"""
A self-contained Mistral-7B forward pass that reads the repo's INT4 shards.

Why not `transformers.MistralForCausalLM`?
  * the training host pins transformers 5.x, whose internal module names and
    attention plumbing differ from the 4.x layout the weights were exported
    under -- monkey-patching nn.Linear -> Int4Linear across versions is
    brittle, and there is no network access to fix it if it breaks;
  * we need a KV cache that is cheap on CPU and nothing else;
  * a decoder-only Mistral block is ~150 lines. Owning it removes the risk.

`transformers` is still used, but only for `AutoTokenizer`, whose API is
stable across versions.

Architecture (Mistral-7B-Instruct-v0.3):
    32 layers, d_model 4096, 32 heads / 8 KV heads (GQA), head_dim 128,
    SwiGLU MLP with intermediate 14336, RMSNorm, RoPE theta 1e6,
    vocab 32768, no sliding-window attention.
"""
from __future__ import annotations

import gc
import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .int4_linear import Int4Linear


@dataclass
class MistralConfig:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str) -> "MistralConfig":
        d = json.load(open(path))
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


def build_rope(head_dim: int, max_pos: int, theta: float, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32)
                           / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, H, T, D];  cos/sin: [T, D/2]"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class Attention(nn.Module):
    def __init__(self, cfg: MistralConfig, group: int, materialize: bool,
                 dtype: torch.dtype):
        super().__init__()
        self.cfg = cfg
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        mk = lambda i, o: Int4Linear(i, o, group, False, materialize, dtype)  # noqa: E731
        self.q_proj = mk(cfg.hidden_size, self.nh * self.hd)
        self.k_proj = mk(cfg.hidden_size, self.nkv * self.hd)
        self.v_proj = mk(cfg.hidden_size, self.nkv * self.hd)
        self.o_proj = mk(self.nh * self.hd, cfg.hidden_size)

    def forward(self, x, cos, sin, cache, mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Always return a cache, including on the prefill pass -- otherwise
        # the first call produces nothing to decode from.
        if cache is not None:
            pk, pv = cache
            if pk is not None:
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)

        if self.nkv != self.nh:
            rep = self.nh // self.nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: MistralConfig, group: int, materialize: bool,
                 dtype: torch.dtype):
        super().__init__()
        mk = lambda i, o: Int4Linear(i, o, group, False, materialize, dtype)  # noqa: E731
        self.gate_proj = mk(cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = mk(cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = mk(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: MistralConfig, group: int, materialize: bool,
                 dtype: torch.dtype):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, group, materialize, dtype)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg, group, materialize, dtype)

    def forward(self, x, cos, sin, cache, mask):
        h, new_cache = self.self_attn(self.input_layernorm(x), cos, sin, cache, mask)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class MistralInt4(nn.Module):
    """Decoder-only Mistral running on 4-bit weights."""

    def __init__(self, cfg: MistralConfig, group: int = 128,
                 materialize: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg
        self.dtype_ = dtype
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size,
                                         dtype=torch.float16)
        self.layers = nn.ModuleList(
            [Block(cfg, group, materialize, dtype) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = Int4Linear(cfg.hidden_size, cfg.vocab_size, group,
                                  False, materialize, dtype)
        cos, sin = build_rope(cfg.head_dim, cfg.max_position_embeddings,
                              cfg.rope_theta, dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, caches=None, pos: int = 0,
                causal: bool = True, key_pad: torch.Tensor | None = None):
        B, T = ids.shape
        x = self.embed_tokens(ids).to(self.dtype_)
        cos = self.rope_cos[pos:pos + T]
        sin = self.rope_sin[pos:pos + T]

        total = pos + T
        # A large finite negative is used instead of -inf on purpose.  With
        # left-padded batches a pad query row would otherwise be entirely
        # -inf, softmax would emit NaN, and the NaN would then leak into every
        # real token through the next layer's KV cache.  -1e9 keeps those rows
        # finite (uniform) while still giving real tokens ~0 weight on pads.
        NEG = -1e9
        mask = None
        if (causal and T > 1) or key_pad is not None:
            mask = torch.zeros((B, 1, T, total), dtype=x.dtype)
            if causal and T > 1:
                cm = torch.full((T, total), NEG, dtype=x.dtype)
                mask = mask + torch.triu(cm, diagonal=1 + pos)[None, None]
            if key_pad is not None:
                mask = mask.masked_fill(
                    ~key_pad[:, None, None, :total].bool(), NEG)

        new_caches = []
        for i, blk in enumerate(self.layers):
            c = caches[i] if caches is not None else None
            x, nc = blk(x, cos, sin, c, mask)
            new_caches.append(nc)
        x = self.norm(x)
        return self.lm_head(x), new_caches

    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(cls, path: str, materialize: bool = False,
                        dtype: torch.dtype = torch.float32,
                        verbose: bool = True) -> "MistralInt4":
        cfg = MistralConfig.from_json(os.path.join(path, "config.json"))
        man = json.load(open(os.path.join(path, "quant_manifest.json")))
        group = int(man.get("group_size", 128))
        model = cls(cfg, group, materialize, dtype)

        # group tensor names by file so each shard is opened exactly once
        by_file: dict[str, list[str]] = {}
        for name, fn in man["weight_map"].items():
            by_file.setdefault(fn, []).append(name)

        mods = dict(model.named_modules())

        def resolve(name: str) -> str:
            # checkpoint keys are HF-style ("model.layers.0...."), our module
            # tree is rooted one level higher, and lm_head has no prefix.
            return name[6:] if name.startswith("model.") else name
        embed_chunks: dict[int, torch.Tensor] = {}
        pending: dict[str, dict[str, torch.Tensor]] = {}
        loaded = 0

        for fi, (fn, names) in enumerate(sorted(by_file.items()), 1):
            blob = load_file(os.path.join(path, fn))
            for name in names:
                t = blob[name]
                if ".embed_tokens.weight.chunk" in name:
                    idx = int(name.rsplit("chunk", 1)[1])
                    embed_chunks[idx] = t
                    continue
                if name.endswith((".qweight", ".scale", ".zero")):
                    base, kind = name.rsplit(".", 1)
                    mod_name = base[: -len(".weight")] if base.endswith(".weight") else base
                    mod_name = resolve(mod_name)
                    pending.setdefault(mod_name, {})[kind] = t
                    d = pending[mod_name]
                    if len(d) == 3:
                        m = mods.get(mod_name)
                        if m is None:
                            raise KeyError(f"no module {mod_name}")
                        m.load(d["qweight"], d["scale"], d["zero"])
                        pending.pop(mod_name)
                        loaded += 1
                    continue
                # plain fp16 tensor: norms
                mod_name, leaf = name.rsplit(".", 1)
                mod_name = resolve(mod_name)
                m = mods.get(mod_name)
                if m is not None and hasattr(m, leaf):
                    setattr(m, leaf, t.float() if leaf == "weight"
                            and isinstance(m, RMSNorm) else t)
                    loaded += 1
            del blob
            gc.collect()
            if verbose and fi % 8 == 0:
                print(f"    [llm] shard {fi}/{len(by_file)}", flush=True)

        if embed_chunks:
            emb = torch.cat([embed_chunks[i] for i in sorted(embed_chunks)], dim=0)
            model.embed_tokens.weight = nn.Parameter(emb, requires_grad=False)
            del embed_chunks
        if pending:
            raise RuntimeError(f"incomplete int4 tensors: {list(pending)[:4]}")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        gc.collect()
        if verbose:
            print(f"    [llm] loaded {loaded} tensors from {len(by_file)} shards",
                  flush=True)
        return model

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 96,
                 temperature: float = 0.0, eos_id: int = 2,
                 stop_strings=None, tokenizer=None,
                 key_pad: torch.Tensor | None = None) -> list[list[int]]:
        """
        Greedy (temperature=0) or sampled batch generation with a KV cache.

        `key_pad` is a [B, T] bool mask over the prompt (True = real token).
        Prompts must be LEFT-padded so the final position is always real.
        RoPE is relative, so a shared positional offset across the batch is
        harmless; only the padding mask matters.
        """
        B = ids.shape[0]
        if key_pad is None:
            key_pad = torch.ones_like(ids, dtype=torch.bool)
        logits, caches = self(ids, caches=None, pos=0, key_pad=key_pad)
        pos = ids.shape[1]
        nxt = self._pick(logits[:, -1], temperature)
        out = [[int(nxt[b])] for b in range(B)]
        done = [int(nxt[b]) == eos_id for b in range(B)]

        for _ in range(max_new_tokens - 1):
            if all(done):
                break
            key_pad = torch.cat(
                [key_pad, torch.ones((B, 1), dtype=torch.bool)], dim=1)
            logits, caches = self(nxt.view(B, 1), caches=caches, pos=pos,
                                  key_pad=key_pad)
            pos += 1
            nxt = self._pick(logits[:, -1], temperature)
            for b in range(B):
                if done[b]:
                    continue
                tok = int(nxt[b])
                if tok == eos_id:
                    done[b] = True
                    continue
                out[b].append(tok)
                if stop_strings and tokenizer is not None and len(out[b]) % 8 == 0:
                    txt = tokenizer.decode(out[b])
                    if any(s in txt for s in stop_strings):
                        done[b] = True
        return out

    @staticmethod
    def _pick(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)


def estimate_ram_gb(materialize: bool, dtype: torch.dtype) -> float:
    """Rough resident-set estimate for the 7B checkpoint."""
    if not materialize:
        return 4.0
    return 7.0 * (2 if dtype in (torch.float16, torch.bfloat16) else 4)
