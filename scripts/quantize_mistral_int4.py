#!/usr/bin/env python3
"""
Turn mistralai/Mistral-7B-Instruct-v0.3 (14.5 GB bf16) into a ~3.9 GB
group-wise INT4 checkpoint that loads with nothing but torch + safetensors.

Why not GGUF / bitsandbytes / AWQ / GPTQ?
    The target machine (Snowflake trial) ships torch + transformers +
    safetensors and cannot pip-install anything.  Every off-the-shelf 4-bit
    format needs an extra runtime (llama.cpp, bitsandbytes, autoawq,
    gptqmodel, gguf ...).  So we define a dead-simple format and ship a small
    pure-torch kernel for it in src/llm/int4_linear.py.

Format, per 2-D linear weight W [out, in]:
    <name>.qweight : uint8   [out, in//2]    two 4-bit codes packed per byte
    <name>.scale   : float16 [out, in//G]    per-(row, group) scale
    <name>.zero    : float16 [out, in//G]    per-(row, group) offset
    W ~= q.float() * scale + zero            (asymmetric round-to-nearest)
Token embeddings are kept in fp16 (row-chunked); norms stay fp16.

Two constraints drive the output layout:
  * GitHub hard-rejects blobs > 100 MB  -> every emitted file is <= 90 MB
  * the box has ~2 GB RAM               -> tensors are streamed one at a time
                                           and flushed as soon as a bin fills
Because the shards are already small and self-describing, the training host
loads them *directly*; there is no reassembly step.

    python scripts/quantize_mistral_int4.py --out models/mistral7b-int4
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import requests
from safetensors.numpy import save_file

REPO = "mistralai/Mistral-7B-Instruct-v0.3"
HF = "https://huggingface.co"
SHARDS = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]
SIDECARS = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
]

GROUP = 128
MAX_FILE = 90 * 1000 * 1000        # stay comfortably under GitHub's 100 MB
EMBED_CHUNK_ROWS = 8192            # 8192 x 4096 fp16 = 67 MB


def bf16_to_f32(raw: np.ndarray, shape) -> np.ndarray:
    """bfloat16 == the top 16 bits of a float32, so shift left and reinterpret."""
    u16 = raw.view(np.uint16).astype(np.uint32)
    return ((u16 << 16).view(np.float32)).reshape(shape)


def quantize_rows(w: np.ndarray, group: int, n_grid: int = 10):
    """
    w float32 [out, in] -> (uint8 packed [out, in//2], fp16 scale, fp16 zero).

    Plain round-to-nearest uses each group's exact min/max, so a single
    outlier weight stretches the 16-level grid and the other 127 weights in
    the group lose precision.  Measured against the original bf16 tensor that
    costs ~0.156 relative error instead of the ~0.107 the bit budget allows.

    So for every group we search a handful of symmetric clip ratios and keep
    whichever minimises squared reconstruction error.  Outliers get clamped
    (they are cheap to lose) and the bulk of the distribution gets a finer
    step.  This is the cheap part of what GPTQ/AWQ do, costs nothing at
    inference time and not one extra byte on disk.
    """
    out_f, in_f = w.shape
    ng = in_f // group
    qs = np.empty((out_f, in_f), dtype=np.uint8)
    scales = np.empty((out_f, ng), dtype=np.float32)
    zeros = np.empty((out_f, ng), dtype=np.float32)

    ratios = np.linspace(1.0, 0.55, n_grid, dtype=np.float32)
    BLK = 256
    for r0 in range(0, out_f, BLK):
        r1 = min(r0 + BLK, out_f)
        blk = w[r0:r1].reshape(r1 - r0, ng, group)
        mn = blk.min(axis=2)
        mx = blk.max(axis=2)
        centre = (mx + mn) * 0.5
        half = (mx - mn) * 0.5

        best_err = np.full((r1 - r0, ng), np.inf, dtype=np.float32)
        best_sc = np.empty((r1 - r0, ng), dtype=np.float32)
        best_lo = np.empty((r1 - r0, ng), dtype=np.float32)
        for ratio in ratios:
            h = np.maximum(half * ratio, 1e-9)
            lo = centre - h
            sc = np.maximum((2.0 * h) / 15.0, 1e-9).astype(np.float32)
            q = np.rint((blk - lo[:, :, None]) / sc[:, :, None])
            np.clip(q, 0, 15, out=q)
            err = ((q * sc[:, :, None] + lo[:, :, None]) - blk)
            err = np.einsum("ijk,ijk->ij", err, err)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_sc = np.where(better, sc, best_sc)
            best_lo = np.where(better, lo, best_lo)
            del q, err, h, lo, sc

        q = np.rint((blk - best_lo[:, :, None]) / best_sc[:, :, None])
        np.clip(q, 0, 15, out=q)
        qs[r0:r1] = q.astype(np.uint8).reshape(r1 - r0, in_f)
        scales[r0:r1] = best_sc
        zeros[r0:r1] = best_lo
        del blk, mn, mx, centre, half, best_err, best_sc, best_lo, q

    packed = (qs[:, 0::2] | (qs[:, 1::2] << 4)).astype(np.uint8)
    del qs
    gc.collect()
    return packed, scales.astype(np.float16), zeros.astype(np.float16)


def dequant_check(packed, scale, zero, ref, group) -> float:
    """Relative Frobenius error on a small row slice -> sanity metric."""
    n = min(64, packed.shape[0])
    lo = (packed[:n] & 0x0F).astype(np.float32)
    hi = (packed[:n] >> 4).astype(np.float32)
    out = np.empty((n, packed.shape[1] * 2), dtype=np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    ng = out.shape[1] // group
    out = out.reshape(n, ng, group) * scale[:n].astype(np.float32)[:, :, None] \
        + zero[:n].astype(np.float32)[:, :, None]
    out = out.reshape(n, -1)
    r = ref[:n]
    return float(np.linalg.norm(out - r) / max(np.linalg.norm(r), 1e-9))


def download(url: str, dest: str, desc: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  [cached] {desc}", flush=True)
        return
    tmp = dest + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    t0 = time.time()
    for attempt in range(6):
        try:
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, stream=True, timeout=300, headers=headers,
                              allow_redirects=True) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + done
                nxt = done + (1 << 29)
                with open(tmp, "ab") as fh:
                    for chunk in r.iter_content(1 << 22):
                        fh.write(chunk)
                        done += len(chunk)
                        if done >= nxt:
                            nxt = done + (1 << 29)
                            pct = 100 * done / total if total else 0
                            print(f"    {desc} {pct:5.1f}%  {done/1e9:5.2f} GB  "
                                  f"{done/1e6/max(time.time()-t0,1e-9):.1f} MB/s",
                                  flush=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"    ~ retry {attempt+1}: {exc}", flush=True)
            done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            time.sleep(5)
    os.rename(tmp, dest)
    print(f"  [done] {desc} {os.path.getsize(dest)/1e9:.2f} GB in "
          f"{time.time()-t0:.0f}s", flush=True)


class BinWriter:
    """Greedily packs tensors into <= MAX_FILE safetensors shards."""

    def __init__(self, outdir: str, group: int):
        self.outdir = outdir
        self.group = group
        self.buf: dict[str, np.ndarray] = {}
        self.nbytes = 0
        self.n = 0
        self.index: dict[str, str] = {}
        self.pending: list[str] = []

    def add(self, name: str, arr: np.ndarray) -> None:
        if self.nbytes + arr.nbytes > MAX_FILE and self.buf:
            self.flush()
        self.buf[name] = arr
        self.pending.append(name)
        self.nbytes += arr.nbytes

    def flush(self) -> None:
        if not self.buf:
            return
        self.n += 1
        fn = f"q-{self.n:05d}.safetensors"
        save_file(self.buf, os.path.join(self.outdir, fn),
                  metadata={"format": f"trader_v4-int4-g{self.group}"})
        for k in self.pending:
            self.index[k] = fn
        sz = os.path.getsize(os.path.join(self.outdir, fn))
        print(f"    -> {fn}  {sz/1e6:6.1f} MB  ({len(self.buf)} tensors)", flush=True)
        self.buf.clear()
        self.pending.clear()
        self.nbytes = 0
        gc.collect()


def read_header(path: str):
    with open(path, "rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(hlen).decode())
    return hlen, header


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/mistral7b-int4")
    ap.add_argument("--work", default="/home/user/_hfwork")
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--clip-grid", type=int, default=10,
                    help="clip ratios searched per group; 1 = plain RTN")
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    print("=== sidecars ===", flush=True)
    for fn in SIDECARS:
        dest = os.path.join(args.out, fn)
        if os.path.exists(dest) and os.path.getsize(dest) > 16:
            continue
        try:
            r = requests.get(f"{HF}/{REPO}/resolve/main/{fn}", timeout=180)
            if r.ok:
                open(dest, "wb").write(r.content)
                print(f"  {fn}  {len(r.content)/1e6:.2f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {fn}: {exc}", flush=True)

    writer = BinWriter(args.out, args.group)
    meta: dict[str, dict] = {}
    n_q = n_fp = 0
    errs: list[float] = []

    for si, shard in enumerate(SHARDS, 1):
        src = os.path.join(args.work, shard)
        print(f"\n=== shard {si}/{len(SHARDS)} : download ===", flush=True)
        download(f"{HF}/{REPO}/resolve/main/{shard}", src, shard)

        print(f"=== shard {si}/{len(SHARDS)} : quantise ===", flush=True)
        hlen, header = read_header(src)
        keys = sorted(k for k in header if k != "__metadata__")
        t0 = time.time()

        for ki, k in enumerate(keys):
            info = header[k]
            shape = info["shape"]
            s, e = info["data_offsets"]
            with open(src, "rb") as raw:
                raw.seek(8 + hlen + s)
                buf = np.frombuffer(raw.read(e - s), dtype=np.uint8)
            arr = (bf16_to_f32(buf, shape) if info["dtype"] == "BF16"
                   else np.frombuffer(buf, dtype=np.float32).reshape(shape))
            del buf

            is_embed = "embed_tokens" in k
            is_lin = (len(shape) == 2 and not is_embed
                      and shape[1] % args.group == 0)

            if is_embed:
                # keep fp16, split into row-chunks so each blob stays < 90 MB
                rows = shape[0]
                nch = 0
                for r0 in range(0, rows, EMBED_CHUNK_ROWS):
                    r1 = min(r0 + EMBED_CHUNK_ROWS, rows)
                    writer.add(f"{k}.chunk{nch}", arr[r0:r1].astype(np.float16))
                    nch += 1
                meta[k] = {"kind": "fp16_chunked", "shape": list(shape),
                           "chunks": nch, "chunk_rows": EMBED_CHUNK_ROWS}
                n_fp += 1
            elif is_lin:
                pk, sc, ze = quantize_rows(arr, args.group, args.clip_grid)
                if ki % 40 == 0:
                    errs.append(dequant_check(pk, sc, ze, arr, args.group))
                writer.add(k + ".qweight", pk)
                writer.add(k + ".scale", sc)
                writer.add(k + ".zero", ze)
                meta[k] = {"kind": "int4", "shape": list(shape),
                           "group": args.group}
                n_q += 1
                del pk, sc, ze
            else:
                writer.add(k, arr.astype(np.float16))
                meta[k] = {"kind": "fp16", "shape": list(shape)}
                n_fp += 1

            del arr
            if ki % 25 == 0:
                gc.collect()
                print(f"    [{ki+1}/{len(keys)}] {k}  ({time.time()-t0:.0f}s)",
                      flush=True)

        if not args.keep_shards:
            os.remove(src)
            print(f"  freed {shard}", flush=True)

    writer.flush()

    cfg = json.load(open(os.path.join(args.out, "config.json")))
    manifest = {
        "source_repo": REPO,
        "format": "trader_v4-int4",
        "group_size": args.group,
        "scheme": "asymmetric, MSE-optimal clip search;  W ~= q.float()*scale + zero",
        "clip_grid": args.clip_grid,
        "packing": "uint8; low nibble = even input index, high nibble = odd",
        "embed_scheme": f"fp16, row chunks of {EMBED_CHUNK_ROWS}",
        "n_int4_tensors": n_q,
        "n_fp16_tensors": n_fp,
        "mean_rel_quant_error": round(float(np.mean(errs)), 5) if errs else None,
        "max_rel_quant_error": round(float(np.max(errs)), 5) if errs else None,
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "weight_map": writer.index,
        "tensors": meta,
    }
    json.dump(manifest, open(os.path.join(args.out, "quant_manifest.json"), "w"),
              indent=1)

    tot = sum(os.path.getsize(os.path.join(args.out, f))
              for f in os.listdir(args.out))
    big = [f for f in os.listdir(args.out)
           if os.path.getsize(os.path.join(args.out, f)) > 99_000_000]
    print(f"\nTOTAL {tot/1e9:.2f} GB   int4={n_q}  fp16={n_fp}  "
          f"files={writer.n}", flush=True)
    print(f"mean rel quant err: {manifest['mean_rel_quant_error']}", flush=True)
    print(f"files over 99 MB: {big if big else 'none'}", flush=True)
    print("QUANTIZE COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
