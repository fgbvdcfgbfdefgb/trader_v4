"""
CPU-side Mistral-7B advisor daemon.

Runs as its own process alongside the GPU trainers.  It never touches a GPU
and never blocks them: trainers read the SQLite cache and fall back to the
numeric prior on a miss, while this loop grinds through the backlog.

Throughput notes
----------------
In int4 non-materialised mode the dominant cost is unpacking the weights,
and that cost is paid PER FORWARD CALL, not per sequence.  So batching is
close to free: 8 prompts in one batch cost roughly what 1 prompt costs.
`--llm-batch-size` therefore matters far more than thread count.

Each (coin, day) is answered exactly once, ever, and persisted.  Across a
multi-day training run the cache converges to full coverage of the sampled
day universe.
"""
from __future__ import annotations

import os
import time
import traceback
from datetime import date

import torch

from ..data.features import day_price_context, prior_day_context
from .advisor import (AdvisorCache, AdvisorSignal, build_prompt, numeric_prior,
                      parse_response, select_headlines)


def _load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def run_daemon(dataset_root: str, model_dir: str, cache_path: str, *,
               batch_size: int = 4, threads: int = 4, max_new_tokens: int = 72,
               materialize: bool = False, dtype: str = "float32",
               poll_seconds: float = 5.0, prewarm: bool = True,
               symbols=("BTC", "ETH", "LTC"), stop_flag: str | None = None,
               log_every: int = 1, verbose: bool = True) -> None:
    torch.set_num_threads(max(1, threads))
    torch.set_grad_enabled(False)

    from ..data.loader import Dataset
    from .mistral_int4 import MistralInt4

    t0 = time.time()
    if verbose:
        print(f"[advisor] loading dataset from {dataset_root}", flush=True)
    ds = Dataset(dataset_root, verbose=False)
    cache = AdvisorCache(cache_path)

    if verbose:
        print(f"[advisor] loading int4 Mistral from {model_dir} "
              f"(materialize={materialize}, dtype={dtype})", flush=True)
    model = MistralInt4.from_pretrained(
        model_dir, materialize=materialize,
        dtype=getattr(torch, dtype), verbose=verbose)
    tok = _load_tokenizer(model_dir)
    if verbose:
        print(f"[advisor] ready in {time.time()-t0:.0f}s, "
              f"threads={threads} batch={batch_size}", flush=True)

    # deterministic pre-warm order: newest days first, they matter most
    prewarm_queue: list[tuple[str, date]] = []
    if prewarm:
        for sym, days in ds.days.items():
            coin = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "LTCUSDT": "LTC"}[sym]
            if coin in symbols:
                prewarm_queue += [(coin, d) for d in days]
        prewarm_queue.sort(key=lambda x: x[1], reverse=True)

    done = 0
    llm_ok = 0
    while True:
        if stop_flag and os.path.exists(stop_flag):
            print("[advisor] stop flag seen, exiting", flush=True)
            return

        # trainers' explicit requests take priority over the pre-warm sweep
        batch = [(c, date.fromisoformat(d)) for c, d in cache.pending(batch_size)]
        while len(batch) < batch_size and prewarm_queue:
            coin, d = prewarm_queue.pop(0)
            if cache.get(coin, d) is None and (coin, d) not in batch:
                batch.append((coin, d))
        if not batch:
            time.sleep(poll_seconds)
            continue

        prompts, metas = [], []
        for coin, d in batch:
            try:
                sym = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "LTC": "LTCUSDT"}[coin]
                heads = select_headlines(ds.news, coin, d)
                ctx = prior_day_context(ds.daily, coin, d)
                ctx.update(day_price_context(ds.minute, sym, d))
                prompts.append(build_prompt(coin, d, heads, ctx))
                metas.append((coin, d, heads, ctx))
            except Exception:  # noqa: BLE001
                traceback.print_exc()

        if not prompts:
            time.sleep(poll_seconds)
            continue

        try:
            enc = [tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                   for p in prompts]
            maxlen = max(len(e) for e in enc)
            pad_id = tok.pad_token_id or 0
            # left-pad so every sequence's last token is the real final token
            ids = torch.full((len(enc), maxlen), pad_id, dtype=torch.long)
            key_pad = torch.zeros((len(enc), maxlen), dtype=torch.bool)
            for i, e in enumerate(enc):
                ids[i, maxlen - len(e):] = e
                key_pad[i, maxlen - len(e):] = True

            ts = time.time()
            outs = model.generate(ids, max_new_tokens=max_new_tokens,
                                  temperature=0.0, eos_id=tok.eos_token_id or 2,
                                  key_pad=key_pad)
            dt = time.time() - ts

            for (coin, d, heads, ctx), toks in zip(metas, outs):
                raw = tok.decode(toks, skip_special_tokens=True)
                sig = parse_response(raw)
                if sig is None:
                    sig = numeric_prior(heads, ctx)
                    sig.rationale = "llm parse failed -> prior"
                else:
                    llm_ok += 1
                sig.n_headlines = len(heads)
                cache.put(coin, d, sig)
                done += 1

            if verbose and done % max(1, log_every * batch_size) < batch_size:
                st = cache.stats()
                print(f"[advisor] {done} answered ({llm_ok} parsed) "
                      f"{dt/len(prompts):.1f}s/item  "
                      f"cache={st['verdicts']} pending={st['pending']}", flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            time.sleep(5.0)


def daemon_entry(kwargs: dict) -> None:
    """multiprocessing entry point."""
    try:
        run_daemon(**kwargs)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()
