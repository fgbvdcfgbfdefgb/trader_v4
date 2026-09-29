#!/usr/bin/env python3
"""
Fill the advisor cache ahead of (or instead of) training.

Training never waits on the LLM -- it falls back to the numeric prior on a
cache miss -- but the agents learn from a richer signal once real verdicts
exist.  Run this on a big-CPU box, or overnight, to pre-compute them.

Each (coin, day) is answered exactly once and stored forever, so this is
resumable: kill it and rerun, it picks up where it stopped.

    # every day in the dataset, 8 CPU threads, 8 prompts per batch
    python precompute_advisor.py --threads 8 --batch-size 8

    # just the most recent two years
    python precompute_advisor.py --since 2024-01-01

    # fast mode on a big-RAM box: dequantise once (~14 GB) instead of per call
    python precompute_advisor.py --materialize --dtype bfloat16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.offline import enforce as enforce_offline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model-dir", default="models/mistral7b-int4")
    ap.add_argument("--cache-path",
                    default="runs/ppo_btc_eth_ltc/advisor_cache.sqlite")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="biggest lever on throughput: the int4 unpack cost "
                         "is per forward call, so a batch of 8 costs ~1x")
    ap.add_argument("--max-new-tokens", type=int, default=72)
    ap.add_argument("--materialize", action="store_true",
                    help="dequantise weights once (~14GB fp16 / 28GB fp32)")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--since", default=None, help="only days >= this date")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--coins", nargs="*", default=["BTC", "ETH", "LTC"])
    ap.add_argument("--allow-network", action="store_true")
    args = ap.parse_args()

    if not args.allow_network:
        enforce_offline(True)

    import torch
    threads = args.threads or (os.cpu_count() or 4)
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    from src.data.features import day_price_context, prior_day_context
    from src.data.loader import COIN_OF, Dataset
    from src.llm.advisor import (AdvisorCache, build_prompt, numeric_prior,
                                 parse_response, select_headlines)
    from src.llm.mistral_int4 import MistralInt4
    from transformers import AutoTokenizer

    print(f"[pre] threads={threads} batch={args.batch_size} "
          f"materialize={args.materialize} dtype={args.dtype}", flush=True)
    ds = Dataset(args.data_root, verbose=True)
    cache = AdvisorCache(args.cache_path)

    since = date.fromisoformat(args.since) if args.since else None
    todo: list[tuple[str, date]] = []
    for sym, days in ds.days.items():
        coin = COIN_OF[sym]
        if coin not in args.coins:
            continue
        for d in days:
            if since and d < since:
                continue
            todo.append((coin, d))
    todo.sort(key=lambda x: x[1], reverse=True)
    todo = [t for t in todo if cache.get(*t) is None]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[pre] {len(todo)} (coin, day) pairs to answer; "
          f"{cache.stats()['verdicts']} already cached", flush=True)
    if not todo:
        return 0

    t0 = time.time()
    model = MistralInt4.from_pretrained(args.model_dir,
                                        materialize=args.materialize,
                                        dtype=getattr(torch, args.dtype))
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[pre] model ready in {time.time()-t0:.0f}s", flush=True)

    ok = 0
    t0 = time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        prompts, metas = [], []
        for coin, d in batch:
            sym = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "LTC": "LTCUSDT"}[coin]
            heads = select_headlines(ds.news, coin, d)
            ctx = prior_day_context(ds.daily, coin, d)
            ctx.update(day_price_context(ds.minute, sym, d))
            prompts.append(build_prompt(coin, d, heads, ctx))
            metas.append((coin, d, heads, ctx))

        enc = [tok(p, return_tensors="pt",
                   add_special_tokens=False)["input_ids"][0] for p in prompts]
        L = max(len(e) for e in enc)
        ids = torch.full((len(enc), L), tok.pad_token_id or 0, dtype=torch.long)
        kp = torch.zeros((len(enc), L), dtype=torch.bool)
        for j, e in enumerate(enc):
            ids[j, L - len(e):] = e
            kp[j, L - len(e):] = True

        outs = model.generate(ids, max_new_tokens=args.max_new_tokens,
                              temperature=0.0, eos_id=tok.eos_token_id or 2,
                              key_pad=kp)
        for (coin, d, heads, ctx), toks in zip(metas, outs):
            sig = parse_response(tok.decode(toks, skip_special_tokens=True))
            if sig is None:
                sig = numeric_prior(heads, ctx)
                sig.rationale = "llm parse failed -> prior"
            else:
                ok += 1
            sig.n_headlines = len(heads)
            cache.put(coin, d, sig)

        n = i + len(batch)
        rate = n / max(time.time() - t0, 1e-9)
        print(f"  {n}/{len(todo)}  parsed={ok}  {1/max(rate,1e-9):.1f}s/item  "
              f"eta {(len(todo)-n)/max(rate,1e-9)/3600:.1f} h", flush=True)

    print(f"[pre] done. cache: {cache.stats()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
