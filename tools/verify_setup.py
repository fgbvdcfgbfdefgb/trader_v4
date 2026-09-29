#!/usr/bin/env python3
"""
Run this FIRST on the training host.  It checks, offline, that every piece
the trainer needs is present and self-consistent, and prints exactly what is
missing if not.

    python tools/verify_setup.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK, WARN, BAD = "  [ok]  ", "  [warn]", "  [FAIL]"
problems: list[str] = []
warnings: list[str] = []


def section(t: str) -> None:
    print(f"\n{'='*70}\n  {t}\n{'='*70}")


def check(cond: bool, msg: str, fatal: bool = True) -> bool:
    if cond:
        print(OK + msg)
    elif fatal:
        print(BAD + msg)
        problems.append(msg)
    else:
        print(WARN + " " + msg)
        warnings.append(msg)
    return cond


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    section("1. Python packages")
    import importlib
    for mod, need in [("torch", True), ("numpy", True), ("pandas", True),
                      ("pyarrow", True), ("matplotlib", True),
                      ("safetensors", True), ("transformers", False)]:
        try:
            m = importlib.import_module(mod)
            check(True, f"{mod} {getattr(m,'__version__','?')}")
        except ImportError:
            check(False, f"{mod} is MISSING"
                  + ("" if need else " (only needed for the LLM advisor)"), need)

    import torch
    section("2. Hardware")
    n = torch.cuda.device_count()
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  GPUs           : {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    cuda:{i}  {p.name}  {p.total_memory/1e9:.1f} GB")
    print(f"  CPU cores      : {os.cpu_count()}")
    if n == 0:
        warnings.append("no GPU -> will train on CPU (slow but works)")
        print(WARN + " no GPU visible; training will fall back to CPU")
    print(f"  -> will launch {max(1,n)} parallel agent(s)")

    section("3. Minute data")
    tot = 0
    for sym in ("BTCUSDT", "ETHUSDT", "LTCUSDT"):
        files = sorted(glob.glob(f"data/minute/{sym}/*.parquet"))
        mb = sum(os.path.getsize(f) for f in files) / 1e6
        tot += mb
        check(len(files) > 0,
              f"{sym}: {len(files)} year files, {mb:.0f} MB "
              f"({os.path.basename(files[0])[:4] if files else '-'}"
              f"..{os.path.basename(files[-1])[:4] if files else '-'})")
    print(f"  total minute data: {tot:.0f} MB")

    section("4. Daily + news data")
    check(os.path.exists("data/daily/market_daily.parquet"),
          "data/daily/market_daily.parquet")
    if os.path.exists("data/daily/market_daily.parquet"):
        import pandas as pd
        d = pd.read_parquet("data/daily/market_daily.parquet")
        print(f"         {len(d)} days x {d.shape[1]-1} features, "
              f"{d['date'].min().date()} .. {d['date'].max().date()}")
    hl = glob.glob("data/news/headlines/*.jsonl.gz")
    check(len(hl) > 0, f"news headlines: {len(hl)} year files", fatal=False)
    check(os.path.exists("data/news/news_tone.parquet"),
          "GDELT tone series (optional)", fatal=False)

    section("5. int4 Mistral checkpoint")
    md = "models/mistral7b-int4"
    has_model = os.path.isdir(md) and os.path.exists(f"{md}/quant_manifest.json")
    if not check(has_model, f"{md}/quant_manifest.json", fatal=False):
        print("         -> agents will run with the numeric prior only "
              "(still fully functional)")
    else:
        man = json.load(open(f"{md}/quant_manifest.json"))
        shards = sorted(glob.glob(f"{md}/q-*.safetensors"))
        size = sum(os.path.getsize(f) for f in shards) / 1e9
        check(len(shards) > 0, f"{len(shards)} shards, {size:.2f} GB")
        refs = set(man["weight_map"].values())
        have = {os.path.basename(f) for f in shards}
        check(refs <= have, f"all {len(refs)} referenced shards present"
              + ("" if refs <= have else f"  MISSING {sorted(refs-have)[:3]}"))
        big = [f for f in shards if os.path.getsize(f) > 100_000_000]
        check(not big, "every shard under GitHub's 100 MB limit"
              + ("" if not big else f"  OVERSIZE: {big}"))
        print(f"         group={man['group_size']}  "
              f"int4 tensors={man['n_int4_tensors']}  "
              f"rel err={man.get('mean_rel_quant_error')}")
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            check(os.path.exists(f"{md}/{f}"), f"{f}")

    section("6. Load the dataset for real")
    try:
        from src.data.loader import Dataset
        ds = Dataset("data", verbose=True)
        n_days = sum(len(v) for v in ds.days.values())
        check(n_days > 100, f"{n_days} tradable day-episodes available")
        section("7. Build one episode end to end")
        from src.envs.trading_env import TradingEnv
        env = TradingEnv(ds, None, use_llm=False, seed=0)
        o = env.reset()
        import numpy as np
        check(np.isfinite(o).all(), f"observation {o.shape}, all finite")
        steps = 0
        done = False
        while not done and steps < 5000:
            o, r, done, info = env.step(env.n_actions // 2)
            steps += 1
        check(steps > 10, f"episode ran {steps} steps -> "
                          f"equity ${info['equity']:.2f}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        check(False, f"dataset/env failed: {exc}")

    section("RESULT")
    if problems:
        print(f"  {len(problems)} BLOCKING problem(s):")
        for p in problems:
            print(f"    - {p}")
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    - {w}")
    if not problems:
        print("  Everything required is present. Start training with:\n")
        print("      python train.py --epochs 2000\n")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
