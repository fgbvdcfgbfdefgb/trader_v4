#!/usr/bin/env python3
"""
Download the *daily, market-moving* context series that sit underneath the
minute bars.  Everything here is free, key-less and full-history.

Sources
-------
alternative.me     Crypto Fear & Greed index                 (2018-02 -> now)
blockchain.info    BTC on-chain fundamentals (9 charts)      (2009    -> now)
OKX v5             Perp funding rate, BTC/ETH/LTC            (~2019   -> now)
Wikimedia          Daily pageviews for 4 crypto articles     (2015-07 -> now)
FRED (St. Louis)   Macro: USD index, fed funds, 10y, VIX,
                   S&P 500, HY spread, 2s10s, WTI, gold      (varies  -> now)

Output: data/daily/market_daily.parquet  (one row per UTC day, wide format)
        data/daily/_raw/*.json|csv       (raw payloads, kept for provenance)

Resumable: raw payloads are cached on disk; re-running only refetches misses.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

S = requests.Session()
S.headers.update({"User-Agent": "trader_v4-datafetch/1.0 (research)"})

BLOCKCHAIN_CHARTS = [
    "n-transactions", "hash-rate", "difficulty", "miners-revenue",
    "n-unique-addresses", "mempool-size", "avg-block-size",
    "estimated-transaction-volume-usd", "market-price",
]

FRED_SERIES = {
    "usd_broad_index": "DTWEXBGS",
    "fed_funds_rate": "DFF",
    "treasury_10y": "DGS10",
    "yield_curve_2s10s": "T10Y2Y",
    "vix": "VIXCLS",
    "sp500": "SP500",
    "hy_credit_spread": "BAMLH0A0HYM2",
    "wti_oil": "DCOILWTICO",
    "nasdaq100": "NASDAQ100",
}

WIKI_ARTICLES = ["Bitcoin", "Ethereum", "Litecoin", "Cryptocurrency"]

OKX_SWAPS = {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP", "LTC": "LTC-USDT-SWAP"}


def cached(path: str, fn, *, binary: bool = False, max_age_d: float = 3.0):
    """Disk-cache a fetch. Returns str/bytes or None."""
    if os.path.exists(path) and os.path.getsize(path) > 32:
        age = (time.time() - os.path.getmtime(path)) / 86400
        if age < max_age_d:
            return open(path, "rb" if binary else "r").read()
    try:
        val = fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {os.path.basename(path)}: {exc}", flush=True)
        if os.path.exists(path):
            return open(path, "rb" if binary else "r").read()
        return None
    if val is None:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if binary else "w") as fh:
        fh.write(val)
    return val


def get(url: str, tries: int = 5, timeout: int = 90, **kw) -> str:
    last = None
    for i in range(tries):
        try:
            r = S.get(url, timeout=timeout, **kw)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def to_day(ts_s) -> pd.Series:
    return pd.to_datetime(ts_s, unit="s", utc=True).dt.floor("D")


# --------------------------------------------------------------------------- #
def fetch_fear_greed(raw: str) -> pd.DataFrame:
    txt = cached(f"{raw}/fear_greed.json",
                 lambda: get("https://api.alternative.me/fng/?limit=0&format=json"))
    if not txt:
        return pd.DataFrame()
    d = json.loads(txt)["data"]
    df = pd.DataFrame({
        "date": to_day(pd.Series([int(x["timestamp"]) for x in d])),
        "fear_greed": pd.Series([float(x["value"]) for x in d]),
    })
    print(f"  fear_greed: {len(df)} days", flush=True)
    return df.drop_duplicates("date")


def fetch_blockchain(raw: str) -> pd.DataFrame:
    out = None
    for chart in BLOCKCHAIN_CHARTS:
        url = (f"https://api.blockchain.info/charts/{chart}"
               f"?timespan=all&rollingAverage=24hours&format=json")
        txt = cached(f"{raw}/bc_{chart}.json", lambda u=url: get(u))
        if not txt:
            continue
        try:
            vals = json.loads(txt).get("values", [])
        except Exception:  # noqa: BLE001
            continue
        if not vals:
            continue
        col = "btc_" + chart.replace("-", "_")
        df = pd.DataFrame({
            "date": to_day(pd.Series([v["x"] for v in vals])),
            col: pd.Series([v["y"] for v in vals], dtype="float64"),
        }).drop_duplicates("date")
        out = df if out is None else out.merge(df, on="date", how="outer")
        print(f"  {col}: {len(df)} days", flush=True)
        time.sleep(0.4)
    return out if out is not None else pd.DataFrame()


def fetch_okx_funding(raw: str) -> pd.DataFrame:
    """Walk funding history backwards, 100 rows per page."""
    out = None
    for tag, inst in OKX_SWAPS.items():
        path = f"{raw}/okx_funding_{tag}.json"
        if os.path.exists(path) and os.path.getsize(path) > 256:
            rows = json.load(open(path))
        else:
            rows, before = [], None
            for _ in range(600):  # 600*100*8h  >> full history
                url = ("https://www.okx.com/api/v5/public/funding-rate-history"
                       f"?instId={inst}&limit=100")
                if before:
                    url += f"&after={before}"
                try:
                    data = json.loads(get(url, tries=3, timeout=45)).get("data", [])
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! okx {tag}: {exc}", flush=True)
                    break
                if not data:
                    break
                rows.extend(data)
                before = data[-1]["fundingTime"]
                time.sleep(0.25)
            if rows:
                os.makedirs(raw, exist_ok=True)
                json.dump(rows, open(path, "w"))
        if not rows:
            continue
        df = pd.DataFrame({
            "date": to_day(pd.Series([int(r["fundingTime"]) // 1000 for r in rows])),
            f"{tag.lower()}_funding_rate": pd.Series(
                [float(r["fundingRate"]) for r in rows], dtype="float64"),
        })
        df = df.groupby("date", as_index=False).mean()
        out = df if out is None else out.merge(df, on="date", how="outer")
        print(f"  {tag} funding: {len(df)} days", flush=True)
    return out if out is not None else pd.DataFrame()


def fetch_wiki(raw: str, today: str) -> pd.DataFrame:
    out = None
    for art in WIKI_ARTICLES:
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/user/{art}/daily/20150701/{today}")
        txt = cached(f"{raw}/wiki_{art}.json", lambda u=url: get(u))
        if not txt:
            continue
        try:
            items = json.loads(txt).get("items", [])
        except Exception:  # noqa: BLE001
            continue
        if not items:
            continue
        col = f"wiki_views_{art.lower()}"
        df = pd.DataFrame({
            "date": pd.to_datetime([i["timestamp"][:8] for i in items],
                                   format="%Y%m%d", utc=True),
            col: pd.Series([i["views"] for i in items], dtype="float64"),
        })
        out = df if out is None else out.merge(df, on="date", how="outer")
        print(f"  {col}: {len(df)} days", flush=True)
        time.sleep(0.4)
    return out if out is not None else pd.DataFrame()


def fetch_fred(raw: str) -> pd.DataFrame:
    out = None
    for name, sid in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        txt = cached(f"{raw}/fred_{sid}.csv", lambda u=url: get(u))
        if not txt:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt))
        except Exception:  # noqa: BLE001
            continue
        dcol, vcol = df.columns[0], df.columns[1]
        df = pd.DataFrame({
            "date": pd.to_datetime(df[dcol], utc=True, errors="coerce"),
            name: pd.to_numeric(df[vcol], errors="coerce"),
        }).dropna(subset=["date"])
        out = df if out is None else out.merge(df, on="date", how="outer")
        print(f"  {name} ({sid}): {df[name].notna().sum()} obs", flush=True)
        time.sleep(0.3)
    return out if out is not None else pd.DataFrame()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/daily")
    args = ap.parse_args()
    raw = os.path.join(args.out, "_raw")
    os.makedirs(raw, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")

    parts = []
    for label, fn in [
        ("fear & greed", lambda: fetch_fear_greed(raw)),
        ("blockchain.info", lambda: fetch_blockchain(raw)),
        ("okx funding", lambda: fetch_okx_funding(raw)),
        ("wikipedia", lambda: fetch_wiki(raw, today)),
        ("fred macro", lambda: fetch_fred(raw)),
    ]:
        print(f"[{label}]", flush=True)
        try:
            df = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {label} failed entirely: {exc}", flush=True)
            df = pd.DataFrame()
        if df is not None and len(df):
            parts.append(df)

    if not parts:
        print("NO DAILY DATA", flush=True)
        return 1

    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="date", how="outer")
    merged = merged.sort_values("date", ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.tz_localize(None)

    # Trim to the crypto era and forward-fill slow-moving macro series only.
    merged = merged[merged["date"] >= "2016-01-01"].reset_index(drop=True)
    macro_cols = list(FRED_SERIES.keys())
    for c in macro_cols:
        if c in merged:
            merged[c] = merged[c].ffill(limit=7)

    for c in merged.columns:
        if c != "date":
            merged[c] = merged[c].astype("float32")

    path = os.path.join(args.out, "market_daily.parquet")
    merged.to_parquet(path, compression="zstd", index=False)
    print(f"\nWROTE {path}: {merged.shape[0]} days x {merged.shape[1] - 1} features "
          f"({os.path.getsize(path) / 1e6:.2f} MB)", flush=True)
    print("columns:", ", ".join(c for c in merged.columns if c != "date"), flush=True)
    print("DAILY DOWNLOAD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
