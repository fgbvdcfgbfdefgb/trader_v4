#!/usr/bin/env python3
"""
Download full-history 1-minute OHLCV klines for BTC/ETH/LTC from Binance's
public data dump (data.binance.vision) and store them as per-year Parquet.

Why per-year files?  GitHub hard-rejects any blob > 100 MB, and a single
full-history minute file would be borderline.  One file per (symbol, year)
keeps every blob around 8-15 MB and lets the training loader mmap only the
years it needs.

This script is RESUMABLE: already-written year files are skipped.
Run:  python scripts/download_minute.py --out data/minute
"""
from __future__ import annotations

import argparse
import calendar
import concurrent.futures as cf
import io
import os
import sys
import time
import zipfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

BASE = "https://data.binance.vision/data/spot"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]

# Binance spot kline CSV layout (12 cols, no header on older files)
COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "trader_v4-datafetch/1.0"})


def month_iter(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def day_iter(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d = date.fromordinal(d.toordinal() + 1)


def fetch(url: str, tries: int = 4) -> bytes | None:
    """GET with retry. Returns None on a genuine 404 (data not published)."""
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=90)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            if i == tries - 1:
                print(f"    ! give up {url}: {exc}", flush=True)
                return None
            time.sleep(1.5 * (i + 1))
    return None


def parse_zip(blob: bytes, src: str) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = z.namelist()[0]
            raw = z.read(name)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! bad zip {src}: {exc}", flush=True)
        return None

    head = raw[:64].decode("utf-8", "ignore")
    skip = 1 if "open_time" in head else 0
    df = pd.read_csv(
        io.BytesIO(raw), header=None, names=COLS, skiprows=skip,
        usecols=["open_time", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades", "taker_buy_base"],
        dtype={"open_time": np.int64, "open": np.float64, "high": np.float64,
               "low": np.float64, "close": np.float64, "volume": np.float64,
               "quote_volume": np.float64, "trades": np.int64,
               "taker_buy_base": np.float64},
    )
    # Some 2025+ dumps switched open_time to microseconds. Normalise to ms.
    if len(df) and df["open_time"].iloc[0] > 1e14:
        df["open_time"] //= 1000
    return df


def download_symbol_year(symbol: str, year: int, today: date) -> pd.DataFrame | None:
    """Monthly archives for the year, falling back to daily for the current month."""
    frames = []
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    for y, m in month_iter(start, end):
        if (y, m) > (today.year, today.month):
            break
        url = f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{y:04d}-{m:02d}.zip"
        blob = fetch(url)
        if blob is None:
            # Current / most recent month may only exist as daily files yet.
            dstart = date(y, m, 1)
            dend = min(date(y, m, calendar.monthrange(y, m)[1]), today)
            got = []
            with cf.ThreadPoolExecutor(max_workers=6) as ex:
                futs = {
                    ex.submit(
                        fetch,
                        f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{d.isoformat()}.zip",
                    ): d
                    for d in day_iter(dstart, dend)
                }
                for fu in cf.as_completed(futs):
                    b = fu.result()
                    if b:
                        got.append((futs[fu], b))
            for d, b in sorted(got):
                f = parse_zip(b, f"{symbol} {d}")
                if f is not None and len(f):
                    frames.append(f)
            continue
        f = parse_zip(blob, f"{symbol} {y}-{m}")
        if f is not None and len(f):
            frames.append(f)
        del blob

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    del frames
    df = df.drop_duplicates(subset="open_time").sort_values("open_time", ignore_index=True)
    df = df[df["close"] > 0]
    df["trades"] = df["trades"].astype(np.int32)
    for c in ("volume", "quote_volume", "taker_buy_base"):
        df[c] = df[c].astype(np.float32)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/minute")
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--start-year", type=int, default=2017)
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    os.makedirs(args.out, exist_ok=True)

    for symbol in args.symbols:
        sdir = os.path.join(args.out, symbol)
        os.makedirs(sdir, exist_ok=True)
        for year in range(args.start_year, today.year + 1):
            path = os.path.join(sdir, f"{year}.parquet")
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                print(f"[skip] {symbol} {year}", flush=True)
                continue
            t0 = time.time()
            df = download_symbol_year(symbol, year, today)
            if df is None or df.empty:
                print(f"[none] {symbol} {year} (not listed yet)", flush=True)
                continue
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(
                table, path, compression="zstd", compression_level=9,
                use_dictionary=False, data_page_size=1 << 20,
            )
            mb = os.path.getsize(path) / 1e6
            print(f"[ok]   {symbol} {year}: {len(df):>8,} bars -> {mb:6.1f} MB "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del df, table

    print("MINUTE DOWNLOAD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
