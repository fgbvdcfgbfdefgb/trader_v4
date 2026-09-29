#!/usr/bin/env python3
"""
Crawl per-day crypto news headlines from GDELT 2.0 (free, no API key) so the
LLM advisor has something real to read for each training day.

Two products:

1. data/news/headlines/<YYYY>.jsonl.gz
      one JSON record per (day, coin) holding up to N de-duplicated headlines
      that GDELT *first saw* on that UTC day.  Nothing published later is ever
      included -> no look-ahead leakage into the RL episode for that day.

2. data/news/news_tone.parquet
      GDELT daily "tone" and "volume" timelines per coin, full history, cheap.
      Used as a dense numeric fallback for days with no harvested headlines.

The crawler is RESUMABLE and polite: progress is journalled per-day, 429s back
off exponentially, and re-running only fetches the days still missing.

    python scripts/download_news.py --start 2017-08-17 --workers 3
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import requests

# GDELT's DOC API only indexes 2017-01-01 onwards.
GDELT_MIN = date(2017, 1, 1)
DOC = "https://api.gdeltproject.org/api/v2/doc/doc"

COIN_QUERIES = {
    "BTC": '(bitcoin OR "BTC")',
    "ETH": '(ethereum OR "ether" OR "ETH")',
    "LTC": '(litecoin OR "LTC")',
}
# Keyword tagging for the combined crawl
COIN_PATTERNS = {
    "BTC": re.compile(r"\b(bitcoin|btc)\b", re.I),
    "ETH": re.compile(r"\b(ethereum|ether|eth)\b", re.I),
    "LTC": re.compile(r"\b(litecoin|ltc)\b", re.I),
}

_lock = threading.Lock()
_sessions = threading.local()

# GDELT enforces ~1 request / 5 s per IP and answers 429 otherwise.  A single
# global gate is the only thing that actually keeps us under it; per-thread
# sleeps do not compose.  MIN_INTERVAL is deliberately above the documented
# limit so retries do not push us over.
MIN_INTERVAL = 5.5
_gate = threading.Lock()
_last_call = [0.0]


def _throttle() -> None:
    with _gate:
        wait = MIN_INTERVAL - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def sess() -> requests.Session:
    if not hasattr(_sessions, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "trader_v4-newsfetch/1.0 (research)"})
        _sessions.s = s
    return _sessions.s


def gdelt(params: dict, tries: int = 6, timeout: int = 60):
    """GDELT GET with exponential backoff on 429/5xx. Returns dict or None."""
    url = DOC + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    delay = 6.0
    for i in range(tries):
        try:
            _throttle()
            r = sess().get(url, timeout=timeout)
            if r.status_code in (429, 502, 503, 504):
                time.sleep(delay + random.random())
                delay = min(delay * 1.6, 60)
                continue
            r.raise_for_status()
            txt = r.text.strip()
            if not txt or txt[0] not in "{[":
                return None
            return json.loads(txt)
        except json.JSONDecodeError:
            return None
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                return None
            time.sleep(delay + random.random())
            delay = min(delay * 2, 45)
    return None


# --------------------------------------------------------------------------- #
#  1. daily headlines
# --------------------------------------------------------------------------- #
def fetch_day(day: date, max_per_coin: int) -> dict[str, list[dict]]:
    """One combined query per day, then tag each headline to the coins it names."""
    d0 = day.strftime("%Y%m%d000000")
    d1 = (day + timedelta(days=1)).strftime("%Y%m%d000000")
    q = quote('(bitcoin OR ethereum OR litecoin OR "crypto market") sourcelang:eng')
    js = gdelt({
        "query": q, "mode": "artlist", "maxrecords": "250",
        "startdatetime": d0, "enddatetime": d1,
        "sort": "hybridrel", "format": "json",
    })
    buckets: dict[str, list[dict]] = defaultdict(list)
    if not js:
        return buckets
    seen: dict[str, set] = defaultdict(set)
    for a in js.get("articles", []):
        title = (a.get("title") or "").strip()
        if len(title) < 15:
            continue
        stamp = a.get("seendate", "")
        # hard guard: never let a later-dated article into this day's bucket
        if stamp[:8] != day.strftime("%Y%m%d"):
            continue
        rec = {
            "t": title,
            "src": a.get("domain", ""),
            "ts": stamp,
        }
        key = re.sub(r"[^a-z0-9]+", "", title.lower())[:90]
        for coin, pat in COIN_PATTERNS.items():
            if len(buckets[coin]) >= max_per_coin:
                continue
            if pat.search(title) and key not in seen[coin]:
                seen[coin].add(key)
                buckets[coin].append(rec)
    return buckets


def crawl_headlines(out_dir: str, start: date, end: date, workers: int,
                    max_per_coin: int) -> None:
    import concurrent.futures as cf

    os.makedirs(out_dir, exist_ok=True)
    done_path = os.path.join(out_dir, "_done_days.txt")
    done: set[str] = set()
    if os.path.exists(done_path):
        done = {l.strip() for l in open(done_path) if l.strip()}

    days = []
    d = max(start, GDELT_MIN)
    while d <= end:
        if d.isoformat() not in done:
            days.append(d)
        d += timedelta(days=1)
    # Most-recent-first: the newest regimes matter most if we run out of time.
    days.reverse()
    print(f"[headlines] {len(days)} days to fetch "
          f"({len(done)} already done)", flush=True)
    if not days:
        return

    writers: dict[int, gzip.GzipFile] = {}
    done_fh = open(done_path, "a")
    n_ok = 0
    t0 = time.time()

    def _writer(year: int):
        if year not in writers:
            writers[year] = gzip.open(
                os.path.join(out_dir, f"{year}.jsonl.gz"), "at", compresslevel=6)
        return writers[year]

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, dd, max_per_coin): dd for dd in days}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            dd = futs[fu]
            try:
                buckets = fu.result()
            except Exception:  # noqa: BLE001
                buckets = {}
            with _lock:
                w = _writer(dd.year)
                total = 0
                for coin, items in buckets.items():
                    if not items:
                        continue
                    total += len(items)
                    w.write(json.dumps({
                        "date": dd.isoformat(), "coin": coin,
                        "n": len(items), "headlines": items,
                    }, ensure_ascii=False) + "\n")
                done_fh.write(dd.isoformat() + "\n")
                if i % 5 == 0:
                    w.flush()
                    done_fh.flush()
                if total:
                    n_ok += 1
            if i % 20 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(days) - i) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(days)} days  ({n_ok} with news)  "
                      f"{rate:.2f} d/s  eta {eta:.0f} min", flush=True)

    for w in writers.values():
        w.close()
    done_fh.close()
    print(f"[headlines] finished, {n_ok} days had usable headlines", flush=True)


# --------------------------------------------------------------------------- #
#  2. daily tone / volume timelines  (dense numeric fallback)
# --------------------------------------------------------------------------- #
def crawl_tone(out_dir: str, start: date, end: date) -> None:
    frames = []
    for coin, q in COIN_QUERIES.items():
        for mode, col in (("timelinetone", "tone"), ("timelinevol", "vol")):
            chunks = []
            # The timeline endpoints happily return the full multi-year range
            # in a single response, so this is 6 requests total, not 60.
            for (a, b) in [(max(start, GDELT_MIN), end)]:
                cache = os.path.join(out_dir, "_raw",
                                     f"gdelt_{mode}_{coin}_full.json")
                if os.path.exists(cache) and os.path.getsize(cache) > 64:
                    js = json.load(open(cache))
                else:
                    js = gdelt({
                        "query": quote(q + " sourcelang:eng"), "mode": mode,
                        "startdatetime": a.strftime("%Y%m%d000000"),
                        "enddatetime": b.strftime("%Y%m%d235959"),
                        "format": "json", "timelinesmooth": "0",
                    })
                    if js:
                        os.makedirs(os.path.dirname(cache), exist_ok=True)
                        json.dump(js, open(cache, "w"))
                if not js or not js.get("timeline"):
                    continue
                pts = js["timeline"][0].get("data", [])
                if pts:
                    chunks.append(pd.DataFrame({
                        "date": pd.to_datetime(
                            [p["date"] for p in pts], format="%Y%m%dT%H%M%SZ",
                            errors="coerce", utc=True),
                        f"{coin.lower()}_news_{col}": [p["value"] for p in pts],
                    }))
            if chunks:
                df = pd.concat(chunks, ignore_index=True).dropna(subset=["date"])
                df["date"] = df["date"].dt.floor("D")
                df = df.groupby("date", as_index=False).mean()
                frames.append(df)
                print(f"  {coin} {col}: {len(df)} days", flush=True)

    if not frames:
        print("[tone] nothing fetched", flush=True)
        return
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="date", how="outer")
    merged = merged.sort_values("date", ignore_index=True)
    merged["date"] = merged["date"].dt.tz_localize(None)
    for c in merged.columns:
        if c != "date":
            merged[c] = merged[c].astype("float32")
    p = os.path.join(out_dir, "news_tone.parquet")
    merged.to_parquet(p, compression="zstd", index=False)
    print(f"[tone] wrote {p}: {merged.shape[0]} days x {merged.shape[1]-1} cols",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/news")
    ap.add_argument("--start", default="2017-08-17")
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-per-coin", type=int, default=24)
    ap.add_argument("--skip-headlines", action="store_true")
    ap.add_argument("--skip-tone", action="store_true")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = (date.fromisoformat(args.end) if args.end
           else datetime.now(timezone.utc).date() - timedelta(days=1))
    os.makedirs(args.out, exist_ok=True)

    if not args.skip_tone:
        print("=== GDELT tone/volume timelines ===", flush=True)
        crawl_tone(args.out, start, end)
    if not args.skip_headlines:
        print("=== GDELT daily headlines ===", flush=True)
        crawl_headlines(os.path.join(args.out, "headlines"), start, end,
                        args.workers, args.max_per_coin)
    print("NEWS DOWNLOAD COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
