#!/usr/bin/env python3
"""
Secondary headline source: Hacker News via the Algolia search API.

Why a second source?  GDELT has the better crypto coverage but enforces
1 request / 5 s per IP, so a full 2017-2026 daily crawl takes ~5 hours.
HN Algolia is key-less, unthrottled and answers in ~0.2 s, so it can cover
the entire history in minutes.  Coverage is thinner and tech-skewed, but it
gives every training day *something* real to read, and the GDELT crawler
tops the same store up in the background.

Writes the identical record shape as download_news.py:
    {"date","coin","n","headlines":[{"t","src","ts"}...]}
into data/news/headlines/<YEAR>.jsonl.gz, so NewsStore merges both sources
transparently.

    python scripts/download_news_hn.py --start 2017-08-17
"""
from __future__ import annotations

import argparse
import calendar
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import requests

API = "https://hn.algolia.com/api/v1/search_by_date"

# Algolia treats `query` as a bag of tokens, not boolean syntax, so each coin
# needs its own pass. Extra generic terms catch market-wide moves.
QUERIES = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "ether"],
    "LTC": ["litecoin"],
}
SHARED = ["cryptocurrency", "crypto market", "coinbase", "binance", "sec crypto"]

S = requests.Session()
S.headers.update({"User-Agent": "trader_v4-newsfetch/1.0 (research)"})


def week_windows(start: date, end: date):
    d = start
    while d <= end:
        b = min(d + timedelta(days=6), end)
        yield d, b
        d = b + timedelta(days=1)


def fetch(query: str, a: date, b: date, tries: int = 4) -> list[dict]:
    lo = int(datetime(a.year, a.month, a.day, tzinfo=timezone.utc).timestamp())
    hi = int(datetime(b.year, b.month, b.day, tzinfo=timezone.utc).timestamp()) + 86399
    params = {
        "query": query, "tags": "story",
        "numericFilters": f"created_at_i>{lo},created_at_i<{hi}",
        "hitsPerPage": 100,
    }
    for i in range(tries):
        try:
            r = S.get(API, params=params, timeout=45)
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            r.raise_for_status()
            return r.json().get("hits", [])
        except Exception:  # noqa: BLE001
            if i == tries - 1:
                return []
            time.sleep(1.5 * (i + 1))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/news/headlines")
    ap.add_argument("--start", default="2017-08-17")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-per-coin", type=int, default=24)
    ap.add_argument("--sleep", type=float, default=0.12)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = (date.fromisoformat(args.end) if args.end
           else datetime.now(timezone.utc).date() - timedelta(days=1))
    os.makedirs(args.out, exist_ok=True)

    marker = os.path.join(args.out, "_done_weeks_hn.txt")
    done = set()
    if os.path.exists(marker):
        done = {l.strip() for l in open(marker) if l.strip()}

    weeks = [(a, b) for a, b in week_windows(start, end)
             if a.isoformat() not in done]
    weeks.reverse()   # newest first
    print(f"[hn] {len(weeks)} weeks to fetch ({len(done)} done)", flush=True)

    writers: dict[int, gzip.GzipFile] = {}
    mk = open(marker, "a")
    t0 = time.time()
    n_records = 0

    for wi, (a, b) in enumerate(weeks, 1):
        # coin -> day -> list of headlines
        buckets: dict[str, dict[date, list[dict]]] = defaultdict(lambda: defaultdict(list))
        seen: dict[str, set] = defaultdict(set)

        for coin, terms in QUERIES.items():
            for q in terms + SHARED:
                for hit in fetch(q, a, b):
                    title = (hit.get("title") or "").strip()
                    if len(title) < 15:
                        continue
                    ts = hit.get("created_at_i")
                    if not ts:
                        continue
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    d = dt.date()
                    if not (a <= d <= b):
                        continue
                    # shared terms only count if the title is actually on-topic
                    if q in SHARED:
                        tl = title.lower()
                        if not any(t in tl for t in terms):
                            continue
                    key = title.lower()[:90]
                    if key in seen[coin]:
                        continue
                    seen[coin].add(key)
                    buckets[coin][d].append({
                        "t": title,
                        "src": (hit.get("url") or "news.ycombinator.com")
                        .split("/")[2] if hit.get("url") else "news.ycombinator.com",
                        "ts": dt.strftime("%Y%m%dT%H%M%SZ"),
                    })
                time.sleep(args.sleep)

        for coin, days in buckets.items():
            for d, items in days.items():
                if not items:
                    continue
                items.sort(key=lambda h: h["ts"], reverse=True)
                items = items[:args.max_per_coin]
                if d.year not in writers:
                    writers[d.year] = gzip.open(
                        os.path.join(args.out, f"{d.year}.jsonl.gz"), "at",
                        compresslevel=6)
                writers[d.year].write(json.dumps({
                    "date": d.isoformat(), "coin": coin,
                    "n": len(items), "headlines": items,
                }, ensure_ascii=False) + "\n")
                n_records += len(items)

        mk.write(a.isoformat() + "\n")
        if wi % 10 == 0:
            for w in writers.values():
                w.flush()
            mk.flush()
            rate = wi / max(time.time() - t0, 1e-9)
            print(f"  {wi}/{len(weeks)} weeks  {n_records} headlines  "
                  f"eta {(len(weeks)-wi)/max(rate,1e-9)/60:.1f} min", flush=True)

    for w in writers.values():
        w.close()
    mk.close()
    print(f"[hn] done: {n_records} headlines", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
