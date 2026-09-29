"""
Offline data access.  Nothing in this module touches the network.

Layout it expects (all produced by scripts/*.py and committed to the repo):

    data/minute/<SYMBOL>/<YEAR>.parquet   1-minute OHLCV
    data/daily/market_daily.parquet       26 daily context features
    data/news/news_tone.parquet           GDELT tone/volume per coin
    data/news/headlines/<YEAR>.jsonl.gz   per (day, coin) headline buckets
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "LTCUSDT"]
COIN_OF = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "LTCUSDT": "LTC"}
SYMBOL_OF = {v: k for k, v in COIN_OF.items()}
MIN_BARS_PER_DAY = 1200          # tolerate exchange outages, reject thin days
BARS_PER_DAY = 1440


class MinuteStore:
    """Per-(symbol, year) parquet with a lazy, LRU-bounded year cache."""

    def __init__(self, root: str, symbols: list[str] | None = None,
                 max_years_cached: int = 6):
        self.root = root
        self.symbols = symbols or SYMBOLS
        self._cache: dict[tuple[str, int], pd.DataFrame] = {}
        self._order: list[tuple[str, int]] = []
        self.max_years_cached = max_years_cached
        self.day_index: dict[str, dict[date, tuple[int, int]]] = {}
        self._build_index()

    # ------------------------------------------------------------------ #
    def _years(self, symbol: str) -> list[int]:
        pat = os.path.join(self.root, symbol, "*.parquet")
        return sorted(int(os.path.basename(p)[:-8]) for p in glob.glob(pat))

    def _build_index(self) -> None:
        """Map every usable UTC day -> (row_start, row_end) inside its year."""
        cache_path = os.path.join(self.root, "_day_index.json")
        if os.path.exists(cache_path):
            try:
                raw = json.load(open(cache_path))
                self.day_index = {
                    s: {date.fromisoformat(d): tuple(v) for d, v in days.items()}
                    for s, days in raw.items()}
                if all(s in self.day_index for s in self.symbols):
                    return
            except Exception:  # noqa: BLE001
                pass

        for s in self.symbols:
            self.day_index[s] = {}
            for y in self._years(s):
                df = self._load_year(s, y)
                days = (df["open_time"].values // 86_400_000).astype(np.int64)
                # contiguous run-length encode
                change = np.flatnonzero(np.diff(days)) + 1
                starts = np.concatenate(([0], change))
                ends = np.concatenate((change, [len(days)]))
                for st, en in zip(starts, ends):
                    if en - st < MIN_BARS_PER_DAY:
                        continue
                    d = datetime.fromtimestamp(int(days[st]) * 86400,
                                               tz=timezone.utc).date()
                    self.day_index[s][d] = (int(st), int(en))
        try:
            json.dump({s: {d.isoformat(): list(v) for d, v in days.items()}
                       for s, days in self.day_index.items()},
                      open(cache_path, "w"))
        except Exception:  # noqa: BLE001
            pass

    def _load_year(self, symbol: str, year: int) -> pd.DataFrame:
        key = (symbol, year)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.root, symbol, f"{year}.parquet")
        df = pd.read_parquet(path)
        self._cache[key] = df
        self._order.append(key)
        while len(self._order) > self.max_years_cached:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return df

    # ------------------------------------------------------------------ #
    def available_days(self, symbol: str) -> list[date]:
        return sorted(self.day_index.get(symbol, {}))

    def day_bars(self, symbol: str, d: date, warmup: int = 0) -> pd.DataFrame:
        """
        Bars for `d`, optionally prefixed with `warmup` bars from before it.

        The warmup slice is strictly earlier in time, so indicators computed on
        it cannot leak future information into the episode.
        """
        st, en = self.day_index[symbol][d]
        df = self._load_year(symbol, d.year)
        lo = st
        pre = None
        if warmup > 0:
            lo = st - warmup
            if lo < 0:
                prev_year = d.year - 1
                need = -lo
                try:
                    pdf = self._load_year(symbol, prev_year)
                    pre = pdf.iloc[max(0, len(pdf) - need):]
                except Exception:  # noqa: BLE001
                    pre = None
                lo = 0
        out = df.iloc[lo:en]
        if pre is not None and len(pre):
            out = pd.concat([pre, out], ignore_index=True)
        return out.reset_index(drop=True)


class DailyStore:
    """Daily context: market_daily.parquet + GDELT tone, indexed by date."""

    def __init__(self, daily_path: str, tone_path: str | None = None):
        df = pd.read_parquet(daily_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if tone_path and os.path.exists(tone_path):
            t = pd.read_parquet(tone_path)
            t["date"] = pd.to_datetime(t["date"]).dt.date
            df = df.merge(t, on="date", how="outer").sort_values("date")
        self.df = df.reset_index(drop=True)
        self.cols = [c for c in self.df.columns if c != "date"]
        self._by_date = {d: i for i, d in enumerate(self.df["date"])}
        self._mat = self.df[self.cols].to_numpy(dtype=np.float32)
        # robust normalisation constants (median / IQR), computed once
        finite = np.where(np.isfinite(self._mat), self._mat, np.nan)
        self._med = np.nanmedian(finite, axis=0)
        q1 = np.nanpercentile(finite, 25, axis=0)
        q3 = np.nanpercentile(finite, 75, axis=0)
        self._iqr = np.where((q3 - q1) > 1e-9, q3 - q1, 1.0)

    def row(self, d: date) -> np.ndarray:
        """Raw feature row for day d (NaN-filled if missing)."""
        i = self._by_date.get(d)
        if i is None:
            return np.full(len(self.cols), np.nan, dtype=np.float32)
        return self._mat[i]

    def norm_row(self, d: date) -> np.ndarray:
        r = self.row(d)
        z = (r - self._med) / self._iqr
        z = np.clip(z, -5.0, 5.0)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def get(self, d: date, col: str, default=float("nan")) -> float:
        if col not in self.cols:
            return default
        i = self._by_date.get(d)
        if i is None:
            return default
        v = self._mat[i, self.cols.index(col)]
        return float(v) if np.isfinite(v) else default


class NewsStore:
    """(coin, day) -> list of headline dicts.  Loaded once, held in RAM (~20 MB)."""

    def __init__(self, root: str):
        self.by_key: dict[tuple[str, date], list[dict]] = defaultdict(list)
        hdir = os.path.join(root, "headlines")
        for path in sorted(glob.glob(os.path.join(hdir, "*.jsonl.gz"))):
            try:
                with gzip.open(path, "rt") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        d = date.fromisoformat(rec["date"])
                        self.by_key[(rec["coin"], d)].extend(rec.get("headlines", []))
            except Exception:  # noqa: BLE001
                continue
        self.n_records = sum(len(v) for v in self.by_key.values())

    def get(self, coin: str, d: date) -> list[dict]:
        return self.by_key.get((coin, d), [])

    def coverage(self) -> dict:
        days = defaultdict(set)
        for (coin, d) in self.by_key:
            days[coin].add(d)
        return {c: len(v) for c, v in days.items()}


class Dataset:
    """Bundles the three stores and exposes the day universe per coin."""

    def __init__(self, root: str = "data", symbols: list[str] | None = None,
                 min_date: str | None = None, max_date: str | None = None,
                 verbose: bool = True):
        self.root = root
        self.symbols = symbols or SYMBOLS
        self.minute = MinuteStore(os.path.join(root, "minute"), self.symbols)
        self.daily = DailyStore(
            os.path.join(root, "daily", "market_daily.parquet"),
            os.path.join(root, "news", "news_tone.parquet"))
        self.news = NewsStore(os.path.join(root, "news"))

        lo = date.fromisoformat(min_date) if min_date else date(2000, 1, 1)
        hi = date.fromisoformat(max_date) if max_date else date(2100, 1, 1)
        self.days: dict[str, list[date]] = {}
        for s in self.symbols:
            ds = [d for d in self.minute.available_days(s) if lo <= d <= hi]
            # need a previous day for warm-up + strictly-prior context
            have = set(self.minute.available_days(s))
            self.days[s] = [d for d in ds if (d - timedelta(days=1)) in have]
        if verbose:
            tot = sum(len(v) for v in self.days.values())
            print(f"[data] {tot} tradable days  "
                  + "  ".join(f"{COIN_OF[s]}={len(self.days[s])}" for s in self.symbols),
                  flush=True)
            print(f"[data] daily features: {len(self.daily.cols)}  "
                  f"news headlines: {self.news.n_records} "
                  f"({self.news.coverage()})", flush=True)

    def split(self, train_frac: float = 0.85):
        """Chronological split -- the tail is held out, never trained on."""
        tr, va = {}, {}
        for s, ds in self.days.items():
            k = int(len(ds) * train_frac)
            tr[s], va[s] = ds[:k], ds[k:]
        return tr, va
