"""
Feature engineering for the minute-level observation.

Everything is causal: feature at bar t uses bars <= t only.  The warm-up
prefix supplied by MinuteStore.day_bars() comes from *earlier* days, so
indicators are already converged when the episode's first bar is reached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP = 240          # minutes of history needed before the episode starts
N_PRICE_FEATURES = 17


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    au = _ema(up, period)
    ad = _ema(dn, period)
    rs = au / np.maximum(ad, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(w, min_periods=2).std().bfill().to_numpy()


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(w, min_periods=1).mean().to_numpy()


def build_minute_features(df: pd.DataFrame) -> np.ndarray:
    """
    df: OHLCV bars including the warm-up prefix.
    returns float32 [T, N_PRICE_FEATURES], already scaled to roughly [-3, 3].
    """
    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    openp = df["open"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)
    tbb = df["taker_buy_base"].to_numpy(np.float64) if "taker_buy_base" in df else vol * 0.5
    trades = df["trades"].to_numpy(np.float64) if "trades" in df else np.ones_like(vol)

    logc = np.log(np.maximum(close, 1e-12))
    r1 = np.diff(logc, prepend=logc[0])

    # multi-horizon momentum, volatility-normalised
    sd60 = np.maximum(_rolling_std(r1, 60), 1e-8)
    feats = [
        np.clip(r1 / sd60, -6, 6),
        np.clip((logc - np.roll(logc, 5)) / (sd60 * np.sqrt(5)), -6, 6),
        np.clip((logc - np.roll(logc, 15)) / (sd60 * np.sqrt(15)), -6, 6),
        np.clip((logc - np.roll(logc, 60)) / (sd60 * np.sqrt(60)), -6, 6),
        np.clip((logc - np.roll(logc, 240)) / (sd60 * np.sqrt(240)), -6, 6),
    ]

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    ema120 = _ema(close, 120)
    feats.append(np.clip((close / np.maximum(ema12, 1e-12) - 1) / 0.002, -6, 6))
    feats.append(np.clip((close / np.maximum(ema120, 1e-12) - 1) / 0.01, -6, 6))
    macd = (ema12 - ema26) / np.maximum(close, 1e-12)
    feats.append(np.clip(macd / 0.002, -6, 6))

    feats.append((_rsi(close, 14) - 50.0) / 25.0)

    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    atr = _ema(tr, 14) / np.maximum(close, 1e-12)
    feats.append(np.clip((atr - 0.001) / 0.002, -4, 6))

    sd240 = np.maximum(_rolling_std(r1, 240), 1e-8)
    feats.append(np.clip(np.log(sd60 / sd240), -3, 3))

    vmean = np.maximum(_rolling_mean(vol, 120), 1e-9)
    feats.append(np.clip(np.log(np.maximum(vol, 1e-9) / vmean), -4, 4))
    tmean = np.maximum(_rolling_mean(trades, 120), 1e-9)
    feats.append(np.clip(np.log(np.maximum(trades, 1e-9) / tmean), -4, 4))

    imb = np.where(vol > 1e-12, (2.0 * tbb - vol) / np.maximum(vol, 1e-12), 0.0)
    feats.append(np.clip(imb, -1, 1))
    feats.append(np.clip(_rolling_mean(imb, 30), -1, 1))

    rng = (high - low) / np.maximum(close, 1e-12)
    feats.append(np.clip((rng - 0.001) / 0.002, -4, 6))
    # guard the denominator itself: np.where still evaluates both branches,
    # which warns (and produces NaN) on zero-range bars
    hl = high - low
    body = np.divide(close - openp, hl, out=np.zeros_like(close),
                     where=hl > 1e-12)
    feats.append(np.clip(body, -1, 1))

    out = np.stack(feats, axis=1).astype(np.float32)
    out[:WARMUP] = 0.0  # rolled features are undefined at the very start
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def time_features(n: int, start_minute: int = 0) -> np.ndarray:
    """Cyclical minute-of-day encoding, [n, 2]."""
    m = (np.arange(n) + start_minute) % 1440
    ang = 2 * np.pi * m / 1440.0
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)


def prior_day_context(daily, coin: str, day):
    """
    Scalar facts about the day BEFORE `day`, for the LLM prompt.
    Strictly causal: reads the daily table at day-1 and earlier only.
    """
    from datetime import timedelta
    prev = day - timedelta(days=1)
    ctx = {
        "fear_greed": daily.get(prev, "fear_greed"),
        "funding_rate": daily.get(prev, f"{coin.lower()}_funding_rate"),
        "news_tone": daily.get(prev, f"{coin.lower()}_news_tone"),
        "news_vol": daily.get(prev, f"{coin.lower()}_news_vol"),
    }
    return ctx


def day_price_context(minute_store, symbol: str, day) -> dict:
    """1d / 7d return and realised vol computed from bars strictly before `day`."""
    from datetime import timedelta
    closes = []
    d = day - timedelta(days=1)
    for _ in range(8):
        if d in minute_store.day_index.get(symbol, {}):
            bars = minute_store.day_bars(symbol, d)
            closes.append((d, float(bars["close"].iloc[-1]),
                           float(bars["volume"].sum())))
        d -= timedelta(days=1)
    if not closes:
        return {}
    closes.reverse()
    px = np.array([c[1] for c in closes], dtype=np.float64)
    vols = np.array([c[2] for c in closes], dtype=np.float64)
    out = {"prev_close": float(px[-1])}
    if len(px) >= 2:
        out["ret_1d"] = float(px[-1] / px[-2] - 1.0)
        rets = np.diff(np.log(px))
        out["realised_vol"] = float(np.std(rets)) if len(rets) > 1 else float("nan")
    if len(px) >= 8:
        out["ret_7d"] = float(px[-1] / px[0] - 1.0)
    if len(vols) >= 4 and np.std(vols[:-1]) > 0:
        out["volume_z"] = float((vols[-1] - np.mean(vols[:-1])) / np.std(vols[:-1]))
    return out
