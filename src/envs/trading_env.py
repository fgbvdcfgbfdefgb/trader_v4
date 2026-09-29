"""
One episode == one UTC trading day on one randomly chosen coin.

Account model
-------------
* starts every episode with a fresh $2,000 of fake cash
* the agent picks a TARGET EXPOSURE (fraction of equity held in the coin)
  rather than raw buy/sell, which keeps the action space stationary across
  coins whose prices differ by three orders of magnitude
* rebalancing costs `fee_bps` + `slippage_bps` on the traded notional, so
  churning every bar is actively punished
* the agent only acts every `decision_interval` minutes (default 5), but
  equity is marked to market every minute

Reward = change in log-equity, scaled, minus a drawdown penalty and a small
turnover penalty.  Log-equity makes the reward scale-free, which matters when
episodes hop between a $100k BTC day and a $60 LTC day.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from ..data.features import (WARMUP, build_minute_features, day_price_context,
                             prior_day_context, time_features)
from ..llm.advisor import SIGNAL_DIM, AdvisorSignal, numeric_prior, select_headlines

LAGS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
ACCOUNT_DIM = 8
TIME_DIM = 2
COIN_DIM = 3


@dataclass
class EpisodeRecord:
    """Everything the PNG report needs, captured during the rollout."""
    symbol: str = ""
    coin: str = ""
    day: date | None = None
    ts: np.ndarray = field(default_factory=lambda: np.empty(0))
    price: np.ndarray = field(default_factory=lambda: np.empty(0))
    equity: np.ndarray = field(default_factory=lambda: np.empty(0))
    position: np.ndarray = field(default_factory=lambda: np.empty(0))
    action: np.ndarray = field(default_factory=lambda: np.empty(0))
    reward: np.ndarray = field(default_factory=lambda: np.empty(0))
    trade_idx: list = field(default_factory=list)
    trade_side: list = field(default_factory=list)
    trade_price: list = field(default_factory=list)
    trade_notional: list = field(default_factory=list)
    start_balance: float = 0.0
    end_balance: float = 0.0
    fees_paid: float = 0.0
    n_trades: int = 0
    max_drawdown: float = 0.0
    buy_hold_end: float = 0.0
    advisor: dict = field(default_factory=dict)
    headlines: list = field(default_factory=list)

    @property
    def pnl(self) -> float:
        return self.end_balance - self.start_balance

    @property
    def pnl_pct(self) -> float:
        return 100.0 * (self.end_balance / max(self.start_balance, 1e-9) - 1.0)


class TradingEnv:
    """Single-episode trading environment. Not gym-dependent by design."""

    def __init__(self, dataset, cache=None, *, start_balance: float = 2000.0,
                 fee_bps: float = 10.0, slippage_bps: float = 2.0,
                 decision_interval: int = 5, allow_short: bool = False,
                 n_exposure_levels: int = 5, reward_scale: float = 100.0,
                 dd_penalty: float = 0.5, turnover_penalty: float = 0.02,
                 use_llm: bool = True, day_pool: dict | None = None,
                 seed: int = 0):
        self.ds = dataset
        self.cache = cache
        self.start_balance = start_balance
        self.fee = (fee_bps + slippage_bps) / 10_000.0
        self.interval = max(1, decision_interval)
        self.reward_scale = reward_scale
        self.dd_penalty = dd_penalty
        self.turnover_penalty = turnover_penalty
        self.use_llm = use_llm
        self.rng = np.random.default_rng(seed)
        self.day_pool = day_pool if day_pool is not None else dataset.days
        self.symbols = [s for s in self.day_pool if self.day_pool[s]]

        if allow_short:
            self.exposures = np.linspace(-1.0, 1.0, 2 * n_exposure_levels - 1)
        else:
            self.exposures = np.linspace(0.0, 1.0, n_exposure_levels)
        self.n_actions = len(self.exposures)

        self.daily_dim = len(self.ds.daily.cols)
        self.obs_dim = (len(LAGS) * 17 + ACCOUNT_DIM + TIME_DIM
                        + self.daily_dim + SIGNAL_DIM + COIN_DIM)
        self._reset_state()

    # ------------------------------------------------------------------ #
    def _reset_state(self):
        self.t = 0
        self.cash = self.start_balance
        self.units = 0.0
        self.equity = self.start_balance
        self.peak = self.start_balance
        self.max_dd = 0.0
        self.n_trades = 0
        self.fees_paid = 0.0
        self.entry_price = 0.0
        self.time_in_pos = 0
        self.rec = EpisodeRecord()

    # ------------------------------------------------------------------ #
    def sample_day(self) -> tuple[str, date]:
        s = self.symbols[self.rng.integers(len(self.symbols))]
        days = self.day_pool[s]
        return s, days[self.rng.integers(len(days))]

    def reset(self, symbol: str | None = None, day: date | None = None):
        if symbol is None or day is None:
            symbol, day = self.sample_day()
        self.symbol = symbol
        self.coin = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "LTCUSDT": "LTC"}[symbol]
        self.day = day

        bars = self.ds.minute.day_bars(symbol, day, warmup=WARMUP)
        self.feat = build_minute_features(bars)
        self.close = bars["close"].to_numpy(np.float64)
        self.ts = bars["open_time"].to_numpy(np.int64)
        self.offset = len(bars) - (self.ds.minute.day_index[symbol][day][1]
                                   - self.ds.minute.day_index[symbol][day][0])
        self.n_steps_total = len(self.close) - self.offset
        self.tf = time_features(len(self.close),
                                start_minute=-self.offset % 1440)

        self.daily_vec = self.ds.daily.norm_row(day - timedelta(days=1))
        self.signal = self._get_signal()
        self.signal_vec = self.signal.to_vector()

        self._reset_state()
        self.t = self.offset
        self.equity_hist = [self.equity]
        self.price_hist = [float(self.close[self.t])]
        self.pos_hist = [0.0]
        self.act_hist = []
        self.rew_hist = []
        self.ts_hist = [int(self.ts[self.t])]
        self.first_price = float(self.close[self.t])
        return self._obs()

    # ------------------------------------------------------------------ #
    def _get_signal(self) -> AdvisorSignal:
        """
        Cache hit -> the real LLM verdict.  Miss -> deterministic prior, and
        the day is queued for the CPU daemon.  Either way the observation has
        the same shape, so a resumed run never sees a geometry change.
        """
        heads = select_headlines(self.ds.news, self.coin, self.day)
        self._headlines = heads
        if self.use_llm and self.cache is not None:
            sig = self.cache.get(self.coin, self.day)
            if sig is not None:
                sig.n_headlines = len(heads)
                return sig
            self.cache.request(self.coin, self.day)
        ctx = prior_day_context(self.ds.daily, self.coin, self.day)
        ctx.update(day_price_context(self.ds.minute, self.symbol, self.day))
        return numeric_prior(heads, ctx)

    # ------------------------------------------------------------------ #
    def _obs(self) -> np.ndarray:
        idx = np.clip(self.t - np.array(LAGS), 0, len(self.feat) - 1)
        px = self.feat[idx].reshape(-1)

        price = self.close[self.t]
        pos_val = self.units * price
        exposure = pos_val / max(self.equity, 1e-9)
        unreal = (price / self.entry_price - 1.0) if self.entry_price > 0 else 0.0
        steps_left = (len(self.close) - self.t) / max(self.n_steps_total, 1)
        acct = np.array([
            exposure,
            np.clip(unreal * 20.0, -5, 5),
            np.tanh(self.time_in_pos / 120.0),
            self.cash / max(self.equity, 1e-9),
            np.clip((self.equity / self.start_balance - 1.0) * 10.0, -5, 5),
            -np.clip(self.max_dd * 10.0, 0, 5),
            np.tanh(self.n_trades / 10.0),
            steps_left,
        ], dtype=np.float32)

        coin_oh = np.zeros(COIN_DIM, dtype=np.float32)
        coin_oh[["BTC", "ETH", "LTC"].index(self.coin)] = 1.0

        return np.concatenate([
            px, acct, self.tf[self.t], self.daily_vec, self.signal_vec, coin_oh
        ]).astype(np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action: int):
        target = float(self.exposures[int(action)])
        price = float(self.close[self.t])

        # ---- rebalance to the target exposure ----
        desired_val = target * self.equity
        current_val = self.units * price
        delta_val = desired_val - current_val
        turnover = 0.0
        if abs(delta_val) > max(2.0, 0.01 * self.equity):
            cost = abs(delta_val) * self.fee
            self.cash -= delta_val + cost
            self.units += delta_val / price
            self.fees_paid += cost
            self.n_trades += 1
            turnover = abs(delta_val) / max(self.equity, 1e-9)
            self.rec.trade_idx.append(self.t - self.offset)
            self.rec.trade_side.append(1 if delta_val > 0 else -1)
            self.rec.trade_price.append(price)
            self.rec.trade_notional.append(abs(delta_val))
            if self.units * price > 1e-6 and self.entry_price == 0.0:
                self.entry_price = price
            elif abs(self.units * price) < 1e-6:
                self.entry_price = 0.0
                self.time_in_pos = 0

        prev_equity = self.equity

        # ---- advance `interval` minutes, marking to market each bar ----
        end = min(self.t + self.interval, len(self.close) - 1)
        for k in range(self.t + 1, end + 1):
            p = float(self.close[k])
            self.equity = self.cash + self.units * p
            self.peak = max(self.peak, self.equity)
            dd = 1.0 - self.equity / max(self.peak, 1e-9)
            self.max_dd = max(self.max_dd, dd)
            self.equity_hist.append(self.equity)
            self.price_hist.append(p)
            self.pos_hist.append(self.units * p / max(self.equity, 1e-9))
            self.ts_hist.append(int(self.ts[k]))
        self.t = end
        if abs(self.units) > 1e-12:
            self.time_in_pos += self.interval

        # ---- reward ----
        growth = np.log(max(self.equity, 1e-6) / max(prev_equity, 1e-6))
        dd_now = 1.0 - self.equity / max(self.peak, 1e-9)
        reward = (self.reward_scale * growth
                  - self.dd_penalty * dd_now
                  - self.turnover_penalty * turnover)
        reward = float(np.clip(reward, -20.0, 20.0))

        self.act_hist.append(int(action))
        self.rew_hist.append(reward)

        done = self.t >= len(self.close) - 1
        if done:
            # liquidate at the close so PnL is realised and comparable
            if abs(self.units) > 1e-12:
                p = float(self.close[self.t])
                val = self.units * p
                cost = abs(val) * self.fee
                self.cash += val - cost
                self.fees_paid += cost
                self.units = 0.0
                self.equity = self.cash
                self.equity_hist[-1] = self.equity
            self._finalise()

        return self._obs(), reward, done, self._info()

    # ------------------------------------------------------------------ #
    def _finalise(self):
        r = self.rec
        r.symbol, r.coin, r.day = self.symbol, self.coin, self.day
        r.ts = np.array(self.ts_hist, dtype=np.int64)
        r.price = np.array(self.price_hist, dtype=np.float64)
        r.equity = np.array(self.equity_hist, dtype=np.float64)
        r.position = np.array(self.pos_hist, dtype=np.float32)
        r.action = np.array(self.act_hist, dtype=np.int32)
        r.reward = np.array(self.rew_hist, dtype=np.float32)
        r.start_balance = self.start_balance
        r.end_balance = self.equity
        r.fees_paid = self.fees_paid
        r.n_trades = self.n_trades
        r.max_drawdown = self.max_dd
        r.buy_hold_end = self.start_balance * (
            float(self.close[-1]) / max(self.first_price, 1e-9))
        r.advisor = {
            "bias": self.signal.bias, "conviction": self.signal.conviction,
            "volatility": self.signal.volatility,
            "event_risk": self.signal.event_risk, "tag": self.signal.tag,
            "source": self.signal.source, "rationale": self.signal.rationale,
            "n_headlines": self.signal.n_headlines, "tone": self.signal.tone,
        }
        r.headlines = [h.get("t", "") for h in getattr(self, "_headlines", [])[:8]]

    def _info(self) -> dict:
        return {
            "equity": self.equity, "pnl": self.equity - self.start_balance,
            "n_trades": self.n_trades, "max_dd": self.max_dd,
            "coin": self.coin, "day": self.day,
        }

    @property
    def record(self) -> EpisodeRecord:
        return self.rec


class VecTradingEnv:
    """
    N independent episodes stepped in lockstep inside one process.

    Subprocess workers would be wrong here: the parquet + news stores are
    already resident and shared, so the only cost is the python loop, while
    subprocesses would duplicate ~400 MB of data per worker.
    """

    def __init__(self, n: int, make_env):
        self.envs = [make_env(i) for i in range(n)]
        self.n = n
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions
        self._done = np.zeros(n, dtype=bool)

    def reset(self) -> np.ndarray:
        self._done[:] = False
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        obs, rew, done, infos = [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, d, inf = e.step(int(actions[i]))
            obs.append(o)
            rew.append(r)
            done.append(d)
            infos.append(inf)
        return (np.stack(obs), np.array(rew, dtype=np.float32),
                np.array(done, dtype=bool), infos)

    def records(self):
        return [e.record for e in self.envs]
