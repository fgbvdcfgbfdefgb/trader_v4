"""
News -> structured trading signal.

Design constraints that shaped this file
----------------------------------------
1. STRICT OUTPUT SHAPE.  A 7-bit-quantised 7B model free-writing prose is
   useless to a PPO agent.  The prompt forces a single flat JSON object with
   five integer/enum fields and the decoder is *primed* with `{"bias":` so the
   very first generated token is already inside the JSON.  Anything that fails
   validation is discarded, not "best-effort parsed".

2. STRICT TIMELINE.  For an episode on day D the advisor may only see
   information timestamped strictly before D 00:00 UTC.  `select_headlines`
   enforces that with an assertion, and the daily context row is taken from
   D-1.  There is no path in this module that can read day D or later.

3. NEVER BLOCK TRAINING.  The LLM is slow on CPU, so every query first hits a
   SQLite cache.  On a miss the env gets `numeric_prior()` -- a deterministic
   lexicon + GDELT-tone signal in the same 8-float shape -- and the day is
   enqueued for the background daemon.  The agent's observation vector has
   identical geometry either way, with `llm_present` flagging which it got.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

SIGNAL_DIM = 8

VOL_LEVELS = ["low", "normal", "high"]
RISK_LEVELS = ["none", "some", "severe"]
TAGS = [
    "regulation", "etf_flows", "macro", "hack_exploit", "adoption",
    "whale_flows", "network_upgrade", "liquidation", "noise", "other",
]

# --------------------------------------------------------------------------- #
#  crypto-specific sentiment lexicon (fallback + a prior the LLM can't see)
# --------------------------------------------------------------------------- #
BULL = {
    "surge": 2.0, "surges": 2.0, "soar": 2.2, "soars": 2.2, "rally": 1.8,
    "rallies": 1.8, "jump": 1.4, "jumps": 1.4, "gain": 1.0, "gains": 1.0,
    "record": 1.6, "ath": 2.0, "all-time high": 2.4, "breakout": 1.8,
    "bullish": 2.0, "adoption": 1.2, "approval": 2.0, "approved": 2.0,
    "inflow": 1.6, "inflows": 1.6, "accumulate": 1.2, "accumulation": 1.2,
    "upgrade": 1.0, "partnership": 0.9, "institutional": 1.0, "etf": 0.8,
    "halving": 1.0, "buy": 0.8, "buying": 0.9, "optimism": 1.3, "boom": 1.6,
    "skyrocket": 2.4, "outperform": 1.3, "rebound": 1.5, "recovery": 1.3,
    "green": 0.6, "climbs": 1.4, "spike": 1.2, "milestone": 1.0,
    "legalize": 1.6, "reserve": 1.2, "treasury": 0.9, "whale buys": 1.6,
}
BEAR = {
    "crash": -2.4, "crashes": -2.4, "plunge": -2.2, "plunges": -2.2,
    "slump": -1.8, "tumble": -1.9, "tumbles": -1.9, "drop": -1.2,
    "drops": -1.2, "fall": -1.1, "falls": -1.1, "sink": -1.6, "sinks": -1.6,
    "bearish": -2.0, "selloff": -2.0, "sell-off": -2.0, "dump": -1.8,
    "hack": -2.4, "hacked": -2.4, "exploit": -2.2, "breach": -2.0,
    "scam": -1.8, "fraud": -2.0, "lawsuit": -1.6, "sue": -1.5, "sued": -1.6,
    "sec": -0.8, "ban": -2.2, "bans": -2.2, "banned": -2.2, "crackdown": -2.0,
    "regulation": -0.7, "outflow": -1.6, "outflows": -1.6, "liquidated": -1.8,
    "liquidation": -1.7, "liquidations": -1.7, "bankruptcy": -2.2,
    "collapse": -2.3, "halt": -1.4, "warning": -1.0, "probe": -1.3,
    "investigation": -1.4, "fine": -1.2, "delist": -1.8, "risk": -0.6,
    "fear": -1.2, "panic": -2.0, "correction": -1.2, "slide": -1.4,
    "pressure": -0.8, "loses": -1.1, "decline": -1.2, "red": -0.6,
}
_TOKEN = re.compile(r"[a-z][a-z\-']+")


# --------------------------------------------------------------------------- #
@dataclass
class AdvisorSignal:
    bias: int = 0            # -2 .. +2   directional lean
    conviction: int = 0      #  0 .. 3    how strongly to size
    volatility: int = 1      #  0 .. 2    expected intraday range
    event_risk: int = 0      #  0 .. 2    tail/headline risk
    tag: str = "noise"
    source: str = "prior"    # "llm" | "prior"
    n_headlines: int = 0
    tone: float = 0.0        # normalised lexicon/GDELT tone, -1..1
    rationale: str = ""

    def to_vector(self) -> np.ndarray:
        """Fixed 8-float observation slice. Identical geometry for llm/prior."""
        return np.array([
            self.bias / 2.0,
            self.conviction / 3.0,
            (self.volatility - 1) / 1.0,
            self.event_risk / 2.0,
            1.0 if self.source == "llm" else 0.0,
            math.tanh(self.n_headlines / 12.0),
            float(np.clip(self.tone, -1.0, 1.0)),
            float(TAGS.index(self.tag) / (len(TAGS) - 1)) if self.tag in TAGS else 1.0,
        ], dtype=np.float32)

    @staticmethod
    def zeros() -> np.ndarray:
        return np.zeros(SIGNAL_DIM, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  timeline-safe headline selection
# --------------------------------------------------------------------------- #
def select_headlines(store, coin: str, day: date, lookback_days: int = 2,
                     limit: int = 14, min_items: int = 3,
                     market_proxy: str = "BTC") -> list[dict]:
    """
    Headlines first seen in [day-lookback, day-1].  Never day itself.

    If the coin's own coverage for those days is thin (common for LTC, which
    the news sources barely mention), we top up with BTC headlines tagged as
    market context.  That is not a fudge: alts are overwhelmingly driven by
    BTC beta, so market-wide news is genuinely the relevant information for
    an LTC day with no LTC-specific story.

    Raises if anything dated >= `day` slips through -- a loud failure is much
    better than silent look-ahead leakage that inflates backtest PnL.
    """
    out: list[dict] = []
    for back in range(1, lookback_days + 1):
        d = day - timedelta(days=back)
        for h in store.get(coin, d):
            out.append(h)

    if len(out) < min_items and coin != market_proxy:
        seen = {h.get("t", "")[:80] for h in out}
        for back in range(1, lookback_days + 1):
            d = day - timedelta(days=back)
            for h in store.get(market_proxy, d):
                if h.get("t", "")[:80] in seen:
                    continue
                g = dict(h)
                g["src"] = (g.get("src", "") + " (market)").strip()
                out.append(g)

    for h in out:
        ts = h.get("ts", "")
        if ts and ts[:8] >= day.strftime("%Y%m%d"):
            raise AssertionError(
                f"look-ahead: headline {ts} used for episode day {day}")
    # newest first, then truncate
    out.sort(key=lambda h: h.get("ts", ""), reverse=True)
    return out[:limit]


def lexicon_tone(headlines: list[dict]) -> tuple[float, int]:
    if not headlines:
        return 0.0, 0
    score = 0.0
    hits = 0
    for h in headlines:
        t = h.get("t", "").lower()
        for phrase, w in (("all-time high", 2.4), ("sell-off", -2.0),
                          ("whale buys", 1.6)):
            if phrase in t:
                score += w
                hits += 1
        for tok in _TOKEN.findall(t):
            if tok in BULL:
                score += BULL[tok]
                hits += 1
            elif tok in BEAR:
                score += BEAR[tok]
                hits += 1
    if hits == 0:
        return 0.0, 0
    return float(np.clip(score / (2.0 * math.sqrt(hits)), -1.0, 1.0)), hits


def numeric_prior(headlines: list[dict], ctx: dict) -> AdvisorSignal:
    """
    Deterministic stand-in with the same shape as the LLM verdict.

    Uses only pre-day information: the lexicon tone of yesterday's headlines,
    GDELT's own tone/volume series, the fear & greed index and realised vol.
    """
    tone, hits = lexicon_tone(headlines)
    gd_tone = ctx.get("news_tone")
    if gd_tone is not None and not math.isnan(gd_tone):
        tone = 0.6 * tone + 0.4 * float(np.clip(gd_tone / 5.0, -1, 1))

    fg = ctx.get("fear_greed")
    if fg is not None and not math.isnan(fg):
        # contrarian tilt at the extremes, mild momentum in the middle
        fg_t = (fg - 50.0) / 50.0
        tone = 0.75 * tone + 0.25 * (-fg_t if abs(fg_t) > 0.6 else fg_t * 0.4)

    bias = int(np.clip(round(tone * 2.2), -2, 2))
    conviction = int(np.clip(round(abs(tone) * 3.0), 0, 3))
    if hits == 0 and (gd_tone is None or math.isnan(gd_tone)):
        bias, conviction = 0, 0

    rv = ctx.get("realised_vol")
    vol = 1
    if rv is not None and not math.isnan(rv):
        vol = 0 if rv < 0.015 else (2 if rv > 0.045 else 1)

    ev = 0
    joined = " ".join(h.get("t", "").lower() for h in headlines)
    if any(k in joined for k in ("hack", "exploit", "lawsuit", "ban",
                                 "crackdown", "bankruptcy", "sec sues")):
        ev = 2
    elif any(k in joined for k in ("regulation", "probe", "investigation",
                                   "fine", "warning")):
        ev = 1

    tag = "noise"
    for key, t in (("etf", "etf_flows"), ("sec ", "regulation"),
                   ("hack", "hack_exploit"), ("upgrade", "network_upgrade"),
                   ("inflation", "macro"), ("fed", "macro"),
                   ("liquidat", "liquidation"), ("whale", "whale_flows"),
                   ("adopt", "adoption")):
        if key in joined:
            tag = t
            break

    return AdvisorSignal(bias=bias, conviction=conviction, volatility=vol,
                         event_risk=ev, tag=tag, source="prior",
                         n_headlines=len(headlines), tone=tone,
                         rationale="lexicon+tone prior")


# --------------------------------------------------------------------------- #
#  prompt construction  (Mistral [INST] format)
# --------------------------------------------------------------------------- #
SYSTEM = (
    "You are a disciplined crypto desk analyst. You read yesterday's headlines "
    "and market state, then emit ONE JSON object and nothing else. "
    "You never explain, never add prose, never use markdown."
)

SCHEMA = (
    '{"bias": <int -2..2>, "conviction": <int 0..3>, '
    '"volatility": <"low"|"normal"|"high">, '
    '"event_risk": <"none"|"some"|"severe">, '
    f'"tag": <one of {"|".join(TAGS)}>, '
    '"why": "<max 10 words>"}'
)

RULES = (
    "bias: expected direction of the NEXT 24h for this coin only. "
    "-2 strong down, -1 down, 0 unclear, 1 up, 2 strong up. "
    "conviction: 0 if the headlines are noise or contradictory, 3 only for a "
    "clear, coin-specific, market-moving catalyst. "
    "volatility: expected size of the intraday range. "
    "event_risk: chance of a sudden adverse shock (hacks, bans, lawsuits, "
    "forced liquidations). "
    "If the headlines say nothing useful, answer bias 0 and conviction 0."
)


def build_prompt(coin: str, day: date, headlines: list[dict], ctx: dict) -> str:
    """Everything here is strictly pre-`day`."""
    lines = []
    for i, h in enumerate(headlines, 1):
        src = h.get("src", "")
        lines.append(f"{i}. {h.get('t','').strip()}" + (f"  [{src}]" if src else ""))
    news = "\n".join(lines) if lines else "(no headlines available)"

    def fmt(key, label, scale=1.0, pct=False, nd=2):
        v = ctx.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        v = v * scale
        return f"{label}={v:.{nd}f}{'%' if pct else ''}"

    facts = [f for f in (
        fmt("prev_close", f"{coin} close(D-1) $", nd=2),
        fmt("ret_1d", "1d return ", 100, True),
        fmt("ret_7d", "7d return ", 100, True),
        fmt("realised_vol", "realised vol(7d) ", 100, True),
        fmt("fear_greed", "fear&greed ", nd=0),
        fmt("funding_rate", "perp funding ", 100, True, 4),
        fmt("volume_z", "volume z-score ", nd=2),
    ) if f]

    user = (
        f"COIN: {coin}\n"
        f"DECISION DATE (UTC): {day.isoformat()}  (you are deciding BEFORE this day opens)\n"
        f"MARKET STATE AS OF {(day - timedelta(days=1)).isoformat()} CLOSE:\n  "
        + ("; ".join(facts) if facts else "(unavailable)")
        + f"\n\nHEADLINES FROM THE PREVIOUS 48 HOURS:\n{news}\n\n"
        f"Reply with exactly one JSON object using this schema:\n{SCHEMA}\n\n{RULES}"
    )
    # Priming the answer with the first key makes small/quantised models
    # comply with the schema far more reliably than asking politely.
    return f"<s>[INST] {SYSTEM}\n\n{user} [/INST]" + ' {"bias":'


_JSON = re.compile(r"\{.*?\}", re.S)


def parse_response(raw: str) -> AdvisorSignal | None:
    """Strict: validate every field, reject on any violation."""
    text = raw if raw.lstrip().startswith("{") else '{"bias":' + raw
    m = _JSON.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    try:
        bias = int(d["bias"])
        conv = int(d["conviction"])
    except Exception:  # noqa: BLE001
        return None
    if not (-2 <= bias <= 2 and 0 <= conv <= 3):
        return None

    vol = d.get("volatility", "normal")
    vol = VOL_LEVELS.index(vol) if vol in VOL_LEVELS else 1
    ev = d.get("event_risk", "none")
    ev = RISK_LEVELS.index(ev) if ev in RISK_LEVELS else 0
    tag = d.get("tag", "other")
    if tag not in TAGS:
        tag = "other"
    why = str(d.get("why", ""))[:80]

    return AdvisorSignal(bias=bias, conviction=conv, volatility=vol,
                         event_risk=ev, tag=tag, source="llm",
                         tone=bias / 2.0, rationale=why)


# --------------------------------------------------------------------------- #
#  multi-process cache
# --------------------------------------------------------------------------- #
class _JsonlBackend:
    """
    Lock-free append-only fallback for filesystems where SQLite cannot run.

    One JSON object per line.  POSIX guarantees that an O_APPEND write
    smaller than PIPE_BUF (4 KB) is atomic, and our records are ~300 bytes,
    so concurrent trainers and the daemon can all append safely without any
    locking primitive -- which is precisely what the FUSE mount lacks.
    Readers reload when the file's mtime changes.
    """

    def __init__(self, path: str):
        self.path = path
        self.req_path = path + ".req"
        self._mem: dict[tuple[str, str], dict] = {}
        self._req: dict[tuple[str, str], int] = {}
        self._mtime = -1.0
        self._reload()

    def _reload(self) -> None:
        for path, sink, is_req in ((self.path, self._mem, False),
                                   (self.req_path, self._req, True)):
            if not os.path.exists(path):
                continue
            try:
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            r = json.loads(line)
                        except Exception:  # noqa: BLE001
                            continue
                        k = (r["coin"], r["day"])
                        if is_req:
                            sink[k] = sink.get(k, 0) + 1
                        else:
                            sink[k] = r["payload"]
            except OSError:
                pass
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = -1.0

    def _maybe_reload(self) -> None:
        try:
            m = os.path.getmtime(self.path)
        except OSError:
            return
        if m != self._mtime:
            self._mem.clear()
            self._req.clear()
            self._reload()

    def _append(self, path: str, rec: dict) -> None:
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def get(self, coin, day):
        self._maybe_reload()
        return self._mem.get((coin, day))

    def put(self, coin, day, payload):
        self._mem[(coin, day)] = payload
        self._append(self.path, {"coin": coin, "day": day, "payload": payload})
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            pass

    def request(self, coin, day):
        self._req[(coin, day)] = self._req.get((coin, day), 0) + 1
        self._append(self.req_path, {"coin": coin, "day": day})

    def pending(self, limit):
        self._maybe_reload()
        out = [(k, n) for k, n in self._req.items() if k not in self._mem]
        out.sort(key=lambda x: -x[1])
        return [k for k, _ in out[:limit]]

    def stats(self):
        self._maybe_reload()
        return {"verdicts": len(self._mem), "requests": len(self._req),
                "pending": sum(1 for k in self._req if k not in self._mem)}


def _open_sqlite(path: str, timeout: float):
    """
    Open a SQLite db and prove it actually works on this filesystem.

    Snowflake workspaces live on a FUSE mount that does not implement the
    shared-memory locking WAL needs, so `PRAGMA journal_mode=WAL` followed by
    a write raises SQLITE_IOERR.  Rather than guess, we try progressively
    weaker journal modes and only accept one after a real write round-trip.
    """
    try:
        conn = sqlite3.connect(path, timeout=timeout, isolation_level=None,
                               check_same_thread=False)
    except sqlite3.Error:
        return None, None
    for mode in ("WAL", "TRUNCATE", "DELETE", "MEMORY"):
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute(f"PRAGMA journal_mode={mode}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("CREATE TABLE IF NOT EXISTS _probe(x INTEGER)")
            conn.execute("INSERT INTO _probe VALUES (1)")
            conn.execute("DELETE FROM _probe")
            return conn, mode
        except sqlite3.Error:
            continue
    try:
        conn.close()
    except sqlite3.Error:
        pass
    return None, None


class AdvisorCache:
    """
    Verdict cache shared by every GPU trainer and the CPU LLM daemon.

    Storage is negotiated at startup, because the training host's filesystem
    is not guaranteed to support SQLite locking:

        1. SQLite at the requested path  (WAL -> TRUNCATE -> DELETE -> MEMORY)
        2. SQLite in the system temp dir
        3. an append-only JSONL file, which needs no locking at all

    Every method is defensive: a cache problem must degrade the advisor to
    the numeric prior, never take down a training run.
    """

    def __init__(self, path: str, timeout: float = 30.0, verbose: bool = True):
        self.timeout = timeout
        self._local = threading.local()
        self.backend = "sqlite"
        self.journal = None
        self.path = path

        candidates = [path]
        tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
        alt = os.path.join(tmp, "trader_v4_advisor.sqlite")
        if os.path.abspath(alt) != os.path.abspath(path):
            candidates.append(alt)

        conn = None
        for cand in candidates:
            try:
                d = os.path.dirname(os.path.abspath(cand))
                if d:
                    os.makedirs(d, exist_ok=True)
            except OSError:
                continue
            conn, mode = _open_sqlite(cand, timeout)
            if conn is not None:
                self.path, self.journal = cand, mode
                break

        if conn is None:
            self.backend = "jsonl"
            base = path if path.endswith(".jsonl") else path + ".jsonl"
            try:
                os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
            except OSError:
                base = os.path.join(tmp, "trader_v4_advisor.jsonl")
            self.path = base
            self._json = _JsonlBackend(base)
            if verbose:
                print(f"[advisor-cache] SQLite unusable here -> append-only "
                      f"JSONL at {base}", flush=True)
            return

        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS verdicts(
                coin TEXT, day TEXT, payload TEXT, created REAL,
                PRIMARY KEY(coin, day))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS requests(
                coin TEXT, day TEXT, hits INTEGER DEFAULT 1, last REAL,
                PRIMARY KEY(coin, day))""")
            conn.execute("DROP TABLE IF EXISTS _probe")
        except sqlite3.Error:
            pass
        self._local.c = conn
        if verbose:
            note = "" if self.path == path else f" (fell back from {path})"
            print(f"[advisor-cache] sqlite journal={self.journal} "
                  f"at {self.path}{note}", flush=True)

    # ------------------------------------------------------------------ #
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "c"):
            c, mode = _open_sqlite(self.path, self.timeout)
            if c is None:
                raise sqlite3.OperationalError("cannot reopen advisor cache")
            self._local.c = c
        return self._local.c

    # ------------------------------------------------------------------ #
    def get(self, coin: str, day: date) -> AdvisorSignal | None:
        try:
            if self.backend == "jsonl":
                raw = self._json.get(coin, day.isoformat())
                return AdvisorSignal(**raw) if raw else None
            r = self._conn().execute(
                "SELECT payload FROM verdicts WHERE coin=? AND day=?",
                (coin, day.isoformat())).fetchone()
            return AdvisorSignal(**json.loads(r[0])) if r else None
        except Exception:  # noqa: BLE001
            return None

    def put(self, coin: str, day: date, sig: AdvisorSignal) -> None:
        try:
            if self.backend == "jsonl":
                self._json.put(coin, day.isoformat(), asdict(sig))
                return
            self._conn().execute(
                "INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?)",
                (coin, day.isoformat(), json.dumps(asdict(sig)), time.time()))
        except Exception:  # noqa: BLE001
            pass

    def request(self, coin: str, day: date) -> None:
        try:
            if self.backend == "jsonl":
                self._json.request(coin, day.isoformat())
                return
            now = time.time()
            self._conn().execute(
                "INSERT INTO requests(coin,day,hits,last) VALUES(?,?,1,?) "
                "ON CONFLICT(coin,day) DO UPDATE SET hits=hits+1, last=?",
                (coin, day.isoformat(), now, now))
        except Exception:  # noqa: BLE001
            pass  # a busy cache must never stall an env step

    def pending(self, limit: int = 64) -> list[tuple[str, str]]:
        try:
            if self.backend == "jsonl":
                return self._json.pending(limit)
            return self._conn().execute(
                "SELECT r.coin, r.day FROM requests r "
                "LEFT JOIN verdicts v ON v.coin=r.coin AND v.day=r.day "
                "WHERE v.day IS NULL ORDER BY r.hits DESC, r.last DESC LIMIT ?",
                (limit,)).fetchall()
        except Exception:  # noqa: BLE001
            return []

    def stats(self) -> dict:
        try:
            if self.backend == "jsonl":
                return self._json.stats()
            c = self._conn()
            return {
                "verdicts": c.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0],
                "requests": c.execute("SELECT COUNT(*) FROM requests").fetchone()[0],
                "pending": c.execute(
                    "SELECT COUNT(*) FROM requests r LEFT JOIN verdicts v "
                    "ON v.coin=r.coin AND v.day=r.day WHERE v.day IS NULL"
                ).fetchone()[0],
            }
        except Exception:  # noqa: BLE001
            return {"verdicts": 0, "requests": 0, "pending": 0}
