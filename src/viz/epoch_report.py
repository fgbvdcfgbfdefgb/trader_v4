"""
One PNG per epoch per agent: `runs/<run>/agent<k>/png/epoch_000123.png`.

Twelve panels covering the featured episode, the account, the advisor and the
optimiser, so a single image answers "did this epoch make money, how, and was
the policy still learning?".
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

BG = "#11141a"
FG = "#e8eaf0"
GRID = "#2a2f3a"
GREEN = "#26d07c"
RED = "#ff5c5c"
BLUE = "#4da3ff"
AMBER = "#ffb84d"
PURPLE = "#b07cff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": "#9aa3b2", "ytick.color": "#9aa3b2",
    "grid.color": GRID, "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.titleweight": "bold", "figure.autolayout": False,
    # Dollar signs are everywhere in this report; without this matplotlib
    # treats "$1,234" as the start of a LaTeX math block and raises.
    "text.parse_math": False,
})


def _style(ax, title: str = "", ylabel: str = ""):
    ax.grid(True, alpha=0.25, lw=0.6)
    for s in ax.spines.values():
        s.set_alpha(0.35)
    if title:
        ax.set_title(title, loc="left", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def _money(v: float) -> str:
    s = "-" if v < 0 else ""
    return f"{s}${abs(v):,.2f}"


def render_epoch(path: str, *, epoch: int, agent_id: int, run_name: str,
                 record, epoch_stats: dict, history: list[dict],
                 train_stats: dict, hparams: dict, advisor_stats: dict | None = None):
    """
    record      : EpisodeRecord of the featured (best-PnL) episode this epoch
    epoch_stats : aggregates over every episode in the epoch
    history     : list of previous epoch_stats dicts (for the trend panels)
    train_stats : PPO diagnostics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(19, 12.2), dpi=104)
    gs = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.20,
                  left=0.045, right=0.985, top=0.915, bottom=0.05)

    pnl = epoch_stats.get("mean_pnl", 0.0)
    head_col = GREEN if pnl >= 0 else RED
    fig.suptitle(
        f"{run_name}   ·   agent {agent_id}   ·   epoch {epoch}",
        x=0.045, y=0.972, ha="left", fontsize=15, fontweight="bold", color=FG)
    fig.text(0.045, 0.941,
             f"epoch mean PnL {_money(pnl)}  ({epoch_stats.get('mean_pnl_pct',0):+.2f}%)"
             f"   ·   cumulative {_money(epoch_stats.get('cum_pnl',0.0))}"
             f"   ·   win rate {100*epoch_stats.get('win_rate',0):.0f}%"
             f"   ·   {epoch_stats.get('n_episodes',0)} day-episodes"
             f"   ·   {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             ha="left", fontsize=10, color=head_col)

    t = np.arange(len(record.price)) if len(record.price) else np.arange(1)

    # ---------------- 1. price + trades ---------------- #
    ax = fig.add_subplot(gs[0, :2])
    if len(record.price):
        ax.plot(t, record.price, color=BLUE, lw=1.1, zorder=3)
        pos = record.position
        if len(pos) == len(t):
            ax.fill_between(t, record.price.min(), record.price,
                            where=pos > 0.02, color=GREEN, alpha=0.08, zorder=1)
            ax.fill_between(t, record.price.min(), record.price,
                            where=pos < -0.02, color=RED, alpha=0.08, zorder=1)
        for i, side, p in zip(record.trade_idx, record.trade_side,
                              record.trade_price):
            i = int(np.clip(i, 0, len(t) - 1))
            ax.scatter(i, p, marker="^" if side > 0 else "v", s=42,
                       color=GREEN if side > 0 else RED,
                       edgecolors="#0b0d11", linewidths=0.5, zorder=5)
    _style(ax, f"featured episode · {record.coin} · "
               f"{record.day}  ({record.n_trades} trades)", "price (USDT)")
    ax.set_xlabel("minute of day")

    # ---------------- 2. advisor panel ---------------- #
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    a = record.advisor or {}
    src = a.get("source", "prior")
    badge = "MISTRAL-7B int4" if src == "llm" else "numeric prior"
    badge_c = PURPLE if src == "llm" else "#7a8396"
    ax.text(0, 1.0, "NEWS ADVISOR", fontsize=10, fontweight="bold",
            color=FG, va="top")
    ax.text(0.62, 1.0, badge, fontsize=8.5, fontweight="bold", color=badge_c,
            va="top")
    bias = a.get("bias", 0)
    bias_c = GREEN if bias > 0 else (RED if bias < 0 else "#9aa3b2")
    rows = [
        ("bias", f"{bias:+d}", bias_c),
        ("conviction", f"{a.get('conviction',0)}/3", FG),
        ("volatility", ["low", "normal", "high"][int(a.get("volatility", 1))], FG),
        ("event risk", ["none", "some", "severe"][int(a.get("event_risk", 0))], FG),
        ("tag", str(a.get("tag", "-")), AMBER),
        ("headlines", str(a.get("n_headlines", 0)), FG),
    ]
    y = 0.88
    for k, v, c in rows:
        ax.text(0.02, y, k, fontsize=8.5, color="#9aa3b2", va="top")
        ax.text(0.52, y, v, fontsize=8.5, color=c, va="top", fontweight="bold")
        y -= 0.072
    why = str(a.get("rationale", ""))[:70]
    if why:
        ax.text(0.02, y - 0.01, f'"{why}"', fontsize=8, color="#9aa3b2",
                va="top", style="italic", wrap=True)
        y -= 0.07
    y -= 0.03
    ax.text(0.02, y, "headlines used (strictly pre-day):", fontsize=8,
            color="#7a8396", va="top")
    y -= 0.055
    for h in (record.headlines or [])[:5]:
        ax.text(0.02, y, "· " + h[:58], fontsize=7.3, color="#c3cad6", va="top")
        y -= 0.05
    if not record.headlines:
        ax.text(0.02, y, "· (none available for this date)", fontsize=7.3,
                color="#6b7383", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    # ---------------- 3. equity vs buy & hold ---------------- #
    ax = fig.add_subplot(gs[1, 0])
    if len(record.equity):
        ax.plot(record.equity, color=GREEN if record.pnl >= 0 else RED, lw=1.5,
                label="agent", zorder=3)
        if len(record.price) == len(record.equity) and record.price[0] > 0:
            bh = record.start_balance * record.price / record.price[0]
            ax.plot(bh, color="#7a8396", lw=1.0, ls="--", label="buy & hold")
        ax.axhline(record.start_balance, color="#4a5160", lw=0.8, ls=":")
        ax.legend(frameon=False, fontsize=7.5, loc="best")
    _style(ax, f"equity · {_money(record.pnl)} ({record.pnl_pct:+.2f}%)", "USD")

    # ---------------- 4. cumulative PnL ---------------- #
    ax = fig.add_subplot(gs[1, 1])
    if history:
        cum = np.array([h.get("cum_pnl", 0.0) for h in history], dtype=float)
        ep = np.array([h.get("epoch", i) for i, h in enumerate(history)])
        ax.plot(ep, cum, color=AMBER, lw=1.4)
        ax.fill_between(ep, 0, cum, where=cum >= 0, color=GREEN, alpha=0.15)
        ax.fill_between(ep, 0, cum, where=cum < 0, color=RED, alpha=0.15)
        ax.axhline(0, color="#4a5160", lw=0.8)
    _style(ax, "cumulative PnL across epochs", "USD")
    ax.set_xlabel("epoch")

    # ---------------- 5. per-epoch PnL ---------------- #
    ax = fig.add_subplot(gs[1, 2])
    if history:
        tail = history[-80:]
        vals = np.array([h.get("mean_pnl", 0.0) for h in tail], dtype=float)
        eps = np.array([h.get("epoch", i) for i, h in enumerate(tail)])
        ax.bar(eps, vals, color=[GREEN if v >= 0 else RED for v in vals],
               width=max(1, len(tail) / 80), alpha=0.85)
        ax.axhline(0, color="#4a5160", lw=0.8)
        if len(vals) >= 10:
            k = min(20, len(vals))
            ma = np.convolve(vals, np.ones(k) / k, mode="valid")
            ax.plot(eps[k - 1:], ma, color=BLUE, lw=1.3, label=f"MA{k}")
            ax.legend(frameon=False, fontsize=7.5)
    _style(ax, "per-epoch mean PnL (last 80)", "USD")
    ax.set_xlabel("epoch")

    # ---------------- 6. action distribution ---------------- #
    ax = fig.add_subplot(gs[2, 0])
    dist = epoch_stats.get("action_dist") or []
    if len(dist):
        labels = hparams.get("exposure_labels") or [str(i) for i in range(len(dist))]
        ax.bar(range(len(dist)), dist, color=BLUE, alpha=0.85)
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
    _style(ax, "action distribution (target exposure)", "frequency")

    # ---------------- 7. reward per step ---------------- #
    ax = fig.add_subplot(gs[2, 1])
    if len(record.reward):
        ax.plot(record.reward, color=PURPLE, lw=0.8, alpha=0.9)
        ax.axhline(0, color="#4a5160", lw=0.8)
        if len(record.reward) > 20:
            k = 20
            ma = np.convolve(record.reward, np.ones(k) / k, mode="valid")
            ax.plot(np.arange(k - 1, len(record.reward)), ma, color=AMBER, lw=1.2)
    _style(ax, "per-step reward (featured episode)", "reward")
    ax.set_xlabel("decision step")

    # ---------------- 8. drawdown ---------------- #
    ax = fig.add_subplot(gs[2, 2])
    if len(record.equity):
        eq = record.equity
        dd = 100.0 * (1.0 - eq / np.maximum.accumulate(eq))
        ax.fill_between(np.arange(len(dd)), 0, -dd, color=RED, alpha=0.35)
        ax.plot(-dd, color=RED, lw=1.0)
    _style(ax, f"drawdown · max {100*record.max_drawdown:.2f}%", "%")

    # ---------------- 9. PPO losses ---------------- #
    ax = fig.add_subplot(gs[3, 0])
    if history:
        ep = [h.get("epoch", i) for i, h in enumerate(history)]
        for key, col, lab in (("policy_loss", BLUE, "policy"),
                              ("value_loss", AMBER, "value"),
                              ("entropy", GREEN, "entropy")):
            ys = [h.get(key, np.nan) for h in history]
            if np.isfinite(np.asarray(ys, dtype=float)).any():
                ax.plot(ep, ys, lw=1.1, color=col, label=lab)
        ax.legend(frameon=False, fontsize=7.5, ncol=3)
    _style(ax, "PPO losses", "")
    ax.set_xlabel("epoch")

    # ---------------- 10. win rate / sharpe ---------------- #
    ax = fig.add_subplot(gs[3, 1])
    if history:
        ep = [h.get("epoch", i) for i, h in enumerate(history)]
        wr = [100 * h.get("win_rate", np.nan) for h in history]
        ax.plot(ep, wr, color=GREEN, lw=1.2, label="win rate %")
        ax.axhline(50, color="#4a5160", lw=0.8, ls=":")
        ax2 = ax.twinx()
        sh = [h.get("sharpe", np.nan) for h in history]
        ax2.plot(ep, sh, color=PURPLE, lw=1.1, label="sharpe")
        ax2.tick_params(colors="#9aa3b2")
        ax2.set_ylabel("sharpe", color=PURPLE)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    _style(ax, "win rate & rolling sharpe", "%")
    ax.set_xlabel("epoch")

    # ---------------- 11+12. stats table ---------------- #
    ax = fig.add_subplot(gs[3, 2])
    ax.axis("off")
    adv = advisor_stats or {}
    left = [
        ("start balance", _money(record.start_balance)),
        ("end balance", _money(record.end_balance)),
        ("episode PnL", f"{_money(record.pnl)} ({record.pnl_pct:+.2f}%)"),
        ("buy & hold", _money(record.buy_hold_end - record.start_balance)),
        ("fees paid", _money(record.fees_paid)),
        ("trades", str(record.n_trades)),
        ("max drawdown", f"{100*record.max_drawdown:.2f}%"),
        ("epoch mean PnL", _money(epoch_stats.get("mean_pnl", 0.0))),
        ("epoch best / worst", f"{_money(epoch_stats.get('best_pnl',0))} / "
                               f"{_money(epoch_stats.get('worst_pnl',0))}"),
    ]
    right = [
        ("lr", f"{train_stats.get('lr', 0):.2e}"),
        ("approx KL", f"{train_stats.get('approx_kl', 0):.4f}"),
        ("clip frac", f"{train_stats.get('clipfrac', 0):.3f}"),
        ("explained var", f"{train_stats.get('explained_var', float('nan')):.3f}"),
        ("entropy", f"{train_stats.get('entropy', 0):.3f}"),
        ("total steps", f"{epoch_stats.get('total_steps', 0):,}"),
        ("advisor cached", f"{adv.get('verdicts', 0):,}"),
        ("advisor pending", f"{adv.get('pending', 0):,}"),
        ("llm-backed eps", f"{100*epoch_stats.get('llm_frac', 0):.0f}%"),
    ]
    ax.text(0.0, 1.0, "EPISODE & ACCOUNT", fontsize=9, fontweight="bold",
            color=FG, va="top")
    ax.text(0.54, 1.0, "OPTIMISER & ADVISOR", fontsize=9, fontweight="bold",
            color=FG, va="top")
    for col, items in ((0.0, left), (0.54, right)):
        y = 0.90
        for k, v in items:
            ax.text(col, y, k, fontsize=8, color="#9aa3b2", va="top")
            ax.text(col + 0.29, y, v, fontsize=8, color=FG, va="top",
                    fontweight="bold")
            y -= 0.095
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)

    tmp = path + ".tmp.png"
    fig.savefig(tmp, facecolor=BG)
    plt.close(fig)
    os.replace(tmp, path)
    return path
