#!/usr/bin/env python3
"""
Offline multi-GPU PPO crypto trader.

    one epoch          = `--episodes-per-epoch` full UTC trading days
    one episode        = one day, one randomly chosen coin, fresh $2,000
    one agent per GPU  = an independent PPO run with its own hyper-parameters,
                         its own checkpoint and its own PNG stream
    one CPU daemon     = int4 Mistral-7B reading that day's prior-48h news and
                         emitting a structured signal the agents observe

Everything reads from the repo.  Outbound sockets are blocked by default.

    python train.py --epochs 2000
    python train.py --epochs 2000 --agents 4          # force 4 even on 1 GPU
    python train.py --no-llm                          # skip the advisor
    python train.py --resume                          # default; just rerun it
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import date, datetime, timezone

# offline guard must be installed before torch / transformers import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.offline import enforce as enforce_offline, set_env_offline  # noqa: E402

set_env_offline()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from src.data.loader import COIN_OF, Dataset  # noqa: E402
from src.envs.trading_env import TradingEnv, VecTradingEnv  # noqa: E402
from src.llm.advisor import AdvisorCache  # noqa: E402
from src.rl.networks import ActorCritic  # noqa: E402
from src.rl.ppo import PPO, PPOConfig, RolloutBuffer  # noqa: E402
from src.utils.checkpoint import CheckpointManager  # noqa: E402
from src.utils.seeding import agent_hparams, seed_everything  # noqa: E402
from src.viz.epoch_report import render_epoch  # noqa: E402


# --------------------------------------------------------------------------- #
def build_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data / run
    p.add_argument("--data-root", default="data")
    p.add_argument("--run-name", default="ppo_btc_eth_ltc")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--symbols", nargs="*",
                   default=["BTCUSDT", "ETHUSDT", "LTCUSDT"])
    p.add_argument("--train-frac", type=float, default=0.85,
                   help="chronological split; the tail is never trained on")
    p.add_argument("--min-date", default=None)
    p.add_argument("--max-date", default=None)

    # training shape
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--episodes-per-epoch", type=int, default=8,
                   help="day-episodes per epoch; 1 = literally one day per epoch")
    p.add_argument("--n-envs", type=int, default=4,
                   help="episodes stepped in lockstep inside one agent")
    p.add_argument("--decision-interval", type=int, default=5,
                   help="minutes between agent decisions (1440/this = steps/day)")

    # account
    p.add_argument("--start-balance", type=float, default=2000.0)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--exposure-levels", type=int, default=5)

    # ppo
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dd-penalty", type=float, default=0.5)
    p.add_argument("--turnover-penalty", type=float, default=0.02)
    p.add_argument("--reward-scale", type=float, default=100.0)

    # parallelism
    p.add_argument("--agents", type=int, default=0,
                   help="0 = one agent per visible GPU (min 1)")
    p.add_argument("--vary-hparams", action="store_true", default=True)
    p.add_argument("--no-vary-hparams", dest="vary_hparams",
                   action="store_false")
    p.add_argument("--torch-threads", type=int, default=2)

    # llm advisor
    p.add_argument("--no-llm", dest="use_llm", action="store_false", default=True)
    p.add_argument("--llm-model-dir", default="models/mistral7b-int4")
    p.add_argument("--llm-batch-size", type=int, default=4)
    p.add_argument("--llm-threads", type=int, default=0,
                   help="0 = max(2, cpu_count - agents - 1)")
    p.add_argument("--llm-max-new-tokens", type=int, default=72)
    p.add_argument("--llm-materialize", action="store_true",
                   help="dequantise weights once: ~14GB RAM (fp16) but much faster")
    p.add_argument("--llm-dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--cache-path", default=None)

    # io
    p.add_argument("--png-every", type=int, default=1)
    p.add_argument("--png-keep-last", type=int, default=400,
                   help="prune older PNGs, but always keep every 50th")
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def episode_stats(records, history, start_balance):
    pnls = np.array([r.pnl for r in records], dtype=float)
    pcts = np.array([r.pnl_pct for r in records], dtype=float)
    bh = np.array([r.buy_hold_end - r.start_balance for r in records], dtype=float)
    llm = np.mean([1.0 if (r.advisor or {}).get("source") == "llm" else 0.0
                   for r in records]) if records else 0.0
    prev_cum = history[-1]["cum_pnl"] if history else 0.0
    all_pcts = [h["mean_pnl_pct"] for h in history[-60:]] + [float(pcts.mean())]
    sharpe = 0.0
    if len(all_pcts) > 3 and np.std(all_pcts) > 1e-9:
        sharpe = float(np.mean(all_pcts) / np.std(all_pcts) * np.sqrt(252))
    return {
        "mean_pnl": float(pnls.mean()), "mean_pnl_pct": float(pcts.mean()),
        "best_pnl": float(pnls.max()), "worst_pnl": float(pnls.min()),
        "cum_pnl": float(prev_cum + pnls.mean()),
        "win_rate": float((pnls > 0).mean()),
        "vs_buy_hold": float((pnls - bh).mean()),
        "mean_trades": float(np.mean([r.n_trades for r in records])),
        "mean_fees": float(np.mean([r.fees_paid for r in records])),
        "mean_max_dd": float(np.mean([r.max_drawdown for r in records])),
        "sharpe": sharpe, "llm_frac": float(llm),
        "n_episodes": len(records),
    }


def prune_pngs(png_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    files = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    if len(files) <= keep_last:
        return
    for f in files[:-keep_last]:
        try:
            n = int(f.split("_")[1].split(".")[0])
        except Exception:  # noqa: BLE001
            continue
        if n % 50 == 0:
            continue           # keep a permanent milestone trail
        try:
            os.remove(os.path.join(png_dir, f))
        except OSError:
            pass


# --------------------------------------------------------------------------- #
def run_agent(agent_id: int, n_agents: int, args, gpu_id: int | None):
    tag = f"agent{agent_id}"
    try:
        if not args.allow_network:
            enforce_offline(True)
        torch.set_num_threads(max(1, args.torch_threads))

        h = agent_hparams(agent_id, {
            "lr": args.lr, "ent_coef": args.ent_coef, "gamma": args.gamma,
            "dd_penalty": args.dd_penalty, "seed": args.seed,
        }, vary=args.vary_hparams)
        seed_everything(h["seed"])

        device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)

        run_dir = os.path.join(args.runs_dir, args.run_name, tag)
        png_dir = os.path.join(run_dir, "png")
        os.makedirs(png_dir, exist_ok=True)
        ckpt = CheckpointManager(run_dir)

        print(f"[{tag}] device={device} lr={h['lr']:.2e} ent={h['ent_coef']} "
              f"gamma={h['gamma']} dd={h['dd_penalty']}", flush=True)

        ds = Dataset(args.data_root, args.symbols, args.min_date, args.max_date,
                     verbose=(agent_id == 0))
        train_days, val_days = ds.split(args.train_frac)
        cache = AdvisorCache(args.cache_path) if args.use_llm else None

        def make_env(i, pool=train_days):
            return TradingEnv(
                ds, cache, start_balance=args.start_balance,
                fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                decision_interval=args.decision_interval,
                allow_short=args.allow_short,
                n_exposure_levels=args.exposure_levels,
                reward_scale=args.reward_scale, dd_penalty=h["dd_penalty"],
                turnover_penalty=args.turnover_penalty,
                use_llm=args.use_llm, day_pool=pool,
                seed=h["seed"] + 101 * i)

        n_envs = max(1, min(args.n_envs, args.episodes_per_epoch))
        venv = VecTradingEnv(n_envs, make_env)
        obs_dim, n_actions = venv.obs_dim, venv.n_actions

        policy = ActorCritic(obs_dim, n_actions, args.hidden,
                             args.n_layers).to(device)
        cfg = PPOConfig(lr=h["lr"], gamma=h["gamma"], gae_lambda=args.gae_lambda,
                        clip_coef=args.clip_coef, vf_coef=args.vf_coef,
                        ent_coef=h["ent_coef"], update_epochs=args.update_epochs,
                        minibatch_size=args.minibatch_size)
        algo = PPO(policy, cfg, device, args.epochs)

        start_epoch = 0
        best_metric = -1e18
        history = ckpt.read_metrics()
        if args.resume:
            state = ckpt.load(map_location=device)
            if state is not None:
                try:
                    policy.load_state_dict(state["policy"])
                    algo.opt.load_state_dict(state["optimizer"])
                    start_epoch = int(state["epoch"]) + 1
                    best_metric = float(state.get("best_metric", -1e18))
                    ckpt.restore_rng(state)
                    for i, e in enumerate(venv.envs):
                        try:
                            e.rng.bit_generator.state = state["rng"]["env"][i]
                        except Exception:  # noqa: BLE001
                            pass
                    history = [r for r in history if r.get("epoch", 0) < start_epoch]
                    print(f"[{tag}] resumed at epoch {start_epoch} "
                          f"(best={best_metric:.2f})", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{tag}] resume failed ({exc}); starting fresh",
                          flush=True)

        steps_per_ep = max(1, 1440 // args.decision_interval)
        rounds = max(1, args.episodes_per_epoch // n_envs)
        buf = RolloutBuffer(rounds * (steps_per_ep + 3), n_envs, obs_dim, device)
        exposure_labels = [f"{x:+.0%}" if args.allow_short else f"{x:.0%}"
                           for x in venv.envs[0].exposures]
        total_steps = sum(h_.get("total_steps", 0) for h_ in history[-1:]) \
            if history else 0

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            buf.reset()
            records = []
            obs = torch.as_tensor(venv.reset(), dtype=torch.float32)

            for _ in range(rounds):
                if buf.ptr > 0:
                    obs = torch.as_tensor(venv.reset(), dtype=torch.float32)
                for _step in range(steps_per_ep + 2):
                    o = obs.to(device)
                    policy.norm.update(o) if hasattr(policy.norm, "update") else None
                    with torch.no_grad():
                        act, logp, _ent, val = policy.act(o)
                    nobs, rew, done, _infos = venv.step(act.cpu().numpy())
                    buf.add(obs, act.cpu(), logp.cpu(), rew, done, val.cpu())
                    obs = torch.as_tensor(nobs, dtype=torch.float32)
                    if done.all():
                        break
                records.extend(venv.records())

            with torch.no_grad():
                _, last_val = policy(obs.to(device))
            lr = algo.set_lr(epoch)
            train_stats = algo.update(buf, last_val, obs_dim)
            train_stats["lr"] = lr

            st = episode_stats(records, history, args.start_balance)
            total_steps += buf.ptr * n_envs
            st["total_steps"] = int(total_steps)
            st["epoch"] = epoch
            st["agent"] = agent_id
            st["wall_s"] = round(time.time() - t0, 2)
            acts = np.concatenate([r.action for r in records if len(r.action)]) \
                if records else np.zeros(1, dtype=int)
            st["action_dist"] = np.bincount(
                acts, minlength=n_actions).astype(float).tolist()
            st["action_dist"] = [v / max(1.0, sum(st["action_dist"]))
                                 for v in st["action_dist"]]
            st.update({k: v for k, v in train_stats.items()
                       if k in ("policy_loss", "value_loss", "entropy",
                                "approx_kl", "clipfrac", "explained_var", "lr")})

            history.append(st)
            ckpt.append_metrics(st)

            is_best = st["cum_pnl"] > best_metric
            best_metric = max(best_metric, st["cum_pnl"])
            if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
                env_rng = [e.rng.bit_generator.state for e in venv.envs]
                ckpt.save(epoch=epoch, policy=policy, optimizer=algo.opt,
                          env_rng=env_rng, cfg=vars(args),
                          best_metric=best_metric, is_best=is_best)

            if args.png_every and epoch % args.png_every == 0:
                featured = max(records, key=lambda r: r.pnl) if records else None
                if featured is not None:
                    try:
                        render_epoch(
                            os.path.join(png_dir, f"epoch_{epoch:06d}.png"),
                            epoch=epoch, agent_id=agent_id,
                            run_name=args.run_name, record=featured,
                            epoch_stats=st, history=history,
                            train_stats=train_stats,
                            hparams={"exposure_labels": exposure_labels},
                            advisor_stats=cache.stats() if cache else None)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
                prune_pngs(png_dir, args.png_keep_last)

            if epoch % 5 == 0 or epoch == start_epoch:
                print(f"[{tag}] ep {epoch:5d}  pnl ${st['mean_pnl']:+8.2f} "
                      f"({st['mean_pnl_pct']:+6.2f}%)  cum ${st['cum_pnl']:+10.2f}  "
                      f"win {100*st['win_rate']:3.0f}%  trades {st['mean_trades']:4.1f}  "
                      f"ent {train_stats['entropy']:.3f}  "
                      f"llm {100*st['llm_frac']:3.0f}%  {st['wall_s']:.1f}s",
                      flush=True)

            if args.eval_every and epoch > 0 and epoch % args.eval_every == 0:
                ev = evaluate(policy, ds, cache, args, h, val_days, device,
                              args.eval_episodes)
                ev.update({"epoch": epoch, "agent": agent_id, "phase": "val"})
                ckpt.append_metrics(ev)
                print(f"[{tag}] ep {epoch:5d}  VAL  pnl ${ev['val_mean_pnl']:+8.2f} "
                      f"({ev['val_mean_pnl_pct']:+6.2f}%)  "
                      f"win {100*ev['val_win_rate']:3.0f}%", flush=True)

        print(f"[{tag}] finished {args.epochs} epochs", flush=True)

    except KeyboardInterrupt:
        print(f"[{tag}] interrupted; latest checkpoint is on disk", flush=True)
    except Exception:  # noqa: BLE001
        print(f"[{tag}] CRASHED", flush=True)
        traceback.print_exc()
        raise


@torch.no_grad()
def evaluate(policy, ds, cache, args, h, day_pool, device, n_episodes: int):
    """Greedy rollout on the held-out chronological tail."""
    env = TradingEnv(ds, cache, start_balance=args.start_balance,
                     fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                     decision_interval=args.decision_interval,
                     allow_short=args.allow_short,
                     n_exposure_levels=args.exposure_levels,
                     reward_scale=args.reward_scale, dd_penalty=h["dd_penalty"],
                     turnover_penalty=args.turnover_penalty,
                     use_llm=args.use_llm, day_pool=day_pool,
                     seed=h["seed"] + 77_777)
    policy.eval()
    pnls, pcts, bh = [], [], []
    for _ in range(n_episodes):
        o = env.reset()
        done = False
        while not done:
            t = torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            a, _, _, _ = policy.act(t, deterministic=True)
            o, _r, done, _i = env.step(int(a.item()))
        r = env.record
        pnls.append(r.pnl)
        pcts.append(r.pnl_pct)
        bh.append(r.buy_hold_end - r.start_balance)
    policy.train()
    pnls = np.array(pnls)
    return {
        "val_mean_pnl": float(pnls.mean()),
        "val_mean_pnl_pct": float(np.mean(pcts)),
        "val_win_rate": float((pnls > 0).mean()),
        "val_vs_buy_hold": float(np.mean(pnls - np.array(bh))),
        "val_episodes": n_episodes,
    }


# --------------------------------------------------------------------------- #
def main():
    args = build_args()
    if not args.allow_network:
        enforce_offline(True)

    os.makedirs(os.path.join(args.runs_dir, args.run_name), exist_ok=True)
    if args.cache_path is None:
        args.cache_path = os.path.join(args.runs_dir, args.run_name,
                                       "advisor_cache.sqlite")

    n_gpu = torch.cuda.device_count()
    n_agents = args.agents if args.agents > 0 else max(1, n_gpu)
    gpu_ids = [i % n_gpu for i in range(n_agents)] if n_gpu else [None] * n_agents

    print("=" * 78)
    print(f"  run            : {args.run_name}")
    print(f"  torch          : {torch.__version__}  cuda={torch.cuda.is_available()}")
    print(f"  GPUs visible   : {n_gpu}"
          + (f"  ({', '.join(torch.cuda.get_device_name(i) for i in range(n_gpu))})"
             if n_gpu else "  -> running on CPU"))
    print(f"  agents         : {n_agents}  -> gpus {gpu_ids}")
    print(f"  epochs         : {args.epochs}   episodes/epoch: {args.episodes_per_epoch}")
    print(f"  balance        : ${args.start_balance:,.0f} per episode")
    print(f"  decision every : {args.decision_interval} min "
          f"({1440//args.decision_interval} steps/day)")
    print(f"  advisor        : {'int4 Mistral-7B on CPU' if args.use_llm else 'disabled'}")
    print(f"  network        : {'ALLOWED' if args.allow_network else 'BLOCKED (offline)'}")
    print("=" * 78, flush=True)

    with open(os.path.join(args.runs_dir, args.run_name, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)

    procs = []
    stop_flag = os.path.join(args.runs_dir, args.run_name, "STOP")
    if os.path.exists(stop_flag):
        os.remove(stop_flag)

    if args.use_llm and os.path.isdir(args.llm_model_dir):
        from src.llm.daemon import daemon_entry
        cpu = os.cpu_count() or 4
        threads = args.llm_threads or max(2, cpu - n_agents - 1)
        kw = dict(dataset_root=args.data_root, model_dir=args.llm_model_dir,
                  cache_path=args.cache_path, batch_size=args.llm_batch_size,
                  threads=threads, max_new_tokens=args.llm_max_new_tokens,
                  materialize=args.llm_materialize, dtype=args.llm_dtype,
                  stop_flag=stop_flag)
        p = mp.Process(target=daemon_entry, args=(kw,), name="advisor",
                       daemon=False)
        p.start()
        procs.append(p)
        print(f"[main] advisor daemon pid={p.pid} threads={threads}", flush=True)
    elif args.use_llm:
        print(f"[main] !! {args.llm_model_dir} not found -- agents will use the "
              f"numeric prior only", flush=True)

    workers = []
    for i in range(n_agents):
        p = mp.Process(target=run_agent, args=(i, n_agents, args, gpu_ids[i]),
                       name=f"agent{i}")
        p.start()
        workers.append(p)
        print(f"[main] agent {i} pid={p.pid} gpu={gpu_ids[i]}", flush=True)

    def _sigterm(signum, frame):
        print("\n[main] shutting down; checkpoints are already on disk", flush=True)
        open(stop_flag, "w").write("stop")
        for w in workers:
            w.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)

    for w in workers:
        w.join()
    open(stop_flag, "w").write("stop")
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    print("[main] all agents done", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
