"""
Crash-safe, resumable checkpointing.

Rules this enforces:
  * every write is atomic (tmp file + os.replace) so a kill -9 mid-save can
    never leave a truncated checkpoint;
  * two rotating slots (latest / previous) so even a corrupt "latest" is
    recoverable;
  * RNG state for python, numpy and torch is part of the checkpoint, so a
    resumed run continues the *same* stochastic trajectory rather than
    silently re-sampling the same days;
  * metrics are appended to a JSONL sidecar, which survives regardless of
    whether the .pt write completed.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time

import numpy as np
import torch


def atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.latest = os.path.join(run_dir, "ckpt_latest.pt")
        self.prev = os.path.join(run_dir, "ckpt_prev.pt")
        self.best = os.path.join(run_dir, "ckpt_best.pt")
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # ------------------------------------------------------------------ #
    def save(self, *, epoch: int, policy, optimizer, env_rng, cfg: dict,
             best_metric: float, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "cfg": cfg,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "env": env_rng,
            },
            "saved_at": time.time(),
        }
        if os.path.exists(self.latest):
            try:
                shutil.copy2(self.latest, self.prev)
            except Exception:  # noqa: BLE001
                pass
        atomic_save(state, self.latest)
        if is_best:
            atomic_save(state, self.best)

    # ------------------------------------------------------------------ #
    def load(self, map_location="cpu"):
        for path in (self.latest, self.prev):
            if not os.path.exists(path):
                continue
            try:
                return torch.load(path, map_location=map_location,
                                  weights_only=False)
            except Exception as exc:  # noqa: BLE001
                print(f"[ckpt] {os.path.basename(path)} unreadable ({exc}), "
                      f"trying fallback", flush=True)
        return None

    def restore_rng(self, state: dict) -> None:
        r = state.get("rng", {})
        try:
            random.setstate(r["python"])
            np.random.set_state(r["numpy"])
            t = r["torch"]
            torch.set_rng_state(t if t.dtype == torch.uint8 else t.to(torch.uint8))
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] could not restore RNG ({exc}); continuing", flush=True)

    # ------------------------------------------------------------------ #
    def append_metrics(self, row: dict) -> None:
        with open(self.metrics_path, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()

    def read_metrics(self) -> list[dict]:
        if not os.path.exists(self.metrics_path):
            return []
        out = []
        with open(self.metrics_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
        return out
