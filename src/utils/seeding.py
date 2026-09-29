"""Deterministic-ish seeding and per-agent hyper-parameter diversification."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def agent_hparams(agent_id: int, base: dict, vary: bool = True) -> dict:
    """
    Give each parallel agent a genuinely different search point.

    Running N identical agents on N GPUs buys nothing but variance reduction;
    spreading learning rate, entropy bonus, discount and risk appetite turns
    the extra GPUs into a small population search whose best member is
    usually much better than the mean.
    """
    h = dict(base)
    h["seed"] = base.get("seed", 0) + 9973 * agent_id
    if not vary or agent_id == 0:
        return h
    lr_mult = [1.0, 0.5, 2.0, 0.33, 3.0, 0.7, 1.5, 0.25][agent_id % 8]
    ent = [0.01, 0.02, 0.005, 0.03, 0.0025, 0.015, 0.04, 0.0075][agent_id % 8]
    gam = [0.997, 0.999, 0.99, 0.995, 0.9985, 0.9975, 0.98, 0.9995][agent_id % 8]
    ddp = [0.5, 0.2, 1.0, 0.35, 1.5, 0.75, 0.1, 2.0][agent_id % 8]
    h["lr"] = base["lr"] * lr_mult
    h["ent_coef"] = ent
    h["gamma"] = gam
    h["dd_penalty"] = ddp
    return h
