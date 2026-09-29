"""
PPO-clip with GAE(lambda).  Written out rather than pulled from
stable-baselines3 because the training host has no network access and SB3 is
not in its image.

One epoch == `episodes_per_epoch` full trading days rolled out across
`n_envs` parallel environments, followed by `update_epochs` passes of
minibatch SGD over the collected buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 512
    target_kl: float = 0.03
    anneal_lr: bool = True
    normalise_adv: bool = True


class RolloutBuffer:
    def __init__(self, capacity: int, n_envs: int, obs_dim: int, device):
        self.obs = torch.zeros((capacity, n_envs, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, n_envs), dtype=torch.long)
        self.logprobs = torch.zeros((capacity, n_envs), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, n_envs), dtype=torch.float32)
        self.dones = torch.zeros((capacity, n_envs), dtype=torch.float32)
        self.values = torch.zeros((capacity, n_envs), dtype=torch.float32)
        self.capacity = capacity
        self.n_envs = n_envs
        self.device = device
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logprobs[i] = logprob
        self.rewards[i] = torch.as_tensor(reward)
        self.dones[i] = torch.as_tensor(done, dtype=torch.float32)
        self.values[i] = value
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        n = self.ptr
        adv = torch.zeros((n, self.n_envs), dtype=torch.float32)
        last_gae = torch.zeros(self.n_envs, dtype=torch.float32)
        for t in reversed(range(n)):
            next_nonterm = 1.0 - self.dones[t]
            next_val = last_value.cpu() if t == n - 1 else self.values[t + 1]
            delta = (self.rewards[t] + gamma * next_val * next_nonterm
                     - self.values[t])
            last_gae = delta + gamma * lam * next_nonterm * last_gae
            adv[t] = last_gae
        ret = adv + self.values[:n]
        return adv, ret

    def flat(self, adv, ret, obs_dim):
        n = self.ptr
        return (self.obs[:n].reshape(-1, obs_dim).to(self.device),
                self.actions[:n].reshape(-1).to(self.device),
                self.logprobs[:n].reshape(-1).to(self.device),
                adv.reshape(-1).to(self.device),
                ret.reshape(-1).to(self.device),
                self.values[:n].reshape(-1).to(self.device))


class PPO:
    def __init__(self, policy: nn.Module, cfg: PPOConfig, device, total_epochs: int):
        self.policy = policy
        self.cfg = cfg
        self.device = device
        self.total_epochs = max(1, total_epochs)
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    def set_lr(self, epoch: int) -> float:
        lr = self.cfg.lr
        if self.cfg.anneal_lr:
            lr = self.cfg.lr * max(0.05, 1.0 - epoch / self.total_epochs)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def update(self, buf: RolloutBuffer, last_value: torch.Tensor,
               obs_dim: int) -> dict:
        cfg = self.cfg
        adv, ret = buf.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)
        b_obs, b_act, b_logp, b_adv, b_ret, b_val = buf.flat(adv, ret, obs_dim)

        n = b_obs.shape[0]
        idx = np.arange(n)
        clipfracs, kls = [], []
        pl = vl = el = 0.0
        n_batches = 0
        stop = False

        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                if len(mb) < 8:
                    continue
                mb_t = torch.as_tensor(mb, device=self.device)
                newlogp, entropy, newval = self.policy.evaluate(
                    b_obs[mb_t], b_act[mb_t])
                logratio = newlogp - b_logp[mb_t]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    kls.append(approx_kl)
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b_adv[mb_t]
                if cfg.normalise_adv and len(mb) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef,
                                            1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unclipped = (newval - b_ret[mb_t]) ** 2
                v_clipped = b_val[mb_t] + torch.clamp(
                    newval - b_val[mb_t], -cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max(
                    v_unclipped, (v_clipped - b_ret[mb_t]) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         cfg.max_grad_norm)
                self.opt.step()

                pl += pg_loss.item()
                vl += v_loss.item()
                el += ent.item()
                n_batches += 1

            if cfg.target_kl and kls and np.mean(kls[-8:]) > cfg.target_kl:
                stop = True
                break

        k = max(n_batches, 1)
        y_pred, y_true = b_val.cpu().numpy(), b_ret.cpu().numpy()
        var_y = np.var(y_true)
        ev = float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        return {
            "policy_loss": pl / k, "value_loss": vl / k, "entropy": el / k,
            "approx_kl": float(np.mean(kls)) if kls else 0.0,
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "explained_var": float(ev), "early_stop": bool(stop),
            "n_batches": n_batches,
        }
