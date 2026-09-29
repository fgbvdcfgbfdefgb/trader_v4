"""Actor-critic network. Small on purpose -- the bottleneck is data, not depth."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def ortho(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class RunningNorm(nn.Module):
    """
    Welford observation normaliser kept *inside* the module so it is written
    to the checkpoint.  Resuming a run with a stale normaliser silently
    destroys a policy, so this must never live outside the state_dict.
    """

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-5):
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm = x.mean(0)
        bv = x.var(0, unbiased=False)
        bc = torch.tensor(float(x.shape[0]), device=x.device)
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var.copy_((m_a + m_b + delta.pow(2) * self.count * bc / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps),
                           -self.clip, self.clip)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256,
                 n_layers: int = 2, normalise_obs: bool = True):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.norm = RunningNorm(obs_dim) if normalise_obs else nn.Identity()

        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(n_layers):
            layers += [ortho(nn.Linear(d, hidden)), nn.Tanh()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.pi = ortho(nn.Linear(hidden, n_actions), gain=0.01)
        self.v = ortho(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.body(self.norm(obs))
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
