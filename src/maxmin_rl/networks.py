"""Hybrid actor (stochastic SAC head + deterministic DDPG head) and twin-critic building block."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .projections import Projection


class HybridActor(nn.Module):
    """One trunk, two heads.

    * Incremental block ``x`` -> SAC head: ``tanh``-squashed Gaussian step, projected
      back onto the feasible set *inside the graph*. The returned action is the
      projected delta ``project(x + step) - x``, never the raw proposal.
    * Simplex block ``p`` -> DDPG head: deterministic ``softmax(sigmoid(.))`` shares.

    The returned log-probability is that of the tanh-squashed Gaussian sample (the
    change-of-variables correction for ``tanh`` is included); it is *not* corrected
    for the projection, which is treated as constraint enforcement.
    """

    reparam_noise = 1e-6

    def __init__(
        self,
        state_dim: int,
        n_p: int,
        n_x: int,
        project_x: Projection,
        hidden: Sequence[int] = (200, 200),
        max_step: float = 0.45,
        sigma_scale: float = 0.2,
    ) -> None:
        super().__init__()
        h1, h2 = hidden
        self.n_p, self.n_x = n_p, n_x
        self.project_x = project_x
        self.max_step = max_step
        self.sigma_scale = sigma_scale

        self.fc1, self.bn1 = nn.Linear(state_dim, h1), nn.BatchNorm1d(h1)
        self.fc2, self.bn2 = nn.Linear(h1, h2), nn.BatchNorm1d(h2)
        self.p_head = nn.Linear(h2, n_p) if n_p > 0 else None
        self.mu_head = nn.Linear(h2, n_x)
        self.sigma_head = nn.Linear(h2, n_x)

    def forward(self, state: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        h = F.leaky_relu(self.bn1(self.fc1(state)), 0.01)
        h = F.leaky_relu(self.bn2(self.fc2(h)), 0.01)
        shares = None
        if self.p_head is not None:
            shares = torch.softmax(torch.sigmoid(self.p_head(h)), dim=-1)
        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) * self.sigma_scale + 1e-8
        return shares, mu, sigma

    def sample(
        self, state: torch.Tensor, reparameterize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob)`` with ``action = concat([p, projected_delta])``."""
        shares, mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        raw = dist.rsample() if reparameterize else dist.sample()
        squashed = torch.tanh(raw)

        x_now = state[:, self.n_p :]
        x_next = self.project_x(x_now + self.max_step * squashed)
        delta = x_next - x_now

        log_prob = dist.log_prob(raw) - torch.log(1 - squashed.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = delta if shares is None else torch.cat([shares, delta], dim=1)
        return action, log_prob


class Critic(nn.Module):
    """Q(s, a): a state branch and an action branch, merged by two more layers."""

    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden
        self.s1, self.sbn1 = nn.Linear(state_dim, h1), nn.BatchNorm1d(h1)
        self.s2, self.sbn2 = nn.Linear(h1, h2), nn.BatchNorm1d(h2)
        self.a1 = nn.Linear(action_dim, h1)
        self.m1 = nn.Linear(h1 + h2, h2)
        self.m2 = nn.Linear(h2, h2)
        self.q = nn.Linear(h2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = self.sbn1(F.leaky_relu(self.s1(state), 0.01))
        s = self.sbn2(F.leaky_relu(self.s2(s), 0.01))
        a = F.leaky_relu(self.a1(action), 0.01)
        h = F.leaky_relu(self.m1(torch.cat([s, a], dim=1)), 0.01)
        h = F.leaky_relu(self.m2(h), 0.01)
        return self.q(h)
