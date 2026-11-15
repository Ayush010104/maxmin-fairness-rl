"""Finite-capacity FIFO replay buffer with optional prioritised sampling."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

_PRIORITY_EPS = 1e-6


class ReplayBuffer:
    """Fixed-size circular buffer.

    Keeping the capacity well below the training horizon makes the buffer discard
    stale transitions from early, low-objective regimes automatically.

    Sampling is uniform by default. Passing ``exponent`` to :meth:`sample` draws
    transitions with probability proportional to ``priority ** exponent`` instead;
    priorities are refreshed with :meth:`update_priorities`. A new transition inherits
    the priority of the slot it overwrites, and no importance-sampling correction is
    applied.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.count = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.count % self.capacity
        self.states[i], self.actions[i] = state, action
        self.rewards[i], self.next_states[i], self.dones[i] = reward, next_state, float(done)
        self.count += 1

    def reset_priorities(self) -> None:
        self.priorities[:] = 1.0

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Set priority to ``log(1 + |td|)`` for the sampled slots."""
        self.priorities[indices] = np.log1p(np.abs(td_errors)) + _PRIORITY_EPS

    def sample(
        self, batch_size: int, rng: np.random.Generator, exponent: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Return ``(states, actions, rewards, next_states, dones, indices)``."""
        n = len(self)
        if exponent is None:
            idx = rng.integers(0, n, size=batch_size)
        else:
            probs = self.priorities[:n].astype(np.float64) ** exponent
            idx = rng.choice(n, size=batch_size, p=probs / probs.sum())
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return (
            to(self.states),
            to(self.actions),
            to(self.rewards),
            to(self.next_states),
            to(self.dones),
            idx,
        )
