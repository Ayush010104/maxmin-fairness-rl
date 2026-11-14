"""Finite-capacity FIFO replay buffer."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer.

    Keeping the capacity well below the training horizon makes the buffer discard
    stale transitions from early, low-objective regimes automatically.
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

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.count % self.capacity
        self.states[i], self.actions[i] = state, action
        self.rewards[i], self.next_states[i], self.dones[i] = reward, next_state, float(done)
        self.count += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[torch.Tensor, ...]:
        idx = rng.integers(0, len(self), size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return (
            to(self.states),
            to(self.actions),
            to(self.rewards),
            to(self.next_states),
            to(self.dones),
        )
