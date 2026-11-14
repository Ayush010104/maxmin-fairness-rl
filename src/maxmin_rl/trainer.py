"""High-level training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .agent import AgentConfig, HybridSACDDPGAgent
from .env import FairnessEnv
from .problem import FairnessProblem


@dataclass
class TrainResult:
    """Outcome of one training run.

    ``best_*`` is the best solution seen at any step (by minimum objective), which is
    what the algorithm returns; ``min_history`` is the per-step minimum objective.
    """

    min_history: np.ndarray
    composite_history: np.ndarray
    initial_min: float
    best_min: float
    best_step: int
    best_x: np.ndarray
    best_p: np.ndarray
    best_objectives: np.ndarray


def train(
    problem: FairnessProblem,
    steps: int = 10_000,
    seed: int = 0,
    config: Optional[AgentConfig] = None,
    verbose: bool = False,
    log_every: int = 1000,
    **env_kwargs,
) -> TrainResult:
    """Train a fresh agent on ``problem`` for ``steps`` environment steps.

    Extra keyword arguments (``gm_weight``, ``min_weight``, ``reward_scale``,
    ``start_candidates``) are forwarded to :class:`FairnessEnv`.
    """
    torch.manual_seed(seed)
    env = FairnessEnv(problem, max_steps=steps, **env_kwargs)
    state = env.reset(seed)
    agent = HybridSACDDPGAgent(
        env.state_dim, problem.n_p, problem.n_x, problem.project_x, config, seed
    )
    initial_min = float(env.objectives.min())
    min_history, composite_history = [], []

    for t in range(steps):
        action = agent.act(state)
        res = env.step(action)
        agent.remember(state, action, res.reward, res.state, res.done)
        agent.learn(res.target)
        state = res.state
        min_history.append(float(res.objectives.min()))
        composite_history.append(res.composite)
        if verbose and (t + 1) % log_every == 0:
            print(f"step {t + 1:>6}/{steps}  min={min_history[-1]:.4f}  best={env.best_min:.4f}")

    return TrainResult(
        min_history=np.asarray(min_history),
        composite_history=np.asarray(composite_history),
        initial_min=initial_min,
        best_min=env.best_min,
        best_step=env.best_step,
        best_x=env.best_x,
        best_p=env.best_p,
        best_objectives=env.best_objectives,
    )
