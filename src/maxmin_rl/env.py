"""Generic MDP wrapper that turns a :class:`FairnessProblem` into an RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .problem import FairnessProblem
from .rewards import composite_metric, log_ratio_reward


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    composite: float
    objectives: np.ndarray
    target: Optional[np.ndarray]
    done: bool


class FairnessEnv:
    """State = current variables, action = incremental update, reward = log-ratio.

    The action is ``concat([p_new, dx])``: the new simplex block ``p`` is taken as
    given, and ``x`` is updated as ``x <- x + dx``. The agent stores the *projected
    delta* ``dx = project(x + step) - x``, so the stored action always explains the
    transition it caused and ``x`` stays feasible by construction.

    The best-so-far solution (by minimum objective) is tracked in ``best_*``.
    """

    def __init__(
        self,
        problem: FairnessProblem,
        max_steps: int = 10_000,
        gm_weight: float = 0.4,
        min_weight: float = 0.6,
        reward_scale: float = 1000.0,
        start_candidates: int = 1,
    ) -> None:
        if problem.n_x <= 0:
            raise ValueError("problem.n_x must be positive")
        self.problem = problem
        self.max_steps = max_steps
        self.gm_weight = gm_weight
        self.min_weight = min_weight
        self.reward_scale = reward_scale
        self.start_candidates = max(1, start_candidates)
        self.state_dim = problem.n_p + problem.n_x
        self.action_dim = problem.n_p + problem.n_x
        self.rng = np.random.default_rng()
        self.t = 0

    def _evaluate(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        o = np.asarray(self.problem.objectives(x, p), dtype=np.float64)
        if o.shape != (self.problem.n_objectives,):
            raise ValueError(
                f"objectives() returned shape {o.shape}, expected ({self.problem.n_objectives},)"
            )
        if not np.all(np.isfinite(o)) or np.any(o <= 0):
            raise ValueError("objectives() must be finite and strictly positive")
        return o

    def _composite(self, o: np.ndarray) -> float:
        return composite_metric(o, self.gm_weight, self.min_weight)

    def _state(self) -> np.ndarray:
        return np.concatenate([self.p, self.x]).astype(np.float32)

    def _update_best(self, o: np.ndarray) -> None:
        if float(o.min()) > self.best_min:
            self.best_min = float(o.min())
            self.best_x, self.best_p = self.x.copy(), self.p.copy()
            self.best_objectives, self.best_step = o.copy(), self.t

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self.rng = np.random.default_rng(seed)
        self.t = 0
        best = None
        for _ in range(self.start_candidates):
            x0, p0 = self.problem.initial_point(self.rng)
            o0 = self._evaluate(np.asarray(x0, np.float64), np.asarray(p0, np.float64))
            if best is None or o0.min() > best[2].min():
                best = (np.asarray(x0, np.float64), np.asarray(p0, np.float64), o0)
        self.x, self.p, o0 = best
        self.objectives = o0
        self.composite = self._composite(o0)
        self.best_min = -np.inf
        self._update_best(o0)
        return self._state()

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        n_p = self.problem.n_p
        self.p = action[:n_p].copy()
        self.x = self.x + action[n_p:]
        self.t += 1

        o = self._evaluate(self.x, self.p)
        c_new = self._composite(o)
        reward = log_ratio_reward(c_new, self.composite, self.reward_scale)
        self.composite, self.objectives = c_new, o
        self._update_best(o)

        return StepResult(
            state=self._state(),
            reward=reward,
            composite=c_new,
            objectives=o,
            target=self.problem.fairness_target(o),
            done=self.t >= self.max_steps,
        )
