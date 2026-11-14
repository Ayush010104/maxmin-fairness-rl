"""The interface a user implements to plug a problem into the agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

from .projections import identity
from .rewards import inverse_objective_target


class FairnessProblem(ABC):
    """A max-min fairness problem over two blocks of variables.

    * ``x`` (size ``n_x``): updated *incrementally*. The stochastic (SAC) head
      proposes a step and ``project_x`` maps ``x + step`` back onto the feasible set.
    * ``p`` (size ``n_p``): chosen *directly* each step as a point on the simplex
      (budget shares summing to 1) by the deterministic (DDPG) head. Set
      ``n_p = 0`` if your problem has no such block.

    The state seen by the agent is ``concat([p, x])``.
    """

    n_objectives: int
    n_x: int
    n_p: int = 0

    @abstractmethod
    def initial_point(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Return a (random) feasible starting point ``(x0, p0)``.

        ``p0`` must have shape ``(n_p,)`` (an empty array if ``n_p == 0``).
        """

    @abstractmethod
    def objectives(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Return the ``n_objectives`` strictly positive values to be made fair."""

    def project_x(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable projection of ``(B, n_x)`` proposals onto the feasible set."""
        return identity(x)

    def fairness_target(self, objectives: np.ndarray) -> Optional[np.ndarray]:
        """Target distribution for the cross-entropy regulariser on ``p``.

        Defaults to the normalised inverse objectives, which requires
        ``n_p == n_objectives``. Return ``None`` to disable the regulariser.
        """
        if self.n_p > 0 and self.n_p == self.n_objectives:
            return inverse_objective_target(objectives)
        return None
