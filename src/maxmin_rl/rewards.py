"""Fairness reward: composite metric, log-ratio reward and the inverse-objective target.

All objectives ``o_k`` must be strictly positive (rates, utilities, ...). Shift or
rescale them in your problem if they are not.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def geometric_mean(objectives: np.ndarray, eps: float = _EPS) -> float:
    """``(prod_k o_k)^(1/K)``, computed in log-space for stability."""
    o = np.maximum(np.asarray(objectives, dtype=np.float64), eps)
    return float(np.exp(np.mean(np.log(o))))


def composite_metric(
    objectives: np.ndarray, gm_weight: float = 0.4, min_weight: float = 0.6
) -> float:
    """``C = gm_weight * GM(o) + min_weight * min(o)``.

    The geometric mean keeps every objective in the learning signal; the minimum
    keeps pressure on the current bottleneck.
    """
    o = np.asarray(objectives, dtype=np.float64)
    return gm_weight * geometric_mean(o) + min_weight * float(o.min())


def log_ratio_reward(c_new: float, c_old: float, scale: float = 1000.0) -> float:
    """``scale * log2(C_{t+1} / C_t)``: relative improvement, exactly zero when nothing changes."""
    return float(scale * np.log2(max(c_new, _EPS) / max(c_old, _EPS)))


def inverse_objective_target(objectives: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalised inverse objectives: the worse an objective, the larger its share."""
    inv = 1.0 / (np.asarray(objectives, dtype=np.float64) + eps)
    return inv / inv.sum()
