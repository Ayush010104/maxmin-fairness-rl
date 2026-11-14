"""In-graph projections onto a constraint set.

A projection maps a batch of raw proposals ``x + step`` of shape ``(B, n_x)`` back
onto the feasible set. It runs *inside* the autograd graph of the actor, so
gradients flow through it. Any differentiable ``Callable[[Tensor], Tensor]`` works.
"""

from __future__ import annotations

from typing import Callable

import torch

Projection = Callable[[torch.Tensor], torch.Tensor]


def identity(x: torch.Tensor) -> torch.Tensor:
    """No constraint."""
    return x


def unit_modulus(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project interleaved ``[re_0, im_0, re_1, im_1, ...]`` pairs onto the unit circle."""
    re, im = x[..., 0::2], x[..., 1::2]
    norm = torch.sqrt(re * re + im * im).clamp_min(eps)
    return torch.stack([re / norm, im / norm], dim=-1).reshape(x.shape)


def box(low: float, high: float) -> Projection:
    """Element-wise clamp to ``[low, high]``."""

    def project(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, low, high)

    return project
