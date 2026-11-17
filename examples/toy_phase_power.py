"""A small, self-contained example problem (not tied to any physical system).

``K`` users share a budget ``p`` (simplex block). Each user's value depends on
``M`` unit-modulus phases ``x`` through ``g_k = |sum_m A[k, m] * exp(j*theta_m)|^2``::

    o_k = log2(1 + snr * K * p_k * g_k)

The user couplings ``A`` are deliberately unequal, so a max-min fair solution has to
steer the phases towards the weak users *and* shift budget towards them.

Run:  python examples/toy_phase_power.py --steps 3000
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch

from maxmin_rl import AgentConfig, FairnessProblem, train, unit_modulus


class ToyPhasePower(FairnessProblem):
    def __init__(self, n_users: int = 4, n_phases: int = 16, snr: float = 10.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        coupling = rng.normal(size=(n_users, n_phases)) + 1j * rng.normal(size=(n_users, n_phases))
        coupling /= np.sqrt(2 * n_phases)
        self.A = coupling * np.geomspace(1.0, 0.25, n_users)[:, None]
        self.snr = snr
        self.n_objectives = n_users
        self.n_x = 2 * n_phases  # interleaved [re, im] per phase
        self.n_p = n_users

    def initial_point(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        theta = rng.uniform(0, 2 * np.pi, size=self.n_x // 2)
        x = np.empty(self.n_x)
        x[0::2], x[1::2] = np.cos(theta), np.sin(theta)
        return x, np.full(self.n_p, 1.0 / self.n_p)

    def objectives(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        z = x[0::2] + 1j * x[1::2]
        gain = np.abs(self.A @ z) ** 2
        return np.log2(1.0 + self.snr * self.n_objectives * p * gain)

    def project_x(self, x: torch.Tensor) -> torch.Tensor:
        return unit_modulus(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="save a convergence plot to this path (needs matplotlib)")
    args = parser.parse_args()

    problem = ToyPhasePower(seed=args.seed)
    # A shorter run than the default 10k steps, so switch the entropy temperature earlier.
    config = AgentConfig(alpha_switch_step=int(0.8 * args.steps))
    result = train(problem, steps=args.steps, seed=args.seed, config=config, verbose=True)

    print(f"\nstarting min objective : {result.initial_min:.4f}")
    print(f"best min objective     : {result.best_min:.4f}  (step {result.best_step})")
    print("per-objective at best  :", np.round(result.best_objectives, 3))
    print("budget shares at best  :", np.round(result.best_p, 3))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
        ax.plot(result.min_history, lw=0.6, alpha=0.5, label="minimum objective per step")
        ax.plot(np.maximum.accumulate(result.min_history), lw=1.8, label="best so far")
        ax.set_xlabel("environment step")
        ax.set_ylabel("min objective")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.plot)


if __name__ == "__main__":
    main()
