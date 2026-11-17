"""Multi-seed evaluation on the toy problem: the agent against simple baselines and one ablation.

Every method gets the same budget of objective evaluations (``--steps``) and is run on the
same set of problem instances (one per seed, so the unequal user couplings differ across
seeds). Reported: mean best-minimum-objective with a 95% bootstrap confidence interval,
and the fraction of seeds on which the method beats the equal-budget random search.

    python examples/benchmark.py --steps 3000 --seeds 5 --plot docs/benchmark.png

This is a sanity benchmark of the library on a synthetic problem. It does not reproduce
the paper's experiments.
"""

from __future__ import annotations

import argparse

import numpy as np

from maxmin_rl import AgentConfig, train
from toy_phase_power import ToyPhasePower  # run from the repo root or examples/


def random_search(problem: ToyPhasePower, steps: int, seed: int) -> float:
    """Best minimum objective over ``steps`` uniformly random feasible points."""
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(steps):
        x, _ = problem.initial_point(rng)
        p = rng.dirichlet(np.ones(problem.n_p))
        best = max(best, float(problem.objectives(x, p).min()))
    return best


def equal_budget_start(problem: ToyPhasePower, seed: int) -> float:
    """Random phases with an equal budget split: the agent's starting regime."""
    x, p = problem.initial_point(np.random.default_rng(seed))
    return float(problem.objectives(x, p).min())


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--plot", default=None)
    args = parser.parse_args()

    cfg = dict(alpha_switch_step=int(0.8 * args.steps), device="cpu")
    methods = {
        "initial point (equal split)": lambda pb, s: equal_budget_start(pb, s),
        "random search (same budget)": lambda pb, s: random_search(pb, args.steps, s),
        "agent, no CE regulariser": lambda pb, s: train(
            pb, args.steps, s, AgentConfig(use_ce=False, **cfg)
        ).best_min,
        "agent (full)": lambda pb, s: train(pb, args.steps, s, AgentConfig(**cfg)).best_min,
    }
    scores = {name: [] for name in methods}
    for seed in range(args.seeds):
        problem = ToyPhasePower(seed=seed)
        for name, fn in methods.items():
            scores[name].append(fn(problem, seed))
        print(f"seed {seed} done", flush=True)

    rs = np.array(scores["random search (same budget)"])
    print(f"\n{'method':32s} {'mean':>8s}   95% CI           beats random search")
    for name, vals in scores.items():
        v = np.array(vals)
        lo, hi = bootstrap_ci(v)
        wins = f"{int((v > rs).sum())}/{len(v)}" if "random" not in name else "-"
        print(f"{name:32s} {v.mean():8.3f}   [{lo:.3f}, {hi:.3f}]   {wins}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(scores)
        fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
        for i, name in enumerate(names):
            v = np.array(scores[name])
            lo, hi = bootstrap_ci(v)
            ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", capsize=3)
            ax.scatter(v, np.full(len(v), i), s=10, alpha=0.4, color="gray")
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel(f"best minimum objective ({args.seeds} seeds, mean and 95% bootstrap CI)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(args.plot)


if __name__ == "__main__":
    main()
