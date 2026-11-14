import sys
from pathlib import Path

import numpy as np
import pytest

from maxmin_rl import AgentConfig, FairnessEnv, FairnessProblem, train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from toy_phase_power import ToyPhasePower  # noqa: E402

SMALL = dict(
    actor_hidden=(32, 32),
    critic_hidden=(32, 32),
    batch_size=32,
    buffer_size=500,
    alpha_switch_step=None,
    device="cpu",
)


def test_env_tracks_the_best_solution_and_stays_feasible():
    problem = ToyPhasePower(n_users=3, n_phases=4)
    env = FairnessEnv(problem, max_steps=50)
    state = env.reset(seed=1)
    rng = np.random.default_rng(1)
    running_best = env.best_min
    for _ in range(50):
        shares = rng.dirichlet(np.ones(problem.n_p))
        theta_new = rng.uniform(0, 2 * np.pi, size=problem.n_x // 2)
        target = np.empty(problem.n_x)
        target[0::2], target[1::2] = np.cos(theta_new), np.sin(theta_new)
        res = env.step(np.concatenate([shares, target - env.x]))
        running_best = max(running_best, res.objectives.min())
        assert env.best_min == pytest.approx(running_best)
    assert res.done
    best_o = problem.objectives(env.best_x, env.best_p)
    assert best_o.min() == pytest.approx(env.best_min)


def test_env_rejects_non_positive_objectives():
    class Bad(ToyPhasePower):
        def objectives(self, x, p):
            return np.zeros(self.n_objectives)

    with pytest.raises(ValueError):
        FairnessEnv(Bad(n_users=2, n_phases=2)).reset(seed=0)


def test_training_runs_and_best_never_regresses():
    problem = ToyPhasePower(n_users=3, n_phases=4)
    res = train(problem, steps=300, seed=0, config=AgentConfig(**SMALL))
    assert res.min_history.shape == (300,)
    assert np.isfinite(res.min_history).all()
    assert res.best_min >= res.initial_min
    assert res.best_min == pytest.approx(res.min_history.max()) or res.best_min >= res.min_history.max()
    assert res.best_p.sum() == pytest.approx(1.0, abs=1e-5)
    xr, xi = res.best_x[0::2], res.best_x[1::2]
    assert np.allclose(np.hypot(xr, xi), 1.0, atol=1e-4)


def test_training_is_reproducible_for_a_fixed_seed():
    problem = ToyPhasePower(n_users=3, n_phases=4)
    cfg = AgentConfig(**SMALL)
    a = train(problem, steps=120, seed=7, config=cfg)
    b = train(problem, steps=120, seed=7, config=cfg)
    np.testing.assert_allclose(a.min_history, b.min_history)


class NoSimplex(FairnessProblem):
    """A problem with only an incremental block and no budget shares."""

    n_objectives, n_x, n_p = 2, 4, 0

    def initial_point(self, rng):
        return rng.uniform(0.5, 1.5, size=self.n_x), np.empty(0)

    def objectives(self, x, p):
        return np.array([1.0 + x[:2].sum() ** 2, 1.0 + x[2:].sum() ** 2])


def test_problem_without_a_simplex_block_trains():
    res = train(NoSimplex(), steps=150, seed=0, config=AgentConfig(**SMALL))
    assert np.isfinite(res.min_history).all()
    assert res.best_p.shape == (0,)
