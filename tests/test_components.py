import math

import numpy as np
import pytest
import torch

from maxmin_rl import (
    HybridSACDDPGAgent,
    ce_lambda,
    composite_metric,
    geometric_mean,
    inverse_objective_target,
    log_ratio_reward,
    unit_modulus,
)
from maxmin_rl.networks import HybridActor
from maxmin_rl.replay import ReplayBuffer


def test_unit_modulus_projects_pairs_onto_circle():
    x = torch.randn(5, 8) * 3
    y = unit_modulus(x)
    norms = torch.sqrt(y[:, 0::2] ** 2 + y[:, 1::2] ** 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_geometric_mean_and_composite():
    o = np.array([1.0, 4.0])
    assert geometric_mean(o) == pytest.approx(2.0)
    assert composite_metric(o, gm_weight=0.4, min_weight=0.6) == pytest.approx(0.4 * 2.0 + 0.6 * 1.0)


def test_log_ratio_reward_is_zero_without_change_and_signed():
    assert log_ratio_reward(3.0, 3.0) == 0.0
    assert log_ratio_reward(4.0, 2.0, scale=1.0) == pytest.approx(1.0)
    assert log_ratio_reward(1.0, 2.0) < 0


def test_inverse_target_is_a_distribution_favouring_the_worst():
    t = inverse_objective_target(np.array([1.0, 2.0, 4.0]))
    assert t.sum() == pytest.approx(1.0)
    assert t[0] > t[1] > t[2]


@pytest.mark.parametrize("loss, expected", [(3.7, 0.1), (-42.0, 1.0), (0.05, 0.001), (1234.0, 100.0)])
def test_ce_lambda_is_one_decade_below_policy_loss(loss, expected):
    assert ce_lambda(loss, 0.1) == pytest.approx(expected)
    assert ce_lambda(loss, 0.1) == pytest.approx(0.1 * 10 ** math.floor(math.log10(abs(loss))))


def test_stored_action_is_the_projected_delta():
    n_p, n_x = 3, 8
    actor = HybridActor(n_p + n_x, n_p, n_x, unit_modulus, hidden=(16, 16)).eval()
    x = unit_modulus(torch.randn(6, n_x))
    state = torch.cat([torch.full((6, n_p), 1 / n_p), x], dim=1)
    action, log_prob = actor.sample(state, reparameterize=False)

    shares, delta = action[:, :n_p], action[:, n_p:]
    assert torch.allclose(shares.sum(dim=1), torch.ones(6), atol=1e-5)
    landed = x + delta
    norms = torch.sqrt(landed[:, 0::2] ** 2 + landed[:, 1::2] ** 2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert torch.isfinite(log_prob).all()


def test_reparameterised_sample_carries_gradients():
    actor = HybridActor(2 + 4, 2, 4, lambda v: v, hidden=(16, 16))
    state = torch.randn(8, 2 + 4)
    action, log_prob = actor.sample(state, reparameterize=True)
    (action.sum() + log_prob.sum()).backward()
    assert actor.mu_head.weight.grad is not None
    assert actor.p_head.weight.grad is not None


def test_agent_learn_waits_for_a_full_batch():
    agent = HybridSACDDPGAgent(6, 2, 4, lambda v: v, seed=0)
    assert agent.learn(np.array([0.5, 0.5])) is None


def test_prioritised_sampling_prefers_high_priority_slots():
    buf = ReplayBuffer(50, 3, 2, torch.device("cpu"))
    for i in range(50):
        buf.add(np.full(3, i), np.zeros(2), 0.0, np.zeros(3), False)
    buf.reset_priorities()
    buf.update_priorities(np.array([7]), np.array([1e4]))
    rng = np.random.default_rng(0)
    weight = (np.log1p(1e4) + 1e-6) ** 1.3
    expected = weight / (weight + 49)  # the other 49 slots keep priority 1
    *_, idx = buf.sample(4000, rng, exponent=1.3)
    assert abs((idx == 7).mean() - expected) < 0.03
    *_, uniform_idx = buf.sample(4000, rng)
    assert abs((uniform_idx == 7).mean() - 1 / 50) < 0.02


@pytest.mark.parametrize("n_p", [2, 4, 8])
def test_share_head_is_confined_to_the_documented_interval(n_p):
    actor = HybridActor(n_p + 4, n_p, 4, lambda v: v, hidden=(16, 16)).eval()
    with torch.no_grad():
        actor.p_head.weight.mul_(1e3)  # saturate the sigmoid in both directions
        shares, _, _ = actor(torch.randn(256, n_p + 4) * 5)
    lo, hi = 1 / (1 + (n_p - 1) * math.e), math.e / (math.e + n_p - 1)
    assert shares.min().item() >= lo - 1e-6
    assert shares.max().item() <= hi + 1e-6
