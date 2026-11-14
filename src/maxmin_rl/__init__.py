"""Deep RL for max-min fairness over any constrained optimisation problem."""

from .agent import AgentConfig, HybridSACDDPGAgent, ce_lambda
from .env import FairnessEnv, StepResult
from .problem import FairnessProblem
from .projections import box, identity, unit_modulus
from .rewards import (
    composite_metric,
    geometric_mean,
    inverse_objective_target,
    log_ratio_reward,
)
from .trainer import TrainResult, train

__version__ = "0.1.0"

__all__ = [
    "AgentConfig",
    "FairnessEnv",
    "FairnessProblem",
    "HybridSACDDPGAgent",
    "StepResult",
    "TrainResult",
    "box",
    "ce_lambda",
    "composite_metric",
    "geometric_mean",
    "identity",
    "inverse_objective_target",
    "log_ratio_reward",
    "train",
    "unit_modulus",
]
