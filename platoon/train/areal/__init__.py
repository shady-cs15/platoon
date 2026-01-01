"""AReaL training backend for Platoon.

This module provides the AReaL-based RL trainer for distributed training.
"""

from platoon.train.areal.config_defs import (
    PlatoonArealRLTrainerConfig,
    RolloutConfig,
    WorkflowConfig,
)
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.rl import PlatoonArealRLTrainer

__all__ = [
    "PlatoonArealRLTrainerConfig",
    "PlatoonArealRLTrainer",
    "ArealProxySession",
    "RolloutConfig",
    "WorkflowConfig",
]

