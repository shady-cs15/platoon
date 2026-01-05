"""AReaL training backend for Platoon.

This module provides the AReaL-based RL trainer for distributed training.
"""

# Apply areal patches before importing areal-dependent modules
from platoon.train.areal.patches import apply_all_patches
apply_all_patches()

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

