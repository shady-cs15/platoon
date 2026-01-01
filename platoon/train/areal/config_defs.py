"""Configuration definitions for AReaL RL training."""

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig
from areal.engine.ppo.actor import PPOActorConfig

from platoon.utils.train import VariableBatchInferenceEngineConfig


@dataclass
class RolloutConfig:
    """Configuration for rollout execution."""
    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = 'rollout_results'
    verbose: bool = True
    timeout: int | None = None
    return_dict: bool = False
    group_size: int = 1
    

@dataclass
class WorkflowConfig:
    """Configuration for the rollout workflow."""
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class PlatoonArealRLTrainerConfig(GRPOConfig):
    """Main configuration for the AReaL RL trainer."""
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)

