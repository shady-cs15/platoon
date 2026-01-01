from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AdamParams:
    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

# TODO: inference config defaults, may need to be different for training vs. eval
@dataclass
class WorkflowConfig:
    group_size: int = 8


# TODO:
# - timeout
@dataclass
class TrainConfig:
    batch_size: int = 32
    # Number of minibatches per batch. Batch must be divisible by num_minibatches.
    # Results in num_minibatches weight updates per batch.
    num_minibatches: int = 1
    # Number of microbatches per minibatch. Minibatch must be divisible by num_microbatches.
    # Used for gradient accumulation --> A single weight update per microbatch.
    # With tinker, gradient accumulation is less important, but this may still be useful for streaming
    # to overlap rollout sampling with forward backward computation within the same batch.
    num_microbatches: int = 1
    optimizer: AdamParams = field(default_factory=AdamParams)
    loss_fn: str = 'cispo'
    loss_fn_config: dict = {'clip_low_threshold': 0., 'clip_high_threshold': 5.}
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    num_concurrent_rollout_workflow_workers: int | None = None

    def __post_init__(self):
        if self.num_concurrent_rollout_workflow_workers is None:
            self.num_concurrent_rollout_workflow_workers = self.batch_size

@dataclass
class TrainEventTriggerConfig:
    strategy: Literal['epoch', 'step', 'none'] = "epoch"
    every: int = 1

@dataclass
class CheckpointConfig(TrainEventTriggerConfig):
    load_checkpoint_path: str | None = None

@dataclass
class EvalConfig(TrainEventTriggerConfig):
    num_concurrent_rollout_workflow_workers: int = 256
    workflow_config: WorkflowConfig = field(default_factory=lambda: WorkflowConfig(group_size=1))

@dataclass
class PlatoonTinkerRLTrainerConfig:
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    log_path: str
