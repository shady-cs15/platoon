"""Shared configuration definitions for Platoon.

This module contains configuration classes that are used across different
parts of the codebase (rollouts, training, inference, etc.).
"""

from dataclasses import dataclass, field


@dataclass
class InferenceParams:
    """Inference parameters for rollout-time model calls.

    Defaults preserve current behavior to keep existing configs backwards compatible.
    """

    temperature: float | None = 1.0
    top_p: float | None = None
    max_completion_tokens: int = 512


@dataclass
class RolloutConfig:
    """Configuration for rollout execution.

    This configuration is used for running agent rollouts, whether for
    training, evaluation, or standalone inference.
    """

    model_name: str | None = None
    model_endpoint: str | None = None
    model_api_key: str | None = None
    train: bool = False
    max_steps: int | None = None
    output_dir: str = "rollout_results"
    verbose: bool = True
    timeout: int | None = None  # Trajectory timeout (entire rollout)
    step_timeout: int = 300  # Per-step timeout (agent.act + env.step)
    return_dict: bool = False
    subagent_success_threshold: float | None = None  # When set, binarize subagent reward (1.0 if >= threshold, else 0.0)
    rubric_model: str | None = None
    rubric_base_url: str | None = None
    rubric_api_key: str | None = None
    rubric_api_key_env: str | None = None
    propagate_root_success: bool = False  # Skip subtask LLM judge and copy root reward/success to all child trajectories
    hierarchical_subagent_judging: bool = False  # Use LLM-judged correct+useful credit assignment for subagents
    failed_root_local_reward_weight: float = 0.2  # Weight for local rewards when root task fails
    hierarchical_gated_bonus_weight: float = 0.4  # Gated usefulness bonus weight for hierarchical delegation rewards
    hierarchical_unconditional_bonus_weight: float = 0.0  # Ungated usefulness bonus weight for hierarchical delegation rewards
    inference_params: InferenceParams = field(default_factory=InferenceParams)

    def __post_init__(self) -> None:
        # Support loading from plain dicts from config loaders and subprocess paths.
        if isinstance(self.inference_params, dict):
            self.inference_params = InferenceParams(**self.inference_params)
