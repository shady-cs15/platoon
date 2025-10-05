from dataclasses import dataclass
from typing import Any, Literal, Optional
from platoon.utils.llm_client import LLMClient


@dataclass
class LLMClientSpec:
    kind: Literal["openai", "areal"] = "openai"
    model_name: str = "neulab/claude-sonnet-4-20250514"
    base_url: Optional[str] = None
    engine: Optional[str] = None
    # Optional engine spec for spawn-safe reconstruction (Areal)
    areal_engine_spec: "ArealEngineSpec | None" = None


@dataclass
class ArealEngineSpec:
    class_path: str  # e.g., "areal.engine.sglang_remote:RemoteSGLangEngine"
    config: dict[str, Any]
    initialize_kwargs: Optional[dict[str, Any]] = None

@dataclass # TODO: We should consider just passing these params as inputs to the generate_rollouts_batch method.
class RolloutGeneratorConfig:
    max_parallel_processes: int
    model_name: str
    max_steps_per_rollout: int
    output_dir: str
    verbose: bool = True
    per_rollout_timeout_seconds: int = 600
    model_endpoint: str | None = None
    llm_client: LLMClient | None = None
    llm_client_spec: LLMClientSpec | None = None