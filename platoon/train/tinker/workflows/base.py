from typing import Protocol
from platoon.train.tinker.proxy import ModelInfo
import tinker

class RolloutWorkflow(Protocol):
    async def arun_episode(self, model_info: ModelInfo, data: dict) -> list[tinker.Datum] | None:
        ...
