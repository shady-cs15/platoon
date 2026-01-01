from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.proxy import ModelInfo
from typing import Callable
from platoon.envs.base import Task
import asyncio
import tinker

class GroupCenteredRolloutWorkflow:
    def __init__(
        self, 
        rollout_fn: Callable[[Task, dict], dict],
        get_task_fn: Callable[[str], Task],
        config: WorkflowConfig
    ):
        self.config = config

    async def arun_episode_single(self, model_info: ModelInfo, data: dict, rollout_number: int=0) -> list[tinker.Datum] | None:
        ...

    async def arun_episode(self, model_info: ModelInfo, data: dict) -> list[tinker.Datum] | None:
        ...
        # results = await asyncio.gather(
        #     *[self.arun_episode_single(model_info, data, i) for i in range(8)] #range(self.config.rollout_config.num_rollouts)]
        # )

