from pathlib import Path

from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, CodeActObservation

from platoon.deepplanning.shoppingplanning.executor import ShoppingPlanningCodeExecutor


class ShoppingPlanningEnv(CodeActEnv):
    def __init__(self, task: Task):
        # task.misc should contain "case_dir" from your tasks.py
        case_dir = Path(task.misc["case_dir"])
        code_executor = ShoppingPlanningCodeExecutor(task=task, case_dir=case_dir)
        super().__init__(task, code_executor)

    async def reset(self) -> CodeActObservation:
        obs = await super().reset()
        action_space = await self._code_executor.describe_action_space()
        self._state.action_space = action_space
        obs.action_space = action_space
        return obs

    async def evaluate(self) -> tuple[float, dict]:
        raise NotImplementedError("ShoppingPlanningEnv does not support evaluation")