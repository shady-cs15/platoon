from platoon.envs.codeact import (
    CodeActEnv,
    CodeActObservation,
)
from platoon.envs.base import Task


class ShoppingPlanningEnv(CodeActEnv):
    def __init__(
        self, 
        task: Task,
        db_path: Path,
        tool_schema_path: Path,

    ):
        super().__init__(task)

    async def reset(self) -> CodeActObservation:
        """Reset the environment and set action space."""
        obs = await super().reset()
        # Set action space description on both state and observation
        # (obs may be a deepcopy of state, so we need to set both)
        action_space = await self._code_executor.describe_action_space()
        self._state.action_space = action_space
        obs.action_space = action_space
        return obs

    async def evaluate(self) -> tuple[float, dict]:
        pass