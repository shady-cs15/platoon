from pathlib import Path
from platoon.envs.base import Task
from platoon.envs.codeact import (
    IPythonCodeExecutor,
    ForkableCodeExecutor,
)

class ShoppingPlanningCodeExecutor(IPythonCodeExecutor, ForkableCodeExecutor):
    """Code executor for shopping planning."""

    def __init__(
        self, 
        task: Task,
        db_path: Path,
        tool_schema_path: Path,
    ) -> None:
        """Initialize the shopping planning code executor.
        
        Args:
            task: The task to execute
            db_path: The path to the database
            tool_schema_path: The path to the tool schema
        """
        self.db_path = db_path
        self.tool_schema_path = tool_schema_path
        super().__init__(task)

    async def reset(self) -> "ShoppingPlanningCodeExecutor":
        """Reset the code executor."""
        return self

    async def step(self, action: str) -> str:
        """Step the code executor."""
        return action