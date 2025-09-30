from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from platoon.agents.base import Action
from platoon.envs.base import Observation, Task
from platoon.episode.trajectory import TrajectoryStep

def thought_processor(thought: str) -> str:
    return f"<think>\n{thought.strip()}\n</think>\n"

def code_processor(code: str) -> str:
    return f"<python>\n{code.strip()}\n</python>\n"

def output_processor(output: str) -> str:
    return f"<output>\n{output.strip()}\n</output>\n"

def error_processor(error: str) -> str:
    return f"<error>\n{error.strip()}\n</error>\n"

@dataclass
class CodeActStep(TrajectoryStep):
    code: str | None = None
    thought: str | None = None
    output: str | None = None
    error: str | None = None
    reward: float = 0.
    misc: dict = field(default_factory=dict)

    def __post_init__(self):
        self.thought_processor = thought_processor
        self.code_processor = code_processor
        self.output_processor = output_processor
        self.error_processor = error_processor

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
       if self.thought:
           thought = self.thought_processor(self.thought)
       else:
           thought = ""
       
       if self.code:
           code = self.code_processor(self.code)
       else:
           code = ""
       
       if self.output:
           output = self.output_processor(self.output)
       else:
           output = ""
           
       if self.error:
           error = self.error_processor(self.error)
       else:
           error = ""
       
       return thought + code + output + error


@dataclass
class CodeActAction(Action):
    action: str | None = None
    parsed_code: str | None = None
    parsed_thought: str | None = None
    misc: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return str(
            CodeActStep(
                code=self.parsed_code,
                thought=self.parsed_thought,
                misc=self.misc
            )
        )

# TODO: We probably should make action space more flexible (maybe add a more flexible tool abstraction with a registry)
# action space should maybe also support ToolGroups somehow? Maybe not here, but we can resolve ToolGroups in the env/codeexecutor.
@dataclass
class CodeActObservation(Observation):
    action_space: str = ""
    history: list[CodeActStep] = field(default_factory=list)

@runtime_checkable
class CodeExecutor(Protocol):

    async def run(self, code: str) -> CodeActStep:
        ...

    async def describe_action_space(self) -> str:
        ...

    async def reset(self) -> CodeExecutor:
        ...

    async def close(self) -> None:
        ...

@runtime_checkable
class ForkableCodeExecutor(CodeExecutor, Protocol):

    async def fork(self, task: Task) -> ForkableCodeExecutor:
        ...
