from __future__ import annotations

from copy import deepcopy
from typing import Protocol, runtime_checkable

from platoon.envs.base import Task

from .types import CodeExecutor, ForkableCodeExecutor, CodeActObservation, CodeActAction
from platoon.episode.context import finish_message
from platoon.episode.context import current_trajectory_collection, current_trajectory, error_message


@runtime_checkable
class CodeActEnv(Protocol):

    # TODO: Come back to think about how we might want to add a factory + config to be string-based configurable.
    def __init__(self, task: Task, code_executor: CodeExecutor, return_obs_copy: bool = True, parent_state: CodeActObservation | None = None, **kwargs):
        self._code_executor = code_executor
        self._task = task
        self._parent_state = parent_state
        self._state = CodeActObservation(task=task, history=[])
        if parent_state is not None:
            self._state.misc["parent_state"] = parent_state
        self._return_obs_copy = return_obs_copy
        self._init_kwargs = kwargs

    @property
    def task(self) -> Task:
        return self._task

    @property
    def code_executor(self) -> CodeExecutor:
        return self._code_executor

    async def reset(self) -> CodeActObservation:
        self._code_executor = await self._code_executor.reset()
        self._state = CodeActObservation(task=self._state.task, history=[])
        if self._parent_state is not None:
            self._state.misc["parent_state"] = self._parent_state
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        # This is not needed here, but we keep it here for demonstration in case an env wants to add steps/obs during reset.
        for step in self._state.history:
            traj_collection.add_trajectory_step(traj.id, step)
        return await self.observe()
    
    async def evaluate(self) -> tuple[float, dict]:
        return 0., {}
    
    async def step(self, action: CodeActAction) -> CodeActObservation:
        step = await self._code_executor.run(action.parsed_code)
        
        if finish_message.get(None) is not None or error_message.get(None) is not None:
            self._state.finished = True
            self._state.misc["finish_message"] = finish_message.get()

        step.thought = action.parsed_thought
        step.reward, reward_info = await self.evaluate()
        step.misc['action_misc'] = action.misc
        step.misc['reward_misc'] = reward_info
        self._state.reward += step.reward
        self._state.history.append(step)
        
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, self._state.history[-1])
        if self._state.finished:
            traj.reward = self._state.reward
        
        return await self.observe()
    
    async def close(self) -> None:
        self._state = CodeActObservation(task=self._state.task, history=[])
        await self._code_executor.close()

    async def observe(self, return_copy: bool | None = None) -> CodeActObservation:
        if return_copy is None:
            return_copy = self._return_obs_copy
            
        if return_copy:
            return deepcopy(self._state)
        else:
            return self._state
    
    async def fork(self, task: Task) -> CodeActEnv:
        if isinstance(self._code_executor, ForkableCodeExecutor):
            return type(self)(
                task=task,
                code_executor=await self._code_executor.fork(task=task),
                return_obs_copy=self._return_obs_copy,
                parent_state=await self.observe(),
                **deepcopy(self._init_kwargs)
            )
        else:
            raise ValueError("CodeExecutor is not forkable. "
            "Either implement fork() for your CodeExecutor or implement a new ForkableEnv for this task."
            )
