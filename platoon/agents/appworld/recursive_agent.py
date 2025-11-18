from __future__ import annotations

from copy import copy, deepcopy
from pathlib import Path
import json
import os

from platoon.utils.llm_client import ConversationWithMetadata
from platoon.agents.appworld.codeact import AppWorldCodeActAgent, AppWorldCodeActPromptBuilder, Supervisor
from platoon.envs.base import SubTask, Task
from platoon.envs.codeact import CodeActObservation, CodeActAction, CodeActStep
from platoon.utils.llm_client import Conversation
from platoon.utils.prompt_retriever import PromptRetriever
from platoon.utils.prompt_retriever import get_prompt

class AppWorldRecursiveCodeActPromptBuilder(AppWorldCodeActPromptBuilder):
    def __init__(self, *args, use_parent_state: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
        self.use_parent_state = use_parent_state

    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        messages: list[ConversationWithMetadata] = []
        reconstructed_obs: dict[str, CodeActObservation] = {}
        
        supervisor = None

        for i, traj in enumerate(traj_collection_dump["trajectories"].values()):
            if i == 0:
                task_id = traj["task"]["id"]
                task_specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
                with open(task_specs_path, "r") as f:
                    task_specs = json.load(f)
                supervisor = Supervisor(**task_specs["supervisor"])
            
            # If we don't use parent state, we can skip early if reward is below threshold.
            # Otherwise, we need to wait until we have reconstructed the observation, because
            # a low reward parent can still have a high reward child.
            if traj["reward"] < reward_threshold and not self.use_parent_state:
                    continue
            
            task = None
            current_task_is_subtask = False
            if traj["task"] is not None:
                if "parent_tasks" in traj["task"] and len(traj["task"]["parent_tasks"]) > 0:
                    task = SubTask.from_dict(traj["task"])
                    current_task_is_subtask = True
                else:
                    task = Task.from_dict(traj["task"])
            
            appworld_env_prompts_path = Path(__file__).parent.parent.parent / "envs" / "appworld" / "prompts"
            action_space_description = get_prompt(
                "user-recursive-action-space-description",
                prompts_dir=appworld_env_prompts_path,
                supervisor=supervisor,
                current_task_is_subtask=current_task_is_subtask
            )
            reward = traj["reward"]
            misc = traj["misc"]
            history = []
            
            obs = CodeActObservation(
                task=task,
                reward=reward,
                misc=misc,
                history=history,
                action_space=action_space_description
            )
            
            traj_misc = {
                "id": traj["id"],
                "task": traj["task"],
                "parent_info": traj["parent_info"],
                "reward": traj["reward"],
                **traj["misc"],
            }

            if self.use_parent_state and traj["parent_info"] is not None:
                obs.misc["parent_state"] = deepcopy(reconstructed_obs[traj["parent_info"]["id"]])
                obs.misc["parent_state"].history = obs.misc["parent_state"].history[:traj["parent_info"]["fork_step"]]
            
            reconstructed_obs[traj["id"]] = obs

            for step_dict in traj["steps"]:
                step = CodeActStep(**step_dict)
                step.misc["traj_misc"] = traj_misc
                step.misc["step_num"] = len(obs.history)
                agent_action = CodeActAction(
                    parsed_code=step.code,
                    parsed_thought=step.thought,
                )
                if traj["reward"] >= reward_threshold:
                    messages.append(ConversationWithMetadata(
                        messages=self.build_messages(obs, agent_action),
                        misc=step.misc
                    ))
                obs.history.append(step)
            
        return messages

    def build_messages(self, obs: CodeActObservation, agent_action: CodeActAction | None = None) -> Conversation:
        obs = copy(obs)
        obs.task = SubTask.from_task(obs.task)
        return super().build_messages(obs, agent_action)

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-recursive-env-specific-system-context")
        return super().build_system_prompt(obs, **context)

    def build_user_prompt(self, obs: CodeActObservation, **context) -> str:
        if self.use_parent_state and 'parent_state' in obs.misc:
            parent_action_history = self.build_action_history_description(obs.misc['parent_state']) 
            return super().build_user_prompt(obs, parent_action_history_description=parent_action_history, **context)
        else:
            return super().build_user_prompt(obs, **context)

class AppWorldRecursiveCodeActAgent(AppWorldCodeActAgent):

    def __init__(self, use_parent_state: bool = False, **kwargs):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = AppWorldRecursiveCodeActPromptBuilder(use_parent_state=use_parent_state)
        super().__init__(**kwargs)

    async def fork(self, task: Task) -> AppWorldRecursiveCodeActAgent:
        return AppWorldRecursiveCodeActAgent(
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        ) # TODO: Have cleaner kwargs story.