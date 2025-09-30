from pathlib import Path
import json
from dataclasses import dataclass
import os

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder
from platoon.envs.codeact.types import CodeActObservation, CodeActAction, CodeActStep
from platoon.envs.base import SubTask, Task
from platoon.utils.prompt_retriever import PromptRetriever
from platoon.utils.prompt_retriever import get_prompt
from platoon.utils.llm_client import ConversationWithMetadata

@dataclass
class Supervisor:
    first_name: str
    last_name: str
    email: str
    phone_number: str


class AppWorldCodeActPromptBuilder(CodeActPromptBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
    
    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        messages: list[ConversationWithMetadata] = []
        
        for traj in traj_collection_dump["trajectories"].values():
            if traj["reward"] < reward_threshold:
                    continue
            task_id = traj["task"]["id"]
            task_specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
            with open(task_specs_path, "r") as f:
                task_specs = json.load(f)
            supervisor = Supervisor(**task_specs["supervisor"])
            appworld_env_prompts_path = Path(__file__).parent.parent.parent / "envs" / "appworld" / "prompts"
            action_space_description = get_prompt("user-action-space-description", prompts_dir=appworld_env_prompts_path, supervisor=supervisor)
            task = None
            if traj["task"] is not None:
                if "parent_tasks" in traj["task"]:
                    task = SubTask.from_dict(traj["task"])
                else:
                    task = Task.from_dict(traj["task"])
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
            
            for step_dict in traj["steps"]:
                step = CodeActStep(**step_dict)
                step.misc["traj_misc"] = traj_misc
                step.misc["step_num"] = len(obs.history)
                agent_action = CodeActAction(
                    parsed_code=step.code,
                    parsed_thought=step.thought,
                )
                messages.append(ConversationWithMetadata(
                    messages=self.build_messages(obs, agent_action),
                    misc=step.misc
                ))
                obs.history.append(step)
            
        return messages

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-env-specific-system-context")
        return super().build_system_prompt(obs, **context)

class AppWorldCodeActAgent(CodeActAgent):

    def __init__(self, *args, **kwargs):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = AppWorldCodeActPromptBuilder()
        super().__init__(*args, **kwargs)
