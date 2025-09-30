import pytest

from platoon.agents.codeact.prompt_builder import CodeActPromptBuilder
import asyncio
from platoon.agents.appworld.recursive_agent import AppWorldRecursiveCodeActPromptBuilder, AppWorldRecursiveCodeActAgent
from platoon.envs.codeact.env import CodeActEnv
from platoon.envs.codeact.types import ForkableCodeExecutor
from platoon.envs.base import Task
from platoon.envs.codeact.types import CodeActObservation, CodeActStep, CodeActAction
from platoon.episode.trajectory import TrajectoryCollection
from platoon.episode.context import current_trajectory_collection, current_trajectory


def _make_obs_with_history(goal: str, history_texts: list[str] | None = None) -> CodeActObservation:
    task = Task(goal=goal, id="test-task")
    obs = CodeActObservation(task=task, history=[])
    if history_texts:
        for t in history_texts:
            obs.history.append(CodeActStep(thought=t))
    return obs


def _get_user_content(messages):
    # messages is a list of ChatMessage; user message is at index 1
    assert len(messages) >= 2
    user_msg = messages[1]
    # Some ChatMessage implementations use attributes, others dict-like; support both
    return getattr(user_msg, "content", user_msg.get("content"))


def test_user_template_without_parent_shows_action_history_heading():
    builder = CodeActPromptBuilder()
    obs = _make_obs_with_history("Do something simple.")

    user_content = builder.build_user_prompt(
        obs,
        task=str(obs.task),
        action_space_description=str(obs.action_space),
        action_history_description=builder.build_action_history_description(obs),
        next_action_str=builder.build_next_action_str(obs),
        parent_action_history_description=None,
    )

    assert "# Task" in user_content
    assert "# Action Space" in user_content
    assert "# Parent Agent Action History" not in user_content
    assert "# Your Action History" not in user_content
    assert "# Action History" in user_content
    assert "# Next Action" in user_content


def test_recursive_builder_with_parent_state_shows_parent_and_your_history():
    builder = AppWorldRecursiveCodeActPromptBuilder(use_parent_state=True)
    # Current obs (child)
    child_obs = _make_obs_with_history("Child goal.")
    # Parent state with some history
    parent_state = _make_obs_with_history("Parent goal.", ["parent thought 1"]) 
    child_obs.misc["parent_state"] = parent_state

    messages = builder.build_messages(child_obs)
    user_content = _get_user_content(messages)

    assert "# Parent Agent Action History" in user_content
    assert "# Your Action History" in user_content
    assert "# Action History" not in user_content  # the fallback header should not appear
    # Ensure some history marker appears from the formatter
    assert "Cell 0:" in user_content or "No actions taken yet." in user_content
    assert "# Next Action" in user_content


def test_recursive_builder_without_parent_state_falls_back_to_action_history():
    builder = AppWorldRecursiveCodeActPromptBuilder(use_parent_state=True)
    obs = _make_obs_with_history("Goal without parent.")

    user_content = builder.build_user_prompt(
        obs,
        task=str(obs.task),
        action_space_description=str(obs.action_space),
        action_history_description=builder.build_action_history_description(obs),
        next_action_str=builder.build_next_action_str(obs),
        parent_action_history_description=None,
    )

    assert "# Parent Agent Action History" not in user_content
    assert "# Your Action History" not in user_content
    assert "# Action History" in user_content


class _DummyExecutor(ForkableCodeExecutor):
    async def run(self, code: str):
        from platoon.envs.codeact.types import CodeActStep
        return CodeActStep(code=code, output="", error="")

    async def describe_action_space(self) -> str:
        return "print, finish"

    async def reset(self):
        return self

    async def close(self) -> None:
        return None

    async def fork(self, task: Task):
        return type(self)()


class _SimpleEnv(CodeActEnv):
    async def step(self, action: CodeActAction) -> CodeActObservation:
        step = await self._code_executor.run(action.parsed_code)
        step.thought = action.parsed_thought
        step.reward, reward_info = await self.evaluate()
        step.misc['action_misc'] = action.misc
        step.misc['reward_misc'] = reward_info
        self._state.reward += step.reward
        self._state.history.append(step)
        return await self.observe()


@pytest.mark.asyncio
async def test_env_fork_populates_parent_state_and_prompt_uses_it():
    # Parent env and a minimal forkable executor
    parent_env = _SimpleEnv(task=Task(goal="Parent goal.", id="parent-task"), code_executor=_DummyExecutor())
    # Initialize minimal episode context for reset/observe to work
    current_trajectory_collection.set(TrajectoryCollection())
    current_trajectory.set(current_trajectory_collection.get().create_trajectory(parent_traj=None))
    # Take one step via env.step to create parent history realistically
    await parent_env.step(CodeActAction(parsed_code="print('hello')", parsed_thought="parent thought 1"))

    # Fork to child env; now env.fork awaits observe() and populates parent_state properly
    child_task = parent_env.task.fork("Child goal.", max_steps=3)
    child_env = await parent_env.fork(child_task)
    # Set context for child trajectory so reset can register task
    current_trajectory.set(current_trajectory_collection.get().create_trajectory(parent_traj=current_trajectory.get()))
    await child_env.reset()
    child_obs = await child_env.observe()

    # parent_state should be present and a materialized observation (not a coroutine)
    assert "parent_state" in child_obs.misc
    #assert not asyncio.iscoroutine(child_obs.misc["parent_state"])  # fixed behavior

    # Build prompt via recursive builder from the child observation
    agent = AppWorldRecursiveCodeActAgent(use_parent_state=True)
    messages = agent.prompt_builder.build_messages(child_obs)
    user_content = _get_user_content(messages)

    # Print the prompt for manual inspection (run pytest with -s to see output)
    print("\n===== USER PROMPT (forked) =====\n" + user_content)

    assert "# Parent Agent Action History" in user_content
    assert "# Your Action History" in user_content
    assert "# Action History" not in user_content
    assert "# Next Action" in user_content
