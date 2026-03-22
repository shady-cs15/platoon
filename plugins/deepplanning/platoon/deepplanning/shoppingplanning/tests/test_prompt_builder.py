# Tests for ShoppingPlanningPromptBuilder.
# cd /root/platoon
# PYTHONPATH=/root/platoon:/root/platoon/plugins/deepplanning uv run pytest -q \
# plugins/deepplanning/platoon/deepplanning/shoppingplanning/tests/test_prompt_builder.py

import pytest

from platoon.envs.base import Task
from platoon.envs.codeact.types import CodeActObservation
from platoon.deepplanning.shoppingplanning.prompt_builder import ShoppingPlanningPromptBuilder


@pytest.mark.parametrize("level, expected_snippet", [
    (1, "Core Mission"),
    (2, "while strictly adhering to their specified budget"),
    (3, "products and coupons"),
])
def test_build_system_prompt_selects_level_prompt(level: int, expected_snippet: str):
    pb = ShoppingPlanningPromptBuilder(prompt_mode="sequence_extension", include_reasoning=True)
    task = Task(goal="dummy", id="t0", misc={"level": level})
    obs = CodeActObservation(task=task, action_space="(action space here)", history=[])

    s = pb.build_system_prompt(obs)

    assert expected_snippet in s
    assert "writing Python code" in s
    assert "call finish('...')" in s
    assert "<thought>" in s
    assert "<python>" in s


def test_build_system_prompt_no_reasoning_mode():
    pb = ShoppingPlanningPromptBuilder(prompt_mode="sequence_extension", include_reasoning=False)
    task = Task(goal="dummy", id="t0", misc={"level": 1})
    obs = CodeActObservation(task=task, action_space="(action space here)", history=[])

    s = pb.build_system_prompt(obs)

    assert "<python>" in s
    assert "<thought>" not in s  # should not instruct thought tags


def test_build_messages_includes_task_and_action_space():
    pb = ShoppingPlanningPromptBuilder(prompt_mode="sequence_extension", include_reasoning=True)

    task = Task(goal="buy stuff", id="t0", misc={"level": 1})
    obs = CodeActObservation(
        task=task,
        action_space="Available actions:\n- search_products(query: str, limit: int = 20)\n",
        history=[],
    )

    msgs = pb.build_messages(obs)

    assert msgs[0]["role"] == "system"
    assert "Shopping Assistant" in msgs[0]["content"] or "Core Mission" in msgs[0]["content"]

    assert msgs[1]["role"] == "user"
    # initial user message should include Task and Action Space sections (fallback builder)
    assert "Task" in msgs[1]["content"]
    assert "Action Space" in msgs[1]["content"]
    assert "search_products" in msgs[1]["content"]