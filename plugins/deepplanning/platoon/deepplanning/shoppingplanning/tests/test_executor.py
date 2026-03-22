# Tests for ShoppingPlanningCodeExecutor.
# Cmd to run tests:
# PYTHONPATH=/root/platoon:/root/platoon/plugins/deepplanning uv run pytest -q \
# plugins/deepplanning/platoon/deepplanning/shoppingplanning/tests/test_executor.py

import asyncio
from pathlib import Path

import pytest

from platoon.envs.base import Task
from platoon.deepplanning.shoppingplanning.executor import ShoppingPlanningCodeExecutor


def _shoppingplanning_root() -> Path:
    # .../shoppingplanning/tests/test_executor.py -> .../shoppingplanning
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def case_dir() -> Path:
    p = _shoppingplanning_root() / "database" / "database_level1" / "case_1"
    assert p.exists(), f"Expected shopping db case dir to exist: {p}"
    return p


def test_executor_describe_action_space_contains_tools(case_dir: Path):
    task = Task(goal="test", id="deepplanning.shopping.level1.train.1", misc={})
    ex = ShoppingPlanningCodeExecutor(task=task, case_dir=case_dir)

    s = asyncio.run(ex.describe_action_space())

    # A few representative actions (don’t assert the whole string; keep it robust)
    assert "search_products" in s
    assert "get_cart_info" in s
    assert "add_product_to_cart" in s
    assert "finish" in s


def test_executor_run_can_call_tool(case_dir: Path):
    task = Task(goal="test", id="deepplanning.shopping.level1.train.1", misc={})
    ex = ShoppingPlanningCodeExecutor(task=task, case_dir=case_dir)

    code = """
cart = get_cart_info()
print("cart_type", type(cart).__name__)
if hasattr(cart, "keys"):
    print("cart_keys", sorted(list(cart.keys())))
"""

    step = asyncio.run(ex.run(code))

    # If the code had an exception, IPythonCodeExecutor puts it in `error`
    assert not step.error, f"Expected no stderr/traceback, got: {step.error}"
    assert "cart_type" in (step.output or "")