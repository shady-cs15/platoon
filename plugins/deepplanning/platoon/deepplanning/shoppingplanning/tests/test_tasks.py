# Tests for shopping planning task extraction.
# Cmd to run: 
# PYTHONPATH=/root/platoon:/root/platoon/plugins/deepplanning uv run pytest -q \
# platoon/deepplanning/shoppingplanning/tests/test_tasks.py

import json
from pathlib import Path

import pytest

from platoon.deepplanning.shoppingplanning import tasks as t


def _base_dir() -> Path:
    # .../shoppingplanning/tasks.py -> .../shoppingplanning
    return Path(t.__file__).resolve().parent


def _query_meta_path(level: int) -> Path:
    return _base_dir() / "data" / f"level_{level}_query_meta.json"


def _num_queries(level: int) -> int:
    p = _query_meta_path(level)
    assert p.exists(), f"Missing query meta file: {p}"
    rows = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(rows, list) and rows, f"Expected non-empty list in {p}"
    return len(rows)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_get_task_ids_respects_limit(level: int):
    ids = t.get_task_ids("train", level=level, limit=3)
    assert isinstance(ids, list)
    assert len(ids) == 3
    assert all(isinstance(x, str) for x in ids)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_get_task_ids_format(level: int):
    ids = t.get_task_ids("train", level=level, limit=5)
    assert ids, "Expected non-empty ids"

    for task_id in ids:
        parts = task_id.split(".")
        # deepplanning.shopping.level{level}.{split}.{case_id}
        assert parts[0] == "deepplanning"
        assert parts[1] == "shopping"
        assert parts[2] == f"level{level}"
        assert parts[3] in ("train", "val")
        assert parts[4].isdigit()


@pytest.mark.parametrize("level", [1, 2, 3])
def test_train_val_split_is_disjoint(level: int):
    train_ids = set(t.get_task_ids("train", level=level, limit=None))
    val_ids = set(t.get_task_ids("val", level=level, limit=None))

    assert train_ids, "Expected some train ids"
    assert val_ids, "Expected some val ids"
    assert train_ids.isdisjoint(val_ids), "Train/val splits should not overlap"

    # Should not exceed total number of queries for that level
    assert len(train_ids) + len(val_ids) <= _num_queries(level)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_get_task_returns_task_with_expected_fields(level: int):
    task_id = t.get_task_ids("train", level=level, limit=1)[0]
    task = t.get_task(task_id)

    assert task.id == task_id
    assert isinstance(task.goal, str) and len(task.goal) > 0

    assert task.misc["domain"] == "shopping"
    assert task.misc["level"] == level
    assert str(task.misc["case_id"]).isdigit()

    case_dir = Path(task.misc["case_dir"])
    tool_schema = Path(task.misc["tool_schema_path"])

    assert case_dir.exists(), f"Expected case_dir to exist: {case_dir}"
    assert tool_schema.exists(), f"Expected tool_schema_path to exist: {tool_schema}"