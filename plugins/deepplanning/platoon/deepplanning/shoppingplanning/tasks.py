import hashlib
import json
from pathlib import Path
from typing import Literal

from platoon.envs.base import Task

_BASE = Path(__file__).resolve().parent 
print("_BASE:", _BASE)

_QUERIES = {}  # cache: level -> list[dict]

def _load_queries(level: int) -> list[dict]:
    if level not in _QUERIES:
        p = _BASE / "data" / f"level_{level}_query_meta.json"
        _QUERIES[level] = json.loads(p.read_text(encoding="utf-8"))
    return _QUERIES[level]

def _is_val(case_id: str, *, seed: int = 42, p_val: float = 0.1) -> bool:
    h = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    # use first 8 hex chars as uint32
    x = int(h[:8], 16) / 0xFFFFFFFF
    return x < p_val

def get_task_ids(
    split: Literal["train", "val"],
    *,
    level: int = 1,
    limit: int | None = None,
    seed: int = 42,
    p_val: float = 0.1,
) -> list[str]:
    rows = _load_queries(level)
    ids = []
    for r in rows:
        cid = str(r["id"])
        is_val = _is_val(cid, seed=seed, p_val=p_val)
        if (split == "val") != is_val:
            continue
        ids.append(f"deepplanning.shopping.level{level}.{split}.{cid}")
        if limit is not None and len(ids) >= limit:
            break
    return ids

def get_task(task_id: str) -> Task:
    # deepplanning.shopping.level{level}.{split}.{case_id}
    parts = task_id.split(".")
    level_str = parts[2]  # "level1"
    level = int(level_str.removeprefix("level"))
    case_id = parts[4]

    rows = _load_queries(level)
    # index by id (small, linear scan OK to start)
    row = next(r for r in rows if str(r["id"]) == str(case_id))
    query = row["query"]

    case_dir = _BASE / "database" / f"database_level{level}" / f"case_{case_id}"
    tool_schema = _BASE / "tools" / "shopping_tool_schema.json"

    return Task(
        goal=query,
        id=task_id,
        max_steps=None,  # later: set based on max_llm_calls or env step budget
        misc={
            "domain": "shopping",
            "level": level,
            "case_id": case_id,
            "case_dir": str(case_dir),
            "tool_schema_path": str(tool_schema),
        },
    )