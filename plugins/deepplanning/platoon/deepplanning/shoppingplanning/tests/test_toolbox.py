import inspect
import json
from pathlib import Path

import pytest

from platoon.deepplanning.shoppingplanning.toolbox import ShoppingPlanningToolBox


def _shoppingplanning_root() -> Path:
    # .../shoppingplanning/tests/test_tools.py -> .../shoppingplanning
    return Path(__file__).resolve().parent.parent


def _schema_path() -> Path:
    return _shoppingplanning_root() / "tools" / "shopping_tool_schema.json"


def _load_tool_schema() -> list[dict]:
    p = _schema_path()
    assert p.exists(), f"Missing tool schema: {p}"
    return json.loads(p.read_text(encoding="utf-8"))


def _schema_tool_specs() -> dict[str, dict]:
    """
    Returns: tool_name -> {"properties": set[str], "required": set[str]}
    """
    specs: dict[str, dict] = {}
    for entry in _load_tool_schema():
        assert entry.get("type") == "function"
        fn = entry["function"]
        name = fn["name"]
        params = fn.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        specs[name] = {
            "properties": set(props.keys()),
            "required": set(required),
        }
    assert specs, "No tools found in schema"
    return specs


@pytest.fixture(scope="session")
def case_dir() -> Path:
    # Use an existing checked-in case directory to allow tool instantiation.
    # Adjust if you prefer another level/case.
    p = _shoppingplanning_root() / "database" / "database_level1" / "case_1"
    assert p.exists(), f"Expected shopping db case dir to exist: {p}"
    return p


@pytest.fixture(scope="session")
def tools(case_dir: Path) -> ShoppingPlanningToolBox:
    return ShoppingPlanningToolBox(case_dir)


def test_schema_and_registry_have_same_tool_names(tools: ShoppingPlanningToolBox):
    schema_names = set(_schema_tool_specs().keys())
    registry_names = set(tools._tools.keys())  # intentional: we want to validate registry coverage

    # Exact match is ideal: schema describes exactly what tools should exist.
    assert registry_names == schema_names, (
        "Mismatch between tool registry and schema.\n"
        f"Only in registry: {sorted(registry_names - schema_names)}\n"
        f"Only in schema: {sorted(schema_names - registry_names)}"
    )


def test_wrapper_exposes_all_schema_tools_as_methods():
    schema_names = set(_schema_tool_specs().keys())

    missing = [name for name in sorted(schema_names) if not hasattr(ShoppingPlanningToolBox, name)]
    assert not missing, f"ShoppingPlanningToolBox is missing wrapper methods: {missing}"

    noncallable = [name for name in sorted(schema_names) if not callable(getattr(ShoppingPlanningToolBox, name))]
    assert not noncallable, f"ShoppingPlanningToolBox has non-callable attributes for tools: {noncallable}"


@pytest.mark.parametrize("tool_name", sorted(_schema_tool_specs().keys()))
def test_wrapper_signature_matches_schema(tool_name: str):
    specs = _schema_tool_specs()[tool_name]
    expected_props: set[str] = specs["properties"]
    expected_required: set[str] = specs["required"]

    fn = getattr(ShoppingPlanningToolBox, tool_name)
    sig = inspect.signature(fn)

    # Exclude `self`
    params = [p for p in sig.parameters.values() if p.name != "self"]

    # Reject **kwargs/*args wrappers: we want explicit signatures
    assert all(
        p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params
    ), f"{tool_name} wrapper should not use *args/**kwargs; make params explicit"

    wrapper_names = {p.name for p in params}
    assert wrapper_names == expected_props, (
        f"{tool_name} wrapper params != schema properties.\n"
        f"Wrapper params: {sorted(wrapper_names)}\n"
        f"Schema props:   {sorted(expected_props)}"
    )

    for p in params:
        if p.name in expected_required:
            assert p.default is inspect._empty, f"{tool_name}.{p.name} is required in schema; wrapper must not default"
        else:
            assert p.default is not inspect._empty, f"{tool_name}.{p.name} is optional in schema; wrapper should default"