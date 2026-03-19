from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import rubric.core.checklist as rubric_checklist_module
import rubric.utils.llm_tools as rubric_llm_tools
from rubric.utils.llm_client import create_llm_client


_RUBRIC_OVERRIDE_LOCK = asyncio.Lock()


@asynccontextmanager
async def configured_rubric_judge(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> AsyncIterator[None]:
    if not any([model, base_url, api_key, api_key_env]):
        yield
        return

    resolved_api_key = api_key
    if resolved_api_key is None and api_key_env:
        resolved_api_key = os.getenv(api_key_env)
        if resolved_api_key is None:
            raise ValueError(f"Rubric judge API key env var {api_key_env!r} is not set.")

    async with _RUBRIC_OVERRIDE_LOCK:
        previous_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "RUBRIC_DEFAULT_LLM": os.environ.get("RUBRIC_DEFAULT_LLM"),
        }
        previous_checklist_model = rubric_checklist_module.LLM_MODEL_NAME
        previous_tool_model = rubric_llm_tools.LLM_MODEL_NAME

        try:
            if model is not None:
                rubric_checklist_module.LLM_MODEL_NAME = model
                rubric_llm_tools.LLM_MODEL_NAME = model
                os.environ["RUBRIC_DEFAULT_LLM"] = model
            if base_url is not None:
                os.environ["OPENAI_BASE_URL"] = base_url
            if resolved_api_key is not None:
                os.environ["OPENAI_API_KEY"] = resolved_api_key
            yield
        finally:
            rubric_checklist_module.LLM_MODEL_NAME = previous_checklist_model
            rubric_llm_tools.LLM_MODEL_NAME = previous_tool_model
            for env_name, previous_value in previous_env.items():
                if previous_value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous_value


async def abinary_judge_subtask(
    *,
    goal: str,
    action_history: str,
    final_message: str | None,
    err_message: str | None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> tuple[float, str]:
    resolved_api_key = api_key
    if resolved_api_key is None and api_key_env:
        resolved_api_key = os.getenv(api_key_env)
        if resolved_api_key is None:
            raise ValueError(f"Rubric judge API key env var {api_key_env!r} is not set.")

    llm_client = create_llm_client(api_key=resolved_api_key, model=model, base_url=base_url)
    system_prompt = (
        "You are a strict evaluator for autonomous agent subtasks. "
        "Decide whether the agent successfully completed the assigned subtask. "
        "Return JSON only with keys: success and reasoning. "
        "success must be exactly 1.0 for success or 0.0 for failure. "
        "Mark success only if the assigned subtask was fully completed based on the evidence. "
        "If the output is partial, uncertain, inconsistent, incomplete, or likely missed items/pages/entities, return 0.0."
    )
    user_prompt = (
        f"# Subtask Goal\n{goal}\n\n"
        f"# Agent Action History\n{action_history}\n\n"
        f"# Final Message\n{final_message}\n\n"
        f"# Error Message\n{err_message}\n\n"
        "Evaluate whether the subtask was fully completed. "
        "Return exactly one JSON object like "
        '{"success": 0.0, "reasoning": "short explanation"} '
        'or {"success": 1.0, "reasoning": "short explanation"}.'
    )
    response = await llm_client.asystem_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.0,
    )
    parsed = _parse_binary_judge_response(response)
    return parsed["success"], parsed["reasoning"]


def _parse_binary_judge_response(response: str) -> dict[str, str | float]:
    try:
        json_start = response.find("{")
        json_end = response.rfind("}")
        if json_start == -1 or json_end == -1 or json_end < json_start:
            raise ValueError("No JSON object found in judge response.")
        parsed = json.loads(response[json_start : json_end + 1])
    except Exception as exc:
        raise ValueError(f"Could not parse binary judge response: {response}") from exc

    success = parsed.get("success")
    reasoning = parsed.get("reasoning", "")
    if success not in (0, 0.0, 1, 1.0):
        raise ValueError(f"Binary judge returned invalid success value: {success!r}")
    return {"success": float(success), "reasoning": str(reasoning)}
