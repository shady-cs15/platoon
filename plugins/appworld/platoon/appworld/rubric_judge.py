from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import rubric.core.checklist as rubric_checklist_module
import rubric.utils.llm_tools as rubric_llm_tools


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
