from __future__ import annotations

from platoon.agents.codeact import CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation

import platoon.deepplanning.shoppingplanning.agent_lib.prompts as prompt_lib


class ShoppingPlanningPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for DeepPlanning ShoppingPlanning using CodeAct-style Python actions.

    - Uses the benchmark's system prompt for the task level (1/2/3).
    - Instructs the agent to act by calling Python functions (your toolbox actions)
      and to end by calling finish(...).

    Snippet to print full prompt for debugging:
    ===========================================
    cd /root/platoon
    PYTHONPATH=/root/platoon:/root/platoon/plugins/deepplanning uv run python - <<'PY'
    from platoon.envs.base import Task
    from platoon.envs.codeact.types import CodeActObservation
    from platoon.deepplanning.shoppingplanning.prompt_builder import ShoppingPlanningPromptBuilder

    pb = ShoppingPlanningPromptBuilder(prompt_mode="sequence_extension", include_reasoning=True)

    task = Task(goal="(dummy goal)", id="t0", misc={"level": 1})
    obs = CodeActObservation(
        task=task,
        action_space="(put your action_space string here, or leave blank)",
        history=[],
    )

    msgs = pb.build_messages(obs)
    print("=== SYSTEM PROMPT ===")
    print(msgs[0]["content"])
    print("\n=== INITIAL USER PROMPT ===")
    print(msgs[1]["content"])
    PY
    """

    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)

    def _get_level(self, obs: CodeActObservation) -> int:
        # tasks.py sets task.misc["level"]
        try:
            level = int(getattr(obs.task, "misc", {}).get("level", 1))
        except Exception:
            level = 1
        return level if level in (1, 2, 3) else 1

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        level = self._get_level(obs)

        base = getattr(prompt_lib, f"SYSTEM_PROMPT_level{level}", None)
        if base is None:
            base = prompt_lib.SYSTEM_PROMPT_level1

        # IMPORTANT: In CodeAct, the "tools" are Python functions in the sandbox namespace.
        # We rely on env.reset() populating obs.action_space to show the available functions.
        codeact_instructions = (
            "You will interact with a shopping environment by writing Python code.\n"
            "Your code will be executed, and you will see its output.\n\n"
            "Rules:\n"
            "- Use ONLY the functions provided in the Action Space to search/filter/sort products and modify the cart.\n"
            "- Before finishing, ALWAYS call get_cart_info() to verify the cart is correct.\n"
            "- When you are done, call finish('...') with a short explanation.\n"
        )

        if include_reasoning:
            format_instructions = (
                "For each step, output:\n"
                "- <thought> 1–3 short sentences about what you will do next </thought>\n"
                "- <python> valid Python code calling the available functions </python>\n"
            )
        else:
            format_instructions = (
                "For each step, output:\n"
                "- <python> valid Python code calling the available functions </python>\n"
            )

        return base.strip() + "\n\n" + codeact_instructions + "\n" + format_instructions