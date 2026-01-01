from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class NumberSearchPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for NumberSearch agent.
    
    Inherits prompt_mode support from CodeActPromptBuilder:
    - "sequence_extension" (default): Multi-turn conversation for sequence extension
    - "no_sequence_extension": Legacy single-user-message format
    """
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return """Solve the problem step by step. Write your thoughts in <thought> </thought> tags and the final action in <python> </python> tags.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<thought>
thought process here
</thought>
<python>
guess(42)
</python>
"""


class NumberSearchAgent(CodeActAgent):
    """Agent for NumberSearch environment.
    
    Args:
        prompt_mode: The prompt format to use ("sequence_extension" or "no_sequence_extension")
    """
    
    def __init__(self, prompt_mode: PromptMode = "sequence_extension", **kwargs):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = NumberSearchPromptBuilder(prompt_mode=prompt_mode)
        super().__init__(prompt_mode=prompt_mode, **kwargs)
