
from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder
from platoon.envs.codeact import CodeActObservation


# TODO: We need to fix action space description in prompt builder.
class NumberSearchPromptBuilder(CodeActPromptBuilder):
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return """Solve the problem step by step. Write your thoughts in <think> </think> tags and the final action in <python> </python> tags.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<think>
thought process here
</think>
<python>
guess(42)
</python>
"""

class NumberSearchAgent(CodeActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(prompt_builder=NumberSearchPromptBuilder(), *args, **kwargs)
