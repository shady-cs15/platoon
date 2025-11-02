from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder
from platoon.envs.codeact import CodeActObservation


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

    # def build_user_prompt(self, obs: CodeActObservation, **context) -> str:
    #     low: int = obs.task.misc["low"]
    #     high: int = obs.task.misc["high"]
    #     return (
    #         f"Guess the correct number between {low} and {high}. "
    #         "Think in <think> </think> and then call <python>guess(<number>)</python>."
    #     )


class NumberSearchAgent(CodeActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(prompt_builder=NumberSearchPromptBuilder(), *args, **kwargs)


