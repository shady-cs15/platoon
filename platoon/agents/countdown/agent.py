from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder
from platoon.envs.codeact import CodeActObservation

class CountDownPromptBuilder(CodeActPromptBuilder):
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return """Solve the problem step by step. Write your thoughts in <think> </think> tags and the solution in <python> </python> tags.
The answer message should be a formula consisting of arithmetic operations (+, -, *, /) that results in the target number.

Write the final answer in <python> </python> using the finish(message: str) function.
Otherwise, the grader will not be able to parse your answer.

Example:
<think>
thought process here
</think>
<python>
finish("(1 + 2) * 2 * 4")
</python>
"""

    def build_user_prompt(self, obs: CodeActObservation, **context) -> str:
        numbers: list[int] = obs.task.misc["numbers"]
        target: int = obs.task.misc["target"]
        return f"""Now, solve the following problem. Using the numbers {numbers}, create an equation that equals {target}. 
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. 
Show your work in <think> </think> tags and return the final answer in <python> </python> tags, for example <python>finish('(1 + 2) * 2 * 4')</python>.
"""

class CountDownAgent(CodeActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(prompt_builder=CountDownPromptBuilder(), *args, **kwargs)
