from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.envs.base import Task
from platoon.agents.actions.common import finish
from platoon.envs.countdown.eval import compute_score

class CountDownEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task, IPythonCodeExecutor(task, actions=(finish,)))
        
    async def evaluate(self) -> tuple[float, dict]:
        return compute_score(self._state.history[-1].output, self._task.misc)
