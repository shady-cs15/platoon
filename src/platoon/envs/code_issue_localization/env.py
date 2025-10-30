from platoon.envs.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished
import ast

def f1_reward_function(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

def reward_function(final_message, instance):
    predicted_files = set(ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0]))
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    return f1_reward_function(predicted_files, true_files), predicted_files, true_files

class CodeIssueLocalizationEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        if is_finished(await self.observe()):
            finish_message = self._conversation.agent_final_response()
            reward, predicted_files, true_files = reward_function(finish_message, self.task['misc']['instance'])
            return reward, {"predicted_files": predicted_files, "true_files": true_files}
        return 0.0, {}
