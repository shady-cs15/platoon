import ast

from openhands.sdk.conversation import get_agent_final_response

from platoon.openhands.env import OpenHandsEnv
from platoon.utils.openhands_utils import is_finished


def f1_reward_function(predicted_files, true_files):
    pred, true = set(predicted_files), set(true_files)
    tp = len(pred & true)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(true) if true else 0.0
    if not pred and not true:
        return 1.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def reward_function(final_message: str, instance: dict) -> tuple[float, set[str], set[str]]:
    true_files = set(x[0] for x in ast.literal_eval(instance["target"]))
    score = 0.0
    repo_dir = str(instance["repo_dir"])
    if not repo_dir.endswith("/"):
        repo_dir += "/"
    try:
        predicted_files: set[str] = set(
            ast.literal_eval(final_message.split("<file-list>")[1].split("</file-list>")[0])
        )
        relative_predicted_files = set()
        for file_path in predicted_files:
            if file_path.startswith(repo_dir):
                relative_path = file_path[len(repo_dir) :]
            else:
                relative_path = file_path
            relative_predicted_files.add(relative_path)
        score = f1_reward_function(relative_predicted_files, true_files)
    except Exception as e:
        print(f"Error parsing final message: {e}")
        return 0.0, set(), true_files

    return score, relative_predicted_files, true_files


class CodeGreEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        if is_finished(await self.observe()):
            finish_message = get_agent_final_response(self._conversation.state.events)
            print(f"Finish message: {finish_message}")
            reward, predicted_files, true_files = reward_function(finish_message, self.task.misc)
            return reward, {"predicted_files": predicted_files, "true_files": true_files}
        return 0.0, {}
