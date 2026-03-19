import sys

from copy import deepcopy
from areal.api.cli_args import load_expr_config
from appworld import load_task_ids
from datasets import Dataset
from dataclasses import dataclass

from platoon.appworld.rollout import run_rollout, run_recursive_rollout, run_depth_aware_rollout
from platoon.appworld.tasks import get_task
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow


@dataclass
class AppWorldArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = False
    depth_aware: bool = False
    delegation_bonus_unconditional_weight: float = 1.0  # 1.0 = fully unconditional (legacy), 0.0 = fully gated on success


def make_reward_processor(unconditional_weight: float = 1.0):
    """Create a reward processor with configurable delegation bonus gating."""
    def reward_processor(traj: dict) -> tuple[float, dict]:
        rewards_dict = {}
        for step in traj["steps"]:
            reward_misc = step.get("misc", {}).get("reward_misc", {})
            for reward_key, reward_value in reward_misc.items():
                if reward_key.startswith("reward/"):
                    if reward_key not in rewards_dict:
                        rewards_dict[reward_key] = 0.0
                    rewards_dict[reward_key] += reward_value
        success_reward = rewards_dict.get("reward/success", 0.0)
        score = success_reward

        launched = rewards_dict.get("reward/subagent_launched", 0.0)
        if launched > 0:
            subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
            score += 0.4 * subagent_success_rate
        return score, rewards_dict
    return reward_processor

def main(args):
    config, _ = load_expr_config(args, AppWorldArealTrainerConfig)
    config: AppWorldArealTrainerConfig = config
    
    if config.depth_aware:
        rollout_fn = run_depth_aware_rollout
    elif config.recursive:
        rollout_fn = run_recursive_rollout
    else:
        rollout_fn = run_rollout

    reward_processor = make_reward_processor(config.delegation_bonus_unconditional_weight)

    train_dataset = Dataset.from_list([{"task_id": x} for x in load_task_ids(dataset_name="train")])
    val_dataset = Dataset.from_list([{"task_id": x} for x in load_task_ids(dataset_name="dev")])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        proxy_server = trainer.proxy_server
        eval_proxy_server = trainer.eval_proxy_server
        workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            config.workflow_config,
            proxy_server,
            "train_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
            filter_errors=False,
        )

        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1

        eval_workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            eval_workflow_config,
            eval_proxy_server,
            "eval_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
            filter_errors=False,
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
