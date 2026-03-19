import sys

from copy import deepcopy
from areal.api.cli_args import load_expr_config
from datasets import Dataset
from dataclasses import dataclass

from platoon.appworld.rollout import run_rollout, run_recursive_rollout, run_depth_aware_rollout
from platoon.appworld.task_sets import resolve_eval_task_ids, resolve_train_task_ids, summarize_task_selection
from platoon.appworld.tasks import get_task
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow


@dataclass
class AppWorldArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = False
    depth_aware: bool = False
    delegation_bonus_gated_weight: float = 0.3
    delegation_bonus_unconditional_weight: float = 0.1
    train_split: str = "train"
    eval_split: str = "dev"
    task_filter: str = "none"


def make_reward_processor(
    gated_weight: float = 0.3,
    unconditional_weight: float = 0.1,
):
    """Create a reward processor with separate gated and ungated delegation bonus terms."""
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
            score += gated_weight * success_reward * subagent_success_rate
            score += unconditional_weight * subagent_success_rate
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

    reward_processor = make_reward_processor(
        gated_weight=config.delegation_bonus_gated_weight,
        unconditional_weight=config.delegation_bonus_unconditional_weight,
    )

    selection_summary = summarize_task_selection()
    train_task_ids = resolve_train_task_ids(config.train_split, config.task_filter)
    eval_task_ids = resolve_eval_task_ids(config.eval_split)

    print(
        "AppWorld task selection:",
        {
            **selection_summary,
            "train_split": config.train_split,
            "task_filter": config.task_filter,
            "train_selected": len(train_task_ids),
            "eval_split": config.eval_split,
            "eval_selected": len(eval_task_ids),
        },
    )

    train_dataset = Dataset.from_list([{"task_id": x} for x in train_task_ids])
    val_dataset = Dataset.from_list([{"task_id": x} for x in eval_task_ids])

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
