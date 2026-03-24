import sys

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field

from areal.api.cli_args import load_expr_config
from datasets import Dataset

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
    root_reward_propagation: bool = False
    train_split: str = "train"
    eval_split: str = "dev"
    task_filter: str = "none"
    curriculum_filters: list[str] = field(default_factory=list)
    curriculum_steps: list[int] = field(default_factory=list)


def make_reward_processor(
    gated_weight: float = 0.3,
    unconditional_weight: float = 0.1,
):
    """Create a reward processor with separate gated and ungated delegation bonus terms.

    Args:
        gated_weight: Delegation bonus multiplied by root success (only rewarded when root succeeds).
        unconditional_weight: Delegation bonus regardless of root success.
    """
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

def _make_curriculum_callback(
    curriculum_filters: list[str],
    curriculum_steps: list[int],
    train_split: str,
) -> Callable[["PlatoonArealRLTrainer", int], None]:
    """Create a callback that switches task filters at curriculum step boundaries.

    Args:
        curriculum_filters: Ordered list of task filters for each phase.
        curriculum_steps: Cumulative step count at which each phase ends.
        train_split: Train split name passed to resolve_train_task_ids.

    Returns:
        Callback compatible with PlatoonArealRLTrainer.train(on_step_callback=...).
    """
    boundaries: list[tuple[int, str]] = []
    cumulative = 0
    for filt, steps in zip(curriculum_filters, curriculum_steps):
        cumulative += steps
        boundaries.append((cumulative, filt))

    state = {"current_filter": curriculum_filters[0]}

    def callback(trainer: "PlatoonArealRLTrainer", global_step: int) -> None:
        target_filter = boundaries[-1][1]
        for boundary_step, filt in boundaries:
            if global_step < boundary_step:
                target_filter = filt
                break

        if target_filter != state["current_filter"]:
            new_task_ids = resolve_train_task_ids(train_split, target_filter)
            new_dataset = Dataset.from_list([{"task_id": x} for x in new_task_ids])
            trainer.replace_train_dataset(new_dataset)
            state["current_filter"] = target_filter
            print(
                f"Curriculum: switched to task_filter='{target_filter}' at step {global_step}"
                f" ({len(new_task_ids)} tasks)"
            )

    return callback


def main(args):
    config, _ = load_expr_config(args, AppWorldArealTrainerConfig)
    config: AppWorldArealTrainerConfig = config

    if config.depth_aware:
        rollout_fn = run_depth_aware_rollout
    elif config.recursive:
        rollout_fn = run_recursive_rollout
    else:
        rollout_fn = run_rollout

    # When root_reward_propagation is enabled, skip subtask LLM judges and
    # propagate root reward/success into child trajectories after rollout.
    # The reward_processor then naturally gives leaf agents the base success
    # (no delegation bonus) and intermediate agents the full reward (with bonus).
    if config.root_reward_propagation:
        config.workflow_config.rollout_config.propagate_root_success = True

    reward_processor = make_reward_processor(
        gated_weight=config.delegation_bonus_gated_weight,
        unconditional_weight=config.delegation_bonus_unconditional_weight,
    )

    # Build curriculum callback if configured
    on_step_callback = None
    if config.curriculum_filters and config.curriculum_steps:
        if len(config.curriculum_filters) != len(config.curriculum_steps):
            raise ValueError(
                f"curriculum_filters ({len(config.curriculum_filters)}) and "
                f"curriculum_steps ({len(config.curriculum_steps)}) must have the same length"
            )
        on_step_callback = _make_curriculum_callback(
            config.curriculum_filters,
            config.curriculum_steps,
            config.train_split,
        )
        # Use first phase's filter for the initial dataset
        config.task_filter = config.curriculum_filters[0]
        # Set max_train_steps so the loop runs long enough for all phases
        config.max_train_steps = sum(config.curriculum_steps)

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
            **({"curriculum": list(zip(config.curriculum_filters, config.curriculum_steps))}
               if config.curriculum_filters else {}),
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
            on_step_callback=on_step_callback,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
