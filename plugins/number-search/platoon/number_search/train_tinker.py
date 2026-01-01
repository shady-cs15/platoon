"""Number Search training script using Tinker backend.

Usage:
    python -m platoon.number_search.train_tinker --config number_search_tinker.yaml
    python -m platoon.number_search.train_tinker --config number_search_tinker.yaml --train.batch_size 64
"""

import asyncio
import sys
from pathlib import Path

from datasets import Dataset

from platoon.number_search.tasks import get_task_ids, get_task
from platoon.number_search.rollout import run_rollout
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.workflows import GroupCenteredRolloutWorkflow
from platoon.utils.config import load_config


async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "number_search_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )
    
    # Create datasets
    train_dataset = Dataset.from_list([
        {"task_id": x} for x in get_task_ids("train", 1000)
    ])
    eval_dataset = Dataset.from_list([
        {"task_id": x} for x in get_task_ids("val", 100)
    ])
    
    # Create trainer and run with context manager for proper cleanup
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    async with trainer:
        # Create workflows
        train_workflow = GroupCenteredRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            stats_scope="train",
            filter_errors=False,
        )
        
        eval_workflow = GroupCenteredRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            stats_scope="eval",
            filter_errors=False,
        )
        
        # Run training
        await trainer.train(
            train_workflow=train_workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))

