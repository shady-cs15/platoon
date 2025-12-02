from platoon.train.rl import PlatoonStepWiseRLTrainer, PlatoonStepWiseRLTrainerConfig
from platoon.train.workflows.step_wise import StepWiseArealWorkflow
from areal.api.cli_args import load_expr_config
from platoon.codegrep.tasks import get_task_ids, get_task
from platoon.codegrep.rollout import run_rollout
from datasets import Dataset
import sys

def main(args):
    config, _ = load_expr_config(args, PlatoonStepWiseRLTrainerConfig)
    config: PlatoonStepWiseRLTrainerConfig = config
    
    # TODO: Design a TaskLoader protocol and add configs + factory for this.
    train_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("train")])
    val_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("val")])
    
    
    with PlatoonStepWiseRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
    
        proxy_server = trainer.proxy_server
        workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device)
        eval_workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device)
        
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
