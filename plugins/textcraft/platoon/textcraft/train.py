import sys
from datasets import Dataset
from platoon.train.rl import PlatoonStepWiseRLTrainer, PlatoonStepWiseRLTrainerConfig
from areal.api.cli_args import load_expr_config
from platoon.textcraft.tasks import get_task_ids, get_task
#from platoon.textcraft.train_workflow import TextCraftArealWorkflow
from platoon.textcraft.rollout import run_rollout, run_recursive_rollout
from platoon.train.workflows.step_wise import StepWiseArealWorkflow


def main(args):
    config, _ = load_expr_config(args, PlatoonStepWiseRLTrainerConfig)
    config: PlatoonStepWiseRLTrainerConfig = config
    
    # TODO: Design a TaskLoader protocol and add configs + factory for this.
    train_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("train", 1000)])
    val_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("val", 100)])
    
    
    with PlatoonStepWiseRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
    
        proxy_server = trainer.proxy_server
        #workflow = TextCraftArealWorkflow(config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device)
        #eval_workflow = TextCraftArealWorkflow(config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device)
        workflow = StepWiseArealWorkflow(run_recursive_rollout, get_task, config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device, filter_errors=True)
        eval_workflow = StepWiseArealWorkflow(run_recursive_rollout, get_task, config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device)
        
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
