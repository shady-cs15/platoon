#%%
import os
import sys
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor

from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from appworld import load_task_ids
from datasets import Dataset
from platoon.agents.appworld.areal_workflow import AppWorldArealWorkflow, AppWorldArealRecursiveWorkflow
from dataclasses import field, dataclass
from areal.api.cli_args import InferenceEngineConfig
from areal.utils.data import concat_padded_tensors
from areal.api.engine_api import TrainEngine
from areal.utils.functional import gather_logprobs_entropy
from areal.api.cli_args import PPOActorConfig
from areal.utils import stats_tracker


def reinforce_loss_fn(logits, data, temperature: float = 1.0):
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"].bool()
    rewards = data["token_rewards"]

    logprobs, entropy = gather_logprobs_entropy(
        logits, torch.roll(input_ids, shifts=-1, dims=-1), temperature
    )
    entropy = entropy.detach()
    # Align mask to next-token prediction positions
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)
    loss = -logprobs * rewards
    loss = torch.where(loss_mask, loss, 0.0)
    

    stats_tracker.denominator(
        n_valid_tokens=loss_mask.bool(),
    )
    stats_tracker.stat(
        entropy=entropy.float(),
        actor_loss=loss.float(),
        token_rewards=rewards,
        logprobs=logprobs.detach(),
        denominator="n_valid_tokens",
    )

    return loss.sum() / loss_mask.count_nonzero()

class ReinforceActor:
    def __init__(self, engine: TrainEngine):
        self.engine = engine

    def train_reinforce(self, data: Dict[str, Any]):
        # Enable gradient checkpointing
        all_stats = []
        self.engine.train()
        train_stat = self.engine.train_batch(
            data,
            loss_fn=lambda logits, data: reinforce_loss_fn(logits, data, self.engine.config.temperature),
            loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
        )
        
        stats_tracker.scalar(**train_stat)
        all_stats.append(
                stats_tracker.export(reduce_group=self.engine.data_parallel_group)
            )
        return all_stats

class FSDPReinforceActor(FSDPPPOActor):
    def __init__(self, config: PPOActorConfig):
        super().__init__(config=config)
        self.actor = ReinforceActor(self)

    def train_reinforce(self, *args, **kwargs):
        return self.actor.train_reinforce(*args, **kwargs)

@dataclass
class VariableBatchInferenceEngineConfig(InferenceEngineConfig):
    shuffle_cross_task: bool = field(default=True)
    ensure_batch_divisible_by: int = field(default=1)

@dataclass
class AppWorldReinforcePlusPlusConfig(GRPOConfig):
    workflow_config: dict = field(default_factory=dict)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)

#%%


os.environ["APPWORLD_ROOT"] = "/mnt/efs/platoon/src/platoon/envs/appworld"

args = ["--config", "/mnt/efs/platoon/src/platoon/train/appworld_custom_reinforce.yaml", "experiment_name=debugrollout", "trial_name=debugrollout"] #sys.argv[1:]
config, _ = load_expr_config(args, AppWorldReinforcePlusPlusConfig)
config: AppWorldReinforcePlusPlusConfig

tokenizer = load_hf_tokenizer(config.tokenizer_path)

allocation_mode = AllocationMode.from_str(config.allocation_mode)
parallel_strategy = allocation_mode.train
assert parallel_strategy is not None



train_ids = load_task_ids(dataset_name="train")
train_index_remainder = len(train_ids) % 4
if train_index_remainder != 0:
    train_ids = train_ids[:-train_index_remainder]
    #train_ids = train_ids[:4] # debugging
train_ids = [x for i, x in enumerate(train_ids) if i % 4 == 0]
train_dataset = Dataset.from_list([{ "task_id": x } for x in train_ids])

# Create dataset and dataloaderss
train_dataloader = StatefulDataLoader(
    train_dataset,
    batch_size=config.train_dataset.batch_size // 4,
    shuffle=config.train_dataset.shuffle,
    num_workers=config.train_dataset.num_workers,
    collate_fn=lambda x: x,
    drop_last=config.train_dataset.drop_last,
)

#%%

# Initialize inference engine
rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size) #1)
eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
# NOTE: eval does not have any offpolicyness control
eval_rollout.config.max_head_offpolicyness = int(1e12)
eval_rollout.initialize()

# Create rollout workflow
if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
    config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
workflow = AppWorldArealRecursiveWorkflow(
    config=config.workflow_config
)

data_generator = cycle_dataloader(train_dataloader)

#%%

batch = rollout.rollout_batch(
    next(data_generator),
    workflow=workflow,
    should_accept=lambda sample: True,
)

# Save generated data for later use
torch.save(batch, "batch_data.pt")

rollout.destroy()

