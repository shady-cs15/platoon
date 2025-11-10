from dataclasses import dataclass, field
from datasets import Dataset
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.alloc_mode import AllocationMode
from areal.utils import seeding, stats_tracker  
from areal.api.workflow_api import RolloutWorkflow
from platoon.utils.train import VariableBatchInferenceEngineConfig, set_expandable_segments, post_process_and_redistribute_tensor_container
from platoon.train.aent.aent_args import AEntPPOActorConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.api.cli_args import GRPOConfig
from areal.platforms import current_platform
from areal.utils.data import (
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
import os
import datetime
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import torch
from areal.engine.ppo.actor import FSDPPPOActor
from platoon.train.aent.actor import FSDPAEntPPOActor
from copy import deepcopy


@dataclass
class PlatoonStepWiseRLTrainerConfig(GRPOConfig):
    workflow_config: dict = field(default_factory=dict) # TODO: Harden this.
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: AEntPPOActorConfig = field(default_factory=AEntPPOActorConfig)


# TODO: Refactor for latest AReaL version and support megatron training backend.
class PlatoonStepWiseRLTrainer:
    
    def __init__(
        self,
        config: PlatoonStepWiseRLTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast | None = None
    ):
        set_expandable_segments(True)
        
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        
        rank = int(os.environ.get("RANK", 0))
        
        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(config.tokenizer_path)
            
        if self.tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.eos_token_id)
        
        seeding.set_random_seed(config.seed, key=f"trainer{rank}")
        allocation_mode = AllocationMode.from_str(config.allocation_mode)
        parallel_strategy = allocation_mode.train
        assert parallel_strategy is not None
        
        
        self.actor = FSDPAEntPPOActor(config=config.actor)
        self.actor.create_process_group(parallel_strategy=parallel_strategy)
        
        # TODO: should do this for other groups as well? Make configurable
        dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=7200), self.actor.data_parallel_group)
        
        
        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=config.train_dataset.batch_size // self.actor.data_parallel_world_size,
            sampler=DistributedSampler(
                train_dataset, 
                num_replicas=self.actor.data_parallel_world_size, 
                rank=self.actor.data_parallel_rank
                shuffle=config.train_dataset.shuffle,
                drop_last=config.train_dataset.drop_last,
            ),
            num_workers=config.train_dataset.num_workers,
            collate_fn=lambda x: x,
        )
        
        self.val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=config.val_dataset.batch_size // self.actor.data_parallel_world_size,
            sampler=DistributedSampler(
                val_dataset, 
                num_replicas=self.actor.data_parallel_world_size, 
                rank=self.actor.data_parallel_rank,
                shuffle=config.val_dataset.shuffle,
                drop_last=config.val_dataset.drop_last,
            ),
            num_workers=config.val_dataset.num_workers,
            collate_fn=lambda x: x,
        )
        
        self.ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )
    
        
        self.rollout = RemoteSGLangEngine(config.rollout)
        self.rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
        self.eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
        # NOTE: eval does not have any offpolicyness control
        self.eval_rollout.config.max_head_offpolicyness = int(1e12)
        self.eval_rollout.initialize()
        
        self.actor.initialize(None, self.ft_spec)
        
        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            self.ref = FSDPPPOActor(config=config.ref)
            self.ref.create_process_group(parallel_strategy=parallel_strategy)
            self.ref.initialize(None, self.ft_spec)
        
        # NOTE: Weight update meta only requires address and free port of rank 0,
        # but `WeightUpdateMeta.from_fsdp_xccl` has to be executed on all ranks
        # due to `engine.get_param_specs()`.
        # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
        weight_update_meta = [
            WeightUpdateMeta.from_fsdp_xccl(
                AllocationMode.from_str(config.allocation_mode), self.actor
            )
        ]
        dist.broadcast_object_list(weight_update_meta, src=0)
        self.weight_update_meta = weight_update_meta[0]
        
         # Run training.
        self.saver = Saver(config.saver, self.ft_spec)
        self.stats_logger = StatsLogger(config, self.ft_spec)
        self.evaluator = Evaluator(config.evaluator, self.ft_spec)

        self.recover_handler = RecoverHandler(config.recover, self.ft_spec)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=weight_update_meta,
        )
        

            
    def train(
        self,
        workflow: RolloutWorkflow,
        eval_workflow: RolloutWorkflow
    ):
        config = self.config
        start_step = (
            self.recover_info.last_step_info.next().global_step
            if self.recover_info is not None
            else 0
        )
        
        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch
        
        data_generator = cycle_dataloader(self.train_dataloader)
        for global_step in range(start_step, max_steps):
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )
            
            with stats_tracker.record_timing("rollout"):
                batch = None
                if self.actor.is_data_parallel_head():
                    if config.async_training:
                        batch = self.rollout.prepare_batch(
                            self.train_dataloader,
                            workflow=workflow,
                            should_accept=lambda sample: True,
                        )
                    else:
                        batch = self.rollout.rollout_batch(
                            next(data_generator),
                            workflow=workflow,
                            should_accept=lambda sample: True,
                        )
                        
                    batch = tensor_container_to(batch, self.actor.device)
                
                    reward_mask = torch.ones_like(batch['task_reward'], dtype=torch.bool)
                    output_token_mask = torch.ones_like(batch['num_output_tokens'], dtype=torch.bool)
                    input_token_mask = torch.ones_like(batch['num_input_tokens'], dtype=torch.bool)
                    num_steps_mask = torch.ones_like(batch['num_steps'], dtype=torch.bool)
                    
                    stats_tracker.get('reinforce_actor').denominator(task_reward_mask=reward_mask, num_output_tokens_mask=output_token_mask, num_input_tokens_mask=input_token_mask, num_steps_mask=num_steps_mask)
                    stats_tracker.get('reinforce_actor').stat(task_reward=batch['task_reward'], denominator="task_reward_mask")
                    stats_tracker.get('reinforce_actor').stat(num_output_tokens=batch['num_output_tokens'], denominator="num_output_tokens_mask")
                    stats_tracker.get('reinforce_actor').stat(num_input_tokens=batch['num_input_tokens'], denominator="num_input_tokens_mask")
                    stats_tracker.get('reinforce_actor').stat(num_steps=batch['num_steps'], denominator="num_steps_mask")
                    
                    torch.cuda.empty_cache()
                    
                    if config.rollout.shuffle_cross_task or config.rollout.ensure_batch_divisible_by > 1:
                        batch = post_process_and_redistribute_tensor_container(batch, shuffle=config.rollout.shuffle_cross_task, ensure_divisible_by=config.rollout.ensure_batch_divisible_by, group=self.actor.data_parallel_group)
                
                batch = broadcast_tensor_container(
                    batch,
                    src_rank=self.actor.current_data_parallel_head(),
                    group=self.actor.context_and_model_parallel_group,
                )
                
            torch.cuda.empty_cache()
            # Create barrier to synchronize all rollout processes.
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
                with stats_tracker.record_timing("recompute_logp"):
                    logp = self.actor.compute_logp(batch)
                    # Ensure prox_logp token width matches attention_mask for proper microbatch splitting
                    attn_len = batch["attention_mask"].shape[1]
                    if torch.is_tensor(logp) and logp.ndim == 2 and logp.shape[1] != attn_len:
                        if logp.shape[1] < attn_len:
                            pad_width = attn_len - logp.shape[1]
                            logp = torch.nn.functional.pad(logp, (0, pad_width), value=0.0)
                        else:
                            logp = logp[:, :attn_len]
                    batch["prox_logp"] = logp
                    log_gpu_stats("recompute logp")

            if self.ref is not None:
                with stats_tracker.record_timing("ref_logp"):
                    ref_logp = self.ref.compute_logp(batch)
                    # Align ref_logp token width with attention_mask as well
                    attn_len = batch["attention_mask"].shape[1]
                    if torch.is_tensor(ref_logp) and ref_logp.ndim == 2 and ref_logp.shape[1] != attn_len:
                        if ref_logp.shape[1] < attn_len:
                            pad_width = attn_len - ref_logp.shape[1]
                            ref_logp = torch.nn.functional.pad(ref_logp, (0, pad_width), value=0.0)
                        else:
                            ref_logp = ref_logp[:, :attn_len]
                    batch["ref_logp"] = ref_logp
                    log_gpu_stats("ref logp")

            with stats_tracker.record_timing("compute_advantage"):
                self.actor.compute_advantages(batch)
                log_gpu_stats("compute advantages")

            with (
                stats_tracker.record_timing("train_step"),
                stats_tracker.scope("grpo_actor"),
            ):
                stats = self.actor.aent_ppo_update(batch, global_step)
                self.actor.step_lr_scheduler()
                log_gpu_stats("ppo update")

            # pause inference for updating weights, save, and evaluation
            self.rollout.pause()

            with stats_tracker.record_timing("update_weights"):
                if dist.get_rank() == 0:
                    future = self.rollout.update_weights(self.weight_update_meta)
                self.actor.upload_weights(self.weight_update_meta)
                if dist.get_rank() == 0:
                    future.result()
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()

                self.actor.set_version(global_step + 1)
                self.rollout.set_version(global_step + 1)
                self.eval_rollout.set_version(global_step + 1)

            with stats_tracker.record_timing("save"):
                self.saver.save(self.actor, epoch, step, global_step, tokenizer=self.tokenizer)

            with stats_tracker.record_timing("checkpoint_for_recover"):
                self.recover_handler.dump(
                    self.actor,
                    step_info,
                    self.saver,
                    self.evaluator,
                    self.stats_logger,
                    self.train_dataloader,
                    tokenizer=self.tokenizer,
                )

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            with stats_tracker.record_timing("eval"):

                def evaluate_fn():
                    if self.actor.is_data_parallel_head():
                        print(f"Evaluating...")
                        cnt = 0
                        for data in self.val_dataloader:
                            for item in data:
                                self.eval_rollout.submit(item, eval_workflow)
                                cnt += 1
                        try:
                            results = self.eval_rollout.wait(cnt, timeout=1800) # TODO: Make the eval timeout configurable.
                            print(f"Evaluated {cnt} tasks")
                            print(f"Task Rewards: {results['task_reward']}")
                            
                            tr = torch.as_tensor(results['task_reward'], dtype=torch.float32, device=self.actor.device)
                            mask = torch.ones_like(tr, dtype=torch.bool)

                            stats_tracker.get("eval").denominator(task_reward_mask=mask)
                            stats_tracker.get("eval").stat("task_reward_mask", task_reward=tr)  # default: AVG/MIN/MAX
                        except TimeoutError as e:
                            print(f"Evaluation timed out after 1500 seconds: {e}")
                        
                    dist.barrier(device_ids=[self.actor.device.index])
                    current_platform.synchronize() 

                self.evaluator.evaluate(
                    evaluate_fn,
                    epoch,
                    step,
                    global_step,
                )

            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # Upload statistics to the logger (e.g., wandb)
            stats[0].update(
                stats_tracker.export_all(reduce_group=self.actor.data_parallel_group)
            )
            self.stats_logger.commit(epoch, step, global_step, stats)

            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # Resume rollout
            self.rollout.resume()
    
    def close(self):
        self.stats_logger.close()
        self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        self.actor.destroy()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
