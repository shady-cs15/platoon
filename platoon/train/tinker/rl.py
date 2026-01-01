# Advanced functionality to add later:
# - WandB integration
# - Profiling and additional logging

import asyncio
import logging
import tinker
from dataclasses import dataclass
from datasets import Dataset
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.train import save_checkpoint_and_get_sampling_client
from platoon.train.tinker.proxy import register_tinker_llm, ModelInfo
from platoon.train.tinker.workflows.base import RolloutWorkflow
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, TrainEventTriggerConfig

logger = logging.getLogger(__name__)

class TerminateTrainLoop(Exception):
    """Exception raised to terminate the train loop."""


class PlatoonTinkerDataloader:

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle_seed: int | None = 42,
        drop_last: bool = True,
    ):
        if shuffle_seed is not None:
            dataset = dataset.shuffle(seed=shuffle_seed)

        self.dataset = dataset
        if shuffle_seed is not None:
            self.dataset = self.dataset.shuffle(seed=shuffle_seed)
        self.batched_dataset = dataset.batch(
            batch_size=batch_size,
            drop_last=drop_last,
        )
    
    def get_batch(self, batch_num: int) -> list[dict]:
        batch_num = batch_num % len(self.batched_dataset)
        batch = self.batched_dataset[batch_num]
        # Convert dict of lists to list of dicts for training.
        keys = list(batch.keys())
        length = len(next(iter(batch.values()))) if keys else 0
        return [{k: batch[k][i] for k in keys} for i in range(length)]

    def __len__(self) -> int:
        return len(self.batched_dataset)



@dataclass
class TrainLoopSharedState:
    config: PlatoonTinkerRLTrainerConfig
    shutdown_event: asyncio.Event
    train_step: int
    train_dataloader: PlatoonTinkerDataloader
    eval_dataloader: PlatoonTinkerDataloader | None
    optimizer_params: tinker.AdamParams
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    sampling_client_step: int
    sampling_client_updated_event: asyncio.Event
    model_info: ModelInfo
    
    def _get_event_frequency(self, config: TrainEventTriggerConfig) -> int:
        """Normalize the event frequency to the number of training steps.
        Args:
            config: The event trigger configuration.
        Returns:
            The event frequency normalized to the number of training steps.
        """
        if config.strategy == 'epoch':
            return self.num_train_batches_per_epoch * self.config.every
        elif config.strategy == 'step':
            return config.every
        
        raise ValueError(f"Cannot compute event frequency for strategy: {config.strategy}")

    @property
    def eval_every(self) -> int:
        return self._get_event_frequency(self.config.eval)

    @property
    def save_every(self) -> int:
        return self._get_event_frequency(self.config.checkpoint)


    @property
    def num_train_batches_per_epoch(self) -> int:
        return len(self.train_dataloader)
    
    @property
    def num_train_batches(self) -> int:
        return self.num_train_batches_per_epoch * self.config.train.num_epochs


class PlatoonTinkerRLTrainer:
    def __init__(
        self,
        config: PlatoonTinkerRLTrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_info = register_tinker_llm(
            self.config.train.model_name,
            self.config.train.renderer_name
        )

    async def _dataloader_loop(
        self, 
        shared_state: TrainLoopSharedState, 
        start_batch: int,
        end_batch: int,
        data_queue: asyncio.Queue[dict],
        dataloader: PlatoonTinkerDataloader,
        max_staleness: int | None = None,
    ):
        i_batch = start_batch
        while not shared_state.shutdown_event.is_set() and i_batch < end_batch:
            
            # Control staleness by waiting for training to catch up.
            if max_staleness is not None and i_batch - shared_state.train_step > max_staleness:
                await asyncio.sleep(20.0) 
                continue 

            batch = dataloader.get_batch(i_batch)
            for data in batch:
                await data_queue.put(data)
            i_batch += 1

    async def _rollout_workflow_worker_loop( 
        self, 
        shared_state: TrainLoopSharedState, 
        workflow: RolloutWorkflow,
        task_data_queue: asyncio.Queue[dict],
        rollout_result_queue: asyncio.Queue[list[tinker.Datum | None]],
    ):
        while not shared_state.shutdown_event.is_set():
            data = await task_data_queue.get()
            rollout = await workflow.arun_episode(data)
            rollout_result_queue.put_nowait(rollout)

    async def _train_loop(
        self, 
        task_group: asyncio.TaskGroup,
        shared_state: TrainLoopSharedState, 
        start_batch: int, 
        end_batch: int,
        train_workflow: RolloutWorkflow,
    ):

        # Initialize queues to pipeline data loading, rollout sampling and training.
        task_data_queue = asyncio.Queue[dict](maxsize=self.config.train.batch_size)
        task_rollout_result_queue = asyncio.Queue[list[tinker.Datum | None]]()

        # Launch background tasks to start streaming in rollouts for training.
        background_streaming_tasks = [
            task_group.create_task(self._dataloader_loop(shared_state, start_batch, end_batch, task_data_queue, max_staleness=self.config.train.max_staleness), name="train_data_loader_loop"),
            # Concurrency for task processing (rollout workflow sampling) is controlled by num_concurrent_rollout_workflow_workers.
            *[
                task_group.create_task(self._rollout_workflow_worker_loop(shared_state, train_workflow), name=f"train_rollout_workflow_worker_{i}")
                for i in range(self.config.train.num_concurrent_rollout_workflow_workers)
            ],
        ]

        # Start training loop.
        shared_state.train_step = start_batch

        while shared_state.train_step < end_batch:
            
            assert self.config.train.batch_size % self.config.train.num_minibatches == 0, (
                f"batch_size {self.config.train.batch_size} must be divisible by num_minibatches {self.config.train.num_minibatches}"
            )
            tasks_per_minibatch = self.config.train.batch_size // self.config.train.num_minibatches
            
            assert tasks_per_minibatch % self.config.train.num_microbatches == 0, (
                f"tasks_per_minibatch {tasks_per_minibatch} must be divisible by num_microbatches {self.config.train.num_microbatches}"
            )

            tasks_per_microbatch = tasks_per_minibatch // self.config.train.num_microbatches

            # We perform num_minibatches weight updates per batch.
            for minibatch_num in range(self.config.train.num_minibatches):

                forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
                
                # Microbatches are used for gradient accumulation. While gradient accumulation may be less important with tinker, this can help performance.
                # Microbatches are processed as they become available, allowing us to overlap rollout sampling with training even within the same batch.
                # This is a second-level of pipelining orthogonal to the async/off-policy pipelining at the batch level.
                for microbatch_num in range(self.config.train.num_microbatches):
                    
                    task_rollout_results: list[tinker.Datum] = []
                    
                    for task_num in range(tasks_per_microbatch):

                        rollout = await task_rollout_result_queue.get()

                        # Filter out stale rollouts. TODO: Consider requeuing.
                        if self.config.train.max_staleness is not None \
                            and shared_state.train_step - rollout['checkpoint_version'] > self.config.train.max_staleness:

                            logger.info(f"Stale rollout detected in batch {shared_state.train_step}. Filtering out.")
                            rollout = None

                        if rollout is not None:
                            task_rollout_results.extend(rollout)

                    forward_backward_futures.append(
                        await shared_state.training_client.forward_backward_async(
                            # Mask is not needed in forward backward computation since adv is set to 0 for masked tokens.
                            [{k: v for k, v in result.items() if k != 'mask' and k != 'checkpoint_version'} for result in task_rollout_results], 
                            loss_fn=self.config.train.loss_fn,
                            loss_fn_config=self.config.train.loss_fn_config,
                        )
                    )

                optim_future = await shared_state.training_client.optim_step_async(shared_state.optimizer_params)

                # Consume all forward backward results.
                for microbatch_num, forward_backward_future in enumerate(forward_backward_futures):
                    forward_backward_result = await forward_backward_future.result_async()
                    # TODO: Gather logprob and other metadata from training results for stats logging

                # Wait for optimizer step to complete.
                await optim_future.result_async()
            
            
            # Update sampling client.
            sampling_client, _ = await save_checkpoint_and_get_sampling_client(
                training_client=shared_state.training_client,
                i_batch=shared_state.train_step,
                log_path=self.config.log_path,
                save_every=shared_state.save_every,
            )
            shared_state.model_info.llm.update_sampling_client(sampling_client)
            shared_state.sampling_client = sampling_client
            shared_state.train_step += 1
            shared_state.sampling_client_step += 1
        
        await self.shutdown_loops(shared_state)
    
    async def _eval_loop(self, shared_state: TrainLoopSharedState, workflow: RolloutWorkflow):

        while not shared_state.shutdown_event.is_set():

            await shared_state.sampling_client_updated_event.wait()

            if shared_state.train_step % shared_state.eval_every == 0 and shared_state.train_step > 0:

                eval_data_queue = asyncio.Queue[dict](maxsize=self.config.eval.batch_size)
                eval_rollout_result_queue = asyncio.Queue[list[tinker.Datum | None]]()

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(
                        self._dataloader_loop(
                            shared_state,
                            0, len(shared_state.eval_dataloader), 
                            eval_data_queue, 
                            shared_state.eval_dataloader,
                            None
                        ), name="eval_data_loader_loop"
                    )
                    for i in range(self.config.eval.num_concurrent_rollout_workflow_workers):
                        tg.create_task(
                            self._rollout_workflow_worker_loop(
                                shared_state, 
                                workflow, 
                                eval_data_queue, 
                                eval_rollout_result_queue
                            ), name=f"eval_rollout_workflow_worker_{i}",
                        )


    async def shutdown_loops(self, shared_state: TrainLoopSharedState):
        shared_state.shutdown_event.set()
        shared_state.sampling_client_updated_event.set()
        raise TerminateTrainLoop()

    async def _create_training_client(self, service_client: tinker.ServiceClient, resume_info: dict | None) -> tinker.TrainingClient:
        if resume_info:
            # Resuming interrupted training - load optimizer state for proper continuation
            training_client = (
                await service_client.create_training_client_from_state_with_optimizer_async(
                    resume_info["state_path"]
                )
            )
            logger.info(f"Resumed training from {resume_info['state_path']}")
        elif self.config.checkpoint.load_checkpoint_path:
            # Starting fresh from a checkpoint - load weights only (fresh optimizer)
            training_client = await service_client.create_training_client_from_state_async(
                self.config.checkpoint.load_checkpoint_path
            )
            logger.info(f"Loaded weights from {self.config.checkpoint.load_checkpoint_path}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                self.config.train.model_name, rank=self.config.train.lora_rank
            )
            logger.info(f"Created new training client for {self.config.train.model_name} with rank {self.config.train.lora_rank}")

        return training_client

    
    async def train(
        self, 
        train_workflow: RolloutWorkflow, 
        eval_workflow: RolloutWorkflow
    ):
        
        shutdown_event = asyncio.Event()

        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)

        resume_info = checkpoint_utils.get_resume_info(self.config.log_path)

        training_client = await self._create_training_client(service_client, resume_info)

        # Initial sampling client to use
        path_dict = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=f"{start_batch:06d}",
            log_path=self.config.log_path,
            loop_state={"batch": start_batch},
            kind="both",
        )
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        sampling_client_step = start_batch
        sampling_client_updated_event = asyncio.Event()
        sampling_client_updated_event.set()

        self.model_info.llm.update_sampling_client(sampling_client)

        optimizer_params = tinker.AdamParams(
            learning_rate=self.config.train.optimizer.lr,
            beta1=self.config.train.optimizer.beta1,
            beta2=self.config.train.optimizer.beta2,
            eps=self.config.train.optimizer.eps,
        )

        train_dataloader = PlatoonTinkerDataloader(self.train_dataset, self.config.train.batch_size)
        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = PlatoonTinkerDataloader(self.eval_dataset, self.config.eval.batch_size)

        shared_state = TrainLoopSharedState(
            config=self.config,
            shutdown_event=shutdown_event,
            optimizer_params=optimizer_params,
            service_client=service_client,
            training_client=training_client,
            sampling_client=sampling_client,
            sampling_client_step=sampling_client_step,
            sampling_client_updated_event=sampling_client_updated_event,
            model_info=self.model_info,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

        
        start_batch = resume_info.get("batch", 0)
        end_batch = shared_state.num_train_batches
  

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._train_loop(tg, shared_state, start_batch, end_batch, train_workflow), name="train_loop")
                if self.eval_dataset is not None and self.config.eval.every > 0:
                    tg.create_task(self._eval_loop(shared_state, eval_workflow), name="eval_loop")
        except* TerminateTrainLoop:
            pass

        # Save final checkpoint
        if start_batch < end_batch:
            await checkpoint_utils.save_checkpoint_async(
                training_client=shared_state.training_client,
                name="final",
                log_path=self.config.log_path,
                kind="both",
                loop_state={"batch": end_batch},
            )
        else:
            logger.info("Training was already complete; nothing to do")

        logger.info("Training complete successfully!")
