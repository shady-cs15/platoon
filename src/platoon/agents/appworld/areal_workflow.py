from platoon.agents.appworld.rollout import run_single_rollout_process, run_single_recursive_rollout_process
from areal.utils.data import concat_padded_tensors
from platoon.generators.types import LLMClientSpec, ArealEngineSpec

import asyncio
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from dataclasses import asdict, is_dataclass


def _engine_config_to_dict(engine) -> dict:
    cfg = getattr(engine, "config", None)
    if cfg is None:
        return {}
    try:
        if is_dataclass(cfg):
            return asdict(cfg)
    except Exception:
        pass
    if hasattr(cfg, "as_dict"):
        try:
            return cfg.as_dict()
        except Exception:
            pass
    if hasattr(cfg, "to_dict"):
        try:
            return cfg.to_dict()
        except Exception:
            pass
    try:
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    except Exception:
        return {}

class AppWorldArealWorkflow:
    def __init__(self, config):
        self.config = config
        
        # import multiprocessing as mp
        # try:
        #     mp.set_start_method("spawn")
        #     self.logger.info("Set start method to spawn in rollout generator.")
        # except RuntimeError:
        #     # The start method can only be set once per interpreter session.
        #     self.logger.warning("Failed to set start method to spawn in rollout generator.")
        #     pass
    

    async def arun_episode(self, engine, data):
        
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
        
        config = deepcopy(self.config)
        # Provide a spawn-safe spec describing how to build the engine in the child
        config['llm_client_spec'] = LLMClientSpec(
            kind="areal",
            model_name=self.config['model_name'],
            areal_engine_spec=ArealEngineSpec(
                class_path="areal.engine.sglang_remote:RemoteSGLangEngine",
                config=_engine_config_to_dict(engine),
                initialize_kwargs={
                    "train_data_parallel_size": getattr(engine, "train_data_parallel_size", None)
                } if hasattr(engine, "train_data_parallel_size") else None,
            ),
        )
        task_id = data['task_id']
        
        args = (task_id, config)
        
        results = await loop.run_in_executor(executor, run_single_rollout_process, args)
        
        areal_completion_data_list = []
        for trajectory in results['trajectories'].values():
            areal_completion_data_list.append(trajectory.misc['areal_completion_data'])
            
        return concat_padded_tensors(areal_completion_data_list)
    

class AppWorldArealRecursiveWorkflow(AppWorldArealWorkflow):
    def __init__(self, config):
        super().__init__(config)
        
    async def arun_episode(self, engine, data):
        loop = asyncio.get_event_loop()
        executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
        
        config = deepcopy(self.config)
        config['llm_client_spec'] = LLMClientSpec(
            kind="areal",
            model_name=self.config['model_name'],
            areal_engine_spec=ArealEngineSpec(
                class_path="areal.engine.sglang_remote:RemoteSGLangEngine",
                config=_engine_config_to_dict(engine),
                initialize_kwargs={
                    "train_data_parallel_size": getattr(engine, "train_data_parallel_size", None)
                } if hasattr(engine, "train_data_parallel_size") else None,
            ),
        )
        task_id = data['task_id']
        
        args = (task_id, config)
        results = await loop.run_in_executor(executor, run_single_recursive_rollout_process, args)
        
        areal_completion_data_list = []
        for trajectory in results['trajectories'].values():
            areal_completion_data_list.append(trajectory['misc']['areal_completion_data'])
            
        print(areal_completion_data_list)
        
        return concat_padded_tensors(areal_completion_data_list)