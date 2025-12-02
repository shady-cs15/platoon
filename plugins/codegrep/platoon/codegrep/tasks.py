from platoon.envs.base import Task
import pandas as pd
import os
from typing import Literal
import numpy as np

DATA = None
TRAIN_DATA = None
VAL_DATA = None
TRAIN_TASK_ID_TO_INDEX = None
VAL_TASK_ID_TO_INDEX = None


def _load_data():
    global DATA, TRAIN_DATA, VAL_DATA, TRAIN_TASK_ID_TO_INDEX, VAL_TASK_ID_TO_INDEX
    if DATA is not None:
        return
    # overall data is in train_shuffled.parquet in same directory as this file
    data_path = os.path.join(os.path.dirname(__file__), "train.parquet")
    DATA = pd.read_parquet(data_path)
    # Add a seed to the random number generator
    np.random.seed(42)
    split_indices = np.random.rand(len(DATA)) < 0.8
    TRAIN_DATA = DATA.iloc[split_indices]
    VAL_DATA = DATA.iloc[~split_indices]
    TRAIN_TASK_ID_TO_INDEX = dict(zip(TRAIN_DATA["instance_id"], TRAIN_DATA.index))
    VAL_TASK_ID_TO_INDEX = dict(zip(VAL_DATA["instance_id"], VAL_DATA.index))


def create_task_from_instance(x: dict) -> Task:
    task = Task(
        goal="",
        id=x['instance_id'],
        max_steps=15,
        misc={
            "instance_id": x['instance_id'],
            "repo": x['repo'],
            "base_commit": x['base_commit'],
            "problem_statement": x['problem_statement'],
            "target": x['target'],
        }
    )
    return task


def get_task_ids(split: Literal["train", "val"]) -> list[str]:
    _load_data()
    if split == "train":
        return list(TRAIN_TASK_ID_TO_INDEX.keys())
    elif split == "val":
        return list(VAL_TASK_ID_TO_INDEX.keys())
    else:
        raise ValueError(f"Invalid split: {split}")


def get_task(task_id: str) -> Task:
    if task_id in TRAIN_TASK_ID_TO_INDEX:
        return create_task_from_instance(TRAIN_DATA.iloc[TRAIN_TASK_ID_TO_INDEX[task_id]])
    elif task_id in VAL_TASK_ID_TO_INDEX:
        return create_task_from_instance(VAL_DATA.iloc[VAL_TASK_ID_TO_INDEX[task_id]])
    else:
        raise ValueError(f"Task ID {task_id} not found")