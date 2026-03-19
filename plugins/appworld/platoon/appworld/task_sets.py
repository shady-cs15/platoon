from __future__ import annotations

from appworld import load_task_ids


CHALLENGE_TRAIN_TASK_IDS: tuple[str, ...] = (
    "6104387_1",
    "6104387_2",
    "6104387_3",
    "afc0fce_1",
    "afc0fce_2",
    "afc0fce_3",
    "27e1026_1",
    "27e1026_2",
    "27e1026_3",
    "22cc237_1",
    "22cc237_2",
    "22cc237_3",
    "34d9492_1",
    "34d9492_2",
    "34d9492_3",
    "ce359b5_1",
    "ce359b5_2",
    "ce359b5_3",
    "229360a_1",
    "229360a_2",
    "229360a_3",
    "7d7fbf6_1",
    "7d7fbf6_2",
    "7d7fbf6_3",
    "771d8fc_1",
    "771d8fc_2",
    "771d8fc_3",
    "3c13f5a_1",
    "3c13f5a_2",
    "3c13f5a_3",
    "e7a10f8_1",
    "e7a10f8_2",
    "e7a10f8_3",
    "6ea6792_1",
    "6ea6792_2",
    "6ea6792_3",
)

CHALLENGE_DEV_TASK_IDS: tuple[str, ...] = (
    "50e1ac9_1",
    "50e1ac9_2",
    "50e1ac9_3",
    "fac291d_1",
    "fac291d_2",
    "fac291d_3",
    "d4e9306_1",
    "d4e9306_2",
    "d4e9306_3",
    "57c3486_1",
    "57c3486_2",
    "57c3486_3",
    "68ee2c9_1",
    "68ee2c9_2",
    "68ee2c9_3",
    "6171bbc_1",
    "6171bbc_2",
    "6171bbc_3",
    "6c2c621_1",
    "6c2c621_2",
    "6c2c621_3",
    "4fab96f_1",
    "4fab96f_2",
    "4fab96f_3",
)

CHALLENGE_TASK_IDS: frozenset[str] = frozenset(CHALLENGE_TRAIN_TASK_IDS + CHALLENGE_DEV_TASK_IDS)

_VALID_TRAIN_SPLITS = {"train", "train_plus_dev"}
_VALID_EVAL_SPLITS = {"dev", "test_normal", "test_challenge"}
_VALID_FILTERS = {"none", "challenge"}


def resolve_train_task_ids(train_split: str, task_filter: str) -> list[str]:
    if train_split not in _VALID_TRAIN_SPLITS:
        raise ValueError(f"Invalid train_split={train_split!r}. Expected one of {_VALID_TRAIN_SPLITS}.")
    if task_filter not in _VALID_FILTERS:
        raise ValueError(f"Invalid task_filter={task_filter!r}. Expected one of {_VALID_FILTERS}.")

    split_names = ["train"] if train_split == "train" else ["train", "dev"]
    task_ids: list[str] = []
    for split_name in split_names:
        task_ids.extend(load_task_ids(dataset_name=split_name))
    return _apply_filter(task_ids, task_filter)


def resolve_eval_task_ids(eval_split: str) -> list[str]:
    if eval_split not in _VALID_EVAL_SPLITS:
        raise ValueError(f"Invalid eval_split={eval_split!r}. Expected one of {_VALID_EVAL_SPLITS}.")
    return list(load_task_ids(dataset_name=eval_split))


def summarize_task_selection() -> dict[str, int]:
    train_ids = list(load_task_ids(dataset_name="train"))
    dev_ids = list(load_task_ids(dataset_name="dev"))
    return {
        "train_raw": len(train_ids),
        "dev_raw": len(dev_ids),
        "train_challenge": len(_apply_filter(train_ids, "challenge")),
        "dev_challenge": len(_apply_filter(dev_ids, "challenge")),
        "train_plus_dev_challenge": len(_apply_filter(train_ids + dev_ids, "challenge")),
    }


def _apply_filter(task_ids: list[str], task_filter: str) -> list[str]:
    if task_filter == "none":
        return list(task_ids)
    if task_filter == "challenge":
        return [task_id for task_id in task_ids if task_id in CHALLENGE_TASK_IDS]
    raise ValueError(f"Invalid task_filter={task_filter!r}.")
