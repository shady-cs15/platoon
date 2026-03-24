from __future__ import annotations

from typing import Any

from platoon.episode.trajectory import TrajectoryCollection


def _get_trajectories(trajectory_collection: dict[str, Any] | TrajectoryCollection) -> dict[str, Any]:
    return (
        trajectory_collection["trajectories"]
        if isinstance(trajectory_collection, dict)
        else trajectory_collection.trajectories
    )


def _get_steps(trajectory: Any) -> list[Any]:
    return trajectory.get("steps", []) if isinstance(trajectory, dict) else trajectory.steps


def _get_step_reward_misc(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        return step.setdefault("misc", {}).setdefault("reward_misc", {})
    if step.misc is None:
        step.misc = {}
    return step.misc.setdefault("reward_misc", {})


def propagate_root_success(
    trajectory_collection: dict[str, Any] | TrajectoryCollection,
) -> dict[str, Any] | TrajectoryCollection:
    """Rewrite recursive rollout rewards so all trajectories use root success.

    Copies the root trajectory's reward/success into every child trajectory's
    final step.  Also rewrites reward/subagent_succeeded so that the standard
    reward_processor computes delegation bonuses correctly: intermediate agents
    (which launched subagents) get the bonus, leaf agents do not.
    """
    trajectories = _get_trajectories(trajectory_collection)
    if not trajectories:
        return trajectory_collection

    root_trajectory = next(iter(trajectories.values()))
    root_steps = _get_steps(root_trajectory)

    root_success = 0.0
    if root_steps:
        root_success = float(_get_step_reward_misc(root_steps[-1]).get("reward/success", 0.0))

    for trajectory in trajectories.values():
        steps = _get_steps(trajectory)
        if steps:
            _get_step_reward_misc(steps[-1])["reward/success"] = root_success
        for step in steps:
            reward_misc = _get_step_reward_misc(step)
            launched = float(reward_misc.get("reward/subagent_launched", 0.0))
            if launched > 0:
                reward_misc["reward/subagent_succeeded"] = launched * root_success

    return trajectory_collection
