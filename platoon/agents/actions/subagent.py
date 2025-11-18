import asyncio

from platoon.envs.base import ForkableEnv
from platoon.agents.base import ForkableAgent
from platoon.episode.loop import run_episode
from platoon.episode.context import budget_tracker, current_agent, current_env


async def launch_subagent(goal: str, max_steps: int=15) -> str:
    """Launch a subagent to solve a task.

    Args:
        goal: The goal of the subagent.
        max_steps: The maximum number of steps the subagent can take.
    
    Returns:
        Returns the answer or finish message for the goal.
    """
    agent: ForkableAgent = current_agent.get()
    env: ForkableEnv = current_env.get()
    task = env.task

    subtask = task.fork(goal, max_steps)
    forked_agent = await agent.fork(subtask)
    forked_env = await env.fork(subtask)

    try:
        budget_tracker.get().reserve_budget(max_steps + 1, raise_on_failure=True)
    except ValueError as e:
        return (f"Not enough budget to launch subagent for goal {goal}. {e} "
        "Note: launch_subagent will automatically reserve max_steps + 1 steps "
        "since you will need one or more steps to process the result of the subagent and complete the task. "
        "You could try requesting a smaller budget or perform the task yourself.")
    
    traj = await asyncio.create_task(run_episode(forked_agent, forked_env))
    
    budget_tracker.get().release_budget(max_steps + 1)
    
    # Compute recursive steps used by the subagent (including its descendants) via budget tracker API
    used_recursive = int(budget_tracker.get().used_budget_for(traj.id))
    remaining_total = int(budget_tracker.get().remaining_budget())

    budget_message = (
        f"\n\nBudget used by subagent: {used_recursive}/{max_steps} steps. "
        f"Total remaining budget for the current task is {remaining_total} steps.\n"
    )

    return (traj.finish_message or traj.error_message or "") + budget_message
