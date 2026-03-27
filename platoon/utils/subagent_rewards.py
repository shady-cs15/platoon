from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from platoon.episode.trajectory import TrajectoryCollection

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Hierarchical LLM-judged credit assignment
# ---------------------------------------------------------------------------

_HIERARCHICAL_JUDGE_SYSTEM_PROMPT = (
    "You are judging one spawned subagent in a recursive multi-agent task.\n\n"
    "You are given:\n"
    "- the root task\n"
    "- the parent agent's goal\n"
    "- the child agent's assigned goal\n"
    "- the child agent's actions and final answer\n"
    "- what the parent did after the child returned\n"
    "- the final root outcome\n\n"
    "Score:\n"
    "1. correct: did the child correctly solve its assigned subtask?\n"
    "2. useful: did the child materially help the parent/root solve the overall task?\n\n"
    "Important:\n"
    "- A child can be correct but not useful.\n"
    "- Judge usefulness based on actual contribution, not polished wording.\n\n"
    'Return JSON only: {"correct": 0 or 1, "useful": 0 or 1, "reason": "short explanation"}'
)

_MAX_HISTORY_CHARS = 2000


@dataclass
class HierarchicalJudgeResult:
    """Result of hierarchical LLM judge for a single child trajectory."""

    correct: float
    useful: float
    reason: str


def _get_trajectory_goal(trajectory: Any) -> str:
    """Extract the goal text from a trajectory's task."""
    if isinstance(trajectory, dict):
        task = trajectory.get("task")
        if task is None:
            return ""
        if isinstance(task, dict):
            return str(task.get("goal", ""))
        return str(getattr(task, "goal", ""))
    task = getattr(trajectory, "task", None)
    if task is None:
        return ""
    return str(getattr(task, "goal", ""))


def _get_parent_info(trajectory: Any) -> tuple[str | None, int]:
    """Return (parent_id, fork_step) or (None, 0)."""
    if isinstance(trajectory, dict):
        pi = trajectory.get("parent_info")
        if pi is None:
            return None, 0
        if isinstance(pi, dict):
            return pi.get("id"), pi.get("fork_step", 0)
        return getattr(pi, "id", None), getattr(pi, "fork_step", 0)
    pi = getattr(trajectory, "parent_info", None)
    if pi is None:
        return None, 0
    return getattr(pi, "id", None), getattr(pi, "fork_step", 0)


def _extract_step_field(step: Any, field: str) -> str:
    """Extract a field from a step, checking direct attrs/keys first, then misc.

    CodeActStep stores thought/code/output as direct dataclass fields, but
    dict-serialized steps (from dataclasses.asdict) put them as top-level keys.
    Only fall back to misc for legacy or non-CodeActStep step types.
    """
    if isinstance(step, dict):
        val = step.get(field) or step.get("misc", {}).get(field, "")
    else:
        val = getattr(step, field, None)
        if val is None:
            misc = getattr(step, "misc", {}) or {}
            val = misc.get(field, "")
    return str(val or "")


def _summarize_steps(steps: list[Any], max_chars: int = _MAX_HISTORY_CHARS) -> str:
    """Build a condensed summary of trajectory steps for the judge prompt."""
    parts: list[str] = []
    total = 0
    for i, step in enumerate(steps):
        thought = _extract_step_field(step, "thought")[:200]
        code = _extract_step_field(step, "code")[:200]
        output = _extract_step_field(step, "output")[:200]
        finish_msg = _extract_step_field(step, "finish_message")[:200]
        line = f"Step {i}: thought={thought} code={code} output={output}"
        if finish_msg:
            line += f" finish={finish_msg}"
        if total + len(line) > max_chars:
            parts.append("... (truncated)")
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def _get_finish_message(trajectory: Any) -> str:
    """Extract finish message from a trajectory."""
    if isinstance(trajectory, dict):
        return str(trajectory.get("finish_message", "") or "")
    return str(getattr(trajectory, "finish_message", "") or "")


def _build_judge_user_prompt(
    root_goal: str,
    parent_goal: str,
    child_goal: str,
    child_steps_summary: str,
    child_finish_message: str,
    parent_steps_after_child: str,
    root_outcome: str,
) -> str:
    """Assemble the user prompt for the hierarchical judge."""
    return (
        f"# Root Task\n{root_goal}\n\n"
        f"# Parent Goal\n{parent_goal}\n\n"
        f"# Child Goal\n{child_goal}\n\n"
        f"# Child Actions and Final Answer\n{child_steps_summary}\n"
        f"Final answer: {child_finish_message}\n\n"
        f"# Parent Steps After Child Returned\n{parent_steps_after_child}\n\n"
        f"# Root Outcome\n{root_outcome}\n\n"
        "Evaluate the child agent. Return exactly one JSON object."
    )


def _extract_json_by_bracket_matching(text: str) -> str | None:
    """Find the first top-level JSON object in text using bracket counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_thinking_tags(response: str) -> str:
    """Remove <think>...</think> blocks from model responses."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _parse_hierarchical_judge_response(response: str) -> HierarchicalJudgeResult:
    """Parse the judge response into a HierarchicalJudgeResult.

    Handles models that include reasoning/thinking before JSON output.
    Degrades gracefully: returns zeros on parse failure rather than raising.
    """
    try:
        # Strip thinking tags if present
        cleaned = _strip_thinking_tags(response)

        # Try 1: whole response is JSON
        parsed = None
        try:
            parsed = json.loads(cleaned.strip())
        except (json.JSONDecodeError, ValueError):
            pass

        # Try 2: extract JSON by bracket matching (handles extra text around JSON)
        if parsed is None:
            json_str = _extract_json_by_bracket_matching(cleaned)
            if json_str is not None:
                try:
                    parsed = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass

        # Try 3: if cleaning didn't help, try bracket matching on original response
        if parsed is None and cleaned != response:
            json_str = _extract_json_by_bracket_matching(response)
            if json_str is not None:
                try:
                    parsed = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass

        if parsed is None:
            raise ValueError("No valid JSON object found in judge response.")

        correct = 1.0 if parsed.get("correct") in (1, 1.0, True) else 0.0
        useful = 1.0 if parsed.get("useful") in (1, 1.0, True) else 0.0
        reason = str(parsed.get("reason", ""))
        return HierarchicalJudgeResult(correct=correct, useful=useful, reason=reason)
    except Exception as exc:
        logger.warning("Failed to parse hierarchical judge response: %s — %s", response[:200], exc)
        return HierarchicalJudgeResult(correct=0.0, useful=0.0, reason=f"parse_error: {exc}")


async def judge_subagent_hierarchical(
    *,
    root_goal: str,
    parent_goal: str,
    child_goal: str,
    child_steps_summary: str,
    child_finish_message: str,
    parent_steps_after_child: str,
    root_outcome: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> HierarchicalJudgeResult:
    """Call an LLM judge to score a child trajectory on correct + useful.

    Args:
        root_goal: The root task's goal text.
        parent_goal: The parent agent's goal text.
        child_goal: The child agent's assigned subtask goal.
        child_steps_summary: Summarized action history of the child.
        child_finish_message: The child's finish/answer message.
        parent_steps_after_child: Summary of parent's steps after receiving child result.
        root_outcome: Whether the root task succeeded or failed.
        model: LLM model for judging.
        base_url: LLM API base URL.
        api_key: LLM API key.
        api_key_env: Env var name for the API key.

    Returns:
        HierarchicalJudgeResult with correct, useful, and reason.
    """
    import os

    from rubric.utils.llm_client import create_llm_client

    resolved_api_key = api_key
    if resolved_api_key is None and api_key_env:
        resolved_api_key = os.getenv(api_key_env)
        if resolved_api_key is None:
            logger.warning("Hierarchical judge API key env var %r not set, defaulting to zeros.", api_key_env)
            return HierarchicalJudgeResult(correct=0.0, useful=0.0, reason="missing_api_key")

    user_prompt = _build_judge_user_prompt(
        root_goal=root_goal,
        parent_goal=parent_goal,
        child_goal=child_goal,
        child_steps_summary=child_steps_summary,
        child_finish_message=child_finish_message,
        parent_steps_after_child=parent_steps_after_child,
        root_outcome=root_outcome,
    )

    try:
        llm_client = create_llm_client(api_key=resolved_api_key, model=model, base_url=base_url)
        response = await llm_client.asystem_completion(
            system_prompt=_HIERARCHICAL_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return _parse_hierarchical_judge_response(response)
    except Exception as exc:
        logger.warning("Hierarchical judge call failed: %s", exc)
        return HierarchicalJudgeResult(correct=0.0, useful=0.0, reason=f"judge_error: {exc}")


# ---------------------------------------------------------------------------
# Tree analysis helpers
# ---------------------------------------------------------------------------

def _build_children_map(trajectories: dict[str, Any]) -> dict[str | None, list[str]]:
    """Build a mapping from parent trajectory ID to list of immediate child IDs.

    Args:
        trajectories: Dict mapping trajectory IDs to trajectory objects.

    Returns:
        Dict mapping parent_id (or None for root) to list of child trajectory IDs.
    """
    children: dict[str | None, list[str]] = {}
    for traj_id, traj in trajectories.items():
        parent_id, _ = _get_parent_info(traj)
        children.setdefault(parent_id, []).append(traj_id)
    return children


def _classify_node(
    traj_id: str,
    children_map: dict[str | None, list[str]],
    root_traj_id: str,
) -> str:
    """Classify a trajectory node as root, leaf, or intermediate.

    Args:
        traj_id: The trajectory ID to classify.
        children_map: Mapping from parent_id to list of child IDs.
        root_traj_id: The root trajectory's ID.

    Returns:
        One of "root", "leaf", or "intermediate".
    """
    if traj_id == root_traj_id:
        return "root"
    has_children = bool(children_map.get(traj_id))
    return "intermediate" if has_children else "leaf"


async def compute_hierarchical_rewards(
    trajectory_collection: dict[str, Any] | TrajectoryCollection,
    *,
    failed_root_local_reward_weight: float = 0.2,
    hierarchical_gated_bonus_weight: float = 0.4,
    hierarchical_unconditional_bonus_weight: float = 0.0,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any] | TrajectoryCollection:
    """Compute hierarchical rewards using LLM-judged correct + useful scores.

    Reward equations:
        - Root:
          R = root_success
              + gated_weight * root_success * avg(child_useful)
              + unconditional_weight * avg(child_useful)
          (always trainable)
        - Leaf:
          R = w(root) * correct
          (trainable only if useful == 1; loss_mask zeroed otherwise)
        - Intermediate:
          R = w(root) * (
                  correct
                  + gated_weight * correct * avg(child_useful)
                  + unconditional_weight * avg(child_useful)
              )
          (trainable only if useful == 1; loss_mask zeroed otherwise)

    Where:
        - w(root) = 1.0 if root succeeds, else failed_root_local_reward_weight
        - gated_weight = hierarchical_gated_bonus_weight
        - unconditional_weight = hierarchical_unconditional_bonus_weight
        - avg(child_useful) uses immediate children only

    This function:
    1. Identifies the root trajectory and its success
    2. Builds the parent-child tree
    3. For each non-root trajectory, calls the LLM judge to get correct/useful
    4. Computes rewards and writes them into reward_misc

    Args:
        trajectory_collection: The trajectory collection to process.
        failed_root_local_reward_weight: Weight for local rewards when root fails.
        hierarchical_gated_bonus_weight: Gated usefulness bonus weight for hierarchical delegation rewards.
        hierarchical_unconditional_bonus_weight: Ungated usefulness bonus weight for hierarchical delegation rewards.
        model: LLM model for the hierarchical judge.
        base_url: LLM API base URL.
        api_key: LLM API key.
        api_key_env: Env var name for the API key.

    Returns:
        The modified trajectory collection with hierarchical rewards written.
    """
    trajectories = _get_trajectories(trajectory_collection)
    if not trajectories:
        return trajectory_collection

    # Identify root trajectory (first in dict — matches existing convention)
    root_traj_id = next(iter(trajectories))
    root_traj = trajectories[root_traj_id]
    root_steps = _get_steps(root_traj)
    root_goal = _get_trajectory_goal(root_traj)

    root_success = 0.0
    if root_steps:
        root_success = float(_get_step_reward_misc(root_steps[-1]).get("reward/success", 0.0))

    w_root = 1.0 if root_success > 0 else failed_root_local_reward_weight
    root_outcome = "SUCCESS" if root_success > 0 else "FAILED"

    children_map = _build_children_map(trajectories)

    # Judge all non-root trajectories concurrently
    judge_results: dict[str, HierarchicalJudgeResult] = {}
    pending_ids: list[str] = []
    pending_coros: list[Any] = []

    for traj_id, traj in trajectories.items():
        if traj_id == root_traj_id:
            continue

        parent_id, fork_step = _get_parent_info(traj)
        if parent_id is None:
            continue

        parent_traj = trajectories.get(parent_id)
        if parent_traj is None:
            logger.warning("Parent trajectory %s not found for child %s", parent_id, traj_id)
            judge_results[traj_id] = HierarchicalJudgeResult(correct=0.0, useful=0.0, reason="missing_parent")
            continue

        parent_goal = _get_trajectory_goal(parent_traj)
        child_goal = _get_trajectory_goal(traj)
        child_steps = _get_steps(traj)
        child_steps_summary = _summarize_steps(child_steps)
        child_finish = _get_finish_message(traj)

        # Parent steps from fork_step onward: fork_step is the index of the
        # next step added *after* the child was spawned, which is typically
        # the step where the parent awaits/uses the child's result.
        parent_steps = _get_steps(parent_traj)
        parent_steps_after = parent_steps[fork_step:] if fork_step < len(parent_steps) else []
        parent_after_summary = _summarize_steps(parent_steps_after, max_chars=1000)

        pending_ids.append(traj_id)
        pending_coros.append(
            judge_subagent_hierarchical(
                root_goal=root_goal,
                parent_goal=parent_goal,
                child_goal=child_goal,
                child_steps_summary=child_steps_summary,
                child_finish_message=child_finish,
                parent_steps_after_child=parent_after_summary,
                root_outcome=root_outcome,
                model=model,
                base_url=base_url,
                api_key=api_key,
                api_key_env=api_key_env,
            )
        )

    if pending_coros:
        results = await asyncio.gather(*pending_coros, return_exceptions=True)
        for traj_id, result in zip(pending_ids, results):
            if isinstance(result, BaseException):
                logger.warning("Judge call failed for %s: %s", traj_id, result)
                judge_results[traj_id] = HierarchicalJudgeResult(
                    correct=0.0, useful=0.0, reason=f"judge_error: {result}"
                )
            else:
                judge_results[traj_id] = result

    def _avg_immediate_child_useful(traj_id: str) -> float:
        """Compute average usefulness over immediate children of a trajectory."""
        immediate_children = children_map.get(traj_id, [])
        child_useful_scores: list[float] = []
        for child_id in immediate_children:
            child_jr = judge_results.get(child_id)
            if child_jr is not None:
                child_useful_scores.append(child_jr.useful)
        if not child_useful_scores:
            return 0.0
        return sum(child_useful_scores) / len(child_useful_scores)

    # Write rewards
    for traj_id, traj in trajectories.items():
        steps = _get_steps(traj)
        if not steps:
            continue

        node_type = _classify_node(traj_id, children_map, root_traj_id)
        final_reward_misc = _get_step_reward_misc(steps[-1])

        if node_type == "root":
            avg_child_useful = _avg_immediate_child_useful(traj_id)
            root_reward = (
                root_success
                + hierarchical_gated_bonus_weight * root_success * avg_child_useful
                + hierarchical_unconditional_bonus_weight * avg_child_useful
            )
            final_reward_misc["reward/success"] = root_reward
            final_reward_misc["reward/subagent_correct"] = 0.0
            final_reward_misc["reward/subagent_useful"] = 0.0
            final_reward_misc["reward/child_usefulness_avg"] = avg_child_useful
            final_reward_misc["reward/hierarchical_trainable"] = 1.0
            final_reward_misc["hierarchical_node_type"] = "root"

        elif node_type == "leaf":
            jr = judge_results.get(traj_id)
            correct = jr.correct if jr else 0.0
            useful = jr.useful if jr else 0.0
            reason = jr.reason if jr else "no_judge_result"

            reward = w_root * correct
            final_reward_misc["reward/success"] = reward
            final_reward_misc["reward/subagent_correct"] = correct
            final_reward_misc["reward/subagent_useful"] = useful
            final_reward_misc["reward/child_usefulness_avg"] = 0.0
            final_reward_misc["reward/hierarchical_trainable"] = useful
            final_reward_misc["hierarchical_reason"] = reason
            final_reward_misc["hierarchical_node_type"] = "leaf"

        else:  # intermediate
            jr = judge_results.get(traj_id)
            correct = jr.correct if jr else 0.0
            useful = jr.useful if jr else 0.0
            reason = jr.reason if jr else "no_judge_result"

            avg_child_useful = _avg_immediate_child_useful(traj_id)
            reward = w_root * (
                correct
                + hierarchical_gated_bonus_weight * correct * avg_child_useful
                + hierarchical_unconditional_bonus_weight * avg_child_useful
            )
            final_reward_misc["reward/success"] = reward
            final_reward_misc["reward/subagent_correct"] = correct
            final_reward_misc["reward/subagent_useful"] = useful
            final_reward_misc["reward/child_usefulness_avg"] = avg_child_useful
            final_reward_misc["reward/hierarchical_trainable"] = useful
            final_reward_misc["hierarchical_reason"] = reason
            final_reward_misc["hierarchical_node_type"] = "intermediate"

    logger.info(
        "Hierarchical rewards computed: root_success=%.1f, w_root=%.2f, "
        "%d children judged, %d intermediate nodes",
        root_success,
        w_root,
        len(judge_results),
        sum(1 for t in trajectories if _classify_node(t, children_map, root_traj_id) == "intermediate"),
    )

    return trajectory_collection
