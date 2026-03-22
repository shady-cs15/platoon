#!/usr/bin/env python3
"""Analyze multi-agent RL rollouts.

Usage:
    python analyze_rollouts.py /fsx/areal/experiments/appworld-challenge-recursive-sat-0
    python analyze_rollouts.py /fsx/areal/experiments/appworld-recursive-sat-21
"""
import json
import glob
import os
import sys


def analyze(experiment_dir):
    train_dir = os.path.join(experiment_dir, "train_rollout")
    if not os.path.exists(train_dir):
        print(f"No train_rollout directory found in {experiment_dir}")
        sys.exit(1)

    steps = sorted(int(d) for d in os.listdir(train_dir) if d.isdigit())
    if not steps:
        print("No steps found.")
        sys.exit(1)

    header = (
        f"{'Step':>4} | "
        f"{'spawn':>5} {'s_ok':>4} {'s_rate':>6} | "
        f"{'no_sp':>5} {'n_ok':>4} {'n_rate':>6} | "
        f"{'child':>5} {'c_ok':>4} {'c_rate':>6} {'c_mean':>6} | "
        f"{'tasks':>5} {'t_ok':>4} {'t_rate':>6} {'r_rate':>6} | "
        f"{'sp+':>3} {'sp+%':>4} {'ns+':>3} {'ns+%':>4} {'tie':>3} {'tie%':>4}"
    )
    print(header)
    print("-" * len(header))

    for step in steps:
        path = os.path.join(train_dir, str(step), "events")
        if not os.path.exists(path):
            continue

        all_trajs = {}
        for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
            for line in open(f):
                e = json.loads(line)
                etype = e["type"]
                if etype == "trajectory_created":
                    tid = e["trajectory"]["id"]
                    all_trajs[tid] = {
                        "pi": e["trajectory"]["parent_info"],
                        "task": None,
                        "reward": None,
                    }
                elif etype == "trajectory_task_set":
                    tid = e["trajectory_id"]
                    if tid in all_trajs:
                        all_trajs[tid]["task"] = e["task"]
                elif etype == "trajectory_finished":
                    tid = e["trajectory_id"]
                    if tid in all_trajs:
                        all_trajs[tid]["reward"] = e.get("reward", 0) or 0

        ptc: dict[str, list[str]] = {}
        for tid, t in all_trajs.items():
            if t["pi"]:
                ptc.setdefault(t["pi"]["id"], []).append(tid)

        # --- Spawn / nospawn / children stats ---
        spawn_s = spawn_f = nospawn_s = nospawn_f = 0
        child_rewards: list[float] = []

        task_groups: dict[str, list[dict]] = {}

        for tid, t in all_trajs.items():
            if t["pi"] is not None:
                if t["reward"] is not None:
                    child_rewards.append(t["reward"])
                continue
            if t["reward"] is None:
                continue
            spawned = len(ptc.get(tid, [])) > 0
            success = t["reward"] >= 1.0
            if spawned:
                if success:
                    spawn_s += 1
                else:
                    spawn_f += 1
            else:
                if success:
                    nospawn_s += 1
                else:
                    nospawn_f += 1

            task_id = t["task"]["id"] if t["task"] else "unknown"
            task_groups.setdefault(task_id, []).append(
                {"reward": t["reward"], "spawned": spawned}
            )

        ts = spawn_s + spawn_f
        tn = nospawn_s + nospawn_f
        n_children = len(child_rewards)
        c_succ = sum(1 for r in child_rewards if r >= 1.0)
        c_mean = sum(child_rewards) / max(n_children, 1)

        sr = f"{spawn_s/max(ts,1)*100:.1f}%" if ts > 0 else "  N/A"
        nr = f"{nospawn_s/max(tn,1)*100:.1f}%"
        cr = f"{c_succ/max(n_children,1)*100:.0f}%" if n_children > 0 else " N/A"

        # --- Per-task RL signal (spawn=0 if no spawn attempted) ---
        n_tasks = len(task_groups)
        sp_win = nosp_win = tie = 0

        for task_id, rollouts in task_groups.items():
            sp = [r["reward"] for r in rollouts if r["spawned"]]
            nosp = [r["reward"] for r in rollouts if not r["spawned"]]
            sp_mean = sum(sp) / len(sp) if sp else 0.0
            nosp_mean = sum(nosp) / len(nosp) if nosp else 0.0

            if sp_mean > nosp_mean:
                sp_win += 1
            elif nosp_mean > sp_mean:
                nosp_win += 1
            else:
                tie += 1

        sp_pct = f"{sp_win/max(n_tasks,1)*100:.0f}%"
        ns_pct = f"{nosp_win/max(n_tasks,1)*100:.0f}%"
        ti_pct = f"{tie/max(n_tasks,1)*100:.0f}%"

        # --- Task success rate & rollout success rate ---
        tasks_with_success = sum(
            1 for rollouts in task_groups.values()
            if any(r["reward"] >= 1.0 for r in rollouts)
        )
        total_root_rollouts = ts + tn
        total_root_successes = spawn_s + nospawn_s
        task_sr = f"{tasks_with_success/max(n_tasks,1)*100:.0f}%"
        rollout_sr = f"{total_root_successes/max(total_root_rollouts,1)*100:.1f}%"

        print(
            f"{step:>4} | "
            f"{ts:>5} {spawn_s:>4} {sr:>6} | "
            f"{tn:>5} {nospawn_s:>4} {nr:>6} | "
            f"{n_children:>5} {c_succ:>4} {cr:>6} {c_mean:>6.3f} | "
            f"{n_tasks:>5} {tasks_with_success:>4} {task_sr:>6} {rollout_sr:>6} | "
            f"{sp_win:>3} {sp_pct:>4} {nosp_win:>3} {ns_pct:>4} {tie:>3} {ti_pct:>4}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)
    analyze(sys.argv[1])
