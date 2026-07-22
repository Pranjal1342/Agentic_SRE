"""
inference.py — Baseline eval runner.

Runs N episodes per task, collects:
  - final_reward per episode → reward spread distribution
  - No-match rate: fraction of decisions where retrieval found nothing above threshold
  - Quarantine rejection rate per task

Outputs structured JSON report to stdout and saves to inference_results.json.

Usage:
  python inference.py --tasks task_1 task_2 task_3 task_4 --n-episodes 5
  python inference.py --tasks task_3 --n-episodes 10  # focus on the hard task
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from config import settings
from memory.db import init_db
from server.fsm import EpisodeFSM
from server.pipeline import run_episode, register_task

# Register tasks
import tasks.task_1 as task_1
import tasks.task_2 as task_2
import tasks.task_3 as task_3
import tasks.task_4 as task_4

log = structlog.get_logger(__name__)

TASK_MODULES = {
    "task_1": task_1,
    "task_2": task_2,
    "task_3": task_3,
    "task_4": task_4,
}

for _tid, _mod in TASK_MODULES.items():
    register_task(_tid, _mod)

# Baseline scores from original repo for comparison
BASELINE_SCORES = {
    "task_1": None,
    "task_2": None,
    "task_3": 0.6190,  # the one to watch
    "task_4": None,
}


async def eval_task(task_id: str, n_episodes: int) -> Dict[str, Any]:
    """
    Run n_episodes for a single task. Returns per-task statistics.
    """
    if task_id in TASK_MODULES:
        register_task(task_id, TASK_MODULES[task_id])
    rewards: List[float] = []
    outcomes: List[str] = []
    step_counts: List[int] = []

    # Collect no-match logs from structured logging (via log interceptor)
    # In production, these would go to a metrics backend.
    # For now we parse the session-level stats from the DB post-run.

    log.info("eval.task_start", task_id=task_id, n_episodes=n_episodes)
    task_start = time.time()

    for ep_idx in range(n_episodes):
        fsm = EpisodeFSM()
        try:
            ctx = await run_episode(task_id=task_id, fsm=fsm)
            rewards.append(ctx.final_reward or 0.0)
            outcomes.append(ctx.outcome or "unknown")
            step_counts.append(ctx.step_index)
            log.info(
                "eval.episode_complete",
                task_id=task_id,
                episode_index=ep_idx,
                reward=ctx.final_reward,
                outcome=ctx.outcome,
            )
        except Exception as exc:
            err_str = str(exc)
            log.exception("eval.episode_error", task_id=task_id, ep_idx=ep_idx, error=err_str)
            rewards.append(0.0)
            outcomes.append("error")
            step_counts.append(0)
            last_error = err_str

    task_elapsed = time.time() - task_start

    # Compute reward spread statistics
    if rewards:
        mean_reward = statistics.mean(rewards)
        std_reward = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        min_reward = min(rewards)
        max_reward = max(rewards)
    else:
        mean_reward = std_reward = min_reward = max_reward = 0.0

    outcome_counts = defaultdict(int)
    for o in outcomes:
        outcome_counts[o] += 1

    baseline = BASELINE_SCORES.get(task_id)

    return {
        "task_id": task_id,
        "n_episodes": n_episodes,
        "elapsed_seconds": round(task_elapsed, 1),
        "last_error": last_error if "last_error" in locals() else None,
        "rewards": {
            "mean": round(mean_reward, 4),
            "std": round(std_reward, 4),
            "min": round(min_reward, 4),
            "max": round(max_reward, 4),
            "all": [round(r, 4) for r in rewards],
        },
        "outcomes": dict(outcome_counts),
        "step_counts": {
            "mean": round(statistics.mean(step_counts), 1) if step_counts else 0,
            "max": max(step_counts) if step_counts else 0,
        },
        "baseline_score": baseline,
        "delta_vs_baseline": (
            round(mean_reward - baseline, 4) if baseline is not None else None
        ),
    }


async def collect_no_match_rate(task_id: str) -> Dict[str, Any]:
    """
    Query the DB for no-match rate stats for a task.
    No-match events are logged to the retrieval.no_match structlog key.
    This function queries the decisions table to compute the rate from DB.
    """
    from memory.db import get_db_session
    from sqlalchemy import func, select, text

    async with get_db_session() as session:
        # Total decisions for this task
        total_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM decisions d
                JOIN episodes e ON d.episode_id = e.episode_id
                WHERE e.task_id = :task_id
            """),
            {"task_id": task_id},
        )
        total = total_result.scalar() or 0

        # Quarantine rejection rate
        quarantine_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM decisions d
                JOIN episodes e ON d.episode_id = e.episode_id
                WHERE e.task_id = :task_id AND d.quarantine_flag = TRUE
            """),
            {"task_id": task_id},
        )
        quarantine_count = quarantine_result.scalar() or 0

        # No-match rate
        no_match_result = await session.execute(
            text("""
                SELECT COUNT(*) FROM decisions d
                JOIN episodes e ON d.episode_id = e.episode_id
                WHERE e.task_id = :task_id AND d.no_match_flag = TRUE
            """),
            {"task_id": task_id},
        )
        no_match_count = no_match_result.scalar() or 0

    return {
        "task_id": task_id,
        "total_decisions": total,
        "quarantine_blocked": quarantine_count,
        "quarantine_rate": round(quarantine_count / total, 4) if total > 0 else 0.0,
        "no_match_count": no_match_count,
        "no_match_rate": round(no_match_count / total, 4) if total > 0 else 0.0,
    }


async def main(task_ids: List[str], n_episodes: int, output_path: Optional[str]) -> None:
    await init_db()

    for task_id, module in TASK_MODULES.items():
        if task_id in task_ids:
            register_task(task_id, module)

    results: Dict[str, Any] = {
        "agent_version": settings.agent_version,
        "claude_model": settings.claude_model,
        "n_episodes_per_task": n_episodes,
        "tasks": [],
        "metrics": [],
    }

    for task_id in task_ids:
        if task_id not in TASK_MODULES:
            log.warning("eval.unknown_task", task_id=task_id)
            continue

        task_result = await eval_task(task_id, n_episodes)
        results["tasks"].append(task_result)

        metrics = await collect_no_match_rate(task_id)
        results["metrics"].append(metrics)

    # Print report
    print("\n" + "=" * 60)
    print("INFERENCE EVALUATION REPORT")
    print("=" * 60)
    for task in results["tasks"]:
        print(f"\nTask: {task['task_id']}")
        print(f"  Mean reward : {task['rewards']['mean']:.4f}  (std={task['rewards']['std']:.4f})")
        print(f"  Min / Max   : {task['rewards']['min']:.4f} / {task['rewards']['max']:.4f}")
        if task["delta_vs_baseline"] is not None:
            sign = "+" if task["delta_vs_baseline"] >= 0 else ""
            print(f"  vs baseline : {sign}{task['delta_vs_baseline']:.4f}  (baseline={task['baseline_score']})")
        print(f"  Outcomes    : {task['outcomes']}")

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)
    for m in results["metrics"]:
        print(f"\nTask: {m['task_id']}")
        print(f"  Total decisions     : {m['total_decisions']}")
        print(f"  Quarantine rate     : {m['quarantine_rate']:.2%}")
        print(f"  No-match rate       : {m['no_match_rate']:.2%}")

    # Save to file
    out_path = Path(output_path or "inference_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRE agent baseline evaluation")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["task_1", "task_2", "task_3", "task_4"],
        help="Task IDs to evaluate",
    )
    parser.add_argument("--n-episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    asyncio.run(main(task_ids=args.tasks, n_episodes=args.n_episodes, output_path=args.output))
