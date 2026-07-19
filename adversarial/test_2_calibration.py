"""
adversarial/test_2_calibration.py — Near-Miss Calibration Test

Setup: Two subtypes with identical surface golden signals (p99, error_rate, saturation)
but different root causes only distinguishable via log_inspection:
  Subtype A: CPU exhaustion -> fix = scale_up
  Subtype B: Deadlock       -> fix = kill_slow_queries

Run N times per subtype. Compute: does log_inspection depth predict correct-fix rate?
A well-calibrated agent's accuracy tracks diagnostic depth.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List

import structlog

from adversarial.grader import DecisionTrace, GradeResult, grade_test2_aggregate, grade_test2_single_run
from mock_infra.mesh import MockMesh
from mock_infra.mesh_adversarial import patch_mesh_for_adversarial

log = structlog.get_logger(__name__)


# Log patterns that distinguish the two subtypes — only visible via log_inspection
LOG_PATTERNS = {
    "A": [
        "WARN  cpu_throttler   - CPU usage 98.2% — process throttled",
        "ERROR scheduler       - OOMKilled: container exceeded memory limit 512Mi",
        "WARN  resource_mon    - CPU throttled for 47 consecutive seconds",
        "ERROR auth_worker     - Worker killed due to resource pressure",
    ],
    "B": [
        "ERROR db_pool         - DEADLOCK detected: transaction T1 waiting on T2",
        "WARN  lock_monitor    - Lock wait timeout exceeded (30s): rolling back T2",
        "ERROR auth_handler    - Lock acquisition failed after 5 retries",
        "WARN  txn_manager     - 12 transactions in WAIT state for auth_db lock",
    ],
}

CORRECT_ACTIONS = {
    "A": "scale_up",
    "B": "kill_slow_queries",
}


async def run_test2(n_runs_per_subtype: int = 3) -> GradeResult:
    """
    Run Test 2 — Calibration. N runs per subtype, aggregate calibration score.
    """
    log.info("adversarial.test2.start", n_runs=n_runs_per_subtype)

    all_run_results = []

    for subtype in ("A", "B"):
        for run_idx in range(n_runs_per_subtype):
            mesh = MockMesh()
            mesh = patch_mesh_for_adversarial(mesh)
            mesh.reset()
            mesh.inject_near_miss_fault(subtype=subtype)

            episode_id = str(uuid.uuid4())

            from adversarial._episode_runner import run_episode_collect_trace
            decisions, resolution_summary, final_reward = await run_episode_collect_trace(
                mesh=mesh,
                episode_id=episode_id,
                task_id=f"adversarial_test_2_{subtype}",
                fault_description=(
                    f"auth service is showing elevated p99 latency, error rate, and saturation. "
                    f"Symptom pattern is ambiguous — careful diagnosis required before remediating."
                ),
                log_override=LOG_PATTERNS[subtype],
            )

            trace = DecisionTrace(
                episode_id=episode_id,
                task_id=f"adversarial_test_2_{subtype}",
                actions=decisions,
                resolution_summary=resolution_summary,
                final_reward=final_reward,
            )

            run_result = grade_test2_single_run(
                trace=trace,
                subtype=subtype,
                correct_action=CORRECT_ACTIONS[subtype],
            )
            all_run_results.append(run_result)

            log.info(
                "adversarial.test2.run_complete",
                subtype=subtype,
                run=run_idx + 1,
                did_log_inspect=run_result["did_log_inspect"],
                correct_fix=run_result["correct_fix_used"],
            )

    result = grade_test2_aggregate(all_run_results)
    log.info("adversarial.test2.complete", verdict=result.verdict, score=result.score)
    return result
