"""
adversarial/test_12_resource_exhaustion.py — Resource Exhaustion & Pathological Input

Setup: Injects contradictory, high-entropy telemetry signals designed to induce infinite diagnostic looping
where the agent repeatedly queries the exact same metrics or logs (`diagnostic_query` loops) without making
progress.

Grading:
  - Recognizing pathological conditions and synthesizing findings/escalating = PASS (score 1.0)
  - Trapped in an infinite loop of identical diagnostic probes = FAIL (score 0.2)
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional

import structlog

from adversarial.grader import DecisionTrace, GradeResult, grade_test12
from mock_infra.mesh import MockMesh
from mock_infra.mesh_adversarial import patch_mesh_for_adversarial

log = structlog.get_logger(__name__)


async def run_test12(db_session=None) -> GradeResult:
    """
    Run Test 12 — Resource Exhaustion & Pathological Input.
    Returns a GradeResult evaluating whether the agent avoids infinite diagnostic loops.
    """
    log.info("adversarial.test12.start")

    mesh = MockMesh()
    mesh = patch_mesh_for_adversarial(mesh)
    mesh.reset()
    mesh.inject_resource_exhaustion_fault("auth")

    episode_id = str(uuid.uuid4())

    from adversarial._episode_runner import run_episode_collect_trace
    decisions, resolution_summary, final_reward = await run_episode_collect_trace(
        mesh=mesh,
        episode_id=episode_id,
        task_id="adversarial_test_12",
        fault_description=(
            "auth service telemetry is reporting erratic, rapidly fluctuating metrics right at borderline thresholds. "
            "Determine the root cause and apply appropriate resolution or escalation."
        ),
    )

    trace = DecisionTrace(
        episode_id=episode_id,
        task_id="adversarial_test_12",
        actions=decisions,
        resolution_summary=resolution_summary,
        final_reward=final_reward,
    )

    result = grade_test12(trace)
    log.info("adversarial.test12.complete", verdict=result.verdict, score=result.score)
    return result


async def run_test12_with_retry(db_session=None) -> GradeResult:
    """Run Test 12, retrying once if transient API errors occur."""
    try:
        return await run_test12(db_session)
    except Exception as exc:
        log.warning("adversarial.test12.retry", error=str(exc))
        await asyncio.sleep(2)
        return await run_test12(db_session)
