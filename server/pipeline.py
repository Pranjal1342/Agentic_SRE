"""
server/pipeline.py — Main episode pipeline.

Orchestrates: task injection → FSM reset → agent reasoning loop →
grader → memory write (open/close episode, credit assignment).
The memory service is called from here — never from inside the FSM or mock_infra.
"""
from __future__ import annotations

import asyncio
import time
import structlog

from config import settings
from server.fsm import EpisodeFSM, EpisodeContext
from graders.reward import compute_reward
from memory.write import open_episode, close_episode, write_decision
from memory.credit_assignment import backfill_credit_labels
from agents.reasoning_loop import ReasoningLoop

log = structlog.get_logger(__name__)

# Task registry — populated by task modules
TASK_REGISTRY: dict = {}


def register_task(task_id: str, task_module) -> None:
    TASK_REGISTRY[task_id] = task_module


async def _probe_sustained_metrics(fsm, window_probes: Optional[int] = None, probe_interval_s: Optional[float] = None) -> dict:
    """
    Sample golden-signal metrics repeatedly after the agent resolves, instead of
    once. A temporary mask (e.g. scale_up hiding a leak) looks fine on the first
    snapshot and degrades again shortly after; a real fix holds. Returns the
    worst-case (most-degraded) value per service/signal across the window, so a
    late regression isn't averaged away by an early good reading.
    """
    if window_probes is None:
        window_probes = settings.window_probes
    if probe_interval_s is None:
        probe_interval_s = settings.probe_interval_s

    snapshots = []
    for i in range(window_probes):
        snapshots.append(fsm.telemetry.collect_metrics())
        if i < window_probes - 1:
            await asyncio.sleep(probe_interval_s)

    worst_case: dict = {}
    for snap in snapshots:
        for service, signals in snap.items():
            worst_case.setdefault(service, {})
            for signal, value in signals.items():
                # all three current signals are lower-is-better, so "worst" = max observed
                if signal not in worst_case[service] or value > worst_case[service][signal]:
                    worst_case[service][signal] = value
    return worst_case


async def run_episode(task_id: str, fsm: EpisodeFSM) -> EpisodeContext:
    """
    Run a single complete episode:
    1. Look up and inject the task fault
    2. FSM reset (environment only)
    3. Open episode row in memory
    4. Run agent reasoning loop (which writes per-decision rows)
    5. Grade the outcome
    6. Close episode row in memory
    7. Backfill credit labels
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id: '{task_id}'. Registered: {list(TASK_REGISTRY)}")

    task = TASK_REGISTRY[task_id]

    # ── 1. FSM reset (env state only) ─────────────────────────────────────────
    ctx = fsm.reset(task_id=task_id)

    # ── 2. Inject fault into environment ──────────────────────────────────────
    task.inject_fault(fsm.mesh, fsm.db)
    golden_targets = task.golden_signal_targets()

    log.info(
        "pipeline.episode_start",
        episode_id=ctx.episode_id,
        task_id=task_id,
        golden_targets=golden_targets,
    )

    # ── 3. Open episode row in memory (OUTSIDE env reset boundary) ────────────
    await open_episode(
        episode_id=ctx.episode_id,
        task_id=task_id,
        agent_version=settings.agent_version,
        started_at=ctx.started_at,
    )

    # ── 4. Agent reasoning loop ───────────────────────────────────────────────
    loop = ReasoningLoop(
        tools=fsm.tools,
        telemetry=fsm.telemetry,
        fsm=fsm,
        episode_id=ctx.episode_id,
        golden_targets=golden_targets,
    )
    await loop.run()

    # ── 5. Grade outcome ──────────────────────────────────────────────────────
    final_metrics = await _probe_sustained_metrics(fsm)
    reward = compute_reward(final_metrics, golden_targets)

    # Determine outcome label
    if reward >= 0.85:
        outcome = "success"
    elif reward >= 0.4:
        outcome = "partial"
    else:
        outcome = "failure"

    # Incorporate timeout
    if ctx.timed_out:
        outcome = "failure"

    if fsm.tools.resolution_submitted:
        resolution_summary = getattr(fsm.tools, "_resolution_summary", "") or (
            f"Episode ended with outcome={outcome}, reward={reward:.4f} (submitted, no summary text)"
        )
    else:
        resolution_summary = f"Episode ended with outcome={outcome}, reward={reward:.4f} (no resolution submitted)"

    fsm.resolve(outcome=outcome, reward=reward, summary=resolution_summary)

    log.info(
        "pipeline.episode_end",
        episode_id=ctx.episode_id,
        outcome=outcome,
        reward=reward,
    )

    # ── 6. Close episode row in memory ────────────────────────────────────────
    await close_episode(
        episode_id=ctx.episode_id,
        outcome=outcome,
        final_reward=reward,
        resolution_summary=resolution_summary,
        ended_at=time.time(),
    )

    # ── 7. Backfill credit labels ─────────────────────────────────────────────
    await backfill_credit_labels(episode_id=ctx.episode_id, outcome=outcome)

    return ctx
