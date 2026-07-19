"""
memory/write.py — Per-decision write path.

Key rules (from brief §6):
- Vector entry (decisions row) and any related causal_edges rows from the
  same decision MUST be written in a single transaction — no partial writes.
- credit_label starts NULL and is backfilled by credit_assignment.py post-episode.
- quarantine_reason must be the gate's actual textual reason, not just the flag.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select

from memory.db import get_db_session
from memory.models import CausalEdge, Decision, Episode

log = structlog.get_logger(__name__)


# ── Episode lifecycle ─────────────────────────────────────────────────────────

async def open_episode(
    episode_id: str,
    task_id: str,
    agent_version: str,
    started_at: float,
) -> None:
    """Create the episode row at the start of an episode."""
    started_dt = datetime.utcfromtimestamp(started_at)
    async with get_db_session() as session:
        episode = Episode(
            episode_id=episode_id,
            task_id=task_id,
            agent_version=agent_version,
            started_at=started_dt,
        )
        session.add(episode)
    log.info("memory.episode.opened", episode_id=episode_id, task_id=task_id)


async def close_episode(
    episode_id: str,
    outcome: str,
    final_reward: float,
    resolution_summary: str,
    ended_at: float,
) -> None:
    """Update episode row with outcome after it completes."""
    ended_dt = datetime.utcfromtimestamp(ended_at)
    async with get_db_session() as session:
        result = await session.execute(
            select(Episode).where(Episode.episode_id == episode_id)
        )
        episode = result.scalar_one_or_none()
        if episode is None:
            log.warning("memory.episode.not_found", episode_id=episode_id)
            return
        episode.ended_at = ended_dt
        episode.outcome = outcome
        episode.final_reward = final_reward
        episode.resolution_summary = resolution_summary
        session.add(episode)
    log.info("memory.episode.closed", episode_id=episode_id, outcome=outcome, reward=final_reward)


# ── Per-decision write ────────────────────────────────────────────────────────

async def write_decision(
    episode_id: str,
    step_index: int,
    state_signature: str,
    state_embedding: Optional[List[float]],
    action_type: str,
    action_payload: Dict[str, Any],
    result_stdout: Optional[str] = None,
    result_stderr: Optional[str] = None,
    exit_code: Optional[int] = None,
    quarantine_flag: bool = False,
    no_match_flag: bool = False,
    quarantine_reason: Optional[str] = None,
    causal_edges: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Write a decision row + any causal edges in a SINGLE transaction.
    Returns the decision_id for use in subsequent edge writes.

    causal_edges: list of dicts with keys:
      - to_decision (str, UUID of subsequent decision — may be None if not yet known)
      - relation_type (str): 'caused' | 'preceded' | 'remediated_by'
      - confidence (float): default 1.0
    Note: edges where to_decision is not yet known are written with from_decision
    and the to_decision is backfilled by the caller after the next decision is written.
    """
    decision_id = str(uuid.uuid4())

    async with get_db_session() as session:
        decision = Decision(
            decision_id=decision_id,
            episode_id=episode_id,
            step_index=step_index,
            state_signature=state_signature,
            state_embedding=state_embedding,
            action_type=action_type,
            action_payload=action_payload,
            result_stdout=result_stdout,
            result_stderr=result_stderr,
            exit_code=exit_code,
            quarantine_flag=quarantine_flag,
            no_match_flag=no_match_flag,
            quarantine_reason=quarantine_reason,
            # credit_label starts NULL — backfilled post-episode
        )
        session.add(decision)

        # Write causal edges in the SAME transaction
        if causal_edges:
            for edge_spec in causal_edges:
                to_decision = edge_spec.get("to_decision")
                if to_decision is None:
                    continue  # Skip incomplete edges; caller backfills
                edge = CausalEdge(
                    edge_id=str(uuid.uuid4()),
                    from_decision=decision_id,
                    to_decision=to_decision,
                    relation_type=edge_spec.get("relation_type", "preceded"),
                    confidence=float(edge_spec.get("confidence", 1.0)),
                    last_reinforced=datetime.utcnow(),
                )
                session.add(edge)

    log.debug(
        "memory.decision.written",
        decision_id=decision_id,
        episode_id=episode_id,
        step_index=step_index,
        action_type=action_type,
        quarantine_flag=quarantine_flag,
    )
    return decision_id


async def write_causal_edge(
    from_decision: str,
    to_decision: str,
    relation_type: str = "preceded",
    confidence: float = 1.0,
) -> str:
    """Write a single causal edge (used when to_decision becomes known after the fact)."""
    edge_id = str(uuid.uuid4())
    async with get_db_session() as session:
        edge = CausalEdge(
            edge_id=edge_id,
            from_decision=from_decision,
            to_decision=to_decision,
            relation_type=relation_type,
            confidence=confidence,
            last_reinforced=datetime.utcnow(),
        )
        session.add(edge)
    return edge_id
