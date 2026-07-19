"""
memory/consolidate.py — Scheduled offline consolidation job.

Reads labeled decisions → clusters by embedding similarity (k-means) →
upserts lessons rows with aggregated outcome stats.
Also applies decay to stale/unreinforced lessons and causal edges.

This job runs BETWEEN episodes (never live during an episode).
Parallel episodes do NOT see each other's lesson writes mid-batch (§8.1).
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sklearn.cluster import KMeans
from sqlalchemy import delete, select, update

from config import settings
from memory.db import get_db_session
from memory.models import CausalEdge, Decision, Lesson

log = structlog.get_logger(__name__)


async def run_consolidation() -> Dict[str, int]:
    """
    Main consolidation entry point. Called by the APScheduler job.

    Returns stats dict: {"decisions_processed", "lessons_written", "lessons_decayed", "edges_decayed"}
    """
    log.info("consolidation.start")
    stats: Dict[str, int] = {
        "decisions_processed": 0,
        "lessons_written": 0,
        "lessons_decayed": 0,
        "edges_decayed": 0,
    }

    # ── 1. Load labeled decisions with embeddings ──────────────────────────────
    decisions = await _load_labeled_decisions()
    stats["decisions_processed"] = len(decisions)

    if len(decisions) < settings.consolidation_min_cluster_size:
        log.info("consolidation.insufficient_data", count=len(decisions))
        return stats

    # ── 2. Cluster and write lessons ───────────────────────────────────────────
    lessons_written = await _cluster_and_write_lessons(decisions)
    stats["lessons_written"] = lessons_written

    # ── 3. Decay stale lessons and edges ──────────────────────────────────────
    lessons_decayed = await _decay_lessons()
    edges_decayed = await _decay_edges()
    stats["lessons_decayed"] = lessons_decayed
    stats["edges_decayed"] = edges_decayed

    log.info("consolidation.complete", **stats)
    return stats


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _load_labeled_decisions() -> List[Decision]:
    """Load all decisions that have a credit_label and a state_embedding."""
    async with get_db_session() as session:
        result = await session.execute(
            select(Decision)
            .where(
                Decision.credit_label.isnot(None),
                Decision.state_embedding.isnot(None),
            )
            .order_by(Decision.created_at)
        )
        return list(result.scalars().all())


async def _cluster_and_write_lessons(decisions: List[Decision]) -> int:
    """
    Cluster decision embeddings with k-means, then upsert a lesson per cluster.
    Returns number of lessons written/updated.
    """
    embeddings = np.array([d.state_embedding for d in decisions], dtype=np.float32)

    # Determine k: sqrt heuristic, bounded
    n = len(decisions)
    k = max(1, min(int(math.sqrt(n / 2)), 50))

    log.info("consolidation.clustering", n_decisions=n, k=k)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    lessons_written = 0
    for cluster_idx in range(k):
        mask = labels == cluster_idx
        cluster_decisions = [d for d, m in zip(decisions, mask) if m]

        if len(cluster_decisions) < settings.consolidation_min_cluster_size:
            continue

        centroid = centroids[cluster_idx].tolist()
        lesson = _build_lesson(cluster_decisions, centroid)
        await _upsert_lesson(lesson)
        lessons_written += 1

    return lessons_written


def _build_lesson(decisions: List[Decision], centroid: List[float]) -> Dict[str, Any]:
    """Aggregate a cluster of decisions into a single lesson."""
    success_count = sum(
        1 for d in decisions if d.credit_label in ("success",)
    )
    total = len(decisions)
    success_rate = success_count / total if total > 0 else 0.0

    # Best action: action from the highest-credit decision in the cluster
    # Priority: success > neutral > failure > quarantine_blocked
    priority = {"success": 0, "neutral": 1, "failure": 2, "quarantine_blocked": 3}
    sorted_decisions = sorted(decisions, key=lambda d: priority.get(d.credit_label or "neutral", 1))
    best_decision = sorted_decisions[0]

    # Build outcome summary from the cluster
    action_types = list({d.action_type for d in decisions})
    quarantine_reasons = list({
        d.quarantine_reason for d in decisions
        if d.quarantine_reason
    })[:3]  # top 3 unique reasons

    outcome_summary = (
        f"Cluster of {total} decisions with action types: {', '.join(action_types)}. "
        f"Success rate: {success_rate:.0%}."
    )
    if quarantine_reasons:
        outcome_summary += f" Known rejection reasons: {'; '.join(quarantine_reasons)}."

    return {
        "state_cluster_embedding": centroid,
        "best_action": {
            "action_type": best_decision.action_type,
            "action_payload": best_decision.action_payload,
            "credit_label": best_decision.credit_label,
        },
        "outcome_summary": outcome_summary,
        "sample_count": total,
        "success_rate": success_rate,
        "last_updated": datetime.utcnow(),
    }


async def _upsert_lesson(lesson: Dict[str, Any]) -> None:
    """Insert a new lesson row (consolidation always inserts; no dedup for now)."""
    async with get_db_session() as session:
        new_lesson = Lesson(
            lesson_id=str(uuid.uuid4()),
            state_cluster_embedding=lesson["state_cluster_embedding"],
            best_action=lesson["best_action"],
            outcome_summary=lesson["outcome_summary"],
            sample_count=lesson["sample_count"],
            success_rate=lesson["success_rate"],
            last_updated=lesson["last_updated"],
        )
        session.add(new_lesson)


# ── Decay ─────────────────────────────────────────────────────────────────────

async def _decay_lessons() -> int:
    """
    Delete lessons not updated within decay_halflife_days * 2 days.
    (Exponential decay approximated by hard cutoff at 2x half-life.)
    """
    cutoff = datetime.utcnow() - timedelta(days=settings.lesson_decay_halflife_days * 2)
    async with get_db_session() as session:
        result = await session.execute(
            delete(Lesson)
            .where(Lesson.last_updated < cutoff)
            .returning(Lesson.lesson_id)
        )
        deleted = len(result.fetchall())
    if deleted:
        log.info("consolidation.lessons_decayed", count=deleted)
    return deleted


async def _decay_edges() -> int:
    """Delete causal edges not reinforced within the same window."""
    cutoff = datetime.utcnow() - timedelta(days=settings.lesson_decay_halflife_days * 2)
    async with get_db_session() as session:
        result = await session.execute(
            delete(CausalEdge)
            .where(CausalEdge.last_reinforced < cutoff)
            .returning(CausalEdge.edge_id)
        )
        deleted = len(result.fetchall())
    if deleted:
        log.info("consolidation.edges_decayed", count=deleted)
    return deleted
