"""
memory/retrieve.py — Retrieval-before-decision.

Queries the `lessons` table ONLY (never raw `decisions` at inference time).
Embeds the current state signature → cosine similarity search →
returns top-k lessons above threshold.

Logs no-match events when nothing clears the similarity threshold.
This no-match rate is the key trigger metric for deciding if DPO is needed.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sqlalchemy import select, text

from config import settings
from memory.db import get_db_session
from memory.models import Lesson, LessonSchema

log = structlog.get_logger(__name__)

# Module-level embedding model (lazy-loaded, shared across calls)
_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(settings.embed_model)
        log.info("retriever.embed_model_loaded", model=settings.embed_model)
    return _embed_model


def embed_state(state_signature: str) -> List[float]:
    """Embed a state signature string into a 384-dim vector."""
    try:
        model = _get_embed_model()
        vec = model.encode(state_signature, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)
    except Exception as exc:
        log.warning("retriever.embed_fallback", error=str(exc))
        return [0.0] * 384


async def retrieve_lessons(
    state_signature: str,
    task_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    step_index: Optional[int] = None,
) -> Tuple[List[LessonSchema], bool]:
    """
    Retrieve the top-k most similar lessons from the lessons table.

    Returns:
      (lessons, no_match)
      - lessons: list of LessonSchema with similarity field populated
      - no_match: True if no lesson cleared the similarity threshold (log this!)

    The no_match flag is the trigger metric (§7) — log it with context.
    """
    embedding = embed_state(state_signature)
    embedding_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    lessons: List[LessonSchema] = []
    no_match = True

    try:
        async with get_db_session() as session:
            # pgvector cosine similarity: 1 - cosine_distance
            result = await session.execute(
                text("""
                    SELECT
                        lesson_id,
                        best_action,
                        outcome_summary,
                        sample_count,
                        success_rate,
                        last_updated,
                        1 - (state_cluster_embedding <=> :embedding ::vector) AS similarity
                    FROM lessons
                    WHERE state_cluster_embedding IS NOT NULL
                    ORDER BY state_cluster_embedding <=> :embedding ::vector
                    LIMIT :top_k
                """),
                {"embedding": embedding_str, "top_k": settings.retrieval_top_k},
            )
            rows = result.fetchall()
    except Exception as exc:
        log.warning("retriever.db_unavailable", error=str(exc), note="Continuing without memory retrieval")
        rows = []

    for row in rows:
        similarity = float(row.similarity)
        if similarity >= settings.similarity_threshold:
            no_match = False
            lessons.append(
                LessonSchema(
                    lesson_id=str(row.lesson_id),
                    best_action=row.best_action,
                    outcome_summary=row.outcome_summary,
                    sample_count=row.sample_count,
                    success_rate=row.success_rate,
                    similarity=round(similarity, 4),
                )
            )

    # ── Log no-match event (§7 trigger metric) ────────────────────────────────
    if no_match:
        log.info(
            "retrieval.no_match",
            task_id=task_id,
            episode_id=episode_id,
            step_index=step_index,
            state_signature_len=len(state_signature),
            total_lessons_checked=len(rows),
            threshold=settings.similarity_threshold,
        )
    else:
        log.debug(
            "retrieval.matched",
            task_id=task_id,
            episode_id=episode_id,
            step_index=step_index,
            matched_count=len(lessons),
            top_similarity=lessons[0].similarity if lessons else None,
        )

    return lessons, no_match


def format_lessons_for_context(lessons: List[LessonSchema]) -> str:
    """
    Format retrieved lessons as a context string for injection into the
    agent's reasoning prompt.
    """
    if not lessons:
        return ""

    lines = ["## Relevant Past Experience (retrieved from memory)\n"]
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"### Lesson {i} (similarity={lesson.similarity:.2f}, success_rate={lesson.success_rate:.0%})")
        if lesson.outcome_summary:
            lines.append(f"**Situation**: {lesson.outcome_summary}")
        if lesson.best_action:
            action_str = json.dumps(lesson.best_action, indent=2)
            lines.append(f"**Best action that worked**:\n```json\n{action_str}\n```")
        lines.append(f"*Based on {lesson.sample_count} past episodes.*\n")

    return "\n".join(lines)
