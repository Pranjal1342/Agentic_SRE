"""
memory/models.py — SQLAlchemy ORM models + Pydantic schemas for the memory service.

Tables: episodes, decisions, causal_edges, lessons
Schema matches init.sql exactly.
"""
from __future__ import annotations

from pydantic import ConfigDict

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from memory.db import Base


# ── ORM Models ────────────────────────────────────────────────────────────────

class Episode(Base):
    __tablename__ = "episodes"

    episode_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id: Mapped[str] = mapped_column(String, nullable=False)
    agent_version: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    final_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    resolution_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    decisions: Mapped[List["Decision"]] = relationship(back_populates="episode")


class Decision(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    episode_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("episodes.episode_id"), nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    state_signature: Mapped[str] = mapped_column(Text, nullable=False)
    state_embedding: Mapped[Optional[Any]] = mapped_column(Vector(384), nullable=True)
    action_type: Mapped[str] = mapped_column(String, nullable=False)
    action_payload: Mapped[Dict] = mapped_column(JSONB, nullable=False, default=dict)
    result_stdout: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_stderr: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    credit_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # backfilled
    quarantine_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    no_match_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    quarantine_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    episode: Mapped["Episode"] = relationship(back_populates="decisions")
    outgoing_edges: Mapped[List["CausalEdge"]] = relationship(
        foreign_keys="CausalEdge.from_decision",
        back_populates="from_decision_rel",
    )
    incoming_edges: Mapped[List["CausalEdge"]] = relationship(
        foreign_keys="CausalEdge.to_decision",
        back_populates="to_decision_rel",
    )


class CausalEdge(Base):
    __tablename__ = "causal_edges"

    edge_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    from_decision: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("decisions.decision_id"), nullable=False)
    to_decision: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("decisions.decision_id"), nullable=False)
    relation_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # caused/preceded/remediated_by
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    last_reinforced: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    from_decision_rel: Mapped["Decision"] = relationship(foreign_keys=[from_decision], back_populates="outgoing_edges")
    to_decision_rel: Mapped["Decision"] = relationship(foreign_keys=[to_decision], back_populates="incoming_edges")


class Lesson(Base):
    __tablename__ = "lessons"

    lesson_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    state_cluster_embedding: Mapped[Optional[Any]] = mapped_column(Vector(384), nullable=True)
    best_action: Mapped[Optional[Dict]] = mapped_column(JSONB, nullable=True)
    outcome_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sample_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    success_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_updated: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


# ── Pydantic schemas (for validation / serialization) ────────────────────────

class EpisodeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    episode_id: str
    task_id: str
    agent_version: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    outcome: Optional[str] = None
    final_reward: Optional[float] = None
    resolution_summary: Optional[str] = None


class DecisionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    decision_id: str
    episode_id: str
    step_index: int
    state_signature: str
    action_type: str
    action_payload: Dict[str, Any]
    result_stdout: Optional[str] = None
    result_stderr: Optional[str] = None
    exit_code: Optional[int] = None
    credit_label: Optional[str] = None
    quarantine_flag: bool = False
    no_match_flag: bool = False
    quarantine_reason: Optional[str] = None


class LessonSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    lesson_id: str
    best_action: Optional[Dict[str, Any]] = None
    outcome_summary: Optional[str] = None
    sample_count: Optional[int] = None
    success_rate: Optional[float] = None
    similarity: Optional[float] = None  # set by retriever, not stored
