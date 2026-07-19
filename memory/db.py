"""
memory/db.py — Connection/session management for the memory Postgres instance.

This is COMPLETELY SEPARATE from the mock_infra mock_db. It connects to the
real Postgres+pgvector instance defined in POSTGRES_URL. The environment
FSM reset() never touches this connection or any tables it manages.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config import settings

log = structlog.get_logger(__name__)

# ── SQLAlchemy base ───────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Engine + session factory (module-level singletons) ───────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.postgres_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for a DB session. Handles commit/rollback."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Schema init ───────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Create all tables if they don't exist. In production, init.sql is run by
    docker-entrypoint-initdb.d; this is a fallback for local dev without Docker.
    """
    from sqlalchemy import text

    engine = get_engine()
    async with engine.begin() as conn:
        # Ensure pgvector extension is available
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create tables via SQLAlchemy metadata
        await conn.run_sync(Base.metadata.create_all)

    log.info("memory.db.init_complete", url=settings.postgres_url)


async def health_check() -> bool:
    """Returns True if the memory DB is reachable."""
    from sqlalchemy import text
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        log.error("memory.db.health_check_failed", error=str(exc))
        return False
