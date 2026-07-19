"""
main.py — Application entrypoint.

Starts:
1. FastAPI app (HTTP + WebSocket)
2. APScheduler background job for offline consolidation

The consolidation job is fully offline/scheduled — it does NOT run during episodes.
Parallel episodes do NOT see each other's lesson writes mid-batch (per §8.1).
"""
from __future__ import annotations

import asyncio
import os
import sys

import structlog
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from memory.consolidate import run_consolidation
from server.app import app

log = structlog.get_logger(__name__)


def setup_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


async def consolidation_job() -> None:
    """Wrapper for the APScheduler async job."""
    try:
        stats = await run_consolidation()
        log.info("scheduler.consolidation_complete", **stats)
    except Exception as exc:
        log.exception("scheduler.consolidation_error", error=str(exc))


def main() -> None:
    setup_logging()

    # Set up APScheduler for offline consolidation
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        consolidation_job,
        "interval",
        minutes=settings.consolidation_interval_minutes,
        id="consolidation",
        replace_existing=True,
    )

    log.info(
        "app.starting",
        version=settings.agent_version,
        consolidation_interval_minutes=settings.consolidation_interval_minutes,
    )

    # Wire scheduler start/stop into app lifespan
    @app.on_event("startup")
    async def start_scheduler():
        scheduler.start()
        log.info("scheduler.started")

    @app.on_event("shutdown")
    async def stop_scheduler():
        scheduler.shutdown(wait=False)
        log.info("scheduler.stopped")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )


if __name__ == "__main__":
    main()
