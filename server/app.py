"""
server/app.py — FastAPI application.

HTTP endpoints:
  POST /episodes/           — start a new episode (runs full pipeline)
  GET  /episodes/{id}       — get episode status
  GET  /health              — health check

WebSocket:
  WS /ws/train              — continuous training mode (episodes stream over WS)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from config import settings
from memory.db import init_db, get_db_session
from server.fsm import EpisodeFSM
from server.pipeline import run_episode, register_task

# Import and register task modules
import tasks.task_1 as task_1
import tasks.task_2 as task_2
import tasks.task_3 as task_3
import tasks.task_4 as task_4

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    log.info("app.startup")
    await init_db()

    # Register tasks
    register_task("task_1", task_1)
    register_task("task_2", task_2)
    register_task("task_3", task_3)
    register_task("task_4", task_4)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("app.shutdown")


app = FastAPI(
    title="Agentic SRE OpenEnv v2",
    description="SRE incident response RL environment with persistent case-based memory.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────

class StartEpisodeRequest(BaseModel):
    task_id: str
    run_id: Optional[str] = None  # optional label for batch runs


class EpisodeResponse(BaseModel):
    episode_id: str
    task_id: str
    outcome: Optional[str] = None
    final_reward: Optional[float] = None
    step_count: Optional[int] = None
    resolution_summary: Optional[str] = None


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


@app.post("/episodes/", response_model=EpisodeResponse)
async def start_episode(req: StartEpisodeRequest) -> EpisodeResponse:
    """
    Run a full episode synchronously. Blocks until the agent resolves or times out.
    For continuous training, use the WebSocket endpoint instead.
    """
    fsm = EpisodeFSM()
    try:
        ctx = await run_episode(task_id=req.task_id, fsm=fsm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.exception("episode.error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    return EpisodeResponse(
        episode_id=ctx.episode_id,
        task_id=ctx.task_id,
        outcome=ctx.outcome,
        final_reward=ctx.final_reward,
        step_count=ctx.step_index,
        resolution_summary=ctx.resolution_summary,
    )


# ── WebSocket: continuous training ────────────────────────────────────────────

@app.websocket("/ws/train")
async def ws_train(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for continuous training mode.
    Client sends: {"task_id": "task_1", "n_episodes": 10}
    Server streams: one JSON result per completed episode.

    Note (per brief §8.1): parallel episodes do NOT share lesson writes mid-batch.
    Consolidation is fully offline/scheduled. Only episodes/decisions write live.
    """
    await websocket.accept()
    log.info("ws.train.connected")

    try:
        raw = await websocket.receive_text()
        req = json.loads(raw)
        task_id = req.get("task_id", "task_1")
        n_episodes = int(req.get("n_episodes", 1))

        for i in range(n_episodes):
            fsm = EpisodeFSM()
            try:
                ctx = await run_episode(task_id=task_id, fsm=fsm)
                result = {
                    "episode_index": i,
                    "episode_id": ctx.episode_id,
                    "task_id": ctx.task_id,
                    "outcome": ctx.outcome,
                    "final_reward": ctx.final_reward,
                    "step_count": ctx.step_index,
                }
            except Exception as exc:
                log.exception("ws.episode.error", error=str(exc))
                result = {"episode_index": i, "error": str(exc)}

            await websocket.send_text(json.dumps(result))

        await websocket.send_text(json.dumps({"status": "done", "total": n_episodes}))

    except WebSocketDisconnect:
        log.info("ws.train.disconnected")
    except Exception as exc:
        log.exception("ws.error", error=str(exc))
        await websocket.close(code=1011)
