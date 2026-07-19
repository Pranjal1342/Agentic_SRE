"""
server/fsm.py — Episode FSM.

States: IDLE → ACTIVE → RESOLVING → DONE

The FSM manages episode lifecycle. Its reset() wipes ALL environment state
(mesh, db, telemetry) every episode. The memory service is COMPLETELY SEPARATE
from this reset boundary — FSM reset() never touches Postgres memory tables.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from config import settings
from mock_infra.mesh import MockMesh
from mock_infra.mock_db import MockDatabase
from mock_infra.telemetry import TelemetryCollector
from agents.quarantine_agent import QuarantineAgent
from mcp.tools import MCPTools

log = structlog.get_logger(__name__)


class FSMState(str, Enum):
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    RESOLVING = "RESOLVING"
    DONE = "DONE"


@dataclass
class EpisodeContext:
    """All per-episode state. Completely wiped on reset()."""
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    started_at: float = field(default_factory=time.time)
    step_index: int = 0
    fsm_state: FSMState = FSMState.IDLE
    outcome: Optional[str] = None         # success / failure / partial
    final_reward: Optional[float] = None
    resolution_summary: Optional[str] = None
    timed_out: bool = False


class EpisodeFSM:
    """
    Manages the FSM state machine for a single episode.
    Owns the environment objects (mesh, db, telemetry, tools).
    Does NOT own or touch the memory service.
    """

    def __init__(self) -> None:
        self._mesh = MockMesh()
        self._db = MockDatabase()
        self._telemetry = TelemetryCollector(self._mesh)
        self._quarantine = QuarantineAgent()
        self._tools: Optional[MCPTools] = None
        self._ctx: Optional[EpisodeContext] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self, task_id: str) -> EpisodeContext:
        """
        Reset ALL environment state. Start a new episode.
        CRITICAL: does not touch memory service. Memory persistence is
        maintained by the pipeline layer, completely outside this method.
        """
        # 1. Wipe environment state
        self._mesh.reset()
        self._db.reset()

        # 2. Create fresh episode context
        self._ctx = EpisodeContext(task_id=task_id)
        self._ctx.fsm_state = FSMState.ACTIVE

        # 3. Wire up fresh MCP tools for this episode
        self._tools = MCPTools(
            mesh=self._mesh,
            telemetry=self._telemetry,
            mock_db=self._db,
            quarantine_gate=self._quarantine,
        )

        log.info("episode.reset", episode_id=self._ctx.episode_id, task_id=task_id)
        return self._ctx

    def step(self) -> bool:
        """
        Advance step index. Returns False if episode should terminate
        (max steps or timeout reached).
        """
        if self._ctx is None:
            raise RuntimeError("FSM not initialized — call reset() first")

        self._ctx.step_index += 1
        elapsed = time.time() - self._ctx.started_at

        if self._ctx.step_index > settings.max_steps_per_episode:
            log.warning("episode.max_steps_reached", episode_id=self._ctx.episode_id)
            self._transition_resolving(outcome="partial")
            return False

        if elapsed > settings.episode_timeout_seconds:
            log.warning("episode.timeout", episode_id=self._ctx.episode_id, elapsed=elapsed)
            self._ctx.timed_out = True
            self._transition_resolving(outcome="failure")
            return False

        return True

    def resolve(self, outcome: str, reward: float, summary: str) -> None:
        """Called when the agent submits a resolution or episode ends."""
        if self._ctx is None:
            return
        self._ctx.outcome = outcome
        self._ctx.final_reward = reward
        self._ctx.resolution_summary = summary
        self._ctx.fsm_state = FSMState.DONE
        log.info(
            "episode.resolved",
            episode_id=self._ctx.episode_id,
            outcome=outcome,
            reward=reward,
        )

    def transition_done(self) -> None:
        if self._ctx:
            self._ctx.fsm_state = FSMState.DONE

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def ctx(self) -> Optional[EpisodeContext]:
        return self._ctx

    @property
    def tools(self) -> Optional[MCPTools]:
        return self._tools

    @property
    def mesh(self) -> MockMesh:
        return self._mesh

    @property
    def db(self) -> MockDatabase:
        return self._db

    @property
    def telemetry(self) -> TelemetryCollector:
        return self._telemetry

    @property
    def is_active(self) -> bool:
        return self._ctx is not None and self._ctx.fsm_state == FSMState.ACTIVE

    @property
    def is_done(self) -> bool:
        return self._ctx is not None and self._ctx.fsm_state == FSMState.DONE

    # ── Internal transitions ──────────────────────────────────────────────────

    def _transition_resolving(self, outcome: str) -> None:
        if self._ctx:
            self._ctx.fsm_state = FSMState.RESOLVING
            self._ctx.outcome = outcome
