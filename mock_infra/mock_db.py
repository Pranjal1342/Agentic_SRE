"""
mock_infra/mock_db.py — Simulates database (Postgres-like) backend for the
microservice mesh. Tracks connection pool state, query latency, and can simulate
pool exhaustion for task_4. Completely separate from the memory service Postgres.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MockDBState:
    pool_size: int = 20
    active_connections: int = 3
    waiting_requests: int = 0
    query_p99_ms: float = 15.0
    pool_exhausted: bool = False
    slow_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "active_connections": self.active_connections,
            "waiting_requests": self.waiting_requests,
            "available_connections": max(0, self.pool_size - self.active_connections),
            "query_p99_ms": round(self.query_p99_ms, 1),
            "pool_exhausted": self.pool_exhausted,
            "slow_queries": list(self.slow_queries),
            "utilization_pct": round(self.active_connections / self.pool_size * 100, 1),
        }


class MockDatabase:
    """
    Simulated database backend. Tracks connection pool, query latency.
    reset() restores healthy state — called by FSM reset() each episode.
    """

    def __init__(self) -> None:
        self._state = MockDBState()
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore healthy DB state. Called by FSM reset()."""
        self._state = MockDBState(
            pool_size=20,
            active_connections=3,
            waiting_requests=0,
            query_p99_ms=15.0,
            pool_exhausted=False,
            slow_queries=[],
        )

    # ── Observation ───────────────────────────────────────────────────────────

    def observe(self) -> dict:
        """Return current DB state with small noise."""
        rng = random.Random(int(time.time() * 1000) % 99999)
        state = self._state.to_dict()
        state["query_p99_ms"] = max(1.0, state["query_p99_ms"] + rng.gauss(0, 0.5))
        return state

    def get_slow_queries(self) -> List[dict]:
        """Return simulated slow query report."""
        if not self._state.pool_exhausted and self._state.query_p99_ms < 100:
            return []
        rng = random.Random(int(time.time() * 100) % 99999)
        return [
            {
                "query": q,
                "duration_ms": round(self._state.query_p99_ms * rng.uniform(1.0, 3.0), 1),
                "calls": rng.randint(50, 500),
            }
            for q in self._state.slow_queries[:5]
        ]

    # ── Fault injection ───────────────────────────────────────────────────────

    def inject_pool_exhaustion(self, waiting: int = 48) -> None:
        """Task 4 — simulate DB connection pool exhaustion."""
        self._state.active_connections = self._state.pool_size  # all slots taken
        self._state.waiting_requests = waiting
        self._state.query_p99_ms = 8000.0  # queries pile up
        self._state.pool_exhausted = True
        self._state.slow_queries = [
            "SELECT * FROM users WHERE id = $1 FOR UPDATE",
            "UPDATE sessions SET last_active = NOW() WHERE token = $1",
            "INSERT INTO audit_log (user_id, action, ts) VALUES ($1, $2, $3)",
        ]

    def inject_slow_queries(self, p99_ms: float = 2000.0) -> None:
        """Inject slow query degradation."""
        self._state.query_p99_ms = p99_ms
        self._state.slow_queries = [
            "SELECT count(*) FROM orders o JOIN order_items oi ON o.id = oi.order_id",
            "SELECT * FROM products WHERE category IN (SELECT id FROM categories WHERE active=true)",
        ]

    # ── Remediation ───────────────────────────────────────────────────────────

    def apply_remediation(self, action_type: str, params: dict) -> dict:
        """Apply DB-level remediation. Called after Quarantine gate approval."""
        result: dict = {"success": False, "action": action_type}

        if action_type == "increase_db_pool":
            new_size = int(params.get("pool_size", 50))
            self._state.pool_size = new_size
            self._state.active_connections = min(self._state.active_connections, new_size)
            self._state.waiting_requests = 0
            self._state.pool_exhausted = False
            self._state.query_p99_ms = 20.0
            result.update({"success": True, "new_pool_size": new_size})

        elif action_type == "kill_slow_queries":
            self._state.slow_queries.clear()
            self._state.query_p99_ms = 15.0
            result.update({"success": True, "effect": "slow_queries_terminated"})

        elif action_type == "vacuum_analyze":
            self._state.query_p99_ms = max(10.0, self._state.query_p99_ms * 0.5)
            result.update({"success": True, "effect": "vacuum_analyze_ran"})

        else:
            result["error"] = f"Unknown DB action: {action_type}"

        return result
