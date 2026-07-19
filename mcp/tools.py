"""
mcp/tools.py — MCP tool wrappers around mock_infra actions.

Exposes 4 tools to the agent reasoning loop:
  - diagnostic_query(service, metric)
  - log_inspection(service, time_window_minutes)
  - remediation(action_type, target, params)
  - submit_resolution(summary)

Each tool validates typed inputs before dispatching. The remediation tool
passes through the Quarantine gate before touching mock_infra. No behavioral
logic lives here — this is the interface layer only.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from mock_infra.mesh import SERVICES


# ── Input schemas ─────────────────────────────────────────────────────────────

VALID_METRICS = frozenset([
    "p99_latency_ms", "error_rate_pct", "saturation_pct",
    "all",  # returns full golden signals
])

VALID_ACTION_TYPES = frozenset([
    "restart_service",
    "scale_up",
    "rollback",
    "increase_db_pool",
    "kill_slow_queries",
    "vacuum_analyze",
    "circuit_breaker",
])


class DiagnosticQueryInput(BaseModel):
    service: str = Field(..., description="Target service name")
    metric: str = Field(default="all", description="Metric name or 'all'")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        if v not in SERVICES:
            raise ValueError(f"Unknown service '{v}'. Valid: {sorted(SERVICES)}")
        return v

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        if v not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{v}'. Valid: {sorted(VALID_METRICS)}")
        return v


class LogInspectionInput(BaseModel):
    service: str = Field(..., description="Target service name")
    time_window_minutes: int = Field(default=5, ge=1, le=60, description="Log window in minutes")

    @field_validator("service")
    @classmethod
    def validate_service(cls, v: str) -> str:
        if v not in SERVICES:
            raise ValueError(f"Unknown service '{v}'. Valid: {sorted(SERVICES)}")
        return v


class RemediationInput(BaseModel):
    action_type: str = Field(..., description="Remediation action type")
    target: str = Field(..., description="Target service or resource")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        if v not in VALID_ACTION_TYPES:
            raise ValueError(f"Unknown action_type '{v}'. Valid: {sorted(VALID_ACTION_TYPES)}")
        return v


class SubmitResolutionInput(BaseModel):
    summary: str = Field(..., min_length=10, description="Resolution summary (min 10 chars)")


# ── Tool result schema ────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    tool: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    quarantine_blocked: bool = False
    quarantine_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


# ── MCP Tool dispatcher ───────────────────────────────────────────────────────

class MCPTools:
    """
    Stateful tool dispatcher. Holds references to the mesh/telemetry/db instances
    for the current episode. A fresh MCPTools is constructed each episode.
    The quarantine_gate is injected at construction time.
    """

    def __init__(self, mesh, telemetry, mock_db, quarantine_gate) -> None:
        self._mesh = mesh
        self._telemetry = telemetry
        self._db = mock_db
        self._quarantine = quarantine_gate
        self._resolution_submitted: bool = False

    # ── Tool: diagnostic_query ────────────────────────────────────────────────

    def diagnostic_query(self, service: str, metric: str = "all") -> ToolResult:
        """Query current golden signals for a service."""
        try:
            inp = DiagnosticQueryInput(service=service, metric=metric)
        except Exception as exc:
            return ToolResult(tool="diagnostic_query", success=False, error=str(exc))

        obs = self._telemetry.full_observation(service=inp.service)
        if inp.metric != "all" and "metrics" in obs and obs["metrics"]:
            data = {inp.metric: obs["metrics"].get(inp.metric)}
        else:
            data = obs
        return ToolResult(tool="diagnostic_query", success=True, data=data)

    # ── Tool: log_inspection ──────────────────────────────────────────────────

    def log_inspection(self, service: str, time_window_minutes: int = 5) -> ToolResult:
        """Inspect logs for a service over a time window."""
        try:
            inp = LogInspectionInput(service=service, time_window_minutes=time_window_minutes)
        except Exception as exc:
            return ToolResult(tool="log_inspection", success=False, error=str(exc))

        logs = self._telemetry.collect_logs(inp.service, inp.time_window_minutes)
        traces = self._telemetry.collect_traces(inp.service)
        return ToolResult(
            tool="log_inspection",
            success=True,
            data={"logs": logs, "traces": traces, "service": inp.service},
        )

    # ── Tool: remediation ─────────────────────────────────────────────────────

    def remediation(
        self, action_type: str, target: str, params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Apply a remediation action. Passes through the Quarantine gate first.
        If blocked, returns quarantine_blocked=True with reason.
        """
        params = params or {}
        try:
            inp = RemediationInput(action_type=action_type, target=target, params=params)
        except Exception as exc:
            return ToolResult(tool="remediation", success=False, error=str(exc))

        # ── Quarantine gate check (FROZEN logic) ──────────────────────────────
        allowed, reason = self._quarantine.check(
            action_type=inp.action_type,
            target=inp.target,
            params=inp.params,
        )
        if not allowed:
            return ToolResult(
                tool="remediation",
                success=False,
                quarantine_blocked=True,
                quarantine_reason=reason,
            )

        # ── Dispatch to mock_infra ────────────────────────────────────────────
        # DB-level actions route to mock_db, mesh-level to mesh
        db_actions = {"increase_db_pool", "kill_slow_queries", "vacuum_analyze"}
        if inp.action_type in db_actions:
            result = self._db.apply_remediation(inp.action_type, inp.params)
            # Also update mesh state for affected service
            if inp.action_type == "increase_db_pool" and inp.target in self._mesh.get_services():
                self._mesh.apply_remediation(inp.action_type, inp.target, inp.params)
        else:
            result = self._mesh.apply_remediation(inp.action_type, inp.target, inp.params)

        return ToolResult(
            tool="remediation",
            success=result.get("success", False),
            data=result,
            error=result.get("error"),
        )

    # ── Tool: submit_resolution ───────────────────────────────────────────────

    def submit_resolution(self, summary: str) -> ToolResult:
        """Submit episode resolution. Triggers FSM transition to DONE."""
        try:
            inp = SubmitResolutionInput(summary=summary)
        except Exception as exc:
            return ToolResult(tool="submit_resolution", success=False, error=str(exc))

        if self._resolution_submitted:
            return ToolResult(
                tool="submit_resolution",
                success=False,
                error="Resolution already submitted for this episode.",
            )

        self._resolution_submitted = True
        return ToolResult(
            tool="submit_resolution",
            success=True,
            data={"summary": inp.summary, "message": "Episode resolution submitted."},
        )

    @property
    def resolution_submitted(self) -> bool:
        return self._resolution_submitted

    # ── Smoke test ────────────────────────────────────────────────────────────

    @classmethod
    def smoke_test(cls) -> None:
        """Basic smoke test — validates tool schemas and dispatch without a live env."""
        from mock_infra.mesh import MockMesh
        from mock_infra.telemetry import TelemetryCollector
        from mock_infra.mock_db import MockDatabase

        class _FakeGate:
            def check(self, **_):
                return True, ""

        mesh = MockMesh()
        mesh.inject_latency_spike("auth")
        tools = cls(
            mesh=mesh,
            telemetry=TelemetryCollector(mesh),
            mock_db=MockDatabase(),
            quarantine_gate=_FakeGate(),
        )
        r = tools.diagnostic_query("auth")
        assert r.success, f"diagnostic_query failed: {r}"
        r = tools.log_inspection("auth", 3)
        assert r.success, f"log_inspection failed: {r}"
        r = tools.remediation("restart_service", "auth")
        assert r.success, f"remediation failed: {r}"
        print("MCP tools smoke test PASSED")


if __name__ == "__main__":
    MCPTools.smoke_test()
