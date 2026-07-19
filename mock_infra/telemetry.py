"""
mock_infra/telemetry.py — Generates structured telemetry (metrics, logs, traces)
from the current mesh state. Returns typed observation dicts consumed by the agent.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

from mock_infra.mesh import MockMesh


# Log templates per fault type
_LOG_TEMPLATES: Dict[str, List[str]] = {
    "latency_spike": [
        "WARN  p99 latency exceeded threshold: {latency:.0f}ms (threshold: 200ms)",
        "ERROR timeout waiting for upstream response after {latency:.0f}ms",
        "WARN  slow consumer detected on queue depth={queue}",
    ],
    "error_spike": [
        "ERROR {method} {path} 502 Bad Gateway upstream={upstream}",
        "ERROR connection refused to {upstream}: max retries exceeded",
        "WARN  circuit breaker half-open: error_rate={rate:.1f}%",
    ],
    "db_pool": [
        "ERROR HikariCP connection pool exhausted (pool_size={pool_size}, waiting={waiting})",
        "WARN  getConnection() timed out after 30000ms",
        "ERROR org.postgresql.util.PSQLException: FATAL: remaining connection slots reserved",
    ],
    "saturation": [
        "WARN  CPU saturation at {sat:.0f}% — shedding requests",
        "WARN  thread pool at capacity: active={active}, queue={queue}",
        "ERROR OOMKilled: container exceeded memory limit",
    ],
    "normal": [
        "INFO  health check OK",
        "INFO  processed {req}/s requests",
        "DEBUG cache hit ratio: {ratio:.2f}",
    ],
}


def _pick_log(category: str, **kwargs) -> str:
    rng = random.Random(int(time.time() * 1000) % 99999)
    template = rng.choice(_LOG_TEMPLATES.get(category, _LOG_TEMPLATES["normal"]))
    defaults = {
        "latency": 250, "queue": 1200, "method": "GET", "path": "/api/v1/users",
        "upstream": "user-service", "rate": 5.0, "pool_size": 10, "waiting": 48,
        "sat": 85, "active": 200, "req": 1200, "ratio": 0.82,
    }
    defaults.update(kwargs)
    try:
        return template.format(**defaults)
    except KeyError:
        return template


class TelemetryCollector:
    """
    Generates structured telemetry observations from the current mesh state.
    All outputs are deterministic given the mesh state (plus small noise).
    """

    def __init__(self, mesh: MockMesh) -> None:
        self._mesh = mesh

    def collect_metrics(self) -> Dict[str, dict]:
        """Return current golden signals for all services."""
        return self._mesh.observe_all()

    def collect_logs(self, service: str, time_window_minutes: int = 5) -> List[str]:
        """
        Return simulated log lines for a service over the requested window.
        Log content reflects active fault state.
        """
        obs = self._mesh.observe_service(service)
        if obs is None:
            return [f"ERROR: unknown service '{service}'"]

        logs: List[str] = []
        faults = obs.get("active_faults", [])
        latency = obs["p99_latency_ms"]
        error_rate = obs["error_rate_pct"]
        sat = obs["saturation_pct"]

        # Add fault-specific log lines
        if any("latency" in f or "timeout" in f for f in faults):
            for _ in range(min(time_window_minutes, 3)):
                logs.append(_pick_log("latency_spike", latency=latency))

        if error_rate > 2.0 or any("error" in f or "cascaded" in f for f in faults):
            for _ in range(min(time_window_minutes, 3)):
                logs.append(_pick_log("error_spike", rate=error_rate, upstream=service))

        if any("db_pool" in f for f in faults):
            for _ in range(2):
                logs.append(_pick_log("db_pool", pool_size=10, waiting=random.randint(30, 60)))

        if sat > 70.0 or any("saturation" in f for f in faults):
            logs.append(_pick_log("saturation", sat=sat, active=200, queue=800))

        # Always pad with some normal logs
        for _ in range(max(1, time_window_minutes - len(logs))):
            logs.append(_pick_log("normal", req=random.randint(800, 1500), ratio=random.uniform(0.7, 0.95)))

        return logs[:time_window_minutes * 3]

    def collect_traces(self, service: str) -> List[dict]:
        """
        Return simulated distributed trace summaries for a service.
        """
        obs = self._mesh.observe_service(service)
        if obs is None:
            return []

        latency = obs["p99_latency_ms"]
        error_rate = obs["error_rate_pct"]
        traces = []
        rng = random.Random(int(time.time() * 100) % 99999)

        for i in range(5):
            duration = latency * rng.uniform(0.5, 1.5)
            is_error = rng.random() < (error_rate / 100.0)
            traces.append({
                "trace_id": f"trace-{service}-{i:04d}",
                "service": service,
                "duration_ms": round(duration, 1),
                "status": "ERROR" if is_error else "OK",
                "spans": rng.randint(2, 8),
            })
        return traces

    def full_observation(self, service: Optional[str] = None) -> dict:
        """
        Full structured observation: metrics + logs + traces.
        If service is None, returns mesh-wide metrics only.
        """
        if service:
            return {
                "metrics": self._mesh.observe_service(service),
                "logs": self.collect_logs(service),
                "traces": self.collect_traces(service),
            }
        return {
            "metrics": self.collect_metrics(),
            "mesh_summary": self._summarize_mesh(),
        }

    def _summarize_mesh(self) -> dict:
        """High-level mesh health summary."""
        metrics = self.collect_metrics()
        unhealthy = []
        for svc, m in metrics.items():
            if m["p99_latency_ms"] > 500 or m["error_rate_pct"] > 5.0 or m["saturation_pct"] > 80.0:
                unhealthy.append(svc)
        return {
            "total_services": len(metrics),
            "unhealthy_services": unhealthy,
            "mesh_healthy": len(unhealthy) == 0,
        }
