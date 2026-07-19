"""
tasks/task_1.py — Single-service latency spike.

Fault: auth service p99 latency spikes 5x due to upstream dependency timeout.
The correct remediation sequence:
  1. Diagnose auth golden signals
  2. Inspect auth logs to confirm timeout errors
  3. Restart or scale_up auth service
"""
from typing import Any, Dict


TASK_ID = "task_1"
DESCRIPTION = "Single-service latency spike on auth service"


def inject_fault(mesh, db) -> None:
    """Inject a latency spike into the auth service."""
    mesh.inject_latency_spike(service="auth", multiplier=5.0)


def golden_signal_targets() -> Dict[str, Dict[str, float]]:
    """
    Target golden signals that must be restored for full reward.
    Only auth is scored — other services are expected to remain healthy.
    """
    return {
        "auth": {
            "p99_latency_ms": 100.0,    # ≤100ms (healthy baseline: 45ms)
            "error_rate_pct": 1.0,       # ≤1%
            "saturation_pct": 40.0,      # ≤40%
        }
    }


def reset(mesh, db) -> None:
    """Reset environment state. Called by FSM reset() — do not call manually."""
    mesh.reset()
    db.reset()
