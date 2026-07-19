"""
tasks/task_2.py — Cascading error rate increase.

Fault: api-gateway starts returning 502s; errors cascade to user-service and
payment-service. Root cause: api-gateway misconfiguration causing upstream failures.
The correct remediation sequence:
  1. Observe mesh-wide error rates
  2. Identify api-gateway as origin (higher error rate than downstream)
  3. Rollback or restart api-gateway
  4. Verify cascade clears
"""
from typing import Any, Dict


TASK_ID = "task_2"
DESCRIPTION = "Cascading error rate increase originating at api-gateway"


def inject_fault(mesh, db) -> None:
    """Inject cascading error rate from api-gateway."""
    mesh.inject_cascading_errors(origin_service="api-gateway", spread_factor=0.3)


def golden_signal_targets() -> Dict[str, Dict[str, float]]:
    """All services must return to near-healthy error rates."""
    return {
        "api-gateway": {
            "error_rate_pct": 1.0,
            "p99_latency_ms": 200.0,
        },
        "user-service": {
            "error_rate_pct": 0.5,
        },
        "payment-service": {
            "error_rate_pct": 1.0,
        },
    }


def reset(mesh, db) -> None:
    mesh.reset()
    db.reset()
