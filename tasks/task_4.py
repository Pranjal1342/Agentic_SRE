"""
tasks/task_4.py — DB connection pool exhaustion.

Fault: user-service DB connection pool is exhausted (all 20 slots taken,
48 requests waiting). This manifests as latency spikes and error rate
increase on user-service.

The correct remediation sequence:
  1. Diagnose user-service (high latency + errors)
  2. Inspect user-service logs (see HikariCP pool exhaustion messages)
  3. Identify DB pool exhaustion (not a service bug)
  4. Apply increase_db_pool remediation (target=user-service, pool_size=50)
  5. Verify user-service recovers
"""
from typing import Any, Dict


TASK_ID = "task_4"
DESCRIPTION = "DB connection pool exhaustion on user-service"


def inject_fault(mesh, db) -> None:
    """Inject DB pool exhaustion into user-service and the mock DB."""
    db.inject_pool_exhaustion(waiting=48)
    mesh.inject_db_pool_exhaustion(affected_service="user-service")


def golden_signal_targets() -> Dict[str, Dict[str, float]]:
    return {
        "user-service": {
            "p99_latency_ms": 200.0,
            "error_rate_pct": 1.0,
            "saturation_pct": 50.0,
        }
    }


def reset(mesh, db) -> None:
    mesh.reset()
    db.reset()
