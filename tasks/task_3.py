"""
tasks/task_3.py — Multi-stage causal decomposition.

This is the HARD task. Baseline score: 0.6190. Memory+retrieval is specifically
intended to improve this.

Fault chain (4 stages, each masking the previous):
  Stage 1: auth slow (8x latency) — primary root cause
  Stage 2: api-gateway queue buildup + saturation (consequence of auth slowdown)
  Stage 3: user-service errors from upstream timeouts (consequence of gateway saturation)
  Stage 4: payment-service saturation (consequence of user-service errors)

The correct remediation sequence (causal, not symptomatic):
  1. Observe all 4 services
  2. Identify auth as root cause (highest latency, stage 1 fault)
  3. Fix auth FIRST (restart or scale_up)
  4. Verify gateway queue clears
  5. Verify user-service errors clear
  6. Verify payment-service saturation resolves
  7. Only then submit resolution

WRONG approach (common failure that boosts the baseline to 0.619 but not 1.0):
  - Remediating downstream services (scale_up gateway, rollback user-service)
    without fixing auth first causes incomplete remediation.
  - Payment-service restart is blocked by the Quarantine gate (high-risk action
    on protected service) — this is a deliberate test of gate interaction.
"""
from typing import Any, Dict


TASK_ID = "task_3"
DESCRIPTION = "Multi-stage causal decomposition: auth→gateway→user-service→payment"


def inject_fault(mesh, db) -> None:
    """Inject 4-stage causal fault chain."""
    mesh.inject_multi_stage_fault()


def golden_signal_targets() -> Dict[str, Dict[str, float]]:
    """All 4 services must be restored. Tight targets for full credit."""
    return {
        "auth": {
            "p99_latency_ms": 100.0,
            "error_rate_pct": 0.5,
            "saturation_pct": 30.0,
        },
        "api-gateway": {
            "p99_latency_ms": 200.0,
            "saturation_pct": 50.0,
        },
        "user-service": {
            "error_rate_pct": 1.0,
        },
        "payment-service": {
            "saturation_pct": 60.0,
        },
    }


def reset(mesh, db) -> None:
    mesh.reset()
    db.reset()
