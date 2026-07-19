"""
mock_infra/mesh.py — Simulated microservice mesh.

Four services: auth, api-gateway, user-service, payment-service.
Each service tracks golden signals (p99 latency ms, error rate %, saturation %).
Fault injection methods support the 4 task scenarios.
reset() restores healthy baselines — called by FSM reset(), completely isolated
from the memory service.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


SERVICES = ["auth", "api-gateway", "user-service", "payment-service"]

# Healthy baseline golden signals
_HEALTHY_BASELINE: Dict[str, Dict[str, float]] = {
    "auth": {"p99_latency_ms": 45.0, "error_rate_pct": 0.1, "saturation_pct": 20.0},
    "api-gateway": {"p99_latency_ms": 80.0, "error_rate_pct": 0.2, "saturation_pct": 30.0},
    "user-service": {"p99_latency_ms": 60.0, "error_rate_pct": 0.1, "saturation_pct": 25.0},
    "payment-service": {"p99_latency_ms": 120.0, "error_rate_pct": 0.3, "saturation_pct": 35.0},
}


@dataclass
class ServiceState:
    name: str
    p99_latency_ms: float
    error_rate_pct: float
    saturation_pct: float
    # Active faults injected into this service
    active_faults: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "service": self.name,
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "error_rate_pct": round(self.error_rate_pct, 3),
            "saturation_pct": round(self.saturation_pct, 1),
            "active_faults": list(self.active_faults),
        }


class MockMesh:
    """
    Simulated microservice mesh. Stateful per-episode; reset() must be called
    at the start of each episode by the FSM. No memory service interaction here.
    """

    def __init__(self) -> None:
        self._states: Dict[str, ServiceState] = {}
        self._noise_seed: int = 0
        self.reset()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Restore all services to healthy baseline. Called by FSM reset()."""
        self._states = {
            name: ServiceState(name=name, **copy.deepcopy(vals))
            for name, vals in _HEALTHY_BASELINE.items()
        }
        self._noise_seed = int(time.time() * 1000) % 10_000

    # ── Observation ───────────────────────────────────────────────────────────

    def observe_all(self) -> Dict[str, dict]:
        """Return current golden signals for all services with small Gaussian noise."""
        rng = random.Random(self._noise_seed)
        self._noise_seed = (self._noise_seed + 1) % 100_000
        result = {}
        for name, s in self._states.items():
            result[name] = {
                "p99_latency_ms": max(0.0, s.p99_latency_ms + rng.gauss(0, 2.0)),
                "error_rate_pct": max(0.0, s.error_rate_pct + rng.gauss(0, 0.02)),
                "saturation_pct": max(0.0, min(100.0, s.saturation_pct + rng.gauss(0, 0.5))),
                "active_faults": list(s.active_faults),
            }
        return result

    def observe_service(self, service: str) -> Optional[dict]:
        """Return current golden signals for a single service."""
        if service not in self._states:
            return None
        all_obs = self.observe_all()
        return all_obs[service]

    # ── Fault injection ───────────────────────────────────────────────────────

    def inject_latency_spike(self, service: str, multiplier: float = 5.0) -> None:
        """Task 1 — single-service latency spike."""
        if service in self._states:
            s = self._states[service]
            s.p99_latency_ms *= multiplier
            s.saturation_pct = min(100.0, s.saturation_pct + 20.0)
            fault = f"latency_spike(x{multiplier})"
            if fault not in s.active_faults:
                s.active_faults.append(fault)

    def inject_cascading_errors(self, origin_service: str, spread_factor: float = 0.3) -> None:
        """Task 2 — error rate increase that cascades downstream."""
        if origin_service not in self._states:
            return
        self._states[origin_service].error_rate_pct = min(
            100.0, self._states[origin_service].error_rate_pct + 15.0
        )
        self._states[origin_service].active_faults.append("error_spike_origin")
        # Cascade to connected services
        for name, s in self._states.items():
            if name != origin_service:
                s.error_rate_pct = min(100.0, s.error_rate_pct + spread_factor * 15.0)
                s.active_faults.append(f"cascaded_errors(from={origin_service})")

    def inject_multi_stage_fault(self) -> None:
        """
        Task 3 — multi-stage causal decomposition.
        auth slow → api-gateway queues → user-service errors → payment-service saturation.
        This is the hard case (baseline 0.6190).
        """
        self._states["auth"].p99_latency_ms *= 8.0
        self._states["auth"].active_faults.append("auth_latency_stage1")

        self._states["api-gateway"].p99_latency_ms *= 4.0
        self._states["api-gateway"].saturation_pct = min(100.0, self._states["api-gateway"].saturation_pct + 40.0)
        self._states["api-gateway"].active_faults.append("queue_buildup_stage2")

        self._states["user-service"].error_rate_pct = min(100.0, self._states["user-service"].error_rate_pct + 12.0)
        self._states["user-service"].active_faults.append("upstream_timeout_errors_stage3")

        self._states["payment-service"].saturation_pct = min(100.0, self._states["payment-service"].saturation_pct + 50.0)
        self._states["payment-service"].active_faults.append("saturation_stage4")

    def inject_db_pool_exhaustion(self, affected_service: str = "user-service") -> None:
        """Task 4 — DB connection pool exhaustion."""
        if affected_service in self._states:
            s = self._states[affected_service]
            s.p99_latency_ms *= 6.0
            s.error_rate_pct = min(100.0, s.error_rate_pct + 8.0)
            s.saturation_pct = min(100.0, s.saturation_pct + 60.0)
            s.active_faults.append("db_pool_exhausted")

    # ── Remediation actions ───────────────────────────────────────────────────

    def apply_remediation(self, action_type: str, target: str, params: dict) -> dict:
        """
        Apply a remediation action to a service. Returns result dict.
        This is called AFTER the Quarantine gate has approved the action.
        """
        if target not in self._states:
            return {"success": False, "error": f"Unknown service: {target}"}

        s = self._states[target]
        result: dict = {"success": False, "service": target, "action": action_type}

        if action_type == "restart_service":
            # Full restart: reset to healthy baseline
            baseline = _HEALTHY_BASELINE[target]
            s.p99_latency_ms = baseline["p99_latency_ms"]
            s.error_rate_pct = baseline["error_rate_pct"]
            s.saturation_pct = baseline["saturation_pct"]
            s.active_faults.clear()
            result.update({"success": True, "effect": "service_restored_to_baseline"})

        elif action_type == "scale_up":
            factor = float(params.get("factor", 1.5))
            s.saturation_pct = max(0.0, s.saturation_pct / factor)
            s.p99_latency_ms = max(10.0, s.p99_latency_ms * 0.7)
            result.update({"success": True, "effect": f"saturation_reduced_by_{factor}x"})

        elif action_type == "rollback":
            # Rollback to healthy baseline
            baseline = _HEALTHY_BASELINE[target]
            s.p99_latency_ms = baseline["p99_latency_ms"]
            s.error_rate_pct = baseline["error_rate_pct"]
            s.saturation_pct = baseline["saturation_pct"]
            s.active_faults = [f for f in s.active_faults if "stage" not in f]
            result.update({"success": True, "effect": "rolled_back_to_last_stable"})

        elif action_type == "increase_db_pool":
            pool_size = int(params.get("pool_size", 50))
            s.saturation_pct = max(0.0, s.saturation_pct - 50.0)
            s.p99_latency_ms = max(10.0, s.p99_latency_ms / 4.0)
            s.error_rate_pct = max(0.0, s.error_rate_pct - 6.0)
            s.active_faults = [f for f in s.active_faults if "db_pool" not in f]
            result.update({"success": True, "effect": f"db_pool_increased_to_{pool_size}"})

        elif action_type == "circuit_breaker":
            # Prevents cascading but doesn't fix root cause
            s.error_rate_pct = max(0.0, s.error_rate_pct - 5.0)
            result.update({"success": True, "effect": "circuit_breaker_opened"})

        else:
            result["error"] = f"Unknown action_type: {action_type}"

        return result

    def get_services(self) -> List[str]:
        return list(self._states.keys())
