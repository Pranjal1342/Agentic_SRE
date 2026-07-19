"""
tests/test_mesh.py — Unit tests for mock_infra.mesh

Tests fault injection, observation, and remediation.
These tests are independent of Postgres/memory service.
"""
import pytest
from mock_infra.mesh import MockMesh, SERVICES


def test_healthy_baseline():
    """Fresh mesh should have healthy golden signals."""
    mesh = MockMesh()
    obs = mesh.observe_all()
    assert set(obs.keys()) == set(SERVICES)
    for svc, m in obs.items():
        assert m["p99_latency_ms"] < 200, f"{svc}: baseline latency too high"
        assert m["error_rate_pct"] < 1.0, f"{svc}: baseline error rate too high"
        assert m["saturation_pct"] < 50.0, f"{svc}: baseline saturation too high"


def test_reset_restores_baseline():
    """reset() must fully restore healthy state after fault injection."""
    mesh = MockMesh()
    mesh.inject_latency_spike("auth", multiplier=10.0)
    assert mesh.observe_service("auth")["p99_latency_ms"] > 200

    mesh.reset()
    obs = mesh.observe_service("auth")
    assert obs["p99_latency_ms"] < 200
    assert obs["active_faults"] == []


def test_task1_fault_injection():
    """Task 1: auth latency spike should affect only auth."""
    mesh = MockMesh()
    mesh.inject_latency_spike("auth", multiplier=5.0)
    auth = mesh.observe_service("auth")
    assert auth["p99_latency_ms"] > 200
    assert "latency_spike(x5.0)" in auth["active_faults"]

    # Other services should be healthy
    gw = mesh.observe_service("api-gateway")
    assert gw["p99_latency_ms"] < 200


def test_task2_cascading_errors():
    """Task 2: cascading errors from api-gateway should spread to other services."""
    mesh = MockMesh()
    mesh.inject_cascading_errors("api-gateway", spread_factor=0.3)
    gw = mesh.observe_service("api-gateway")
    assert gw["error_rate_pct"] > 5.0

    # Should cascade
    us = mesh.observe_service("user-service")
    assert us["error_rate_pct"] > 0.5


def test_task3_multi_stage_fault():
    """Task 3: all 4 services should be degraded in causal order."""
    mesh = MockMesh()
    mesh.inject_multi_stage_fault()

    auth = mesh.observe_service("auth")
    assert auth["p99_latency_ms"] > 300, "auth should be very slow"

    gw = mesh.observe_service("api-gateway")
    assert gw["saturation_pct"] > 50, "gateway should be saturated"

    us = mesh.observe_service("user-service")
    assert us["error_rate_pct"] > 5.0, "user-service should have errors"

    ps = mesh.observe_service("payment-service")
    assert ps["saturation_pct"] > 50, "payment-service should be saturated"


def test_task4_db_pool_exhaustion():
    """Task 4: db pool exhaustion should degrade user-service."""
    mesh = MockMesh()
    mesh.inject_db_pool_exhaustion("user-service")
    us = mesh.observe_service("user-service")
    assert us["p99_latency_ms"] > 300
    assert us["saturation_pct"] > 50
    assert "db_pool_exhausted" in us["active_faults"]


def test_remediation_restart():
    """restart_service should restore to baseline."""
    mesh = MockMesh()
    mesh.inject_latency_spike("auth", multiplier=5.0)
    result = mesh.apply_remediation("restart_service", "auth", {})
    assert result["success"] is True
    obs = mesh.observe_service("auth")
    assert obs["p99_latency_ms"] < 100


def test_remediation_unknown_service():
    """Remediation on unknown service should fail gracefully."""
    mesh = MockMesh()
    result = mesh.apply_remediation("restart_service", "nonexistent", {})
    assert result["success"] is False
    assert "Unknown service" in result["error"]
