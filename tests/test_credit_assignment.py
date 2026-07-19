"""
tests/test_credit_assignment.py — Unit tests for credit assignment logic.

Uses a mocked session to test label assignment without a live DB.
"""
import pytest
from unittest.mock import MagicMock
from memory.credit_assignment import _assign_label
from memory.models import Decision


def _make_decision(**kwargs) -> Decision:
    """Factory for Decision with defaults."""
    d = Decision(
        decision_id="test-id",
        episode_id="ep-id",
        step_index=kwargs.get("step_index", 0),
        state_signature="state",
        action_type=kwargs.get("action_type", "diagnostic_query"),
        action_payload={},
        credit_label=None,
        quarantine_flag=kwargs.get("quarantine_flag", False),
        quarantine_reason=kwargs.get("quarantine_reason", None),
    )
    return d


def test_quarantine_blocked_gets_quarantine_label():
    d = _make_decision(
        action_type="restart_service",
        quarantine_flag=True,
        quarantine_reason="action_type 'restart_service' is not allowed on payment-service",
    )
    label = _assign_label(d, outcome="failure", step_position=2, total_steps=5)
    assert label == "quarantine_blocked"


def test_diagnostic_action_always_neutral():
    for action in ("diagnostic_query", "log_inspection"):
        d = _make_decision(action_type=action)
        for outcome in ("success", "failure", "partial"):
            label = _assign_label(d, outcome=outcome, step_position=0, total_steps=3)
            assert label == "neutral", f"{action} should always be neutral, got {label}"


def test_remediation_in_success_episode_gets_success():
    d = _make_decision(action_type="restart_service")
    label = _assign_label(d, outcome="success", step_position=1, total_steps=3)
    assert label == "success"


def test_remediation_in_failure_episode_gets_failure():
    d = _make_decision(action_type="scale_up")
    label = _assign_label(d, outcome="failure", step_position=0, total_steps=3)
    assert label == "failure"


def test_last_step_in_partial_episode_gets_neutral():
    d = _make_decision(action_type="restart_service")
    # Last step (position = total_steps - 1)
    label = _assign_label(d, outcome="partial", step_position=4, total_steps=5)
    assert label == "neutral"


def test_early_step_in_partial_episode_gets_failure():
    d = _make_decision(action_type="restart_service")
    # Early step
    label = _assign_label(d, outcome="partial", step_position=0, total_steps=5)
    assert label == "failure"


def test_submit_resolution_success():
    d = _make_decision(action_type="submit_resolution")
    label = _assign_label(d, outcome="success", step_position=5, total_steps=6)
    assert label == "success"


def test_submit_resolution_failure():
    d = _make_decision(action_type="submit_resolution")
    label = _assign_label(d, outcome="failure", step_position=5, total_steps=6)
    assert label == "failure"
