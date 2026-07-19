"""
tests/test_reward.py — Unit tests for the graders/reward.py dense reward function.
"""
import pytest
from graders.reward import compute_reward, reward_breakdown


def _make_targets():
    return {
        "auth": {
            "p99_latency_ms": 100.0,
            "error_rate_pct": 1.0,
            "saturation_pct": 40.0,
        }
    }


def test_perfect_restoration():
    """Metrics exactly at target → reward = 1.0."""
    targets = _make_targets()
    metrics = {
        "auth": {
            "p99_latency_ms": 45.0,
            "error_rate_pct": 0.1,
            "saturation_pct": 20.0,
        }
    }
    reward = compute_reward(metrics, targets)
    assert reward == 1.0, f"Expected 1.0, got {reward}"


def test_complete_failure():
    """Metrics far beyond max → reward near 0."""
    targets = _make_targets()
    metrics = {
        "auth": {
            "p99_latency_ms": 10000.0,
            "error_rate_pct": 100.0,
            "saturation_pct": 100.0,
        }
    }
    reward = compute_reward(metrics, targets)
    assert reward < 0.1, f"Expected near 0, got {reward}"


def test_partial_restoration():
    """Partially restored metrics should give intermediate reward."""
    targets = _make_targets()
    metrics = {
        "auth": {
            "p99_latency_ms": 200.0,   # worse than target (100ms)
            "error_rate_pct": 0.5,      # better than target
            "saturation_pct": 30.0,     # better than target
        }
    }
    reward = compute_reward(metrics, targets)
    assert 0.3 < reward < 1.0, f"Expected intermediate reward, got {reward}"


def test_reward_bounded():
    """Reward must always be in [0, 1]."""
    targets = _make_targets()
    # Extreme cases
    for extreme_latency in [0.1, 999999.0]:
        metrics = {"auth": {"p99_latency_ms": extreme_latency, "error_rate_pct": 0.0, "saturation_pct": 0.0}}
        reward = compute_reward(metrics, targets)
        assert 0.0 <= reward <= 1.0, f"Reward out of bounds: {reward}"


def test_empty_targets():
    """Empty targets → reward = 0 (no signals to score)."""
    reward = compute_reward({}, {})
    assert reward == 0.0


def test_multi_service_reward():
    """Multi-service targets: all services must be considered."""
    targets = {
        "auth": {"p99_latency_ms": 100.0},
        "api-gateway": {"error_rate_pct": 1.0},
    }
    # auth restored, api-gateway still broken
    metrics = {
        "auth": {"p99_latency_ms": 50.0},
        "api-gateway": {"error_rate_pct": 30.0},
    }
    reward = compute_reward(metrics, targets)
    # Should be roughly 50% (one service restored, one not)
    assert 0.3 < reward < 0.8, f"Expected partial reward, got {reward}"


def test_reward_breakdown_keys():
    """breakdown() should include per-service and per-signal keys."""
    targets = _make_targets()
    metrics = {"auth": {"p99_latency_ms": 50.0, "error_rate_pct": 0.5, "saturation_pct": 20.0}}
    breakdown = reward_breakdown(metrics, targets)
    assert "auth" in breakdown
    assert "p99_latency_ms" in breakdown["auth"]
    assert "final_reward" in breakdown
