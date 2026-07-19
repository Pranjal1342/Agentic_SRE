"""
tests/test_mcp_tools.py — Integration tests for MCP tool layer.

Tests tool dispatch, schema validation, quarantine passthrough,
and typed error handling — without needing a live Postgres instance.
"""
import pytest
from mock_infra.mesh import MockMesh
from mock_infra.telemetry import TelemetryCollector
from mock_infra.mock_db import MockDatabase
from mcp.tools import MCPTools


class _PassGate:
    """Quarantine gate that always allows."""
    def check(self, **_):
        return True, ""


class _BlockGate:
    """Quarantine gate that always blocks."""
    def check(self, **kwargs):
        return False, f"Test block: {kwargs.get('action_type')} is rejected by policy"


@pytest.fixture
def tools_pass():
    mesh = MockMesh()
    return MCPTools(
        mesh=mesh,
        telemetry=TelemetryCollector(mesh),
        mock_db=MockDatabase(),
        quarantine_gate=_PassGate(),
    )


@pytest.fixture
def tools_block():
    mesh = MockMesh()
    return MCPTools(
        mesh=mesh,
        telemetry=TelemetryCollector(mesh),
        mock_db=MockDatabase(),
        quarantine_gate=_BlockGate(),
    )


def test_diagnostic_query_all(tools_pass):
    result = tools_pass.diagnostic_query("auth", "all")
    assert result.success is True
    assert result.data is not None
    assert "metrics" in result.data or "p99_latency_ms" in str(result.data)


def test_diagnostic_query_specific_metric(tools_pass):
    result = tools_pass.diagnostic_query("auth", "p99_latency_ms")
    assert result.success is True


def test_diagnostic_query_unknown_service(tools_pass):
    result = tools_pass.diagnostic_query("unknown-svc", "all")
    assert result.success is False
    assert result.error is not None


def test_diagnostic_query_unknown_metric(tools_pass):
    result = tools_pass.diagnostic_query("auth", "cpu_mhz")
    assert result.success is False


def test_log_inspection(tools_pass):
    result = tools_pass.log_inspection("auth", 5)
    assert result.success is True
    assert result.data is not None
    assert "logs" in result.data
    assert isinstance(result.data["logs"], list)


def test_remediation_allowed(tools_pass):
    result = tools_pass.remediation("restart_service", "auth", {})
    assert result.success is True
    assert result.quarantine_blocked is False


def test_remediation_blocked(tools_block):
    result = tools_block.remediation("restart_service", "auth", {})
    assert result.success is False
    assert result.quarantine_blocked is True
    assert result.quarantine_reason is not None
    assert "Test block" in result.quarantine_reason


def test_remediation_unknown_action(tools_pass):
    result = tools_pass.remediation("nuke_everything", "auth", {})
    assert result.success is False
    assert result.error is not None


def test_submit_resolution(tools_pass):
    result = tools_pass.submit_resolution("Incident resolved: restarted auth service")
    assert result.success is True
    assert tools_pass.resolution_submitted is True


def test_submit_resolution_too_short(tools_pass):
    result = tools_pass.submit_resolution("ok")
    assert result.success is False


def test_submit_resolution_idempotent(tools_pass):
    """Can't submit resolution twice."""
    tools_pass.submit_resolution("Incident resolved: restarted auth service")
    result = tools_pass.submit_resolution("Another summary that is long enough to pass validation")
    assert result.success is False
    assert "already submitted" in result.error


def test_smoke_test():
    """Run the built-in smoke test."""
    MCPTools.smoke_test()
