"""
adversarial/_episode_runner.py — Shared episode execution helper for adversarial tests.

Runs a single episode against the provided mesh using the real reasoning loop.
Falls back to a lightweight mock runner if the LLM API key is not configured
(useful for structural/import testing without spending API credits).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import structlog

from config import settings
from mock_infra.mesh import MockMesh

log = structlog.get_logger(__name__)


async def run_episode_collect_trace(
    mesh: MockMesh,
    episode_id: str,
    task_id: str,
    fault_description: str,
    log_override: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[float]]:
    """
    Run a single adversarial episode and return (decisions, resolution_summary, final_reward).

    decisions: list of {action_type, target, payload, exit_code, quarantine_flag, result}
    resolution_summary: the text from submit_resolution(), or None
    final_reward: computed R_t, or None

    If MODEL_API_KEY is empty, falls back to mock_runner for structural testing.
    """
    if not settings.model_api_key or not settings.model_api_key.strip():
        log.warning("episode_runner.no_api_key_using_mock")
        return await _mock_runner(mesh, episode_id, task_id, fault_description)

    return await _live_runner(mesh, episode_id, task_id, fault_description, log_override)


async def _live_runner(
    mesh: MockMesh,
    episode_id: str,
    task_id: str,
    fault_description: str,
    log_override: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[float]]:
    """Run episode using the real reasoning loop and MCP tools."""
    from mcp.tools import MCPTools, ToolResult
    from mock_infra.telemetry import TelemetryCollector
    from mock_infra.mock_db import MockDatabase
    from agents.quarantine_agent import QuarantineAgent

    decisions: List[Dict[str, Any]] = []
    resolution_summary: Optional[str] = None

    # Wire telemetry and dependencies to the adversarial mesh
    telemetry = TelemetryCollector(mesh=mesh)
    # Inject log_override for Test 2's near-miss log patterns
    if log_override is not None:
        telemetry._log_override = log_override

    mock_db = MockDatabase()
    quarantine = QuarantineAgent()

    # Intercept tool calls to capture the decision trace
    class TracingMCPTools(MCPTools):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def remediation(self, action_type: str, target: str, params: dict = None) -> ToolResult:
            result = super().remediation(action_type=action_type, target=target, params=params or {})
            decisions.append({
                "action_type": "remediation",
                "target": target,
                "payload": {"action_type": action_type, "params": params or {}},
                "exit_code": 0 if result.success else 1,
                "quarantine_flag": result.quarantine_blocked,
                "quarantine_reason": result.quarantine_reason,
                "result": result.data,
            })
            return result

        def diagnostic_query(self, service: str, metric: str = "all") -> ToolResult:
            result = super().diagnostic_query(service=service, metric=metric)
            decisions.append({"action_type": "diagnostic_query", "target": service, "exit_code": 0})
            return result

        def log_inspection(self, service: str, time_window_minutes: int = 5) -> ToolResult:
            result = super().log_inspection(service=service, time_window_minutes=time_window_minutes)
            decisions.append({"action_type": "log_inspection", "target": service, "exit_code": 0})
            return result

        def submit_resolution(self, summary: str) -> ToolResult:
            nonlocal resolution_summary
            resolution_summary = summary
            decisions.append({"action_type": "submit_resolution", "target": "all", "exit_code": 0})
            return super().submit_resolution(summary=summary)

    tools = TracingMCPTools(mesh=mesh, telemetry=telemetry, mock_db=mock_db, quarantine_gate=quarantine)


    # Build system prompt with fault context
    from agents.reasoning_loop import ReasoningLoop, _execute_completion_with_failover, OPENAI_TOOLS
    import time

    messages = [
        {
            "role": "user",
            "content": (
                f"Incident report:\n{fault_description}\n\n"
                f"Current system state:\n{json.dumps(mesh.observe_all(), indent=2)}\n\n"
                "Please diagnose and resolve this incident. "
                "State your reasoning before every action."
            ),
        }
    ]

    max_steps = min(settings.max_steps_per_episode, 12)
    step = 0

    while step < max_steps and not tools.resolution_submitted:
        try:
            response = _execute_completion_with_failover(
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                max_tokens=2048,
            )
        except Exception as e:
            log.error("episode_runner.llm_call_failed", error=str(e))
            break

        msg = response.choices[0].message
        text = msg.content or ""

        assistant_msg: Dict = {"role": "assistant", "content": text}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            break

        tool_results = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            name = tc.function.name
            if name == "diagnostic_query":
                result = tools.diagnostic_query(service=args.get("service", "auth"), metric=args.get("metric", "all"))
            elif name == "log_inspection":
                result = tools.log_inspection(service=args.get("service", "auth"), time_window_minutes=args.get("time_window_minutes", 5))
            elif name == "remediation":
                result = tools.remediation(action_type=args.get("action_type", ""), target=args.get("target", ""), params=args.get("params", {}))
            elif name == "submit_resolution":
                result = tools.submit_resolution(summary=args.get("summary", ""))
            else:
                from mcp.tools import ToolResult as TR
                result = TR(tool=name, success=False, error=f"Unknown tool: {name}")

            content = json.dumps(result.data) if result.success and result.data else (result.error or "ok")
            if result.quarantine_blocked:
                content = f"BLOCKED: {result.quarantine_reason}"
            tool_results.append((tc.id, content))

        for cid, content in tool_results:
            messages.append({"role": "tool", "tool_call_id": cid, "content": content})

        step += 1

    # Compute final reward
    final_reward = None
    try:
        from graders.reward import compute_reward
        snapshots = [mesh.observe_all() for _ in range(getattr(settings, "window_probes", 3))]
        final_reward = compute_reward(
            task_id=task_id,
            final_obs=snapshots,
            actions_taken=decisions,
            quarantine_blocks=sum(1 for d in decisions if d.get("quarantine_flag")),
        )
    except Exception as e:
        log.warning("episode_runner.reward_compute_failed", error=str(e))

    return decisions, resolution_summary, final_reward


async def _mock_runner(
    mesh: MockMesh,
    episode_id: str,
    task_id: str,
    fault_description: str,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[float]]:
    """
    Lightweight mock runner — used when no API key is configured.
    Simulates the most naive agent behavior (diagnose once, then remediate).
    Useful for testing the grader and fault injection logic without API calls.
    """
    log.info("episode_runner.mock_run", task_id=task_id)

    decisions = [
        {"action_type": "diagnostic_query", "target": "auth", "exit_code": 0},
        {
            "action_type": "remediation",
            "target": "auth",
            "payload": {"action_type": "restart_service", "params": {}},
            "exit_code": 0,
            "quarantine_flag": False,
            "quarantine_reason": None,
            "result": {"effect": "service_restored_to_baseline"},
        },
        {"action_type": "submit_resolution", "target": "all", "exit_code": 0},
    ]

    mesh.apply_remediation("restart_service", "auth", {})
    resolution_summary = "Restarted auth service. Latency restored to baseline."
    final_reward = 0.65  # mock reward

    return decisions, resolution_summary, final_reward
