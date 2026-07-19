"""
agents/reasoning_loop.py — ToRA-style agent reasoning loop (per episode).

Pattern: Reason → Act → Observe → Revise

Provider-agnostic: uses OpenAI-compatible API via config.model_base_url /
config.model_api_key / config.model_name. Switching providers = env var change only.

Includes:
  - Retry-with-exponential-backoff (§5B requirement — free/trial tiers are rate-limited)
  - GLM-5.2 thinking mode via extra_body (on by default, configurable)
  - Single-tool probe (probe_tool_calling) to verify format before full wiring
  - Claude backend fallback when model_backend=claude

Per brief §5:
1. State goal (golden-signal targets) before first action
2. Query memory (lessons table) for similar past states → inject into context
3. State rationale for chosen action BEFORE calling tool
4. On failed/rejected action: reflect, retry with revision
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

from config import settings
from mcp.tools import MCPTools, ToolResult
from memory.retrieve import embed_state, format_lessons_for_context, retrieve_lessons
from memory.write import write_causal_edge, write_decision

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions — OpenAI function-calling format
# (also used as source-of-truth for Claude's input_schema)
# ─────────────────────────────────────────────────────────────────────────────
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "diagnostic_query",
            "description": "Query current golden signals (p99 latency, error rate, saturation) for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["auth", "api-gateway", "user-service", "payment-service"],
                        "description": "Target service name",
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["p99_latency_ms", "error_rate_pct", "saturation_pct", "all"],
                        "description": "Metric to query, or 'all' for full golden signals",
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_inspection",
            "description": "Inspect recent logs and distributed traces for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "enum": ["auth", "api-gateway", "user-service", "payment-service"],
                    },
                    "time_window_minutes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 60,
                        "description": "How many minutes of logs to inspect",
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remediation",
            "description": (
                "Apply a remediation action to a service. "
                "Will be validated by the Quarantine gate before execution. "
                "If rejected, you will receive the exact rejection reason — reflect on it and revise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": [
                            "restart_service", "scale_up", "rollback",
                            "increase_db_pool", "kill_slow_queries", "vacuum_analyze",
                            "circuit_breaker",
                        ],
                    },
                    "target": {
                        "type": "string",
                        "enum": ["auth", "api-gateway", "user-service", "payment-service", "database"],
                    },
                    "params": {
                        "type": "object",
                        "description": "Action parameters. E.g. {'factor': 2.0} for scale_up.",
                    },
                },
                "required": ["action_type", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_resolution",
            "description": "Submit your final resolution summary when the incident is resolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Detailed summary of what was wrong, what you did, and why.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

# Claude format derived from OpenAI definitions
CLAUDE_TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in OPENAI_TOOLS
]


# ─────────────────────────────────────────────────────────────────────────────
# Provider client helpers
# ─────────────────────────────────────────────────────────────────────────────

class ProviderPool:
    """Manages rotation across configured API keys/tiers on rate-limits (429) and quota exhaustion (402)."""
    def __init__(self):
        self.providers = []
        self.current_idx = 0
        self._init_providers()

    def _init_providers(self):
        # 1. Primary from settings/env
        if settings.model_api_key and settings.model_api_key.strip():
            self.providers.append({
                "base_url": settings.model_base_url.strip(),
                "api_key": settings.model_api_key.strip(),
                "model_name": settings.model_name.strip(),
                "name": "Primary Configured"
            })

        # 2. Add all known backup tiers if present or configured in environment variables
        known_tiers = [
            {"name": "OpenRouter (Tier 4)", "base_url": "https://openrouter.ai/api/v1", "api_key": os.getenv("OPENROUTER_API_KEY", ""), "model_name": "z-ai/glm-5.2"},
            {"name": "ZenMux (Tier 1)", "base_url": "https://api.zenmux.ai/v1", "api_key": os.getenv("ZENMUX_API_KEY", ""), "model_name": "z-ai/glm-5.2-free"},
            {"name": "Z.ai Direct (Tier 2)", "base_url": "https://api.z.ai/v1", "api_key": os.getenv("ZAI_API_KEY", ""), "model_name": "glm-5.2"},
            {"name": "Zhipu Direct (Tier 3)", "base_url": "https://open.bigmodel.cn/api/paas/v4/", "api_key": os.getenv("ZHIPU_API_KEY", ""), "model_name": "glm-5.2"},
            {"name": "HuggingFace Router (Tier 0)", "base_url": "https://router.huggingface.co/v1", "api_key": os.getenv("HF_API_KEY", ""), "model_name": "zai-org/GLM-5.2:novita"},
        ]
        seen_keys = {p["api_key"] for p in self.providers}
        for t in known_tiers:
            if t["api_key"] and t["api_key"] not in seen_keys:
                self.providers.append(t)
                seen_keys.add(t["api_key"])

    def next_provider(self) -> bool:
        if len(self.providers) <= 1:
            return False
        self.current_idx = (self.current_idx + 1) % len(self.providers)
        p = self.providers[self.current_idx]
        log.warning("api.provider_failover", switched_to=p["name"], base_url=p["base_url"], model=p["model_name"])
        return True

    def get_active(self) -> Dict[str, Any]:
        if not self.providers:
            return {"base_url": settings.model_base_url, "api_key": settings.model_api_key or "none", "model_name": settings.model_name, "name": "Default"}
        return self.providers[self.current_idx]

    def get_client(self):
        from openai import OpenAI
        active = self.get_active()
        return OpenAI(api_key=active["api_key"], base_url=active["base_url"])


provider_pool = ProviderPool()


def _get_openai_client():
    return provider_pool.get_client()


def _get_claude_client():
    import anthropic
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Retry-with-exponential-backoff + Automatic Provider Failover (§5B)
# ─────────────────────────────────────────────────────────────────────────────

def _call_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on rate-limit / transient errors.
    Uses config.model_max_retries, model_retry_base_delay, model_retry_max_delay.
    """
    import openai

    max_retries = settings.model_max_retries
    base_delay = settings.model_retry_base_delay
    max_delay = settings.model_retry_max_delay

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            log.warning(
                "model.rate_limited",
                attempt=attempt + 1,
                max_retries=max_retries,
                retry_in=delay,
                model=settings.model_name,
            )
            time.sleep(delay)
        except openai.APIStatusError as exc:
            # Retry on 5xx server errors only
            if exc.status_code and exc.status_code >= 500:
                last_exc = exc
                if attempt == max_retries:
                    break
                delay = min(base_delay * (2 ** attempt), max_delay)
                log.warning("model.server_error", status=exc.status_code, retry_in=delay)
                time.sleep(delay)
            else:
                raise  # 4xx (bad request, auth, etc.) — don't retry
        except Exception:
            raise  # Non-API errors — don't retry

    raise last_exc


def _execute_completion_with_failover(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] = OPENAI_TOOLS, tool_choice: str = "auto", max_tokens: int = 4096):
    """
    Execute chat completion across providers in ProviderPool.
    Automatically rotates to the next API key when hitting 402 (Depleted Credits), 401 (Auth Error), or exhausted 429 (Rate Limit).
    """
    import openai
    max_retries = settings.model_max_retries
    base_delay = settings.model_retry_base_delay
    max_delay = settings.model_retry_max_delay

    last_exc = None
    total_attempts = 0
    max_total = max_retries * max(len(provider_pool.providers), 1) + 5

    while total_attempts < max_total:
        total_attempts += 1
        active = provider_pool.get_active()
        client = provider_pool.get_client()
        model_name = active["model_name"]
        base_url = active["base_url"]

        extra_body = {}
        if settings.model_thinking_mode == "on":
            if "openrouter.ai" in base_url:
                extra_body = {"reasoning": {"enabled": True}}
            else:
                extra_body = {"thinking": {"mode": "on"}}

        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                **({"extra_body": extra_body} if extra_body else {}),
            )
        except openai.RateLimitError as exc:
            last_exc = exc
            delay = min(base_delay * (2 ** ((total_attempts - 1) % max_retries)), max_delay)
            log.warning("model.rate_limited", provider=active["name"], attempt=total_attempts, retry_in=delay)
            time.sleep(delay)
            # If we hit multiple rate limits, rotate provider to avoid blocking
            if len(provider_pool.providers) > 1 and (total_attempts % 2 == 0):
                provider_pool.next_provider()
        except openai.APIStatusError as exc:
            last_exc = exc
            err_msg = str(exc).lower()
            if exc.status_code in [401, 402, 403] or (exc.status_code in [400, 404] and ("model" in err_msg or "not a valid" in err_msg or "endpoint" in err_msg)):
                # Out of credits / auth failure / invalid model ID on this provider — rotate immediately!
                log.warning("model.provider_error", status=exc.status_code, provider=active["name"], error=str(exc)[:150])
                if not provider_pool.next_provider():
                    raise  # No other providers to fall back to
                continue
            elif exc.status_code == 429:
                delay = min(base_delay * (2 ** ((total_attempts - 1) % max_retries)), max_delay)
                log.warning("model.status_429", provider=active["name"], retry_in=delay)
                time.sleep(delay)
                if len(provider_pool.providers) > 1 and (total_attempts % 2 == 0):
                    provider_pool.next_provider()
            elif exc.status_code and exc.status_code >= 500:
                delay = min(base_delay * (2 ** ((total_attempts - 1) % max_retries)), max_delay)
                log.warning("model.server_error", status=exc.status_code, provider=active["name"], retry_in=delay)
                time.sleep(delay)
                if len(provider_pool.providers) > 1 and (total_attempts % 3 == 0):
                    provider_pool.next_provider()
            else:
                raise  # Other schema / bad request errors — don't rotate
        except Exception as exc:
            last_exc = exc
            raise

    raise last_exc


# ─────────────────────────────────────────────────────────────────────────────
# Single-tool probe (§5B: verify format before wiring full tool set)
# ─────────────────────────────────────────────────────────────────────────────

def probe_tool_calling() -> dict:
    """
    Send one test call with a single MCP tool exposed and inspect the raw response.
    Run this BEFORE running full episodes to confirm GLM-5.2's function-calling
    response format matches what the reasoning loop expects.

    Returns a dict with:
      - raw_response: the raw API response object (inspect manually)
      - tool_calls_found: list of (name, args) extracted
      - format_ok: True if the format matches expected OpenAI function-calling spec
      - notes: any discrepancies found
    """
    try:
        response = _execute_completion_with_failover(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need to check the latency of the auth service. "
                        "Please call the diagnostic_query tool with service='auth' and metric='all'."
                    ),
                }
            ],
            tools=single_tool,
            tool_choice="auto",
            max_tokens=512,
        )
    except Exception as exc:
        return {
            "raw_response": None,
            "tool_calls_found": [],
            "format_ok": False,
            "notes": f"API call failed: {exc}",
        }

    msg = response.choices[0].message
    tool_calls_found = []
    notes = []
    format_ok = True

    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                tool_calls_found.append({"name": tc.function.name, "args": args, "id": tc.id})
            except json.JSONDecodeError as e:
                notes.append(f"JSON parse error on arguments: {e}")
                format_ok = False
    else:
        notes.append("No tool_calls in response — model may have responded with text only.")
        format_ok = False
        if msg.content:
            notes.append(f"Text response: {msg.content[:200]}")

    # Verify expected fields exist
    if tool_calls_found:
        tc = tool_calls_found[0]
        if tc["name"] != "diagnostic_query":
            notes.append(f"Wrong tool called: {tc['name']}")
            format_ok = False
        if "service" not in tc["args"]:
            notes.append("Expected 'service' arg not found in tool call args")
            format_ok = False

    result = {
        "raw_response": response.model_dump() if hasattr(response, "model_dump") else str(response),
        "model": settings.model_name,
        "base_url": settings.model_base_url,
        "tool_calls_found": tool_calls_found,
        "format_ok": format_ok,
        "finish_reason": response.choices[0].finish_reason,
        "notes": notes,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Loop
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningLoop:
    """
    ToRA-style agent reasoning loop for a single episode.
    Provider-agnostic: backend selected via config.model_backend.
    """

    def __init__(
        self,
        tools: MCPTools,
        telemetry,
        fsm,
        episode_id: str,
        golden_targets: Dict[str, Any],
    ) -> None:
        self._tools = tools
        self._telemetry = telemetry
        self._fsm = fsm
        self._episode_id = episode_id
        self._golden_targets = golden_targets
        self._messages: List[Dict] = []
        self._step_index = 0
        self._prev_decision_id: Optional[str] = None
        self._backend = settings.model_backend

    async def run(self) -> None:
        initial_obs = self._telemetry.full_observation()
        system_prompt = self._build_system_prompt(initial_obs)

        self._messages = [
            {
                "role": "user",
                "content": (
                    "A new incident has been detected. "
                    "Please diagnose and resolve it. "
                    "Begin by stating your goal and the current golden-signal targets you need to restore, "
                    "then proceed to diagnose before taking any remediation action."
                ),
            }
        ]

        log.info(
            "reasoning_loop.start",
            episode_id=self._episode_id,
            backend=self._backend,
            model=settings.model_name,
        )

        while self._fsm.is_active and not self._tools.resolution_submitted:
            if not self._fsm.step():
                break

            # ── Query memory before deciding ──────────────────────────────────
            state_sig = self._build_state_signature()
            lessons, no_match = await retrieve_lessons(
                state_signature=state_sig,
                task_id=self._fsm.ctx.task_id,
                episode_id=self._episode_id,
                step_index=self._step_index,
            )
            memory_context = format_lessons_for_context(lessons)

            if self._step_index > 0:
                current_state_str = json.dumps(self._telemetry.collect_metrics(), indent=2)
                memory_note = (
                    f"\n\n{memory_context}\n\n" if memory_context
                    else "\nNo relevant past experience found for this situation.\n\n"
                )
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"Current system state:\n{current_state_str}"
                        f"{memory_note}"
                        f"Please state your rationale for the next action before calling any tool."
                    ),
                })

            # ── Call model (with retry) ────────────────────────────────────────
            if self._backend == "claude":
                tool_calls, assistant_content = self._call_claude(system_prompt)
            else:
                tool_calls, assistant_content = self._call_openai_compatible(system_prompt)

            if assistant_content:
                log.debug("reasoning_loop.model_text", text=assistant_content[:200])

            if not tool_calls:
                break  # model responded with text only — end turn

            # ── Process tool calls ────────────────────────────────────────────
            tool_results = []
            for tool_name, tool_input, call_id in tool_calls:
                log.info(
                    "reasoning_loop.tool_call",
                    episode_id=self._episode_id,
                    step=self._step_index,
                    tool=tool_name,
                    input=tool_input,
                )

                result = self._dispatch_tool(tool_name, tool_input)

                # Write decision to memory
                embedding = embed_state(state_sig)
                decision_id = await write_decision(
                    episode_id=self._episode_id,
                    step_index=self._step_index,
                    state_signature=state_sig,
                    state_embedding=embedding,
                    action_type=tool_name,
                    action_payload=tool_input,
                    result_stdout=json.dumps(result.data) if result.data else None,
                    result_stderr=result.error,
                    exit_code=0 if result.success else 1,
                    quarantine_flag=result.quarantine_blocked,
                    no_match_flag=no_match,
                    quarantine_reason=result.quarantine_reason,
                )

                if self._prev_decision_id is not None:
                    await write_causal_edge(
                        from_decision=self._prev_decision_id,
                        to_decision=decision_id,
                        relation_type="preceded",
                    )
                self._prev_decision_id = decision_id

                if result.quarantine_blocked:
                    result_content = (
                        f"ACTION BLOCKED by Quarantine Gate.\n"
                        f"Reason: {result.quarantine_reason}\n\n"
                        f"Reflect on why this was rejected and try a different approach."
                    )
                elif result.success:
                    result_content = json.dumps(result.data or {"status": "success"})
                else:
                    result_content = f"Error: {result.error}"

                tool_results.append((call_id, tool_name, result_content))
                self._step_index += 1

            # Feed results back
            if self._backend == "claude":
                self._messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": cid, "content": content}
                        for cid, _, content in tool_results
                    ],
                })
            else:
                for cid, tname, content in tool_results:
                    self._messages.append({"role": "tool", "tool_call_id": cid, "content": content})

            if self._tools.resolution_submitted:
                break

        log.info("reasoning_loop.complete", episode_id=self._episode_id, steps=self._step_index)

    # ── Backend calls ─────────────────────────────────────────────────────────

    def _call_openai_compatible(self, system_prompt: str) -> Tuple[List, str]:
        """
        Call OpenAI-compatible API (GLM, Groq, Z.ai, ZenMux, OpenRouter).
        Passes MODEL_THINKING_MODE via extra_body for GLM-5.2.
        Returns (tool_calls, text_content).
        """
        messages = [{"role": "system", "content": system_prompt}] + self._messages

        response = _execute_completion_with_failover(
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            max_tokens=4096,
        )

        msg = response.choices[0].message
        text_content = msg.content or ""

        # Extract OpenRouter / GLM reasoning details if returned by provider
        reasoning_details = None
        if hasattr(msg, "reasoning_details") and msg.reasoning_details is not None:
            reasoning_details = msg.reasoning_details
        elif hasattr(msg, "model_extra") and isinstance(msg.model_extra, dict):
            reasoning_details = msg.model_extra.get("reasoning_details") or msg.model_extra.get("reasoning")
        elif hasattr(msg, "reasoning") and msg.reasoning is not None:
            reasoning_details = msg.reasoning

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append((tc.function.name, args, tc.id))

        # Add to message history with tool_calls and reasoning_details (preserved for multi-turn)
        assistant_msg: Dict = {"role": "assistant", "content": text_content}
        if reasoning_details is not None:
            assistant_msg["reasoning_details"] = reasoning_details
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        self._messages.append(assistant_msg)

        return tool_calls, text_content

    def _call_claude(self, system_prompt: str) -> Tuple[List, str]:
        """Call Anthropic Claude. Returns (tool_calls, text_content)."""
        import anthropic
        client = _get_claude_client()

        response = _call_with_retry(
            client.messages.create,
            model=settings.claude_model,
            max_tokens=4096,
            system=system_prompt,
            tools=CLAUDE_TOOLS,
            messages=self._messages,
        )

        self._messages.append({"role": "assistant", "content": response.content})

        tool_calls = []
        text_parts = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append((block.name, block.input, block.id))
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        return tool_calls, " ".join(text_parts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self, initial_obs: dict) -> str:
        targets_str = json.dumps(self._golden_targets, indent=2)
        obs_str = json.dumps(initial_obs.get("metrics", {}), indent=2)
        return f"""You are an expert SRE (Site Reliability Engineer) agent tasked with diagnosing and resolving an active incident.

## Your Goal
Restore all services to their golden-signal targets:
```json
{targets_str}
```

## Current State (at episode start)
```json
{obs_str}
```

## Instructions
1. **State your goal** explicitly at the start.
2. **Diagnose before remediating** — use diagnostic_query and log_inspection to understand root cause.
3. **State your rationale** before EVERY tool call. Format: "Rationale: [why]"
4. **If an action is blocked** by the Quarantine gate, read the rejection reason and try a different approach.
5. **Think causally** — for multi-service incidents, find root cause before fixing downstream symptoms.
6. **Submit resolution** only when signals are restored or options exhausted.

## Available Services: auth, api-gateway, user-service, payment-service

## Rules
- Do NOT attempt the same rejected action twice.
- Do NOT use shell commands or anything outside the provided tools.
"""

    def _build_state_signature(self) -> str:
        try:
            metrics = self._telemetry.collect_metrics()
            degraded = []
            for svc, m in metrics.items():
                issues = []
                if m.get("p99_latency_ms", 0) > 500:
                    issues.append(f"latency={m['p99_latency_ms']:.0f}ms")
                if m.get("error_rate_pct", 0) > 5.0:
                    issues.append(f"errors={m['error_rate_pct']:.1f}%")
                if m.get("saturation_pct", 0) > 80.0:
                    issues.append(f"sat={m['saturation_pct']:.0f}%")
                if issues:
                    degraded.append(f"{svc}:[{','.join(issues)}]")
            return f"step={self._step_index} degraded={';'.join(degraded) or 'none'}"
        except Exception:
            return f"step={self._step_index}"

    def _dispatch_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolResult:
        try:
            if tool_name == "diagnostic_query":
                return self._tools.diagnostic_query(
                    service=tool_input["service"],
                    metric=tool_input.get("metric", "all"),
                )
            elif tool_name == "log_inspection":
                return self._tools.log_inspection(
                    service=tool_input["service"],
                    time_window_minutes=tool_input.get("time_window_minutes", 5),
                )
            elif tool_name == "remediation":
                return self._tools.remediation(
                    action_type=tool_input["action_type"],
                    target=tool_input["target"],
                    params=tool_input.get("params", {}),
                )
            elif tool_name == "submit_resolution":
                return self._tools.submit_resolution(summary=tool_input["summary"])
            else:
                return ToolResult(tool=tool_name, success=False, error=f"Unknown tool: {tool_name}")
        except Exception as exc:
            log.exception("reasoning_loop.dispatch_error", tool=tool_name, error=str(exc))
            return ToolResult(tool=tool_name, success=False, error=str(exc))
