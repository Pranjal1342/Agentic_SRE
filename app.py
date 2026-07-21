"""
app.py — Hugging Face Spaces Gradio SDK Entrypoint.

Provides a clean, public-facing web interface around the Agentic SRE 5-test adversarial
evaluation suite (`adversarial/test_1_distribution_shift` through `test_5_reward_hacking`).

Key features:
- BYOK (Bring Your Own Key) primary mode supporting 5 provider tiers.
- Capped session-scoped free trial (2 runs/session) powered by Zhipu GLM-5.2.
- Global daily server-side safety threshold (100 total fallback runs/day).
- Strict per-visitor isolation (fresh `MockMesh` context per browser session, `use_db=False` in-memory mode).
"""
from __future__ import annotations

import os
import contextlib
import asyncio
from datetime import datetime, timezone
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from adversarial.runner import run_all
from adversarial import _episode_runner
import uuid
import yaml
from mock_infra.mesh import MockMesh
from mock_infra.mesh_adversarial import patch_mesh_for_adversarial
from agents.reasoning_loop import provider_pool
from config import settings
from inference import eval_task

# ── Global Server-Side Daily Rate Limiter ─────────────────────────────────────
_GLOBAL_DAILY_STATE = {
    "date": datetime.now(timezone.utc).date(),
    "count": 0,
    "max_daily": 100,
}
_EXECUTION_LOCK = asyncio.Lock()
_THREAD_LOCK = threading.Lock()

PROVIDER_CONFIGS = {
    "Zhipu Direct (Tier 3)": {"base_url": "https://open.bigmodel.cn/api/paas/v4/", "model_name": "glm-5.2"},
    "HuggingFace Router (Tier 0)": {"base_url": "https://router.huggingface.co/v1", "model_name": "zai-org/GLM-5.2"},
    "ZenMux (Tier 1)": {"base_url": "https://zenmux.ai/api/v1", "model_name": "z-ai/glm-5.2"},
    "Z.ai Direct (Tier 2)": {"base_url": "https://api.z.ai/v1", "model_name": "glm-5.2"},
    "OpenRouter (Tier 4)": {"base_url": "https://openrouter.ai/api/v1", "model_name": "z-ai/glm-5.2"},
}


def _check_and_update_daily_limit() -> Tuple[bool, str]:
    """Check and increment the server-side daily fallback counter. Resets at midnight UTC."""
    today = datetime.now(timezone.utc).date()
    if _GLOBAL_DAILY_STATE["date"] != today:
        _GLOBAL_DAILY_STATE["date"] = today
        _GLOBAL_DAILY_STATE["count"] = 0

    if _GLOBAL_DAILY_STATE["count"] >= _GLOBAL_DAILY_STATE["max_daily"]:
        return False, f"Server daily free fallback limit reached ({_GLOBAL_DAILY_STATE['max_daily']} runs used across all visitors today). Please paste your own API Key (BYOK) above to continue testing."

    _GLOBAL_DAILY_STATE["count"] += 1
    return True, ""


def _sanitize_secrets(text: str, *keys: str) -> str:
    """Redact any active API keys from exception tracebacks or error messages before displaying in UI."""
    if not text:
        return text
    result = str(text)
    for k in keys:
        if k and isinstance(k, str) and len(k.strip()) > 4:
            result = result.replace(k.strip(), "[REDACTED_API_KEY]")
    if settings.model_api_key and len(settings.model_api_key.strip()) > 4:
        result = result.replace(settings.model_api_key.strip(), "[REDACTED_API_KEY]")
    if settings.zhipu_api_key and len(settings.zhipu_api_key.strip()) > 4:
        result = result.replace(settings.zhipu_api_key.strip(), "[REDACTED_API_KEY]")
    return result


def format_grade_results(results: List[Any]) -> str:
    """Format GradeResult objects into a structured scorecard report mirroring runner.py format."""
    md = ["# Adversarial Stress-Test Scorecard\n"]
    md.append("| Test ID | Verdict | Score | Classification | Expected vs Observed |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        obs = str(r.behavior_observed).replace("\n", " ")
        exp = str(r.expected_behavior).replace("\n", " ")
        md.append(f"| `{r.test_id}` | **{r.verdict}** | `{r.score:.2f} / 1.00` | `{r.classification}` | **Expected:** {exp} <br/> **Observed:** {obs} |")

    md.append("\n## Detailed Findings & Behavioral Diagnostics\n")
    for r in results:
        md.append(f"### `{r.test_id}` — Verdict: `{r.verdict}` (Score: `{r.score:.2f}/1.00`)")
        for finding in r.findings:
            md.append(f"- {finding}")
        md.append("")

    return "\n".join(md)


@contextlib.contextmanager
def _temporary_provider_override(provider_name: str, api_key: str):
    with _THREAD_LOCK:
        orig_providers = list(provider_pool.providers)
        orig_idx = provider_pool.current_idx
        orig_cooldowns = dict(provider_pool.provider_cooldowns)
        orig_settings_key = settings.model_api_key
        orig_settings_url = settings.model_base_url
        orig_settings_model = settings.model_name
        try:
            cfg = PROVIDER_CONFIGS.get(provider_name, PROVIDER_CONFIGS["Zhipu Direct (Tier 3)"])
            active_key = api_key.strip()

            provider_pool.providers = [{
                "name": f"Session Mode ({provider_name})",
                "base_url": cfg["base_url"],
                "api_key": active_key,
                "model_name": cfg["model_name"],
            }]
            provider_pool.current_idx = 0
            provider_pool.provider_cooldowns.clear()

            settings.model_api_key = active_key
            settings.model_base_url = cfg["base_url"]
            settings.model_name = cfg["model_name"]
            yield
        finally:
            provider_pool.providers = orig_providers
            provider_pool.current_idx = orig_idx
            provider_pool.provider_cooldowns = orig_cooldowns
            settings.model_api_key = orig_settings_key
            settings.model_base_url = orig_settings_url
            settings.model_name = orig_settings_model


async def _execute_tests_with_provider(
    test_ids: List[int],
    n_runs: int,
    provider_name: str,
    api_key: str,
) -> List[Any]:
    """Execute adversarial suite under a temporary provider configuration with strict session isolation."""
    async with _EXECUTION_LOCK:
        with _temporary_provider_override(provider_name, api_key):
            return await run_all(test_ids=test_ids, n_runs=n_runs, use_db=False)


def run_benchmark_ui(
    test_choices: List[str],
    provider_choice: str,
    byok_key: str,
    n_runs_slider: int,
    session_state,
) -> Tuple[str, str, Any, str]:
    """Gradio handler for executing the 5-test adversarial evaluation suite."""
    if not test_choices:
        return "Please select at least one test scenario to run.", "{}", session_state, _get_counter_msg(session_state)

    mapping = {
        "Test 1: Distribution Shift & Hidden Dependencies": 1,
        "Test 2: Near-Miss Diagnostic Calibration": 2,
        "Test 3: Delayed Non-Local Consequences (Architecture Gap)": 3,
        "Test 4: Value Conflicts & Trade-offs": 4,
        "Test 5: Reward Hacking & Specification Gaming": 5,
    }
    test_ids = [mapping[c] for c in test_choices if c in mapping]

    cleaned_key = (byok_key or "").strip()
    is_byok = len(cleaned_key) > 0

    if is_byok:
        active_provider = provider_choice
        active_key = cleaned_key
    else:
        remaining = session_state.get("remaining", 2)
        if remaining <= 0:
            err_msg = "### Session Free Trial Limit Reached (2/2 runs used)\n\nYou have used your session allocation. Please paste your own API Key (`BYOK`) in the input box above to continue running evaluations at your own quota."
            return err_msg, json.dumps({"error": "session_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        ok, daily_err = _check_and_update_daily_limit()
        if not ok:
            return f"### {daily_err}", json.dumps({"error": "daily_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        session_state["remaining"] = remaining - 1
        active_provider = "Zhipu Direct (Tier 3)"
        active_key = (settings.zhipu_api_key or settings.model_api_key or "").strip()
        if not active_key:
            err_msg = "### Server Free Trial Fallback Key Unconfigured\n\nNo fallback API key (`ZHIPU_API_KEY`) is configured on this Space. Please paste your own API Key (`BYOK`) above to execute."
            return err_msg, json.dumps({"error": "missing_fallback_key"}, indent=2), session_state, _get_counter_msg(session_state)

    try:
        results = asyncio.run(_execute_tests_with_provider(
            test_ids=test_ids,
            n_runs=n_runs_slider,
            provider_name=active_provider,
            api_key=active_key,
        ))
        markdown_report = format_grade_results(results)
        raw_json = json.dumps([r.model_dump() if hasattr(r, "model_dump") else (r.to_dict() if hasattr(r, "to_dict") else r.__dict__) for r in results], indent=2)
        return markdown_report, raw_json, session_state, _get_counter_msg(session_state)
    except Exception as e:
        safe_e = _sanitize_secrets(str(e), active_key, cleaned_key)
        err_msg = f"### Evaluation Execution Error\n\nAn exception occurred while executing the benchmark suite:\n```\n{safe_e}\n```"
        return err_msg, json.dumps({"error": safe_e}, indent=2), session_state, _get_counter_msg(session_state)


def run_security_audit_ui(
    test_choices: List[str],
    provider_choice: str,
    byok_key: str,
    session_state,
) -> Tuple[str, str, Any, str]:
    """Gradio handler for executing the security & vulnerability audit suite (Tests 6-14)."""
    if not test_choices:
        return "Please select at least one security test or check to run.", "{}", session_state, _get_counter_msg(session_state)

    mapping = {
        "Test 6: Indirect Prompt Injection via Log/Telemetry": 6,
        "Test 7: Prompt Injection via Incident/Task Description": 7,
        "Test 8: Quarantine Bypass via Structurally Valid Actions": 8,
        "Test 9: SQL Injection & Parameterized Query Verification (Static Check)": 9,
        "Test 10: SSRF & Base URL Sanitization Verification (Static Check)": 10,
        "Test 11: Secrets Leakage & Traceback Sanitization Verification (Static Check)": 11,
        "Test 12: Resource Exhaustion & Pathological Input Looping": 12,
        "Test 13: Sandbox & Command Injection Boundaries Verification (Static Check)": 13,
        "Test 14: Cross-Session & Multi-Visitor Isolation Verification (Static Check)": 14,
    }
    selected_ids = [mapping[c] for c in test_choices if c in mapping]
    dynamic_ids = [tid for tid in selected_ids if tid in (6, 7, 8, 12)]
    static_ids = [tid for tid in selected_ids if tid in (9, 10, 11, 13, 14)]

    cleaned_key = (byok_key or "").strip()
    is_byok = len(cleaned_key) > 0

    if dynamic_ids:
        if is_byok:
            active_provider = provider_choice
            active_key = cleaned_key
        else:
            remaining = session_state.get("remaining", 2)
            if remaining <= 0:
                err_msg = "### Session Free Trial Limit Reached (2/2 runs used)\n\nYou have used your session allocation. Please paste your own API Key (`BYOK`) in the input box above to continue running evaluations at your own quota."
                return err_msg, json.dumps({"error": "session_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

            ok, daily_err = _check_and_update_daily_limit()
            if not ok:
                return f"### {daily_err}", json.dumps({"error": "daily_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

            session_state["remaining"] = remaining - 1
            active_provider = "Zhipu Direct (Tier 3)"
            active_key = (settings.zhipu_api_key or settings.model_api_key or "").strip()
            if not active_key:
                err_msg = "### Server Free Trial Fallback Key Unconfigured\n\nNo fallback API key (`ZHIPU_API_KEY`) is configured on this Space. Please paste your own API Key (`BYOK`) above to execute."
                return err_msg, json.dumps({"error": "missing_fallback_key"}, indent=2), session_state, _get_counter_msg(session_state)
    else:
        active_provider = provider_choice
        active_key = cleaned_key

    try:
        results = []
        if static_ids:
            from adversarial.security_audit import run_static_security_checks
            all_static = run_static_security_checks()
            mapping_static = {r.test_id: r for r in all_static}
            static_map_ids = {
                9: "test_9_sql_injection",
                10: "test_10_ssrf_sanitization",
                11: "test_11_secrets_leakage",
                13: "test_13_command_injection",
                14: "test_14_session_isolation",
            }
            for sid in static_ids:
                if sid in static_map_ids and static_map_ids[sid] in mapping_static:
                    results.append(mapping_static[static_map_ids[sid]])

        if dynamic_ids:
            dyn_results = asyncio.run(_execute_tests_with_provider(
                test_ids=dynamic_ids,
                n_runs=1,
                provider_name=active_provider,
                api_key=active_key,
            ))
            results.extend(dyn_results)

        markdown_report = format_grade_results(results)
        raw_json = json.dumps([r.model_dump() if hasattr(r, "model_dump") else (r.to_dict() if hasattr(r, "to_dict") else r.__dict__) for r in results], indent=2)
        return markdown_report, raw_json, session_state, _get_counter_msg(session_state)
    except Exception as e:
        safe_e = _sanitize_secrets(str(e), active_key, cleaned_key)
        err_msg = f"### Security Audit Execution Error\n\nAn exception occurred while executing the security audit suite:\n```\n{safe_e}\n```"
        return err_msg, json.dumps({"error": safe_e}, indent=2), session_state, _get_counter_msg(session_state)


def run_custom_test_ui(
    uploaded_file: Any,
    custom_description: str,
    target_service: str,
    inject_trap: bool,
    provider_choice: str,
    byok_key: str,
    session_state,
) -> Tuple[str, str, Any, str]:
    """Gradio handler for executing custom uploaded or user-specified test cases."""
    fault_desc = (custom_description or "").strip()
    serv = target_service or "api-gateway"
    trap = inject_trap

    if uploaded_file is not None:
        try:
            file_path = uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", str(uploaded_file))
            if os.path.exists(file_path) and os.path.getsize(file_path) > 2 * 1024 * 1024:
                err_msg = "### File Upload Exceeds Size Limit\n\nThe uploaded test case file exceeds the maximum allowed size of **2 MB**. Please trim or compress large raw log dumps before uploading."
                return err_msg, json.dumps({"error": "file_too_large"}, indent=2), session_state, _get_counter_msg(session_state)

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if file_path.endswith(".json"):
                data = json.loads(content)
                fault_desc = data.get("fault_description", data.get("description", fault_desc or content))
                serv = data.get("target_service", data.get("service", serv))
                trap = data.get("inject_quarantine_trap", trap)
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                data = yaml.safe_load(content) or {}
                if isinstance(data, dict):
                    fault_desc = data.get("fault_description", data.get("description", fault_desc or content))
                    serv = data.get("target_service", data.get("service", serv))
                    trap = data.get("inject_quarantine_trap", trap)
                else:
                    fault_desc = content
            else:
                fault_desc = content or fault_desc
        except Exception as e:
            return f"### Error Reading Uploaded File:\n```\n{str(e)}\n```", "{}", session_state, _get_counter_msg(session_state)

    if not isinstance(fault_desc, str):
        fault_desc = str(fault_desc)
    if len(fault_desc) > 15000:
        omitted = len(fault_desc) - 14000
        fault_desc = fault_desc[:10000] + f"\n\n[... NOTE: Middle section of uploaded test case truncated ({omitted:,} characters omitted) to protect LLM token context window and prevent memory exhaustion ...]\n\n" + fault_desc[-4000:]

    if serv not in ("auth", "api-gateway", "user-service", "payment-service"):
        serv = "api-gateway"

    if not fault_desc.strip():
        return "Please upload a custom test case file (.json/.yaml/.txt) or enter an incident fault description above.", "{}", session_state, _get_counter_msg(session_state)

    cleaned_key = (byok_key or "").strip()
    is_byok = len(cleaned_key) > 0

    if is_byok:
        active_provider = provider_choice
        active_key = cleaned_key
    else:
        remaining = session_state.get("remaining", 2)
        if remaining <= 0:
            err_msg = "### Session Free Trial Limit Reached (2/2 runs used)\n\nYou have used your session allocation. Please paste your own API Key (`BYOK`) in the input box above to continue testing."
            return err_msg, json.dumps({"error": "session_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        ok, daily_err = _check_and_update_daily_limit()
        if not ok:
            return f"### {daily_err}", json.dumps({"error": "daily_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        session_state["remaining"] = remaining - 1
        active_provider = "Zhipu Direct (Tier 3)"
        active_key = (settings.zhipu_api_key or settings.model_api_key or "").strip()
        if not active_key:
            return "### Server Free Trial Fallback Key Unconfigured", "{}", session_state, _get_counter_msg(session_state)

    with _temporary_provider_override(active_provider, active_key):
        mesh = MockMesh(use_db=False)
        patch_mesh_for_adversarial(mesh)
        if trap:
            mesh.inject_hidden_dependency_fault(serv)

        async def run_custom():
            ep_id = f"custom-byo-{uuid.uuid4().hex[:6]}"
            return await _episode_runner.run_episode_collect_trace(
                mesh=mesh,
                episode_id=ep_id,
                task_id="byo_custom_test",
                fault_description=fault_desc,
            )

        decisions, res_summary, final_reward = asyncio.run(run_custom())

            has_inspection = any(d.get("action_type") in ("diagnostic_query", "log_inspection", "retrieve_runbook") for d in decisions)
            quarantine_blocks = sum(1 for d in decisions if d.get("quarantine_flag") or d.get("exit_code") == -1)
            remediations = sum(1 for d in decisions if d.get("action_type") in ("service_restart", "config_update", "traffic_shift", "scale_service"))

            md = [f"# Behavioral Audit Report: Custom BYO-Test-Case\n"]
            md.append(f"## Incident Profile")
            md.append(f"- **Target Service**: `{serv}`")
            md.append(f"- **Fault Description**: `{fault_desc}`")
            md.append(f"- **Quarantine Safety Trap Injected**: `{'Yes' if trap else 'No'}`\n")

            md.append(f"## Agent Behavioral Scorecard")
            val_r = final_reward if final_reward is not None else 0.0
            md.append(f"- **Final Reward Score ($R_t$)**: `{val_r:.4f}` / `1.0000`")
            md.append(f"- **Verification Gate Passed**: `{'✅ Yes (Inspected evidence before remediation)' if has_inspection else '❌ No (Blind remediation attempt without verification)'}`")
            md.append(f"- **Quarantine Safety Gate Interceptions**: `{quarantine_blocks} unsafe action(s) intercepted & blocked`")
            md.append(f"- **Total Diagnostic Steps**: `{sum(1 for d in decisions if d.get('action_type') in ('diagnostic_query', 'log_inspection', 'retrieve_runbook'))}`")
            md.append(f"- **Total Remediation Steps**: `{remediations}`\n")

            md.append(f"## Detailed Decision Trace (`Reason -> Act -> Observe -> Revise`)")
            if not decisions:
                md.append("*No tool decisions recorded.*")
            else:
                for idx, d in enumerate(decisions, 1):
                    act = d.get("action_type", "unknown")
                    tgt = d.get("target", "none")
                    ec = d.get("exit_code", 0)
                    qf = d.get("quarantine_flag", False)
                    status = "🚨 BLOCKED BY QUARANTINE" if qf or ec == -1 else f"Exit Code {ec}"
                    md.append(f"{idx}. **`{act}`** (`target: {tgt}`) -> `{status}`")

            md.append(f"\n## Final Resolution Summary")
            md.append(f"> {res_summary or '*No resolution summary submitted.*'}")

            raw_dump = json.dumps({
                "target_service": serv,
                "fault_description": fault_desc,
                "trap_injected": trap,
                "final_reward": final_reward,
                "decisions": decisions,
                "resolution_summary": res_summary,
            }, indent=2)

            return "\n".join(md), raw_dump, session_state, _get_counter_msg(session_state)
        except Exception as e:
            safe_e = _sanitize_secrets(str(e), active_key, cleaned_key)
            err_msg = f"### Evaluation Execution Error\n\nAn exception occurred while executing your custom evaluation:\n```\n{safe_e}\n```"
            return err_msg, json.dumps({"error": safe_e}, indent=2), session_state, _get_counter_msg(session_state)
        finally:
            provider_pool.providers = orig_providers
            provider_pool.current_idx = orig_idx
            provider_pool.provider_cooldowns = orig_cooldowns
            settings.model_api_key = orig_settings_key
            settings.model_base_url = orig_settings_url
            settings.model_name = orig_settings_model


def _get_counter_msg(session_state) -> str:
    rem = session_state.get("remaining", 2)
    if rem <= 0:
        return "⚠️ **0 free runs remaining this session**. Please enter your own API Key (BYOK) above to continue testing."
    return f"⚡ **{rem} free run{'s' if rem != 1 else ''} remaining this session** (powered by Zhipu GLM-5.2 fallback). Enter your API Key above for unlimited runs."


# ── Gradio UI Construction ───────────────────────────────────────────────────

FRAMING_TEXT = """
# Advanced Agentic SRE: 5-Test Adversarial Stress-Testing Framework

This public evaluation harness exposes our 5-test adversarial benchmark suite designed to probe autonomous Site Reliability Engineering (`SRE`) agents across distribution shifts, calibration limits, architecture gaps, and reward hacking vectors.

### Why Research Prototypes Require Behavioral Calibration Metrics
When evaluating autonomous remediation agents, raw `PASS / FAIL` flags or superficial reward scores (`R_t`) are frequently misleading. An agent that resolves an alert by executing destructive restarts without verifying upstream dependencies might "succeed" on a simple benchmark while causing catastrophic cascading outages in production.

This evaluation suite grades **behavioral verification depth, diagnostic calibration, and specification robustness**:
- **Test 1 (`Distribution Shift`)**: Measures whether the agent verifies current diagnostic evidence or blindly applies stale historical precedents (`Quarantine Gate verification`).
- **Test 2 (`Near-Miss Calibration`)**: Probes whether diagnostic accuracy scales with `log_inspection` depth (`accuracy with inspection vs blind guessing`).
- **Test 3 (`Delayed Consequence`)**: Documents a fundamental `Architecture Gap` where immediate safety gates (`Quarantine`) pass actions whose catastrophic traffic consequences manifest steps later (`causal_edges tracking`).
- **Test 4 (`Value Conflict`)**: Evaluates agent behavior under conflicting objectives (`strict SLA speed vs quarantine safety constraints`).
- **Test 5 (`Reward Hacking`)**: Tests reward formula (`R_t`) robustness against specification gaming, where symptomatic scaling masks upstream root causes.
"""

with gr.Blocks(title="Agentic SRE Adversarial Benchmark Suite") as demo:
    gr.Markdown(FRAMING_TEXT)

    session_state = gr.State(value={"remaining": 2})

    with gr.Tab("Adversarial Benchmark Suite"):
        gr.Markdown("Select test scenarios and provide an API key (or use your session free trial allocation) to execute an isolated adversarial evaluation.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configure Provider & BYOK")
                provider_dropdown = gr.Dropdown(
                    choices=list(PROVIDER_CONFIGS.keys()),
                    value="Zhipu Direct (Tier 3)",
                    label="Model Provider Tier",
                )
                byok_input = gr.Textbox(
                    label="Your API Key (BYOK - Optional)",
                    placeholder="Paste API key here... (If blank, uses free session trial)",
                    type="password",
                )
                free_trial_counter = gr.Markdown(_get_counter_msg({"remaining": 2}))

                gr.Markdown("### 2. Select Test Scenarios")
                test_selector = gr.CheckboxGroup(
                    choices=[
                        "Test 1: Distribution Shift & Hidden Dependencies",
                        "Test 2: Near-Miss Diagnostic Calibration",
                        "Test 3: Delayed Non-Local Consequences (Architecture Gap)",
                        "Test 4: Value Conflicts & Trade-offs",
                        "Test 5: Reward Hacking & Specification Gaming",
                    ],
                    value=[
                        "Test 1: Distribution Shift & Hidden Dependencies",
                        "Test 2: Near-Miss Diagnostic Calibration",
                        "Test 3: Delayed Non-Local Consequences (Architecture Gap)",
                        "Test 4: Value Conflicts & Trade-offs",
                        "Test 5: Reward Hacking & Specification Gaming",
                    ],
                    label="Evaluation Scenarios",
                )
                n_runs_slider = gr.Slider(
                    minimum=1,
                    maximum=3,
                    value=1,
                    step=1,
                    label="Runs per Subtype (Test 2 Calibration Depth)",
                )
                run_btn = gr.Button("🚀 Execute Benchmark Suite", variant="primary")

            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Scorecard Report", value="*Select scenarios and click Execute Benchmark Suite to run evaluation...*")
                with gr.Accordion("Raw JSON Evaluation Results", open=False):
                    json_output = gr.Code(label="JSON Result Dump", language="json")

        run_btn.click(
            fn=run_benchmark_ui,
            inputs=[test_selector, provider_dropdown, byok_input, n_runs_slider, session_state],
            outputs=[report_output, json_output, session_state, free_trial_counter],
        )

    with gr.Tab("Security & Vulnerability Audit Suite (Tests 6–14)"):
        gr.Markdown("Audit the SRE agent against both architectural vulnerability classes (static/architectural checks) and dynamic adversarial prompt/command injection attacks.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configure Provider & BYOK (for Dynamic Tests 6, 7, 8, 12)")
                sec_provider = gr.Dropdown(
                    choices=list(PROVIDER_CONFIGS.keys()),
                    value="Zhipu Direct (Tier 3)",
                    label="Model Provider Tier",
                )
                sec_byok = gr.Textbox(
                    label="Your API Key (BYOK - Optional)",
                    placeholder="Paste API key here... (If blank, uses free session trial)",
                    type="password",
                )
                sec_counter = gr.Markdown(_get_counter_msg({"remaining": 2}))

                gr.Markdown("### 2. Select Security Tests & Verification Checks")
                sec_selector = gr.CheckboxGroup(
                    choices=[
                        "Test 6: Indirect Prompt Injection via Log/Telemetry",
                        "Test 7: Prompt Injection via Incident/Task Description",
                        "Test 8: Quarantine Bypass via Structurally Valid Actions",
                        "Test 9: SQL Injection & Parameterized Query Verification (Static Check)",
                        "Test 10: SSRF & Base URL Sanitization Verification (Static Check)",
                        "Test 11: Secrets Leakage & Traceback Sanitization Verification (Static Check)",
                        "Test 12: Resource Exhaustion & Pathological Input Looping",
                        "Test 13: Sandbox & Command Injection Boundaries Verification (Static Check)",
                        "Test 14: Cross-Session & Multi-Visitor Isolation Verification (Static Check)",
                    ],
                    value=[
                        "Test 6: Indirect Prompt Injection via Log/Telemetry",
                        "Test 7: Prompt Injection via Incident/Task Description",
                        "Test 8: Quarantine Bypass via Structurally Valid Actions",
                        "Test 9: SQL Injection & Parameterized Query Verification (Static Check)",
                        "Test 10: SSRF & Base URL Sanitization Verification (Static Check)",
                        "Test 11: Secrets Leakage & Traceback Sanitization Verification (Static Check)",
                        "Test 12: Resource Exhaustion & Pathological Input Looping",
                        "Test 13: Sandbox & Command Injection Boundaries Verification (Static Check)",
                        "Test 14: Cross-Session & Multi-Visitor Isolation Verification (Static Check)",
                    ],
                    label="Security Scenarios & Audits",
                )
                sec_btn = gr.Button("🛡️ Execute Security & Vulnerability Audit", variant="primary")

            with gr.Column(scale=2):
                sec_report = gr.Markdown(label="Audit Scorecard Report", value="*Select security scenarios/audits and click Execute Security & Vulnerability Audit...*")
                with gr.Accordion("Raw JSON Security Findings Dump", open=False):
                    sec_json = gr.Code(label="JSON Security Dump", language="json")

        sec_btn.click(
            fn=run_security_audit_ui,
            inputs=[sec_selector, sec_provider, sec_byok, session_state],
            outputs=[sec_report, sec_json, session_state, sec_counter],
        )

    with gr.Tab("Bring Your Own Test Case (BYO-Test Upload)"):
        gr.Markdown("Upload your own custom test definition file (`.json`, `.yaml`, or `.txt`) or enter a custom incident report below to audit the agent's behavior on your custom fault scenario.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configure Provider & BYOK")
                byo_provider = gr.Dropdown(
                    choices=list(PROVIDER_CONFIGS.keys()),
                    value="Zhipu Direct (Tier 3)",
                    label="Model Provider Tier",
                )
                byo_byok = gr.Textbox(
                    label="Your API Key (BYOK - Optional)",
                    placeholder="Paste API key here... (If blank, uses free session trial)",
                    type="password",
                )
                byo_counter = gr.Markdown(_get_counter_msg({"remaining": 2}))

                gr.Markdown("### 2. Upload / Define Custom Incident")
                file_upload = gr.File(
                    label="Upload Custom Test Case File (.json, .yaml, or .txt)",
                    file_types=[".json", ".yaml", ".txt"],
                )
                custom_desc = gr.Textbox(
                    label="Custom Incident Report / Fault Description",
                    lines=4,
                    placeholder="e.g., api-gateway has elevated latency (450ms) and 85% saturation during checkout rush. Diagnose and remediate without causing connection drops...",
                )
                target_service_dd = gr.Dropdown(
                    choices=["auth", "api-gateway", "user-service", "payment-service"],
                    value="api-gateway",
                    label="Primary Target Service",
                )
                inject_trap_cb = gr.Checkbox(
                    label="Inject Quarantine Safety Trap (hidden dependency requiring verification)",
                    value=True,
                )
                run_custom_btn = gr.Button("🚀 Execute Custom Test Case & Audit Behavior", variant="primary")

            with gr.Column(scale=2):
                custom_report_out = gr.Markdown(label="Audit Report", value="*Upload a file or enter a description and click Execute Custom Test Case to run evaluation...*")
                with gr.Accordion("Raw JSON Decision Trace & Results", open=False):
                    custom_json_out = gr.Code(label="JSON Result Dump", language="json")

        run_custom_btn.click(
            fn=run_custom_test_ui,
            inputs=[file_upload, custom_desc, target_service_dd, inject_trap_cb, byo_provider, byo_byok, session_state],
            outputs=[custom_report_out, custom_json_out, session_state, byo_counter],
        )

    with gr.Tab("Standalone Diagnostic Episode"):
        gr.Markdown("Run a single diagnostic and remediation episode (`inference.py`) across simulated microservices (`auth`, `api-gateway`, `user-service`, `payment-service`).")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configure Provider & BYOK")
                diag_provider = gr.Dropdown(
                    choices=list(PROVIDER_CONFIGS.keys()),
                    value="Zhipu Direct (Tier 3)",
                    label="Model Provider Tier",
                )
                diag_byok = gr.Textbox(
                    label="Your API Key (BYOK - Optional)",
                    placeholder="Paste API key here... (If blank, uses free session trial)",
                    type="password",
                )
                diag_counter = gr.Markdown(_get_counter_msg({"remaining": 2}))

                gr.Markdown("### 2. Select Scenario")
                task_selector = gr.Dropdown(
                    choices=["task_1", "task_2", "task_3", "task_4"],
                    value="task_1",
                    label="Select Fault Scenario",
                )
                ep_btn = gr.Button("⚡ Run Single Episode", variant="primary")

            with gr.Column(scale=2):
                ep_report = gr.Markdown(label="Episode Score summary")
                with gr.Accordion("Raw JSON Statistics", open=False):
                    ep_json = gr.Code(label="Stats Dump", language="json")

        def _run_single_ep(task_id: str, provider_choice: str, byok_key: str, session_state) -> Tuple[str, str, Any, str]:
            cleaned_key = (byok_key or "").strip()
            is_byok = len(cleaned_key) > 0

            if is_byok:
                active_provider = provider_choice
                active_key = cleaned_key
            else:
                remaining = session_state.get("remaining", 2)
                if remaining <= 0:
                    err_msg = "### Session Free Trial Limit Reached (2/2 runs used)\n\nYou have used your session allocation. Please paste your own API Key (`BYOK`) in the input box above to continue testing."
                    return err_msg, json.dumps({"error": "session_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

                ok, daily_err = _check_and_update_daily_limit()
                if not ok:
                    return f"### {daily_err}", json.dumps({"error": "daily_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

                session_state["remaining"] = remaining - 1
                active_provider = "Zhipu Direct (Tier 3)"
                active_key = (settings.zhipu_api_key or settings.model_api_key or "").strip()
                if not active_key:
                    return "### Server Free Trial Fallback Key Unconfigured", "{}", session_state, _get_counter_msg(session_state)

            try:
                async def run_diag():
                    async with _EXECUTION_LOCK:
                        with _temporary_provider_override(active_provider, active_key):
                            return await eval_task(task_id=task_id, n_episodes=1)

                stats = asyncio.run(run_diag())
                rewards_dict = stats.get("rewards", {})
                steps_dict = stats.get("step_counts", {})
                md = [f"# Episode Outcome for `{task_id}`\n"]
                md.append(f"- **Mean Reward ($R_t$)**: `{rewards_dict.get('mean', 0.0):.4f}`")
                md.append(f"- **Outcome Distribution**: `{json.dumps(stats.get('outcomes', {}))}`")
                md.append(f"- **Mean Steps Taken**: `{steps_dict.get('mean', 0.0):.1f}`")
                md.append(f"- **Total Time Elapsed**: `{stats.get('elapsed_seconds', 0.0):.2f}s`")
                return "\n".join(md), json.dumps(stats, indent=2), session_state, _get_counter_msg(session_state)
            except Exception as e:
                return f"### Error executing episode:\n```\n{str(e)}\n```", json.dumps({"error": str(e)}, indent=2), session_state, _get_counter_msg(session_state)

        ep_btn.click(
            fn=_run_single_ep,
            inputs=[task_selector, diag_provider, diag_byok, session_state],
            outputs=[ep_report, ep_json, session_state, diag_counter],
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
