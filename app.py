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

import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

try:
    import spaces
except ImportError:
    class _DummySpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator
    spaces = _DummySpaces()

# Prevent HF Hub token writes on read-only Space containers
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _MonkeyPatchHfFolder:
        @staticmethod
        def save_token(token: str) -> None:
            pass
        @staticmethod
        def get_token() -> str | None:
            return None
        @staticmethod
        def delete_token() -> None:
            pass
        @staticmethod
        def path_token() -> str:
            return ""
    huggingface_hub.HfFolder = _MonkeyPatchHfFolder

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


async def _execute_tests_with_provider(
    test_ids: List[int],
    n_runs: int,
    provider_name: str,
    api_key: str,
) -> List[Any]:
    """Execute adversarial suite under a temporary provider configuration with strict session isolation."""
    async with _EXECUTION_LOCK:
        orig_providers = list(provider_pool.providers)
        orig_idx = provider_pool.current_idx
        orig_cooldowns = dict(provider_pool.provider_cooldowns)
        orig_settings_key = settings.model_api_key
        orig_settings_url = settings.model_base_url
        orig_settings_model = settings.model_name

        try:
            cfg = PROVIDER_CONFIGS.get(provider_name, PROVIDER_CONFIGS["Zhipu Direct (Tier 3)"])
            active_key = api_key.strip()

            # Temporarily configure the pool and global settings for this specific evaluation
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

            # Run in in-memory mode (use_db=False) to ensure zero risk to dev databases
            return await run_all(test_ids=test_ids, n_runs=n_runs, use_db=False)
        finally:
            # Restore global state cleanly after run completes
            provider_pool.providers = orig_providers
            provider_pool.current_idx = orig_idx
            provider_pool.provider_cooldowns = orig_cooldowns
            settings.model_api_key = orig_settings_key
            settings.model_base_url = orig_settings_url
            settings.model_name = orig_settings_model


@spaces.GPU(duration=120)
def run_benchmark_ui(
    test_choices: List[str],
    provider_choice: str,
    byok_key: str,
    n_runs_slider: int,
    session_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any], str]:
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

    # Check whether visitor provided a BYOK key or needs free trial fallback
    cleaned_key = (byok_key or "").strip()
    is_byok = len(cleaned_key) > 0

    if is_byok:
        active_provider = provider_choice
        active_key = cleaned_key
    else:
        # Free trial fallback checks
        remaining = session_state.get("remaining", 2)
        if remaining <= 0:
            err_msg = "### Session Free Trial Limit Reached (2/2 runs used)\n\nYou have used your session allocation. Please paste your own API Key (`BYOK`) in the input box above to continue running evaluations at your own quota."
            return err_msg, json.dumps({"error": "session_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        ok, daily_err = _check_and_update_daily_limit()
        if not ok:
            return f"### {daily_err}", json.dumps({"error": "daily_limit_reached"}, indent=2), session_state, _get_counter_msg(session_state)

        # Decrement visitor's session counter
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
        err_msg = f"### Evaluation Execution Error\n\nAn exception occurred while executing the benchmark suite:\n```\n{str(e)}\n```"
        return err_msg, json.dumps({"error": str(e)}, indent=2), session_state, _get_counter_msg(session_state)


@spaces.GPU(duration=120)
def run_custom_test_ui(
    uploaded_file: Any,
    custom_description: str,
    target_service: str,
    inject_trap: bool,
    provider_choice: str,
    byok_key: str,
    session_state: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any], str]:
    """Gradio handler for executing custom uploaded or user-specified test cases."""
    fault_desc = (custom_description or "").strip()
    serv = target_service or "api-gateway"
    trap = inject_trap

    if uploaded_file is not None:
        try:
            file_path = uploaded_file if isinstance(uploaded_file, str) else getattr(uploaded_file, "name", str(uploaded_file))
            with open(file_path, "r", encoding="utf-8") as f:
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

    if not fault_desc:
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

    orig_providers = list(provider_pool.providers)
    orig_idx = provider_pool.current_idx
    orig_cooldowns = dict(provider_pool.provider_cooldowns)
    orig_settings_key = settings.model_api_key
    orig_settings_url = settings.model_base_url
    orig_settings_model = settings.model_name

    try:
        cfg = PROVIDER_CONFIGS.get(active_provider, PROVIDER_CONFIGS["Zhipu Direct (Tier 3)"])
        provider_pool.providers = [cfg["pool_provider"]]
        provider_pool.current_idx = 0
        provider_pool.provider_cooldowns = {}
        settings.model_api_key = active_key
        settings.model_base_url = cfg["base_url"]
        settings.model_name = cfg["model_name"]

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
        md.append(f"- **Final Reward Score ($R_t$)**: `{final_reward:.4f if final_reward is not None else 0.0:.4f}` / `1.0000`")
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
    finally:
        provider_pool.providers = orig_providers
        provider_pool.current_idx = orig_idx
        provider_pool.provider_cooldowns = orig_cooldowns
        settings.model_api_key = orig_settings_key
        settings.model_base_url = orig_settings_url
        settings.model_name = orig_settings_model


def _get_counter_msg(session_state: Dict[str, Any]) -> str:
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

        @spaces.GPU(duration=120)
        def _run_single_ep(task_id: str) -> Tuple[str, str]:
            try:
                stats = asyncio.run(eval_task(task_id=task_id, n_episodes=1))
                md = [f"# Episode Outcome for `{task_id}`\n"]
                md.append(f"- **Mean Reward ($R_t$)**: `{stats.get('mean_reward', 0.0):.4f}`")
                md.append(f"- **Outcome Distribution**: `{json.dumps(stats.get('outcome_counts', {}))}`")
                md.append(f"- **Mean Steps Taken**: `{stats.get('mean_steps', 0.0):.1f}`")
                md.append(f"- **Total Time Elapsed**: `{stats.get('elapsed_seconds', 0.0):.2f}s`")
                return "\n".join(md), json.dumps(stats, indent=2)
            except Exception as e:
                return f"### Error executing episode:\n```\n{str(e)}\n```", json.dumps({"error": str(e)}, indent=2)

        ep_btn.click(fn=_run_single_ep, inputs=[task_selector], outputs=[ep_report, ep_json])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
