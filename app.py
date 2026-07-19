"""
app.py — Hugging Face Spaces Gradio SDK Entrypoint.

Provides a web interface to run the Agentic SRE 5-test adversarial evaluation suite
and interactive diagnostic episodes without requiring a paid Docker space or local CLI.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

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

import gradio as gr

from adversarial.runner import run_all
from inference import eval_task


def format_grade_results(results: List[Any]) -> str:
    """Format GradeResult objects into a markdown scorecard report."""
    md = ["# Adversarial Stress-Test Scorecard\n"]
    md.append("| Test ID | Verdict | Score | Classification | Expected vs Observed |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        obs = str(r.behavior_observed).replace("\n", " ")
        exp = str(r.expected_behavior).replace("\n", " ")
        md.append(f"| `{r.test_id}` | **{r.verdict}** | `{r.score:.2f} / 1.00` | `{r.classification}` | **Expected:** {exp} <br/> **Observed:** {obs} |")

    md.append("\n## Detailed Findings\n")
    for r in results:
        md.append(f"### {r.test_id} ({r.verdict} - {r.score:.2f})")
        for finding in r.findings:
            md.append(f"- {finding}")
        md.append("")

    return "\n".join(md)


def run_benchmark_ui(test_choices: List[str], use_db: bool) -> tuple[str, str]:
    """Handler for the Adversarial Benchmark tab."""
    if not test_choices:
        return "Please select at least one test to run.", "{}"

    # Map choice names to integer IDs
    mapping = {
        "Test 1: Distribution Shift & Hidden Dependencies": 1,
        "Test 2: Diagnostic Calibration": 2,
        "Test 3: Delayed Non-Local Consequences (Architecture Gap)": 3,
        "Test 4: Value Conflicts & Trade-offs": 4,
        "Test 5: Reward Hacking & Specification Gaming": 5,
    }
    test_ids = [mapping[c] for c in test_choices if c in mapping]

    try:
        results = asyncio.run(run_all(test_ids=test_ids, n_runs=1, use_db=use_db))
        markdown_report = format_grade_results(results)
        raw_json = json.dumps([r.model_dump() for r in results], indent=2)
        return markdown_report, raw_json
    except Exception as e:
        return f"Error executing benchmark suite: {str(e)}", json.dumps({"error": str(e)}, indent=2)


def run_episode_ui(task_id: str) -> tuple[str, str]:
    """Handler for running a standalone diagnostic task episode."""
    try:
        stats = asyncio.run(eval_task(task_id=task_id, n_episodes=1))
        md = [f"# Episode Results for `{task_id}`\n"]
        md.append(f"- **Mean Reward (R_t)**: `{stats.get('mean_reward', 0.0):.4f}`")
        md.append(f"- **Outcome Distribution**: `{json.dumps(stats.get('outcome_counts', {}))}`")
        md.append(f"- **Mean Steps Taken**: `{stats.get('mean_steps', 0.0):.1f}`")
        md.append(f"- **Total Time Elapsed**: `{stats.get('elapsed_seconds', 0.0):.2f}s`")
        
        return "\n".join(md), json.dumps(stats, indent=2)
    except Exception as e:
        return f"Error executing episode: {str(e)}", json.dumps({"error": str(e)}, indent=2)


with gr.Blocks(title="Agentic SRE Benchmark Suite") as demo:
    gr.Markdown(
        """
        # Advanced Agentic SRE & Adversarial Stress-Testing Framework
        
        An autonomous Site Reliability Engineering (SRE) diagnosis and remediation framework powered by structured Large Language Model reasoning loops (`Reason -> Act -> Observe -> Revise`). Pairs live diagnostic tool calling with immediate Quarantine safety gates, sustained multi-snapshot temporal verification, and a 5-test adversarial evaluation suite.
        """
    )

    with gr.Tab("Adversarial Benchmark Suite"):
        gr.Markdown("Select and execute the adversarial stress-test evaluations to benchmark agent diagnostic calibration, safety gates, and reward formula ($R_t$) robustness.")
        with gr.Row():
            with gr.Column(scale=1):
                test_selector = gr.CheckboxGroup(
                    choices=[
                        "Test 1: Distribution Shift & Hidden Dependencies",
                        "Test 2: Diagnostic Calibration",
                        "Test 3: Delayed Non-Local Consequences (Architecture Gap)",
                        "Test 4: Value Conflicts & Trade-offs",
                        "Test 5: Reward Hacking & Specification Gaming",
                    ],
                    value=[
                        "Test 1: Distribution Shift & Hidden Dependencies",
                        "Test 2: Diagnostic Calibration",
                        "Test 3: Delayed Non-Local Consequences (Architecture Gap)",
                        "Test 4: Value Conflicts & Trade-offs",
                        "Test 5: Reward Hacking & Specification Gaming",
                    ],
                    label="Select Adversarial Tests",
                )
                db_checkbox = gr.Checkbox(
                    value=False,
                    label="Enable PostgreSQL DB Connection (requires database attached to Space; keep unchecked for offline mock mode)",
                )
                run_btn = gr.Button("Execute Benchmark Suite", variant="primary")
            
            with gr.Column(scale=2):
                report_output = gr.Markdown(label="Scorecard Report")
                json_output = gr.Code(label="Raw JSON Results", language="json")

        run_btn.click(
            fn=run_benchmark_ui,
            inputs=[test_selector, db_checkbox],
            outputs=[report_output, json_output],
        )

    with gr.Tab("Run Diagnostic Episode"):
        gr.Markdown("Run an interactive diagnostic and remediation episode (`inference.py`) across simulated microservices (`auth`, `payment-service`, `api-gateway`).")
        with gr.Row():
            with gr.Column(scale=1):
                task_selector = gr.Dropdown(
                    choices=["task_1", "task_2", "task_3", "task_4"],
                    value="task_1",
                    label="Select Infrastructure Fault Scenario",
                )
                ep_btn = gr.Button("Run Diagnostic Episode", variant="primary")
            
            with gr.Column(scale=2):
                ep_report = gr.Markdown(label="Episode Statistics")
                ep_json = gr.Code(label="Raw JSON Stats", language="json")

        ep_btn.click(
            fn=run_episode_ui,
            inputs=[task_selector],
            outputs=[ep_report, ep_json],
        )

    with gr.Tab("Framework Architecture"):
        gr.Markdown(
            """
            ### Core Components
            1. **Autonomous Reasoning Loop (`agents/reasoning_loop.py`)**: Structured function calling (`log_inspection`, `get_metric`, `scale_up`, `restart_service`, `rollback`, `graceful_drain`). Requires diagnostic evidence verification before executing high-risk remediations.
            2. **Immediate Safety Gate (`quarantine.py`)**: State machine intercepting destructive or unverified commands (`quarantine_block`).
            3. **Sustained Metric Verification (`server/pipeline.py` & `graders/reward.py`)**: Evaluates SLA attainment across multiple temporal snapshots (`window_probes`) to prevent specification gaming where temporary relief masks underlying root causes.
            """
        )

if __name__ == "__main__":
    demo.launch()
