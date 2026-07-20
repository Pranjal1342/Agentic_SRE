---
title: Agentic SRE
sdk: gradio
sdk_version: "5.12.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Agentic SRE: Containerized SRE Incident Simulation & Adversarial Evaluation Harness

## Research Prototype Notice
This repository contains a research prototype and benchmark harness designed to stress-test autonomous Site Reliability Engineering (`SRE`) diagnostic agents inside an isolated, containerized mock microservice mesh (`MockMesh`). It operates in a simulated environment to demonstrate open behavioral problems in AI agent alignment, calibration, non-local architecture limits, and safety verification. It is **not** a production monitoring, alerting, or auto-remediation tool.

---

## Project Overview

When evaluating autonomous remediation agents, binary pass/fail scorecards or superficial reward formulas ($R_t$) are frequently misleading. An agent that resolves an alert by executing destructive service restarts without verifying upstream session-cache dependencies might succeed on a simplified benchmark while causing catastrophic cascading outages in production.

This framework implements an OpenEnv-compatible containerized RL/agentic environment simulating multi-service SRE incident response. It provides a deterministic finite-state machine (`FSM`) lifecycle, a typed action/observation space, a dense reward function scoring worst-case degradation across sustained temporal verification windows, and a safety-focused Quarantine Agent that gates remediation actions against both prompt injections and structural destructive command attempts.

---

## System Architecture

The system architecture is structured into five core subsystems matching the specification outlined in `agentic_sre_rebuild_brief.md`:

```
agent_sre_env/
├── adversarial/                        # 5-test behavioral evaluation & 9-test security audit suite
├── agents/                             # Reasoning loop and Quarantine safety interception wrapper
├── graders/                            # Dense reward formulas ($R_t$) with multi-probe snapshot scoring
├── knowledge_base/                     # Reference runbooks and system documentation
├── memory/                             # Causal memory models, trace persistence, and consolidation jobs
├── mock_infra/                         # Simulated microservice topology, decay models, and telemetry mesh
├── rag/                                # Vector index and runbook lookup utilities
├── server/                             # FastAPI endpoint orchestration and FSM lifecycle tracking
├── tasks/                              # Task definitions (tasks 1 through 4)
├── config.py                           # Global configuration and provider pool endpoints
├── Dockerfile                          # Container environment build instructions (non-root execution)
├── docker-compose.yml                  # Multi-service local environment orchestration
├── inference.py                        # Standalone CLI entrypoint for executing diagnostic episodes
├── init.sql                            # PostgreSQL schema initialization for causal edges tracking
└── app.py                              # Gradio web application exposing public stress-test suites
```

### 1. Environment Layer (`mock_infra/` & `server/fsm.py`)
- **Simulated Topology**: Models a four-service microservice mesh (`auth`, `api-gateway`, `user-service`, `payment-service`) with explicit dependency graph edges.
- **Deterministic Telemetry with Noise**: Generates realistic metrics (`p99_latency_ms`, `error_rate_pct`, `cpu_util_pct`, `memory_util_pct`, `db_pool_saturation_pct`) derived from underlying parametric decay states plus small Gaussian variance (`mock_infra/telemetry.py`).
- **FSM Episode Lifecycle**: Manages episode transitions (`server/fsm.py`) across structured states (`IDLE`, `INVESTIGATING`, `MITIGATING`, `VERIFYING`, `RESOLVED`, `ESCALATED`, `QUARANTINED`), enforcing maximum step limits (`max_steps_per_episode = 12`).

### 2. MCP Tool Layer (`mcp/tools.py`)
All interaction between the agent and the simulated mesh occurs through strictly typed Pydantic tool definitions. To prevent command injection and boundary violations, tools execute purely via dictionary state lookups and modifications (`zero exec(), eval(), or subprocess.run() calls`):
- **Diagnostic Tools**: `log_inspection`, `get_metric`, `observe_service`, `retrieve_runbook`, `diagnostic_query`.
- **Remediation Tools**: `scale_up`, `restart_service`, `rollback`, `graceful_drain`, `silence_alerts`.

### 3. Reasoning Loop (`agents/reasoning_loop.py` & `agents/quarantine_agent.py`)
- **Structured Tool-Calling Loop**: Orchestrates iterative multi-turn LLM generation (`agents/reasoning_loop.py`), parsing tool calls, executing actions against the mesh, and appending observations to the episode trace.
- **Quarantine Agent Wrapper**: Acts as an immediate safety interception gate (`agents/quarantine_agent.py`). It intercepts remediation actions before execution, validating that required prerequisite diagnostic evidence (`log_inspection` or `get_metric`) exists and blocking unsafe destructive overrides during active incidents or locked states.

### 4. Memory Layer (`memory/`)
- **Causal & Episodic Models**: Defines SQLAlchemy tables (`episodes`, `decisions`, `causal_edges`) supporting both asynchronous (`AsyncEngine`) and synchronous (`Engine`) database backends (`memory/models.py`, `memory/db.py`).
- **Trace Persistence**: Records complete action traces, observation payloads, and rationale strings at each step (`memory/write.py`).
- **Runbook & Edge Retrieval**: Retrieves relevant historical incident experiences and known causal relationships based on active symptom signatures (`memory/retrieve.py`).

### 5. Consolidation Job (`memory/consolidate.py` & `memory/credit_assignment.py`)
- **Post-Episode Consolidation**: Executes offline batch analysis after episode completion (`consolidate_episode_memory`).
- **Causal Credit Assignment**: Evaluates multi-step trajectories to assign causal credit (`compute_trajectory_credit`), linking remediation actions to eventual metric recovery or delayed degradation across non-local boundaries.

---

## Verified vs. Deferred Capabilities (Accuracy Disclosures)

To maintain strict accuracy, the operational status of core architectural components is categorized below based on confirmation in the current engineering session:

### Confirmed Working Capabilities
- **In-Memory & PostgreSQL Execution**: Full episode lifecycle execution confirmed working in both offline in-memory mode (`use_db=False`) and persistent PostgreSQL mode (`use_db=True`).
- **Adversarial Benchmark Suite (`Tests 1–5`)**: Confirmed executing and grading cleanly across distribution shifts, diagnostic calibration curves, causal edge tracking, value conflicts, and reward hacking audits (`adversarial/runner.py`).
- **Security & Vulnerability Audit Suite (`Tests 6–14`)**: Confirmed executing and passing 9/9 verification checks across prompt injections, quarantine bypasses, SQL injection parameterization, SSRF sanitization, traceback secret redaction, and session isolation (`adversarial/security_audit.py`).
- **Public Gradio Space Application (`app.py`)**: Confirmed running with dual-layer thread and async locking (`_THREAD_LOCK`, `_EXECUTION_LOCK`), full secret sanitization (`_sanitize_secrets`), BYOK routing, session free-trial decrements, and BYO-Test upload safeguards.

### Deferred, Partially Built, or Measurement-Only Capabilities
- **Direct Preference Optimization (`DPO`) Training**: DPO preference optimization and model fine-tuning pipelines are **not implemented**. The memory retrieval layer (`memory/retrieve.py`) and consolidation job (`memory/consolidate.py`) are instrumented to compute a `no-match-rate` metric (`no_match_rate`), which serves strictly as a measurement trigger to inform future dataset curation decisions.
- **Self-Editing (`SICA`-style) Agent Loop**: Self-Improving Causal Agent (`SICA`) self-editing and rule-mutation behaviors are **deferred** behind a plateau trigger (`R_t` improvement plateau over consecutive evaluation epochs) and are not currently built into the active reasoning loop.
- **`causal_edges` Retroactive-Detection Mechanism (`Test 3`)**: The `causal_edges` tracking table and graph extraction logic function as a **retroactive measurement and mitigation tool, not an immediate pre-action prevention mechanism**. When an agent executes an initial scale-up on `api-gateway` (`Test 3`), the immediate Quarantine gate permits the action because local metrics appear healthy. The delayed downstream consequence (`user-service` connection pool exhaustion) occurs several steps later. The `causal_edges` table records the graph edge `(api-gateway:scale_up -> user-service:pool_exhaustion)` during post-episode consolidation (`consolidate.py`) so future episodes can retrieve the pattern to inform mitigation decisions. It does not prevent the initial occurrence.

---

## Provider Pool Status & BYOK Requirements

The framework uses a multi-provider failover pool (`config.py` and `app.py`) supporting five LLM inference endpoints. Based on live verification during this session, the readiness status of each provider is listed below:

| Provider Name | Tier | Model | Confirmed Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Zhipu Direct** | `Tier 3` | `glm-5.2` | **Confirmed Working (`200 OK`)** | Fully verified. Powering active CLI evaluation runs and session free-trial fallbacks. Obtain keys at `open.bigmodel.cn`. |
| **OpenRouter Router** | `Tier 5` | `glm-5.2` | **Reachable, Requires Funding** | Confirmed reachable and configured, but returned `HTTP 402 - This request requires more credits` when tested without active account balance. Obtain keys and add credits at `openrouter.ai/settings/credits`. |
| **ZenMux** | `Tier 1` | `glm-5.2` | **Reachable, Requires Funding** | Confirmed reachable and configured, but returned `HTTP 402 - Access denied: model only available to accounts with balance`. Obtain keys at `zenmux.net`. |
| **Z.ai Direct** | `Tier 2` | `glm-5.2` | **Requires Credential Correction** | Returned `HTTP 404 Not Found` when tested with standard credentials under default routing paths. Requires verified endpoint correction. |
| **HuggingFace Router** | `Tier 4` | `glm-5.2` | **Configured, Requires Quota** | Configured in provider pool. Requires a valid Hugging Face API token (`HF_TOKEN`) with active router inference quota (`router.huggingface.co/zhipuai`). |

---

## Installation & Setup Instructions

### Prerequisites
- Operating System: Windows, macOS, or Linux
- Python: 3.10 or higher
- (Optional) PostgreSQL instance for persistent `causal_edges` graph storage and multi-episode consolidation

### 1. Virtual Environment Setup
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables Configuration (`.env`)
Create a `.env` file in the project root directory following the model-backend configuration pattern:
```env
# Database Connection String (Use in-memory or point to PostgreSQL)
DATABASE_URL=postgresql+asyncio://postgres:postgres@localhost:5432/agent_sre

# Model Backend & Provider Selection
PRIMARY_PROVIDER=openrouter
FALLBACK_PROVIDER=anthropic
MODEL_NAME=zhipuai/glm-4-plus
MODEL_BASE_URL=https://openrouter.ai/api/v1

# API Credentials for Supported Providers
ZHIPU_API_KEY=your_zhipu_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
ZENMUX_API_KEY=your_zenmux_api_key_here
ZAI_API_KEY=your_zai_api_key_here
HF_TOKEN=your_huggingface_token_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. PostgreSQL Database Initialization (Optional)
If running with database persistence (`use_db=True`), initialize the schema using `init.sql`:
```bash
psql -U postgres -d agent_sre -f init.sql
```
Or start the complete local stack using Docker Compose:
```bash
docker-compose up -d
```

---

## Running Inference & Adversarial Benchmark Suites

### Standalone CLI Episode Inference (`inference.py`)
To run a standalone diagnostic episode against simulated infrastructure faults (tasks 1 through 4):
```bash
python inference.py --task 1
python inference.py --task 2
python inference.py --task 3
python inference.py --task 4
```

### Adversarial Evaluation Suites (`adversarial/`)
The framework includes two comprehensive evaluation suites. For full rationale and grading specifications, link directly to the engineering briefs:
- `adversarial_test_cases_brief.md` — Behavioral alignment, diagnostic calibration, non-local architecture gaps, value conflicts, and reward hacking (Tests 1–5).
- `security_adversarial_test_brief.md` — Security audits, prompt injections, quarantine bypasses, static checks, and resource exhaustion looping (Tests 6–14).

#### Grading Philosophy: Calibration Over Pass/Fail
The grading harness (`adversarial/grader.py` and `adversarial/security_audit.py`) evaluates **behavioral verification depth, diagnostic calibration curves, and escalation decisions rather than raw pass/fail flags**. An agent that immediately restarts a service might resolve a localized metric spike but fail the behavioral audit if it bypassed diagnostic log inspection (`Test 1`).

A `FAIL` verdict or gap classification (`architecture_gap`, `reward_function_gap`) indicates that the test successfully exposed an open behavioral or structural boundary in autonomous SRE alignment. It represents a valid diagnostic measurement rather than a broken test execution.

#### Execution Commands
Run the core behavioral benchmark suite (Tests 1–5) in offline in-memory mode:
```bash
python -m adversarial.runner --no-db
```

Run specific core behavioral tests by ID:
```bash
python -m adversarial.runner --tests 1 2 5 --no-db
```

Run the complete 9-part security and vulnerability audit suite (Tests 6–14):
```bash
python -m adversarial.security_audit
```

---

## Public Stress-Test Demo Summary (`SPACE_README.md`)

The repository includes a Gradio web application (`app.py` deployed via Gradio SDK on port `7860`) suitable for public stress-testing on Hugging Face Spaces. For exact public demo documentation, see `SPACE_README.md`.

### Key Features of the Public Demo
- **Bring Your Own Key (`BYOK`) Routing**: Visitors can select from the five supported provider endpoints (`config.py`) and enter their own API key (`type="password"`). When a BYOK key is provided, all session counters are bypassed.
- **Session Free-Trial Allocation**: Visitors without an API key receive **2 free evaluation runs per browser session**, powered by the confirmed working `Zhipu Direct (Tier 3)` (`glm-5.2`) fallback.
- **Global Daily Cap**: To prevent automated traffic bursts from exhausting API balance, server-side tracking (`_GLOBAL_DAILY_STATE`) limits total free-trial fallback runs to **100 runs per calendar day (UTC)** across all visitors.
- **BYO-Test Upload Safeguards**: In the Bring Your Own Test Case tab, custom `.json`, `.yaml`, or `.txt` uploads are protected by strict resource boundaries:
  - **2 MB Hard Disk Size Limit**: Rejects files exceeding 2 MB immediately before loading into memory.
  - **15,000-Character Prompt Limit**: Truncates lengthy log dumps to retain the first 10,000 and last 4,000 characters, embedding a clear truncation summary note to protect LLM context windows.
  - **Target Service Clamping**: Whitelists target services against the known `MockMesh` topology (`auth`, `api-gateway`, `user-service`, `payment-service`).

---

## Current Known Limitations

To ensure transparency regarding system capabilities, current limitations are stated plainly below:
1. **Simulated Telemetry Simplifications**: The `MockMesh` environment (`mock_infra/telemetry.py`) generates metrics using parametric decay formulas (`p99_latency_ms`, `error_rate_pct`, `saturation_pct`) plus Gaussian noise. It does not capture the full chaotic variance, kernel-level thread deadlocks, or network packet drops of real production Linux operating systems.
2. **Causal Edge Tracking Requires Historical Data**: The `causal_edges` retroactive tracking mechanism (`memory/models.py`) depends on historical trajectory consolidation (`consolidate.py`). It does not prevent zero-day non-local architectural side effects during an agent's first execution against an unknown topology.
3. **Unbuilt Optimization & Self-Editing Loops**: Direct Preference Optimization (`DPO`) fine-tuning and SICA self-editing loop behaviors are unbuilt/deferred and do not actively mutate system prompts or agent weights at runtime.
4. **Static Provider Failover Logic**: Multi-provider failover in `provider_pool` relies on HTTP error status detection (`402`, `429`, `503`) and does not dynamically measure real-time latency tokens-per-second (`TPS`) before routing inference requests.
