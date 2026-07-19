# Engineering Brief: Agentic SRE OpenEnv v2 (Rebuild)

Paste this whole document into Antigravity as the task brief. Read it fully before writing any code. If anything below is ambiguous or you'd need to guess, ask me first — do not silently assume on the flagged open questions in Section 8.

---

## 1. Context

Rebuilding `agentic-sre-openenv` from scratch. This is a containerized RL/agentic environment (OpenEnv spec) simulating SRE incident response: a mock microservice mesh, an FSM-driven episode lifecycle, a typed action/observation space, a dense reward function, and a Quarantine Agent that gates remediation actions for safety (prompt-injection defense + structural type restrictions on destructive commands).

The original design deliberately had **zero external database dependencies** — FAISS and a document-hierarchy graph were built offline and baked into the Docker image, read-only at runtime, for a clean `docker build && docker run` startup and full episode isolation (no state persists across episodes).

**This rebuild changes that on purpose.** The goal now is an agent that accumulates experience across episodes — remembers specific past mistakes and successes, and retrieves them before making similar decisions again. That requires real persistence. Postgres + pgvector is the chosen store (already used in an earlier version of this project, so it's a known-good stack, not a new unknown).

## 2. Goal (what "self-learning" means here, precisely)

Not full online RL. Not weight updates in this pass. Specifically:

1. **Agentic loop** (single episode): FSM-orchestrated, tool-using, plans before acting, reflects on failures within the episode. This layer already mostly exists — preserve and formalize it, don't redesign it.
2. **Case-based memory** (across episodes): every decision the agent makes is logged with enough context to later say "this specific choice, in this specific situation, worked or didn't." Failures get flagged with a precise reason via the Quarantine Agent's own rejection logic wherever possible — that's a free, precise credit-assignment signal, use it.
3. **Retrieval-before-decision**: before choosing an action, the agent queries a consolidated memory store for similar past situations and their outcomes, and that retrieval is injected into context.
4. **Consolidation**: raw per-decision logs get periodically clustered/summarized into a smaller, queryable "lessons" table. Retrieval reads from the consolidated table, never the raw ledger.
5. **Instrumentation for a future decision, not a current build**: log the rate at which retrieval finds *no* matching prior situation. This metric — not intuition — decides later whether weight-level training (DPO on ranked rollout pairs) is actually needed. Do not build DPO in this pass. Just make sure the metric exists so that decision can be made with evidence later.

## 3. Explicit Non-Goals for This Build

Do not implement these now. They were each considered and deliberately deferred — don't reintroduce them without asking:

- **Full online RL (PPO/GRPO)** — too expensive for the hardware (see Section 4), and there's no evidence yet that memory + retrieval doesn't already solve most of the gap.
- **DPO training loop** — instrument the trigger metric (Section 2.5) only; don't build the training pipeline itself yet.
- **Self-editing agents (SICA/SEAL-style prompt or weight self-modification)** — explicitly premature until memory+retrieval plateaus over a real measurement window. Do not let any agent component rewrite its own prompts or the Quarantine Agent's criteria.
- **Multi-agent persona routing / model orchestration (Fugu-style)** — an optimization layer on top of a working loop, not part of building the loop.
- **Auto-curriculum / task generation (Exploration & Discovery pattern)** — needs a stable baseline to compare against first.
- **JEPA, TEE** — no identified failure mode in this repo requires either.
- **Letting the Quarantine Agent evolve from accumulated memory.** This is a hard boundary, not a preference: the Quarantine/verification gate must stay frozen and read-only relative to the memory system it feeds. It writes rejection reasons into memory; nothing writes back into its criteria. If this boundary is unclear anywhere in the implementation, stop and ask rather than guessing which side of the line something falls on.

## 4. Hardware / Environment Constraints

- Local dev: RTX 3050 laptop GPU, 4GB VRAM. Anything requiring real training compute goes to Colab, not local.
- Antigravity IDE (VSCodium-based), Claude Sonnet as the coding agent (you).
- Docker remains the deployment target, but the "no external database dependency" property is intentionally being given up in this rebuild — Postgres is now a required runtime dependency, not baked-in static data. This is a known, accepted tradeoff, not an oversight — don't try to route around it by re-baking a static index.

## 5. Architecture — Layers

```
┌─────────────────────────────────────────────────────────┐
│ Environment (FSM, mock mesh, mock DB, telemetry)          │  <- stays fully episode-isolated,
│  - reset() wipes ALL environment state every episode      │     exactly as before. DO NOT
│  - unchanged design from original repo                    │     let memory leak into this layer.
└─────────────────────────────────────────────────────────┘
                         │ tool calls (via MCP)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MCP tool layer                                             │  <- wraps mock_infra actions as
│  - diagnostic_query, log_inspection, remediation,          │     MCP tools. Standardizes the
│    submit_resolution exposed as MCP tools                  │     schema, nothing behavioral
└─────────────────────────────────────────────────────────┘     changes yet at this layer.
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Agent reasoning loop (per episode)                          │  <- Planning + Reflection patterns
│  1. State goal (target golden-signal values) before acting  │
│  2. Query memory (lessons table) for similar past states    │
│  3. State rationale for chosen action BEFORE calling tool    │
│     (ToRA-style: reason -> act -> observe -> revise)         │
│  4. On failed/rejected action: reflect, retry with revision │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Quarantine / Verification Gate                              │  <- FROZEN. Reads memory to inform
│  - unchanged logic from original repo                       │     its own decisions is fine.
│  - writes precise rejection reasons to memory                │     Being WRITTEN TO by memory/
│  - NEVER updated by accumulated memory                       │     learned patterns is NOT fine.
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Memory service (Postgres + pgvector)                        │  <- lives OUTSIDE the environment's
│  - separate connection/service from env's mock_infra         │     reset() boundary. Persists
│  - episodes, decisions, causal_edges, lessons tables          │     across episodes on purpose.
│  - see Section 6 for schema                                  │
└─────────────────────────────────────────────────────────┘
                         │ (scheduled, offline)
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Consolidation job                                            │  <- runs between training batches,
│  - clusters raw `decisions` rows into `lessons` rows          │     NOT live during an episode.
│  - decays stale/unreinforced edges and lessons                │     See Section 8, open question 1.
└─────────────────────────────────────────────────────────┘
```

**Hard rule:** the Environment layer's `reset()` must remain exactly as isolated as it was in the original design — no environment state persists across episodes. The Memory service is a completely separate persistence boundary. Do not merge these two concerns into one database or one connection pool.

## 6. Memory Schema (Postgres + pgvector)

```sql
-- One row per episode
CREATE TABLE episodes (
    episode_id      UUID PRIMARY KEY,
    task_id         TEXT NOT NULL,
    agent_version   TEXT NOT NULL,          -- prompt/scaffold version string
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ,
    outcome         TEXT,                   -- success / failure / partial
    final_reward    FLOAT,
    resolution_summary TEXT
);

-- One row per action/decision (per-decision granularity, not per-episode)
CREATE TABLE decisions (
    decision_id     UUID PRIMARY KEY,
    episode_id      UUID REFERENCES episodes(episode_id),
    step_index      INT NOT NULL,
    state_signature TEXT NOT NULL,          -- serialized observation at this step
    state_embedding VECTOR(384),            -- same all-MiniLM-L6-v2 model already used for runbook RAG
    action_type     TEXT NOT NULL,
    action_payload  JSONB NOT NULL,
    result_stdout   TEXT,
    result_stderr   TEXT,
    exit_code       INT,
    credit_label    TEXT,                   -- success / failure / neutral — filled in AFTER episode resolves
    quarantine_flag BOOLEAN DEFAULT FALSE,
    quarantine_reason TEXT,                 -- verbatim reason from the frozen gate
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- Causal edges between decisions
CREATE TABLE causal_edges (
    edge_id         UUID PRIMARY KEY,
    from_decision   UUID REFERENCES decisions(decision_id),
    to_decision     UUID REFERENCES decisions(decision_id),
    relation_type   TEXT,                   -- 'caused' / 'preceded' / 'remediated_by'
    confidence      FLOAT DEFAULT 1.0,
    last_reinforced TIMESTAMPTZ DEFAULT now()
);

-- Consolidated, queryable memory — THIS is what retrieval-before-decision reads from
CREATE TABLE lessons (
    lesson_id        UUID PRIMARY KEY,
    state_cluster_embedding VECTOR(384),
    best_action      JSONB,
    outcome_summary  TEXT,
    sample_count     INT,
    success_rate     FLOAT,
    last_updated     TIMESTAMPTZ
);

CREATE INDEX ON decisions USING ivfflat (state_embedding vector_cosine_ops);
CREATE INDEX ON decisions (episode_id, step_index);
CREATE INDEX ON causal_edges (from_decision);
CREATE INDEX ON lessons USING ivfflat (state_cluster_embedding vector_cosine_ops);
```

Rules:
- Retrieval-before-decision queries `lessons` only. Never query raw `decisions` at inference time — it will get noisy and slow as it grows; it's a ledger for consolidation to read, not a live lookup table.
- `credit_label` starts NULL, backfilled by a post-episode job once `outcome` is known.
- `quarantine_reason` must be the gate's actual textual reason, not just the boolean flag — that's the precise credit-assignment signal, don't throw it away.
- Vector entry (`decisions` row) and any related `causal_edges` rows from the same decision must be written in a single transaction — no partial writes.

## 7. Metrics to Instrument From Day One

Even though DPO/RL isn't being built yet, log these now so the decision to build it later is evidence-based, not a guess:

- **No-match rate**: fraction of decisions where retrieval-before-decision found no `lessons` row above a similarity threshold. Track per task, over time. This is the trigger metric — falling/low means memory is doing the job; persistently high/flat means the agent is hitting genuinely novel situations memory can't help with, which is the actual signal that weight-level training is needed.
- **Reward spread**: distribution of `final_reward` across episodes/sampled rollouts per task. Needed later to check whether `R_t` actually discriminates well enough between rollouts to generate usable DPO preference pairs — check this before ever starting that work, don't assume it.
- **Quarantine rejection rate over time**, per task — sanity check that the gate isn't drifting in behavior even though its logic is frozen (would indicate a bug, not learning).

## 8. Open Questions — Resolve With Me, Don't Guess

1. **Concurrency during training.** If multiple episodes run in parallel (the environment supports continuous WebSocket-based training), do they need to see each other's memory writes mid-batch, or is it acceptable that `lessons` only updates between batches via the offline consolidation job? **Default assumption if not told otherwise: consolidation is fully offline/scheduled, parallel episodes do not see each other's writes mid-batch, only `episodes`/`decisions` writes happen live.** Flag me if this default is wrong for the intended training setup before building the consolidation job.
2. **The "no new dependencies" constraint from earlier design discussion is being treated as resolved-soft**, since Postgres itself is now an accepted new dependency. Default: memory + retrieval get built now; DPO training pipeline does not get built now regardless (see Section 3) — this isn't gated by the dependency question, it's gated by not having evidence it's needed yet.

## 9. Suggested File/Module Layout

Extend the existing layout, don't restructure what already works:

```
📦 agentic-sre-openenv/
├── server/                     # unchanged: FSM, HTTP/WS endpoints, pipeline
├── mock_infra/                 # unchanged: mesh, mock DB, telemetry
├── agents/
│   └── quarantine_agent.py     # unchanged logic; add memory write-out of rejection reasons
├── graders/                    # unchanged: reward computation
├── mcp/                        # NEW: MCP tool wrappers around mock_infra actions
│   └── tools.py
├── memory/                     # NEW: everything from Section 6-7
│   ├── db.py                   # connection/session management, separate from env's mock_infra
│   ├── models.py                # SQLAlchemy/pydantic models for episodes/decisions/edges/lessons
│   ├── write.py                 # per-decision write path (called from reasoning loop)
│   ├── retrieve.py               # retrieval-before-decision, queries `lessons` only
│   ├── credit_assignment.py     # post-episode backward labeling from quarantine reasons
│   └── consolidate.py           # scheduled clustering job: decisions -> lessons, decay
├── rag/                         # existing runbook RAG — keep as is, separate from experience memory
├── tasks/                       # unchanged
├── knowledge_base/              # unchanged
└── inference.py                  # baseline eval — extend to log no-match rate + reward spread
```

## 10. Build Order

Implement in this order; each step should be independently testable before moving to the next:

1. Stand up Postgres + pgvector locally (docker-compose addition), create schema from Section 6.
2. Wrap `mock_infra` actions as MCP tools (Section 5, tool layer) — no behavior change yet, just the interface.
3. Add explicit goal-statement + rationale-before-action to the reasoning loop (Planning pattern).
4. Wire per-decision writes to `decisions` table during episode execution.
5. Wire quarantine rejection reasons into `decisions.quarantine_reason` / `quarantine_flag`.
6. Build post-episode backward credit assignment (`credit_assignment.py`).
7. Build the consolidation job (`consolidate.py`) — even a simple version (cluster by embedding similarity, aggregate outcomes) is enough to start.
8. Build retrieval-before-decision, querying `lessons`, inject into agent context.
9. Instrument no-match rate and reward spread logging in `inference.py`.
10. Re-run baseline eval on `task_1`–`task_4`; compare against the original scores (`task_3` = 0.6190 is the one to watch — that's the multi-stage causal decomposition failure this whole memory layer is meant to help with).

Stop after step 10 and report back before touching anything in Section 3's non-goals list.
