-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────
-- One row per episode
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episodes (
    episode_id          UUID PRIMARY KEY,
    task_id             TEXT NOT NULL,
    agent_version       TEXT NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    outcome             TEXT,           -- success / failure / partial
    final_reward        FLOAT,
    resolution_summary  TEXT
);

-- ─────────────────────────────────────────────
-- One row per action / decision
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS decisions (
    decision_id         UUID PRIMARY KEY,
    episode_id          UUID REFERENCES episodes(episode_id),
    step_index          INT NOT NULL,
    state_signature     TEXT NOT NULL,          -- serialized observation at this step
    state_embedding     VECTOR(384),            -- all-MiniLM-L6-v2
    action_type         TEXT NOT NULL,
    action_payload      JSONB NOT NULL,
    result_stdout       TEXT,
    result_stderr       TEXT,
    exit_code           INT,
    credit_label        TEXT,                   -- success / failure / neutral — backfilled post-episode
    quarantine_flag     BOOLEAN DEFAULT FALSE,
    no_match_flag       BOOLEAN DEFAULT FALSE,
    quarantine_reason   TEXT,                   -- verbatim reason from frozen gate
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────────────────────
-- Causal edges between decisions
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS causal_edges (
    edge_id             UUID PRIMARY KEY,
    from_decision       UUID REFERENCES decisions(decision_id),
    to_decision         UUID REFERENCES decisions(decision_id),
    relation_type       TEXT,                   -- caused / preceded / remediated_by
    confidence          FLOAT DEFAULT 1.0,
    last_reinforced     TIMESTAMPTZ DEFAULT now()
);

-- ─────────────────────────────────────────────
-- Consolidated, queryable memory (retrieval-before-decision reads ONLY this)
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id                   UUID PRIMARY KEY,
    state_cluster_embedding     VECTOR(384),
    best_action                 JSONB,
    outcome_summary             TEXT,
    sample_count                INT,
    success_rate                FLOAT,
    last_updated                TIMESTAMPTZ
);

-- ─────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS decisions_embedding_idx
    ON decisions USING ivfflat (state_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS decisions_episode_step_idx
    ON decisions (episode_id, step_index);

CREATE INDEX IF NOT EXISTS causal_edges_from_idx
    ON causal_edges (from_decision);

CREATE INDEX IF NOT EXISTS lessons_embedding_idx
    ON lessons USING ivfflat (state_cluster_embedding vector_cosine_ops)
    WITH (lists = 100);
