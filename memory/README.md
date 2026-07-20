# Memory & Causal Tracking Subsystem (`memory/`)

This directory contains the persistent experience storage, similarity-based retrieval, and causal graph tracking routines for the Agentic SRE environment.

## Module Purpose

The memory subsystem records agent execution trajectories (actions, observations, decision rationale, and reward outcomes) and extracts structural relationships between remediation attempts and subsequent system state transitions.

For the full architectural specification and requirements of the memory layer, refer directly to `agentic_sre_rebuild_brief.md`.

## Component Summary

- `models.py` — Defines SQLAlchemy ORM tables (`episodes`, `decisions`, `causal_edges`) supporting both asynchronous (`AsyncEngine`) and synchronous (`Engine`) database backends.
- `db.py` — Engine initialization and session management utilities for PostgreSQL or SQLite connections.
- `write.py` — Trace recording functions (`log_episode_decision`, `finalize_episode_trace`) that persist structured agent actions and observation snapshots during active evaluation.
- `retrieve.py` — Similarity search and runbook query logic (`retrieve_relevant_experiences`, `get_causal_edges_for_service`) used to supply historical context to the reasoning loop (`agents/reasoning_loop.py`).
- `consolidate.py` — Post-episode batch processing routine (`consolidate_episode_memory`) that extracts multi-step causal credit and updates edge weights.
- `credit_assignment.py` — Algorithmic logic computing temporal discount weights across action sequences (`compute_trajectory_credit`).

## Verified vs. Deferred Capabilities

To maintain exact alignment with confirmed system state, the operational status of memory features is categorized below:

### Confirmed Working (Current Session)
- **Trajectory Persistence**: Recording full episode decision histories (`episodes`, `decisions`) via `write.py`.
- **Runbook & Causal Edge Retrieval**: Querying historical incidents and known causal edges (`retrieve.py`) to inform current diagnostic decisions.
- **Post-Episode Consolidation**: Offline extraction and persistence of `causal_edges` relationships (`consolidate.py`) with configurable database backends (`memory/models.py`).

### Deferred & Partially Built Features
- **DPO Preference Optimization**: Direct Preference Optimization (`DPO`) dataset generation and model fine-tuning are **not implemented**. The retrieval module (`retrieve.py`) and consolidation job calculate a `no-match-rate` metric (`no_match_rate`), which is instrumented solely as a measurement trigger to inform future training dataset curation decisions.
- **Self-Editing (`SICA`-style) Agent Loop**: Self-Improving Causal Agent (`SICA`) self-editing routines are **deferred** behind a plateau trigger (`R_t` improvement plateau across consecutive evaluation epochs) and are not currently built.
- **`causal_edges` Retroactive-Detection Mechanism (`Test 3`)**: The `causal_edges` graph persistence layer is confirmed as a **measurement and mitigation tool, not an immediate pre-action prevention mechanism**. When an agent executes an action (`api-gateway:scale_up`) that passes initial Quarantine checks but causes a delayed downstream consequence (`user-service:pool_exhaustion`) several steps later, the causal relationship is detected and stored post-episode (`consolidate.py`). This allows subsequent episodes to retrieve the edge and apply informed mitigation strategies, but does not block the initial occurrence.
