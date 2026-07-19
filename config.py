"""
config.py — Typed config via pydantic-settings. Single source of truth for all tunables.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Postgres ──────────────────────────────────────────────────────────────
    postgres_url: str = Field(
        default="postgresql+asyncpg://sre:sre_secret@localhost:15432/sre_memory",
        description="Async SQLAlchemy URL for the memory Postgres instance.",
    )

    # ── Acting model (does diagnosis/remediation reasoning during episodes) ───
    # Provider-agnostic: any OpenAI-compatible endpoint.
    # Switching provider = change these three env vars, zero code changes.
    model_backend: str = Field(
        default="openai_compatible",
        description="'openai_compatible' (GLM, Groq, OpenRouter, Z.ai, ZenMux) or 'claude'.",
    )
    model_base_url: str = Field(
        default="https://api.zenmux.ai/v1",
        description=(
            "Base URL for the OpenAI-compatible acting-model endpoint.\n"
            "  Free/trial: https://api.zenmux.ai/v1  or  https://api.z.ai/v1\n"
            "  Paid direct: https://open.bigmodel.cn/api/paas/v4/\n"
            "  Aggregator:  https://openrouter.ai/api/v1"
        ),
    )
    model_api_key: str = Field(
        default="",
        description="API key for the acting model endpoint.",
    )
    model_name: str = Field(
        default="z-ai/glm-5.2-free",
        description=(
            "Model identifier string (provider-dependent).\n"
            "  ZenMux free:     z-ai/glm-5.2-free\n"
            "  Z.ai direct:     glm-5.2\n"
            "  Zhipu direct:    glm-5.2\n"
            "  OpenRouter:      zhipu-ai/glm-5.2 (check their model list)"
        ),
    )
    model_thinking_mode: str = Field(
        default="on",
        description=(
            "GLM-5.2 thinking mode. 'on' for the agent reasoning loop (always), "
            "'off' only for cheap classification calls if any get added later. "
            "Passed as extra_body to the API call."
        ),
    )

    # ── Backup provider API keys for automatic failover rotation ──────────────
    zenmux_api_key: Optional[str] = Field(
        default="",
        description="Backup API key for ZenMux (Tier 1).",
    )
    zai_api_key: Optional[str] = Field(
        default="",
        description="Backup API key for Z.ai Direct (Tier 2).",
    )
    zhipu_api_key: Optional[str] = Field(
        default="",
        description="Backup API key for Zhipu Direct (Tier 3).",
    )
    openrouter_api_key: Optional[str] = Field(
        default="",
        description="Backup API key for OpenRouter (Tier 4).",
    )
    hf_api_key: Optional[str] = Field(
        default="",
        description="Backup API key for HuggingFace Router (Tier 0).",
    )

    # ── Future DPO fine-tune target (NOT the acting model, NOT built yet) ────
    # Kept as separate named entries from day one so the two roles are never
    # conflated. See brief §2.5 and §3 — DPO is a non-goal for this build.
    # These are placeholders; populate them when DPO work begins.
    dpo_model_name: str = Field(
        default="Qwen/Qwen3-6B",
        description=(
            "Target model for future DPO fine-tuning. Deliberately smaller than the "
            "acting model — needs to fit on local/Colab hardware for fast iteration. "
            "Candidates: Qwen3.6-35B-A3B, MiniMax-M2.1. NOT GLM-5.2."
        ),
    )
    dpo_model_base_url: str = Field(
        default="",
        description="Base URL for the DPO fine-tune target (local vLLM, Colab, etc.). Unused until DPO build begins.",
    )
    dpo_model_api_key: str = Field(
        default="",
        description="API key for the DPO model endpoint. Unused until DPO build begins.",
    )

    # ── Claude backend (only when model_backend=claude) ───────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key (used when model_backend=claude).")
    claude_model: str = Field(default="claude-sonnet-4-5", description="Claude model identifier.")

    # ── Retry / rate-limit handling ───────────────────────────────────────────
    model_max_retries: int = Field(
        default=5,
        description="Max retry attempts for model calls (free/trial tiers are rate-limited).",
    )
    model_retry_base_delay: float = Field(
        default=2.0,
        description="Base delay (seconds) for exponential backoff on model retries.",
    )
    model_retry_max_delay: float = Field(
        default=60.0,
        description="Max delay cap (seconds) for exponential backoff.",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name for state embeddings.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine similarity floor. Below this = no-match event.",
    )
    retrieval_top_k: int = Field(
        default=3,
        ge=1,
        description="Max lessons returned per retrieval query.",
    )

    # ── Consolidation ─────────────────────────────────────────────────────────
    consolidation_interval_minutes: int = Field(
        default=60,
        ge=1,
        description="How often the offline consolidation job runs.",
    )
    consolidation_min_cluster_size: int = Field(
        default=3,
        ge=1,
        description="Minimum decisions per cluster before writing a lesson.",
    )
    consolidation_cluster_distance: float = Field(
        default=0.25,
        ge=0.0,
        le=2.0,
        description="Cosine distance threshold for cluster merging.",
    )
    lesson_decay_halflife_days: int = Field(
        default=30,
        ge=1,
        description="Half-life (days) for decay of stale lessons / edges.",
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_version: str = Field(
        default="v2.0.0",
        description="Prompt/scaffold version string stored on every episode.",
    )

    # ── Episode limits ────────────────────────────────────────────────────────
    max_steps_per_episode: int = Field(default=20, ge=1)
    episode_timeout_seconds: int = Field(default=300, ge=10)
    window_probes: int = Field(default=3, ge=1, description="Number of sustained metric samples to probe during evaluation window.")
    probe_interval_s: float = Field(default=2.0, ge=0.0, description="Interval in seconds between sustained metric probes.")


# Singleton — import and use this everywhere
settings = Settings()
