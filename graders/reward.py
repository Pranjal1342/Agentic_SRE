"""
graders/reward.py — Dense reward function.

Measures golden-signal restoration for each service:
  - p99_latency_ms: lower is better; target is the healthy baseline
  - error_rate_pct: lower is better
  - saturation_pct: lower is better

Final reward is a weighted average across all services and signals,
normalized to [0, 1]. 1.0 = all targets met exactly or better.

The reward function is deliberately dense (not sparse) so that even partial
remediation yields a gradient signal.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union, List
import structlog

log = structlog.get_logger()


# Signal weights (must sum to 1.0 across the 3 signals)
_SIGNAL_WEIGHTS = {
    "p99_latency_ms": 0.4,
    "error_rate_pct": 0.4,
    "saturation_pct": 0.2,
}

# Absolute max values used for normalization (beyond these = 0 score for that signal)
_SIGNAL_MAX = {
    "p99_latency_ms": 5000.0,   # 5 seconds = effectively broken
    "error_rate_pct": 50.0,      # 50% error rate = catastrophic
    "saturation_pct": 100.0,
}


def _signal_score(current: float, target: float, signal: str) -> float:
    """
    Score a single signal for a single service.
    Returns 1.0 if current <= target, scales down linearly above target,
    hits 0.0 at _SIGNAL_MAX.
    """
    if current <= target:
        return 1.0  # At or better than target

    max_val = _SIGNAL_MAX.get(signal, 100.0)
    # Linear decay from target → max_val
    excess = current - target
    max_excess = max(max_val - target, 1e-6)
    score = 1.0 - min(excess / max_excess, 1.0)
    return max(0.0, score)


def compute_reward(
    final_metrics: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]] = None,
    golden_targets: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    task_id: Optional[str] = None,
    final_obs: Optional[Union[Dict[str, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]]] = None,
    actions_taken: Optional[list] = None,
    quarantine_blocks: Optional[int] = None,
    **kwargs: Any,
) -> float:
    """
    Compute the final episode reward ∈ [0, 1].

    Supports:
      - Positional/direct: compute_reward(final_metrics, golden_targets)
      - Keyword: compute_reward(task_id=task_id, final_obs=final_obs, actions_taken=..., quarantine_blocks=...)
      - Temporal window: final_metrics/final_obs can be a single snapshot Dict or a List of snapshots
        over a sustained verification window. When a List is provided, signals are scored on their
        worst-case (max) value across the window to penalize temporary masking (e.g. scale_up on a leak).
    """
    if final_metrics is None and final_obs is not None:
        final_metrics = final_obs

    if golden_targets is None:
        if task_id is not None:
            try:
                from server.pipeline import TASK_REGISTRY, register_task
                if task_id not in TASK_REGISTRY:
                    try:
                        from tasks import task_1, task_2, task_3, task_4
                        register_task("task_1", task_1)
                        register_task("task_2", task_2)
                        register_task("task_3", task_3)
                        register_task("task_4", task_4)
                    except Exception:
                        pass
                if task_id in TASK_REGISTRY:
                    golden_targets = TASK_REGISTRY[task_id].golden_signal_targets()
                else:
                    log.warning("reward.golden_targets_fallback", task_id=task_id, reason="task_id not found in TASK_REGISTRY (using default targets)")
            except Exception as e:
                log.warning("reward.golden_targets_fallback", task_id=task_id, reason=f"exception looking up task_id: {e}")
        else:
            log.warning("reward.golden_targets_fallback", reason="no golden_targets or task_id provided to compute_reward")

        if golden_targets is None:
            golden_targets = {
                "auth": {"error_rate_pct": 1.0, "p99_latency_ms": 200.0},
                "payment-service": {"error_rate_pct": 1.0, "p99_latency_ms": 500.0},
                "api-gateway": {"error_rate_pct": 1.0, "p99_latency_ms": 200.0},
                "user-service": {"error_rate_pct": 0.5, "p99_latency_ms": 300.0},
            }

    if not final_metrics or not golden_targets:
        return 0.0

    snapshots: List[Dict[str, Dict[str, Any]]] = (
        final_metrics if isinstance(final_metrics, list) else [final_metrics]
    )

    total_score = 0.0
    total_weight = 0.0

    for service, targets in golden_targets.items():
        for signal, target_val in targets.items():
            weight = _SIGNAL_WEIGHTS.get(signal, 0.1)
            # Evaluate worst-case (max value) across all temporal snapshots in the verification window
            current_val = max(
                snapshot.get(service, {}).get(signal, _SIGNAL_MAX.get(signal, 100.0))
                for snapshot in snapshots
            )
            score = _signal_score(current_val, target_val, signal)
            total_score += score * weight
            total_weight += weight

    if total_weight == 0.0:
        return 0.0

    reward = total_score / total_weight
    return round(min(max(reward, 0.0), 1.0), 6)


def reward_breakdown(
    final_metrics: Dict[str, Dict[str, Any]],
    golden_targets: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Return per-service, per-signal score breakdown for logging/debugging.
    """
    breakdown: Dict[str, Any] = {}
    for service, targets in golden_targets.items():
        current = final_metrics.get(service, {})
        service_scores = {}
        for signal, target_val in targets.items():
            current_val = current.get(signal, _SIGNAL_MAX.get(signal, 100.0))
            score = _signal_score(current_val, target_val, signal)
            service_scores[signal] = {
                "current": round(current_val, 3),
                "target": target_val,
                "score": round(score, 4),
            }
        breakdown[service] = service_scores
    breakdown["final_reward"] = compute_reward(final_metrics, golden_targets)
    return breakdown
