"""
agents/quarantine_agent.py — FROZEN gate. Logic must NOT be modified by the
memory system. Reads memory to inform decisions is permitted.
Being written to by memory/learned patterns is NOT permitted.

Validates remediation actions against:
1. Structural type restrictions (action_type must be in allowlist, params typed)
2. Prompt-injection defense (no shell metacharacters / template injection in params)
3. Scope restrictions (can't restart services with zero downtime tolerance)

Returns (allowed: bool, reason: str). Writes rejection reasons to memory via
write.py — but is never updated by that memory. This is a hard boundary.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

import structlog

log = structlog.get_logger(__name__)

# ── Frozen allowlists (never updated by memory) ───────────────────────────────

_ALLOWED_ACTION_TYPES = frozenset([
    "restart_service",
    "scale_up",
    "rollback",
    "increase_db_pool",
    "kill_slow_queries",
    "vacuum_analyze",
    "circuit_breaker",
])

_ALLOWED_TARGETS = frozenset([
    "auth",
    "api-gateway",
    "user-service",
    "payment-service",
    "database",
])

# Actions that must NOT be applied to payment-service without explicit high-reward context
_HIGH_RISK_ACTIONS = frozenset(["restart_service", "rollback"])

_HIGH_RISK_PROTECTED_TARGETS = frozenset(["payment-service"])

# Prompt injection patterns to detect in param values
_INJECTION_PATTERNS = re.compile(
    r"(\$\{|`|\||;|&&|\|\||>|<|python |import |eval\(|exec\(|__import__|curl |wget )",
    re.IGNORECASE,
)

# Param value length limit
_MAX_PARAM_VALUE_LENGTH = 256


class QuarantineAgent:
    """
    FROZEN gate. Logic is immutable with respect to the memory system.
    This class may READ from memory for informational purposes in future,
    but its check() logic is never updated by memory writes.
    """

    def check(
        self,
        action_type: str,
        target: str,
        params: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Validate a remediation action. Returns (allowed, reason).
        If allowed=False, reason is the precise rejection explanation —
        this string is stored verbatim in decisions.quarantine_reason.
        """
        # ── Check 1: action_type allowlist ────────────────────────────────────
        if action_type not in _ALLOWED_ACTION_TYPES:
            reason = (
                f"action_type '{action_type}' is not in the approved remediation allowlist. "
                f"Approved types: {sorted(_ALLOWED_ACTION_TYPES)}"
            )
            self._log_rejection(action_type, target, reason)
            return False, reason

        # ── Check 2: target allowlist ─────────────────────────────────────────
        if target not in _ALLOWED_TARGETS:
            reason = (
                f"target '{target}' is not in the approved service allowlist. "
                f"Approved targets: {sorted(_ALLOWED_TARGETS)}"
            )
            self._log_rejection(action_type, target, reason)
            return False, reason

        # ── Check 3: high-risk action on protected target ─────────────────────
        if action_type in _HIGH_RISK_ACTIONS and target in _HIGH_RISK_PROTECTED_TARGETS:
            reason = (
                f"action_type '{action_type}' on target '{target}' is a high-risk operation "
                f"on a protected service. Requires elevated approval (not available in this scope)."
            )
            self._log_rejection(action_type, target, reason)
            return False, reason

        # ── Check 4: param type validation ───────────────────────────────────
        param_error = self._validate_params(action_type, params)
        if param_error:
            reason = f"Invalid parameters for '{action_type}': {param_error}"
            self._log_rejection(action_type, target, reason)
            return False, reason

        # ── Check 5: prompt injection defense ────────────────────────────────
        injection_error = self._check_injection(params)
        if injection_error:
            reason = f"Prompt injection detected in params: {injection_error}"
            self._log_rejection(action_type, target, reason)
            return False, reason

        log.debug("quarantine.allowed", action_type=action_type, target=target)
        return True, ""

    # ── Param validators (frozen) ─────────────────────────────────────────────

    def _validate_params(self, action_type: str, params: Dict[str, Any]) -> str:
        """Returns error string if params are invalid, empty string if valid."""
        if action_type == "scale_up":
            factor = params.get("factor", 1.5)
            try:
                f = float(factor)
                if not (1.0 < f <= 10.0):
                    return f"'factor' must be in (1.0, 10.0], got {f}"
            except (TypeError, ValueError):
                return f"'factor' must be a number, got {type(factor).__name__}"

        if action_type == "increase_db_pool":
            pool_size = params.get("pool_size", 50)
            try:
                p = int(pool_size)
                if not (10 <= p <= 500):
                    return f"'pool_size' must be in [10, 500], got {p}"
            except (TypeError, ValueError):
                return f"'pool_size' must be an integer, got {type(pool_size).__name__}"

        return ""

    def _check_injection(self, params: Dict[str, Any]) -> str:
        """Scan all param values for injection patterns."""
        params_str = json.dumps(params, ensure_ascii=False)
        if len(params_str) > _MAX_PARAM_VALUE_LENGTH * 10:
            return f"params payload too large ({len(params_str)} chars)"
        match = _INJECTION_PATTERNS.search(params_str)
        if match:
            return f"suspicious pattern '{match.group()}' found in params"
        return ""

    def _log_rejection(self, action_type: str, target: str, reason: str) -> None:
        log.warning(
            "quarantine.rejected",
            action_type=action_type,
            target=target,
            reason=reason,
        )
