"""
memory/credit_assignment.py — Post-episode backward credit labeling.

After an episode closes and its outcome is known, this job backfills
credit_label on all decisions for that episode. Uses quarantine rejection
reasons as precise credit-assignment signals where available (free signal
from the frozen gate — don't throw it away).

Credit labels:
  - success  : action contributed positively to episode resolution
  - failure  : action was wrong or directly caused degradation
  - neutral  : informational action (diagnostic_query, log_inspection)
  - quarantine_blocked : rejected by gate (strongest failure signal)
"""
from __future__ import annotations

import structlog
from sqlalchemy import select, update

from memory.db import get_db_session
from memory.models import Decision

log = structlog.get_logger(__name__)

# Actions that are always informational (neutral credit regardless of outcome)
DIAGNOSTIC_ACTIONS = frozenset(["diagnostic_query", "log_inspection"])


async def backfill_credit_labels(episode_id: str, outcome: str) -> int:
    """
    Backfill credit_label for all decisions in an episode.
    Returns the count of rows updated.

    Credit assignment logic:
    1. Quarantine-blocked decisions → always 'quarantine_blocked' (strongest failure signal)
    2. Diagnostic actions → always 'neutral'
    3. Remediation actions in a successful episode → 'success'
    4. Remediation actions in a failed/partial episode → 'failure'
       UNLESS the action was the last before resolution in partial (→ 'neutral')
    """
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(Decision)
                .where(Decision.episode_id == episode_id)
                .order_by(Decision.step_index)
            )
            decisions = result.scalars().all()

            if not decisions:
                log.warning("credit_assignment.no_decisions", episode_id=episode_id)
                return 0

            updated = 0
            total_steps = len(decisions)

            for i, decision in enumerate(decisions):
                if decision.credit_label is not None:
                    # Already labeled (shouldn't happen but be safe)
                    continue

                label = _assign_label(
                    decision=decision,
                    outcome=outcome,
                    step_position=i,
                    total_steps=total_steps,
                )
                decision.credit_label = label
                session.add(decision)
                updated += 1

        log.info(
            "credit_assignment.complete",
            episode_id=episode_id,
            outcome=outcome,
            decisions_labeled=updated,
        )
        return updated
    except Exception as exc:
        log.warning("credit_assignment.db_unavailable", error=str(exc), note="Skipping backfill")
        return 0


def _assign_label(
    decision: Decision,
    outcome: str,
    step_position: int,
    total_steps: int,
) -> str:
    """Determine credit label for a single decision."""
    # Quarantine-blocked → strongest failure signal (gate provided precise reason)
    if decision.quarantine_flag and decision.quarantine_reason:
        return "quarantine_blocked"

    # Diagnostic/observational actions → always neutral
    if decision.action_type in DIAGNOSTIC_ACTIONS:
        return "neutral"

    # submit_resolution → label based on episode outcome
    if decision.action_type == "submit_resolution":
        if outcome == "success":
            return "success"
        elif outcome == "partial":
            return "neutral"
        else:
            return "failure"

    # Remediation actions
    if outcome == "success":
        # All remediation steps that weren't quarantine-blocked → success
        return "success"
    elif outcome == "failure":
        # All remediation in a failed episode → failure
        return "failure"
    elif outcome == "partial":
        # Last remediation step → neutral (couldn't confirm if it was right or wrong)
        # Earlier steps → credit based on position heuristic
        if step_position >= total_steps - 2:
            return "neutral"
        else:
            return "failure"

    return "neutral"
