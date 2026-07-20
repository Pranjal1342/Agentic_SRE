"""
adversarial/runner.py — Adversarial stress-test suite runner.

Runs all 5 tests and prints the structured report specified in the brief.
Can be run with or without Postgres (Tests 1/3 use DB for lesson seeding and
causal edges; Tests 2/4/5 are DB-optional).

Usage:
  python -m adversarial.runner                  # run all 5 tests
  python -m adversarial.runner --tests 1 2 5    # run specific tests
  python -m adversarial.runner --n-runs 5       # Test 2 calibration runs per subtype
  python -m adversarial.runner --no-db          # skip DB-dependent features
  python -m adversarial.runner --output report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from adversarial.grader import GradeResult

log = structlog.get_logger(__name__)


async def run_all(
    test_ids: Optional[List[int]] = None,
    n_runs: int = 3,
    use_db: bool = True,
) -> List[GradeResult]:
    """Run the requested adversarial tests and return all GradeResults."""

    results: List[GradeResult] = []
    session_factory = None
    if use_db:
        try:
            from memory.db import get_session_factory
            session_factory = get_session_factory()
        except Exception as e:
            log.warning("runner.db_unavailable", error=str(e), note="Tests 1/3 will skip DB features")
            session_factory = None

    tests_to_run = test_ids or [1, 2, 3, 4, 5]

    for test_id in tests_to_run:
        print(f"\n{'-'*60}")
        print(f"Running Test {test_id}...")
        print(f"{'-'*60}")

        db_session = session_factory() if session_factory is not None else None
        try:
            if test_id == 1:
                from adversarial.test_1_distribution_shift import run_test1
                result = await run_test1(db_session=db_session)

            elif test_id == 2:
                from adversarial.test_2_calibration import run_test2
                result = await run_test2(n_runs_per_subtype=n_runs)

            elif test_id == 3:
                from adversarial.test_3_delayed_consequence import run_test3
                result = await run_test3(db_session=db_session)

            elif test_id == 4:
                from adversarial.test_4_value_conflict import run_test4
                result = await run_test4()

            elif test_id == 5:
                from adversarial.test_5_reward_hacking import run_test5
                result = await run_test5()

            elif test_id == 6:
                from adversarial.test_6_log_injection import run_test6_with_retry
                result = await run_test6_with_retry(db_session=db_session)

            elif test_id == 7:
                from adversarial.test_7_desc_injection import run_test7_with_retry
                result = await run_test7_with_retry(db_session=db_session)

            elif test_id == 8:
                from adversarial.test_8_quarantine_bypass import run_test8_with_retry
                result = await run_test8_with_retry(db_session=db_session)

            elif test_id == 12:
                from adversarial.test_12_resource_exhaustion import run_test12_with_retry
                result = await run_test12_with_retry(db_session=db_session)

            else:
                print(f"Unknown test ID: {test_id}")
                continue

            results.append(result)
            _print_result(result)

        except Exception as e:
            log.exception(f"runner.test_{test_id}_failed", error=str(e))
            print(f"  ERROR: Test {test_id} failed with exception: {e}")
        finally:
            if db_session is not None:
                try:
                    await db_session.close()
                except Exception:
                    pass

    return results


def _print_result(result: GradeResult) -> None:
    """Print a single test result in the report format specified by the brief."""
    VERDICT_COLORS = {
        "PASS": "\033[92m",          # green
        "FAIL": "\033[91m",          # red
        "WARN": "\033[93m",          # yellow
        "MEASUREMENT": "\033[94m",   # blue
        "WELL-CALIBRATED": "\033[92m",
        "MISCALIBRATED": "\033[91m",
        "MARGINAL": "\033[93m",
        "REWARD_FUNCTION_BUG": "\033[91m",
        "REWARD_FUNCTION_GAP": "\033[93m",
    }
    RESET = "\033[0m"
    color = VERDICT_COLORS.get(result.verdict, "")

    print(f"\n  Verdict    : {color}{result.verdict}{RESET}")
    print(f"  Score      : {result.score:.2f} / 1.00")
    print(f"  Classification: {result.classification}")
    print(f"\n  Observed   : {result.behavior_observed}")
    print(f"  Expected   : {result.expected_behavior}")

    if result.findings:
        print(f"\n  Findings:")
        for finding in result.findings:
            prefix = "  [WARN] " if ("BUG" in finding or "GAP" in finding or "FAIL" in finding) else "  * "
            print(f"{prefix}{finding}")


def _print_summary(results: List[GradeResult]) -> None:
    """Print the final summary table."""
    print(f"\n{'='*60}")
    print("ADVERSARIAL STRESS-TEST SUMMARY")
    print(f"{'='*60}")

    # Separate by classification per brief: agent_behavior vs architecture/reward bugs
    agent_results = [r for r in results if r.classification == "agent_behavior"]
    gap_results = [r for r in results if r.classification == "architecture_gap"]
    reward_results = [r for r in results if r.classification == "reward_function_bug"]

    if agent_results:
        print("\nAgent Behavior Tests:")
        for r in agent_results:
            icon = "[OK]" if r.verdict in ("PASS", "WELL-CALIBRATED") else ("[WARN]" if "MARGINAL" in r.verdict else "[FAIL]")
            print(f"  {icon}  {r.test_id:40s}  {r.verdict} ({r.score:.2f})")

    if gap_results:
        print("\nArchitecture Gap Reports (measurement only, not pass/fail):")
        for r in gap_results:
            print(f"  [DATA]  {r.test_id:40s}  {r.verdict} ({r.score:.2f})")

    if reward_results:
        print("\nReward Function Audit:")
        for r in reward_results:
            icon = "[OK]" if r.score == 1.0 else ("[WARN]" if r.score >= 0.5 else "[FAIL]")
            print(f"  {icon}  {r.test_id:40s}  {r.verdict} ({r.score:.2f})")

    print(f"\n{'-'*60}")
    print("IMPORTANT: Per the brief grading criteria --")
    print("  * AGENT BEHAVIOR: scored on calibration/escalation, NOT raw success.")
    print("  * ARCHITECTURE GAP: measurement only. 'FAIL' here = system limitation.")
    print("  * REWARD FUNCTION: bugs here -> fix R_t, NOT the agent prompt.")
    print(f"{'='*60}\n")


def _save_report(results: List[GradeResult], output_path: str) -> None:
    """Save results to JSON."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tests": len(results),
        "results": [
            {
                "test_id": r.test_id,
                "verdict": r.verdict,
                "score": r.score,
                "classification": r.classification,
                "behavior_observed": r.behavior_observed,
                "expected_behavior": r.expected_behavior,
                "findings": r.findings,
            }
            for r in results
        ],
    }
    path = Path(output_path)
    path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to: {path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial SRE Stress-Test Suite")
    parser.add_argument(
        "--tests", nargs="+", type=int, choices=[1, 2, 3, 4, 5],
        help="Which tests to run (default: all)",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Number of calibration runs per subtype for Test 2 (default: 3)",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="Skip DB-dependent features (Tests 1/3 lesson seeding and causal edges)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON report (default: adversarial_report.json)",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("ADVERSARIAL SRE STRESS-TEST SUITE")
    print("Grading: calibration + escalation, NOT raw success/failure")
    print(f"{'='*60}")

    results = await run_all(
        test_ids=args.tests,
        n_runs=args.n_runs,
        use_db=not args.no_db,
    )

    _print_summary(results)

    output_path = args.output or "adversarial_report.json"
    _save_report(results, output_path)


if __name__ == "__main__":
    asyncio.run(main())
