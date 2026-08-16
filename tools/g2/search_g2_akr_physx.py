#!/usr/bin/env python3
"""Run automatically ranked G2 AKR hypotheses until the physical budget ends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.physical_search import enumerate_physical_trials  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", type=Path, required=True)
    parser.add_argument("--g2-usd", type=Path, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--executor", type=Path, default=PROJECT_ROOT / "tools/g2/execute_g2_akr_physx.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-physical-attempts", type=int, default=12)
    parser.add_argument("--target-successes", type=int, default=3)
    parser.add_argument("--render-stride", type=int, default=8)
    parser.add_argument("--executor-argument", action="append", default=[])
    parser.add_argument("--wait-for-idle-seconds", type=float, default=0.0)
    parser.add_argument("--busy-poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--busy-pattern",
        default="execute_g2_handled_open_physx.py|schedule_g2_physx_isolated.py",
    )
    return parser.parse_args()


def _other_pipeline_busy(pattern: str) -> bool:
    check = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return check.returncode == 0


def _write_search_report(path: Path, search: dict[str, Any]) -> None:
    path.write_text(json.dumps(search, indent=2, sort_keys=True), encoding="utf-8")


def _wait_for_idle(pattern: str, maximum_wait_seconds: float, poll_seconds: float) -> bool:
    if not _other_pipeline_busy(pattern):
        return True
    if maximum_wait_seconds <= 0.0:
        return False
    deadline = time.monotonic() + maximum_wait_seconds
    while _other_pipeline_busy(pattern):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(poll_seconds, remaining))
    return True


def main() -> int:
    args = _parse_args()
    if args.maximum_physical_attempts <= 0 or args.target_successes <= 0:
        raise ValueError("physical attempt and success budgets must be positive")
    if args.wait_for_idle_seconds < 0.0 or args.busy_poll_seconds <= 0.0:
        raise ValueError("idle wait must be non-negative and busy poll period must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite search output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report = json.loads(args.plan_report.read_text(encoding="utf-8"))
    trials = enumerate_physical_trials(report, report_path=args.plan_report)
    search: dict[str, Any] = {
        "schema_version": 1,
        "mode": "automatic_learned_contact_akr_physx_search",
        "plan_report": str(args.plan_report.resolve()),
        "target_successes": args.target_successes,
        "maximum_physical_attempts": args.maximum_physical_attempts,
        "available_trial_count": len(trials),
        "dataset_ready": False,
        "dataset_ready_blockers": ["home_to_precontact_transit_pending", "release_and_retreat_pending"],
        "attempts": [],
    }
    search_path = args.output_dir / "search_report.json"
    _write_search_report(search_path, search)

    for search_index, trial in enumerate(trials[: args.maximum_physical_attempts]):
        search["status"] = "waiting_for_shared_isaac"
        _write_search_report(search_path, search)
        if not _wait_for_idle(args.busy_pattern, args.wait_for_idle_seconds, args.busy_poll_seconds):
            search["termination"] = "shared_isaac_busy_timeout"
            break
        search["status"] = "executing"
        trial_dir = args.output_dir / f"trial_{search_index:04d}_{trial.attempt_id}_path{trial.trajectory_index}"
        command = [
            str(args.isaac_python.resolve()),
            str(args.executor.resolve()),
            "--attempt-dir",
            trial.attempt_dir,
            "--g2-usd",
            str(args.g2_usd.resolve()),
            "--output-dir",
            str(trial_dir.resolve()),
            "--trajectory-index",
            str(trial.trajectory_index),
            "--render-stride",
            str(args.render_stride),
            *args.executor_argument,
        ]
        trial_dir.mkdir(parents=True)
        attempt_record: dict[str, Any] = {
            "search_index": search_index,
            "trial": trial.to_dict(),
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "result_path": str((trial_dir / "result.json").resolve()),
        }
        search["attempts"].append(attempt_record)
        _write_search_report(search_path, search)
        with (trial_dir / "launcher.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        result_path = trial_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "strict_physical_open_success": False,
            "failure_reason": "executor_result_missing",
        }
        attempt_record.update(
            status="completed",
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            executor_return_code=completed.returncode,
            strict_physical_open_success=bool(result.get("strict_physical_open_success")),
            maximum_progress_fraction=result.get("maximum_progress_fraction"),
            failure_reasons=result.get("failure_reasons", [result.get("failure_reason")]),
        )
        _write_search_report(search_path, search)
        success_count = sum(bool(item["strict_physical_open_success"]) for item in search["attempts"])
        if success_count >= args.target_successes:
            search["termination"] = "target_successes_reached"
            break
    else:
        search["termination"] = "physical_budget_or_trial_pool_exhausted"

    search["successful_attempts"] = sum(
        bool(item["strict_physical_open_success"]) for item in search["attempts"]
    )
    search["executed_attempts"] = len(search["attempts"])
    search["status"] = "finished"
    _write_search_report(search_path, search)
    print(search_path)
    return 0 if search["successful_attempts"] >= args.target_successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
