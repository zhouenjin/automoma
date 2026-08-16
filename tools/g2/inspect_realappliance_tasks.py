#!/usr/bin/env python3
"""Extract normalized task articulations from RealAppliance assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from automoma.integrations.realappliance.usd_task import extract_usd_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True, help="RealAppliance model directory")
    parser.add_argument("--asset-id", action="append", required=True, help="Asset ID; repeat for multiple assets")
    parser.add_argument("--task", default="open")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {"task": args.task, "assets": {}, "failures": {}}
    for asset_id in args.asset_id:
        asset_dir = args.asset_root / asset_id
        try:
            extracted = extract_usd_tasks(asset_dir / "Aligned.usd", asset_dir / "gt_part.json", args.task)
            report["assets"][asset_id] = [task.to_dict() for task in extracted]
        except Exception as error:  # noqa: B902 - the report must retain per-asset extraction failures
            report["failures"][asset_id] = {"type": type(error).__name__, "message": str(error)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(args.output)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
