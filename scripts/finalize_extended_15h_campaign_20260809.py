#!/usr/bin/env python3
"""Wait for the 15-hour campaign and generate its consolidated report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
CAMPAIGN_ROOT = PROJECT_ROOT / "runs" / "extended_15h_campaign_20260809"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "extended_15h_analysis_20260810"
STATUS_PATH = CAMPAIGN_ROOT / "analysis_watcher_status.json"
SESSION = "lvs_extended_15h_20260809"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--max-hours",
        type=float,
        default=0.0,
        help="Watcher timeout in hours; 0 disables the timeout.",
    )
    return parser.parse_args()


def _session_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _write(state: str, **extra: object) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        **extra,
    }
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def main() -> None:
    args = parse_args()
    deadline = None if args.max_hours <= 0 else time.monotonic() + args.max_hours * 3600
    while _session_alive():
        if deadline is not None and time.monotonic() >= deadline:
            _write("timeout")
            raise TimeoutError(
                f"Extended campaign watcher exceeded its {args.max_hours:g}-hour limit"
            )
        campaign_status = CAMPAIGN_ROOT / "campaign_status.json"
        snapshot = json.loads(campaign_status.read_text()) if campaign_status.exists() else {}
        _write("waiting", campaign=snapshot)
        time.sleep(args.poll_seconds)

    campaign_status = json.loads((CAMPAIGN_ROOT / "campaign_status.json").read_text())
    if campaign_status.get("state") not in {
        "completed_budget",
        "completed_all",
        "completed_with_failures",
    }:
        _write("campaign_failed_or_incomplete", campaign=campaign_status)
        raise RuntimeError(f"Campaign ended in state {campaign_status.get('state')}")
    _write("analyzing", campaign=campaign_status)
    result = subprocess.run(
        [
            str(PYTHON),
            "scripts/analyze_extended_15h_campaign.py",
            "--campaign-root",
            str(CAMPAIGN_ROOT),
            "--output-root",
            str(OUTPUT_ROOT),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        _write(
            "analysis_failed",
            campaign=campaign_status,
            returncode=result.returncode,
            stderr_tail=result.stderr[-4000:],
        )
        raise RuntimeError("Extended analysis failed; see analysis_watcher_status.json")
    _write(
        "completed",
        campaign=campaign_status,
        report=str(OUTPUT_ROOT / "EXTENDED_15H_RESULTS.md"),
        analyzer_stdout=result.stdout.strip(),
    )


if __name__ == "__main__":
    main()
