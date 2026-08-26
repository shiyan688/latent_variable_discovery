#!/usr/bin/env python3
"""Take over the bounded campaign without interrupting its active GPU jobs."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv-lvs-gpu" / "bin" / "python"
CAMPAIGN_ROOT = PROJECT_ROOT / "runs" / "extended_15h_campaign_20260809"
ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "extended_15h_analysis_20260810"
STATUS_PATH = CAMPAIGN_ROOT / "unlimited_takeover_status.json"
BOUNDED_SESSION = "lvs_extended_15h_20260809"


def _session_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", BOUNDED_SESSION],
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


def _campaign_snapshot() -> dict[str, object]:
    path = CAMPAIGN_ROOT / "campaign_status.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    while _session_alive():
        _write("waiting_for_bounded_controller", campaign=_campaign_snapshot())
        time.sleep(30)

    campaign_command = [
        str(PYTHON),
        "scripts/run_extended_15h_campaign_20260809.py",
        "--gpus",
        "2,3,4,5",
        "--run-until-complete",
        "--poll-seconds",
        "30",
        "--single-job-timeout-minutes",
        "90",
        "--output-root",
        "runs/extended_15h_campaign_20260809",
    ]
    _write("resuming_without_deadline", command=campaign_command, campaign=_campaign_snapshot())
    campaign = subprocess.run(campaign_command, cwd=PROJECT_ROOT, check=False)
    if campaign.returncode != 0:
        _write(
            "campaign_failed",
            returncode=campaign.returncode,
            campaign=_campaign_snapshot(),
        )
        raise SystemExit(campaign.returncode)

    analysis_command = [
        str(PYTHON),
        "scripts/analyze_extended_15h_campaign.py",
        "--campaign-root",
        str(CAMPAIGN_ROOT),
        "--output-root",
        str(ANALYSIS_ROOT),
    ]
    _write("analyzing_complete_campaign", command=analysis_command, campaign=_campaign_snapshot())
    analysis = subprocess.run(analysis_command, cwd=PROJECT_ROOT, check=False)
    if analysis.returncode != 0:
        _write(
            "analysis_failed",
            returncode=analysis.returncode,
            campaign=_campaign_snapshot(),
        )
        raise SystemExit(analysis.returncode)

    _write(
        "completed",
        campaign=_campaign_snapshot(),
        report=str(ANALYSIS_ROOT / "EXTENDED_15H_RESULTS.md"),
    )


if __name__ == "__main__":
    main()
