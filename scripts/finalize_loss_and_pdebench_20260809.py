#!/usr/bin/env python3
"""Monitor the active confirmatory queues and run analysis only after they finish."""
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
SYNTHETIC_ROOT = PROJECT_ROOT / "runs" / "loss_component_ablation_synthetic_20260809"
REAL_ROOT = PROJECT_ROOT / "runs" / "loss_component_ablation_real_20260809"
PDE_ROOT = PROJECT_ROOT / "runs" / "pdebench_burgers_latent_20260809"
ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "loss_component_ablation_analysis_20260809"
STATUS_PATH = PROJECT_ROOT / "runs" / "confirmatory_suite_status_20260809.json"
SESSIONS = (
    "lvs_loss_syn_20260809",
    "lvs_loss_real_20260809",
    "lvs_pdebench_20260809",
)
EXPECTED = {"synthetic": 90, "real": 90, "pde": 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=24.0)
    return parser.parse_args()


def _successful_results(root: Path) -> int:
    total = 0
    for path in root.glob("**/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total += payload.get("status") == "success"
    return total


def _session_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _write_status(state: str, counts: dict[str, int], sessions: dict[str, bool], **extra: object) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "counts": counts,
        "expected": EXPECTED,
        "sessions_alive": sessions,
        **extra,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    deadline = time.monotonic() + args.max_hours * 3600
    while True:
        counts = {
            "synthetic": _successful_results(SYNTHETIC_ROOT),
            "real": _successful_results(REAL_ROOT),
            "pde": _successful_results(PDE_ROOT),
        }
        sessions = {name: _session_alive(name) for name in SESSIONS}
        complete = all(counts[name] == EXPECTED[name] for name in EXPECTED)
        if complete and not any(sessions.values()):
            break
        if not any(sessions.values()) and not complete:
            _write_status("incomplete", counts, sessions)
            raise RuntimeError(f"Controllers stopped before completion: {counts}")
        if time.monotonic() >= deadline:
            _write_status("timeout", counts, sessions)
            raise TimeoutError(f"Confirmatory suite exceeded {args.max_hours} hours")
        _write_status("running", counts, sessions)
        time.sleep(args.poll_seconds)

    _write_status("analyzing", counts, sessions)
    _run([str(PYTHON), "scripts/run_iclr_latent_discovery.py", "summarize", "--output-root", str(SYNTHETIC_ROOT)])
    _run([str(PYTHON), "scripts/run_iclr_real_discovery.py", "summarize", "--output-root", str(REAL_ROOT)])
    _run([str(PYTHON), "scripts/run_pdebench_burgers_latent_study.py", "summarize", "--output-root", str(PDE_ROOT)])
    _run(
        [
            str(PYTHON),
            "scripts/analyze_loss_component_ablation.py",
            "--synthetic-root",
            str(SYNTHETIC_ROOT),
            "--real-root",
            str(REAL_ROOT),
            "--output-root",
            str(ANALYSIS_ROOT),
        ]
    )
    _run(
        [
            str(PYTHON),
            "scripts/analyze_pdebench_burgers_latent.py",
            "--output-root",
            str(PDE_ROOT),
        ]
    )
    _write_status(
        "completed",
        counts,
        sessions,
        outputs={
            "loss_report": str(ANALYSIS_ROOT / "LOSS_COMPONENT_ABLATION_REPORT.md"),
            "pde_report": str(PDE_ROOT / "analysis_report.md"),
        },
    )


if __name__ == "__main__":
    main()
