#!/usr/bin/env python3
"""Monitor two launchers and run aggregate analysis after both reach terminal state."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--synthetic-expected", type=int, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--real-expected", type=int, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    return parser.parse_args()


def _read_status(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _count_successful_results(root: Path) -> int:
    count = 0
    for result_path in root.glob("**/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "success":
            count += 1
    return count


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    state_path = args.analysis_output / "suite_status.json"
    while True:
        synthetic_status = _read_status(args.synthetic_root / "launcher_status.jsonl")
        real_status = _read_status(args.real_root / "launcher_status.jsonl")
        failures = [
            {"domain": domain, **row}
            for domain, rows in (("synthetic", synthetic_status), ("real", real_status))
            for row in rows
            if row.get("returncode") != 0
        ]
        synthetic_completed = _count_successful_results(args.synthetic_root)
        real_completed = _count_successful_results(args.real_root)
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "synthetic_completed": synthetic_completed,
            "synthetic_expected": args.synthetic_expected,
            "real_completed": real_completed,
            "real_expected": args.real_expected,
            "failures": failures,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_state(state_path, state)
        terminal = (
            synthetic_completed >= args.synthetic_expected
            and real_completed >= args.real_expected
        )
        if terminal:
            command = [
                sys.executable,
                str(Path(__file__).resolve().parent / "analyze_iclr_discovery_runs.py"),
                "--synthetic-root", str(args.synthetic_root),
                "--real-root", str(args.real_root),
                "--output-dir", str(args.analysis_output),
            ]
            completed = subprocess.run(command, check=False)
            state.update(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "completed" if completed.returncode == 0 else "analysis_failed",
                    "analysis_returncode": completed.returncode,
                }
            )
            _write_state(state_path, state)
            raise SystemExit(completed.returncode)
        if time.monotonic() - started >= args.timeout_hours * 3600:
            state.update(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "status": "monitor_timeout",
                }
            )
            _write_state(state_path, state)
            raise SystemExit(2)
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
