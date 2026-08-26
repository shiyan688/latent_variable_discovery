#!/usr/bin/env python3
"""Persistent four-GPU overnight continuation and sensitivity experiment queue."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
STATE_ROOT = PROJECT_ROOT / "runs" / "iclr_overnight_20260808"
SYNTHETIC_ROOT = PROJECT_ROOT / "runs" / "iclr_synthetic_continuity_main_20260808"
REAL_MAIN_ROOT = PROJECT_ROOT / "runs" / "iclr_real_broad_pilot_20260808"
QDIM_ROOT = PROJECT_ROOT / "runs" / "iclr_real_qdim_sensitivity_20260808"
SUPPORT01_ROOT = PROJECT_ROOT / "runs" / "iclr_real_support01_sensitivity_20260808"
SUPPORT05_ROOT = PROJECT_ROOT / "runs" / "iclr_real_support05_sensitivity_20260808"
MAIN_ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "iclr_discovery_analysis_20260808"
SENSITIVITY_ANALYSIS_ROOT = PROJECT_ROOT / "runs" / "iclr_real_sensitivity_analysis_20260808"
PREPARED = [
    "data/application_reviewer_clean/prepared_datasets.json",
    "data/real_datasets2/prepared/prepared_datasets.json",
]
LATENT_METHODS = (
    "joint_mse,alternating_mse,joint_fixed,alternating_fixed,"
    "joint_dynamic,alternating_dynamic"
)
ALL_REAL_METHODS = f"{LATENT_METHODS},no_q_mlp,random_forest,support_knn"


def _write_state(phase: str, status: str, **extra: object) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = STATE_ROOT / "status.json"
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "status": status,
        **extra,
    }
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(state_path)


def _prepared_args() -> list[str]:
    output: list[str] = []
    for path in PREPARED:
        output.extend(["--prepared-summary", path])
    return output


def _real_command(
    *, output_root: Path, seeds: str, q_dims: str, gpus: str, support_ratio: float, methods: str
) -> list[str]:
    return [
        str(PYTHON), "scripts/run_iclr_real_discovery.py", "launch",
        *_prepared_args(),
        "--methods", methods,
        "--seeds", seeds,
        "--q-dims", q_dims,
        "--gpus", gpus,
        "--output-root", str(output_root),
        "--epochs", "200",
        "--cal-steps", "200",
        "--support-ratio", str(support_ratio),
        "--batch-size", "256",
        "--hidden-sizes", "256,128",
        "--max-train-per-label", "256",
        "--max-test-per-label", "256",
        "--subsample-seed", "20260808",
        "--resume",
        "--save-artifacts",
    ]


def _run_parallel(phase: str, jobs: Sequence[tuple[str, list[str]]]) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, subprocess.Popen[bytes], object]] = []
    for name, command in jobs:
        handle = (STATE_ROOT / f"{name}.log").open("ab")
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT
        )
        processes.append((name, process, handle))
    _write_state(
        phase,
        "running",
        processes={name: process.pid for name, process, _ in processes},
    )
    return_codes: dict[str, int] = {}
    while len(return_codes) < len(processes):
        for name, process, _ in processes:
            if name in return_codes:
                continue
            return_code = process.poll()
            if return_code is not None:
                return_codes[name] = return_code
        _write_state(phase, "running", return_codes=return_codes)
        time.sleep(20)
    for _, _, handle in processes:
        handle.close()
    if any(return_codes.values()):
        _write_state(phase, "failed", return_codes=return_codes)
        raise SystemExit(1)
    _write_state(phase, "completed", return_codes=return_codes)


def _run_checked(phase: str, command: list[str]) -> None:
    _write_state(phase, "running")
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        _write_state(phase, "failed", returncode=completed.returncode)
        raise SystemExit(completed.returncode)
    _write_state(phase, "completed", returncode=0)


def main() -> None:
    synthetic_command = [
        str(PYTHON), "scripts/run_iclr_latent_discovery.py", "launch",
        "--expression-ids", "3,15,21,36,41,48",
        "--methods", (
            "joint_mse,alternating_mse,joint_fixed,alternating_fixed,"
            "joint_dynamic,alternating_dynamic,no_q_mlp,random_forest,support_knn,oracle_q_mlp"
        ),
        "--seeds", "0,1,2,3,4",
        "--data-seed", "20260808",
        "--gpus", "4,5",
        "--output-root", str(SYNTHETIC_ROOT),
        "--epochs", "300",
        "--cal-steps", "300",
        "--train-labels", "32",
        "--validation-labels", "16",
        "--test-labels", "32",
        "--samples-per-label", "60",
        "--support-ratio", "0.3",
        "--batch-size", "256",
        "--resume",
        "--save-artifacts",
    ]
    real_main_command = _real_command(
        output_root=REAL_MAIN_ROOT,
        seeds="0,1,2,3,4",
        q_dims="2",
        gpus="6,7",
        support_ratio=0.3,
        methods=ALL_REAL_METHODS,
    )
    _run_parallel(
        "main_five_seed_matrix",
        [("synthetic_main", synthetic_command), ("real_main", real_main_command)],
    )
    _run_checked(
        "main_analysis",
        [
            str(PYTHON), "scripts/analyze_iclr_discovery_runs.py",
            "--synthetic-root", str(SYNTHETIC_ROOT),
            "--real-root", str(REAL_MAIN_ROOT),
            "--output-dir", str(MAIN_ANALYSIS_ROOT),
        ],
    )
    _run_parallel(
        "q_dimension_sensitivity",
        [
            (
                "real_qdim",
                _real_command(
                    output_root=QDIM_ROOT,
                    seeds="0,1,2",
                    q_dims="1,4",
                    gpus="4,5,6,7",
                    support_ratio=0.3,
                    methods=LATENT_METHODS,
                ),
            )
        ],
    )
    _run_parallel(
        "support_ratio_sensitivity",
        [
            (
                "real_support01",
                _real_command(
                    output_root=SUPPORT01_ROOT,
                    seeds="0,1,2",
                    q_dims="2",
                    gpus="4,5",
                    support_ratio=0.1,
                    methods=LATENT_METHODS,
                ),
            ),
            (
                "real_support05",
                _real_command(
                    output_root=SUPPORT05_ROOT,
                    seeds="0,1,2",
                    q_dims="2",
                    gpus="6,7",
                    support_ratio=0.5,
                    methods=LATENT_METHODS,
                ),
            ),
        ],
    )
    _run_checked(
        "sensitivity_analysis",
        [
            str(PYTHON), "scripts/analyze_iclr_sensitivity_runs.py",
            "--main-root", str(REAL_MAIN_ROOT),
            "--qdim-root", str(QDIM_ROOT),
            "--support-root", str(SUPPORT01_ROOT),
            "--support-root", str(SUPPORT05_ROOT),
            "--output-dir", str(SENSITIVITY_ANALYSIS_ROOT),
        ],
    )
    _write_state("all", "completed")


if __name__ == "__main__":
    main()
