#!/usr/bin/env python3
"""Run the frozen 1,000-epoch matched-update real/PDE campaign safely."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
RUNS = PROJECT_ROOT / "runs"
DEFAULT_ROOT = RUNS / "matched_1000ep_real_pde_20260817"
REAL_ROOT = DEFAULT_ROOT / "real"
PDE_ROOT = DEFAULT_ROOT / "pdebench"
PDE_SOURCE_ROOT = RUNS / "pdebench_burgers_latent_20260809"
GPU_MEMORY_THRESHOLD_MIB = 128

REAL_DATASETS = (
    (
        "nasa_battery_capacity",
        PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json",
    ),
    (
        "starry_te_seebeck",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
    ),
    (
        "starry_te_electrical_conductivity",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
    ),
    (
        "starry_te_thermal_conductivity",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
    ),
)
REAL_METHODS = (
    "joint_mse_step1",
    "joint_continuity_step1",
    "no_q_mlp",
    "support_knn",
    "random_forest",
)


@dataclass(frozen=True)
class Task:
    task_id: str
    phase: int
    family: str
    output_root: str
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--single-job-timeout-minutes", type=float, default=240.0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _task_id(family: str, values: tuple[Any, ...]) -> str:
    raw = json.dumps([family, *values], sort_keys=True, separators=(",", ":")).encode()
    return f"{family}_{hashlib.sha256(raw).hexdigest()[:14]}"


def _real_task(
    *, phase: int, dataset: str, summary: Path, method: str, seed: int, root: Path
) -> Task:
    command = (
        str(PYTHON),
        "scripts/run_iclr_real_discovery.py",
        "run-job",
        "--prepared-summary",
        str(summary),
        "--dataset",
        dataset,
        "--method",
        method,
        "--seed",
        str(seed),
        "--q-dim",
        "8",
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--epochs",
        "1000",
        "--cal-steps",
        "200",
        "--cal-init-mode",
        "prior_random",
        "--cal-num-starts",
        "4",
        "--cal-selection-ratio",
        "0.25",
        "--cal-selection-min-rows",
        "24",
        "--cal-refine-steps",
        "50",
        "--cal-refine-only-after-selection",
        "--support-ratio",
        "0.3",
        "--batch-size",
        "256",
        "--hidden-sizes",
        "256,128",
        "--max-train-per-label",
        "256",
        "--max-test-per-label",
        "256",
        "--subsample-seed",
        "20260808",
        "--resume",
        "--save-artifacts",
    )
    values = (dataset, method, seed, 8, 0.3, 1000)
    return Task(_task_id("real", values), phase, "real", str(root), command)


def _pde_task(
    *, phase: int, q_dim: int, method: str, seed: int, root: Path
) -> Task:
    command = (
        str(PYTHON),
        "scripts/run_pdebench_burgers_latent_study.py",
        "run-job",
        "--q-dim",
        str(q_dim),
        "--method",
        method,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--epochs",
        "1000",
        "--support-ratio",
        "0.3",
        "--batch-size",
        "256",
        "--resume",
        "--save-artifacts",
    )
    values = (q_dim, method, seed, 0.3, 1000)
    return Task(_task_id("pde", values), phase, "pde", str(root), command)


def build_tasks(seeds: list[int], *, real_root: Path, pde_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for seed in seeds:
        phase = 1 if seed < 3 else 2
        for dataset, summary in REAL_DATASETS:
            for method in REAL_METHODS:
                tasks.append(
                    _real_task(
                        phase=phase,
                        dataset=dataset,
                        summary=summary,
                        method=method,
                        seed=seed,
                        root=real_root,
                    )
                )
        for method in ("joint_mse_step1", "joint_continuity_step1"):
            tasks.append(
                _pde_task(
                    phase=phase,
                    q_dim=16,
                    method=method,
                    seed=seed,
                    root=pde_root,
                )
            )
        tasks.append(
            _pde_task(
                phase=phase,
                q_dim=8,
                method="joint_mse_step1",
                seed=seed,
                root=pde_root,
            )
        )
    return sorted(tasks, key=lambda task: task.phase)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _gpu_memory() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, used = (value.strip() for value in line.split(",", maxsplit=1))
        memory[index] = int(used)
    return memory


def _finished_ids(path: Path) -> set[str]:
    finished: set[str] = set()
    if not path.exists():
        return finished
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("returncode") == 0:
            finished.add(str(row["task_id"]))
    return finished


def _stage_pde(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pdebench_burgers_subset.npz", "subset_manifest.json"):
        source = PDE_SOURCE_ROOT / name
        destination = root / name
        if not destination.exists():
            shutil.copy2(source, destination)


def _write_manifest(
    *, tasks: list[Task], root: Path, gpus: list[str], seeds: list[int], args: argparse.Namespace
) -> None:
    family_counts: dict[str, int] = {}
    for task in tasks:
        family_counts[task.family] = family_counts.get(task.family, 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_plan": str(PROJECT_ROOT / "MATCHED_1000EP_REAL_PDE_PLAN_20260817.md"),
        "planned_tasks": len(tasks),
        "tasks_by_family": family_counts,
        "gpus": gpus,
        "seeds": seeds,
        "epochs": 1000,
        "gpu_memory_available_threshold_mib": GPU_MEMORY_THRESHOLD_MIB,
        "poll_seconds": args.poll_seconds,
        "single_job_timeout_minutes": args.single_job_timeout_minutes,
        "run_until_complete": True,
    }
    _write_json_atomic(root / "campaign_manifest.json", manifest)
    with (root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def _finalize_outputs(root: Path, real_root: Path, pde_root: Path) -> list[dict[str, Any]]:
    commands = (
        (str(PYTHON), "scripts/run_iclr_real_discovery.py", "summarize", "--output-root", str(real_root)),
        (str(PYTHON), "scripts/run_pdebench_burgers_latent_study.py", "summarize", "--output-root", str(pde_root)),
    )
    rows: list[dict[str, Any]] = []
    for command in commands:
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        rows.append(
            {
                "command": list(command),
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        )
    _write_json_atomic(root / "finalize_status.json", {"commands": rows})
    return rows


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    output_root = args.output_root.resolve()
    real_root = output_root / "real"
    pde_root = output_root / "pdebench"
    tasks = build_tasks(seeds, real_root=real_root, pde_root=pde_root)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "planned": len(tasks),
                    "real": sum(task.family == "real" for task in tasks),
                    "pde": sum(task.family == "pde" for task in tasks),
                },
                indent=2,
            )
        )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    _stage_pde(pde_root)
    _write_manifest(tasks=tasks, root=output_root, gpus=gpus, seeds=seeds, args=args)
    status_path = output_root / "task_status.jsonl"
    completed = _finished_ids(status_path)
    pending = [task for task in tasks if task.task_id not in completed]
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]] = {}
    started_at = datetime.now(timezone.utc)
    failures = 0
    gpu_error: str | None = None

    while pending or running:
        now = time.monotonic()
        try:
            memory = _gpu_memory()
            gpu_error = None
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            memory = {}
            gpu_error = repr(error)

        if gpu_error is None:
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                if memory.get(gpu, 10**9) >= GPU_MEMORY_THRESHOLD_MIB:
                    continue
                task = pending.pop(0)
                log_path = output_root / "logs" / f"{task.task_id}.log"
                handle = log_path.open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment.update(
                    {
                        "OMP_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4",
                        "OPENBLAS_NUM_THREADS": "4",
                    }
                )
                process = subprocess.Popen(
                    list(task.command),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                running[gpu] = (process, task, handle, now)

        finished_gpus: list[str] = []
        for gpu, (process, task, handle, task_started) in running.items():
            returncode = process.poll()
            elapsed = now - task_started
            timed_out = elapsed > args.single_job_timeout_minutes * 60
            if returncode is None and timed_out:
                process.terminate()
                try:
                    returncode = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
            if returncode is None:
                continue
            handle.close()
            row = {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "task_id": task.task_id,
                "phase": task.phase,
                "family": task.family,
                "gpu": gpu,
                "elapsed_seconds": elapsed,
                "returncode": returncode,
                "timed_out": timed_out,
                "output_root": task.output_root,
            }
            with status_path.open("a", encoding="utf-8") as status:
                status.write(json.dumps(row) + "\n")
            if returncode == 0:
                completed.add(task.task_id)
            else:
                failures += 1
            finished_gpus.append(gpu)
        for gpu in finished_gpus:
            del running[gpu]

        _write_json_atomic(
            output_root / "campaign_status.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": "waiting_for_gpu_query" if gpu_error else "running",
                "started_at": started_at.isoformat(),
                "planned": len(tasks),
                "completed": len(completed),
                "failed": failures,
                "pending": len(pending),
                "running": {
                    gpu: {
                        "task_id": task.task_id,
                        "family": task.family,
                        "elapsed_seconds": now - task_started,
                    }
                    for gpu, (_, task, _, task_started) in running.items()
                },
                "gpu_memory_mib": memory,
                "gpu_query_error": gpu_error,
            },
        )
        time.sleep(args.poll_seconds)

    finalize = _finalize_outputs(output_root, real_root, pde_root)
    if status_path.exists():
        rows = [json.loads(line) for line in status_path.read_text().splitlines() if line.strip()]
        fields = sorted({key for row in rows for key in row})
        with (output_root / "task_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    _write_json_atomic(
        output_root / "campaign_status.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "completed_all" if len(completed) == len(tasks) and not failures else "completed_with_failures",
            "started_at": started_at.isoformat(),
            "planned": len(tasks),
            "completed": len(completed),
            "failed": failures,
            "pending": 0,
            "running": {},
            "finalize_returncodes": [row["returncode"] for row in finalize],
        },
    )


if __name__ == "__main__":
    main()
