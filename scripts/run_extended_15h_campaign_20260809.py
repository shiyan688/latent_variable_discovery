#!/usr/bin/env python3
"""Deadline-aware four-GPU extension campaign for loss and PDEBench studies."""
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
RUNS = PROJECT_ROOT / "runs"
CAMPAIGN_ROOT = RUNS / "extended_15h_campaign_20260809"
PDE_SOURCE_ROOT = RUNS / "pdebench_burgers_latent_20260809"

CORE_METHODS = (
    "joint_mse",
    "joint_lb_mse",
    "joint_hsic",
    "joint_continuity",
    "joint_q_l2",
    "joint_calprior",
    "joint_hsic_cont",
    "joint_all_mse",
    "joint_fixed",
    "joint_dynamic",
)
ROBUST_METHODS = (
    "joint_mse",
    "joint_hsic",
    "joint_continuity",
    "joint_calprior",
    "joint_fixed",
    "joint_dynamic",
)
BASELINES = ("no_q_mlp", "random_forest", "support_knn")
DOSE_METHODS = (
    "joint_hsic_w005",
    "joint_hsic_w01",
    "joint_hsic_w02",
    "joint_hsic_w10",
    "joint_hsic_w20",
    "joint_cont_w005",
    "joint_cont_w01",
    "joint_cont_w02",
    "joint_cont_w10",
    "joint_cont_w20",
    "joint_ql2_w0001",
    "joint_ql2_w0003",
    "joint_ql2_w003",
    "joint_ql2_w01",
    "joint_calprior_w001",
    "joint_calprior_w003",
    "joint_calprior_w03",
    "joint_calprior_w10",
    "joint_orth_pearson",
    "joint_orth_nhsic",
    "joint_orth_dcor",
    "joint_orth_propensity",
    "joint_orth_adversarial",
)
EXPRESSIONS = (3, 41, 48)
REAL_DATASETS = (
    (
        "starry_te_seebeck",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
    ),
    (
        "nasa_cmapss_fd001_sensor_response",
        PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json",
    ),
    (
        "nasa_battery_capacity",
        PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json",
    ),
)
GPU_PREREQUISITES = {
    "2": "lvs_loss_syn_20260809",
    "3": "lvs_loss_real_20260809",
    "4": "lvs_pdebench_20260809",
    "5": "lvs_pdebench_20260809",
}


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
    parser.add_argument("--wall-hours", type=float, default=15.0)
    parser.add_argument(
        "--run-until-complete",
        action="store_true",
        help="Disable the campaign dispatch deadline and exhaust the frozen task reservoir.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--single-job-timeout-minutes", type=float, default=90.0)
    parser.add_argument("--output-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _task_id(family: str, values: tuple[Any, ...]) -> str:
    raw = json.dumps([family, *values], separators=(",", ":"), default=str).encode()
    return f"{family}_{hashlib.sha256(raw).hexdigest()[:14]}"


def _calibration_args() -> tuple[str, ...]:
    return (
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
    )


def _synthetic_task(
    *, phase: int, family: str, root: Path, expression: int, method: str, seed: int
) -> Task:
    command = (
        str(PYTHON),
        "scripts/run_iclr_latent_discovery.py",
        "run-job",
        "--expression-id",
        str(expression),
        "--method",
        method,
        "--seed",
        str(seed),
        "--data-seed",
        "20260809",
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--epochs",
        "200",
        *_calibration_args(),
        "--train-labels",
        "24",
        "--validation-labels",
        "8",
        "--test-labels",
        "8",
        "--samples-per-label",
        "60",
        "--support-ratio",
        "0.3",
        "--batch-size",
        "256",
        "--resume",
        "--save-artifacts",
    )
    values = (expression, method, seed)
    return Task(_task_id(family, values), phase, family, str(root), command)


def _real_task(
    *,
    phase: int,
    family: str,
    root: Path,
    dataset: str,
    summary: Path,
    method: str,
    seed: int,
    q_dim: int,
    support_ratio: float,
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
        str(q_dim),
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--epochs",
        "200",
        *_calibration_args(),
        "--support-ratio",
        str(support_ratio),
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
    values = (dataset, method, seed, q_dim, support_ratio)
    return Task(_task_id(family, values), phase, family, str(root), command)


def _ratio_name(value: float) -> str:
    return f"r{value:.1f}".replace(".", "p")


def _pde_root(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for name in ("pdebench_burgers_subset.npz", "subset_manifest.json"):
        destination = base / name
        if not destination.exists():
            shutil.copy2(PDE_SOURCE_ROOT / name, destination)
    return base


def _pde_task(
    *,
    phase: int,
    family: str,
    root: Path,
    q_dim: int,
    method: str,
    seed: int,
    support_ratio: float,
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
        "300",
        "--support-ratio",
        str(support_ratio),
        "--batch-size",
        "256",
        "--resume",
        "--save-artifacts",
    )
    values = (q_dim, method, seed, support_ratio)
    return Task(_task_id(family, values), phase, family, str(root), command)


def build_tasks(*, stage_pde_data: bool) -> list[Task]:
    tasks: list[Task] = []
    seed_syn_root = RUNS / "extended_loss_seed_synthetic_20260809"
    seed_real_root = RUNS / "extended_loss_seed_real_20260809"
    qdim_root = RUNS / "extended_loss_qdim_real_20260809"
    dose_syn_root = RUNS / "extended_loss_dose_synthetic_20260809"
    dose_real_root = RUNS / "extended_loss_dose_real_20260809"

    # Phase 1: three extra confirmatory seeds for every component.
    for seed in (3, 4, 5):
        for expression in EXPRESSIONS:
            for method in CORE_METHODS:
                tasks.append(
                    _synthetic_task(
                        phase=1,
                        family="seed_extension_synthetic_core",
                        root=seed_syn_root,
                        expression=expression,
                        method=method,
                        seed=seed,
                    )
                )
        for dataset, summary in REAL_DATASETS:
            for method in CORE_METHODS:
                tasks.append(
                    _real_task(
                        phase=1,
                        family="seed_extension_real_core",
                        root=seed_real_root,
                        dataset=dataset,
                        summary=summary,
                        method=method,
                        seed=seed,
                        q_dim=4,
                        support_ratio=0.3,
                    )
                )

    # Phase 2: PDEBench support sparsity, including all same-support baselines at q=8 joint.
    for ratio in (0.1, 0.2, 0.5, 0.7):
        root = RUNS / "extended_pdebench_support_20260809" / _ratio_name(ratio)
        if stage_pde_data:
            _pde_root(root)
        for seed in (0, 1, 2):
            for q_dim in (4, 8, 16):
                for method in ("joint_mse", "alternating_mse"):
                    tasks.append(
                        _pde_task(
                            phase=2,
                            family=f"pde_support_{_ratio_name(ratio)}",
                            root=root,
                            q_dim=q_dim,
                            method=method,
                            seed=seed,
                            support_ratio=ratio,
                        )
                    )

    # Phase 3: latent dimension screen on all component variants.
    for seed in (0, 1, 2):
        for dataset, summary in REAL_DATASETS:
            for q_dim in (1, 2, 8):
                for method in CORE_METHODS:
                    tasks.append(
                        _real_task(
                            phase=3,
                            family="real_qdim_core",
                            root=qdim_root,
                            dataset=dataset,
                            summary=summary,
                            method=method,
                            seed=seed,
                            q_dim=q_dim,
                            support_ratio=0.3,
                        )
                    )

    # Phase 4: dose response and alternative dependence penalties.
    for seed in (0, 1, 2):
        for expression in EXPRESSIONS:
            for method in DOSE_METHODS:
                tasks.append(
                    _synthetic_task(
                        phase=4,
                        family="loss_dose_synthetic",
                        root=dose_syn_root,
                        expression=expression,
                        method=method,
                        seed=seed,
                    )
                )
        for dataset, summary in REAL_DATASETS:
            for method in DOSE_METHODS:
                tasks.append(
                    _real_task(
                        phase=4,
                        family="loss_dose_real",
                        root=dose_real_root,
                        dataset=dataset,
                        summary=summary,
                        method=method,
                        seed=seed,
                        q_dim=4,
                        support_ratio=0.3,
                    )
                )

    # Phase 5: support-ratio robustness with latent methods and fair baselines.
    for ratio in (0.1, 0.2, 0.5, 0.7):
        root = RUNS / "extended_loss_support_real_20260809" / _ratio_name(ratio)
        for seed in (0, 1, 2):
            for dataset, summary in REAL_DATASETS:
                for method in (*ROBUST_METHODS, *BASELINES):
                    tasks.append(
                        _real_task(
                            phase=5,
                            family=f"real_support_{_ratio_name(ratio)}",
                            root=root,
                            dataset=dataset,
                            summary=summary,
                            method=method,
                            seed=seed,
                            q_dim=4,
                            support_ratio=ratio,
                        )
                    )

    # Phase 6: deepen every key estimate if the time budget remains.
    for seed in (6, 7, 8, 9):
        for expression in EXPRESSIONS:
            for method in CORE_METHODS:
                tasks.append(
                    _synthetic_task(
                        phase=6,
                        family="seed_extension_synthetic_deep",
                        root=seed_syn_root,
                        expression=expression,
                        method=method,
                        seed=seed,
                    )
                )
        for dataset, summary in REAL_DATASETS:
            for method in CORE_METHODS:
                tasks.append(
                    _real_task(
                        phase=6,
                        family="seed_extension_real_deep",
                        root=seed_real_root,
                        dataset=dataset,
                        summary=summary,
                        method=method,
                        seed=seed,
                        q_dim=4,
                        support_ratio=0.3,
                    )
                )
    for seed in (3, 4, 5, 6, 7, 8, 9):
        for dataset, summary in REAL_DATASETS:
            for q_dim in (1, 2, 8):
                for method in CORE_METHODS:
                    tasks.append(
                        _real_task(
                            phase=6,
                            family="real_qdim_deep",
                            root=qdim_root,
                            dataset=dataset,
                            summary=summary,
                            method=method,
                            seed=seed,
                            q_dim=q_dim,
                            support_ratio=0.3,
                        )
                    )
    for seed in (3, 4, 5):
        for expression in EXPRESSIONS:
            for method in DOSE_METHODS:
                tasks.append(
                    _synthetic_task(
                        phase=6,
                        family="loss_dose_synthetic_deep",
                        root=dose_syn_root,
                        expression=expression,
                        method=method,
                        seed=seed,
                    )
                )
        for dataset, summary in REAL_DATASETS:
            for method in DOSE_METHODS:
                tasks.append(
                    _real_task(
                        phase=6,
                        family="loss_dose_real_deep",
                        root=dose_real_root,
                        dataset=dataset,
                        summary=summary,
                        method=method,
                        seed=seed,
                        q_dim=4,
                        support_ratio=0.3,
                    )
                )
        for ratio in (0.1, 0.2, 0.5, 0.7):
            root = RUNS / "extended_loss_support_real_20260809" / _ratio_name(ratio)
            for dataset, summary in REAL_DATASETS:
                for method in (*ROBUST_METHODS, *BASELINES):
                    tasks.append(
                        _real_task(
                            phase=6,
                            family=f"real_support_{_ratio_name(ratio)}_deep",
                            root=root,
                            dataset=dataset,
                            summary=summary,
                            method=method,
                            seed=seed,
                            q_dim=4,
                            support_ratio=ratio,
                        )
                    )
    pde_seed_root = RUNS / "extended_pdebench_seeds_20260809"
    if stage_pde_data:
        _pde_root(pde_seed_root)
    for seed in (3, 4, 5, 6, 7, 8, 9):
        for q_dim in (4, 8, 16):
            for method in ("joint_mse", "alternating_mse"):
                tasks.append(
                    _pde_task(
                        phase=6,
                        family="pde_seed_extension",
                        root=pde_seed_root,
                        q_dim=q_dim,
                        method=method,
                        seed=seed,
                        support_ratio=0.3,
                    )
                )
    return tasks


def _tmux_session_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


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
    output: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, used = (value.strip() for value in line.split(",", maxsplit=1))
        output[index] = int(used)
    return output


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _read_finished_ids(path: Path) -> set[str]:
    finished: set[str] = set()
    if not path.exists():
        return finished
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("returncode") == 0:
            finished.add(row["task_id"])
    return finished


def _write_plan(tasks: list[Task], output_root: Path, args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    for task in tasks:
        counts[task.family] = counts.get(task.family, 0) + 1
        phase_counts[str(task.phase)] = phase_counts.get(str(task.phase), 0) + 1
    manifest_path = output_root / "campaign_manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_manifest = {}
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at": created_at,
        "initial_created_at": previous_manifest.get(
            "initial_created_at", previous_manifest.get("created_at", created_at)
        ),
        "wall_hours": None if args.run_until_complete else args.wall_hours,
        "run_until_complete": args.run_until_complete,
        "gpus": [value for value in args.gpus.split(",") if value],
        "dispatch_policy": (
            "exhaust the frozen task reservoir; no campaign wall-clock deadline"
            if args.run_until_complete
            else "stop new dispatch at deadline; let active jobs finish atomically"
        ),
        "monitoring": {
            "poll_seconds": args.poll_seconds,
            "single_job_timeout_minutes": args.single_job_timeout_minutes,
            "gpu_memory_available_threshold_mib": 128,
        },
        "scientific_questions": [
            "Are component-loss effects stable across seeds?",
            "Does latent dimension explain real-data calibration failures?",
            "How do sparse support ratios alter prediction and latent geometry?",
            "Are regularization effects monotone in weight?",
            "Which dependence penalty offers the best prediction-continuity trade-off?",
            "Do PDEBench conclusions persist across latent dimension, support sparsity, and seeds?",
        ],
        "planned_tasks": len(tasks),
        "tasks_by_phase": phase_counts,
        "tasks_by_family": counts,
    }
    _write_json_atomic(manifest_path, manifest)
    with (output_root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def _summarize_completed(tasks: list[Task], output_root: Path) -> None:
    status_path = output_root / "task_status.jsonl"
    rows: list[dict[str, Any]] = []
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_root / "task_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    roots = sorted({task.output_root for task in tasks if task.task_id in {row.get("task_id") for row in rows if row.get("returncode") == 0}})
    summaries = []
    for raw_root in roots:
        root = Path(raw_root)
        result_count = sum(1 for _ in root.glob("**/result.json"))
        summaries.append({"output_root": raw_root, "result_files": result_count})
    _write_json_atomic(output_root / "output_inventory.json", {"roots": summaries})


def main() -> None:
    args = parse_args()
    if not args.run_until_complete and args.wall_hours <= 0:
        raise ValueError("--wall-hours must be positive unless --run-until-complete is set")
    tasks = build_tasks(stage_pde_data=not args.dry_run)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.family] = counts.get(task.family, 0) + 1
    if args.dry_run:
        print(json.dumps({"tasks": len(tasks), "families": counts}, indent=2))
        return

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    _write_plan(tasks, output_root, args)
    status_log = output_root / "task_status.jsonl"
    completed_ids = _read_finished_ids(status_log)
    pending = [task for task in tasks if task.task_id not in completed_ids]
    pending.sort(key=lambda task: task.phase)
    gpus = [value for value in args.gpus.split(",") if value]
    started_at = datetime.now(timezone.utc)
    deadline_at = None if args.run_until_complete else started_at + timedelta(hours=args.wall_hours)
    deadline_monotonic = None if args.run_until_complete else time.monotonic() + args.wall_hours * 3600
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]] = {}
    failed = 0

    while pending or running:
        now = time.monotonic()
        memory = _gpu_memory()
        dispatch_open = deadline_monotonic is None or now < deadline_monotonic
        if dispatch_open:
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                prerequisite = GPU_PREREQUISITES.get(gpu)
                if prerequisite and _tmux_session_alive(prerequisite):
                    continue
                if memory.get(gpu, 10**9) >= 128:
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

        finished_gpus = []
        for gpu, (process, task, handle, task_started) in running.items():
            returncode = process.poll()
            elapsed = now - task_started
            timed_out = elapsed > args.single_job_timeout_minutes * 60
            if returncode is None and timed_out:
                process.terminate()
                try:
                    returncode = process.wait(timeout=10)
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
            with status_log.open("a", encoding="utf-8") as status:
                status.write(json.dumps(row) + "\n")
            completed_ids.add(task.task_id) if returncode == 0 else None
            failed += returncode != 0
            finished_gpus.append(gpu)
        for gpu in finished_gpus:
            del running[gpu]

        completed_by_family: dict[str, int] = {}
        for task in tasks:
            if task.task_id in completed_ids:
                completed_by_family[task.family] = completed_by_family.get(task.family, 0) + 1
        _write_json_atomic(
            output_root / "campaign_status.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "state": "running" if dispatch_open else "draining",
                "started_at": started_at.isoformat(),
                "dispatch_deadline": deadline_at.isoformat() if deadline_at else None,
                "run_until_complete": args.run_until_complete,
                "planned": len(tasks),
                "completed": len(completed_ids),
                "failed": failed,
                "pending": len(pending),
                "running": {
                    gpu: {
                        "task_id": task.task_id,
                        "family": task.family,
                        "elapsed_seconds": now - task_started,
                    }
                    for gpu, (_, task, _, task_started) in running.items()
                },
                "completed_by_family": completed_by_family,
                "gpu_memory_mib": memory,
            },
        )
        if deadline_monotonic is not None and now >= deadline_monotonic and not running:
            break
        time.sleep(args.poll_seconds)

    _summarize_completed(tasks, output_root)
    _write_json_atomic(
        output_root / "campaign_status.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": (
                "completed_budget"
                if pending
                else "completed_all"
                if len(completed_ids) == len(tasks)
                else "completed_with_failures"
            ),
            "started_at": started_at.isoformat(),
            "dispatch_deadline": deadline_at.isoformat() if deadline_at else None,
            "run_until_complete": args.run_until_complete,
            "planned": len(tasks),
            "completed": len(completed_ids),
            "failed": failed,
            "pending": len(pending),
            "running": {},
        },
    )


if __name__ == "__main__":
    main()
