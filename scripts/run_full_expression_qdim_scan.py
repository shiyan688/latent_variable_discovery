#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_expression_library import build_expression_task, load_expression_library  # noqa: E402


@dataclass
class Job:
    expression_id: int
    q_dim: int
    gpu: str
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all supported expressions with q_dim equal to ground-truth q count.")
    parser.add_argument("--library-csv", type=Path, default=PROJECT_ROOT / "data" / "latent_variable_expressions_xrange10.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "full_expressions_xrange10_newloss")
    parser.add_argument("--gpus", default="5,6")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--start-after-id", type=int, default=None)
    parser.add_argument("--only-expression-ids", default=None)
    parser.add_argument("--label-count", type=int, default=50)
    parser.add_argument("--train-samples-per-label", type=int, default=80)
    parser.add_argument("--test-samples-per-label", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cal-steps", type=int, default=4000)
    parser.add_argument("--cal-lr", type=float, default=0.1)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument("--orth-weight", type=float, default=0.05)
    parser.add_argument(
        "--orth-type",
        choices=("pearson", "hsic", "distance_correlation", "adversarial", "propensity"),
        default="pearson",
    )
    parser.add_argument("--continuity-weight", type=float, default=0.05)
    parser.add_argument("--continuity-grid-size", type=int, default=64)
    parser.add_argument("--cal-q-prior-weight", type=float, default=0.1)
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--symbolic-sample-size", type=int, default=3000)
    parser.add_argument("--dag-max-orders", type=int, default=10000)
    parser.add_argument("--dag-max-time", type=float, default=20000.0)
    parser.add_argument("--quiet", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_csv = resolve(args.library_csv)
    output_root = resolve(args.output_root)
    log_root = output_root / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, library_csv, log_root)
    status_path = output_root / "status.log"
    append_status(status_path, f"start total_jobs={len(jobs)} library={library_csv}")

    running: list[tuple[subprocess.Popen[Any], Job, Any]] = []
    pending = list(jobs)
    finished: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            process, handle = launch_job(args, library_csv, output_root, job)
            running.append((process, job, handle))
            append_status(status_path, f"launched expr={job.expression_id} q_dim={job.q_dim} gpu={job.gpu}")

        next_running: list[tuple[subprocess.Popen[Any], Job, Any]] = []
        for process, job, handle in running:
            rc = process.poll()
            if rc is None:
                next_running.append((process, job, handle))
                continue
            handle.write(f"[finish] {datetime.now().isoformat(timespec='seconds')}\n")
            handle.write(f"[returncode] {rc}\n")
            handle.close()
            finished.append({"expression_id": job.expression_id, "q_dim": job.q_dim, "returncode": rc})
            append_status(status_path, f"finished expr={job.expression_id} q_dim={job.q_dim} rc={rc}")
        running = next_running
        time.sleep(10)

    (output_root / "launcher_summary.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
    append_status(status_path, "done")


def build_jobs(args: argparse.Namespace, library_csv: Path, log_root: Path) -> list[Job]:
    records = load_expression_library(library_csv)
    only_ids = parse_id_list(args.only_expression_ids)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    jobs: list[Job] = []
    gpu_index = 0
    for record in records:
        if only_ids is not None and record.expression_id not in only_ids:
            continue
        if args.start_after_id is not None and record.expression_id <= args.start_after_id:
            continue
        try:
            task = build_expression_task(record)
        except ValueError:
            continue
        gpu = gpus[gpu_index % len(gpus)]
        gpu_index += 1
        log_path = log_root / f"expr{record.expression_id:03d}_qdim{task.ground_truth_latent_dim}_gpu{gpu}.log"
        jobs.append(Job(record.expression_id, task.ground_truth_latent_dim, gpu, log_path))
    return jobs


def launch_job(args: argparse.Namespace, library_csv: Path, output_root: Path, job: Job) -> tuple[subprocess.Popen[Any], Any]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_workflow.py"),
        "--library-csv",
        str(library_csv),
        "--expression-id",
        str(job.expression_id),
        "--backend",
        "torch",
        "--q-dim",
        str(job.q_dim),
        "--label-count",
        str(args.label_count),
        "--train-samples-per-label",
        str(args.train_samples_per_label),
        "--test-samples-per-label",
        str(args.test_samples_per_label),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--cal-steps",
        str(args.cal_steps),
        "--cal-lr",
        str(args.cal_lr),
        "--cal-ratio",
        str(args.cal_ratio),
        "--latent-feature-orthogonality-weight",
        str(args.orth_weight),
        "--latent-feature-orthogonality-type",
        args.orth_type,
        "--latent-curve-continuity-weight",
        str(args.continuity_weight),
        "--latent-curve-continuity-grid-size",
        str(args.continuity_grid_size),
        "--calibration-q-prior-weight",
        str(args.cal_q_prior_weight),
        "--hidden-sizes",
        args.hidden_sizes,
        "--run-symbolic",
        "--symbolic-sample-size",
        str(args.symbolic_sample_size),
        "--symbolic-dag-max-orders",
        str(args.dag_max_orders),
        "--symbolic-dag-max-time",
        str(args.dag_max_time),
        "--output-root",
        str(output_root / "runs"),
        "--quiet",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")
    env["PYTHONUNBUFFERED"] = "1"
    handle = job.log_path.open("w", encoding="utf-8")
    handle.write(f"[start] {datetime.now().isoformat(timespec='seconds')}\n")
    handle.write(f"[expression_id] {job.expression_id}\n")
    handle.write(f"[q_dim] {job.q_dim}\n")
    handle.write(f"[gpu_id] {job.gpu}\n")
    handle.write("[command] " + " ".join(cmd) + "\n")
    handle.flush()
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env)
    return process, handle


def parse_id_list(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def append_status(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
