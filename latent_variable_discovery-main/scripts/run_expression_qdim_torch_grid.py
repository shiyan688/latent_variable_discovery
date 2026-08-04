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


@dataclass(frozen=True)
class Job:
    expression_id: int
    q_dim: int
    gpu: str
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run torch-only q_dim grid for every supported expression.")
    parser.add_argument("--library-csv", type=Path, default=PROJECT_ROOT / "data" / "latent_variable_expressions.csv")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "expressions_qdim123_torch_only")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--q-dims", default="1,2,3")
    parser.add_argument("--only-expression-ids", default=None)
    parser.add_argument("--label-count", type=int, default=50)
    parser.add_argument("--train-samples-per-label", type=int, default=80)
    parser.add_argument("--test-samples-per-label", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cal-steps", type=int, default=1200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument("--orth-weight", type=float, default=0.05)
    parser.add_argument(
        "--orth-type",
        choices=("pearson", "hsic", "nhsic", "distance_correlation", "adversarial", "propensity"),
        default="pearson",
    )
    parser.add_argument(
        "--orth-stats-mode",
        choices=("mean_std", "rich", "rff_kme", "rich_rff_kme"),
        default="mean_std",
    )
    parser.add_argument("--continuity-weight", type=float, default=0.05)
    parser.add_argument("--continuity-grid-size", type=int, default=64)
    parser.add_argument("--cal-q-prior-weight", type=float, default=0.01)
    parser.add_argument("--latent-q-l2-weight", type=float, default=0.0)
    parser.add_argument("--prediction-loss-type", choices=("mse", "label_balanced_mse"), default="mse")
    parser.add_argument("--latent-q-whitening-weight", type=float, default=0.0)
    parser.add_argument("--latent-jacobian-disentanglement-weight", type=float, default=0.0)
    parser.add_argument(
        "--latent-q-canonicalization-mode",
        choices=("none", "output", "train"),
        default="none",
    )
    parser.add_argument("--latent-q-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--latent-q-smoothness-epsilon", type=float, default=0.05)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_csv = resolve(args.library_csv)
    output_root = resolve(args.output_root)
    run_root = output_root / "runs"
    log_root = output_root / "launcher_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args, library_csv, run_root, log_root)
    status_path = output_root / "status.log"
    append_status(
        status_path,
        f"start pending_jobs={len(jobs)} library={library_csv} q_dims={args.q_dims} "
        f"epochs={args.epochs} hidden={args.hidden_sizes}",
    )

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
            returncode = process.poll()
            if returncode is None:
                next_running.append((process, job, handle))
                continue
            handle.write(f"[finish] {datetime.now().isoformat(timespec='seconds')}\n")
            handle.write(f"[returncode] {returncode}\n")
            handle.close()
            finished.append({"expression_id": job.expression_id, "q_dim": job.q_dim, "returncode": returncode})
            append_status(status_path, f"finished expr={job.expression_id} q_dim={job.q_dim} rc={returncode}")
        running = next_running
        time.sleep(10)

    (output_root / "launcher_summary.json").write_text(json.dumps(finished, indent=2), encoding="utf-8")
    append_status(status_path, "done")


def build_jobs(args: argparse.Namespace, library_csv: Path, run_root: Path, log_root: Path) -> list[Job]:
    records = load_expression_library(library_csv)
    only_ids = parse_id_list(args.only_expression_ids)
    q_dims = [int(part.strip()) for part in args.q_dims.split(",") if part.strip()]
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not q_dims:
        raise ValueError("--q-dims cannot be empty.")
    if not gpus:
        raise ValueError("--gpus cannot be empty.")

    supported_ids: list[int] = []
    for record in records:
        if only_ids is not None and record.expression_id not in only_ids:
            continue
        try:
            build_expression_task(record)
        except ValueError:
            continue
        supported_ids.append(record.expression_id)

    jobs: list[Job] = []
    gpu_index = 0
    completed = completed_pairs(run_root) if args.resume else set()
    for expression_id in supported_ids:
        for q_dim in q_dims:
            if (expression_id, q_dim) in completed:
                continue
            gpu = gpus[gpu_index % len(gpus)]
            gpu_index += 1
            log_path = log_root / f"expr{expression_id:03d}_qdim{q_dim}_gpu{gpu}.log"
            jobs.append(Job(expression_id, q_dim, gpu, log_path))
    return jobs


def completed_pairs(run_root: Path) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for summary_path in run_root.glob("*/run_summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") != "success":
            continue
        expression = data.get("expression", {})
        config = data.get("workflow_config", {})
        expression_id = expression.get("expression_id")
        q_dim = config.get("q_dim")
        if expression_id is not None and q_dim is not None:
            pairs.add((int(expression_id), int(q_dim)))
    return pairs


def launch_job(args: argparse.Namespace, library_csv: Path, output_root: Path, job: Job) -> tuple[subprocess.Popen[Any], Any]:
    cmd = [
        str(PROJECT_ROOT / ".venv-lvs-gpu" / "bin" / "python"),
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
        "--latent-feature-stats-mode",
        args.orth_stats_mode,
        "--latent-curve-continuity-weight",
        str(args.continuity_weight),
        "--latent-curve-continuity-grid-size",
        str(args.continuity_grid_size),
        "--calibration-q-prior-weight",
        str(args.cal_q_prior_weight),
        "--latent-q-l2-weight",
        str(args.latent_q_l2_weight),
        "--prediction-loss-type",
        args.prediction_loss_type,
        "--latent-q-whitening-weight",
        str(args.latent_q_whitening_weight),
        "--latent-jacobian-disentanglement-weight",
        str(args.latent_jacobian_disentanglement_weight),
        "--latent-q-canonicalization-mode",
        args.latent_q_canonicalization_mode,
        "--latent-q-smoothness-weight",
        str(args.latent_q_smoothness_weight),
        "--latent-q-smoothness-epsilon",
        str(args.latent_q_smoothness_epsilon),
        "--hidden-sizes",
        args.hidden_sizes,
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
