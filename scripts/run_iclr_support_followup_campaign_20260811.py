#!/usr/bin/env python3
"""Run the frozen 18-job attentive-CNP and encoder-multistart follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import run_iclr_support_encoder_campaign_20260811 as base

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_followup_20260811"
DATASETS = (
    ("nasa_battery_capacity", PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json", 8),
    ("nasa_cmapss_fd001_sensor_response", PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json", 4),
    ("starry_te_seebeck", PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json", 8),
)
METHODS = ("attentive_cnp", "encoder_q_multistart")
SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3,4,5")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--single-job-timeout-minutes", type=float, default=90.0)
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _task_id(dataset: str, method: str, seed: int, q_dim: int) -> str:
    raw = json.dumps([dataset, method, seed, q_dim], separators=(",", ":")).encode()
    return f"support_followup_{hashlib.sha256(raw).hexdigest()[:14]}"


def _command(root: Path, summary: Path, dataset: str, method: str, seed: int, q_dim: int) -> tuple[str, ...]:
    return (
        str(PYTHON), "scripts/run_support_conditioned_real_study.py",
        "--prepared-summary", str(summary), "--dataset", dataset,
        "--method", method, "--seed", str(seed), "--q-dim", str(q_dim),
        "--device", "cuda:0", "--output-root", str(root / "new_methods"),
        "--decoder-epochs", "200", "--encoder-epochs", "200",
        "--support-ratio", "0.3", "--batch-size", "256", "--entity-batch-size", "8",
        "--hidden-sizes", "256,128", "--encoder-hidden-sizes", "128,128",
        "--clip-standard-deviations", "3.0", "--alignment-weight", "0.05",
        "--cal-steps", "200", "--cal-lr", "0.05", "--cal-num-starts", "4",
        "--cal-selection-ratio", "0.25", "--cal-selection-min-rows", "24",
        "--cal-refine-steps", "50", "--max-train-per-label", "256",
        "--max-test-per-label", "256", "--subsample-seed", "20260808",
        "--resume", "--save-artifacts",
    )


def build_tasks(root: Path) -> list[base.Task]:
    tasks: list[base.Task] = []
    for dataset, summary, q_dim in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                tasks.append(base.Task(
                    _task_id(dataset, method, seed, q_dim), "followup", dataset,
                    method, seed, q_dim, _command(root, summary, dataset, method, seed, q_dim)
                ))
    return tasks


def _write_manifest(tasks: list[base.Task], args: argparse.Namespace, gpus: list[str]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    created = base._utc_now()
    path = args.output_root / "campaign_manifest.json"
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    base._write_json_atomic(path, {
        "protocol": "ICLR_SUPPORT_FOLLOWUP_PLAN_20260811.md",
        "created_at": created,
        "initial_created_at": previous.get("initial_created_at", previous.get("created_at", created)),
        "planned_tasks": len(tasks), "datasets": [item[0] for item in DATASETS],
        "methods": list(METHODS), "seeds": list(SEEDS), "gpus": gpus,
        "prior_anchor_root": str(PROJECT_ROOT / "runs" / "iclr_support_encoder_pilot_20260811"),
        "dispatch_policy": "run all 18 frozen jobs; never auto-retry terminal failures",
        "monitoring": {"poll_seconds": args.poll_seconds, "single_job_timeout_minutes": args.single_job_timeout_minutes, "gpu_memory_threshold_mib": args.gpu_memory_threshold_mib},
    })
    with (args.output_root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def run_campaign(args: argparse.Namespace) -> int:
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique GPU indices")
    tasks = build_tasks(args.output_root)
    _write_manifest(tasks, args, gpus)
    if args.dry_run:
        print(json.dumps({"planned_tasks": len(tasks), "gpus": gpus}, indent=2))
        return 0
    event_path = args.output_root / "task_events.jsonl"
    status_path = args.output_root / "campaign_status.json"
    log_root = args.output_root / "job_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    terminal = base._terminal_events(event_path)
    pending = [task for task in tasks if task.task_id not in terminal]
    running: dict[str, base.Running] = {}
    timeout_seconds = args.single_job_timeout_minutes * 60.0
    while pending or running:
        now = time.monotonic()
        for gpu, item in list(running.items()):
            returncode = item.process.poll()
            timed_out = returncode is None and now - item.started_monotonic > timeout_seconds
            if returncode is None and not timed_out:
                continue
            event = "timeout" if timed_out else ("success" if returncode == 0 else "failed")
            if timed_out:
                base._terminate_own_process_group(item.process)
                returncode = item.process.returncode
            item.handle.close()
            row = {"event": event, "task_id": item.task.task_id, "dataset": item.task.dataset,
                   "method": item.task.method, "seed": item.task.seed, "q_dim": item.task.q_dim,
                   "gpu": gpu, "pid": item.process.pid, "started_at": item.started_at,
                   "finished_at": base._utc_now(), "elapsed_seconds": now - item.started_monotonic,
                   "returncode": returncode, "log": str(item.log_path)}
            base._append_event(event_path, row)
            terminal[item.task.task_id] = row
            del running[gpu]
        memory = base._gpu_memory()
        for gpu in gpus:
            if not pending or gpu in running or memory.get(gpu, 10**9) >= args.gpu_memory_threshold_mib:
                continue
            task = pending.pop(0)
            log_path = log_root / f"{task.task_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            started_at = base._utc_now()
            handle.write(f"[{started_at}] CUDA_VISIBLE_DEVICES={gpu} {' '.join(task.command)}\n")
            handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault("MPLCONFIGDIR", "/tmp/lvs-mpl-cache")
            process = subprocess.Popen(task.command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=handle, stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            running[gpu] = base.Running(task, gpu, process, handle, log_path, time.monotonic(), started_at)
            base._append_event(event_path, {"event": "started", "task_id": task.task_id,
                "dataset": task.dataset, "method": task.method, "seed": task.seed,
                "q_dim": task.q_dim, "gpu": gpu, "pid": process.pid,
                "started_at": started_at, "log": str(log_path)})
        base._write_json_atomic(status_path, base._status_payload(tasks, terminal, running, "running"))
        if pending or running:
            time.sleep(max(1, args.poll_seconds))
    state = "completed_all" if all(row.get("event") == "success" for row in terminal.values()) else "completed_with_failures"
    final_status = base._status_payload(tasks, terminal, running, state)
    base._write_json_atomic(status_path, final_status)
    analysis_log = args.output_root / "analysis.log"
    with analysis_log.open("w", encoding="utf-8") as handle:
        analysis = subprocess.run([str(PYTHON), "scripts/analyze_support_followup_20260811.py",
            "--campaign-root", str(args.output_root)], cwd=PROJECT_ROOT,
            stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    final_status["analysis"] = {"returncode": analysis.returncode, "log": str(analysis_log),
        "report": str(args.output_root / "SUPPORT_FOLLOWUP_RESULTS.md"), "completed_at": base._utc_now()}
    if analysis.returncode != 0:
        final_status["state"] = "completed_analysis_failed"
    base._write_json_atomic(status_path, final_status)
    return 0 if state == "completed_all" and analysis.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_campaign(parse_args()))
