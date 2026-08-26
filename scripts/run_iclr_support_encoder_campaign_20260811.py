#!/usr/bin/env python3
"""Frozen four-GPU pilot for learned support conditioning and q refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_ROOT / ".venv-lvs-gpu" / "bin" / "python"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_encoder_pilot_20260811"
DATASETS = (
    (
        "nasa_battery_capacity",
        PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json",
        8,
    ),
    (
        "nasa_cmapss_fd001_sensor_response",
        PROJECT_ROOT / "data" / "real_datasets2" / "prepared" / "prepared_datasets.json",
        4,
    ),
    (
        "starry_te_seebeck",
        PROJECT_ROOT / "data" / "application_full_features" / "prepared_datasets.json",
        8,
    ),
)
NEW_METHODS = ("deepsets_direct", "encoder_q_refine")
ANCHORS = ("joint_continuity", "no_q_mlp", "random_forest", "support_knn")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    dataset: str
    method: str
    seed: int
    q_dim: int
    command: tuple[str, ...]


@dataclass
class Running:
    task: Task
    gpu: str
    process: subprocess.Popen[str]
    handle: TextIO
    log_path: Path
    started_monotonic: float
    started_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,3,4,5")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--single-job-timeout-minutes", type=float, default=90.0)
    parser.add_argument("--gpu-memory-threshold-mib", type=int, default=128)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_id(dataset: str, method: str, seed: int, q_dim: int) -> str:
    raw = json.dumps([dataset, method, seed, q_dim], separators=(",", ":")).encode()
    return f"support_pilot_{hashlib.sha256(raw).hexdigest()[:14]}"


def _new_method_command(
    output_root: Path, summary: Path, dataset: str, method: str, seed: int, q_dim: int
) -> tuple[str, ...]:
    return (
        str(PYTHON),
        "scripts/run_support_conditioned_real_study.py",
        "--prepared-summary", str(summary),
        "--dataset", dataset,
        "--method", method,
        "--seed", str(seed),
        "--q-dim", str(q_dim),
        "--device", "cuda:0",
        "--output-root", str(output_root / "new_methods"),
        "--decoder-epochs", "200",
        "--encoder-epochs", "200",
        "--support-ratio", "0.3",
        "--batch-size", "256",
        "--entity-batch-size", "8",
        "--hidden-sizes", "256,128",
        "--encoder-hidden-sizes", "128,128",
        "--refine-steps", "50",
        "--refine-lr", "0.02",
        "--trust-region-weight", "0.01",
        "--clip-standard-deviations", "3.0",
        "--alignment-weight", "0.05",
        "--max-train-per-label", "256",
        "--max-test-per-label", "256",
        "--subsample-seed", "20260808",
        "--resume",
        "--save-artifacts",
    )


def _anchor_command(
    output_root: Path, summary: Path, dataset: str, method: str, seed: int, q_dim: int
) -> tuple[str, ...]:
    return (
        str(PYTHON),
        "scripts/run_iclr_real_discovery.py",
        "run-job",
        "--prepared-summary", str(summary),
        "--dataset", dataset,
        "--method", method,
        "--seed", str(seed),
        "--q-dim", str(q_dim),
        "--device", "cuda:0",
        "--output-root", str(output_root / "anchors"),
        "--epochs", "200",
        "--cal-steps", "200",
        "--cal-init-mode", "prior_random",
        "--cal-num-starts", "4",
        "--cal-selection-ratio", "0.25",
        "--cal-selection-min-rows", "24",
        "--cal-refine-steps", "50",
        "--cal-refine-only-after-selection",
        "--support-ratio", "0.3",
        "--batch-size", "256",
        "--hidden-sizes", "256,128",
        "--max-train-per-label", "256",
        "--max-test-per-label", "256",
        "--subsample-seed", "20260808",
        "--resume",
        "--save-artifacts",
    )


def build_tasks(output_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for dataset, summary, q_dim in DATASETS:
        for method in (*NEW_METHODS, *ANCHORS):
            for seed in SEEDS:
                command = (
                    _new_method_command(output_root, summary, dataset, method, seed, q_dim)
                    if method in NEW_METHODS
                    else _anchor_command(output_root, summary, dataset, method, seed, q_dim)
                )
                tasks.append(
                    Task(
                        task_id=_task_id(dataset, method, seed, q_dim),
                        family="new_method" if method in NEW_METHODS else "anchor",
                        dataset=dataset,
                        method=method,
                        seed=seed,
                        q_dim=q_dim,
                        command=command,
                    )
                )
    return tasks


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _terminal_events(path: Path) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") in {"success", "failed", "timeout"}:
            events[row["task_id"]] = row
    return events


def _write_manifest(tasks: list[Task], args: argparse.Namespace, gpus: list[str]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "campaign_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    created = _utc_now()
    manifest = {
        "protocol": "ICLR_SUPPORT_ENCODER_PILOT_PLAN_20260811.md",
        "created_at": created,
        "initial_created_at": previous.get("initial_created_at", previous.get("created_at", created)),
        "planned_tasks": len(tasks),
        "datasets": [item[0] for item in DATASETS],
        "methods": [*NEW_METHODS, *ANCHORS],
        "seeds": list(SEEDS),
        "gpus": gpus,
        "dispatch_policy": "run frozen matrix to a terminal state; never auto-retry failures",
        "monitoring": {
            "poll_seconds": args.poll_seconds,
            "single_job_timeout_minutes": args.single_job_timeout_minutes,
            "gpu_memory_threshold_mib": args.gpu_memory_threshold_mib,
        },
    }
    _write_json_atomic(manifest_path, manifest)
    with (args.output_root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def _gpu_memory() -> dict[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    memory: dict[str, int] = {}
    for line in result.stdout.splitlines():
        index, used = (item.strip() for item in line.split(",", 1))
        memory[index] = int(used)
    return memory


def _terminate_own_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _status_payload(
    tasks: list[Task], terminal: dict[str, dict[str, Any]], running: dict[str, Running], state: str
) -> dict[str, Any]:
    successes = sum(row.get("event") == "success" for row in terminal.values())
    failures = sum(row.get("event") == "failed" for row in terminal.values())
    timeouts = sum(row.get("event") == "timeout" for row in terminal.values())
    return {
        "updated_at": _utc_now(),
        "state": state,
        "planned": len(tasks),
        "completed": len(terminal),
        "success": successes,
        "failed": failures,
        "timeout": timeouts,
        "running": len(running),
        "pending": len(tasks) - len(terminal) - len(running),
        "running_jobs": [
            {
                "task_id": item.task.task_id,
                "dataset": item.task.dataset,
                "method": item.task.method,
                "seed": item.task.seed,
                "gpu": item.gpu,
                "pid": item.process.pid,
                "started_at": item.started_at,
                "elapsed_seconds": time.monotonic() - item.started_monotonic,
                "log": str(item.log_path),
            }
            for item in running.values()
        ],
    }


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
    terminal = _terminal_events(event_path)
    pending = [task for task in tasks if task.task_id not in terminal]
    running: dict[str, Running] = {}
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
                _terminate_own_process_group(item.process)
                returncode = item.process.returncode
            item.handle.close()
            row = {
                "event": event,
                "task_id": item.task.task_id,
                "dataset": item.task.dataset,
                "method": item.task.method,
                "seed": item.task.seed,
                "q_dim": item.task.q_dim,
                "gpu": gpu,
                "pid": item.process.pid,
                "started_at": item.started_at,
                "finished_at": _utc_now(),
                "elapsed_seconds": now - item.started_monotonic,
                "returncode": returncode,
                "log": str(item.log_path),
            }
            _append_event(event_path, row)
            terminal[item.task.task_id] = row
            del running[gpu]

        memory = _gpu_memory()
        for gpu in gpus:
            if not pending or gpu in running or memory.get(gpu, 10**9) >= args.gpu_memory_threshold_mib:
                continue
            task = pending.pop(0)
            log_path = log_root / f"{task.task_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            started_at = _utc_now()
            handle.write(f"[{started_at}] CUDA_VISIBLE_DEVICES={gpu} {' '.join(task.command)}\n")
            handle.flush()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.setdefault("MPLCONFIGDIR", "/tmp/lvs-mpl-cache")
            process = subprocess.Popen(
                task.command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            running[gpu] = Running(
                task=task,
                gpu=gpu,
                process=process,
                handle=handle,
                log_path=log_path,
                started_monotonic=time.monotonic(),
                started_at=started_at,
            )
            _append_event(
                event_path,
                {
                    "event": "started",
                    "task_id": task.task_id,
                    "dataset": task.dataset,
                    "method": task.method,
                    "seed": task.seed,
                    "q_dim": task.q_dim,
                    "gpu": gpu,
                    "pid": process.pid,
                    "started_at": started_at,
                    "log": str(log_path),
                },
            )

        _write_json_atomic(status_path, _status_payload(tasks, terminal, running, "running"))
        if pending or running:
            time.sleep(max(1, args.poll_seconds))

    state = "completed_all" if all(row.get("event") == "success" for row in terminal.values()) else "completed_with_failures"
    final_status = _status_payload(tasks, terminal, running, state)
    _write_json_atomic(status_path, final_status)
    analysis_log = args.output_root / "analysis.log"
    with analysis_log.open("w", encoding="utf-8") as handle:
        analysis = subprocess.run(
            [str(PYTHON), "scripts/analyze_support_encoder_pilot_20260811.py", "--campaign-root", str(args.output_root)],
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    final_status["analysis"] = {
        "returncode": analysis.returncode,
        "log": str(analysis_log),
        "report": str(args.output_root / "SUPPORT_ENCODER_PILOT_RESULTS.md"),
        "completed_at": _utc_now(),
    }
    if analysis.returncode != 0:
        final_status["state"] = "completed_analysis_failed"
    _write_json_atomic(status_path, final_status)
    return 0 if state == "completed_all" and analysis.returncode == 0 else 1


def main() -> None:
    raise SystemExit(run_campaign(parse_args()))


if __name__ == "__main__":
    main()
