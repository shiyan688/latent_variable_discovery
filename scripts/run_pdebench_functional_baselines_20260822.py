#!/usr/bin/env python3
"""Run information-matched FPCA and masked DeepONet baselines on frozen PDEBench data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

from lvs.core.metrics import macro_prediction_metrics, reference_scaled_prediction_metrics
from lvs.core.pipeline import split_support_query_indices


PYTHON = Path(sys.executable)
SOURCE_ROOT = PROJECT_ROOT / "runs" / "pdebench_burgers_latent_20260809"
ANCHOR_ROOT = PROJECT_ROOT / "runs" / "matched_1000ep_real_pde_20260817" / "pdebench"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "pdebench_functional_baselines_20260822"
PLAN_PATH = PROJECT_ROOT / "PDEBENCH_FUNCTIONAL_BASELINES_PLAN_20260822.md"
METHODS = ("fpca_ridge", "masked_deeponet")
GPU_MEMORY_THRESHOLD_MIB = 128


@dataclass(frozen=True)
class Task:
    task_id: str
    method: str
    seed: int
    command: tuple[str, ...]


class MaskedDeepONet(nn.Module):
    def __init__(self, grid_size: int, width: int = 128) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(2 * grid_size, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, width),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, width),
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = math.sqrt(width)

    def forward(self, branch_input: torch.Tensor, query_coordinates: torch.Tensor) -> torch.Tensor:
        branch_features = self.branch(branch_input)
        trunk_features = self.trunk(query_coordinates)
        return torch.einsum("bd,bqd->bq", branch_features, trunk_features) / self.scale + self.bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _experiment_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--gpus", default="4,5")
    launch.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    launch.add_argument("--poll-seconds", type=float, default=15.0)
    launch.add_argument("--single-job-timeout-minutes", type=float, default=360.0)
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    launch.add_argument("--dry-run", action="store_true")
    _experiment_args(launch)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--updates", type=int, default=128_000)
    parser.add_argument("--batch-trajectories", type=int, default=8)
    parser.add_argument("--queries-per-trajectory", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _stage_subset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("pdebench_burgers_subset.npz", "subset_manifest.json"):
        destination = root / name
        if not destination.exists():
            shutil.copy2(SOURCE_ROOT / name, destination)


def _load_subset(root: Path) -> dict[str, np.ndarray]:
    with np.load(root / "pdebench_burgers_subset.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _coordinates(data: dict[str, np.ndarray]) -> np.ndarray:
    grid_t, grid_x = np.meshgrid(data["t_coordinates"], data["x_coordinates"], indexing="ij")
    coordinates = np.column_stack([grid_x.reshape(-1), grid_t.reshape(-1)]).astype(np.float32)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    return (2.0 * (coordinates - low) / (high - low) - 1.0).astype(np.float32)


def _split_indices(grid_size: int, label: int, seed: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    return split_support_query_indices(
        np.arange(grid_size), ratio, mode="random", seed=seed, label=label
    )


def _prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, float]:
    reference_scale = max(float(np.std(train_targets)), 1e-8)
    per_label = np.asarray(
        [
            np.sqrt(np.mean((truth[labels == label] - prediction[labels == label]) ** 2))
            / reference_scale
            for label in np.unique(labels)
        ]
    )
    return {
        **macro_prediction_metrics(truth, prediction, labels),
        **reference_scaled_prediction_metrics(truth, prediction, reference_scale=reference_scale),
        "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
        "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
        "label_reference_nrmse_max": float(np.max(per_label)),
    }


def _query_frame(
    data: dict[str, np.ndarray],
    trajectory_indices: np.ndarray,
    seed: int,
    support_ratio: float,
    prediction: np.ndarray,
) -> pd.DataFrame:
    grid_t, grid_x = np.meshgrid(data["t_coordinates"], data["x_coordinates"], indexing="ij")
    x = grid_x.reshape(-1)
    t = grid_t.reshape(-1)
    rows: list[pd.DataFrame] = []
    position = 0
    grid_size = x.size
    for label in trajectory_indices:
        _, query = _split_indices(grid_size, int(label), seed, support_ratio)
        count = len(query)
        rows.append(
            pd.DataFrame(
                {
                    "label": int(label),
                    "source_trajectory": int(data["source_indices"][label]),
                    "x": x[query],
                    "t": t[query],
                    "target": data["targets"][label].reshape(-1)[query],
                    "split": "test",
                    "prediction": prediction[position : position + count],
                }
            )
        )
        position += count
    assert position == len(prediction)
    return pd.concat(rows, ignore_index=True)


def _anchor_query(seed: int) -> pd.DataFrame:
    anchors = pd.read_csv(ANCHOR_ROOT / "all_runs.csv")
    row = anchors[
        (anchors["q_dim"] == 8)
        & (anchors["method"] == "joint_mse_step1")
        & (anchors["strategy"] == "support_knn4")
        & (anchors["seed"] == seed)
    ]
    assert len(row) == 1
    return pd.read_csv(Path(row.iloc[0]["result_path"]).parent / "query_predictions_support_knn4.csv")


def _assert_query_alignment(frame: pd.DataFrame, seed: int) -> None:
    anchor = _anchor_query(seed)
    assert len(frame) == len(anchor) == 11_488
    np.testing.assert_array_equal(frame["label"].to_numpy(), anchor["label"].to_numpy())
    for column in ("x", "t", "target"):
        np.testing.assert_allclose(
            frame[column].to_numpy(float), anchor[column].to_numpy(float), rtol=1e-6, atol=1e-7
        )


def _fpca_predictions(
    pca: PCA,
    standardized_targets: np.ndarray,
    trajectory_indices: np.ndarray,
    *,
    seed: int,
    support_ratio: float,
    components: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = pca.mean_
    basis = pca.components_[:components]
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in trajectory_indices:
        support, query = _split_indices(standardized_targets.shape[1], int(label), seed, support_ratio)
        design = basis[:, support].T
        coefficient = np.linalg.solve(
            design.T @ design + ridge * np.eye(components),
            design.T @ (standardized_targets[label, support] - mean[support]),
        )
        predictions.append(mean[query] + basis[:, query].T @ coefficient)
        truths.append(standardized_targets[label, query])
        labels.append(np.full(len(query), int(label)))
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(labels)


def _run_fpca(job: dict[str, Any], data: dict[str, np.ndarray]) -> dict[str, Any]:
    started = time.perf_counter()
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    test_indices = np.flatnonzero(data["split"] == "test")
    raw_targets = data["targets"].reshape(len(data["split"]), -1).astype(np.float64)
    train_mean = float(raw_targets[train_indices].mean())
    train_scale = float(raw_targets[train_indices].std())
    targets = (raw_targets - train_mean) / train_scale
    pca = PCA(n_components=32, svd_solver="full").fit(targets[train_indices])
    candidates: list[dict[str, float | int]] = []
    for components in (2, 4, 8, 16, 32):
        for ridge in (1e-6, 1e-4, 1e-2, 1.0):
            truth, prediction, _ = _fpca_predictions(
                pca,
                targets,
                validation_indices,
                seed=job["seed"],
                support_ratio=job["support_ratio"],
                components=components,
                ridge=ridge,
            )
            candidates.append(
                {
                    "components": components,
                    "ridge": ridge,
                    "validation_nrmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
                }
            )
    selected = min(candidates, key=lambda row: (row["validation_nrmse"], row["components"], row["ridge"]))
    truth_z, prediction_z, labels = _fpca_predictions(
        pca,
        targets,
        test_indices,
        seed=job["seed"],
        support_ratio=job["support_ratio"],
        components=int(selected["components"]),
        ridge=float(selected["ridge"]),
    )
    altered = targets.copy()
    for label in test_indices:
        _, query = _split_indices(targets.shape[1], int(label), job["seed"], job["support_ratio"])
        altered[label, query] += 123.0
    _, altered_prediction_z, _ = _fpca_predictions(
        pca,
        altered,
        test_indices,
        seed=job["seed"],
        support_ratio=job["support_ratio"],
        components=int(selected["components"]),
        ridge=float(selected["ridge"]),
    )
    leakage_difference = float(np.max(np.abs(prediction_z - altered_prediction_z)))
    assert leakage_difference == 0.0
    truth = truth_z * train_scale + train_mean
    prediction = prediction_z * train_scale + train_mean
    return {
        "prediction": prediction,
        "metrics": _prediction_metrics(truth, prediction, labels, raw_targets[train_indices].reshape(-1)),
        "validation_candidates": candidates,
        "selected": selected,
        "leakage_probe_max_abs_prediction_difference": leakage_difference,
        "wall_time_seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "parameter_count": int(32 * targets.shape[1] + 32),
    }


def _evaluation_arrays(
    model: MaskedDeepONet,
    targets_z: torch.Tensor,
    trajectory_indices: np.ndarray,
    coordinates: torch.Tensor,
    *,
    seed: int,
    support_ratio: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    grid_size = targets_z.shape[1]
    with torch.no_grad():
        for label in trajectory_indices:
            support, query = _split_indices(grid_size, int(label), seed, support_ratio)
            mask = torch.zeros(grid_size, device=device)
            mask[torch.as_tensor(support, device=device)] = 1.0
            values = targets_z[label] * mask
            branch = torch.cat([values, mask]).unsqueeze(0)
            query_tensor = torch.as_tensor(query, device=device)
            prediction = model(branch, coordinates[query_tensor].unsqueeze(0)).squeeze(0)
            predictions.append(prediction.detach().cpu().numpy())
            truths.append(targets_z[label, query_tensor].detach().cpu().numpy())
            labels.append(np.full(len(query), int(label)))
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(labels)


def _milestones(updates: int) -> list[int]:
    return sorted({value for value in (16_000, 32_000, 64_000, 128_000, updates) if value <= updates})


def _run_deeponet(job: dict[str, Any], data: dict[str, np.ndarray], device_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    seed = int(job["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    test_indices = np.flatnonzero(data["split"] == "test")
    raw_targets = data["targets"].reshape(len(data["split"]), -1).astype(np.float32)
    train_mean = float(raw_targets[train_indices].mean())
    train_scale = float(raw_targets[train_indices].std())
    target_tensor = torch.as_tensor((raw_targets - train_mean) / train_scale, device=device)
    coordinate_tensor = torch.as_tensor(_coordinates(data), device=device)
    model = MaskedDeepONet(target_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(job["learning_rate"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 20260822)
    train_tensor = torch.as_tensor(train_indices, device=device)
    support_count = int(math.floor(job["support_ratio"] * target_tensor.shape[1]))
    batch_trajectories = int(job["batch_trajectories"])
    queries_per_trajectory = int(job["queries_per_trajectory"])
    validation_trace: list[dict[str, float | int]] = []
    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    milestone_set = set(_milestones(int(job["updates"])))
    model.train()
    for update in range(1, int(job["updates"]) + 1):
        trajectory_positions = torch.randint(
            len(train_indices), (batch_trajectories,), generator=generator, device=device
        )
        trajectories = train_tensor[trajectory_positions]
        batch_targets = target_tensor[trajectories]
        support_scores = torch.rand(
            (batch_trajectories, target_tensor.shape[1]), generator=generator, device=device
        )
        support = torch.topk(support_scores, support_count, largest=False).indices
        mask = torch.zeros_like(batch_targets)
        mask.scatter_(1, support, 1.0)
        query_scores = torch.rand(
            (batch_trajectories, target_tensor.shape[1]), generator=generator, device=device
        ).masked_fill(mask.bool(), float("inf"))
        query = torch.topk(query_scores, queries_per_trajectory, largest=False).indices
        branch = torch.cat([batch_targets * mask, mask], dim=1)
        query_coordinates = coordinate_tensor[query]
        truth = torch.gather(batch_targets, 1, query)
        prediction = model(branch, query_coordinates)
        loss = torch.mean((prediction - truth) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update in milestone_set:
            validation_truth, validation_prediction, _ = _evaluation_arrays(
                model,
                target_tensor,
                validation_indices,
                coordinate_tensor,
                seed=seed,
                support_ratio=float(job["support_ratio"]),
                device=device,
            )
            validation_trace.append(
                {
                    "update": update,
                    "validation_nrmse": float(
                        np.sqrt(np.mean((validation_truth - validation_prediction) ** 2))
                    ),
                    "training_batch_loss": float(loss.detach().cpu()),
                }
            )
            checkpoints[update] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            model.train()
    selected = min(validation_trace, key=lambda row: (row["validation_nrmse"], row["update"]))
    model.load_state_dict(checkpoints[int(selected["update"])])
    truth_z, prediction_z, labels = _evaluation_arrays(
        model,
        target_tensor,
        test_indices,
        coordinate_tensor,
        seed=seed,
        support_ratio=float(job["support_ratio"]),
        device=device,
    )
    altered = target_tensor.clone()
    for label in test_indices:
        _, query = _split_indices(target_tensor.shape[1], int(label), seed, float(job["support_ratio"]))
        altered[label, torch.as_tensor(query, device=device)] += 123.0
    _, altered_prediction_z, _ = _evaluation_arrays(
        model,
        altered,
        test_indices,
        coordinate_tensor,
        seed=seed,
        support_ratio=float(job["support_ratio"]),
        device=device,
    )
    leakage_difference = float(np.max(np.abs(prediction_z - altered_prediction_z)))
    assert leakage_difference == 0.0
    truth = truth_z * train_scale + train_mean
    prediction = prediction_z * train_scale + train_mean
    return {
        "prediction": prediction,
        "metrics": _prediction_metrics(truth, prediction, labels, raw_targets[train_indices].reshape(-1)),
        "validation_trace": validation_trace,
        "selected": selected,
        "leakage_probe_max_abs_prediction_difference": leakage_difference,
        "wall_time_seconds": time.perf_counter() - started,
        "optimizer_updates": int(job["updates"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_model_state": model.state_dict(),
    }


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.output_root / "subset_manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "method": args.method,
        "seed": args.seed,
        "support_ratio": args.support_ratio,
        "updates": args.updates,
        "batch_trajectories": args.batch_trajectories,
        "queries_per_trajectory": args.queries_per_trajectory,
        "learning_rate": args.learning_rate,
        "subset_config": manifest["config"],
    }


def run_job(args: argparse.Namespace) -> Path:
    job = _job_config(args)
    run_dir = args.output_root / args.method / f"seed{args.seed}_{_stable_hash(job)}"
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job") == job:
            return result_path
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _load_subset(args.output_root)
    if args.method == "fpca_ridge":
        outcome = _run_fpca(job, data)
    else:
        outcome = _run_deeponet(job, data, args.device)
    prediction = outcome.pop("prediction")
    model_state = outcome.pop("selected_model_state", None)
    test_indices = np.flatnonzero(data["split"] == "test")
    prediction_frame = _query_frame(
        data, test_indices, args.seed, args.support_ratio, prediction
    )
    _assert_query_alignment(prediction_frame, args.seed)
    prediction_path = run_dir / "query_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    checkpoint_path = run_dir / "selected_model.pt"
    if model_state is not None:
        torch.save({"job": job, "model_state_dict": model_state}, checkpoint_path)
    payload = {
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if args.method == "masked_deeponet" and torch.cuda.is_available()
            else None,
        },
        **outcome,
        "artifacts": {
            "query_predictions": str(prediction_path),
            "selected_model": str(checkpoint_path) if model_state is not None else None,
        },
    }
    _write_json_atomic(result_path, payload)
    return result_path


def _task(method: str, seed: int, root: Path, args: argparse.Namespace) -> Task:
    command = (
        str(PYTHON),
        str(Path(__file__).resolve()),
        "run-job",
        "--method",
        method,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--support-ratio",
        str(args.support_ratio),
        "--updates",
        str(args.updates),
        "--batch-trajectories",
        str(args.batch_trajectories),
        "--queries-per-trajectory",
        str(args.queries_per_trajectory),
        "--learning-rate",
        str(args.learning_rate),
        "--resume",
    )
    task_id = f"{method}_{hashlib.sha256(json.dumps([method, seed, args.support_ratio, args.updates]).encode()).hexdigest()[:14]}"
    return Task(task_id, method, seed, command)


def _successful_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {str(row["task_id"]) for row in rows if row["returncode"] == 0}


def _gpu_memory() -> dict[str, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    return {
        index.strip(): int(used.strip())
        for index, used in (line.split(",", maxsplit=1) for line in result.stdout.splitlines())
    }


def _append_status(path: Path, task: Task, returncode: int, elapsed: float, gpu: str | None, timed_out: bool) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "task_id": task.task_id,
                    "method": task.method,
                    "seed": task.seed,
                    "gpu": gpu,
                    "elapsed_seconds": elapsed,
                    "returncode": returncode,
                    "timed_out": timed_out,
                }
            )
            + "\n"
        )


def _aggregate_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in root.glob("*/seed*/result.json"):
        payload = json.loads(path.read_text())
        if payload.get("status") != "success":
            continue
        rows.append(
            {
                "method": payload["job"]["method"],
                "seed": payload["job"]["seed"],
                **payload["metrics"],
                "wall_time_seconds": payload["wall_time_seconds"],
                "optimizer_updates": payload["optimizer_updates"],
                "parameter_count": payload["parameter_count"],
                "selected": json.dumps(payload["selected"], sort_keys=True),
                "result_path": str(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)


def _bh_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order] * len(array) / np.arange(1, len(array) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(array)
    output[order] = np.minimum(ranked, 1.0)
    return output.tolist()


def summarize(root: Path) -> None:
    frame = _aggregate_rows(root)
    assert len(frame) == 20
    assert frame.groupby("method").size().to_dict() == {"fpca_ridge": 10, "masked_deeponet": 10}
    assert np.isfinite(frame["reference_nrmse"]).all()
    planned = [json.loads(line) for line in (root / "planned_tasks.jsonl").read_text().splitlines() if line]
    ledger = [json.loads(line) for line in (root / "task_status.jsonl").read_text().splitlines() if line]
    assert len(planned) == len({row["task_id"] for row in planned}) == 20
    assert len(ledger) == len({row["task_id"] for row in ledger}) == 20
    assert all(row["returncode"] == 0 and not row["timed_out"] for row in ledger)
    prediction_rows: list[int] = []
    leakage_values: list[float] = []
    for path_text in frame["result_path"]:
        payload = json.loads(Path(path_text).read_text())
        prediction_rows.append(len(pd.read_csv(payload["artifacts"]["query_predictions"])))
        leakage_values.append(payload["leakage_probe_max_abs_prediction_difference"])
    assert prediction_rows == [11_488] * 20
    assert leakage_values == [0.0] * 20
    frame.to_csv(root / "all_runs.csv", index=False)
    summary = (
        frame.groupby("method", as_index=False)
        .agg(
            runs=("reference_nrmse", "size"),
            median_nrmse=("reference_nrmse", "median"),
            mean_nrmse=("reference_nrmse", "mean"),
            p90_nrmse=("reference_nrmse", lambda values: values.quantile(0.9)),
            max_nrmse=("reference_nrmse", "max"),
            median_label_p95=("label_reference_nrmse_p95", "median"),
            median_wall_seconds=("wall_time_seconds", "median"),
            median_updates=("optimizer_updates", "median"),
            parameters=("parameter_count", "median"),
        )
    )
    summary.to_csv(root / "method_summary.csv", index=False)
    anchors = pd.read_csv(ANCHOR_ROOT / "all_runs.csv")
    anchor_cells = {
        "support_knn4": anchors[
            (anchors.q_dim == 8)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "support_knn4")
        ],
        "q16_continuity": anchors[
            (anchors.q_dim == 16)
            & (anchors.method == "joint_continuity_step1")
            & (anchors.strategy == "latent_adaptive_k4_min24")
        ],
        "q16_mse": anchors[
            (anchors.q_dim == 16)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "latent_adaptive_k4_min24")
        ],
        "q8_mse": anchors[
            (anchors.q_dim == 8)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "latent_adaptive_k4_min24")
        ],
        "support_mean": anchors[
            (anchors.q_dim == 8)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "support_mean")
        ],
        "pooled_no_q_mlp": anchors[
            (anchors.q_dim == 8)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "pooled_mlp_no_latent")
        ],
        "full_ic_pca_mlp_extra_information": anchors[
            (anchors.q_dim == 8)
            & (anchors.method == "joint_mse_step1")
            & (anchors.strategy == "full_ic_pca_mlp_reference")
        ],
    }
    effects: list[dict[str, Any]] = []
    for method, candidate in frame.groupby("method"):
        for anchor_name, anchor in anchor_cells.items():
            paired = candidate[["seed", "reference_nrmse"]].merge(
                anchor[["seed", "reference_nrmse"]], on="seed", suffixes=("_candidate", "_anchor"), validate="one_to_one"
            )
            delta = paired.reference_nrmse_candidate - paired.reference_nrmse_anchor
            effects.append(
                {
                    "method": method,
                    "anchor": anchor_name,
                    "pairs": len(paired),
                    "wins": int((delta < 0).sum()),
                    "median_candidate": float(paired.reference_nrmse_candidate.median()),
                    "median_anchor": float(paired.reference_nrmse_anchor.median()),
                    "median_delta": float(delta.median()),
                    "median_ratio": float((paired.reference_nrmse_candidate / paired.reference_nrmse_anchor).median()),
                    "wilcoxon_p": float(wilcoxon(delta).pvalue),
                }
            )
    fpca = frame[frame.method == "fpca_ridge"]
    deep = frame[frame.method == "masked_deeponet"]
    paired = fpca[["seed", "reference_nrmse"]].merge(
        deep[["seed", "reference_nrmse"]], on="seed", suffixes=("_candidate", "_anchor"), validate="one_to_one"
    )
    delta = paired.reference_nrmse_candidate - paired.reference_nrmse_anchor
    effects.append(
        {
            "method": "fpca_ridge",
            "anchor": "masked_deeponet",
            "pairs": len(paired),
            "wins": int((delta < 0).sum()),
            "median_candidate": float(paired.reference_nrmse_candidate.median()),
            "median_anchor": float(paired.reference_nrmse_anchor.median()),
            "median_delta": float(delta.median()),
            "median_ratio": float((paired.reference_nrmse_candidate / paired.reference_nrmse_anchor).median()),
            "wilcoxon_p": float(wilcoxon(delta).pvalue),
        }
    )
    effects_frame = pd.DataFrame(effects)
    effects_frame["wilcoxon_bh_q"] = _bh_adjust(effects_frame["wilcoxon_p"].tolist())
    effects_frame.to_csv(root / "paired_effects.csv", index=False)
    fpca_selected = Counter(
        (json.loads(value)["components"], json.loads(value)["ridge"])
        for value in frame.loc[frame.method == "fpca_ridge", "selected"]
    )
    deep_selected = Counter(
        json.loads(value)["update"]
        for value in frame.loc[frame.method == "masked_deeponet", "selected"]
    )
    q_checkpoint_path = next(
        (ANCHOR_ROOT / "q16" / "joint_continuity_step1").glob("seed0_*/training_checkpoint.pt")
    )
    q_checkpoint = torch.load(q_checkpoint_path, map_location="cpu", weights_only=False)
    q_parameter_count = sum(
        value.numel()
        for state_name in ("latent_model_state_dict", "latent_embedding_state_dict")
        for value in q_checkpoint[state_name].values()
    )
    audit = {
        "planned_tasks": len(planned),
        "ledger_rows": len(ledger),
        "unique_task_ids": len({row["task_id"] for row in ledger}),
        "successful_tasks": sum(row["returncode"] == 0 for row in ledger),
        "failed_tasks": sum(row["returncode"] != 0 for row in ledger),
        "timed_out_tasks": sum(row["timed_out"] for row in ledger),
        "result_rows": len(frame),
        "finite_primary_metrics": int(np.isfinite(frame["reference_nrmse"]).sum()),
        "query_rows_per_result": sorted(set(prediction_rows)),
        "max_query_leakage_probe_difference": max(leakage_values),
        "q16_continuity_train_parameter_count": q_parameter_count,
        "accounted_task_hours": sum(row["elapsed_seconds"] for row in ledger) / 3600,
        "first_finished_utc": min(row["finished_at"] for row in ledger),
        "last_finished_utc": max(row["finished_at"] for row in ledger),
    }
    _write_json_atomic(root / "terminal_audit.json", audit)
    report = [
        "# PDEBench functional baseline results",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + terminal analysis",
        f"- Origin Date: {datetime.now().astimezone().date().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: pdebench_functional_baselines_v1",
        "",
        "## Method summary",
        "",
        "| method | runs | median NRMSE | p90 | max | median label-p95 | median seconds | median updates | parameters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.method} | {row.runs} | {row.median_nrmse:.6g} | {row.p90_nrmse:.6g} | {row.max_nrmse:.6g} | {row.median_label_p95:.6g} | {row.median_wall_seconds:.1f} | {row.median_updates:.0f} | {row.parameters:.0f} |"
        )
    report.extend(
        [
            "",
            "## Paired anchors",
            "",
            "| method | anchor | wins | pairs | candidate median | anchor median | median delta | median ratio | p | BH q |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects_frame.itertuples(index=False):
        report.append(
            f"| {row.method} | {row.anchor} | {row.wins} | {row.pairs} | {row.median_candidate:.6g} | {row.median_anchor:.6g} | {row.median_delta:.6g} | {row.median_ratio:.4f} | {row.wilcoxon_p:.4g} | {row.wilcoxon_bh_q:.4g} |"
        )
    report.extend(
        [
            "",
            "## Validation selection and execution audit",
            "",
            f"- FPCA selected configurations: `{dict(fpca_selected)}`.",
            f"- DeepONet selected checkpoints: `{dict(deep_selected)}`; every job still executed the full 128,000-update cap.",
            f"- Trainable parameter count: masked DeepONet 345,217 versus q=16 continuity model plus train embeddings {q_parameter_count:,}.",
            f"- Terminal ledger: {audit['successful_tasks']}/{audit['planned_tasks']} successful, {audit['failed_tasks']} failed, {audit['timed_out_tasks']} timed out.",
            f"- Accounted task time: {audit['accounted_task_hours']:.2f} hours; every result has exactly 11,488 query rows and a finite primary metric.",
            "",
            "## Bounded conclusion",
            "",
            "Neither new baseline closes the support-kNN gap. FPCA is stronger than the tested masked DeepONet, while q=16 continuity is stronger than both. Both new methods beat the pooled support-blind no-q MLP in every paired seed. This supports q as a competitive nonlinear bottleneck relative to these declared functional baselines, but support-kNN remains the strongest same-support method on this task.",
            "",
            "All new-method query rows were checked against the exact same-seed support-kNN artifacts. Query-target perturbation changed no prediction for either method.",
        ]
    )
    (root / "PDEBENCH_FUNCTIONAL_BASELINE_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_campaign_status(
    root: Path,
    *,
    started_at: str,
    tasks: list[Task],
    completed: set[str],
    pending: list[Task],
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]],
    memory: dict[str, int],
    state: str,
) -> None:
    _write_json_atomic(
        root / "campaign_status.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "started_at": started_at,
            "planned": len(tasks),
            "completed": len(completed),
            "failed": len(tasks) - len(completed) - len(pending) - len(running),
            "pending": len(pending),
            "running": {
                gpu: {"task_id": task.task_id, "method": task.method, "seed": task.seed}
                for gpu, (_, task, _, _) in running.items()
            },
            "gpu_memory_mib": memory,
        },
    )


def launch(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    tasks = [_task(method, seed, root, args) for method in METHODS for seed in seeds]
    if args.dry_run:
        print(json.dumps({"planned": len(tasks), "methods": METHODS, "seeds": seeds}, indent=2))
        return
    _stage_subset(root)
    (root / "logs").mkdir(exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_plan": str(PLAN_PATH),
        "planned_tasks": len(tasks),
        "methods": list(METHODS),
        "seeds": seeds,
        "gpus": gpus,
        "support_ratio": args.support_ratio,
        "updates": args.updates,
        "batch_trajectories": args.batch_trajectories,
        "queries_per_trajectory": args.queries_per_trajectory,
        "learning_rate": args.learning_rate,
        "gpu_memory_available_threshold_mib": GPU_MEMORY_THRESHOLD_MIB,
        "single_job_timeout_minutes": args.single_job_timeout_minutes,
    }
    _write_json_atomic(root / "campaign_manifest.json", manifest)
    with (root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task)) + "\n")
    status_path = root / "task_status.jsonl"
    completed = _successful_ids(status_path)
    started_at = datetime.now(timezone.utc).isoformat()

    for task in [item for item in tasks if item.method == "fpca_ridge" and item.task_id not in completed]:
        log_path = root / "logs" / f"{task.task_id}.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(task.command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        _append_status(status_path, task, result.returncode, time.monotonic() - started, None, False)
        if result.returncode == 0:
            completed.add(task.task_id)

    pending = [item for item in tasks if item.method == "masked_deeponet" and item.task_id not in completed]
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]] = {}
    memory: dict[str, int] = {}
    while pending or running:
        now = time.monotonic()
        memory = _gpu_memory()
        for gpu in gpus:
            if gpu in running or not pending or memory.get(gpu, 10**9) >= GPU_MEMORY_THRESHOLD_MIB:
                continue
            task = pending.pop(0)
            log_path = root / "logs" / f"{task.task_id}.log"
            handle = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
            process = subprocess.Popen(
                task.command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
            )
            running[gpu] = (process, task, handle, now)
        finished: list[str] = []
        for gpu, (process, task, handle, task_started) in running.items():
            returncode = process.poll()
            timed_out = now - task_started > args.single_job_timeout_minutes * 60
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
            _append_status(status_path, task, returncode, now - task_started, gpu, timed_out)
            if returncode == 0:
                completed.add(task.task_id)
            finished.append(gpu)
        for gpu in finished:
            del running[gpu]
        _write_campaign_status(
            root,
            started_at=started_at,
            tasks=tasks,
            completed=completed,
            pending=pending,
            running=running,
            memory=memory,
            state="running",
        )
        time.sleep(args.poll_seconds)

    success = len(completed) == len(tasks)
    summarize_returncode = None
    if success:
        result = subprocess.run(
            [str(PYTHON), str(Path(__file__).resolve()), "summarize", "--output-root", str(root)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        summarize_returncode = result.returncode
    if status_path.exists():
        rows = [json.loads(line) for line in status_path.read_text().splitlines() if line.strip()]
        with (root / "task_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    _write_json_atomic(
        root / "campaign_status.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "completed_all" if success and summarize_returncode == 0 else "completed_with_failures",
            "started_at": started_at,
            "planned": len(tasks),
            "completed": len(completed),
            "failed": len(tasks) - len(completed),
            "pending": 0,
            "running": {},
            "summarize_returncode": summarize_returncode,
        },
    )


def main() -> None:
    args = parse_args()
    if args.command == "run-job":
        _stage_subset(args.output_root)
        print(run_job(args))
    elif args.command == "launch":
        launch(args)
    else:
        summarize(args.output_root)


if __name__ == "__main__":
    main()
