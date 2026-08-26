#!/usr/bin/env python3
"""Run the frozen q-versus-kNN support-internal selector confirmation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

import run_iclr_real_discovery as real
from lvs.backends.torch_mlp import build_torch_model_factory, parse_hidden_sizes
from lvs.core.metrics import (
    effective_rank,
    fit_procrustes_alignment,
    grouped_rff_signatures,
    knn_overlap,
    local_distance_distortion,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
)
from lvs.core.pipeline import (
    OutputConfig,
    build_dataset_from_arrays,
    calibrate_latent_q_for_test_labels,
    evaluate_latent_q_pipeline,
    train_latent_q_model,
)


PYTHON = PROJECT_ROOT / ".venv-lvs-gpu" / "bin" / "python"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "q_knn_reliability_selector_confirm_20260822"
PLAN_PATH = PROJECT_ROOT / "Q_KNN_RELIABILITY_SELECTOR_PLAN_20260822.md"
GPU_MEMORY_THRESHOLD_MIB = 128
DATASETS = (
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


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    seed: int
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-job")
    run.add_argument("--prepared-summary", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _experiment_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--gpus", default="4,5")
    launch.add_argument("--seeds", default="20,21,22,23,24,25,26,27,28,29")
    launch.add_argument("--poll-seconds", type=float, default=15.0)
    launch.add_argument("--single-job-timeout-minutes", type=float, default=240.0)
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    launch.add_argument("--dry-run", action="store_true")
    _experiment_args(launch)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--q-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--cal-steps", type=int, default=200)
    parser.add_argument("--cal-num-starts", type=int, default=4)
    parser.add_argument("--cal-selection-ratio", type=float, default=0.25)
    parser.add_argument("--cal-selection-min-rows", type=int, default=24)
    parser.add_argument("--cal-refine-steps", type=int, default=50)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--selector-fit-ratio", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--max-train-per-label", type=int, default=256)
    parser.add_argument("--max-test-per-label", type=int, default=256)
    parser.add_argument("--subsample-seed", type=int, default=20260808)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "prepared_summary": str(args.prepared_summary),
        "method": "q_knn_support_internal_selector",
        "latent_method": "joint_continuity_step1",
        "seed": args.seed,
        "q_dim": args.q_dim,
        "epochs": args.epochs,
        "cal_steps": args.cal_steps,
        "cal_num_starts": args.cal_num_starts,
        "cal_selection_ratio": args.cal_selection_ratio,
        "cal_selection_min_rows": args.cal_selection_min_rows,
        "cal_refine_steps": args.cal_refine_steps,
        "support_ratio": args.support_ratio,
        "selector_fit_ratio": args.selector_fit_ratio,
        "selector_seed_offset": 104729,
        "selector_score": "physical_mae",
        "selector_tie_rule": "support_knn",
        "batch_size": args.batch_size,
        "hidden_sizes": args.hidden_sizes,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
    }


def _latent_config(args: argparse.Namespace) -> Any:
    values = {
        "q_dim": args.q_dim,
        "epochs": args.epochs,
        "cal_steps": args.cal_steps,
        "cal_init_mode": "prior_random",
        "cal_num_starts": args.cal_num_starts,
        "cal_selection_ratio": args.cal_selection_ratio,
        "cal_selection_min_rows": args.cal_selection_min_rows,
        "cal_refine_steps": args.cal_refine_steps,
        "cal_refine_only_after_selection": True,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
    }
    return real._latent_config(argparse.Namespace(**values), real.METHODS["joint_continuity_step1"])


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    train_y: np.ndarray,
) -> dict[str, float]:
    reference_scale = max(float(np.std(train_y)), 1e-8)
    per_label = np.asarray(
        [
            np.sqrt(np.mean((truth[labels == label] - prediction[labels == label]) ** 2))
            / reference_scale
            for label in pd.unique(labels)
        ]
    )
    return {
        **real._prediction_payload(truth, prediction, labels, train_y),
        "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
        "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
        "label_reference_nrmse_max": float(per_label.max()),
    }


def _q_geometry(
    query_frame: pd.DataFrame,
    feature_columns: list[str],
    q_by_label: dict[Any, np.ndarray],
    seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    labels, signatures = grouped_rff_signatures(
        np.column_stack(
            [
                query_frame[feature_columns].to_numpy(float),
                query_frame["target"].to_numpy(float),
            ]
        ),
        query_frame["label"].to_numpy(),
        n_components=64,
        seed=20260808,
    )
    q_values = np.vstack([q_by_label[label] for label in labels])
    curve = neighborhood_preservation_curve(
        signatures, q_values, max_k=min(10, (len(labels) - 1) // 2)
    )
    geometry = pairwise_distance_metrics(q_values, signatures)
    distortion = local_distance_distortion(
        signatures, q_values, k=min(5, len(labels) - 1)
    )
    return (
        {
            "response_continuity_auc": float(np.mean([row["continuity"] for row in curve])),
            "response_trustworthiness_auc": float(
                np.mean([row["trustworthiness"] for row in curve])
            ),
            "response_knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve])),
            **{f"response_{key}": value for key, value in geometry.items()},
            **{f"response_{key}": value for key, value in distortion.items()},
            "effective_rank": effective_rank(q_values),
        },
        labels,
        q_values,
    )


def run_job(args: argparse.Namespace) -> Path:
    job = _job_config(args)
    run_dir = (
        args.output_root
        / "results"
        / args.dataset
        / f"seed{args.seed}_{_stable_hash(job)}"
    )
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job") == job:
            return result_path
    run_dir.mkdir(parents=True, exist_ok=True)

    record = real._load_record(args.prepared_summary, args.dataset)
    feature_columns = list(record["feature_columns"])
    train_frame = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["train_csv"])),
        args.max_train_per_label,
        args.subsample_seed,
    )
    test_frame = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["test_csv"])),
        args.max_test_per_label,
        args.subsample_seed + 10000,
    )
    train_x = train_frame[feature_columns].to_numpy(np.float32)
    train_y = train_frame["target"].to_numpy(np.float32)
    train_labels = train_frame["label"].to_numpy()
    test_x = test_frame[feature_columns].to_numpy(np.float32)
    test_y = test_frame["target"].to_numpy(np.float32)
    test_labels = test_frame["label"].to_numpy()
    support_indices, query_indices = real._support_query_indices(
        test_labels, args.support_ratio, args.seed
    )

    started = time.perf_counter()
    config = _latent_config(args)
    train_dataset = build_dataset_from_arrays(
        train_x, train_labels, train_y, feature_names=feature_columns
    )
    test_dataset = build_dataset_from_arrays(
        test_x, test_labels, test_y, feature_names=feature_columns
    )
    training = train_latent_q_model(
        train_dataset,
        build_torch_model_factory(parse_hidden_sizes(args.hidden_sizes)),
        config,
    )
    q_result = evaluate_latent_q_pipeline(
        train_dataset,
        test_dataset,
        training,
        config,
        output_config=OutputConfig(save_csv=False, save_plot=False),
    )
    np.testing.assert_array_equal(q_result.eval_indices, query_indices)
    q_prediction = q_result.eval_predictions
    truth = q_result.eval_targets
    query_labels = q_result.eval_labels
    knn_prediction = real._run_support_knn(
        test_x, test_y, test_labels, support_indices, query_indices
    )

    support_dataset = build_dataset_from_arrays(
        test_x[support_indices],
        test_labels[support_indices],
        test_y[support_indices],
        feature_names=feature_columns,
    )
    selector_seed = args.seed + 104729
    selector_config = replace(
        config,
        calibration_ratio=args.selector_fit_ratio,
        seed=selector_seed,
    )
    q_selector = calibrate_latent_q_for_test_labels(
        support_dataset, training, selector_config
    )
    selector_fit, selector_validation = real._support_query_indices(
        test_labels[support_indices], args.selector_fit_ratio, selector_seed
    )
    np.testing.assert_array_equal(q_selector.eval_indices, selector_validation)
    knn_selector_prediction = real._run_support_knn(
        test_x[support_indices],
        test_y[support_indices],
        test_labels[support_indices],
        selector_fit,
        selector_validation,
    )

    selected_q_by_label: dict[Any, bool] = {}
    selector_rows: list[dict[str, Any]] = []
    for label in pd.unique(q_selector.eval_labels):
        selected = q_selector.eval_labels == label
        q_mae = float(
            np.mean(np.abs(q_selector.eval_predictions[selected] - q_selector.eval_targets[selected]))
        )
        knn_mae = float(
            np.mean(np.abs(knn_selector_prediction[selected] - q_selector.eval_targets[selected]))
        )
        selected_q_by_label[label] = q_mae < knn_mae
        selector_rows.append(
            {
                "label": label,
                "selector_fit_rows": int(np.sum(test_labels[support_indices][selector_fit] == label)),
                "selector_validation_rows": int(selected.sum()),
                "q_selector_mae": q_mae,
                "knn_selector_mae": knn_mae,
                "selected_q": int(selected_q_by_label[label]),
            }
        )

    selector_prediction = np.asarray(
        [
            q_prediction[index]
            if selected_q_by_label[label]
            else knn_prediction[index]
            for index, label in enumerate(query_labels)
        ]
    )
    oracle_q_by_label: dict[Any, bool] = {}
    for row in selector_rows:
        label = row["label"]
        selected = query_labels == label
        q_rmse = float(np.sqrt(np.mean((q_prediction[selected] - truth[selected]) ** 2)))
        knn_rmse = float(np.sqrt(np.mean((knn_prediction[selected] - truth[selected]) ** 2)))
        oracle_q_by_label[label] = q_rmse < knn_rmse
        row.update(
            {
                "q_query_rmse": q_rmse,
                "knn_query_rmse": knn_rmse,
                "oracle_selected_q": int(oracle_q_by_label[label]),
                "selector_matches_query_oracle": int(
                    selected_q_by_label[label] == oracle_q_by_label[label]
                ),
                "query_rmse_regret": (
                    (q_rmse if selected_q_by_label[label] else knn_rmse)
                    - min(q_rmse, knn_rmse)
                ),
            }
        )
    oracle_prediction = np.asarray(
        [
            q_prediction[index] if oracle_q_by_label[label] else knn_prediction[index]
            for index, label in enumerate(query_labels)
        ]
    )

    altered_test_y = test_y.copy()
    altered_test_y[query_indices] += np.float32(123.0 * max(float(np.std(train_y)), 1.0))
    altered_dataset = build_dataset_from_arrays(
        test_x, test_labels, altered_test_y, feature_names=feature_columns
    )
    altered_q = calibrate_latent_q_for_test_labels(altered_dataset, training, config)
    np.testing.assert_array_equal(altered_q.eval_indices, query_indices)
    altered_knn = real._run_support_knn(
        test_x, altered_test_y, test_labels, support_indices, query_indices
    )
    altered_selector = np.asarray(
        [
            altered_q.eval_predictions[index]
            if selected_q_by_label[label]
            else altered_knn[index]
            for index, label in enumerate(query_labels)
        ]
    )
    leakage_difference = float(
        max(
            np.max(np.abs(q_prediction - altered_q.eval_predictions)),
            np.max(np.abs(knn_prediction - altered_knn)),
            np.max(np.abs(selector_prediction - altered_selector)),
        )
    )
    assert leakage_difference <= 1e-7

    query_frame = test_frame.iloc[query_indices].copy()
    query_frame["q_prediction"] = q_prediction
    query_frame["knn_prediction"] = knn_prediction
    query_frame["selector_prediction"] = selector_prediction
    query_frame["selected_component"] = [
        "latent_q" if selected_q_by_label[label] else "support_knn"
        for label in query_labels
    ]
    query_path = run_dir / "query_predictions.csv"
    query_frame.to_csv(query_path, index=False)
    selector_path = run_dir / "selector_diagnostics.csv"
    pd.DataFrame(selector_rows).to_csv(selector_path, index=False)

    q_columns = [column for column in q_result.test_output.columns if column.startswith("q")]
    q_by_label = q_result.test_output.groupby("label", sort=False)[q_columns].mean()
    q_mapping = {label: q_by_label.loc[label].to_numpy(float) for label in q_by_label.index}
    geometry, geometry_labels, geometry_q = _q_geometry(
        query_frame, feature_columns, q_mapping, args.seed
    )
    q_frame = pd.DataFrame({"label": geometry_labels})
    for index in range(geometry_q.shape[1]):
        q_frame[f"q{index + 1}"] = geometry_q[:, index]
    q_path = run_dir / "test_label_q.csv"
    q_frame.to_csv(q_path, index=False)
    checkpoint_path = run_dir / "training_checkpoint.pt"
    torch.save(
        {
            "job": job,
            "model_state_dict": training.model.state_dict(),
            "embedding_state_dict": training.embedding.state_dict(),
            "normalizer": asdict(training.normalizer),
        },
        checkpoint_path,
    )

    payload = {
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "dataset": {
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "support_rows": len(support_indices),
            "query_rows": len(query_indices),
            "test_labels": int(pd.Series(test_labels).nunique()),
        },
        "metrics": {
            "selector": _metrics(truth, selector_prediction, query_labels, train_y),
            "latent_q": _metrics(truth, q_prediction, query_labels, train_y),
            "support_knn": _metrics(truth, knn_prediction, query_labels, train_y),
            "query_oracle_non_deployable": _metrics(
                truth, oracle_prediction, query_labels, train_y
            ),
        },
        "selector": {
            "selected_q_fraction": float(np.mean(list(selected_q_by_label.values()))),
            "query_oracle_q_fraction": float(np.mean(list(oracle_q_by_label.values()))),
            "query_oracle_agreement": float(
                np.mean(
                    [selected_q_by_label[label] == oracle_q_by_label[label] for label in selected_q_by_label]
                )
            ),
            "median_query_rmse_regret": float(
                np.median([row["query_rmse_regret"] for row in selector_rows])
            ),
        },
        "q_geometry": geometry,
        "optimization_counters": asdict(training.optimization_counters),
        "query_leakage_probe_max_abs_difference": leakage_difference,
        "wall_time_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None,
        },
        "artifacts": {
            "query_predictions": str(query_path),
            "selector_diagnostics": str(selector_path),
            "test_label_q": str(q_path),
            "training_checkpoint": str(checkpoint_path),
        },
    }
    _write_json_atomic(result_path, payload)
    return result_path


def _task(dataset: str, summary: Path, seed: int, root: Path, args: argparse.Namespace) -> Task:
    command = (
        str(PYTHON),
        str(Path(__file__).resolve()),
        "run-job",
        "--prepared-summary",
        str(summary),
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
        "--output-root",
        str(root),
        "--q-dim",
        str(args.q_dim),
        "--epochs",
        str(args.epochs),
        "--cal-steps",
        str(args.cal_steps),
        "--cal-num-starts",
        str(args.cal_num_starts),
        "--cal-selection-ratio",
        str(args.cal_selection_ratio),
        "--cal-selection-min-rows",
        str(args.cal_selection_min_rows),
        "--cal-refine-steps",
        str(args.cal_refine_steps),
        "--support-ratio",
        str(args.support_ratio),
        "--selector-fit-ratio",
        str(args.selector_fit_ratio),
        "--batch-size",
        str(args.batch_size),
        "--hidden-sizes",
        args.hidden_sizes,
        "--max-train-per-label",
        str(args.max_train_per_label),
        "--max-test-per-label",
        str(args.max_test_per_label),
        "--subsample-seed",
        str(args.subsample_seed),
        "--resume",
    )
    return Task(
        f"q_knn_selector_{_stable_hash([dataset, seed, args.q_dim, args.epochs])}",
        dataset,
        seed,
        command,
    )


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
        for index, used in (
            line.split(",", maxsplit=1) for line in result.stdout.splitlines()
        )
    }


def _append_status(
    path: Path,
    task: Task,
    returncode: int,
    elapsed: float,
    gpu: str,
    timed_out: bool,
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "task_id": task.task_id,
                    "dataset": task.dataset,
                    "seed": task.seed,
                    "gpu": gpu,
                    "elapsed_seconds": elapsed,
                    "returncode": returncode,
                    "timed_out": timed_out,
                }
            )
            + "\n"
        )


def _bh_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array)
    ranked = array[order] * len(array) / np.arange(1, len(array) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(array)
    output[order] = np.minimum(ranked, 1.0)
    return output.tolist()


def _aggregate(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in root.glob("results/*/seed*/result.json"):
        payload = json.loads(path.read_text())
        query_frame = pd.read_csv(payload["artifacts"]["query_predictions"])
        q_frame = pd.read_csv(payload["artifacts"]["test_label_q"])
        feature_columns = list(
            real._load_record(
                Path(payload["job"]["prepared_summary"]), payload["job"]["dataset"]
            )["feature_columns"]
        )
        q_columns = [column for column in q_frame.columns if column.startswith("q")]
        q_mapping = {
            row.label: np.asarray([getattr(row, column) for column in q_columns])
            for row in q_frame.itertuples(index=False)
        }
        geometry, _, _ = _q_geometry(
            query_frame, feature_columns, q_mapping, payload["job"]["seed"]
        )
        for method, metrics in payload["metrics"].items():
            rows.append(
                {
                    "dataset": payload["job"]["dataset"],
                    "seed": payload["job"]["seed"],
                    "method": method,
                    **metrics,
                    "selected_q_fraction": payload["selector"]["selected_q_fraction"],
                    "query_oracle_agreement": payload["selector"]["query_oracle_agreement"],
                    "response_continuity_auc": geometry["response_continuity_auc"],
                    "response_trustworthiness_auc": geometry["response_trustworthiness_auc"],
                    "response_local_log_distortion_p95": geometry["response_local_log_distortion_p95"],
                    "effective_rank": geometry["effective_rank"],
                    "wall_time_seconds": payload["wall_time_seconds"],
                    "result_path": str(path),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "method", "seed"]).reset_index(drop=True)


def _seed_stability(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, _ in DATASETS:
        entries: list[tuple[int, Path]] = []
        for path in (root / "results" / dataset).glob("seed*/result.json"):
            payload = json.loads(path.read_text())
            entries.append((int(payload["job"]["seed"]), Path(payload["artifacts"]["test_label_q"])))
        for (left_seed, left_path), (right_seed, right_path) in combinations(sorted(entries), 2):
            left = pd.read_csv(left_path)
            right = pd.read_csv(right_path)
            q_columns = [column for column in left.columns if column.startswith("q")]
            merged = left.merge(right, on="label", suffixes=("_left", "_right"), validate="one_to_one")
            left_values = merged[[f"{column}_left" for column in q_columns]].to_numpy(float)
            right_values = merged[[f"{column}_right" for column in q_columns]].to_numpy(float)
            aligned = fit_procrustes_alignment(left_values, right_values).transform(left_values)
            scale = max(
                float(np.sqrt(np.mean((right_values - right_values.mean(axis=0)) ** 2))),
                1e-12,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "procrustes_seed_nrmse": float(
                        np.sqrt(np.mean((aligned - right_values) ** 2)) / scale
                    ),
                    "aligned_seed_knn_overlap": knn_overlap(
                        aligned, right_values, k=min(5, len(merged) - 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(root: Path) -> None:
    frame = _aggregate(root)
    assert len(frame) == 160
    assert frame.groupby(["dataset", "method"]).size().eq(10).all()
    assert np.isfinite(frame["reference_nrmse"]).all()
    planned = [json.loads(line) for line in (root / "planned_tasks.jsonl").read_text().splitlines() if line]
    ledger = [json.loads(line) for line in (root / "task_status.jsonl").read_text().splitlines() if line]
    assert len(planned) == len({row["task_id"] for row in planned}) == 40
    assert len(ledger) == len({row["task_id"] for row in ledger}) == 40
    assert all(row["returncode"] == 0 and not row["timed_out"] for row in ledger)
    query_rows: list[int] = []
    leakage: list[float] = []
    for path in root.glob("results/*/seed*/result.json"):
        payload = json.loads(path.read_text())
        query_rows.append(len(pd.read_csv(payload["artifacts"]["query_predictions"])))
        assert query_rows[-1] == payload["dataset"]["query_rows"]
        leakage.append(payload["query_leakage_probe_max_abs_difference"])
    assert len(query_rows) == 40 and max(leakage) <= 1e-7
    frame.to_csv(root / "all_method_rows.csv", index=False)
    summary = (
        frame.groupby(["dataset", "method"], as_index=False)
        .agg(
            runs=("reference_nrmse", "size"),
            median_nrmse=("reference_nrmse", "median"),
            p90_nrmse=("reference_nrmse", lambda values: values.quantile(0.9)),
            max_nrmse=("reference_nrmse", "max"),
            median_label_p95=("label_reference_nrmse_p95", "median"),
            catastrophic_runs=("reference_nrmse", lambda values: int((values > 1).sum())),
            median_selected_q_fraction=("selected_q_fraction", "median"),
            median_oracle_agreement=("query_oracle_agreement", "median"),
            median_wall_seconds=("wall_time_seconds", "median"),
        )
    )
    summary.to_csv(root / "dataset_method_summary.csv", index=False)

    effects: list[dict[str, Any]] = []
    for dataset in [item[0] for item in DATASETS] + ["__pooled__"]:
        selected = frame if dataset == "__pooled__" else frame[frame.dataset == dataset]
        candidate = selected[selected.method == "selector"]
        for anchor in ("latent_q", "support_knn"):
            reference = selected[selected.method == anchor]
            paired = candidate[["dataset", "seed", "reference_nrmse"]].merge(
                reference[["dataset", "seed", "reference_nrmse"]],
                on=["dataset", "seed"],
                suffixes=("_selector", "_anchor"),
                validate="one_to_one",
            )
            delta = paired.reference_nrmse_selector - paired.reference_nrmse_anchor
            p_value = 1.0 if np.allclose(delta, 0.0) else float(wilcoxon(delta).pvalue)
            effects.append(
                {
                    "dataset": dataset,
                    "anchor": anchor,
                    "pairs": len(paired),
                    "wins": int((delta < 0).sum()),
                    "ties": int(np.isclose(delta, 0).sum()),
                    "median_selector": float(paired.reference_nrmse_selector.median()),
                    "median_anchor": float(paired.reference_nrmse_anchor.median()),
                    "median_delta": float(delta.median()),
                    "wilcoxon_p": p_value,
                }
            )
    effects_frame = pd.DataFrame(effects)
    effects_frame["wilcoxon_bh_q"] = _bh_adjust(effects_frame.wilcoxon_p.tolist())
    effects_frame.to_csv(root / "paired_effects.csv", index=False)
    stability = _seed_stability(root)
    stability.to_csv(root / "q_seed_stability.csv", index=False)

    lookup = summary.set_index(["dataset", "method"])
    nasa_better = min(
        lookup.loc[("nasa_battery_capacity", "latent_q"), "median_nrmse"],
        lookup.loc[("nasa_battery_capacity", "support_knn"), "median_nrmse"],
    )
    gates: dict[str, bool] = {
        "integrity": len(ledger) == 40 and max(leakage) <= 1e-7,
        "nasa_within_5pct_of_better_component": bool(
            lookup.loc[("nasa_battery_capacity", "selector"), "median_nrmse"]
            <= 1.05 * nasa_better
        ),
    }
    for dataset, _ in DATASETS[1:]:
        gates[f"{dataset}_zero_catastrophic"] = bool(
            lookup.loc[(dataset, "selector"), "catastrophic_runs"] == 0
        )
        gates[f"{dataset}_within_10pct_knn"] = bool(
            lookup.loc[(dataset, "selector"), "median_nrmse"]
            <= 1.10 * lookup.loc[(dataset, "support_knn"), "median_nrmse"]
        )
    pooled = frame.groupby("method").reference_nrmse.median()
    gates["pooled_not_worse_than_better_component"] = bool(
        pooled["selector"] <= min(pooled["latent_q"], pooled["support_knn"])
    )
    gates["advance"] = all(gates.values())
    _write_json_atomic(root / "advancement_gates.json", gates)
    audit = {
        "planned_tasks": len(planned),
        "ledger_rows": len(ledger),
        "unique_task_ids": len({row["task_id"] for row in ledger}),
        "successful_tasks": sum(row["returncode"] == 0 for row in ledger),
        "failed_tasks": sum(row["returncode"] != 0 for row in ledger),
        "timed_out_tasks": sum(row["timed_out"] for row in ledger),
        "atomic_results": len(query_rows),
        "finite_method_rows": int(np.isfinite(frame.reference_nrmse).sum()),
        "query_rows_by_job": sorted(set(query_rows)),
        "max_query_leakage_probe_difference": max(leakage),
        "accounted_task_hours": sum(row["elapsed_seconds"] for row in ledger) / 3600,
        "advancement_gates": gates,
    }
    _write_json_atomic(root / "terminal_audit.json", audit)

    report = [
        "# q–kNN reliability-selector confirmation",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + terminal analysis",
        f"- Origin Date: {datetime.now().astimezone().date().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: q_knn_selector_v1",
        "",
        "## Dataset results",
        "",
        "| dataset | method | runs | median NRMSE | p90 | max | catastrophic | selected-q fraction | oracle agreement | median seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report.append(
            f"| {row.dataset} | {row.method} | {row.runs} | {row.median_nrmse:.6g} | {row.p90_nrmse:.6g} | {row.max_nrmse:.6g} | {row.catastrophic_runs} | {row.median_selected_q_fraction:.3f} | {row.median_oracle_agreement:.3f} | {row.median_wall_seconds:.1f} |"
        )
    report.extend(
        [
            "",
            "## Paired selector effects",
            "",
            "| dataset | anchor | wins | ties | pairs | median selector | median anchor | median delta | p | BH q |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in effects_frame.itertuples(index=False):
        report.append(
            f"| {row.dataset} | {row.anchor} | {row.wins} | {row.ties} | {row.pairs} | {row.median_selector:.6g} | {row.median_anchor:.6g} | {row.median_delta:.6g} | {row.wilcoxon_p:.4g} | {row.wilcoxon_bh_q:.4g} |"
        )
    report.extend(["", "## Frozen gates", ""])
    for name, passed in gates.items():
        report.append(f"- {name}: **{'PASS' if passed else 'FAIL'}**")
    report.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Terminal ledger: {audit['successful_tasks']}/40 successful, {audit['failed_tasks']} failed, {audit['timed_out_tasks']} timed out.",
            f"- Maximum query-target perturbation effect: {audit['max_query_leakage_probe_difference']:.3g}.",
            f"- Accounted task time: {audit['accounted_task_hours']:.2f} hours.",
            f"- Final advancement decision: **{'ADVANCE' if gates['advance'] else 'DO NOT ADVANCE'}**.",
            "",
            "The query-oracle row is a non-deployable diagnostic only and must not be presented as a usable method.",
        ]
    )
    (root / "Q_KNN_SELECTOR_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_status(
    root: Path,
    started_at: str,
    tasks: list[Task],
    completed: set[str],
    pending: list[Task],
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]],
    memory: dict[str, int],
) -> None:
    ledger_path = root / "task_status.jsonl"
    ledger = (
        [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
        if ledger_path.exists()
        else []
    )
    _write_json_atomic(
        root / "campaign_status.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "running",
            "started_at": started_at,
            "planned": len(tasks),
            "completed": len(completed),
            "failed": sum(row["returncode"] != 0 for row in ledger),
            "pending": len(pending),
            "running": {
                gpu: {"task_id": task.task_id, "dataset": task.dataset, "seed": task.seed}
                for gpu, (_, task, _, _) in running.items()
            },
            "gpu_memory_mib": memory,
        },
    )


def launch(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    if len(gpus) != len(set(gpus)) or not gpus:
        raise ValueError("--gpus must list at least one unique physical GPU")
    tasks = [
        _task(dataset, summary, seed, root, args)
        for seed in seeds
        for dataset, summary in DATASETS
    ]
    if args.dry_run:
        print(json.dumps({"planned": len(tasks), "datasets": [x[0] for x in DATASETS], "seeds": seeds, "gpus": gpus}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    _write_json_atomic(
        root / "campaign_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_plan": str(PLAN_PATH),
            "planned_tasks": len(tasks),
            "datasets": [item[0] for item in DATASETS],
            "seeds": seeds,
            "gpus": gpus,
            "q_dim": args.q_dim,
            "epochs": args.epochs,
            "support_ratio": args.support_ratio,
            "selector_fit_ratio": args.selector_fit_ratio,
            "gpu_memory_available_threshold_mib": GPU_MEMORY_THRESHOLD_MIB,
            "single_job_timeout_minutes": args.single_job_timeout_minutes,
        },
    )
    with (root / "planned_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task)) + "\n")
    ledger_path = root / "task_status.jsonl"
    completed = _successful_ids(ledger_path)
    pending = [task for task in tasks if task.task_id not in completed]
    running: dict[str, tuple[subprocess.Popen[Any], Task, Any, float]] = {}
    started_at = datetime.now(timezone.utc).isoformat()
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
                task.command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
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
            _append_status(
                ledger_path, task, returncode, now - task_started, gpu, timed_out
            )
            if returncode == 0:
                completed.add(task.task_id)
            finished.append(gpu)
        for gpu in finished:
            del running[gpu]
        _write_status(root, started_at, tasks, completed, pending, running, memory)
        if pending or running:
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
    if ledger_path.exists():
        rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
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
        print(run_job(args))
    elif args.command == "launch":
        launch(args)
    else:
        summarize(args.output_root)


if __name__ == "__main__":
    main()
