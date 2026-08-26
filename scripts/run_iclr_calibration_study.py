#!/usr/bin/env python3
"""Train once and compare held-out latent-q calibration strategies fairly."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

from lvs.backends.torch_mlp import build_torch_model_factory, parse_hidden_sizes
from lvs.core.metrics import (
    effective_rank,
    grouped_rff_signatures,
    local_distance_distortion,
    macro_prediction_metrics,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
    reference_scaled_prediction_metrics,
)
from lvs.core.pipeline import (
    LatentQConfig,
    build_dataset_from_arrays,
    evaluate_latent_q_pipeline,
    train_latent_q_model,
)

PYTHON = PROJECT_ROOT / ".venv-lvs-gpu" / "bin" / "python"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_calibration_strategy_pilot_20260809"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationStrategy:
    name: str
    init_mode: str
    num_starts: int
    steps: int
    selection_ratio: float = 0.0
    refine_steps: int = 0
    selection_min_rows: int = 2
    refine_only_after_selection: bool = False

    @property
    def step_budget(self) -> int:
        return self.num_starts * self.steps + self.refine_steps


SCREENING_STRATEGIES = (
    CalibrationStrategy("legacy_k1_s200", "legacy_random", 1, 200),
    CalibrationStrategy("zero_k1_s200", "zero", 1, 200),
    CalibrationStrategy("mean_k1_s200", "train_mean", 1, 200),
    CalibrationStrategy("prior_k1_s200", "prior_random", 1, 200),
    CalibrationStrategy("legacy_k8_matched", "legacy_random", 8, 20, 0.25, 40),
    CalibrationStrategy("prior_k4_matched", "prior_random", 4, 40, 0.25, 40),
    CalibrationStrategy("prior_k8_matched", "prior_random", 8, 20, 0.25, 40),
    CalibrationStrategy("prior_k8_full", "prior_random", 8, 50, 0.25, 50),
)

LONGSTART_STRATEGIES = (
    CalibrationStrategy("legacy_k1_s200", "legacy_random", 1, 200),
    CalibrationStrategy("prior_k1_s200", "prior_random", 1, 200),
    CalibrationStrategy("prior_k4_s200_inner_r50", "prior_random", 4, 200, 0.25, 50),
    CalibrationStrategy("prior_k8_s200_inner_r50", "prior_random", 8, 200, 0.25, 50),
    CalibrationStrategy("prior_k8_s200_support", "prior_random", 8, 200),
)

ADAPTIVE_STRATEGIES = (
    CalibrationStrategy("legacy_k1_s200", "legacy_random", 1, 200),
    CalibrationStrategy("prior_k1_s200", "prior_random", 1, 200),
    CalibrationStrategy(
        "prior_k4_s200_adaptive24_r50", "prior_random", 4, 200, 0.25, 50, 24, True
    ),
    CalibrationStrategy(
        "prior_k8_s200_adaptive24_r50", "prior_random", 8, 200, 0.25, 50, 24, True
    ),
)

THRESHOLD_STRATEGIES = (
    CalibrationStrategy("legacy_k1_s200", "legacy_random", 1, 200),
    CalibrationStrategy(
        "prior_k4_s200_min2_r50", "prior_random", 4, 200, 0.25, 50, 2, True
    ),
    CalibrationStrategy(
        "prior_k4_s200_min20_r50", "prior_random", 4, 200, 0.25, 50, 20, True
    ),
    CalibrationStrategy(
        "prior_k4_s200_min24_r50", "prior_random", 4, 200, 0.25, 50, 24, True
    ),
    CalibrationStrategy(
        "prior_k4_s200_min32_r50", "prior_random", 4, 200, 0.25, 50, 32, True
    ),
    CalibrationStrategy(
        "prior_k4_s200_min48_r50", "prior_random", 4, 200, 0.25, 50, 48, True
    ),
)

CONFIRMATION_STRATEGIES = (
    CalibrationStrategy("legacy_k1_s200", "legacy_random", 1, 200),
    CalibrationStrategy(
        "prior_k4_s200_min24_r50", "prior_random", 4, 200, 0.25, 50, 24, True
    ),
    CalibrationStrategy(
        "prior_k4_s200_min32_r50", "prior_random", 4, 200, 0.25, 50, 32, True
    ),
)

STRATEGY_PROFILES = {
    "screening": SCREENING_STRATEGIES,
    "longstart": LONGSTART_STRATEGIES,
    "adaptive": ADAPTIVE_STRATEGIES,
    "thresholds": THRESHOLD_STRATEGIES,
    "confirmation": CONFIRMATION_STRATEGIES,
}
# Backward-compatible import used by existing analysis code.
STRATEGIES = SCREENING_STRATEGIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--prepared-summary", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--method", choices=("joint_mse", "alternating_mse"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--q-dim", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--prepared-summary", type=Path, action="append", required=True)
    launch.add_argument(
        "--tasks",
        required=True,
        help="Comma-separated dataset:q_dim pairs.",
    )
    launch.add_argument("--methods", default="joint_mse,alternating_mse")
    launch.add_argument("--seeds", default="0,1,2")
    launch.add_argument("--gpus", default="4,5,6,7")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_args(launch)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--max-train-per-label", type=int, default=256)
    parser.add_argument("--max-test-per-label", type=int, default=256)
    parser.add_argument("--subsample-seed", type=int, default=20260808)
    parser.add_argument(
        "--strategy-profile",
        choices=tuple(STRATEGY_PROFILES),
        default="screening",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_record(summary_path: Path, dataset_name: str) -> dict[str, Any]:
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [record for record in records if record["name"] == dataset_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {dataset_name!r} in {summary_path}.")
    return matches[0]


def _cap_rows_per_label(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame.reset_index(drop=True)
    pieces = []
    for index, (_, group) in enumerate(frame.groupby("label", sort=False)):
        if len(group) > maximum:
            group = group.sample(n=maximum, random_state=seed + index)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _task_pairs(raw: str) -> list[tuple[str, int]]:
    output = []
    for item in raw.split(","):
        name, separator, raw_q_dim = item.strip().rpartition(":")
        if not separator or not name or not raw_q_dim:
            raise ValueError(f"Invalid task {item!r}; expected dataset:q_dim.")
        q_dim = int(raw_q_dim)
        if q_dim <= 0:
            raise ValueError("q_dim must be positive.")
        output.append((name, q_dim))
    if not output:
        raise ValueError("At least one task is required.")
    return output


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    strategies = STRATEGY_PROFILES[args.strategy_profile]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "prepared_summary": str(args.prepared_summary),
        "method": args.method,
        "seed": args.seed,
        "q_dim": args.q_dim,
        "epochs": args.epochs,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "hidden_sizes": args.hidden_sizes,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
        "strategy_profile": args.strategy_profile,
        "strategies": [asdict(strategy) for strategy in strategies],
    }


def _base_config(args: argparse.Namespace) -> LatentQConfig:
    schedule = "alternating" if args.method == "alternating_mse" else "joint"
    return LatentQConfig(
        q_dim=args.q_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        calibration_steps=200,
        calibration_lr=0.05,
        calibration_ratio=args.support_ratio,
        calibration_split_mode="random",
        seed=args.seed,
        device=args.device,
        verbose=False,
        early_stop_enabled=False,
        optimization_schedule=schedule,
        joint_steps_per_cycle=2,
        theta_steps_per_cycle=1,
        q_steps_per_cycle=1,
    )


def _prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, float]:
    reference_scale = max(float(np.std(train_targets)), 1e-8)
    output = {
        **macro_prediction_metrics(truth, prediction, labels),
        **reference_scaled_prediction_metrics(truth, prediction, reference_scale=reference_scale),
    }
    per_label = []
    for label in pd.unique(labels):
        selected = labels == label
        per_label.append(float(np.sqrt(np.mean((truth[selected] - prediction[selected]) ** 2)) / reference_scale))
    output.update(
        {
            "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
            "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
            "label_reference_nrmse_max": float(np.max(per_label)),
        }
    )
    return output


def _spatial_metrics(
    *,
    result: Any,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    subsample_seed: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, list[dict[str, float]]]:
    q_columns = [column for column in result.test_output.columns if column.startswith("q")]
    q_by_label = result.test_output.groupby("label", sort=False)[q_columns].mean()
    query_frame = test_frame.iloc[result.eval_indices].reset_index(drop=True)
    q_labels, response_signatures = grouped_rff_signatures(
        np.column_stack(
            [
                query_frame[feature_columns].to_numpy(float),
                query_frame["target"].to_numpy(float),
            ]
        ),
        query_frame["label"].to_numpy(),
        n_components=64,
        seed=subsample_seed,
    )
    acquisition_labels, acquisition_signatures = grouped_rff_signatures(
        query_frame[feature_columns].to_numpy(float),
        query_frame["label"].to_numpy(),
        n_components=64,
        seed=subsample_seed + 1,
    )
    q_values = np.vstack([q_by_label.loc[label].to_numpy(float) for label in q_labels])
    acquisition_map = {
        label: acquisition_signatures[index] for index, label in enumerate(acquisition_labels)
    }
    acquisition_values = np.vstack([acquisition_map[label] for label in q_labels])
    curve = (
        neighborhood_preservation_curve(
            response_signatures,
            q_values,
            max_k=min(10, (len(q_labels) - 1) // 2),
        )
        if len(q_labels) >= 3
        else []
    )
    response_geometry = pairwise_distance_metrics(q_values, response_signatures)
    acquisition_geometry = pairwise_distance_metrics(q_values, acquisition_values)
    metrics = {
        "response_continuity_auc": float(np.mean([row["continuity"] for row in curve]))
        if curve
        else float("nan"),
        "response_trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in curve]))
        if curve
        else float("nan"),
        "response_knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve]))
        if curve
        else float("nan"),
        **{f"response_{key}": value for key, value in response_geometry.items()},
        **{
            f"response_{key}": value
            for key, value in local_distance_distortion(
                response_signatures, q_values, k=min(5, len(q_labels) - 1)
            ).items()
        },
        **{f"acquisition_{key}": value for key, value in acquisition_geometry.items()},
        "effective_rank": effective_rank(q_values),
    }
    return metrics, q_labels, q_values, curve


def _save_strategy_artifacts(
    *,
    run_dir: Path,
    strategy: str,
    result: Any,
    test_frame: pd.DataFrame,
    q_labels: np.ndarray,
    q_values: np.ndarray,
    curve: list[dict[str, float]],
) -> dict[str, str]:
    prediction_frame = test_frame.iloc[result.eval_indices].copy()
    prediction_frame["prediction"] = result.eval_predictions
    prediction_path = run_dir / f"query_predictions_{strategy}.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    q_frame = pd.DataFrame({"label": q_labels})
    for index in range(q_values.shape[1]):
        q_frame[f"q{index + 1}"] = q_values[:, index]
    q_path = run_dir / f"test_label_q_{strategy}.csv"
    curve_path = run_dir / f"continuity_curve_{strategy}.csv"
    q_frame.to_csv(q_path, index=False)
    pd.DataFrame(curve).to_csv(curve_path, index=False)
    return {
        "query_predictions": str(prediction_path),
        "test_label_q": str(q_path),
        "continuity_curve": str(curve_path),
    }


def run_job(args: argparse.Namespace) -> Path:
    record = _load_record(args.prepared_summary, args.dataset)
    feature_columns = list(record["feature_columns"])
    train_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["train_csv"])),
        args.max_train_per_label,
        args.subsample_seed,
    )
    test_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["test_csv"])),
        args.max_test_per_label,
        args.subsample_seed + 10000,
    )
    required = {"label", "target", *feature_columns}
    for name, frame in (("train", train_frame), ("test", test_frame)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data are missing columns: {sorted(missing)}")

    job = _job_config(args)
    run_dir = (
        args.output_root
        / args.dataset
        / args.method
        / f"seed{args.seed}_q{args.q_dim}_{stable_hash(job)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("status") == "success"
            and existing.get("job") == job
        ):
            return result_path

    train_x = train_frame[feature_columns].to_numpy(np.float32)
    train_labels = train_frame["label"].to_numpy()
    train_y = train_frame["target"].to_numpy(np.float32)
    test_x = test_frame[feature_columns].to_numpy(np.float32)
    test_labels = test_frame["label"].to_numpy()
    test_y = test_frame["target"].to_numpy(np.float32)
    train_dataset = build_dataset_from_arrays(
        train_x, train_labels, train_y, feature_names=feature_columns
    )
    test_dataset = build_dataset_from_arrays(
        test_x, test_labels, test_y, feature_names=feature_columns
    )
    config = _base_config(args)
    training_started = time.perf_counter()
    training = train_latent_q_model(
        train_dataset,
        build_torch_model_factory(parse_hidden_sizes(args.hidden_sizes)),
        config,
    )
    training_seconds = time.perf_counter() - training_started

    checkpoint_path = run_dir / "training_checkpoint.pt"
    if args.save_artifacts:
        torch.save(
            {
                "job": job,
                "config": asdict(config),
                "model_state_dict": training.model.state_dict(),
                "embedding_state_dict": training.embedding.state_dict(),
                "normalizer": asdict(training.normalizer),
                "label_to_index": training.label_to_index,
                "train_history": [asdict(row) for row in training.train_history],
                "optimization_counters": asdict(training.optimization_counters),
            },
            checkpoint_path,
        )

    strategy_payloads: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGY_PROFILES[args.strategy_profile]:
        strategy_config = replace(
            config,
            calibration_init_mode=strategy.init_mode,
            calibration_num_starts=strategy.num_starts,
            calibration_steps=strategy.steps,
            calibration_selection_ratio=strategy.selection_ratio,
            calibration_selection_min_rows=strategy.selection_min_rows,
            calibration_refine_steps=strategy.refine_steps,
            calibration_refine_only_after_selection=strategy.refine_only_after_selection,
        )
        started = time.perf_counter()
        result = evaluate_latent_q_pipeline(
            train_dataset, test_dataset, training, strategy_config
        )
        calibration_seconds = time.perf_counter() - started
        prediction = _prediction_metrics(
            result.eval_targets,
            result.eval_predictions,
            result.eval_labels,
            train_y,
        )
        spatial, q_labels, q_values, curve = _spatial_metrics(
            result=result,
            test_frame=test_frame,
            feature_columns=feature_columns,
            subsample_seed=args.subsample_seed,
        )
        artifacts = (
            _save_strategy_artifacts(
                run_dir=run_dir,
                strategy=strategy.name,
                result=result,
                test_frame=test_frame,
                q_labels=q_labels,
                q_values=q_values,
                curve=curve,
            )
            if args.save_artifacts
            else {}
        )
        strategy_payloads[strategy.name] = {
            "config": asdict(strategy),
            "step_budget": strategy.step_budget,
            "prediction": prediction,
            "spatial": spatial,
            "calibration": {
                key: value
                for key, value in result.metrics.items()
                if key.startswith("calibration_")
            },
            "calibration_seconds": calibration_seconds,
            "artifacts": artifacts,
        }

    payload = {
        "status": "success",
        "job": job,
        "dataset": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_labels": int(pd.Series(train_labels).nunique()),
            "test_labels": int(pd.Series(test_labels).nunique()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None,
        },
        "training_seconds": training_seconds,
        "training_checkpoint": str(checkpoint_path) if args.save_artifacts else None,
        "optimization_counters": asdict(training.optimization_counters),
        "strategies": strategy_payloads,
    }
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(result_path)
    return result_path


def summarize(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("*/*/seed*_q*/result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != "success":
            continue
        for strategy, values in payload["strategies"].items():
            rows.append(
                {
                    **{key: value for key, value in payload["job"].items() if key != "strategies"},
                    **payload["dataset"],
                    "strategy": strategy,
                    "step_budget": values["step_budget"],
                    **values["prediction"],
                    **values["spatial"],
                    **values["calibration"],
                    "training_seconds": payload["training_seconds"],
                    "calibration_seconds": values["calibration_seconds"],
                    "result_path": str(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "all_strategy_runs.csv", index=False)
    if frame.empty:
        return
    metrics = [
        column
        for column in (
            "reference_nrmse",
            "label_reference_nrmse_p90",
            "label_reference_nrmse_p95",
            "response_continuity_auc",
            "response_trustworthiness_auc",
            "response_local_log_distortion_p95",
            "calibration_candidate_q_dispersion_mean",
            "calibration_seconds",
        )
        if column in frame.columns
    ]
    summary = frame.groupby("strategy", as_index=False)[metrics].agg(["count", "mean", "std"])
    summary.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_root / "strategy_summary.csv", index=False)
    baseline = frame[frame["strategy"] == "legacy_k1_s200"]
    contrasts = []
    block_columns = ["dataset", "method", "seed", "q_dim"]
    for strategy in sorted(set(frame["strategy"]) - {"legacy_k1_s200"}):
        comparison = frame[frame["strategy"] == strategy]
        paired = baseline.merge(comparison, on=block_columns, suffixes=("_baseline", "_comparison"))
        for metric in ("reference_nrmse", "label_reference_nrmse_p95"):
            difference = paired[f"{metric}_comparison"] - paired[f"{metric}_baseline"]
            contrasts.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "paired_blocks": int(len(paired)),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "win_rate_lower_is_better": float((difference < 0).mean()),
                }
            )
    pd.DataFrame(contrasts).to_csv(output_root / "paired_strategy_contrasts.csv", index=False)


def launch(args: argparse.Namespace) -> None:
    methods = [value for value in args.methods.split(",") if value]
    invalid_methods = set(methods) - {"joint_mse", "alternating_mse"}
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {sorted(invalid_methods)}")
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    summaries: dict[str, Path] = {}
    for path in args.prepared_summary:
        for record in json.loads(path.read_text(encoding="utf-8")):
            summaries[record["name"]] = path
    jobs = []
    for dataset, q_dim in _task_pairs(args.tasks):
        if dataset not in summaries:
            raise ValueError(f"Dataset {dataset!r} is absent from prepared summaries.")
        for method in methods:
            for seed in seeds:
                jobs.append((summaries[dataset], dataset, q_dim, method, seed))

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "protocol": "one fitted decoder per block; strategy-only calibration contrasts",
        "prepared_summaries": [str(path) for path in args.prepared_summary],
        "tasks": _task_pairs(args.tasks),
        "methods": methods,
        "seeds": seeds,
        "gpus": gpus,
        "epochs": args.epochs,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "hidden_sizes": args.hidden_sizes,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
        "strategy_profile": args.strategy_profile,
        "strategies": [
            asdict(strategy) | {"step_budget": strategy.step_budget}
            for strategy in STRATEGY_PROFILES[args.strategy_profile]
        ],
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pending = list(jobs)
    running: list[tuple[subprocess.Popen[Any], tuple[Any, ...], Any, str]] = []
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            job = pending.pop(0)
            summary_path, dataset, q_dim, method, seed = job
            available = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            if not available:
                break
            gpu = available[0]
            command = [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "run-job",
                "--prepared-summary", str(summary_path),
                "--dataset", dataset,
                "--method", method,
                "--seed", str(seed),
                "--q-dim", str(q_dim),
                "--device", "cuda:0",
                "--output-root", str(args.output_root),
                "--epochs", str(args.epochs),
                "--support-ratio", str(args.support_ratio),
                "--batch-size", str(args.batch_size),
                "--hidden-sizes", args.hidden_sizes,
                "--max-train-per-label", str(args.max_train_per_label),
                "--max-test-per-label", str(args.max_test_per_label),
                "--subsample-seed", str(args.subsample_seed),
                "--strategy-profile", args.strategy_profile,
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update(
                {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            )
            log_path = args.output_root / "logs" / f"{dataset}_{method}_seed{seed}_q{q_dim}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((process, job, handle, gpu))
        next_running = []
        for process, job, handle, gpu in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((process, job, handle, gpu))
                continue
            handle.close()
            with status_path.open("a", encoding="utf-8") as status:
                status.write(
                    json.dumps(
                        {
                            "time": datetime.now(timezone.utc).isoformat(),
                            "job": [str(job[0]), *job[1:]],
                            "returncode": return_code,
                        }
                    )
                    + "\n"
                )
        running = next_running
        time.sleep(2)
    summarize(args.output_root)
    (args.output_root / "controller_status.json").write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "jobs": len(jobs),
            },
            indent=2,
        ),
        encoding="utf-8",
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
