#!/usr/bin/env python3
"""Synthetic true-q validation for held-out calibration strategies."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
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

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.expression_library import (
    load_expression_library,
    sample_expression_dataset,
    select_expression_task,
)
from lvs.core.metrics import (
    alignment_metrics,
    apply_affine_alignment,
    effective_rank,
    fit_affine_alignment,
    fit_cca_alignment,
    knn_overlap,
    local_distance_distortion,
    macro_prediction_metrics,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
    reference_scaled_prediction_metrics,
    score_cca_alignment,
)
from lvs.core.pipeline import (
    LatentQConfig,
    build_dataset_from_arrays,
    evaluate_latent_q_pipeline,
    train_latent_q_model,
)
from scripts.run_iclr_calibration_study import STRATEGY_PROFILES, stable_hash

PYTHON = Path(sys.executable)
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_synthetic_calibration_strategy_20260809"
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--expression-id", type=int, required=True)
    run.add_argument("--method", choices=("joint_mse", "alternating_mse"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--data-seed", type=int, default=20260808)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--expression-ids", default="3,41,48")
    launch.add_argument("--methods", default="joint_mse,alternating_mse")
    launch.add_argument("--seeds", default="0,1,2")
    launch.add_argument("--data-seed", type=int, default=20260808)
    launch.add_argument("--gpus", default="4,5,6,7")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_args(launch)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train-labels", type=int, default=32)
    parser.add_argument("--validation-labels", type=int, default=16)
    parser.add_argument("--test-labels", type=int, default=32)
    parser.add_argument("--samples-per-label", type=int, default=60)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--strategy-profile",
        choices=tuple(STRATEGY_PROFILES),
        default="screening",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)


def _job_config(args: argparse.Namespace, q_dim: int) -> dict[str, Any]:
    strategies = STRATEGY_PROFILES[args.strategy_profile]
    return {
        "schema_version": SCHEMA_VERSION,
        "expression_id": args.expression_id,
        "method": args.method,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "q_dim": q_dim,
        "epochs": args.epochs,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "samples_per_label": args.samples_per_label,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "strategy_profile": args.strategy_profile,
        "strategies": [asdict(strategy) for strategy in strategies],
    }


def _base_config(args: argparse.Namespace, q_dim: int) -> LatentQConfig:
    schedule = "alternating" if args.method == "alternating_mse" else "joint"
    return LatentQConfig(
        q_dim=q_dim,
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


def _arrays(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame.loc[:, features].to_numpy(np.float32),
        frame["label"].to_numpy(),
        frame["target"].to_numpy(np.float32),
    )


def _learned_truth_by_label(
    result: Any,
    truth: pd.DataFrame,
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    q_columns = [column for column in result.test_output.columns if column.startswith("q")]
    latent_columns = [column for column in truth.columns if column.startswith("q")]
    learned_columns = [f"learned_{column}" for column in q_columns]
    learned = result.test_output.groupby("label", sort=False)[q_columns].mean().reset_index()
    learned = learned.rename(columns=dict(zip(q_columns, learned_columns)))
    reference = truth[truth["split"] == split][["label", *latent_columns]]
    merged = learned.merge(reference, on="label", how="inner")
    return (
        merged[learned_columns].to_numpy(float),
        merged[latent_columns].to_numpy(float),
        merged,
    )


def _prediction_metrics(result: Any, train_y: np.ndarray) -> dict[str, float]:
    reference_scale = max(float(np.std(train_y)), 1e-8)
    output = {
        **macro_prediction_metrics(result.eval_targets, result.eval_predictions, result.eval_labels),
        **reference_scaled_prediction_metrics(
            result.eval_targets,
            result.eval_predictions,
            reference_scale=reference_scale,
        ),
    }
    per_label = []
    for label in pd.unique(result.eval_labels):
        selected = result.eval_labels == label
        per_label.append(
            float(
                np.sqrt(
                    np.mean(
                        (result.eval_targets[selected] - result.eval_predictions[selected]) ** 2
                    )
                )
                / reference_scale
            )
        )
    output.update(
        {
            "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
            "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
            "label_reference_nrmse_max": float(np.max(per_label)),
        }
    )
    return output


def _true_q_metrics(
    *,
    validation_learned: np.ndarray,
    validation_truth: np.ndarray,
    test_learned: np.ndarray,
    test_truth: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    alignment = fit_affine_alignment(validation_learned, validation_truth)
    aligned_test = apply_affine_alignment(alignment, test_learned)
    cca_alignment = fit_cca_alignment(validation_learned, validation_truth)
    cca = score_cca_alignment(cca_alignment, test_learned, test_truth)
    curve = neighborhood_preservation_curve(
        test_truth,
        test_learned,
        max_k=min(10, (len(test_truth) - 1) // 2),
    )
    metrics = {
        **alignment_metrics(test_truth, aligned_test),
        **pairwise_distance_metrics(test_learned, test_truth),
        "trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in curve])),
        "continuity_auc": float(np.mean([row["continuity"] for row in curve])),
        "knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve])),
        **local_distance_distortion(
            test_truth, test_learned, k=min(5, len(test_truth) - 1)
        ),
        "cca_mean": float(cca.mean()) if cca.size else float("nan"),
        "knn_overlap": knn_overlap(
            test_learned, test_truth, k=min(3, len(test_truth) - 1)
        ),
        "effective_rank": effective_rank(test_learned),
    }
    return metrics, aligned_test, curve


def _save_artifacts(
    *,
    run_dir: Path,
    strategy: str,
    testing: Any,
    test_frame: pd.DataFrame,
    validation_q: pd.DataFrame,
    test_q: pd.DataFrame,
    aligned_test: np.ndarray,
    curve: list[dict[str, float]],
) -> dict[str, str]:
    prediction_frame = test_frame.iloc[testing.eval_indices].copy()
    prediction_frame["prediction"] = testing.eval_predictions
    prediction_path = run_dir / f"query_predictions_{strategy}.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    for index in range(aligned_test.shape[1]):
        test_q[f"aligned_q{index + 1}"] = aligned_test[:, index]
    validation_path = run_dir / f"validation_label_q_{strategy}.csv"
    test_path = run_dir / f"test_label_q_{strategy}.csv"
    curve_path = run_dir / f"continuity_curve_{strategy}.csv"
    validation_q.to_csv(validation_path, index=False)
    test_q.to_csv(test_path, index=False)
    pd.DataFrame(curve).to_csv(curve_path, index=False)
    return {
        "query_predictions": str(prediction_path),
        "validation_label_q": str(validation_path),
        "test_label_q": str(test_path),
        "continuity_curve": str(curve_path),
    }


def run_job(args: argparse.Namespace) -> Path:
    if args.validation_labels < 2 or args.test_labels < 2:
        raise ValueError("Validation and test label counts must both be at least two.")
    task = select_expression_task(
        load_expression_library(PROJECT_ROOT / "data" / "latent_variable_expressions.csv"),
        expression_id=args.expression_id,
    )
    q_dim = task.ground_truth_latent_dim
    job = _job_config(args, q_dim)
    run_dir = (
        args.output_root
        / f"expr{args.expression_id:03d}"
        / args.method
        / f"seed{args.seed}_q{q_dim}_{stable_hash(job)}"
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

    generated = sample_expression_dataset(
        task,
        label_count=args.train_labels,
        validation_label_count=args.validation_labels,
        test_label_count=args.test_labels,
        train_samples_per_label=args.samples_per_label,
        validation_samples_per_label=args.samples_per_label,
        test_samples_per_label=args.samples_per_label,
        label_split_mode="disjoint",
        seed=args.data_seed,
    )
    assert generated.validation_frame is not None
    features = list(task.feature_columns)
    train_x, train_labels, train_y = _arrays(generated.train_frame, features)
    val_x, val_labels, val_y = _arrays(generated.validation_frame, features)
    test_x, test_labels, test_y = _arrays(generated.test_frame, features)
    train_dataset = build_dataset_from_arrays(
        train_x, train_labels, train_y, feature_names=features
    )
    validation_dataset = build_dataset_from_arrays(
        val_x, val_labels, val_y, feature_names=features
    )
    test_dataset = build_dataset_from_arrays(
        test_x, test_labels, test_y, feature_names=features
    )
    config = _base_config(args, q_dim)
    started = time.perf_counter()
    training = train_latent_q_model(
        train_dataset, build_torch_model_factory((128, 64)), config
    )
    training_seconds = time.perf_counter() - started
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

    truth = generated.latent_truth_frame
    strategy_payloads = {}
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
        validation = evaluate_latent_q_pipeline(
            train_dataset, validation_dataset, training, strategy_config
        )
        testing = evaluate_latent_q_pipeline(
            train_dataset, test_dataset, training, strategy_config
        )
        calibration_seconds = time.perf_counter() - started
        val_learned, val_truth, val_q = _learned_truth_by_label(
            validation, truth, "validation"
        )
        test_learned, test_truth, test_q = _learned_truth_by_label(
            testing, truth, "test"
        )
        spatial, aligned_test, curve = _true_q_metrics(
            validation_learned=val_learned,
            validation_truth=val_truth,
            test_learned=test_learned,
            test_truth=test_truth,
        )
        artifacts = (
            _save_artifacts(
                run_dir=run_dir,
                strategy=strategy.name,
                testing=testing,
                test_frame=generated.test_frame,
                validation_q=val_q,
                test_q=test_q,
                aligned_test=aligned_test,
                curve=curve,
            )
            if args.save_artifacts
            else {}
        )
        strategy_payloads[strategy.name] = {
            "config": asdict(strategy),
            "step_budget": strategy.step_budget,
            "prediction": _prediction_metrics(testing, train_y),
            "spatial": spatial,
            "calibration": {
                key: value
                for key, value in testing.metrics.items()
                if key.startswith("calibration_")
            },
            "calibration_seconds": calibration_seconds,
            "artifacts": artifacts,
        }

    payload = {
        "status": "success",
        "job": job,
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
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def summarize(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("expr*/*/seed*_q*/result.json"):
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
            "label_reference_nrmse_p95",
            "aligned_nrmse",
            "cca_mean",
            "continuity_auc",
            "trustworthiness_auc",
            "local_log_distortion_p95",
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
    keys = ["expression_id", "method", "seed", "q_dim"]
    baseline = frame[frame["strategy"] == "legacy_k1_s200"]
    contrasts = []
    for strategy in sorted(set(frame["strategy"]) - {"legacy_k1_s200"}):
        paired = baseline.merge(
            frame[frame["strategy"] == strategy], on=keys, suffixes=("_baseline", "_comparison")
        )
        for metric in ("reference_nrmse", "aligned_nrmse", "continuity_auc"):
            difference = paired[f"{metric}_comparison"] - paired[f"{metric}_baseline"]
            lower_is_better = metric != "continuity_auc"
            contrasts.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "paired_blocks": int(len(paired)),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "win_rate": float((difference < 0).mean())
                    if lower_is_better
                    else float((difference > 0).mean()),
                    "direction": "lower" if lower_is_better else "higher",
                }
            )
    pd.DataFrame(contrasts).to_csv(output_root / "paired_strategy_contrasts.csv", index=False)


def launch(args: argparse.Namespace) -> None:
    expression_ids = [int(value) for value in args.expression_ids.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    jobs = [
        (expression_id, method, seed)
        for expression_id in expression_ids
        for method in methods
        for seed in seeds
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "protocol": "one fitted decoder per block; validation-fit true-q alignment; strategy-only calibration contrasts",
        "expression_ids": expression_ids,
        "methods": methods,
        "seeds": seeds,
        "data_seed": args.data_seed,
        "gpus": gpus,
        "epochs": args.epochs,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "samples_per_label": args.samples_per_label,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
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
    running: list[tuple[subprocess.Popen[Any], tuple[int, str, int], Any, str]] = []
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            job = pending.pop(0)
            expression_id, method, seed = job
            available = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            gpu = available[0]
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "run-job",
                "--expression-id", str(expression_id),
                "--method", method,
                "--seed", str(seed),
                "--data-seed", str(args.data_seed),
                "--device", "cuda:0",
                "--output-root", str(args.output_root),
                "--epochs", str(args.epochs),
                "--train-labels", str(args.train_labels),
                "--validation-labels", str(args.validation_labels),
                "--test-labels", str(args.test_labels),
                "--samples-per-label", str(args.samples_per_label),
                "--support-ratio", str(args.support_ratio),
                "--batch-size", str(args.batch_size),
                "--strategy-profile", args.strategy_profile,
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update(
                {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            )
            log_path = args.output_root / "logs" / f"expr{expression_id}_{method}_seed{seed}.log"
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
                            "job": list(job),
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
