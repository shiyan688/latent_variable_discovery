#!/usr/bin/env python3
"""Latent-variable proof-of-concept on an exact viscous Burgers shock family."""
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
from lvs.core.metrics import macro_prediction_metrics, reference_scaled_prediction_metrics
from lvs.core.pipeline import (
    LatentQConfig,
    build_dataset_from_arrays,
    denormalize_targets,
    evaluate_latent_q_pipeline,
    normalize_features,
    train_latent_q_model,
)
from scripts.run_iclr_calibration_study import CalibrationStrategy, stable_hash
from scripts.run_iclr_synthetic_calibration_study import (
    _learned_truth_by_label,
    _save_artifacts,
    _true_q_metrics,
)

PYTHON = Path(sys.executable)
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "burgers_latent_poc_20260809"
SCHEMA_VERSION = 1

LATENT_STRATEGIES = (
    CalibrationStrategy("latent_legacy_k1", "legacy_random", 1, 200),
    CalibrationStrategy(
        "latent_adaptive_k4_min24",
        "prior_random",
        4,
        200,
        0.25,
        50,
        24,
        True,
    ),
)

REGULARIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "base": {},
    "whiten_1e3": {"latent_q_whitening_weight": 1e-3},
    "whiten_1e2": {"latent_q_whitening_weight": 1e-2},
    "smooth_1e4": {"latent_q_smoothness_weight": 1e-4},
    "jacobian_1e4": {"latent_jacobian_disentanglement_weight": 1e-4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--q-dim", type=int, required=True)
    run.add_argument("--method", choices=("joint_mse", "alternating_mse"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    run.add_argument(
        "--regularization-profile",
        choices=tuple(REGULARIZATION_PROFILES),
        default="base",
    )
    _shared_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--q-dims", default="1,2,4")
    launch.add_argument("--methods", default="joint_mse,alternating_mse")
    launch.add_argument("--seeds", default="0,1,2")
    launch.add_argument("--gpus", default="4,5,6,7")
    launch.add_argument("--profiles", default="base")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _shared_args(launch)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-seed", type=int, default=20260809)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train-labels", type=int, default=32)
    parser.add_argument("--validation-labels", type=int, default=16)
    parser.add_argument("--test-labels", type=int, default=32)
    parser.add_argument("--x-points", type=int, default=16)
    parser.add_argument("--t-points", type=int, default=8)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)


def _burgers_field(
    x: np.ndarray,
    t: np.ndarray,
    *,
    amplitude: float,
    viscosity: float,
    speed: float = 0.4,
    center: float = -0.2,
) -> np.ndarray:
    coordinate = x - speed * t - center
    return speed - amplitude * np.tanh(amplitude * coordinate / (2.0 * viscosity))


def _analytic_residual(
    x: np.ndarray,
    t: np.ndarray,
    *,
    amplitude: float,
    viscosity: float,
    speed: float = 0.4,
    center: float = -0.2,
) -> np.ndarray:
    coordinate = x - speed * t - center
    kappa = amplitude / (2.0 * viscosity)
    tanh_value = np.tanh(kappa * coordinate)
    sech_squared = 1.0 - tanh_value**2
    u_value = speed - amplitude * tanh_value
    u_t = amplitude * kappa * speed * sech_squared
    u_x = -amplitude * kappa * sech_squared
    u_xx = 2.0 * amplitude * kappa**2 * sech_squared * tanh_value
    return u_t + u_value * u_x - viscosity * u_xx


def make_burgers_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_labels = args.train_labels + args.validation_labels + args.test_labels
    generator = np.random.default_rng(args.data_seed)
    amplitudes = generator.uniform(0.35, 1.15, size=total_labels)
    log_viscosities = generator.uniform(np.log(0.025), np.log(0.16), size=total_labels)
    x_values = np.linspace(-2.0, 2.0, args.x_points)
    t_values = np.linspace(0.0, 1.0, args.t_points)
    grid_x, grid_t = np.meshgrid(x_values, t_values, indexing="xy")
    flat_x = grid_x.reshape(-1)
    flat_t = grid_t.reshape(-1)
    split_boundaries = (
        args.train_labels,
        args.train_labels + args.validation_labels,
    )
    frames = []
    truth_rows = []
    for label in range(total_labels):
        split = (
            "train"
            if label < split_boundaries[0]
            else "validation"
            if label < split_boundaries[1]
            else "test"
        )
        amplitude = float(amplitudes[label])
        log_viscosity = float(log_viscosities[label])
        viscosity = float(np.exp(log_viscosity))
        target = _burgers_field(
            flat_x,
            flat_t,
            amplitude=amplitude,
            viscosity=viscosity,
        )
        frames.append(
            pd.DataFrame(
                {
                    "label": label,
                    "x": flat_x,
                    "t": flat_t,
                    "target": target,
                    "amplitude": amplitude,
                    "log_viscosity": log_viscosity,
                    "split": split,
                }
            )
        )
        truth_rows.append(
            {
                "label": label,
                "q1": amplitude,
                "q2": log_viscosity,
                "viscosity": viscosity,
                "split": split,
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(truth_rows)


def _dataset(frame: pd.DataFrame, features: list[str], *, pooled: bool = False) -> Any:
    labels = np.zeros(len(frame), dtype=np.int64) if pooled else frame["label"].to_numpy()
    return build_dataset_from_arrays(
        frame[features].to_numpy(np.float32),
        labels,
        frame["target"].to_numpy(np.float32),
        feature_names=features,
    )


def _base_config(args: argparse.Namespace, q_dim: int) -> LatentQConfig:
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
        optimization_schedule="alternating" if args.method == "alternating_mse" else "joint",
        joint_steps_per_cycle=2,
        theta_steps_per_cycle=1,
        q_steps_per_cycle=1,
        **REGULARIZATION_PROFILES[args.regularization_profile],
    )


def _prediction_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, float]:
    reference_scale = max(float(np.std(train_targets)), 1e-8)
    metrics = {
        **macro_prediction_metrics(truth, prediction, labels),
        **reference_scaled_prediction_metrics(
            truth, prediction, reference_scale=reference_scale
        ),
    }
    per_label = [
        float(
            np.sqrt(np.mean((truth[labels == label] - prediction[labels == label]) ** 2))
            / reference_scale
        )
        for label in pd.unique(labels)
    ]
    metrics.update(
        {
            "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
            "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
            "label_reference_nrmse_max": float(np.max(per_label)),
        }
    )
    return metrics


def _direct_predictions(training: Any, dataset: Any, indices: np.ndarray) -> np.ndarray:
    features = normalize_features(dataset.features, training.normalizer)
    feature_tensor = torch.tensor(features[indices], dtype=torch.float32, device=training.device)
    constant_q = training.embedding.weight[0].detach().unsqueeze(0).repeat(len(indices), 1)
    training.model.eval()
    with torch.no_grad():
        prediction = training.model(torch.cat([feature_tensor, constant_q], dim=1))
    normalized = prediction.detach().cpu().numpy().reshape(-1)
    return denormalize_targets(normalized, training.normalizer)


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "problem": "viscous_burgers_traveling_shock",
        "q_dim": args.q_dim,
        "method": args.method,
        "seed": args.seed,
        "regularization_profile": args.regularization_profile,
        "regularization": REGULARIZATION_PROFILES[args.regularization_profile],
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "x_points": args.x_points,
        "t_points": args.t_points,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "latent_strategies": [asdict(strategy) for strategy in LATENT_STRATEGIES],
        "baselines_at_q_dim": 2,
    }


def run_job(args: argparse.Namespace) -> Path:
    frame, truth = make_burgers_data(args)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    job = _job_config(args)
    run_dir = (
        args.output_root
        / f"q{args.q_dim}"
        / args.method
        / f"seed{args.seed}_{stable_hash(job)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job") == job:
            return result_path

    feature_columns = ["x", "t"]
    train_dataset = _dataset(train_frame, feature_columns)
    validation_dataset = _dataset(validation_frame, feature_columns)
    test_dataset = _dataset(test_frame, feature_columns)
    config = _base_config(args, args.q_dim)
    started = time.perf_counter()
    latent_training = train_latent_q_model(
        train_dataset, build_torch_model_factory((128, 64)), config
    )
    training_seconds: dict[str, float] = {"latent": time.perf_counter() - started}

    strategy_payloads: dict[str, dict[str, Any]] = {}
    reference_eval_indices: np.ndarray | None = None
    for strategy in LATENT_STRATEGIES:
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
            train_dataset, validation_dataset, latent_training, strategy_config
        )
        testing = evaluate_latent_q_pipeline(
            train_dataset, test_dataset, latent_training, strategy_config
        )
        calibration_seconds = time.perf_counter() - started
        if reference_eval_indices is None:
            reference_eval_indices = testing.eval_indices
        else:
            np.testing.assert_array_equal(reference_eval_indices, testing.eval_indices)
        validation_learned, validation_truth, validation_q = _learned_truth_by_label(
            validation, truth, "validation"
        )
        test_learned, test_truth, test_q = _learned_truth_by_label(
            testing, truth, "test"
        )
        spatial, aligned_test, curve = _true_q_metrics(
            validation_learned=validation_learned,
            validation_truth=validation_truth,
            test_learned=test_learned,
            test_truth=test_truth,
        )
        artifacts = (
            _save_artifacts(
                run_dir=run_dir,
                strategy=strategy.name,
                testing=testing,
                test_frame=test_frame,
                validation_q=validation_q,
                test_q=test_q,
                aligned_test=aligned_test,
                curve=curve,
            )
            if args.save_artifacts
            else {}
        )
        strategy_payloads[strategy.name] = {
            "type": "latent",
            "config": asdict(strategy),
            "prediction": _prediction_metrics(
                testing.eval_targets,
                testing.eval_predictions,
                testing.eval_labels,
                train_frame["target"].to_numpy(float),
            ),
            "spatial": spatial,
            "calibration": {
                key: value
                for key, value in testing.metrics.items()
                if key.startswith("calibration_")
            },
            "calibration_seconds": calibration_seconds,
            "artifacts": artifacts,
        }

    assert reference_eval_indices is not None
    baseline_state_payloads: dict[str, dict[str, Any]] = {}
    if args.q_dim == 2:
        baseline_specs = {
            "pooled_mlp_no_latent": (["x", "t"], False),
            "oracle_parameter_mlp": (["x", "t", "amplitude", "log_viscosity"], True),
        }
        for name, (features, is_oracle) in baseline_specs.items():
            baseline_train = _dataset(train_frame, features, pooled=True)
            baseline_test = _dataset(test_frame, features, pooled=True)
            baseline_config = _base_config(args, 1)
            started = time.perf_counter()
            baseline_training = train_latent_q_model(
                baseline_train, build_torch_model_factory((128, 64)), baseline_config
            )
            training_seconds[name] = time.perf_counter() - started
            baseline_state_payloads[name] = {
                "config": asdict(baseline_config),
                "model_state_dict": baseline_training.model.state_dict(),
                "embedding_state_dict": baseline_training.embedding.state_dict(),
                "normalizer": asdict(baseline_training.normalizer),
            }
            prediction = _direct_predictions(
                baseline_training, baseline_test, reference_eval_indices
            )
            targets = test_frame["target"].to_numpy(float)[reference_eval_indices]
            labels = test_frame["label"].to_numpy()[reference_eval_indices]
            prediction_path = run_dir / f"query_predictions_{name}.csv"
            if args.save_artifacts:
                output = test_frame.iloc[reference_eval_indices].copy()
                output["prediction"] = prediction
                output.to_csv(prediction_path, index=False)
            strategy_payloads[name] = {
                "type": "oracle" if is_oracle else "no_latent_baseline",
                "prediction": _prediction_metrics(
                    targets,
                    prediction,
                    labels,
                    train_frame["target"].to_numpy(float),
                ),
                "spatial": {},
                "calibration": {},
                "calibration_seconds": 0.0,
                "artifacts": {
                    "query_predictions": str(prediction_path)
                    if args.save_artifacts
                    else None
                },
            }

    checkpoint_path = run_dir / "training_checkpoint.pt"
    if args.save_artifacts:
        torch.save(
            {
                "job": job,
                "latent_config": asdict(config),
                "latent_model_state_dict": latent_training.model.state_dict(),
                "latent_embedding_state_dict": latent_training.embedding.state_dict(),
                "latent_normalizer": asdict(latent_training.normalizer),
                "baseline_states": baseline_state_payloads,
            },
            checkpoint_path,
        )
    all_residuals = []
    for row in truth.itertuples(index=False):
        all_residuals.append(
            np.max(
                np.abs(
                    _analytic_residual(
                        test_frame["x"].to_numpy(float)[: args.x_points * args.t_points],
                        test_frame["t"].to_numpy(float)[: args.x_points * args.t_points],
                        amplitude=float(row.q1),
                        viscosity=float(row.viscosity),
                    )
                )
            )
        )
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
        "physics": {
            "equation": "u_t + u u_x = nu u_xx",
            "solution": "u = c - a tanh(a(x-ct-x0)/(2nu))",
            "speed": 0.4,
            "center": -0.2,
            "analytic_residual_max": float(np.max(all_residuals)),
        },
        "training_seconds": training_seconds,
        "training_checkpoint": str(checkpoint_path) if args.save_artifacts else None,
        "strategies": strategy_payloads,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def summarize(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("q*/*/seed*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            continue
        for strategy, values in payload["strategies"].items():
            rows.append(
                {
                    **{key: value for key, value in payload["job"].items() if key != "latent_strategies"},
                    "strategy": strategy,
                    "strategy_type": values["type"],
                    **values["prediction"],
                    **values["spatial"],
                    **values["calibration"],
                    "calibration_seconds": values["calibration_seconds"],
                    "result_path": str(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "all_runs.csv", index=False)
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
            "local_log_distortion_p95",
            "calibration_seconds",
        )
        if column in frame
    ]
    summary = frame.groupby(["q_dim", "strategy"], as_index=False)[metrics].agg(
        ["count", "mean", "std"]
    )
    summary.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_root / "strategy_summary.csv", index=False)


def launch(args: argparse.Namespace) -> None:
    q_dims = [int(value) for value in args.q_dims.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    profiles = [value for value in args.profiles.split(",") if value]
    invalid_profiles = set(profiles) - set(REGULARIZATION_PROFILES)
    if invalid_profiles:
        raise ValueError(f"Unsupported regularization profiles: {sorted(invalid_profiles)}")
    gpus = [value for value in args.gpus.split(",") if value]
    jobs = [
        (q_dim, method, seed, profile)
        for q_dim in q_dims
        for method in methods
        for seed in seeds
        for profile in profiles
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "protocol": "exact Burgers shock; disjoint parameter labels; support/query calibration; q_dim=2 baseline block",
        "q_dims": q_dims,
        "methods": methods,
        "seeds": seeds,
        "regularization_profiles": {
            profile: REGULARIZATION_PROFILES[profile] for profile in profiles
        },
        "gpus": gpus,
        "data_seed": args.data_seed,
        "epochs": args.epochs,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "x_points": args.x_points,
        "t_points": args.t_points,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "latent_strategies": [asdict(strategy) for strategy in LATENT_STRATEGIES],
        "baselines": ["pooled_mlp_no_latent", "oracle_parameter_mlp"],
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending = list(jobs)
    running: list[tuple[subprocess.Popen[Any], tuple[int, str, int, str], Any, str]] = []
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            job = pending.pop(0)
            q_dim, method, seed, profile = job
            available = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            gpu = available[0]
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "run-job",
                "--q-dim", str(q_dim), "--method", method, "--seed", str(seed),
                "--regularization-profile", profile,
                "--device", "cuda:0", "--output-root", str(args.output_root),
                "--data-seed", str(args.data_seed), "--epochs", str(args.epochs),
                "--train-labels", str(args.train_labels),
                "--validation-labels", str(args.validation_labels),
                "--test-labels", str(args.test_labels),
                "--x-points", str(args.x_points), "--t-points", str(args.t_points),
                "--support-ratio", str(args.support_ratio),
                "--batch-size", str(args.batch_size),
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update(
                {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            )
            log_path = args.output_root / "logs" / f"q{q_dim}_{method}_seed{seed}_{profile}.log"
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
