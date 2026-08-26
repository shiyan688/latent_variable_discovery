#!/usr/bin/env python3
"""External latent-variable study on the official PDEBench 1D Burgers data."""
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

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.loss_presets import get_loss_preset
from lvs.core.metrics import (
    alignment_metrics,
    apply_affine_alignment,
    effective_rank,
    fit_affine_alignment,
    knn_overlap,
    local_distance_distortion,
    macro_prediction_metrics,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
    reference_scaled_prediction_metrics,
)
from lvs.core.pipeline import (
    LatentQConfig,
    build_dataset_from_arrays,
    denormalize_targets,
    evaluate_latent_q_pipeline,
    normalize_features,
    split_support_query_indices,
    train_latent_q_model,
)
from scripts.run_iclr_calibration_study import CalibrationStrategy, stable_hash

PYTHON = Path(sys.executable)
DEFAULT_DATA = PROJECT_ROOT / "data" / "external" / "pdebench" / "1D_Burgers_Sols_Nu0.02.hdf5"
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "pdebench_burgers_latent_20260809"
SCHEMA_VERSION = 1
SOURCE_URL = "https://huggingface.co/datasets/pdebench/Burgers/blob/main/1D_Burgers_Sols_Nu0.02.hdf5"
SOURCE_SHA256 = "b1d1ef10a612abef7eedd99873323289416a53c737c6cf04cb59c90020ed1911"
SOURCE_MD5 = "7c8c717a3a7818145877baa57106b090"

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


@dataclass(frozen=True)
class Method:
    name: str
    schedule: str = "joint"
    loss_preset: str = "mse"
    joint_steps_per_cycle: int = 2


METHODS = {
    method.name: method
    for method in (
        Method("joint_mse"),
        Method("joint_mse_step1", joint_steps_per_cycle=1),
        Method("joint_continuity_step1", loss_preset="continuity", joint_steps_per_cycle=1),
        Method("alternating_mse", schedule="alternating"),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _data_args(prepare)

    run = subparsers.add_parser("run-job")
    run.add_argument("--q-dim", type=int, required=True)
    run.add_argument("--method", choices=tuple(METHODS), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _training_args(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--q-dims", default="4,8,16")
    launch.add_argument("--methods", default="joint_mse,alternating_mse")
    launch.add_argument("--seeds", default="0,1,2")
    launch.add_argument("--gpus", default="4,5,6,7")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _training_args(launch)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-seed", type=int, default=20260809)
    parser.add_argument("--train-labels", type=int, default=64)
    parser.add_argument("--validation-labels", type=int, default=16)
    parser.add_argument("--test-labels", type=int, default=32)
    parser.add_argument("--x-points", type=int, default=32)
    parser.add_argument("--t-points", type=int, default=16)


def _training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_subset(args: argparse.Namespace) -> Path:
    data_path = args.data_path.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    total_labels = args.train_labels + args.validation_labels + args.test_labels
    args.output_root.mkdir(parents=True, exist_ok=True)
    subset_path = args.output_root / "pdebench_burgers_subset.npz"
    manifest_path = args.output_root / "subset_manifest.json"
    config = {
        "schema_version": SCHEMA_VERSION,
        "data_seed": args.data_seed,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "x_points": args.x_points,
        "t_points": args.t_points,
        "source_path": str(data_path),
        "source_size": data_path.stat().st_size,
        "source_sha256": SOURCE_SHA256,
        "source_md5": SOURCE_MD5,
    }
    if subset_path.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config") == config and existing.get("status") == "success":
            return subset_path

    actual_sha256 = _sha256(data_path)
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(f"PDEBench source SHA-256 mismatch: {actual_sha256}")

    with h5py.File(data_path, "r") as handle:
        if "tensor" not in handle:
            raise KeyError(f"Expected 'tensor'; found {sorted(handle.keys())}")
        tensor = handle["tensor"]
        if tensor.ndim != 3:
            raise ValueError(f"Expected [trajectory,time,x], got {tensor.shape}")
        trajectory_count, time_count, space_count = map(int, tensor.shape)
        tensor_dtype = str(tensor.dtype)
        if total_labels > trajectory_count:
            raise ValueError("Requested more trajectories than the source contains.")
        rng = np.random.default_rng(args.data_seed)
        source_indices = rng.choice(trajectory_count, size=total_labels, replace=False)
        x_indices = np.unique(np.rint(np.linspace(0, space_count - 1, args.x_points)).astype(int))
        t_indices = np.unique(np.rint(np.linspace(0, time_count - 1, args.t_points)).astype(int))
        if len(x_indices) != args.x_points or len(t_indices) != args.t_points:
            raise ValueError("Requested sampling grid contains duplicate source indices.")
        if "x-coordinate" in handle:
            x_coordinates = np.asarray(handle["x-coordinate"], dtype=np.float32)[x_indices]
        else:
            x_coordinates = np.linspace(0.0, 1.0, space_count, dtype=np.float32)[x_indices]
        if "t-coordinate" in handle:
            t_coordinates = np.asarray(handle["t-coordinate"], dtype=np.float32)[t_indices]
        else:
            t_coordinates = np.linspace(0.0, 1.0, time_count, dtype=np.float32)[t_indices]

        targets = np.empty((total_labels, len(t_indices), len(x_indices)), dtype=np.float32)
        initial_conditions = np.empty((total_labels, len(x_indices)), dtype=np.float32)
        for label, source_index in enumerate(source_indices):
            trajectory = np.asarray(tensor[int(source_index)], dtype=np.float32)
            targets[label] = trajectory[np.ix_(t_indices, x_indices)]
            initial_conditions[label] = trajectory[0, x_indices]

    split = np.full(total_labels, "test", dtype="U10")
    split[: args.train_labels] = "train"
    split[args.train_labels : args.train_labels + args.validation_labels] = "validation"
    temporary = subset_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            targets=targets,
            initial_conditions=initial_conditions,
            source_indices=source_indices.astype(np.int64),
            split=split,
            x_coordinates=x_coordinates,
            t_coordinates=t_coordinates,
            x_indices=x_indices.astype(np.int64),
            t_indices=t_indices.astype(np.int64),
        )
    temporary.replace(subset_path)
    payload = {
        "status": "success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "official_source": SOURCE_URL,
        "source_shape": [trajectory_count, time_count, space_count],
        "source_dtype": tensor_dtype,
        "sampled_source_indices": source_indices.tolist(),
        "x_indices": x_indices.tolist(),
        "t_indices": t_indices.tolist(),
        "subset_path": str(subset_path),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return subset_path


def _load_subset(output_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    subset_path = output_root / "pdebench_burgers_subset.npz"
    if not subset_path.is_file():
        raise FileNotFoundError(f"Run prepare first: {subset_path}")
    with np.load(subset_path, allow_pickle=False) as data:
        targets = data["targets"]
        initial_conditions = data["initial_conditions"]
        source_indices = data["source_indices"]
        split = data["split"]
        x_coordinates = data["x_coordinates"]
        t_coordinates = data["t_coordinates"]
    rows: list[pd.DataFrame] = []
    grid_t, grid_x = np.meshgrid(t_coordinates, x_coordinates, indexing="ij")
    for label in range(len(split)):
        rows.append(
            pd.DataFrame(
                {
                    "label": label,
                    "source_trajectory": int(source_indices[label]),
                    "x": grid_x.reshape(-1),
                    "t": grid_t.reshape(-1),
                    "target": targets[label].reshape(-1),
                    "split": split[label],
                }
            )
        )
    metadata = {
        "initial_conditions": initial_conditions,
        "source_indices": source_indices,
        "split": split,
    }
    return pd.concat(rows, ignore_index=True), metadata


def _dataset(frame: pd.DataFrame, features: list[str], *, pooled: bool = False) -> Any:
    labels = np.zeros(len(frame), dtype=np.int64) if pooled else frame["label"].to_numpy()
    return build_dataset_from_arrays(
        frame[features].to_numpy(np.float32),
        labels,
        frame["target"].to_numpy(np.float32),
        feature_names=features,
    )


def _base_config(args: argparse.Namespace, q_dim: int) -> LatentQConfig:
    method = METHODS[args.method]
    common = dict(
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
        optimization_schedule=method.schedule,
        joint_steps_per_cycle=method.joint_steps_per_cycle,
        theta_steps_per_cycle=1,
        q_steps_per_cycle=1,
    )
    return LatentQConfig(**common, **get_loss_preset(method.loss_preset).config_kwargs())


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
    per_label = [
        float(
            np.sqrt(np.mean((truth[labels == label] - prediction[labels == label]) ** 2))
            / reference_scale
        )
        for label in pd.unique(labels)
    ]
    output.update(
        {
            "label_reference_nrmse_p90": float(np.quantile(per_label, 0.90)),
            "label_reference_nrmse_p95": float(np.quantile(per_label, 0.95)),
            "label_reference_nrmse_max": float(np.max(per_label)),
        }
    )
    return output


def _q_by_label(result: Any) -> np.ndarray:
    q_columns = [column for column in result.test_output.columns if column.startswith("q")]
    return (
        result.test_output.groupby("label", sort=True)[q_columns]
        .mean()
        .to_numpy(dtype=float)
    )


def _initial_condition_metrics(
    *,
    validation_result: Any,
    test_result: Any,
    metadata: dict[str, np.ndarray],
    q_dim: int,
) -> tuple[dict[str, float], np.ndarray, list[dict[str, float]]]:
    split = metadata["split"]
    initial_conditions = metadata["initial_conditions"]
    train_ic = initial_conditions[split == "train"]
    validation_ic = initial_conditions[split == "validation"]
    test_ic = initial_conditions[split == "test"]
    components = min(q_dim, 8, len(train_ic) - 1, train_ic.shape[1])
    pca = PCA(n_components=components, random_state=0).fit(train_ic)
    validation_reference = pca.transform(validation_ic)
    test_reference = pca.transform(test_ic)
    validation_q = _q_by_label(validation_result)
    test_q = _q_by_label(test_result)
    alignment = fit_affine_alignment(validation_q, validation_reference)
    aligned_test = apply_affine_alignment(alignment, test_q)
    curve = neighborhood_preservation_curve(
        test_reference,
        test_q,
        max_k=min(10, (len(test_q) - 1) // 2),
    )
    metrics = {
        **alignment_metrics(test_reference, aligned_test),
        **pairwise_distance_metrics(test_q, test_reference),
        "trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in curve])),
        "continuity_auc": float(np.mean([row["continuity"] for row in curve])),
        "knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve])),
        **local_distance_distortion(test_reference, test_q, k=min(5, len(test_q) - 1)),
        "knn_overlap": knn_overlap(test_q, test_reference, k=min(3, len(test_q) - 1)),
        "effective_rank": effective_rank(test_q),
        "ic_pca_components": int(components),
        "ic_pca_train_variance_explained": float(pca.explained_variance_ratio_.sum()),
    }
    return metrics, aligned_test, curve


def _direct_predictions(training: Any, dataset: Any, indices: np.ndarray) -> np.ndarray:
    features = normalize_features(dataset.features, training.normalizer)
    feature_tensor = torch.tensor(features[indices], dtype=torch.float32, device=training.device)
    constant_q = training.embedding.weight[0].detach().unsqueeze(0).repeat(len(indices), 1)
    training.model.eval()
    with torch.no_grad():
        normalized = training.model(torch.cat([feature_tensor, constant_q], dim=1))
    return denormalize_targets(
        normalized.detach().cpu().numpy().reshape(-1), training.normalizer
    )


def _same_support_baselines(
    test_frame: pd.DataFrame,
    query_indices: np.ndarray,
    *,
    support_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    predictions = {
        "support_mean": np.empty(len(query_indices), dtype=float),
        "support_knn4": np.empty(len(query_indices), dtype=float),
    }
    query_position = {int(index): position for position, index in enumerate(query_indices)}
    feature_scale = np.maximum(test_frame[["x", "t"]].std().to_numpy(float), 1e-8)
    for raw_label in pd.unique(test_frame["label"]):
        label_indices = np.flatnonzero(test_frame["label"].to_numpy() == raw_label)
        support_indices, expected_query = split_support_query_indices(
            label_indices,
            support_ratio,
            mode="random",
            seed=seed,
            label=int(raw_label),
        )
        actual = np.asarray([index for index in expected_query if int(index) in query_position])
        support_x = test_frame.iloc[support_indices][["x", "t"]].to_numpy(float) / feature_scale
        support_y = test_frame.iloc[support_indices]["target"].to_numpy(float)
        query_x = test_frame.iloc[actual][["x", "t"]].to_numpy(float) / feature_scale
        distances = np.sqrt(((query_x[:, None, :] - support_x[None, :, :]) ** 2).sum(axis=2))
        neighbors = np.argsort(distances, axis=1, kind="stable")[:, : min(4, len(support_y))]
        neighbor_distances = np.take_along_axis(distances, neighbors, axis=1)
        weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
        knn_values = (support_y[neighbors] * weights).sum(axis=1) / weights.sum(axis=1)
        for index, knn_value in zip(actual, knn_values):
            position = query_position[int(index)]
            predictions["support_mean"][position] = support_y.mean()
            predictions["support_knn4"][position] = knn_value
    return predictions


def _add_ic_pca_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    metadata: dict[str, np.ndarray],
    components: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, np.ndarray]]:
    split = metadata["split"]
    initial_conditions = metadata["initial_conditions"]
    resolved = min(components, int((split == "train").sum()) - 1, initial_conditions.shape[1])
    pca = PCA(n_components=resolved, random_state=0).fit(initial_conditions[split == "train"])
    transformed = pca.transform(initial_conditions)
    columns = [f"ic_pc{index + 1}" for index in range(resolved)]
    mapping = pd.DataFrame(transformed, columns=columns)
    mapping["label"] = np.arange(len(mapping))
    return (
        train_frame.merge(mapping, on="label", how="left", validate="many_to_one"),
        test_frame.merge(mapping, on="label", how="left", validate="many_to_one"),
        columns,
        {
            "components": pca.components_.copy(),
            "mean": pca.mean_.copy(),
            "explained_variance": pca.explained_variance_.copy(),
            "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
        },
    )


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.output_root / "subset_manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "problem": "pdebench_1d_burgers_nu0.02_random_initial_conditions",
        "q_dim": args.q_dim,
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "subset_config": manifest["config"],
        "latent_strategies": [asdict(strategy) for strategy in LATENT_STRATEGIES],
        "method_config": asdict(METHODS[args.method]),
        "baseline_block": args.q_dim == 8
        and args.method in {"joint_mse", "joint_mse_step1"},
    }


def run_job(args: argparse.Namespace) -> Path:
    frame, metadata = _load_subset(args.output_root)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame[frame["split"] == "validation"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    job = _job_config(args)
    run_dir = args.output_root / f"q{args.q_dim}" / args.method / f"seed{args.seed}_{stable_hash(job)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job") == job:
            return result_path

    features = ["x", "t"]
    train_dataset = _dataset(train_frame, features)
    validation_dataset = _dataset(validation_frame, features)
    test_dataset = _dataset(test_frame, features)
    config = _base_config(args, args.q_dim)
    started = time.perf_counter()
    training = train_latent_q_model(
        train_dataset, build_torch_model_factory((128, 64)), config
    )
    training_seconds: dict[str, float] = {"latent": time.perf_counter() - started}
    optimization_counters: dict[str, Any] = {
        "latent": asdict(training.optimization_counters)
    }
    strategies: dict[str, dict[str, Any]] = {}
    reference_query_indices: np.ndarray | None = None

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
            train_dataset, validation_dataset, training, strategy_config
        )
        testing = evaluate_latent_q_pipeline(
            train_dataset, test_dataset, training, strategy_config
        )
        calibration_seconds = time.perf_counter() - started
        if reference_query_indices is None:
            reference_query_indices = testing.eval_indices
        else:
            np.testing.assert_array_equal(reference_query_indices, testing.eval_indices)
        spatial, aligned_test, curve = _initial_condition_metrics(
            validation_result=validation,
            test_result=testing,
            metadata=metadata,
            q_dim=args.q_dim,
        )
        prediction_path = run_dir / f"query_predictions_{strategy.name}.csv"
        q_path = run_dir / f"test_q_{strategy.name}.csv"
        aligned_path = run_dir / f"test_q_aligned_ic_pca_{strategy.name}.csv"
        curve_path = run_dir / f"continuity_curve_{strategy.name}.csv"
        if args.save_artifacts:
            output = test_frame.iloc[testing.eval_indices].copy()
            output["prediction"] = testing.eval_predictions
            output.to_csv(prediction_path, index=False)
            q_columns = [f"q{index + 1}" for index in range(args.q_dim)]
            q_frame = pd.DataFrame(_q_by_label(testing), columns=q_columns)
            q_frame["label"] = sorted(test_frame["label"].unique())
            q_frame.to_csv(q_path, index=False)
            pd.DataFrame(aligned_test).to_csv(aligned_path, index=False)
            pd.DataFrame(curve).to_csv(curve_path, index=False)
        strategies[strategy.name] = {
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
                key: value for key, value in testing.metrics.items() if key.startswith("calibration_")
            },
            "calibration_seconds": calibration_seconds,
            "artifacts": {
                "query_predictions": str(prediction_path),
                "test_q": str(q_path),
                "aligned_ic_pca": str(aligned_path),
                "continuity_curve": str(curve_path),
            },
        }

    assert reference_query_indices is not None
    baseline_states: dict[str, Any] = {}
    if job["baseline_block"]:
        query_targets = test_frame["target"].to_numpy(float)[reference_query_indices]
        query_labels = test_frame["label"].to_numpy()[reference_query_indices]
        for name, prediction in _same_support_baselines(
            test_frame,
            reference_query_indices,
            support_ratio=args.support_ratio,
            seed=args.seed,
        ).items():
            prediction_path = run_dir / f"query_predictions_{name}.csv"
            if args.save_artifacts:
                output = test_frame.iloc[reference_query_indices].copy()
                output["prediction"] = prediction
                output.to_csv(prediction_path, index=False)
            strategies[name] = {
                "type": "same_support_baseline",
                "prediction": _prediction_metrics(
                    query_targets,
                    prediction,
                    query_labels,
                    train_frame["target"].to_numpy(float),
                ),
                "spatial": {},
                "calibration": {},
                "calibration_seconds": 0.0,
                "artifacts": {
                    "query_predictions": str(prediction_path) if args.save_artifacts else None
                },
            }

        pooled_train = _dataset(train_frame, features, pooled=True)
        pooled_test = _dataset(test_frame, features, pooled=True)
        pooled_config = _base_config(args, 1)
        started = time.perf_counter()
        pooled_training = train_latent_q_model(
            pooled_train, build_torch_model_factory((128, 64)), pooled_config
        )
        training_seconds["pooled_mlp_no_latent"] = time.perf_counter() - started
        optimization_counters["pooled_mlp_no_latent"] = asdict(
            pooled_training.optimization_counters
        )
        pooled_prediction = _direct_predictions(
            pooled_training, pooled_test, reference_query_indices
        )
        pooled_prediction_path = run_dir / "query_predictions_pooled_mlp_no_latent.csv"
        if args.save_artifacts:
            output = test_frame.iloc[reference_query_indices].copy()
            output["prediction"] = pooled_prediction
            output.to_csv(pooled_prediction_path, index=False)
        strategies["pooled_mlp_no_latent"] = {
            "type": "no_latent_baseline",
            "prediction": _prediction_metrics(
                query_targets,
                pooled_prediction,
                query_labels,
                train_frame["target"].to_numpy(float),
            ),
            "spatial": {},
            "calibration": {},
            "calibration_seconds": 0.0,
            "artifacts": {
                "query_predictions": str(pooled_prediction_path)
                if args.save_artifacts
                else None
            },
        }
        baseline_states["pooled_mlp_no_latent"] = {
            "model_state_dict": pooled_training.model.state_dict(),
            "embedding_state_dict": pooled_training.embedding.state_dict(),
            "normalizer": asdict(pooled_training.normalizer),
        }

        ic_train, ic_test, ic_columns, ic_pca_state = _add_ic_pca_features(
            train_frame, test_frame, metadata
        )
        ic_features = ["x", "t", *ic_columns]
        ic_train_dataset = _dataset(ic_train, ic_features, pooled=True)
        ic_test_dataset = _dataset(ic_test, ic_features, pooled=True)
        started = time.perf_counter()
        ic_training = train_latent_q_model(
            ic_train_dataset, build_torch_model_factory((128, 64)), pooled_config
        )
        training_seconds["full_ic_pca_mlp_reference"] = time.perf_counter() - started
        optimization_counters["full_ic_pca_mlp_reference"] = asdict(
            ic_training.optimization_counters
        )
        ic_prediction = _direct_predictions(
            ic_training, ic_test_dataset, reference_query_indices
        )
        ic_prediction_path = run_dir / "query_predictions_full_ic_pca_mlp_reference.csv"
        if args.save_artifacts:
            output = test_frame.iloc[reference_query_indices].copy()
            output["prediction"] = ic_prediction
            output.to_csv(ic_prediction_path, index=False)
        strategies["full_ic_pca_mlp_reference"] = {
            "type": "extra_information_reference",
            "prediction": _prediction_metrics(
                query_targets,
                ic_prediction,
                query_labels,
                train_frame["target"].to_numpy(float),
            ),
            "spatial": {},
            "calibration": {},
            "calibration_seconds": 0.0,
            "artifacts": {
                "query_predictions": str(ic_prediction_path)
                if args.save_artifacts
                else None
            },
            "note": "Uses the full t=0 field compressed by train-only PCA; not a same-information competitor.",
        }
        baseline_states["full_ic_pca_mlp_reference"] = {
            "model_state_dict": ic_training.model.state_dict(),
            "embedding_state_dict": ic_training.embedding.state_dict(),
            "normalizer": asdict(ic_training.normalizer),
            "pca_feature_columns": ic_columns,
            "pca_state": ic_pca_state,
        }

    checkpoint_path = run_dir / "training_checkpoint.pt"
    if args.save_artifacts:
        torch.save(
            {
                "job": job,
                "latent_config": asdict(config),
                "latent_model_state_dict": training.model.state_dict(),
                "latent_embedding_state_dict": training.embedding.state_dict(),
                "latent_normalizer": asdict(training.normalizer),
                "baseline_states": baseline_states,
            },
            checkpoint_path,
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
        "training_seconds": training_seconds,
        "optimization_counters": optimization_counters,
        "latent_config": asdict(config),
        "training_checkpoint": str(checkpoint_path) if args.save_artifacts else None,
        "strategies": strategies,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def summarize(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in output_root.glob("q*/*/seed*/result.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            continue
        for strategy, values in payload["strategies"].items():
            rows.append(
                {
                    **{
                        key: value
                        for key, value in payload["job"].items()
                        if key not in {"latent_strategies", "subset_config"}
                    },
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
    output_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_root / "all_runs.csv", index=False)
    if frame.empty:
        return
    metrics = [
        column
        for column in (
            "reference_nrmse",
            "label_reference_nrmse_p95",
            "aligned_nrmse",
            "distance_spearman",
            "continuity_auc",
            "local_log_distortion_p95",
            "effective_rank",
            "calibration_seconds",
        )
        if column in frame
    ]
    summary = frame.groupby(["q_dim", "method", "strategy"], as_index=False)[metrics].agg(
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
    gpus = [value for value in args.gpus.split(",") if value]
    jobs = [(q_dim, method, seed) for q_dim in q_dims for method in methods for seed in seeds]
    if not (args.output_root / "pdebench_burgers_subset.npz").is_file():
        raise FileNotFoundError("Prepared subset is missing; run prepare first.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "protocol": "PDEBench Nu=0.02; disjoint trajectories; sparse random support/query; initial-condition geometry audit",
        "q_dims": q_dims,
        "methods": methods,
        "seeds": seeds,
        "gpus": gpus,
        "epochs": args.epochs,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "latent_strategies": [asdict(strategy) for strategy in LATENT_STRATEGIES],
        "baselines": [
            "pooled_mlp_no_latent",
            "support_mean",
            "support_knn4",
            "full_ic_pca_mlp_reference",
        ],
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pending = list(jobs)
    running: list[tuple[subprocess.Popen[Any], tuple[int, str, int], Any, str]] = []
    failed_jobs: list[dict[str, Any]] = []
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            q_dim, method, seed = pending.pop(0)
            available = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            gpu = available[0]
            command = [
                str(PYTHON),
                str(Path(__file__).resolve()),
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
                str(args.output_root),
                "--epochs",
                str(args.epochs),
                "--support-ratio",
                str(args.support_ratio),
                "--batch-size",
                str(args.batch_size),
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update(
                {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            )
            log_path = args.output_root / "logs" / f"q{q_dim}_{method}_seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((process, (q_dim, method, seed), handle, gpu))
        next_running = []
        for process, job, handle, gpu in running:
            return_code = process.poll()
            if return_code is None:
                next_running.append((process, job, handle, gpu))
                continue
            handle.close()
            if return_code != 0:
                failed_jobs.append({"job": list(job), "returncode": return_code})
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
                "status": "completed" if not failed_jobs else "failed",
                "jobs": len(jobs),
                "failed_jobs": failed_jobs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if failed_jobs:
        raise RuntimeError(f"{len(failed_jobs)} PDEBench jobs failed; see launcher_status.jsonl")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        print(prepare_subset(args))
    elif args.command == "run-job":
        print(run_job(args))
    elif args.command == "launch":
        launch(args)
    else:
        summarize(args.output_root)


if __name__ == "__main__":
    main()
