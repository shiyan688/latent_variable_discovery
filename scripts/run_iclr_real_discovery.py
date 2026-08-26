#!/usr/bin/env python3
"""Reviewer-facing real-data benchmark for latent-q discovery and baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
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
from lvs.core.loss_presets import LOSS_SWEEP_METHOD_PRESETS, get_loss_preset
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
    OutputConfig,
    build_dataset_from_arrays,
    evaluate_latent_q_pipeline,
    split_support_query_indices,
    train_latent_q_model,
)

PYTHON = Path(sys.executable)
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_real_discovery"


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    schedule: str = "joint"
    weighting: str = "static"
    loss_preset: str = "mse"
    joint_steps_per_cycle: int = 2


METHODS = {
    method.name: method
    for method in (
        Method("joint_mse", "latent"),
        Method("joint_mse_step1", "latent", joint_steps_per_cycle=1),
        Method("alternating_mse", "latent", schedule="alternating"),
        Method("joint_lb_mse", "latent", loss_preset="label_balanced_mse"),
        Method("joint_hsic", "latent", loss_preset="hsic"),
        Method("joint_continuity", "latent", loss_preset="continuity"),
        Method(
            "joint_continuity_step1",
            "latent",
            loss_preset="continuity",
            joint_steps_per_cycle=1,
        ),
        Method("joint_q_l2", "latent", loss_preset="q_l2"),
        Method("joint_calprior", "latent", loss_preset="calibration_prior"),
        Method("joint_hsic_cont", "latent", loss_preset="hsic_continuity"),
        Method("joint_all_mse", "latent", loss_preset="all_mse"),
        Method("joint_fixed", "latent", loss_preset="all_label_balanced"),
        Method(
            "alternating_fixed",
            "latent",
            schedule="alternating",
            loss_preset="all_label_balanced",
        ),
        Method(
            "joint_dynamic",
            "latent",
            weighting="adaptive_loss_scale",
            loss_preset="all_label_balanced",
        ),
        Method(
            "alternating_dynamic",
            "latent",
            schedule="alternating",
            weighting="adaptive_loss_scale",
            loss_preset="all_label_balanced",
        ),
        Method("no_q_mlp", "no_q_mlp"),
        Method("random_forest", "random_forest"),
        Method("support_knn", "support_knn"),
        *(
            Method(name, "latent", loss_preset=preset)
            for name, preset in LOSS_SWEEP_METHOD_PRESETS.items()
        ),
    )
}

PRIMARY_METHODS = (
    "joint_mse",
    "alternating_mse",
    "joint_fixed",
    "alternating_fixed",
    "joint_dynamic",
    "alternating_dynamic",
    "no_q_mlp",
    "random_forest",
    "support_knn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--prepared-summary", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--method", choices=tuple(METHODS), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--q-dim", type=int, default=2)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_arguments(run)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--prepared-summary", type=Path, action="append", required=True)
    launch.add_argument("--datasets", default="")
    launch.add_argument("--methods", default=",".join(PRIMARY_METHODS))
    launch.add_argument("--seeds", default="0,1,2")
    launch.add_argument("--q-dims", default="2")
    launch.add_argument("--gpus", default="4,5")
    launch.add_argument("--gpu-memory-threshold-mib", type=int, default=128)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    _add_shared_arguments(launch)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--cal-steps", type=int, default=300)
    parser.add_argument(
        "--cal-init-mode",
        choices=("legacy_random", "prior_random", "zero", "train_mean"),
        default="legacy_random",
    )
    parser.add_argument("--cal-num-starts", type=int, default=1)
    parser.add_argument("--cal-selection-ratio", type=float, default=0.0)
    parser.add_argument("--cal-selection-min-rows", type=int, default=2)
    parser.add_argument("--cal-refine-steps", type=int, default=0)
    parser.add_argument(
        "--cal-refine-only-after-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument(
        "--support-split-mode", choices=("random", "prefix"), default="random"
    )
    parser.add_argument("--support-order-column", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--max-train-per-label", type=int, default=0)
    parser.add_argument("--max-test-per-label", type=int, default=0)
    parser.add_argument("--subsample-seed", type=int, default=20260808)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _has_successful_result(
    output_root: Path,
    *,
    dataset: str,
    method: str,
    seed: int,
    q_dim: int,
    expected_job: dict[str, Any],
) -> bool:
    """Return whether an exactly matching job already finished successfully."""
    pattern = f"{dataset}/{method}/seed{seed}_q{q_dim}_*/result.json"
    for result_path in output_root.glob(pattern):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        job = result.get("job", {})
        if result.get("status") == "success" and all(
            job.get(key) == value for key, value in expected_job.items()
        ):
            return True
    return False


def _load_record(summary_path: Path, dataset_name: str) -> dict[str, Any]:
    records = json.loads(summary_path.read_text(encoding="utf-8"))
    matches = [record for record in records if record["name"] == dataset_name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one record for {dataset_name!r} in {summary_path}.")
    return matches[0]


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cap_rows_per_label(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame.reset_index(drop=True)
    pieces = []
    for index, (_, group) in enumerate(frame.groupby("label", sort=False)):
        if len(group) > maximum:
            group = group.sample(n=maximum, random_state=seed + index)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _support_query_indices(
    labels: np.ndarray, ratio: float, seed: int, mode: str = "random"
) -> tuple[np.ndarray, np.ndarray]:
    support_parts = []
    query_parts = []
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        support, query = split_support_query_indices(
            indices, ratio, mode=mode, seed=seed, label=label
        )
        support_parts.append(support)
        query_parts.append(query)
    return np.concatenate(support_parts), np.concatenate(query_parts)


def _prediction_payload(
    truth: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    train_targets: np.ndarray,
) -> dict[str, float]:
    return {
        **macro_prediction_metrics(truth, prediction, labels),
        **reference_scaled_prediction_metrics(
            truth, prediction, reference_scale=float(np.std(train_targets))
        ),
    }


def run_job(args: argparse.Namespace) -> Path:
    method = METHODS[args.method]
    record = _load_record(args.prepared_summary, args.dataset)
    feature_columns = list(record["feature_columns"])
    train_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["train_csv"])), args.max_train_per_label, args.subsample_seed
    )
    test_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["test_csv"])), args.max_test_per_label, args.subsample_seed + 10000
    )
    required = {"label", "target", *feature_columns}
    for name, frame in (("train", train_frame), ("test", test_frame)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data are missing columns: {sorted(missing)}")
    if args.support_split_mode == "prefix":
        if not args.support_order_column:
            raise ValueError("--support-order-column is required for prefix support")
        if args.support_order_column not in test_frame:
            raise ValueError(f"test data are missing support order column {args.support_order_column!r}")
        test_frame = test_frame.sort_values(
            ["label", args.support_order_column], kind="stable"
        ).reset_index(drop=True)

    job_config = {
        "dataset": args.dataset,
        "prepared_summary": str(args.prepared_summary),
        "method": args.method,
        "loss_preset": method.loss_preset,
        "seed": args.seed,
        "q_dim": args.q_dim if method.kind == "latent" else 0,
        "epochs": args.epochs,
        "cal_steps": args.cal_steps,
        "cal_init_mode": args.cal_init_mode,
        "cal_num_starts": args.cal_num_starts,
        "cal_selection_ratio": args.cal_selection_ratio,
        "cal_selection_min_rows": args.cal_selection_min_rows,
        "cal_refine_steps": args.cal_refine_steps,
        "cal_refine_only_after_selection": args.cal_refine_only_after_selection,
        "support_ratio": args.support_ratio,
        "support_split_mode": args.support_split_mode,
        "support_order_column": args.support_order_column,
        "batch_size": args.batch_size,
        "hidden_sizes": args.hidden_sizes,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
    }
    run_dir = (
        args.output_root
        / args.dataset
        / args.method
        / f"seed{args.seed}_q{job_config['q_dim']}_{stable_hash(job_config)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "success" and existing.get("job") == job_config:
            return result_path

    train_x = train_frame[feature_columns].to_numpy(np.float32)
    train_labels = train_frame["label"].to_numpy()
    train_y = train_frame["target"].to_numpy(np.float32)
    test_x = test_frame[feature_columns].to_numpy(np.float32)
    test_labels = test_frame["label"].to_numpy()
    test_y = test_frame["target"].to_numpy(np.float32)
    support_indices, query_indices = _support_query_indices(
        test_labels, args.support_ratio, args.seed, args.support_split_mode
    )
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    started = time.perf_counter()
    spatial: dict[str, float] = {}
    artifact_paths: dict[str, str] = {}
    optimization_counters: dict[str, int] = {}
    dynamic_weight_trace: list[dict[str, float]] = []
    latent_config_payload: dict[str, Any] | None = None
    test_metrics_payload: dict[str, Any] = {}

    if method.kind == "latent":
        config = _latent_config(args, method)
        latent_config_payload = asdict(config)
        train_dataset = build_dataset_from_arrays(
            train_x, train_labels, train_y, feature_names=feature_columns
        )
        test_dataset = build_dataset_from_arrays(
            test_x, test_labels, test_y, feature_names=feature_columns
        )
        training = train_latent_q_model(
            train_dataset, build_torch_model_factory(hidden_sizes), config
        )
        result = evaluate_latent_q_pipeline(
            train_dataset,
            test_dataset,
            training,
            config,
            output_config=OutputConfig(save_csv=False, save_plot=False),
        )
        test_metrics_payload = result.metrics
        predictions = result.eval_predictions
        truth = result.eval_targets
        query_labels = result.eval_labels
        query_indices = result.eval_indices
        q_columns = [column for column in result.test_output.columns if column.startswith("q")]
        q_by_label = result.test_output.groupby("label", sort=False)[q_columns].mean()
        query_frame = test_frame.iloc[query_indices].reset_index(drop=True)
        q_labels, response_signatures = grouped_rff_signatures(
            np.column_stack([query_frame[feature_columns].to_numpy(float), query_frame["target"].to_numpy(float)]),
            query_frame["label"].to_numpy(),
            n_components=64,
            seed=args.subsample_seed,
        )
        acquisition_labels, acquisition_signatures = grouped_rff_signatures(
            query_frame[feature_columns].to_numpy(float),
            query_frame["label"].to_numpy(),
            n_components=64,
            seed=args.subsample_seed + 1,
        )
        q_values = np.vstack([q_by_label.loc[label].to_numpy(float) for label in q_labels])
        acquisition_map = {
            label: acquisition_signatures[index] for index, label in enumerate(acquisition_labels)
        }
        acquisition_values = np.vstack([acquisition_map[label] for label in q_labels])
        curve = (
            neighborhood_preservation_curve(
                response_signatures, q_values, max_k=min(10, (len(q_labels) - 1) // 2)
            )
            if len(q_labels) >= 3
            else []
        )
        response_geometry = pairwise_distance_metrics(q_values, response_signatures)
        acquisition_geometry = pairwise_distance_metrics(q_values, acquisition_values)
        spatial = {
            "response_continuity_auc": float(np.mean([row["continuity"] for row in curve]))
            if curve else float("nan"),
            "response_trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in curve]))
            if curve else float("nan"),
            "response_knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve]))
            if curve else float("nan"),
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
        optimization_counters = asdict(training.optimization_counters)
        dynamic_weight_trace = training.dynamic_weight_trace
        if args.save_artifacts:
            artifact_paths = _save_latent_artifacts(
                run_dir=run_dir,
                test_frame=test_frame,
                query_indices=query_indices,
                predictions=predictions,
                q_labels=q_labels,
                q_values=q_values,
                response_signatures=response_signatures,
                curve=curve,
                feature_columns=feature_columns,
                train_output=result.train_output,
                training=training,
                job_config=job_config,
                hidden_sizes=hidden_sizes,
            )
    else:
        truth = test_y[query_indices]
        query_labels = test_labels[query_indices]
        if method.kind == "no_q_mlp":
            predictions = _run_no_q_mlp(
                train_x,
                train_labels,
                train_y,
                test_x[query_indices],
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                hidden_sizes=hidden_sizes,
            )
            optimizer_steps = args.epochs * int(np.ceil(len(train_x) / args.batch_size))
            optimization_counters = {
                "theta_steps": optimizer_steps,
                "q_steps": 0,
                "backward_passes": optimizer_steps,
                "examples_processed": args.epochs * len(train_x),
                "gradient_norm_trace": [],
            }
        elif method.kind == "random_forest":
            predictions = _run_random_forest(
                train_x, train_y, test_x[query_indices], seed=args.seed
            )
        else:
            predictions = _run_support_knn(
                test_x, test_y, test_labels, support_indices, query_indices
            )
        if args.save_artifacts:
            artifact_paths = _save_prediction_artifact(
                run_dir, test_frame, query_indices, predictions
            )

    prediction = _prediction_payload(truth, predictions, query_labels, train_y)
    payload = {
        "status": "success",
        "job": job_config,
        "dataset": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_labels": int(pd.Series(train_labels).nunique()),
            "test_labels": int(pd.Series(test_labels).nunique()),
            "support_rows": int(len(support_indices)),
            "query_rows": int(len(query_indices)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available() and method.kind in {"latent", "no_q_mlp"}
            else None,
        },
        "prediction": prediction,
        "spatial": spatial,
        "latent_config": latent_config_payload,
        "test_metrics": test_metrics_payload,
        "optimization_counters": optimization_counters,
        "dynamic_weight_trace": dynamic_weight_trace,
        "artifacts": artifact_paths,
        "wall_time_seconds": time.perf_counter() - started,
    }
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(result_path)
    return result_path


def _latent_config(args: argparse.Namespace, method: Method) -> LatentQConfig:
    common = dict(
        q_dim=args.q_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        calibration_steps=args.cal_steps,
        calibration_lr=0.05,
        calibration_ratio=args.support_ratio,
        calibration_split_mode=args.support_split_mode,
        calibration_init_mode=args.cal_init_mode,
        calibration_num_starts=args.cal_num_starts,
        calibration_selection_ratio=args.cal_selection_ratio,
        calibration_selection_min_rows=args.cal_selection_min_rows,
        calibration_refine_steps=args.cal_refine_steps,
        calibration_refine_only_after_selection=args.cal_refine_only_after_selection,
        seed=args.seed,
        device=args.device,
        verbose=False,
        early_stop_enabled=False,
        optimization_schedule=method.schedule,
        joint_steps_per_cycle=method.joint_steps_per_cycle,
        theta_steps_per_cycle=1,
        q_steps_per_cycle=1,
        loss_weighting=method.weighting,
        gradnorm_warmup_steps=5,
        gradnorm_interval=5,
        gradnorm_alpha=0.5,
        gradnorm_lr=0.025,
        gradnorm_min_weight=1e-3,
        gradnorm_max_weight=10.0,
        gradnorm_record_trace=method.weighting == "adaptive_loss_scale",
    )
    return LatentQConfig(**common, **get_loss_preset(method.loss_preset).config_kwargs())


def _run_no_q_mlp(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    hidden_sizes: tuple[int, ...],
) -> np.ndarray:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-8)
    target_mean = float(train_y.mean())
    target_std = max(float(train_y.std()), 1e-8)
    x_tensor = torch.tensor((train_x - feature_mean) / feature_std, dtype=torch.float32, device=resolved_device)
    y_tensor = torch.tensor((train_y - target_mean) / target_std, dtype=torch.float32, device=resolved_device)
    label_codes = pd.factorize(train_labels, sort=False)[0]
    label_tensor = torch.tensor(label_codes, dtype=torch.long, device=resolved_device)
    model = build_torch_model_factory(hidden_sizes)(train_x.shape[1]).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(train_x), batch_size):
            indices = permutation[start : start + batch_size].to(resolved_device)
            prediction = model(x_tensor[indices]).squeeze(1)
            losses = (prediction - y_tensor[indices]).pow(2)
            batch_labels = label_tensor[indices]
            loss = torch.stack(
                [losses[batch_labels == label].mean() for label in torch.unique(batch_labels)]
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    query_tensor = torch.tensor(
        (query_x - feature_mean) / feature_std, dtype=torch.float32, device=resolved_device
    )
    model.eval()
    with torch.no_grad():
        prediction = model(query_tensor).squeeze(1).cpu().numpy()
    return prediction * target_std + target_mean


def _run_random_forest(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        max_features=1.0,
        n_jobs=4,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    return model.predict(query_x)


def _run_support_knn(
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    support_indices: np.ndarray,
    query_indices: np.ndarray,
) -> np.ndarray:
    from sklearn.neighbors import KNeighborsRegressor

    scale = np.maximum(test_x[support_indices].std(axis=0), 1e-8)
    center = test_x[support_indices].mean(axis=0)
    prediction_by_index: dict[int, float] = {}
    support_set = set(support_indices.tolist())
    for label in pd.unique(test_labels):
        label_support = np.array(
            [index for index in np.flatnonzero(test_labels == label) if index in support_set], dtype=int
        )
        label_query = np.intersect1d(np.flatnonzero(test_labels == label), query_indices)
        neighbors = min(5, len(label_support))
        model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
        model.fit((test_x[label_support] - center) / scale, test_y[label_support])
        values = model.predict((test_x[label_query] - center) / scale)
        prediction_by_index.update(dict(zip(label_query.tolist(), values.tolist())))
    return np.asarray([prediction_by_index[int(index)] for index in query_indices], dtype=float)


def _save_prediction_artifact(
    run_dir: Path,
    test_frame: pd.DataFrame,
    query_indices: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, str]:
    output = test_frame.iloc[query_indices].copy()
    output["prediction"] = predictions
    path = run_dir / "query_predictions.csv"
    output.to_csv(path, index=False)
    return {"query_predictions": str(path)}


def _save_latent_artifacts(
    *,
    run_dir: Path,
    test_frame: pd.DataFrame,
    query_indices: np.ndarray,
    predictions: np.ndarray,
    q_labels: np.ndarray,
    q_values: np.ndarray,
    response_signatures: np.ndarray,
    curve: list[dict[str, float]],
    feature_columns: list[str],
    train_output: pd.DataFrame,
    training: Any,
    job_config: dict[str, Any],
    hidden_sizes: tuple[int, ...],
) -> dict[str, str]:
    paths = _save_prediction_artifact(run_dir, test_frame, query_indices, predictions)
    query_frame = test_frame.iloc[query_indices].copy()
    feature_means = query_frame.groupby("label", sort=False)[feature_columns].mean()
    q_frame = pd.DataFrame({"label": q_labels})
    for index in range(q_values.shape[1]):
        q_frame[f"q{index + 1}"] = q_values[:, index]
    response_centered = response_signatures - response_signatures.mean(axis=0)
    left, singular_values, _ = np.linalg.svd(response_centered, full_matrices=False)
    coordinates = left[:, :2] * singular_values[:2]
    q_frame["response_pc1"] = coordinates[:, 0]
    q_frame["response_pc2"] = coordinates[:, 1] if coordinates.shape[1] > 1 else 0.0
    for column in feature_columns:
        q_frame[f"feature_mean_{column}"] = [feature_means.loc[label, column] for label in q_labels]
    q_path = run_dir / "test_label_q.csv"
    train_q_path = run_dir / "train_label_q.csv"
    checkpoint_path = run_dir / "training_checkpoint.pt"
    curve_path = run_dir / "continuity_curve.csv"
    neighbor_path = run_dir / "q_nearest_neighbors.csv"
    plot_path = run_dir / "latent_response_geometry.png"
    q_frame.to_csv(q_path, index=False)
    train_output.groupby("label", sort=False)[
        [column for column in train_output if column.startswith("q")]
    ].mean().reset_index().to_csv(train_q_path, index=False)
    torch.save(
        {
            "job": job_config,
            "model_state_dict": training.model.state_dict(),
            "embedding_state_dict": training.embedding.state_dict(),
            "normalizer": asdict(training.normalizer),
            "label_to_index": training.label_to_index,
            "feature_columns": feature_columns,
            "hidden_sizes": hidden_sizes,
        },
        checkpoint_path,
    )
    pd.DataFrame(curve).to_csv(curve_path, index=False)
    _nearest_neighbor_table(q_labels, q_values, response_signatures).to_csv(neighbor_path, index=False)
    _plot_real_geometry(q_values, coordinates, curve, plot_path)
    paths.update(
        {
            "test_label_q": str(q_path),
            "train_label_q": str(train_q_path),
            "training_checkpoint": str(checkpoint_path),
            "continuity_curve": str(curve_path),
            "q_nearest_neighbors": str(neighbor_path),
            "latent_response_geometry_plot": str(plot_path),
        }
    )
    return paths


def _nearest_neighbor_table(
    labels: np.ndarray, q_values: np.ndarray, response_signatures: np.ndarray
) -> pd.DataFrame:
    q_distances = np.linalg.norm(q_values[:, None, :] - q_values[None, :, :], axis=2)
    response_distances = np.linalg.norm(
        response_signatures[:, None, :] - response_signatures[None, :, :], axis=2
    )
    np.fill_diagonal(q_distances, np.inf)
    rows = []
    for index, label in enumerate(labels):
        neighbor = int(np.argmin(q_distances[index]))
        rows.append(
            {
                "label": label,
                "q_nearest_label": labels[neighbor],
                "q_distance": q_distances[index, neighbor],
                "response_distance": response_distances[index, neighbor],
            }
        )
    return pd.DataFrame(rows)


def _plot_real_geometry(
    q_values: np.ndarray,
    response_coordinates: np.ndarray,
    curve: list[dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_centered = q_values - q_values.mean(axis=0)
    left, singular_values, _ = np.linalg.svd(q_centered, full_matrices=False)
    q_coordinates = left[:, :2] * singular_values[:2]
    if q_coordinates.shape[1] == 1:
        q_coordinates = np.column_stack([q_coordinates[:, 0], np.zeros(len(q_coordinates))])
    color = response_coordinates[:, 0]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].scatter(response_coordinates[:, 0], response_coordinates[:, 1], c=color, cmap="viridis")
    axes[0].set_title("held-out response signatures")
    axes[0].set_xlabel("response PC1")
    axes[0].set_ylabel("response PC2")
    axes[1].scatter(q_coordinates[:, 0], q_coordinates[:, 1], c=color, cmap="viridis")
    axes[1].set_title("learned q (PCA view)")
    axes[1].set_xlabel("q PC1")
    axes[1].set_ylabel("q PC2")
    k_values = [row["k"] for row in curve]
    axes[2].plot(k_values, [row["continuity"] for row in curve], "o-", label="continuity")
    axes[2].plot(k_values, [row["trustworthiness"] for row in curve], "s-", label="trustworthiness")
    axes[2].plot(k_values, [row["knn_overlap"] for row in curve], "^-", label="kNN overlap")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("neighborhood score")
    axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def launch(args: argparse.Namespace) -> None:
    selected = {value for value in args.datasets.split(",") if value}
    records: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in args.prepared_summary:
        for record in json.loads(summary_path.read_text(encoding="utf-8")):
            if not selected or record["name"] in selected:
                records.append((summary_path, record))
    methods = [value for value in args.methods.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    q_dims = [int(value) for value in args.q_dims.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    jobs = []
    for seed in seeds:
        for summary_path, record in records:
            for method_name in methods:
                dimensions = q_dims if METHODS[method_name].kind == "latent" else [0]
                for q_dim in dimensions:
                    reported_q_dim = q_dim if METHODS[method_name].kind == "latent" else 0
                    expected_job = {
                        "dataset": record["name"],
                        "prepared_summary": str(summary_path),
                        "method": method_name,
                        "loss_preset": METHODS[method_name].loss_preset,
                        "seed": seed,
                        "q_dim": reported_q_dim,
                        "epochs": args.epochs,
                        "cal_steps": args.cal_steps,
                        "cal_init_mode": args.cal_init_mode,
                        "cal_num_starts": args.cal_num_starts,
                        "cal_selection_ratio": args.cal_selection_ratio,
                        "cal_selection_min_rows": args.cal_selection_min_rows,
                        "cal_refine_steps": args.cal_refine_steps,
                        "cal_refine_only_after_selection": args.cal_refine_only_after_selection,
                        "support_ratio": args.support_ratio,
                        "support_split_mode": args.support_split_mode,
                        "support_order_column": args.support_order_column,
                        "batch_size": args.batch_size,
                        "hidden_sizes": args.hidden_sizes,
                        "max_train_per_label": args.max_train_per_label,
                        "max_test_per_label": args.max_test_per_label,
                        "subsample_seed": args.subsample_seed,
                    }
                    if args.resume and _has_successful_result(
                        args.output_root,
                        dataset=record["name"],
                        method=method_name,
                        seed=seed,
                        q_dim=reported_q_dim,
                        expected_job=expected_job,
                    ):
                        continue
                    jobs.append((summary_path, record["name"], method_name, seed, q_dim))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "group-held-out labels; "
            f"{args.support_split_mode} support/query split within each test label"
        ),
        "prepared_summaries": [str(path) for path in args.prepared_summary],
        "datasets": [record["name"] for _, record in records],
        "methods": methods,
        "seeds": seeds,
        "q_dims": q_dims,
        "gpus": gpus,
        "gpu_memory_threshold_mib": args.gpu_memory_threshold_mib,
        "dispatch_policy": "dispatch only below the memory threshold; run each job once; never auto-retry",
        "epochs": args.epochs,
        "calibration_steps": args.cal_steps,
        "calibration_strategy": {
            "init_mode": args.cal_init_mode,
            "num_starts": args.cal_num_starts,
            "selection_ratio": args.cal_selection_ratio,
            "selection_min_rows": args.cal_selection_min_rows,
            "refine_steps": args.cal_refine_steps,
            "refine_only_after_selection": args.cal_refine_only_after_selection,
        },
        "support_ratio": args.support_ratio,
        "support_split_mode": args.support_split_mode,
        "support_order_column": args.support_order_column,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps({**manifest, "planned_jobs": len(jobs)}, indent=2))
        return
    pending = list(jobs)
    running: list[tuple[subprocess.Popen[Any], tuple[Any, ...], Any, str]] = []
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            memory = _gpu_memory()
            available = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            available = [
                gpu
                for gpu in available
                if memory.get(gpu, 10**9) < args.gpu_memory_threshold_mib
            ]
            if not available:
                break
            job = pending.pop(0)
            summary_path, dataset, method_name, seed, q_dim = job
            gpu = available[0]
            log_path = args.output_root / "logs" / f"{dataset}_{method_name}_seed{seed}_q{q_dim}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON),
                str(Path(__file__).resolve()),
                "run-job",
                "--prepared-summary", str(summary_path),
                "--dataset", dataset,
                "--method", method_name,
                "--seed", str(seed),
                "--q-dim", str(max(q_dim, 1)),
                "--device", "cuda:0",
                "--output-root", str(args.output_root),
                "--epochs", str(args.epochs),
                "--cal-steps", str(args.cal_steps),
                "--cal-init-mode", args.cal_init_mode,
                "--cal-num-starts", str(args.cal_num_starts),
                "--cal-selection-ratio", str(args.cal_selection_ratio),
                "--cal-selection-min-rows", str(args.cal_selection_min_rows),
                "--cal-refine-steps", str(args.cal_refine_steps),
                "--cal-refine-only-after-selection"
                if args.cal_refine_only_after_selection
                else "--no-cal-refine-only-after-selection",
                "--support-ratio", str(args.support_ratio),
                "--support-split-mode", args.support_split_mode,
                "--support-order-column", args.support_order_column,
                "--batch-size", str(args.batch_size),
                "--hidden-sizes", args.hidden_sizes,
                "--max-train-per-label", str(args.max_train_per_label),
                "--max-test-per-label", str(args.max_test_per_label),
                "--subsample-seed", str(args.subsample_seed),
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update(
                {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
            )
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT
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
    summarize_results(args.output_root)


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


def summarize_results(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("*/*/seed*_q*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        rows.append(
            {
                **result["job"],
                **result["dataset"],
                **result["prediction"],
                **result["spatial"],
                "wall_time_seconds": result["wall_time_seconds"],
                "result_path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "all_runs.csv", index=False)
    if frame.empty:
        return
    metric_columns = [
        column
        for column in (
            "reference_nrmse",
            "macro_rmse",
            "macro_r2",
            "response_continuity_auc",
            "response_trustworthiness_auc",
            "response_distance_spearman",
            "response_local_log_distortion_p95",
            "acquisition_distance_spearman",
            "effective_rank",
            "wall_time_seconds",
        )
        if column in frame.columns
    ]
    summary = frame.groupby(["dataset", "method", "q_dim"], as_index=False)[metric_columns].agg(
        ["count", "mean", "std"]
    )
    summary.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(output_root / "method_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.command == "run-job":
        print(run_job(args))
    elif args.command == "launch":
        launch(args)
    else:
        summarize_results(args.output_root)


if __name__ == "__main__":
    main()
