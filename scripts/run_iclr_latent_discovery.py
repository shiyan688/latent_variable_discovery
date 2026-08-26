#!/usr/bin/env python3
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

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.expression_library import load_expression_library, sample_expression_dataset, select_expression_task
from lvs.core.loss_presets import LOSS_SWEEP_METHOD_PRESETS, get_loss_preset
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
    OutputConfig,
    build_dataset_from_arrays,
    evaluate_latent_q_pipeline,
    split_support_query_indices,
    train_latent_q_model,
)

PYTHON = Path(sys.executable)
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_latent_discovery_pilot"


@dataclass(frozen=True)
class Method:
    name: str
    schedule: str = "joint"
    weighting: str = "static"
    loss_preset: str = "mse"
    kind: str = "latent"
    joint_steps_per_cycle: int = 2
    q_scale_constraint: str = "none"
    record_gradient_norms: bool = False
    q_lr_multiplier: float = 1.0
    q_canonicalization_mode: str = "none"


METHODS = {
    method.name: method
    for method in (
        Method("joint_mse"),
        Method("joint_mse_step1", joint_steps_per_cycle=1),
        Method("alternating_mse", schedule="alternating"),
        Method("joint_lb_mse", loss_preset="label_balanced_mse"),
        Method("joint_hsic", loss_preset="hsic"),
        Method("joint_continuity", loss_preset="continuity"),
        Method(
            "joint_continuity_step1",
            loss_preset="continuity",
            joint_steps_per_cycle=1,
        ),
        Method("joint_q_l2", loss_preset="q_l2"),
        Method("joint_calprior", loss_preset="calibration_prior"),
        Method("joint_hsic_cont", loss_preset="hsic_continuity"),
        Method("joint_all_mse", loss_preset="all_mse"),
        Method("joint_fixed", loss_preset="all_label_balanced"),
        Method(
            "alternating_fixed",
            schedule="alternating",
            loss_preset="all_label_balanced",
        ),
        Method(
            "joint_dynamic",
            weighting="adaptive_loss_scale",
            loss_preset="all_label_balanced",
        ),
        Method(
            "alternating_dynamic",
            schedule="alternating",
            weighting="adaptive_loss_scale",
            loss_preset="all_label_balanced",
        ),
        # Latent-recovery ablations (2026-08-12). Baselines above are unchanged; each
        # variant below alters exactly one mechanism relative to joint_continuity so
        # the substitution-recovery comparison stays interpretable.
        Method("joint_continuity_gradlog", loss_preset="continuity", record_gradient_norms=True),
        Method("joint_mse_gradlog", record_gradient_norms=True),
        Method("joint_continuity_fixednorm", loss_preset="continuity", q_scale_constraint="fixed_norm"),
        Method("joint_mse_fixednorm", q_scale_constraint="fixed_norm"),
        # Full affine quotient: substitution recovery needs only a linear f, so the
        # drift is an affine map, not just a global scale. This arm reuses the
        # existing train-time centering+whitening projection.
        Method(
            "joint_continuity_affinequot",
            loss_preset="continuity",
            q_canonicalization_mode="train",
        ),
        Method(
            "alternating_continuity_qlr10",
            schedule="alternating",
            loss_preset="continuity",
            q_lr_multiplier=10.0,
        ),
        Method(
            "alternating_continuity_qlr1",
            schedule="alternating",
            loss_preset="continuity",
        ),
        Method("no_q_mlp", kind="no_q_mlp"),
        Method("random_forest", kind="random_forest"),
        Method("support_knn", kind="support_knn"),
        Method("oracle_q_mlp", kind="oracle_q_mlp"),
        *(Method(name, loss_preset=preset) for name, preset in LOSS_SWEEP_METHOD_PRESETS.items()),
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
    "oracle_q_mlp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICLR latent-discovery pilot with disjoint labels and latent metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run-job")
    run.add_argument("--expression-id", type=int, required=True)
    run.add_argument("--method", choices=tuple(METHODS), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Dataset seed. Defaults to --seed for backward-compatible resampling.",
    )
    run.add_argument("--q-dim", type=int, default=None)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    run.add_argument("--epochs", type=int, default=300)
    run.add_argument("--cal-steps", type=int, default=300)
    _add_calibration_strategy_arguments(run)
    run.add_argument("--train-labels", type=int, default=24)
    run.add_argument("--validation-labels", type=int, default=8)
    run.add_argument("--test-labels", type=int, default=8)
    run.add_argument("--samples-per-label", type=int, default=60)
    run.add_argument("--support-ratio", type=float, default=0.3)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--expression-ids", default="3,15,21,36,41,48")
    launch.add_argument("--methods", default=",".join(PRIMARY_METHODS))
    launch.add_argument("--seeds", default="13,37,73")
    launch.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Fix this seed across model seeds to enable representation-stability analysis.",
    )
    launch.add_argument("--gpus", default="1,2")
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    launch.add_argument("--epochs", type=int, default=300)
    launch.add_argument("--cal-steps", type=int, default=300)
    _add_calibration_strategy_arguments(launch)
    launch.add_argument("--train-labels", type=int, default=24)
    launch.add_argument("--validation-labels", type=int, default=8)
    launch.add_argument("--test-labels", type=int, default=8)
    launch.add_argument("--samples-per-label", type=int, default=60)
    launch.add_argument("--support-ratio", type=float, default=0.3)
    launch.add_argument("--batch-size", type=int, default=256)
    launch.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    launch.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _add_calibration_strategy_arguments(parser: argparse.ArgumentParser) -> None:
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


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _has_successful_result(
    output_root: Path,
    *,
    expression_id: int,
    method: str,
    seed: int,
    expected_job: dict[str, Any],
) -> bool:
    """Return whether an exactly matching job already finished successfully."""
    pattern = f"expr{expression_id:03d}/{method}/seed{seed}_*/result.json"
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


def run_job(args: argparse.Namespace) -> Path:
    if args.validation_labels < 2:
        raise ValueError("validation_labels must be at least 2 for held-out alignment.")
    if args.test_labels < 2:
        raise ValueError("test_labels must be at least 2 for spatial metrics.")
    if args.samples_per_label < 2:
        raise ValueError("samples_per_label must be at least 2 for support/query evaluation.")
    method = METHODS[args.method]
    task = select_expression_task(load_expression_library(PROJECT_ROOT / "data" / "latent_variable_expressions.csv"), expression_id=args.expression_id)
    q_dim = task.ground_truth_latent_dim if args.q_dim is None else args.q_dim
    reported_q_dim = q_dim if method.kind in {"latent", "oracle_q_mlp"} else 0
    data_seed = args.seed if args.data_seed is None else args.data_seed
    job_config = {
        "expression_id": args.expression_id,
        "method": args.method,
        "loss_preset": method.loss_preset,
        "seed": args.seed,
        "data_seed": data_seed,
        "q_dim": reported_q_dim,
        "epochs": args.epochs,
        "cal_steps": args.cal_steps,
        "cal_init_mode": args.cal_init_mode,
        "cal_num_starts": args.cal_num_starts,
        "cal_selection_ratio": args.cal_selection_ratio,
        "cal_selection_min_rows": args.cal_selection_min_rows,
        "cal_refine_steps": args.cal_refine_steps,
        "cal_refine_only_after_selection": args.cal_refine_only_after_selection,
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "samples_per_label": args.samples_per_label,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
    }
    run_dir = args.output_root / f"expr{args.expression_id:03d}" / args.method / f"seed{args.seed}_{stable_hash(job_config)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "success" and existing.get("job") == job_config:
            return result_path

    started = time.perf_counter()
    generated = sample_expression_dataset(
        task,
        label_count=args.train_labels,
        validation_label_count=args.validation_labels,
        test_label_count=args.test_labels,
        train_samples_per_label=args.samples_per_label,
        validation_samples_per_label=args.samples_per_label,
        test_samples_per_label=args.samples_per_label,
        label_split_mode="disjoint",
        seed=data_seed,
    )
    assert generated.validation_frame is not None
    features = list(task.feature_columns)

    def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            frame.loc[:, features].to_numpy(np.float32),
            frame["label"].to_numpy(),
            frame["target"].to_numpy(np.float32),
        )

    train_x, train_labels, train_y = arrays(generated.train_frame)
    val_x, val_labels, val_y = arrays(generated.validation_frame)
    test_x, test_labels, test_y = arrays(generated.test_frame)
    if method.kind != "latent":
        support_indices, query_indices = _support_query_indices(
            test_labels, args.support_ratio, args.seed
        )
        query_x = test_x[query_indices]
        if method.kind == "no_q_mlp":
            predictions = _run_no_q_mlp(
                train_x,
                train_labels,
                train_y,
                query_x,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                hidden_sizes=(128, 64),
            )
        elif method.kind == "random_forest":
            predictions = _run_random_forest(train_x, train_y, query_x, seed=args.seed)
        elif method.kind == "support_knn":
            predictions = _run_support_knn(
                test_x, test_y, test_labels, support_indices, query_indices
            )
        else:
            truth_frame = generated.latent_truth_frame
            train_truth = truth_frame[truth_frame["split"] == "train"].set_index("label")
            test_truth = truth_frame[truth_frame["split"] == "test"].set_index("label")
            latent_columns = [column for column in truth_frame.columns if column.startswith("q")]
            train_oracle_q = np.vstack(
                [train_truth.loc[label, latent_columns].to_numpy(float) for label in train_labels]
            )
            query_oracle_q = np.vstack(
                [test_truth.loc[label, latent_columns].to_numpy(float) for label in test_labels[query_indices]]
            )
            predictions = _run_no_q_mlp(
                np.column_stack([train_x, train_oracle_q]),
                train_labels,
                train_y,
                np.column_stack([query_x, query_oracle_q]),
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                hidden_sizes=(128, 64),
            )
        prediction = {
            **macro_prediction_metrics(test_y[query_indices], predictions, test_labels[query_indices]),
            **reference_scaled_prediction_metrics(
                test_y[query_indices], predictions, reference_scale=float(np.std(train_y))
            ),
        }
        artifact_paths: dict[str, str] = {}
        if args.save_artifacts:
            query_output = generated.test_frame.iloc[query_indices].copy()
            query_output["prediction"] = predictions
            prediction_path = run_dir / "query_predictions.csv"
            query_output.to_csv(prediction_path, index=False)
            artifact_paths["query_predictions"] = str(prediction_path)
        payload = {
            "status": "success",
            "job": job_config,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(torch.device(args.device))
                if torch.cuda.is_available() and method.kind in {"no_q_mlp", "oracle_q_mlp"}
                else None,
            },
            "prediction": prediction,
            "spatial": {},
            "latent_config": None,
            "artifacts": artifact_paths,
            "optimization_counters": {},
            "dynamic_weight_trace": [],
            "wall_time_seconds": time.perf_counter() - started,
        }
        temporary_path = result_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(result_path)
        return result_path
    common = dict(
        q_dim=q_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        calibration_steps=args.cal_steps,
        calibration_lr=0.05,
        calibration_ratio=args.support_ratio,
        calibration_split_mode="random",
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
        q_scale_constraint=method.q_scale_constraint,
        record_gradient_norms=method.record_gradient_norms,
        latent_q_canonicalization_mode=method.q_canonicalization_mode,
    )
    if method.q_lr_multiplier != 1.0:
        # Alternating schedules own separate theta/q optimizers, so a q-specific
        # learning rate is only meaningful there.
        common["theta_lr"] = common["lr"]
        common["q_lr"] = common["lr"] * method.q_lr_multiplier
    config = LatentQConfig(**common, **get_loss_preset(method.loss_preset).config_kwargs())

    output_config = OutputConfig(save_csv=False, save_plot=False)
    train_dataset = build_dataset_from_arrays(
        train_x, train_labels, train_y, feature_names=features
    )
    validation_dataset = build_dataset_from_arrays(
        val_x, val_labels, val_y, feature_names=features
    )
    test_dataset = build_dataset_from_arrays(
        test_x, test_labels, test_y, feature_names=features
    )
    training_artifacts = train_latent_q_model(
        train_dataset,
        build_torch_model_factory((128, 64)),
        config,
    )
    validation = evaluate_latent_q_pipeline(
        train_dataset,
        validation_dataset,
        training_artifacts,
        config,
        output_config=output_config,
    )
    testing = evaluate_latent_q_pipeline(
        train_dataset,
        test_dataset,
        training_artifacts,
        config,
        output_config=output_config,
    )

    truth = generated.latent_truth_frame
    latent_columns = [column for column in truth.columns if column.startswith("q")]

    def learned_by_label(
        frame: pd.DataFrame,
        q_frame: pd.DataFrame,
        split: str,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str]]:
        q_columns = [column for column in q_frame.columns if column.startswith("q")]
        learned_columns = [f"learned_{column}" for column in q_columns]
        learned = q_frame.groupby("label")[q_columns].mean().reset_index()
        learned = learned.rename(columns=dict(zip(q_columns, learned_columns)))
        reference = truth[truth["split"] == split][["label", *latent_columns]]
        merged = learned.merge(reference, on="label", how="inner")
        return (
            merged[learned_columns].to_numpy(float),
            merged[latent_columns].to_numpy(float),
            merged,
            learned_columns,
            latent_columns,
        )

    val_learned, val_truth, val_q_frame, _, _ = learned_by_label(
        generated.validation_frame, validation.test_output, "validation"
    )
    test_learned, test_truth, test_q_frame, learned_columns, truth_columns = learned_by_label(
        generated.test_frame, testing.test_output, "test"
    )
    alignment = fit_affine_alignment(val_learned, val_truth)
    aligned_test = apply_affine_alignment(alignment, test_learned)
    cca_alignment = fit_cca_alignment(val_learned, val_truth)
    cca = score_cca_alignment(cca_alignment, test_learned, test_truth)
    continuity_curve = neighborhood_preservation_curve(
        test_truth, test_learned, max_k=min(10, (len(test_truth) - 1) // 2)
    )
    continuity_summary = {
        "trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in continuity_curve])),
        "continuity_auc": float(np.mean([row["continuity"] for row in continuity_curve])),
        "knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in continuity_curve])),
    }
    spatial = {
        **alignment_metrics(test_truth, aligned_test),
        **pairwise_distance_metrics(test_learned, test_truth),
        **continuity_summary,
        **local_distance_distortion(test_truth, test_learned, k=min(5, len(test_truth) - 1)),
        "cca_mean": float(cca.mean()) if cca.size else float("nan"),
        "knn_overlap": knn_overlap(test_learned, test_truth, k=min(3, len(test_truth) - 1)),
        "effective_rank": effective_rank(test_learned),
    }
    prediction = {
        **macro_prediction_metrics(testing.eval_targets, testing.eval_predictions, testing.eval_labels),
        **reference_scaled_prediction_metrics(
            testing.eval_targets,
            testing.eval_predictions,
            reference_scale=float(np.std(train_y)),
        ),
    }
    artifact_paths: dict[str, str] = {}
    if args.save_artifacts:
        for index in range(aligned_test.shape[1]):
            test_q_frame[f"aligned_q{index + 1}"] = aligned_test[:, index]
        test_q_path = run_dir / "test_label_q.csv"
        validation_q_path = run_dir / "validation_label_q.csv"
        continuity_path = run_dir / "continuity_curve.csv"
        test_q_frame.to_csv(test_q_path, index=False)
        val_q_frame.to_csv(validation_q_path, index=False)
        pd.DataFrame(continuity_curve).to_csv(continuity_path, index=False)
        geometry_plot_path = run_dir / "latent_geometry.png"
        _plot_latent_geometry(
            test_truth,
            test_learned,
            aligned_test,
            continuity_curve,
            geometry_plot_path,
            title=f"expr{args.expression_id:03d} {args.method} seed={args.seed}",
        )
        artifact_paths = {
            "test_label_q": str(test_q_path),
            "validation_label_q": str(validation_q_path),
            "continuity_curve": str(continuity_path),
            "latent_geometry_plot": str(geometry_plot_path),
        }
    payload = {
        "status": "success",
        "job": job_config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)) if torch.cuda.is_available() else None,
        },
        "validation_metrics": validation.metrics,
        "test_metrics": testing.metrics,
        "prediction": prediction,
        "spatial": spatial,
        "latent_config": asdict(config),
        "artifacts": artifact_paths,
        "optimization_counters": asdict(testing.training_artifacts.optimization_counters),
        "dynamic_weight_trace": testing.training_artifacts.dynamic_weight_trace,
        "wall_time_seconds": time.perf_counter() - started,
    }
    temp_result_path = result_path.with_suffix(".json.tmp")
    temp_result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_result_path.replace(result_path)
    return result_path


def launch(args: argparse.Namespace) -> None:
    expression_ids = [int(value) for value in args.expression_ids.split(",") if value]
    methods = [value for value in args.methods.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    gpus = [value for value in args.gpus.split(",") if value]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "disjoint train/validation/test labels; support/query split within held-out labels",
        "expression_ids": expression_ids,
        "methods": methods,
        "method_configs": {name: asdict(METHODS[name]) for name in methods},
        "model_and_support_seeds": seeds,
        "data_seed": args.data_seed,
        "gpus": gpus,
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
        "train_labels": args.train_labels,
        "validation_labels": args.validation_labels,
        "test_labels": args.test_labels,
        "samples_per_label": args.samples_per_label,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "save_artifacts": args.save_artifacts,
    }
    (args.output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    jobs = []
    for seed in seeds:
        for expression_id in expression_ids:
            for method in methods:
                expected_job = {
                    "expression_id": expression_id,
                    "method": method,
                    "loss_preset": METHODS[method].loss_preset,
                    "seed": seed,
                    "data_seed": seed if args.data_seed is None else args.data_seed,
                    "epochs": args.epochs,
                    "cal_steps": args.cal_steps,
                    "cal_init_mode": args.cal_init_mode,
                    "cal_num_starts": args.cal_num_starts,
                    "cal_selection_ratio": args.cal_selection_ratio,
                    "cal_selection_min_rows": args.cal_selection_min_rows,
                    "cal_refine_steps": args.cal_refine_steps,
                    "cal_refine_only_after_selection": args.cal_refine_only_after_selection,
                    "train_labels": args.train_labels,
                    "validation_labels": args.validation_labels,
                    "test_labels": args.test_labels,
                    "samples_per_label": args.samples_per_label,
                    "support_ratio": args.support_ratio,
                    "batch_size": args.batch_size,
                }
                if args.resume and _has_successful_result(
                    args.output_root,
                    expression_id=expression_id,
                    method=method,
                    seed=seed,
                    expected_job=expected_job,
                ):
                    continue
                jobs.append((expression_id, method, seed))
    running: list[tuple[subprocess.Popen[Any], tuple[int, str, int], Any, str]] = []
    pending = list(jobs)
    status_path = args.output_root / "launcher_status.jsonl"
    while pending or running:
        while pending and len(running) < len(gpus):
            job = pending.pop(0)
            expression_id, method, seed = job
            available_gpus = [gpu for gpu in gpus if gpu not in {entry[3] for entry in running}]
            if not available_gpus:
                break
            gpu = available_gpus[0]
            log_path = args.output_root / "logs" / f"expr{expression_id:03d}_{method}_seed{seed}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON), str(Path(__file__).resolve()), "run-job",
                "--expression-id", str(expression_id), "--method", method, "--seed", str(seed),
                "--device", "cuda:0", "--output-root", str(args.output_root),
                "--epochs", str(args.epochs), "--cal-steps", str(args.cal_steps),
                "--cal-init-mode", args.cal_init_mode,
                "--cal-num-starts", str(args.cal_num_starts),
                "--cal-selection-ratio", str(args.cal_selection_ratio),
                "--cal-selection-min-rows", str(args.cal_selection_min_rows),
                "--cal-refine-steps", str(args.cal_refine_steps),
                "--cal-refine-only-after-selection"
                if args.cal_refine_only_after_selection
                else "--no-cal-refine-only-after-selection",
                "--train-labels", str(args.train_labels), "--validation-labels", str(args.validation_labels),
                "--test-labels", str(args.test_labels), "--samples-per-label", str(args.samples_per_label),
                "--support-ratio", str(args.support_ratio), "--batch-size", str(args.batch_size),
                "--resume" if args.resume else "--no-resume",
                "--save-artifacts" if args.save_artifacts else "--no-save-artifacts",
            ]
            if args.data_seed is not None:
                command.extend(["--data-seed", str(args.data_seed)])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            running.append((process, job, handle, gpu))
        next_running = []
        for process, job, handle, gpu in running:
            code = process.poll()
            if code is None:
                next_running.append((process, job, handle, gpu))
                continue
            handle.close()
            with status_path.open("a", encoding="utf-8") as status:
                status.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "job": job, "returncode": code}) + "\n")
        running = next_running
        time.sleep(2)
    summarize_results(args.output_root)


def summarize_results(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("expr*/**/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row = {
            **result["job"],
            **result["prediction"],
            **result["spatial"],
            "wall_time_seconds": result["wall_time_seconds"],
            "result_path": str(path),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "all_runs.csv", index=False)
    if frame.empty:
        return
    summary = frame.groupby(["expression_id", "method"], as_index=False).agg(
        runs=("seed", "count"),
        macro_nrmse=("macro_nrmse", "mean"),
        macro_nrmse_std=("macro_nrmse", "std"),
        macro_rmse=("macro_rmse", "mean"),
        macro_rmse_std=("macro_rmse", "std"),
        aligned_nrmse=("aligned_nrmse", "mean"),
        aligned_nrmse_std=("aligned_nrmse", "std"),
        distance_stress=("distance_stress", "mean"),
        distance_stress_std=("distance_stress", "std"),
        continuity_auc=("continuity_auc", "mean"),
        continuity_auc_std=("continuity_auc", "std"),
        trustworthiness_auc=("trustworthiness_auc", "mean"),
        trustworthiness_auc_std=("trustworthiness_auc", "std"),
        local_log_distortion_p95=("local_log_distortion_p95", "mean"),
        local_log_distortion_p95_std=("local_log_distortion_p95", "std"),
        reference_nrmse=("reference_nrmse", "mean"),
        reference_nrmse_std=("reference_nrmse", "std"),
        cca_mean=("cca_mean", "mean"),
        cca_mean_std=("cca_mean", "std"),
        wall_time_seconds=("wall_time_seconds", "mean"),
    )
    summary.to_csv(output_root / "method_summary.csv", index=False)


def _plot_latent_geometry(
    truth: np.ndarray,
    learned: np.ndarray,
    aligned: np.ndarray,
    continuity_curve: list[dict[str, float]],
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    if truth.shape[1] == 1:
        order = np.argsort(truth[:, 0])
        axes[0].plot(truth[order, 0], aligned[order, 0], "o-", markersize=3)
        axes[0].set_xlabel("true q")
        axes[0].set_ylabel("validation-aligned learned q")
    else:
        color = truth[:, 0]
        axes[0].scatter(truth[:, 0], truth[:, 1], c=color, cmap="viridis", label="true", marker="o")
        axes[0].scatter(aligned[:, 0], aligned[:, 1], c=color, cmap="viridis", label="aligned", marker="x")
        axes[0].set_xlabel("q1")
        axes[0].set_ylabel("q2")
        axes[0].legend(frameon=False)
    true_distances = _upper_pairwise_distances(truth)
    learned_distances = _upper_pairwise_distances(learned)
    axes[1].scatter(true_distances, learned_distances, s=10, alpha=0.6)
    axes[1].set_xlabel("true pairwise distance")
    axes[1].set_ylabel("learned pairwise distance")
    k_values = [row["k"] for row in continuity_curve]
    axes[2].plot(k_values, [row["continuity"] for row in continuity_curve], "o-", label="continuity")
    axes[2].plot(k_values, [row["trustworthiness"] for row in continuity_curve], "s-", label="trustworthiness")
    axes[2].plot(k_values, [row["knn_overlap"] for row in continuity_curve], "^-", label="kNN overlap")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("neighborhood score")
    axes[2].legend(frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _upper_pairwise_distances(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    distances = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    return distances[np.triu_indices(array.shape[0], k=1)]


def _support_query_indices(
    labels: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    support_parts = []
    query_parts = []
    for label in pd.unique(labels):
        support, query = split_support_query_indices(
            np.flatnonzero(labels == label), ratio, mode="random", seed=seed, label=label
        )
        support_parts.append(support)
        query_parts.append(query)
    return np.concatenate(support_parts), np.concatenate(query_parts)


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
    x_tensor = torch.tensor(
        (train_x - feature_mean) / feature_std, dtype=torch.float32, device=resolved_device
    )
    y_tensor = torch.tensor(
        (train_y - target_mean) / target_std, dtype=torch.float32, device=resolved_device
    )
    label_tensor = torch.tensor(
        pd.factorize(train_labels, sort=False)[0], dtype=torch.long, device=resolved_device
    )
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
    model.eval()
    query_tensor = torch.tensor(
        (query_x - feature_mean) / feature_std, dtype=torch.float32, device=resolved_device
    )
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
    support_set = set(support_indices.tolist())
    prediction_by_index: dict[int, float] = {}
    for label in pd.unique(test_labels):
        label_support = np.asarray(
            [index for index in np.flatnonzero(test_labels == label) if index in support_set],
            dtype=int,
        )
        label_query = np.intersect1d(np.flatnonzero(test_labels == label), query_indices)
        model = KNeighborsRegressor(
            n_neighbors=min(5, len(label_support)), weights="distance"
        )
        model.fit((test_x[label_support] - center) / scale, test_y[label_support])
        values = model.predict((test_x[label_query] - center) / scale)
        prediction_by_index.update(dict(zip(label_query.tolist(), values.tolist())))
    return np.asarray([prediction_by_index[int(index)] for index in query_indices], dtype=float)


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
