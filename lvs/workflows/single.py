from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from lvs.core.expression_library import (
    GeneratedExpressionDataset,
    dataset_metadata,
    describe_expression_support,
    load_expression_library,
    sample_expression_dataset,
    save_generated_expression_dataset,
    select_expression_task,
)


DEFAULT_LIBRARY_CSV = Path(__file__).resolve().parents[2] / "data" / "latent_variable_expressions.csv"


@dataclass(frozen=True)
class WorkflowConfig:
    library_csv: Path
    expression_id: Optional[int]
    expression_name: Optional[str]
    label_count: int
    validation_label_count: int
    test_label_count: Optional[int]
    label_split_mode: str
    train_samples_per_label: int
    validation_samples_per_label: Optional[int]
    test_samples_per_label: int
    noise_std: float
    seed: int
    backend: str
    q_dim: int
    output_root: Path
    max_attempts_per_row: int
    epochs: int
    batch_size: int
    lr: float
    auto_target_scale: bool
    target_scale_min_magnitude: float
    target_scale_desired_magnitude: float
    cal_steps: int
    cal_lr: float
    cal_ratio: float
    calibration_split_mode: str
    early_stop_enabled: bool
    early_stop_r2_threshold: float
    early_stop_patience: int
    latent_feature_orthogonality_weight: float
    latent_feature_orthogonality_type: str
    latent_feature_stats_mode: str
    latent_curve_continuity_weight: float
    latent_curve_continuity_grid_size: int
    calibration_q_prior_weight: float
    latent_q_l2_weight: float
    prediction_loss_type: str
    latent_q_whitening_weight: float
    latent_jacobian_disentanglement_weight: float
    latent_q_canonicalization_mode: str
    latent_q_smoothness_weight: float
    latent_q_smoothness_epsilon: float
    optimization_schedule: str
    theta_lr: Optional[float]
    q_lr: Optional[float]
    theta_steps_per_cycle: int
    q_steps_per_cycle: int
    loss_weighting: str
    gradnorm_warmup_steps: int
    gradnorm_interval: int
    gradnorm_alpha: float
    gradnorm_lr: float
    gradnorm_min_weight: float
    gradnorm_max_weight: float
    gradnorm_record_trace: bool
    device: Optional[str]
    quiet: bool
    hidden_sizes: str
    kan_grid: int
    kan_order: int


@dataclass
class WorkflowResult:
    run_dir: Path
    generated_paths: dict[str, Path]
    latent_q_paths: dict[str, Path]
    metrics: dict[str, Any]
    q_dim_model: int
    ground_truth_latent_dim: int


@dataclass(frozen=True)
class TargetScalingDecision:
    applied: bool
    scale_factor: float
    representative_magnitude: float
    min_magnitude_threshold: float
    desired_magnitude: float
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end latent-variable workflow: select a benchmark expression, "
            "generate data, train latent q, calibrate unseen labels, and evaluate predictions."
        )
    )
    add_workflow_arguments(parser)
    return parser


def add_workflow_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_expression_selection: bool = True,
    include_list_expressions: bool = True,
) -> argparse.ArgumentParser:
    parser.add_argument("--library-csv", type=Path, default=DEFAULT_LIBRARY_CSV)
    if include_list_expressions:
        parser.add_argument("--list-expressions", action="store_true")
    if include_expression_selection:
        parser.add_argument("--expression-id", type=int, default=None)
        parser.add_argument("--expression-name", type=str, default=None)
    parser.add_argument("--label-count", type=int, default=50)
    parser.add_argument("--validation-label-count", type=int, default=0)
    parser.add_argument("--test-label-count", type=int, default=None)
    parser.add_argument("--label-split-mode", choices=("shared", "disjoint"), default="shared")
    parser.add_argument("--train-samples-per-label", type=int, default=80)
    parser.add_argument("--validation-samples-per-label", type=int, default=None)
    parser.add_argument("--test-samples-per-label", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("torch", "kan"), default="torch")
    parser.add_argument("--q-dim", "--q_dim", dest="q_dim", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--max-attempts-per-row", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--auto-target-scale", dest="auto_target_scale", action="store_true", default=True)
    parser.add_argument("--disable-auto-target-scale", dest="auto_target_scale", action="store_false")
    parser.add_argument("--target-scale-min-magnitude", type=float, default=1e-3)
    parser.add_argument("--target-scale-desired-magnitude", type=float, default=1.0)
    parser.add_argument("--cal-steps", type=int, default=4000)
    parser.add_argument("--cal-lr", type=float, default=0.10)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument(
        "--calibration-split-mode",
        choices=("prefix", "random"),
        default="prefix",
        help="How each test label is divided into calibration support and evaluation query rows.",
    )
    parser.add_argument("--early-stop", dest="early_stop_enabled", action="store_true", default=True)
    parser.add_argument("--disable-early-stop", dest="early_stop_enabled", action="store_false")
    parser.add_argument("--early-stop-r2-threshold", type=float, default=0.999)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--latent-feature-orthogonality-weight", type=float, default=0.0)
    parser.add_argument(
        "--latent-feature-orthogonality-type",
        choices=("pearson", "hsic", "nhsic", "distance_correlation", "adversarial", "propensity"),
        default="pearson",
    )
    parser.add_argument(
        "--latent-feature-stats-mode",
        choices=("mean_std", "rich", "rff_kme", "rich_rff_kme"),
        default="mean_std",
    )
    parser.add_argument("--latent-curve-continuity-weight", type=float, default=0.0)
    parser.add_argument("--latent-curve-continuity-grid-size", type=int, default=64)
    parser.add_argument("--calibration-q-prior-weight", type=float, default=0.0)
    parser.add_argument("--latent-q-l2-weight", type=float, default=0.0)
    parser.add_argument("--prediction-loss-type", choices=("mse", "label_balanced_mse"), default="mse")
    parser.add_argument("--latent-q-whitening-weight", type=float, default=0.0)
    parser.add_argument("--latent-jacobian-disentanglement-weight", type=float, default=0.0)
    parser.add_argument(
        "--latent-q-canonicalization-mode",
        choices=("none", "output", "train"),
        default="none",
    )
    parser.add_argument("--latent-q-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--latent-q-smoothness-epsilon", type=float, default=0.05)
    parser.add_argument("--optimization-schedule", choices=("joint", "alternating"), default="joint")
    parser.add_argument("--theta-lr", type=float, default=None)
    parser.add_argument("--q-lr", type=float, default=None)
    parser.add_argument("--theta-steps-per-cycle", type=int, default=1)
    parser.add_argument("--q-steps-per-cycle", type=int, default=1)
    parser.add_argument("--loss-weighting", choices=("static", "gradnorm"), default="static")
    parser.add_argument("--gradnorm-warmup-steps", type=int, default=0)
    parser.add_argument("--gradnorm-interval", type=int, default=1)
    parser.add_argument("--gradnorm-alpha", type=float, default=0.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument("--gradnorm-min-weight", type=float, default=1e-3)
    parser.add_argument("--gradnorm-max-weight", type=float, default=1e3)
    parser.add_argument("--gradnorm-record-trace", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hidden-sizes", type=str, default="128,64")
    parser.add_argument("--kan-grid", type=int, default=5)
    parser.add_argument("--kan-order", type=int, default=3)
    return parser


def parse_hidden_sizes(raw_value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    hidden_sizes = tuple(int(part) for part in parts if part)
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty.")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("Each hidden layer size must be a positive integer.")
    return hidden_sizes


def namespace_to_workflow_config(args: argparse.Namespace) -> WorkflowConfig:
    return WorkflowConfig(
        library_csv=args.library_csv,
        expression_id=args.expression_id,
        expression_name=args.expression_name,
        label_count=args.label_count,
        validation_label_count=args.validation_label_count,
        test_label_count=args.test_label_count,
        label_split_mode=args.label_split_mode,
        train_samples_per_label=args.train_samples_per_label,
        validation_samples_per_label=args.validation_samples_per_label,
        test_samples_per_label=args.test_samples_per_label,
        noise_std=args.noise_std,
        seed=args.seed,
        backend=args.backend,
        q_dim=args.q_dim,
        output_root=args.output_root,
        max_attempts_per_row=args.max_attempts_per_row,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        auto_target_scale=args.auto_target_scale,
        target_scale_min_magnitude=args.target_scale_min_magnitude,
        target_scale_desired_magnitude=args.target_scale_desired_magnitude,
        cal_steps=args.cal_steps,
        cal_lr=args.cal_lr,
        cal_ratio=args.cal_ratio,
        calibration_split_mode=args.calibration_split_mode,
        early_stop_enabled=args.early_stop_enabled,
        early_stop_r2_threshold=args.early_stop_r2_threshold,
        early_stop_patience=args.early_stop_patience,
        latent_feature_orthogonality_weight=args.latent_feature_orthogonality_weight,
        latent_feature_orthogonality_type=args.latent_feature_orthogonality_type,
        latent_feature_stats_mode=args.latent_feature_stats_mode,
        latent_curve_continuity_weight=args.latent_curve_continuity_weight,
        latent_curve_continuity_grid_size=args.latent_curve_continuity_grid_size,
        calibration_q_prior_weight=args.calibration_q_prior_weight,
        latent_q_l2_weight=args.latent_q_l2_weight,
        prediction_loss_type=args.prediction_loss_type,
        latent_q_whitening_weight=args.latent_q_whitening_weight,
        latent_jacobian_disentanglement_weight=args.latent_jacobian_disentanglement_weight,
        latent_q_canonicalization_mode=args.latent_q_canonicalization_mode,
        latent_q_smoothness_weight=args.latent_q_smoothness_weight,
        latent_q_smoothness_epsilon=args.latent_q_smoothness_epsilon,
        optimization_schedule=args.optimization_schedule,
        theta_lr=args.theta_lr,
        q_lr=args.q_lr,
        theta_steps_per_cycle=args.theta_steps_per_cycle,
        q_steps_per_cycle=args.q_steps_per_cycle,
        loss_weighting=args.loss_weighting,
        gradnorm_warmup_steps=args.gradnorm_warmup_steps,
        gradnorm_interval=args.gradnorm_interval,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        gradnorm_min_weight=args.gradnorm_min_weight,
        gradnorm_max_weight=args.gradnorm_max_weight,
        gradnorm_record_trace=args.gradnorm_record_trace,
        device=args.device,
        quiet=args.quiet,
        hidden_sizes=args.hidden_sizes,
        kan_grid=args.kan_grid,
        kan_order=args.kan_order,
    )


def list_expressions(library_csv: Path | str) -> None:
    records = load_expression_library(library_csv)
    descriptions = describe_expression_support(records)
    for item in descriptions:
        status = item["status"]
        if status == "supported":
            print(
                f"[supported] id={item['expression_id']:>2} "
                f"name={item['formula_name']} "
                f"x={item['observed_feature_variables']} "
                f"q={item['latent_variables']}"
            )
        else:
            print(
                f"[unsupported] id={item['expression_id']:>2} "
                f"name={item['formula_name']} "
                f"reason={item['reason']}"
            )


def create_run_dir(config: WorkflowConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expr_token = f"expr{config.expression_id:03d}" if config.expression_id is not None else "exprname"
    base_name = f"{timestamp}_{expr_token}_{config.backend}_qdim{config.q_dim}"
    run_dir = config.output_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = config.output_root / f"{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def ensure_run_structure(run_dir: Path) -> dict[str, Path]:
    paths = {
        "configs": run_dir / "configs",
        "artifacts": run_dir / "artifacts",
        "data": run_dir / "artifacts" / "data",
        "latent_q": run_dir / "artifacts" / "latent_q",
        "logs": run_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_pipeline_config(config: WorkflowConfig):
    from lvs.core.pipeline import LatentQConfig

    return LatentQConfig(
        q_dim=config.q_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        calibration_steps=config.cal_steps,
        calibration_lr=config.cal_lr,
        calibration_ratio=config.cal_ratio,
        calibration_split_mode=config.calibration_split_mode,
        seed=config.seed,
        early_stop_enabled=config.early_stop_enabled,
        early_stop_r2_threshold=config.early_stop_r2_threshold,
        early_stop_patience=config.early_stop_patience,
        latent_feature_orthogonality_weight=config.latent_feature_orthogonality_weight,
        latent_feature_orthogonality_type=config.latent_feature_orthogonality_type,
        latent_feature_stats_mode=config.latent_feature_stats_mode,
        latent_curve_continuity_weight=config.latent_curve_continuity_weight,
        latent_curve_continuity_grid_size=config.latent_curve_continuity_grid_size,
        calibration_q_prior_weight=config.calibration_q_prior_weight,
        latent_q_l2_weight=config.latent_q_l2_weight,
        prediction_loss_type=config.prediction_loss_type,
        latent_q_whitening_weight=config.latent_q_whitening_weight,
        latent_jacobian_disentanglement_weight=config.latent_jacobian_disentanglement_weight,
        latent_q_canonicalization_mode=config.latent_q_canonicalization_mode,
        latent_q_smoothness_weight=config.latent_q_smoothness_weight,
        latent_q_smoothness_epsilon=config.latent_q_smoothness_epsilon,
        optimization_schedule=config.optimization_schedule,
        theta_lr=config.theta_lr,
        q_lr=config.q_lr,
        theta_steps_per_cycle=config.theta_steps_per_cycle,
        q_steps_per_cycle=config.q_steps_per_cycle,
        loss_weighting=config.loss_weighting,
        gradnorm_warmup_steps=config.gradnorm_warmup_steps,
        gradnorm_interval=config.gradnorm_interval,
        gradnorm_alpha=config.gradnorm_alpha,
        gradnorm_lr=config.gradnorm_lr,
        gradnorm_min_weight=config.gradnorm_min_weight,
        gradnorm_max_weight=config.gradnorm_max_weight,
        gradnorm_record_trace=config.gradnorm_record_trace,
        device=config.device,
        verbose=not config.quiet,
    )


def detect_target_scaling(
    train_targets: np.ndarray,
    test_targets_or_config: np.ndarray | WorkflowConfig,
    config: WorkflowConfig | None = None,
) -> TargetScalingDecision:
    """Fit target-scaling decisions strictly from training targets.

    The three-argument form remains accepted for callers on the legacy API, but
    the test targets are deliberately ignored.
    """
    resolved_config = test_targets_or_config if config is None else config
    train_values = np.asarray(train_targets, dtype=np.float64).reshape(-1)
    representative_magnitude = float(
        max(
            float(np.max(np.abs(train_values))) if train_values.size else 0.0,
            float(np.std(train_values)) if train_values.size else 0.0,
        )
    )
    config = resolved_config
    if not config.auto_target_scale:
        return TargetScalingDecision(
            applied=False,
            scale_factor=1.0,
            representative_magnitude=representative_magnitude,
            min_magnitude_threshold=config.target_scale_min_magnitude,
            desired_magnitude=config.target_scale_desired_magnitude,
            reason="auto_target_scale_disabled",
        )
    if config.target_scale_min_magnitude <= 0:
        raise ValueError("target_scale_min_magnitude must be positive.")
    if config.target_scale_desired_magnitude <= 0:
        raise ValueError("target_scale_desired_magnitude must be positive.")
    if representative_magnitude <= 0:
        return TargetScalingDecision(
            applied=False,
            scale_factor=1.0,
            representative_magnitude=representative_magnitude,
            min_magnitude_threshold=config.target_scale_min_magnitude,
            desired_magnitude=config.target_scale_desired_magnitude,
            reason="non_positive_representative_magnitude",
        )
    if representative_magnitude >= config.target_scale_min_magnitude:
        return TargetScalingDecision(
            applied=False,
            scale_factor=1.0,
            representative_magnitude=representative_magnitude,
            min_magnitude_threshold=config.target_scale_min_magnitude,
            desired_magnitude=config.target_scale_desired_magnitude,
            reason="target_magnitude_already_reasonable",
        )

    exponent = int(math.ceil(math.log10(config.target_scale_desired_magnitude / representative_magnitude)))
    scale_factor = float(10.0 ** max(0, exponent))
    return TargetScalingDecision(
        applied=not np.isclose(scale_factor, 1.0),
        scale_factor=scale_factor,
        representative_magnitude=representative_magnitude,
        min_magnitude_threshold=config.target_scale_min_magnitude,
        desired_magnitude=config.target_scale_desired_magnitude,
        reason="scaled_small_targets_for_regression",
    )


def build_output_config(config: WorkflowConfig, latent_q_dir: Path):
    from lvs.core.pipeline import OutputConfig

    default_plot_name = (
        "fit_vs_real_latent_q_torch.png"
        if config.backend == "torch"
        else "fit_vs_real_latent_q_kan.png"
    )
    return OutputConfig(
        output_dir=latent_q_dir,
        train_output_name="train_with_q.csv",
        test_output_name="test_with_q.csv",
        plot_output_name=default_plot_name,
        save_csv=True,
        save_plot=True,
        plot_feature_index=0,
        plot_title=f"Fit vs Real Curve ({config.backend.upper()}, q_dim={config.q_dim})",
    )


def run_latent_q_stage(
    generated_dataset: GeneratedExpressionDataset,
    config: WorkflowConfig,
    latent_q_dir: Path,
    target_scaling: TargetScalingDecision,
):
    from lvs.core.pipeline import build_dataset_from_arrays, scale_dataset_targets

    pipeline_config = build_pipeline_config(config)
    output_config = build_output_config(config, latent_q_dir)
    feature_names = generated_dataset.task.feature_columns

    train_dataset = build_dataset_from_arrays(
        features=generated_dataset.train_frame.loc[:, list(feature_names)].to_numpy(),
        labels=generated_dataset.train_frame["label"].to_numpy(),
        targets=generated_dataset.train_frame["target"].to_numpy(),
        feature_names=feature_names,
        label_name="label",
        target_name="target",
    )
    test_dataset = build_dataset_from_arrays(
        features=generated_dataset.test_frame.loc[:, list(feature_names)].to_numpy(),
        labels=generated_dataset.test_frame["label"].to_numpy(),
        targets=generated_dataset.test_frame["target"].to_numpy(),
        feature_names=feature_names,
        label_name="label",
        target_name="target",
    )
    if target_scaling.applied:
        train_dataset = scale_dataset_targets(train_dataset, target_scaling.scale_factor)
        test_dataset = scale_dataset_targets(test_dataset, target_scaling.scale_factor)

    if config.backend == "torch":
        from lvs.backends.torch_mlp import run_torch_latent_q

        return run_torch_latent_q(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=pipeline_config,
            output_config=output_config,
            hidden_sizes=parse_hidden_sizes(config.hidden_sizes),
        )

    from lvs.backends.kan import run_kan_latent_q

    return run_kan_latent_q(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=pipeline_config,
        output_config=output_config,
        kan_grid=config.kan_grid,
        kan_order=config.kan_order,
    )


def summarize_workflow(
    run_dir: Path,
    config: WorkflowConfig,
    generated_dataset: GeneratedExpressionDataset,
    generated_paths: dict[str, Path],
    latent_q_result,
    target_scaling: TargetScalingDecision,
) -> Path:
    summary = {
        "status": "success",
        "run_dir": str(run_dir),
        "expression": dataset_metadata(generated_dataset),
        "workflow_config": workflow_config_to_json(config),
        "target_scaling": asdict(target_scaling),
        "latent_q": {
            "backend": config.backend,
            "q_dim_model": int(config.q_dim),
            "ground_truth_latent_dim": int(generated_dataset.ground_truth_latent_dim),
            "metrics": dict(latent_q_result.metrics),
            "saved_paths": {key: str(value) for key, value in latent_q_result.saved_paths.items()},
        },
        "generated_paths": {key: str(value) for key, value in generated_paths.items()},
    }
    return write_json(run_dir / "run_summary.json", summary)


def workflow_config_to_json(config: WorkflowConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def write_failure_summary(run_dir: Path, config: Optional[WorkflowConfig], stage: str, error_message: str) -> None:
    summary = {
        "status": "failed",
        "stage": stage,
        "error": error_message,
        "workflow_config": None if config is None else workflow_config_to_json(config),
    }
    write_json(run_dir / "run_summary.json", summary)


def generate_expression_dataset(task, config: WorkflowConfig) -> GeneratedExpressionDataset:
    """Generate data using the split protocol recorded in the workflow config."""
    return sample_expression_dataset(
        task,
        label_count=config.label_count,
        validation_label_count=config.validation_label_count,
        test_label_count=config.test_label_count,
        label_split_mode=config.label_split_mode,
        train_samples_per_label=config.train_samples_per_label,
        validation_samples_per_label=config.validation_samples_per_label,
        test_samples_per_label=config.test_samples_per_label,
        noise_std=config.noise_std,
        seed=config.seed,
        max_attempts_per_row=config.max_attempts_per_row,
    )


def run_workflow(config: WorkflowConfig) -> WorkflowResult:
    run_dir = create_run_dir(config)
    paths = ensure_run_structure(run_dir)

    try:
        write_json(paths["configs"] / "workflow_config.json", workflow_config_to_json(config))

        records = load_expression_library(config.library_csv)
        task = select_expression_task(
            records,
            expression_id=config.expression_id,
            formula_name=config.expression_name,
        )
        write_json(paths["configs"] / "expression_task.json", {
            "expression_id": task.expression_id,
            "formula_name": task.formula_name,
            "raw_formula": task.raw_formula,
            "normalized_formula": task.normalized_formula,
            "rhs_expression": task.rhs_expression,
            "observed_feature_variables": list(task.observed_feature_variables),
            "feature_columns": list(task.feature_columns),
            "latent_variables": list(task.latent_variables),
            "ground_truth_latent_dim": int(task.ground_truth_latent_dim),
            "variable_mapping": dict(task.variable_mapping),
            "variable_ranges": {
                key: [value[0], value[1]] for key, value in task.variable_ranges.items()
            },
        })

        generated_dataset = generate_expression_dataset(task, config)
        generated_paths = save_generated_expression_dataset(
            generated_dataset,
            paths["data"],
            train_filename="generated_train.csv",
            test_filename="generated_test.csv",
            latent_truth_filename="latent_truth.csv",
            metadata_filename="expression_metadata.json",
            include_header=True,
        )

        target_scaling = detect_target_scaling(
            generated_dataset.train_frame["target"].to_numpy(),
            config,
        )

        latent_q_result = run_latent_q_stage(
            generated_dataset,
            config,
            paths["latent_q"],
            target_scaling,
        )

        summarize_workflow(
            run_dir=run_dir,
            config=config,
            generated_dataset=generated_dataset,
            generated_paths=generated_paths,
            latent_q_result=latent_q_result,
            target_scaling=target_scaling,
        )

        return WorkflowResult(
            run_dir=run_dir,
            generated_paths=generated_paths,
            latent_q_paths=dict(latent_q_result.saved_paths),
            metrics=dict(latent_q_result.metrics),
            q_dim_model=config.q_dim,
            ground_truth_latent_dim=generated_dataset.ground_truth_latent_dim,
        )
    except Exception as exc:
        write_failure_summary(run_dir, config, stage="workflow", error_message=str(exc))
        raise


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.list_expressions:
        list_expressions(args.library_csv)
        return

    config = namespace_to_workflow_config(args)
    result = run_workflow(config)
    print(f"Run directory: {result.run_dir}")
    print(f"q_dim_model: {result.q_dim_model}")
    print(f"ground_truth_latent_dim: {result.ground_truth_latent_dim}")
    for metric_name, metric_value in result.metrics.items():
        print(f"{metric_name}: {metric_value}")
    for name, path in result.generated_paths.items():
        print(f"Saved {name}: {path}")
    for name, path in result.latent_q_paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
