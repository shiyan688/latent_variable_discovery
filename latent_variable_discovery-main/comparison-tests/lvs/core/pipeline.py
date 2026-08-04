from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = Path(
    os.environ.get(
        "MPLCONFIGDIR",
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "latent-variable-discovery"
        / "matplotlib",
    )
)
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score

EPSILON = 1e-8
ModelFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class CSVColumnConfig:
    feature_cols: tuple[int, ...] = (1,)
    label_col: int = 0
    target_col: int = -1
    has_header: bool = False


@dataclass(frozen=True)
class LatentQConfig:
    q_dim: int = 3
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    calibration_steps: int = 200
    calibration_lr: float = 0.05
    calibration_ratio: float = 0.3
    seed: int = 42
    device: Optional[str] = None
    verbose: bool = True
    early_stop_enabled: bool = True
    early_stop_r2_threshold: float = 0.999
    early_stop_patience: int = 10
    latent_feature_orthogonality_weight: float = 0.0
    latent_feature_orthogonality_type: str = "pearson"
    latent_feature_stats_mode: str = "mean_std"
    latent_curve_continuity_weight: float = 0.0
    latent_curve_continuity_grid_size: int = 64
    calibration_q_prior_weight: float = 0.0
    latent_q_l2_weight: float = 0.0
    prediction_loss_type: str = "mse"
    latent_q_whitening_weight: float = 0.0
    latent_jacobian_disentanglement_weight: float = 0.0
    latent_q_canonicalization_mode: str = "none"
    latent_q_smoothness_weight: float = 0.0
    latent_q_smoothness_epsilon: float = 0.05
    mds_init_enabled: bool = True
    mds_fixed_q_epochs: int = 200
    mds_joint_finetune_lr_ratio: float = 0.1
    mds_joint_finetune_epochs: int = 300
    mds_position_preservation_weight: float = 0.1
    mds_q_regularization_weight: float = 0.0


@dataclass(frozen=True)
class OutputConfig:
    output_dir: Path = Path(".")
    train_output_name: str = "train_with_q.csv"
    test_output_name: str = "test_with_q.csv"
    plot_output_name: Optional[str] = None
    save_csv: bool = True
    save_plot: bool = True
    plot_feature_index: Optional[int] = None
    plot_title: str = "Fit vs Real Curve"


@dataclass(frozen=True)
class LatentQDataset:
    features: np.ndarray
    labels: np.ndarray
    targets: np.ndarray
    targets_original: np.ndarray
    target_scale_factor: float
    feature_names: tuple[str, ...]
    label_name: str
    target_name: str
    base_output_frame: pd.DataFrame


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    r2: float
    mse: float
    mse_original: float


@dataclass(frozen=True)
class NormalizationStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float


@dataclass
class TrainingArtifacts:
    model: nn.Module
    embedding: nn.Embedding
    normalizer: NormalizationStats
    label_to_index: dict[Any, int]
    device: torch.device
    train_history: list[EpochMetrics]
    early_stopped: bool = False
    early_stop_epoch: Optional[int] = None


@dataclass(frozen=True)
class CalibrationArtifacts:
    q_by_label: dict[Any, np.ndarray]
    eval_predictions: np.ndarray
    eval_targets: np.ndarray
    eval_plot_axis: np.ndarray
    eval_indices: np.ndarray


@dataclass
class LatentQPipelineResult:
    training_artifacts: TrainingArtifacts
    train_output: pd.DataFrame
    test_output: pd.DataFrame
    train_q_matrix: np.ndarray
    test_q_matrix: np.ndarray
    eval_predictions: np.ndarray
    eval_targets: np.ndarray
    eval_plot_axis: np.ndarray
    eval_indices: np.ndarray
    plot_feature_name: str
    metrics: dict[str, Any]
    saved_paths: dict[str, Path] = field(default_factory=dict)


def parse_index_list(raw_value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw_value.split(",")]
    indices = tuple(int(part) for part in parts if part)
    if not indices:
        raise ValueError("feature_cols cannot be empty.")
    return indices


def add_common_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_plot_output_name: str,
    default_plot_title: str,
) -> None:
    parser.add_argument("--train-csv", type=Path, default=Path("train.csv"), help="Training CSV path.")
    parser.add_argument("--test-csv", type=Path, default=Path("test.csv"), help="Test CSV path.")
    parser.add_argument(
        "--csv-has-header",
        action="store_true",
        help="Set this flag if the input CSV files include a header row.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default="1",
        help="Comma-separated observed-feature indices. The label column is excluded by default.",
    )
    parser.add_argument("--label-col", type=int, default=0, help="Label column index.")
    parser.add_argument("--target-col", type=int, default=-1, help="Target column index.")
    parser.add_argument(
        "--q-dim",
        "--q_dim",
        "--k",
        dest="q_dim",
        type=int,
        default=2,
        help="Latent q dimension.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Training learning rate.")
    parser.add_argument(
        "--cal-steps",
        type=int,
        default=200,
        help="Calibration steps used to estimate q on the test set.",
    )
    parser.add_argument("--cal-lr", type=float, default=0.05, help="Calibration learning rate.")
    parser.add_argument(
        "--cal-ratio",
        type=float,
        default=0.3,
        help="Per-label calibration split ratio on the test set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--early-stop",
        dest="early_stop_enabled",
        action="store_true",
        default=True,
        help="Enable training early stopping when train metrics stay above the target thresholds.",
    )
    parser.add_argument(
        "--disable-early-stop",
        dest="early_stop_enabled",
        action="store_false",
        help="Disable training early stopping.",
    )
    parser.add_argument(
        "--early-stop-r2-threshold",
        type=float,
        default=0.999,
        help="Early stop condition: train R2 must stay above this threshold.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Number of consecutive qualifying epochs required before early stop triggers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Example: cpu, cuda, cuda:0. Defaults to auto detection.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for optional output files.")
    parser.add_argument(
        "--train-output-name",
        type=str,
        default="train_with_q.csv",
        help="Output CSV name for the training set with q values.",
    )
    parser.add_argument(
        "--test-output-name",
        type=str,
        default="test_with_q.csv",
        help="Output CSV name for the test set with q values.",
    )
    parser.add_argument(
        "--plot-output-name",
        type=str,
        default=default_plot_output_name,
        help="Output plot filename.",
    )
    parser.add_argument("--plot-title", type=str, default=default_plot_title, help="Prediction curve plot title.")
    parser.add_argument(
        "--plot-feature-index",
        type=int,
        default=None,
        help="Feature index used for sorting the prediction curve. Defaults to 1 or 0 depending on feature count.",
    )
    parser.add_argument("--skip-save", action="store_true", help="Skip writing train/test CSV outputs.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip writing the prediction curve plot.")
    parser.add_argument("--quiet", action="store_true", help="Disable epoch-level logging.")
    parser.add_argument(
        "--latent-feature-orthogonality-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a label-level decorrelation penalty between learned q values and observed-feature "
            "distribution statistics. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--latent-feature-orthogonality-type",
        choices=("pearson", "hsic", "nhsic", "distance_correlation", "adversarial", "propensity"),
        default="pearson",
        help=(
            "Orthogonality/dependence penalty used when latent-feature orthogonality weight is positive. "
            "pearson is the original squared-correlation penalty."
        ),
    )
    parser.add_argument(
        "--latent-feature-stats-mode",
        choices=("mean_std", "rich", "rff_kme", "rich_rff_kme"),
        default="mean_std",
        help=(
            "Label-level acquisition-distribution embedding A_l used by the orthogonality penalty. "
            "mean_std is the original [mean, std]; rich adds min/max/range/quantiles/covariance; "
            "rff_kme uses multi-scale random Fourier feature kernel mean embeddings."
        ),
    )
    parser.add_argument(
        "--latent-curve-continuity-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a label-level continuity penalty matching pairwise latent-q distances to "
            "pairwise response-curve distances. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--latent-curve-continuity-grid-size",
        type=int,
        default=64,
        help="Grid size used to summarize each label response curve for continuity loss.",
    )
    parser.add_argument(
        "--calibration-q-prior-weight",
        type=float,
        default=0.0,
        help="Weight for keeping calibrated test q near the training q distribution. Use 0 to disable.",
    )
    parser.add_argument(
        "--latent-q-l2-weight",
        type=float,
        default=0.0,
        help="Weight for an L2 penalty on train label latent descriptors Q.",
    )
    parser.add_argument(
        "--prediction-loss-type",
        choices=("mse", "label_balanced_mse"),
        default="mse",
        help="Training prediction loss. label_balanced_mse averages MSE within each label before averaging labels.",
    )
    parser.add_argument(
        "--latent-q-whitening-weight",
        type=float,
        default=0.0,
        help="Weight for enforcing centered, unit-variance, decorrelated latent q coordinates.",
    )
    parser.add_argument(
        "--latent-jacobian-disentanglement-weight",
        type=float,
        default=0.0,
        help="Weight for decorrelating df/dq directions across q dimensions.",
    )
    parser.add_argument(
        "--latent-q-canonicalization-mode",
        choices=("none", "output", "train"),
        default="none",
        help=(
            "Canonicalize learned q coordinates. output applies a train-q whitening transform to saved q values; "
            "train additionally projects train embeddings after optimizer steps."
        ),
    )
    parser.add_argument(
        "--latent-q-smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for a finite-difference curvature penalty that keeps f(x,q) simple in q.",
    )
    parser.add_argument(
        "--latent-q-smoothness-epsilon",
        type=float,
        default=0.05,
        help="Finite-difference step used by --latent-q-smoothness-weight.",
    )
    parser.add_argument(
        "--mds-init",
        action="store_true",
        default=True,
        help="Enable MDS initialization of Q from curve shape distances.",
    )
    parser.add_argument(
        "--no-mds-init",
        action="store_false",
        dest="mds_init",
        help="Disable MDS initialization and use standard random initialization.",
    )
    parser.add_argument(
        "--mds-fixed-q-epochs",
        type=int,
        default=200,
        help="Number of epochs to train θ with fixed Q during MDS initialization phase.",
    )
    parser.add_argument(
        "--mds-joint-finetune-epochs",
        type=int,
        default=300,
        help="Number of epochs for joint fine-tuning of θ and Q.",
    )
    parser.add_argument(
        "--mds-joint-finetune-lr-ratio",
        type=float,
        default=0.1,
        help="Learning rate ratio for joint fine-tuning phase (multiplied by main lr).",
    )
    parser.add_argument(
        "--mds-q-regularization-weight",
        type=float,
        default=0.0,
        help="Weight for L2 regularization to keep Q close to MDS initialization during joint fine-tuning. Use 0 to disable.",
    )


def namespace_to_shared_configs(args: argparse.Namespace) -> tuple[CSVColumnConfig, LatentQConfig, OutputConfig]:
    column_config = CSVColumnConfig(
        feature_cols=parse_index_list(args.feature_cols),
        label_col=args.label_col,
        target_col=args.target_col,
        has_header=args.csv_has_header,
    )
    pipeline_config = LatentQConfig(
        q_dim=args.q_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        calibration_steps=args.cal_steps,
        calibration_lr=args.cal_lr,
        calibration_ratio=args.cal_ratio,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
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
        mds_init_enabled=args.mds_init,
        mds_fixed_q_epochs=args.mds_fixed_q_epochs,
        mds_joint_finetune_epochs=args.mds_joint_finetune_epochs,
        mds_joint_finetune_lr_ratio=args.mds_joint_finetune_lr_ratio,
        mds_q_regularization_weight=args.mds_q_regularization_weight,
    )
    output_config = OutputConfig(
        output_dir=Path(args.output_dir),
        train_output_name=args.train_output_name,
        test_output_name=args.test_output_name,
        plot_output_name=args.plot_output_name,
        save_csv=not args.skip_save,
        save_plot=not args.skip_plot,
        plot_feature_index=args.plot_feature_index,
        plot_title=args.plot_title,
    )
    return column_config, pipeline_config, output_config


def load_csv_dataset(csv_path: Path | str, column_config: CSVColumnConfig) -> LatentQDataset:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    header = 0 if column_config.has_header else None
    frame = pd.read_csv(path, header=header)
    return build_dataset_from_dataframe(frame, column_config)


def build_dataset_from_dataframe(frame: pd.DataFrame, column_config: CSVColumnConfig) -> LatentQDataset:
    if frame.empty:
        raise ValueError("Input dataframe cannot be empty.")

    n_columns = frame.shape[1]
    feature_cols = tuple(_resolve_index(index, n_columns) for index in column_config.feature_cols)
    label_col = _resolve_index(column_config.label_col, n_columns)
    target_col = _resolve_index(column_config.target_col, n_columns)

    features = frame.iloc[:, list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    labels = frame.iloc[:, label_col].to_numpy(copy=True)
    targets = frame.iloc[:, target_col].to_numpy(dtype=np.float32, copy=True)

    feature_names = tuple(_normalize_column_name(frame.columns[index], index) for index in feature_cols)
    label_name = _normalize_column_name(frame.columns[label_col], label_col)
    target_name = _normalize_column_name(frame.columns[target_col], target_col)

    ordered_indices: list[int] = []
    for index in (label_col, *feature_cols, target_col):
        if index not in ordered_indices:
            ordered_indices.append(index)
    ordered_names = [_normalize_column_name(frame.columns[index], index) for index in ordered_indices]
    base_output_frame = frame.iloc[:, ordered_indices].copy()
    base_output_frame.columns = ordered_names

    return _build_dataset(
        features,
        labels,
        targets,
        targets,
        1.0,
        feature_names,
        label_name,
        target_name,
        base_output_frame,
    )


def build_dataset_from_arrays(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    label_name: str = "label",
    target_name: str = "target",
) -> LatentQDataset:
    feature_array, label_array, target_array = _coerce_dataset_arrays(features, labels, targets)

    if feature_names is None:
        resolved_feature_names = tuple(f"feature_{index}" for index in range(feature_array.shape[1]))
    else:
        resolved_feature_names = tuple(feature_names)
        if len(resolved_feature_names) != feature_array.shape[1]:
            raise ValueError("feature_names length must match the feature dimension.")

    base_output_frame = pd.DataFrame(feature_array, columns=resolved_feature_names)
    base_output_frame.insert(0, label_name, label_array)
    base_output_frame[target_name] = target_array

    return _build_dataset(
        feature_array,
        label_array,
        target_array,
        target_array,
        1.0,
        resolved_feature_names,
        label_name,
        target_name,
        base_output_frame,
    )


def train_latent_q_model(
    train_dataset: LatentQDataset,
    model_factory: ModelFactory,
    config: LatentQConfig,
) -> TrainingArtifacts:
    validated_config = _validate_config(config)
    _validate_matching_feature_dimensions(train_dataset, train_dataset)
    _set_random_seed(validated_config.seed)
    device = _resolve_device(validated_config.device)

    input_dim = train_dataset.features.shape[1] + validated_config.q_dim
    model = model_factory(input_dim)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module instance.")
    model = model.to(device)

    normalizer = fit_normalization(train_dataset.features, train_dataset.targets)
    train_features_normalized = normalize_features(train_dataset.features, normalizer)
    train_targets_normalized = normalize_targets(train_dataset.targets, normalizer)

    unique_labels = [_normalize_label_value(label) for label in pd.unique(train_dataset.labels)]
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    indexed_labels = np.asarray(
        [label_to_index[_normalize_label_value(label)] for label in train_dataset.labels],
        dtype=np.int64,
    )

    feature_tensor = torch.tensor(train_features_normalized, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(train_targets_normalized.reshape(-1, 1), dtype=torch.float32, device=device)
    label_tensor = torch.tensor(indexed_labels, dtype=torch.long, device=device)
    label_feature_stats_tensor = _compute_label_feature_stats(
        feature_tensor,
        label_tensor,
        label_count=len(unique_labels),
        mode=validated_config.latent_feature_stats_mode,
    )
    label_curve_distance_tensor = _compute_label_curve_distance_matrix(
        feature_tensor,
        target_tensor.squeeze(1),
        label_tensor,
        label_count=len(unique_labels),
        grid_size=validated_config.latent_curve_continuity_grid_size,
    )

    embedding = nn.Embedding(len(unique_labels), validated_config.q_dim).to(device)
    if validated_config.mds_init_enabled:
        mds_q = _mds_init_from_distances(label_curve_distance_tensor, validated_config.q_dim)
        embedding.weight.data.copy_(mds_q)
        if validated_config.verbose:
            print(f"MDS initialization completed. Q shape: {mds_q.shape}")
    else:
        nn.init.normal_(embedding.weight, mean=0.0, std=0.1)

    orth_adversary: Optional[nn.Module] = None
    orth_adversary_optimizer: Optional[optim.Optimizer] = None
    orth_adversary_targets: Optional[torch.Tensor] = None
    if (
        validated_config.latent_feature_orthogonality_weight > 0
        and validated_config.latent_feature_orthogonality_type == "adversarial"
    ):
        adversary_hidden = max(16, min(128, validated_config.q_dim * 16))
        orth_adversary = nn.Sequential(
            nn.Linear(validated_config.q_dim, adversary_hidden),
            nn.ReLU(),
            nn.Linear(adversary_hidden, label_feature_stats_tensor.shape[1]),
        ).to(device)
        orth_adversary_optimizer = optim.Adam(orth_adversary.parameters(), lr=validated_config.lr)
        orth_adversary_targets = _standardize_columns(label_feature_stats_tensor).detach()
    mse_loss = nn.MSELoss()

    history: list[EpochMetrics] = []
    sample_count = feature_tensor.shape[0]
    batch_size = min(validated_config.batch_size, sample_count)
    early_stop_streak = 0
    early_stopped = False
    early_stop_epoch: Optional[int] = None

    def _train_epoch(
        model: nn.Module,
        embedding: nn.Embedding,
        optimizer: optim.Optimizer,
        epoch_num: int,
        phase_name: str = "",
        initial_q_weight: Optional[torch.Tensor] = None,
    ) -> EpochMetrics:
        model.train()
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, batch_size):
            selection = permutation[start : start + batch_size]
            batch_features = feature_tensor[selection]
            batch_targets = target_tensor[selection]
            batch_labels = label_tensor[selection]

            if orth_adversary is not None and orth_adversary_optimizer is not None and orth_adversary_targets is not None:
                orth_adversary.train()
                orth_adversary_optimizer.zero_grad()
                detached_q = _standardize_columns(embedding.weight.detach())
                adversary_predictions = orth_adversary(detached_q)
                adversary_loss = mse_loss(adversary_predictions, orth_adversary_targets)
                adversary_loss.backward()
                orth_adversary_optimizer.step()

            q_batch = embedding(batch_labels)
            model_inputs = torch.cat([batch_features, q_batch], dim=1)
            predictions = _ensure_prediction_column(model(model_inputs))
            loss = _prediction_loss(
                predictions,
                batch_targets,
                batch_labels,
                loss_type=validated_config.prediction_loss_type,
            )
            if validated_config.latent_feature_orthogonality_weight > 0:
                loss = loss + validated_config.latent_feature_orthogonality_weight * _latent_feature_orthogonality_penalty(
                    embedding.weight,
                    label_feature_stats_tensor,
                    penalty_type=validated_config.latent_feature_orthogonality_type,
                    adversary=orth_adversary,
                    adversary_targets=orth_adversary_targets,
                )
            if validated_config.latent_curve_continuity_weight > 0:
                loss = loss + validated_config.latent_curve_continuity_weight * _latent_curve_continuity_penalty(
                    embedding.weight,
                    label_curve_distance_tensor,
                )
            if validated_config.latent_q_l2_weight > 0:
                loss = loss + validated_config.latent_q_l2_weight * embedding.weight.pow(2).mean()
            if validated_config.latent_q_whitening_weight > 0:
                loss = loss + validated_config.latent_q_whitening_weight * _latent_q_whitening_penalty(embedding.weight)
            if validated_config.latent_jacobian_disentanglement_weight > 0:
                loss = (
                    loss
                    + validated_config.latent_jacobian_disentanglement_weight
                    * _latent_jacobian_disentanglement_penalty(model, batch_features, q_batch)
                )
            if validated_config.latent_q_smoothness_weight > 0:
                loss = loss + validated_config.latent_q_smoothness_weight * _latent_q_smoothness_penalty(
                    model,
                    batch_features,
                    q_batch,
                    epsilon=validated_config.latent_q_smoothness_epsilon,
                )

            if (
                validated_config.mds_position_preservation_weight > 0
                and initial_q_weight is not None
                and phase_name == "Joint-FT"
            ):
                q_weight = embedding.weight
                n = q_weight.shape[0]
                if n > 1:
                    q_diff = q_weight.unsqueeze(0) - q_weight.unsqueeze(1)
                    q_diff_norm = q_diff.norm(dim=-1)
                    
                    init_diff = initial_q_weight.unsqueeze(0) - initial_q_weight.unsqueeze(1)
                    init_diff_norm = init_diff.norm(dim=-1)
                    
                    mask = init_diff_norm > 1e-8
                    if mask.sum() > 0:
                        ratio = q_diff_norm[mask] / init_diff_norm[mask].clamp_min(1e-8)
                        distance_ratio_loss = (ratio - 1.0).pow(2).mean()
                        position_loss = distance_ratio_loss
                    else:
                        position_loss = (q_weight - initial_q_weight).pow(2).mean()
                else:
                    position_loss = (q_weight - initial_q_weight).pow(2).mean()
                
                loss = loss + validated_config.mds_position_preservation_weight * position_loss

            if (
                validated_config.mds_q_regularization_weight > 0
                and initial_q_weight is not None
                and phase_name == "Joint-FT"
            ):
                q_weight = embedding.weight
                regularization_loss = (q_weight - initial_q_weight).pow(2).mean()
                loss = loss + validated_config.mds_q_regularization_weight * regularization_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if validated_config.latent_q_canonicalization_mode == "train":
                _project_embedding_to_canonical_q_(embedding)

        epoch_metrics = _compute_train_epoch_metrics(
            epoch_number=epoch_num,
            model=model,
            embedding=embedding,
            feature_tensor=feature_tensor,
            label_tensor=label_tensor,
            targets_original=train_dataset.targets_original,
            target_scale_factor=train_dataset.target_scale_factor,
            normalizer=normalizer,
        )
        history.append(epoch_metrics)
        if validated_config.verbose:
            prefix = f"[{phase_name}] " if phase_name else ""
            message = (
                f"{prefix}Epoch {epoch_metrics.epoch}: "
                f"Train R2: {epoch_metrics.r2:.6f}, Train MSE: {epoch_metrics.mse:.6g}"
            )
            if not np.isclose(train_dataset.target_scale_factor, 1.0):
                message += f", Train MSE Original: {epoch_metrics.mse_original:.6g}"
            print(message)
        return epoch_metrics

    current_epoch = 0

    if validated_config.mds_init_enabled:
        mds_q_initial = embedding.weight.data.detach().clone()

        if validated_config.verbose:
            print("=== Phase 1: Fixed Q training (only θ) ===")
        embedding.requires_grad_(False)
        fixed_q_optimizer = optim.Adam(model.parameters(), lr=validated_config.lr)
        for epoch in range(validated_config.mds_fixed_q_epochs):
            current_epoch += 1
            epoch_metrics = _train_epoch(model, embedding, fixed_q_optimizer, current_epoch, "Fixed-Q")
            if _is_early_stop_epoch(epoch_metrics, validated_config):
                early_stop_streak += 1
                if early_stop_streak >= validated_config.early_stop_patience:
                    early_stopped = True
                    early_stop_epoch = epoch_metrics.epoch
                    if validated_config.verbose:
                        print(f"Early stopping at epoch {early_stop_epoch}")
                    break
            else:
                early_stop_streak = 0
        if early_stopped:
            pass
        elif validated_config.mds_joint_finetune_epochs > 0:
            if validated_config.verbose:
                print("=== Phase 2: Joint fine-tuning (θ and Q with regularization) ===")
            embedding.requires_grad_(True)
            joint_lr = validated_config.lr * validated_config.mds_joint_finetune_lr_ratio
            joint_optimizer = optim.Adam(
                list(model.parameters()) + list(embedding.parameters()),
                lr=joint_lr,
            )
            for epoch in range(validated_config.mds_joint_finetune_epochs):
                current_epoch += 1
                epoch_metrics = _train_epoch(
                    model,
                    embedding,
                    joint_optimizer,
                    current_epoch,
                    "Joint-FT",
                    initial_q_weight=mds_q_initial,
                )
                if _is_early_stop_epoch(epoch_metrics, validated_config):
                    early_stop_streak += 1
                    if early_stop_streak >= validated_config.early_stop_patience:
                        early_stopped = True
                        early_stop_epoch = epoch_metrics.epoch
                        if validated_config.verbose:
                            print(f"Early stopping at epoch {early_stop_epoch}")
                        break
                else:
                    early_stop_streak = 0
    else:
        if validated_config.verbose:
            print("=== Standard training (θ and Q together) ===")
        optimizer = optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=validated_config.lr)
        for epoch in range(validated_config.epochs):
            current_epoch += 1
            epoch_metrics = _train_epoch(model, embedding, optimizer, current_epoch)
            if _is_early_stop_epoch(epoch_metrics, validated_config):
                early_stop_streak += 1
                if early_stop_streak >= validated_config.early_stop_patience:
                    early_stopped = True
                    early_stop_epoch = epoch_metrics.epoch
                    if validated_config.verbose:
                        print(
                            "Early stopping triggered: "
                            f"train R2={epoch_metrics.r2:.6f} >= {validated_config.early_stop_r2_threshold:.6f} "
                            f"for {validated_config.early_stop_patience} consecutive epochs."
                        )
                    break
            else:
                early_stop_streak = 0

    return TrainingArtifacts(
        model=model,
        embedding=embedding,
        normalizer=normalizer,
        label_to_index=label_to_index,
        device=device,
        train_history=history,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
    )


def calibrate_latent_q_for_test_labels(
    test_dataset: LatentQDataset,
    training_artifacts: TrainingArtifacts,
    config: LatentQConfig,
    *,
    plot_feature_index: Optional[int] = None,
) -> CalibrationArtifacts:
    validated_config = _validate_config(config)
    plot_index = _resolve_plot_feature_index(plot_feature_index, test_dataset.features.shape[1])

    normalized_test_features = normalize_features(test_dataset.features, training_artifacts.normalizer)
    normalized_test_targets = normalize_targets(test_dataset.targets, training_artifacts.normalizer)

    feature_tensor = torch.tensor(normalized_test_features, dtype=torch.float32, device=training_artifacts.device)
    target_tensor = torch.tensor(normalized_test_targets.reshape(-1, 1), dtype=torch.float32, device=training_artifacts.device)

    q_by_label: dict[Any, np.ndarray] = {}
    eval_predictions: list[np.ndarray] = []
    eval_targets: list[np.ndarray] = []
    eval_plot_axis: list[np.ndarray] = []
    eval_indices: list[np.ndarray] = []

    model = training_artifacts.model
    model.eval()
    mse_loss = nn.MSELoss()
    train_q = training_artifacts.embedding.weight.detach()
    q_prior_mean = train_q.mean(dim=0)
    q_prior_std = train_q.std(dim=0).clamp_min(0.05)

    for raw_label in pd.unique(test_dataset.labels):
        label = _normalize_label_value(raw_label)
        label_indices = np.flatnonzero(test_dataset.labels == raw_label)
        calibration_indices, evaluation_indices = split_calibration_and_eval_indices(
            label_indices,
            validated_config.calibration_ratio,
        )
        initial_q = _initial_q_vector(label, training_artifacts, validated_config.q_dim)
        q_parameter = nn.Parameter(initial_q.to(training_artifacts.device))
        optimizer = optim.Adam([q_parameter], lr=validated_config.calibration_lr)

        for _ in range(validated_config.calibration_steps):
            calibration_features = feature_tensor[calibration_indices]
            repeated_q = q_parameter.unsqueeze(0).repeat(calibration_features.shape[0], 1)
            model_inputs = torch.cat([calibration_features, repeated_q], dim=1)
            predictions = _ensure_prediction_column(model(model_inputs))
            loss = mse_loss(predictions, target_tensor[calibration_indices])
            if validated_config.calibration_q_prior_weight > 0:
                standardized_q = (q_parameter - q_prior_mean) / q_prior_std
                loss = loss + validated_config.calibration_q_prior_weight * torch.mean(standardized_q.pow(2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        q_by_label[label] = q_parameter.detach().cpu().numpy().copy()

        if evaluation_indices.size == 0:
            continue

        with torch.no_grad():
            evaluation_features = feature_tensor[evaluation_indices]
            repeated_q = q_parameter.unsqueeze(0).repeat(evaluation_features.shape[0], 1)
            model_inputs = torch.cat([evaluation_features, repeated_q], dim=1)
            predictions = _ensure_prediction_column(model(model_inputs)).squeeze(1)

        denormalized_predictions = denormalize_targets(
            predictions.cpu().numpy(),
            training_artifacts.normalizer,
        )
        eval_predictions.append(denormalized_predictions / test_dataset.target_scale_factor)
        eval_targets.append(test_dataset.targets_original[evaluation_indices])
        eval_plot_axis.append(test_dataset.features[evaluation_indices, plot_index])
        eval_indices.append(evaluation_indices)

    if not eval_predictions:
        raise RuntimeError("No evaluation samples remain after applying the calibration split.")

    return CalibrationArtifacts(
        q_by_label=q_by_label,
        eval_predictions=np.concatenate(eval_predictions),
        eval_targets=np.concatenate(eval_targets),
        eval_plot_axis=np.concatenate(eval_plot_axis),
        eval_indices=np.concatenate(eval_indices),
    )


def assemble_output_frame(dataset: LatentQDataset, q_matrix: np.ndarray) -> pd.DataFrame:
    q_array = np.asarray(q_matrix, dtype=np.float32)
    if q_array.ndim != 2:
        raise ValueError("q_matrix must be a 2D array.")
    if q_array.shape[0] != len(dataset.labels):
        raise ValueError("q_matrix row count must match the dataset size.")

    output_frame = dataset.base_output_frame.copy()
    for index in range(q_array.shape[1]):
        output_frame[f"q{index + 1}"] = q_array[:, index]
    return output_frame


def save_output_frame(frame: pd.DataFrame, csv_path: Path | str) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_pipeline_outputs(result: LatentQPipelineResult, output_config: OutputConfig) -> dict[str, Path]:
    saved_paths: dict[str, Path] = {}
    output_dir = Path(output_config.output_dir)

    if output_config.save_csv:
        saved_paths["train_csv"] = save_output_frame(result.train_output, output_dir / output_config.train_output_name)
        saved_paths["test_csv"] = save_output_frame(result.test_output, output_dir / output_config.test_output_name)

    if output_config.save_plot:
        if not output_config.plot_output_name:
            raise ValueError("plot_output_name must be provided when save_plot=True.")
        plot_path = output_dir / output_config.plot_output_name
        saved_paths["plot"] = plot_prediction_curve(
            x_axis=result.eval_plot_axis,
            y_true=result.eval_targets,
            y_pred=result.eval_predictions,
            plot_path=plot_path,
            x_label=result.plot_feature_name,
            title=output_config.plot_title,
        )

    result.saved_paths = saved_paths
    return saved_paths


def run_latent_q_pipeline(
    train_dataset: LatentQDataset,
    test_dataset: LatentQDataset,
    model_factory: ModelFactory,
    config: LatentQConfig,
    *,
    output_config: Optional[OutputConfig] = None,
) -> LatentQPipelineResult:
    _validate_matching_feature_dimensions(train_dataset, test_dataset)
    validated_config = _validate_config(config)
    resolved_output_config = output_config or OutputConfig(save_csv=False, save_plot=False)
    plot_index = _resolve_plot_feature_index(resolved_output_config.plot_feature_index, test_dataset.features.shape[1])

    training_artifacts = train_latent_q_model(train_dataset, model_factory, validated_config)
    calibration_artifacts = calibrate_latent_q_for_test_labels(
        test_dataset,
        training_artifacts,
        validated_config,
        plot_feature_index=plot_index,
    )

    train_q_matrix = extract_train_q_matrix(train_dataset.labels, training_artifacts)
    test_q_matrix = build_q_matrix(test_dataset.labels, calibration_artifacts.q_by_label)
    if validated_config.latent_q_canonicalization_mode in {"output", "train"}:
        train_q_matrix, test_q_matrix = _canonicalize_q_outputs(train_q_matrix, test_q_matrix)
    train_output = assemble_output_frame(train_dataset, train_q_matrix)
    test_output = assemble_output_frame(test_dataset, test_q_matrix)

    metrics = {
        "train_r2_last_epoch": training_artifacts.train_history[-1].r2,
        "train_mse_last_epoch": training_artifacts.train_history[-1].mse,
        "train_mse_original_last_epoch": training_artifacts.train_history[-1].mse_original,
        "test_r2": float(r2_score(calibration_artifacts.eval_targets, calibration_artifacts.eval_predictions)),
        "test_mse": float(mean_squared_error(calibration_artifacts.eval_targets, calibration_artifacts.eval_predictions)),
        "target_scale_factor": train_dataset.target_scale_factor,
        "early_stopped": training_artifacts.early_stopped,
        "early_stop_epoch": training_artifacts.early_stop_epoch,
        "epochs_completed": len(training_artifacts.train_history),
        "latent_q_canonicalization_mode": validated_config.latent_q_canonicalization_mode,
    }
    metrics.update(compute_latent_feature_orthogonality_metrics(train_dataset, train_q_matrix))
    metrics.update(_flatten_q_distribution_metrics("train_q", train_q_matrix))
    metrics.update(_flatten_q_distribution_metrics("test_q", test_q_matrix))

    result = LatentQPipelineResult(
        training_artifacts=training_artifacts,
        train_output=train_output,
        test_output=test_output,
        train_q_matrix=train_q_matrix,
        test_q_matrix=test_q_matrix,
        eval_predictions=calibration_artifacts.eval_predictions,
        eval_targets=calibration_artifacts.eval_targets,
        eval_plot_axis=calibration_artifacts.eval_plot_axis,
        eval_indices=calibration_artifacts.eval_indices,
        plot_feature_name=test_dataset.feature_names[plot_index],
        metrics=metrics,
    )

    if output_config is not None:
        save_pipeline_outputs(result, resolved_output_config)

    return result


def _flatten_q_distribution_metrics(prefix: str, q_matrix: np.ndarray) -> dict[str, float]:
    q_array = np.asarray(q_matrix, dtype=float)
    if q_array.size == 0:
        return {}
    metrics: dict[str, float] = {}
    for index in range(q_array.shape[1]):
        column = q_array[:, index]
        q_prefix = f"{prefix}{index + 1}"
        metrics[f"{q_prefix}_min"] = float(np.min(column))
        metrics[f"{q_prefix}_mean"] = float(np.mean(column))
        metrics[f"{q_prefix}_max"] = float(np.max(column))
        metrics[f"{q_prefix}_std"] = float(np.std(column))
    return metrics


def fit_normalization(features: np.ndarray, targets: np.ndarray) -> NormalizationStats:
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + EPSILON
    target_mean = float(targets.mean())
    target_std = float(targets.std() + EPSILON)
    return NormalizationStats(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean,
        target_std=target_std,
    )


def normalize_features(features: np.ndarray, normalizer: NormalizationStats) -> np.ndarray:
    return ((features - normalizer.feature_mean) / normalizer.feature_std).astype(np.float32)


def normalize_targets(targets: np.ndarray, normalizer: NormalizationStats) -> np.ndarray:
    return ((targets - normalizer.target_mean) / normalizer.target_std).astype(np.float32)


def denormalize_targets(targets: np.ndarray, normalizer: NormalizationStats) -> np.ndarray:
    return targets * normalizer.target_std + normalizer.target_mean


def extract_train_q_matrix(labels: np.ndarray, training_artifacts: TrainingArtifacts) -> np.ndarray:
    label_keys = [_normalize_label_value(label) for label in labels]
    with torch.no_grad():
        embedding_weights = training_artifacts.embedding.weight.detach().cpu().numpy()
    return np.vstack([embedding_weights[training_artifacts.label_to_index[label]] for label in label_keys]).astype(np.float32)


def build_q_matrix(labels: np.ndarray, q_by_label: dict[Any, np.ndarray]) -> np.ndarray:
    label_keys = [_normalize_label_value(label) for label in labels]
    return np.vstack([q_by_label[label] for label in label_keys]).astype(np.float32)


def compute_latent_feature_orthogonality_metrics(
    dataset: LatentQDataset,
    q_matrix: np.ndarray,
) -> dict[str, float]:
    label_keys = [_normalize_label_value(label) for label in dataset.labels]
    unique_labels = list(dict.fromkeys(label_keys))
    if len(unique_labels) < 2:
        return {
            "latent_feature_corr_mean_abs": 0.0,
            "latent_feature_corr_max_abs": 0.0,
            "latent_feature_corr_mean_sq": 0.0,
        }

    q_by_label: list[np.ndarray] = []
    feature_stats_by_label: list[np.ndarray] = []
    label_array = np.asarray(label_keys, dtype=object)
    for label in unique_labels:
        indices = np.flatnonzero(label_array == label)
        q_by_label.append(np.asarray(q_matrix[indices], dtype=np.float64).mean(axis=0))
        feature_values = np.asarray(dataset.features[indices], dtype=np.float64)
        feature_stats_by_label.append(
            np.concatenate(
                [
                    feature_values.mean(axis=0),
                    feature_values.std(axis=0),
                ]
            )
        )

    q_values = np.vstack(q_by_label)
    feature_stats = np.vstack(feature_stats_by_label)
    corr = _numpy_column_correlation(q_values, feature_stats)
    abs_corr = np.abs(corr[np.isfinite(corr)])
    if abs_corr.size == 0:
        return {
            "latent_feature_corr_mean_abs": 0.0,
            "latent_feature_corr_max_abs": 0.0,
            "latent_feature_corr_mean_sq": 0.0,
        }
    return {
        "latent_feature_corr_mean_abs": float(abs_corr.mean()),
        "latent_feature_corr_max_abs": float(abs_corr.max()),
        "latent_feature_corr_mean_sq": float(np.mean(abs_corr**2)),
    }


def scale_dataset_targets(dataset: LatentQDataset, scale_factor: float) -> LatentQDataset:
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be a positive finite number.")
    if np.isclose(scale_factor, 1.0):
        return dataset

    scaled_targets = (dataset.targets_original * np.float32(scale_factor)).astype(np.float32)
    return LatentQDataset(
        features=dataset.features.copy(),
        labels=dataset.labels.copy(),
        targets=scaled_targets,
        targets_original=dataset.targets_original.copy(),
        target_scale_factor=float(scale_factor),
        feature_names=dataset.feature_names,
        label_name=dataset.label_name,
        target_name=dataset.target_name,
        base_output_frame=dataset.base_output_frame.copy(),
    )


def split_calibration_and_eval_indices(indices: np.ndarray, calibration_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        raise ValueError("indices cannot be empty.")
    split_point = max(1, int(np.floor(calibration_ratio * indices.size)))
    calibration_indices = indices[:split_point]
    evaluation_indices = indices[split_point:]
    return calibration_indices, evaluation_indices


def plot_prediction_curve(
    *,
    x_axis: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_path: Path | str,
    x_label: str,
    title: str,
) -> Path:
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    sort_order = np.argsort(x_axis)
    x_sorted = x_axis[sort_order]
    y_true_sorted = y_true[sort_order]
    y_pred_sorted = y_pred[sort_order]

    plt.figure(figsize=(8, 5))
    plt.plot(x_sorted, y_true_sorted, label="Ground Truth", linewidth=2)
    plt.plot(x_sorted, y_pred_sorted, label="Prediction", linewidth=2)
    plt.xlabel(x_label)
    plt.ylabel("Target")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def _build_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    targets: np.ndarray,
    targets_original: np.ndarray,
    target_scale_factor: float,
    feature_names: Sequence[str],
    label_name: str,
    target_name: str,
    base_output_frame: pd.DataFrame,
) -> LatentQDataset:
    feature_array, label_array, target_array = _coerce_dataset_arrays(features, labels, targets)
    _, _, target_original_array = _coerce_dataset_arrays(features, labels, targets_original)
    if len(feature_names) != feature_array.shape[1]:
        raise ValueError("feature_names length must match the feature dimension.")

    return LatentQDataset(
        features=feature_array,
        labels=label_array,
        targets=target_array,
        targets_original=target_original_array,
        target_scale_factor=float(target_scale_factor),
        feature_names=tuple(str(name) for name in feature_names),
        label_name=str(label_name),
        target_name=str(target_name),
        base_output_frame=base_output_frame.copy(),
    )


def _coerce_dataset_arrays(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    targets: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)
    if feature_array.ndim != 2:
        raise ValueError("features must be a 2D array or a sequence convertible to 2D.")
    if feature_array.shape[0] == 0:
        raise ValueError("features cannot be empty.")
    if not np.isfinite(feature_array).all():
        raise ValueError("features must contain only finite values.")

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        label_array = label_array.reshape(-1)
    if label_array.shape[0] == 0:
        raise ValueError("labels cannot be empty.")
    if pd.isna(label_array).any():
        raise ValueError("labels cannot contain null values.")

    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim != 1:
        target_array = target_array.reshape(-1)
    if target_array.shape[0] == 0:
        raise ValueError("targets cannot be empty.")
    if not np.isfinite(target_array).all():
        raise ValueError("targets must contain only finite values.")

    if not (feature_array.shape[0] == label_array.shape[0] == target_array.shape[0]):
        raise ValueError("features, labels, and targets must have the same number of rows.")

    return feature_array, label_array, target_array


def _validate_config(config: LatentQConfig) -> LatentQConfig:
    if config.q_dim <= 0:
        raise ValueError("q_dim must be a positive integer.")
    if config.epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if config.lr <= 0:
        raise ValueError("lr must be positive.")
    if config.calibration_steps <= 0:
        raise ValueError("calibration_steps must be a positive integer.")
    if config.calibration_lr <= 0:
        raise ValueError("calibration_lr must be positive.")
    if not 0 < config.calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between 0 and 1.")
    if not 0 <= config.early_stop_r2_threshold <= 1:
        raise ValueError("early_stop_r2_threshold must be between 0 and 1.")
    if config.early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be a positive integer.")
    if config.latent_feature_orthogonality_weight < 0:
        raise ValueError("latent_feature_orthogonality_weight must be non-negative.")
    if config.latent_feature_orthogonality_type not in {
        "pearson",
        "hsic",
        "nhsic",
        "distance_correlation",
        "adversarial",
        "propensity",
    }:
        raise ValueError(
            "latent_feature_orthogonality_type must be one of: "
            "pearson, hsic, nhsic, distance_correlation, adversarial, propensity."
        )
    if config.latent_feature_stats_mode not in {"mean_std", "rich", "rff_kme", "rich_rff_kme"}:
        raise ValueError(
            "latent_feature_stats_mode must be one of: mean_std, rich, rff_kme, rich_rff_kme."
        )
    if config.latent_curve_continuity_weight < 0:
        raise ValueError("latent_curve_continuity_weight must be non-negative.")
    if config.latent_curve_continuity_grid_size <= 1:
        raise ValueError("latent_curve_continuity_grid_size must be greater than 1.")
    if config.calibration_q_prior_weight < 0:
        raise ValueError("calibration_q_prior_weight must be non-negative.")
    if config.latent_q_l2_weight < 0:
        raise ValueError("latent_q_l2_weight must be non-negative.")
    if config.prediction_loss_type not in {"mse", "label_balanced_mse"}:
        raise ValueError("prediction_loss_type must be one of: mse, label_balanced_mse.")
    if config.latent_q_whitening_weight < 0:
        raise ValueError("latent_q_whitening_weight must be non-negative.")
    if config.latent_jacobian_disentanglement_weight < 0:
        raise ValueError("latent_jacobian_disentanglement_weight must be non-negative.")
    if config.latent_q_canonicalization_mode not in {"none", "output", "train"}:
        raise ValueError("latent_q_canonicalization_mode must be one of: none, output, train.")
    if config.latent_q_smoothness_weight < 0:
        raise ValueError("latent_q_smoothness_weight must be non-negative.")
    if config.latent_q_smoothness_epsilon <= 0:
        raise ValueError("latent_q_smoothness_epsilon must be positive.")
    if config.mds_fixed_q_epochs < 0:
        raise ValueError("mds_fixed_q_epochs must be non-negative.")
    if config.mds_joint_finetune_epochs < 0:
        raise ValueError("mds_joint_finetune_epochs must be non-negative.")
    if config.mds_joint_finetune_lr_ratio <= 0 or config.mds_joint_finetune_lr_ratio > 1:
        raise ValueError("mds_joint_finetune_lr_ratio must be between 0 and 1.")
    if config.mds_position_preservation_weight < 0:
        raise ValueError("mds_position_preservation_weight must be non-negative.")
    if config.mds_q_regularization_weight < 0:
        raise ValueError("mds_q_regularization_weight must be non-negative.")
    return config


def _compute_label_feature_stats(
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    label_count: int,
    mode: str = "mean_std",
) -> torch.Tensor:
    if mode == "mean_std":
        return _compute_label_mean_std_stats(feature_tensor, label_tensor, label_count=label_count)
    if mode == "rich":
        return _compute_label_rich_stats(feature_tensor, label_tensor, label_count=label_count)
    if mode == "rff_kme":
        return _compute_label_rff_kme_stats(feature_tensor, label_tensor, label_count=label_count)
    if mode == "rich_rff_kme":
        return torch.cat(
            [
                _compute_label_rich_stats(feature_tensor, label_tensor, label_count=label_count),
                _compute_label_rff_kme_stats(feature_tensor, label_tensor, label_count=label_count),
            ],
            dim=1,
        ).detach()
    raise ValueError(f"Unsupported latent feature stats mode: {mode}")


def _compute_label_mean_std_stats(
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    label_count: int,
) -> torch.Tensor:
    stats: list[torch.Tensor] = []
    for label_index in range(label_count):
        label_features = feature_tensor[label_tensor == label_index]
        if label_features.shape[0] == 0:
            stats.append(torch.zeros(feature_tensor.shape[1] * 2, dtype=feature_tensor.dtype, device=feature_tensor.device))
            continue
        stats.append(torch.cat([label_features.mean(dim=0), label_features.std(dim=0, unbiased=False)]))
    return torch.stack(stats, dim=0).detach()


def _compute_label_rich_stats(
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    label_count: int,
) -> torch.Tensor:
    feature_dim = feature_tensor.shape[1]
    quantile_levels = feature_tensor.new_tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    covariance_dim = feature_dim * (feature_dim + 1) // 2
    empty_dim = feature_dim * 10 + covariance_dim + 1
    stats: list[torch.Tensor] = []
    for label_index in range(label_count):
        label_features = feature_tensor[label_tensor == label_index]
        if label_features.shape[0] == 0:
            stats.append(torch.zeros(empty_dim, dtype=feature_tensor.dtype, device=feature_tensor.device))
            continue
        mean = label_features.mean(dim=0)
        std = label_features.std(dim=0, unbiased=False)
        min_values = label_features.min(dim=0).values
        max_values = label_features.max(dim=0).values
        value_range = max_values - min_values
        quantiles = torch.quantile(label_features, quantile_levels, dim=0).transpose(0, 1).reshape(-1)
        centered = label_features - mean
        covariance = centered.transpose(0, 1).matmul(centered) / max(int(label_features.shape[0]), 1)
        upper_indices = torch.triu_indices(feature_dim, feature_dim, device=feature_tensor.device)
        covariance_upper = covariance[upper_indices[0], upper_indices[1]]
        log_count = torch.log1p(feature_tensor.new_tensor([float(label_features.shape[0])]))
        stats.append(
            torch.cat(
                [
                    mean,
                    std,
                    min_values,
                    max_values,
                    value_range,
                    quantiles,
                    covariance_upper,
                    log_count,
                ]
            )
        )
    return torch.stack(stats, dim=0).detach()


def _compute_label_rff_kme_stats(
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    label_count: int,
    features_per_scale: int = 32,
    scales: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    feature_dim = feature_tensor.shape[1]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1729 + feature_dim * 17 + features_per_scale)
    embeddings: list[torch.Tensor] = []
    for scale in scales:
        weights = torch.randn(feature_dim, features_per_scale, generator=generator, dtype=torch.float32)
        weights = weights / max(float(scale), EPSILON)
        phases = 2.0 * np.pi * torch.rand(features_per_scale, generator=generator, dtype=torch.float32)
        weights = weights.to(device=feature_tensor.device, dtype=feature_tensor.dtype)
        phases = phases.to(device=feature_tensor.device, dtype=feature_tensor.dtype)
        projected = feature_tensor.matmul(weights) + phases
        embeddings.append(np.sqrt(2.0 / features_per_scale) * torch.cos(projected))
    point_embedding = torch.cat(embeddings, dim=1)
    stats: list[torch.Tensor] = []
    for label_index in range(label_count):
        label_embedding = point_embedding[label_tensor == label_index]
        if label_embedding.shape[0] == 0:
            stats.append(torch.zeros(point_embedding.shape[1], dtype=feature_tensor.dtype, device=feature_tensor.device))
            continue
        stats.append(label_embedding.mean(dim=0))
    return torch.stack(stats, dim=0).detach()


def _prediction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    per_sample_loss = (predictions - targets).pow(2).reshape(-1)
    if loss_type == "mse":
        return per_sample_loss.mean()
    if loss_type == "label_balanced_mse":
        label_losses = []
        for label in torch.unique(labels):
            mask = labels == label
            if torch.any(mask):
                label_losses.append(per_sample_loss[mask].mean())
        if not label_losses:
            return per_sample_loss.mean()
        return torch.stack(label_losses).mean()
    raise ValueError(f"Unsupported prediction loss type: {loss_type}")


def _latent_feature_correlation_penalty(q_values: torch.Tensor, feature_stats: torch.Tensor) -> torch.Tensor:
    if q_values.shape[0] < 2 or feature_stats.shape[1] == 0:
        return q_values.new_tensor(0.0)
    q_normalized = _standardize_columns(q_values)
    stats_normalized = _standardize_columns(feature_stats)
    corr = q_normalized.transpose(0, 1).matmul(stats_normalized) / q_values.shape[0]
    return corr.pow(2).mean()


def _latent_feature_orthogonality_penalty(
    q_values: torch.Tensor,
    feature_stats: torch.Tensor,
    *,
    penalty_type: str,
    adversary: Optional[nn.Module] = None,
    adversary_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if q_values.shape[0] < 2 or feature_stats.shape[1] == 0:
        return q_values.new_tensor(0.0)
    if penalty_type == "pearson":
        return _latent_feature_correlation_penalty(q_values, feature_stats)
    if penalty_type in {"hsic", "nhsic"}:
        return _latent_feature_hsic_penalty(q_values, feature_stats)
    if penalty_type == "distance_correlation":
        return _latent_feature_distance_correlation_penalty(q_values, feature_stats)
    if penalty_type == "propensity":
        weights = _feature_stat_inverse_propensity_weights(feature_stats)
        return _latent_feature_weighted_correlation_penalty(q_values, feature_stats, weights)
    if penalty_type == "adversarial":
        if adversary is None or adversary_targets is None:
            return q_values.new_tensor(0.0)
        q_normalized = _standardize_columns(q_values)
        predictions = adversary(q_normalized)
        adversary_mse = (predictions - adversary_targets).pow(2).mean()
        return torch.exp(-adversary_mse)
    raise ValueError(f"Unsupported latent feature orthogonality type: {penalty_type}")


def _latent_feature_weighted_correlation_penalty(
    q_values: torch.Tensor,
    feature_stats: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if q_values.shape[0] < 2 or feature_stats.shape[1] == 0:
        return q_values.new_tensor(0.0)
    normalized_weights = weights.reshape(-1, 1)
    normalized_weights = normalized_weights / normalized_weights.mean().clamp_min(EPSILON)
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(EPSILON)

    q_mean = (normalized_weights * q_values).sum(dim=0, keepdim=True)
    stats_mean = (normalized_weights * feature_stats).sum(dim=0, keepdim=True)
    q_centered = q_values - q_mean
    stats_centered = feature_stats - stats_mean
    q_scale = (normalized_weights * q_centered.pow(2)).sum(dim=0, keepdim=True).sqrt().clamp_min(EPSILON)
    stats_scale = (normalized_weights * stats_centered.pow(2)).sum(dim=0, keepdim=True).sqrt().clamp_min(EPSILON)
    q_normalized = q_centered / q_scale
    stats_normalized = stats_centered / stats_scale
    weighted_q = q_normalized * normalized_weights
    corr = weighted_q.transpose(0, 1).matmul(stats_normalized)
    return corr.pow(2).mean()


def _latent_feature_hsic_penalty(q_values: torch.Tensor, feature_stats: torch.Tensor) -> torch.Tensor:
    q_kernel = _rbf_kernel_with_median_bandwidth(_standardize_columns(q_values))
    stats_kernel = _rbf_kernel_with_median_bandwidth(_standardize_columns(feature_stats))
    return _centered_kernel_alignment(q_kernel, stats_kernel)


def _latent_feature_distance_correlation_penalty(q_values: torch.Tensor, feature_stats: torch.Tensor) -> torch.Tensor:
    q_distances = torch.cdist(_standardize_columns(q_values), _standardize_columns(q_values), p=2)
    stats_distances = torch.cdist(_standardize_columns(feature_stats), _standardize_columns(feature_stats), p=2)
    q_centered = _double_center_distance_matrix(q_distances)
    stats_centered = _double_center_distance_matrix(stats_distances)
    dcov = (q_centered * stats_centered).mean()
    dvar_q = (q_centered * q_centered).mean().clamp_min(EPSILON)
    dvar_stats = (stats_centered * stats_centered).mean().clamp_min(EPSILON)
    return (dcov / torch.sqrt(dvar_q * dvar_stats).clamp_min(EPSILON)).clamp_min(0.0)


def _latent_q_whitening_penalty(q_values: torch.Tensor) -> torch.Tensor:
    if q_values.shape[0] < 2:
        return q_values.new_tensor(0.0)
    mean_penalty = q_values.mean(dim=0).pow(2).mean()
    centered = q_values - q_values.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0).sqrt().clamp_min(EPSILON)
    std_penalty = (std - 1.0).pow(2).mean()
    standardized = centered / std.reshape(1, -1)
    covariance = standardized.transpose(0, 1).matmul(standardized) / standardized.shape[0]
    identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device)
    covariance_penalty = (covariance - identity).pow(2).mean()
    return mean_penalty + std_penalty + covariance_penalty


def _latent_jacobian_disentanglement_penalty(
    model: nn.Module,
    features: torch.Tensor,
    q_values: torch.Tensor,
) -> torch.Tensor:
    if q_values.shape[1] < 2 or q_values.shape[0] < 2:
        return q_values.new_tensor(0.0)
    q_for_grad = q_values.detach().clone().requires_grad_(True)
    predictions = _ensure_prediction_column(model(torch.cat([features, q_for_grad], dim=1)))
    gradients = torch.autograd.grad(
        predictions.sum(),
        q_for_grad,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradients is None:
        return q_values.new_tensor(0.0)
    normalized = _standardize_columns(gradients)
    gram = normalized.transpose(0, 1).matmul(normalized) / normalized.shape[0]
    offdiag_mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    if not torch.any(offdiag_mask):
        return q_values.new_tensor(0.0)
    return gram[offdiag_mask].pow(2).mean()


def _latent_q_smoothness_penalty(
    model: nn.Module,
    features: torch.Tensor,
    q_values: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    if q_values.shape[1] == 0:
        return q_values.new_tensor(0.0)
    step = max(float(epsilon), EPSILON)
    base_predictions = _ensure_prediction_column(model(torch.cat([features, q_values], dim=1)))
    penalties: list[torch.Tensor] = []
    for dim in range(q_values.shape[1]):
        perturbation = torch.zeros_like(q_values)
        perturbation[:, dim] = step
        plus_predictions = _ensure_prediction_column(model(torch.cat([features, q_values + perturbation], dim=1)))
        minus_predictions = _ensure_prediction_column(model(torch.cat([features, q_values - perturbation], dim=1)))
        curvature = (plus_predictions - 2.0 * base_predictions + minus_predictions) / (step * step)
        penalties.append(curvature.pow(2).mean())
    return torch.stack(penalties).mean()


@torch.no_grad()
def _project_embedding_to_canonical_q_(embedding: nn.Embedding) -> None:
    q_values = embedding.weight.data
    if q_values.shape[0] < 2:
        q_values.sub_(q_values.mean(dim=0, keepdim=True))
        return
    centered = q_values - q_values.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).matmul(centered) / max(1, centered.shape[0] - 1)
    jitter = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=covariance.device) * 1e-4
    eigvals, eigvecs = torch.linalg.eigh(covariance + jitter)
    whitening = eigvecs.matmul(torch.diag(torch.rsqrt(eigvals.clamp_min(1e-4)))).matmul(eigvecs.transpose(0, 1))
    q_values.copy_(centered.matmul(whitening))


def _canonicalize_q_outputs(train_q: np.ndarray, test_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_array = np.asarray(train_q, dtype=np.float64)
    test_array = np.asarray(test_q, dtype=np.float64)
    if train_array.ndim != 2 or train_array.shape[0] < 2:
        return train_q, test_q
    mean = train_array.mean(axis=0, keepdims=True)
    centered_train = train_array - mean
    covariance = centered_train.T @ centered_train / max(1, centered_train.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(covariance + np.eye(covariance.shape[0]) * 1e-8)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-8))) @ eigvecs.T
    return (
        ((train_array - mean) @ whitening).astype(np.float32),
        ((test_array - mean) @ whitening).astype(np.float32),
    )


def _feature_stat_inverse_propensity_weights(feature_stats: torch.Tensor) -> torch.Tensor:
    normalized_stats = _standardize_columns(feature_stats)
    squared_distances = torch.cdist(normalized_stats, normalized_stats, p=2).pow(2)
    mask = ~torch.eye(feature_stats.shape[0], dtype=torch.bool, device=feature_stats.device)
    if not torch.any(mask):
        return torch.ones(feature_stats.shape[0], dtype=feature_stats.dtype, device=feature_stats.device)
    selected = squared_distances[mask]
    bandwidth = torch.sqrt(torch.median(selected.detach()).clamp_min(EPSILON))
    affinities = torch.exp(-squared_distances / (2.0 * bandwidth.pow(2).clamp_min(EPSILON)))
    affinities = affinities.masked_fill(~mask, 0.0)
    density = affinities.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    weights = 1.0 / density.clamp_min(0.05)
    return (weights / weights.mean().clamp_min(EPSILON)).detach()


def _standardize_columns(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    scale = centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(EPSILON)
    return centered / scale


def _rbf_kernel_with_median_bandwidth(values: torch.Tensor) -> torch.Tensor:
    squared_distances = torch.cdist(values, values, p=2).pow(2)
    mask = ~torch.eye(values.shape[0], dtype=torch.bool, device=values.device)
    if torch.any(mask):
        bandwidth_squared = torch.median(squared_distances[mask].detach()).clamp_min(EPSILON)
    else:
        bandwidth_squared = values.new_tensor(1.0)
    return torch.exp(-squared_distances / (2.0 * bandwidth_squared))


def _centered_kernel_alignment(left_kernel: torch.Tensor, right_kernel: torch.Tensor) -> torch.Tensor:
    left_centered = _center_kernel_matrix(left_kernel)
    right_centered = _center_kernel_matrix(right_kernel)
    numerator = (left_centered * right_centered).mean()
    left_scale = (left_centered * left_centered).mean().clamp_min(EPSILON)
    right_scale = (right_centered * right_centered).mean().clamp_min(EPSILON)
    return (numerator / torch.sqrt(left_scale * right_scale).clamp_min(EPSILON)).clamp_min(0.0)


def _center_kernel_matrix(kernel: torch.Tensor) -> torch.Tensor:
    return kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()


def _double_center_distance_matrix(distances: torch.Tensor) -> torch.Tensor:
    return distances - distances.mean(dim=0, keepdim=True) - distances.mean(dim=1, keepdim=True) + distances.mean()


def _compute_label_curve_distance_matrix(
    feature_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    label_count: int,
    grid_size: int,
) -> torch.Tensor:
    if label_count < 2:
        return torch.zeros((label_count, label_count), dtype=feature_tensor.dtype, device=feature_tensor.device)
    primary_feature = feature_tensor[:, 0]
    grid = torch.linspace(
        primary_feature.min(),
        primary_feature.max(),
        steps=grid_size,
        dtype=feature_tensor.dtype,
        device=feature_tensor.device,
    )
    profiles: list[torch.Tensor] = []
    global_mean = target_tensor.mean()
    for label_index in range(label_count):
        indices = torch.nonzero(label_tensor == label_index, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            profiles.append(torch.full_like(grid, global_mean))
            continue
        label_x = primary_feature[indices]
        label_y = target_tensor[indices]
        order = torch.argsort(label_x)
        sorted_x = label_x[order]
        sorted_y = label_y[order]
        unique_x, inverse = torch.unique_consecutive(sorted_x, return_inverse=True)
        if unique_x.numel() != sorted_x.numel():
            summed = torch.zeros_like(unique_x)
            counts = torch.zeros_like(unique_x)
            summed.scatter_add_(0, inverse, sorted_y)
            counts.scatter_add_(0, inverse, torch.ones_like(sorted_y))
            sorted_x = unique_x
            sorted_y = summed / counts.clamp_min(EPSILON)
        profiles.append(_interp1d_torch(grid, sorted_x, sorted_y))
    profile_matrix = torch.stack(profiles, dim=0)
    profile_matrix = profile_matrix - profile_matrix.mean(dim=1, keepdim=True)
    scale = profile_matrix.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(EPSILON)
    profile_matrix = profile_matrix / scale
    distances = torch.cdist(profile_matrix, profile_matrix, p=2)
    return _normalize_distance_matrix(distances).detach()


def _interp1d_torch(grid: torch.Tensor, x_values: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
    if x_values.numel() == 1:
        return torch.full_like(grid, y_values[0])
    positions = torch.searchsorted(x_values, grid, right=False).clamp(1, x_values.numel() - 1)
    x0 = x_values[positions - 1]
    x1 = x_values[positions]
    y0 = y_values[positions - 1]
    y1 = y_values[positions]
    weights = (grid - x0) / (x1 - x0).clamp_min(EPSILON)
    interpolated = y0 + weights * (y1 - y0)
    return torch.where(
        grid <= x_values[0],
        y_values[0],
        torch.where(grid >= x_values[-1], y_values[-1], interpolated),
    )


def _latent_curve_continuity_penalty(q_values: torch.Tensor, curve_distances: torch.Tensor) -> torch.Tensor:
    if q_values.shape[0] < 2:
        return q_values.new_tensor(0.0)
    q_distances = _normalize_distance_matrix(torch.cdist(q_values, q_values, p=2))
    mask = ~torch.eye(q_values.shape[0], dtype=torch.bool, device=q_values.device)
    return (q_distances[mask] - curve_distances[mask]).pow(2).mean()


def _normalize_distance_matrix(distances: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
    if not torch.any(mask):
        return torch.zeros_like(distances)
    selected = distances[mask]
    mean = selected.mean()
    std = selected.std(unbiased=False).clamp_min(EPSILON)
    return (distances - mean) / std


def _mds_init_from_distances(distances: torch.Tensor, q_dim: int) -> torch.Tensor:
    n = distances.shape[0]
    if n <= q_dim:
        return torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1

    high_dim = n
    low_dim = q_dim

    normalized_high_dist = distances / high_dim
    normalized_high_dist = normalized_high_dist.clamp_min(EPSILON)

    q_coords = torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    q_coords.requires_grad_(True)

    optimizer = optim.Adam([q_coords], lr=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    for epoch in range(2000):
        optimizer.zero_grad()

        q_diff = q_coords.unsqueeze(0) - q_coords.unsqueeze(1)
        low_dist = q_diff.abs().sum(dim=-1)

        normalized_low_dist = low_dist / low_dim
        normalized_low_dist = normalized_low_dist.clamp_min(EPSILON)

        ratio1 = normalized_high_dist / normalized_low_dist
        ratio2 = normalized_low_dist / normalized_high_dist

        ratio_sum = ratio1 + ratio2
        ratio_sum = ratio_sum.clamp_max(50.0)

        loss = torch.exp(ratio_sum).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([q_coords], max_norm=1.0)
        optimizer.step()
        scheduler.step()

    q_coords = q_coords.detach()
    q_coords = (q_coords - q_coords.mean(dim=0, keepdim=True)) / q_coords.std(dim=0, keepdim=True).clamp_min(EPSILON)
    return q_coords


def _numpy_column_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.ndim == 1:
        left_array = left_array.reshape(-1, 1)
    if right_array.ndim == 1:
        right_array = right_array.reshape(-1, 1)
    if left_array.shape[0] != right_array.shape[0]:
        raise ValueError("Arrays must have the same number of rows.")
    if left_array.shape[0] < 2:
        return np.zeros((left_array.shape[1], right_array.shape[1]), dtype=np.float64)

    left_centered = left_array - left_array.mean(axis=0, keepdims=True)
    right_centered = right_array - right_array.mean(axis=0, keepdims=True)
    left_scale = np.sqrt(np.mean(left_centered**2, axis=0, keepdims=True))
    right_scale = np.sqrt(np.mean(right_centered**2, axis=0, keepdims=True))
    left_normalized = left_centered / np.maximum(left_scale, EPSILON)
    right_normalized = right_centered / np.maximum(right_scale, EPSILON)
    return left_normalized.T @ right_normalized / left_array.shape[0]


def _is_early_stop_epoch(epoch_metrics: EpochMetrics, config: LatentQConfig) -> bool:
    if not config.early_stop_enabled:
        return False
    return epoch_metrics.r2 >= config.early_stop_r2_threshold


def _validate_matching_feature_dimensions(train_dataset: LatentQDataset, test_dataset: LatentQDataset) -> None:
    if train_dataset.features.shape[1] != test_dataset.features.shape[1]:
        raise ValueError(
            "train and test datasets must have the same feature dimension. "
            f"Got {train_dataset.features.shape[1]} and {test_dataset.features.shape[1]}."
        )


def _normalize_column_name(column_name: Any, fallback_index: int) -> str:
    if isinstance(column_name, str) and column_name:
        return column_name
    return f"col_{fallback_index}"


def _normalize_label_value(label: Any) -> Any:
    return label.item() if isinstance(label, np.generic) else label


def _resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_device = torch.device(device_name)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")
    return resolved_device


def _set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_index(index: int, n_columns: int) -> int:
    resolved_index = index if index >= 0 else n_columns + index
    if resolved_index < 0 or resolved_index >= n_columns:
        raise IndexError(f"Column index {index} is out of range for a dataframe with {n_columns} columns.")
    return resolved_index


def _ensure_prediction_column(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 2 or predictions.shape[1] != 1:
        raise ValueError(
            "Model predictions must have shape (batch_size, 1) or (batch_size,). "
            f"Received shape {tuple(predictions.shape)}."
        )
    return predictions


def _compute_train_epoch_metrics(
    *,
    epoch_number: int,
    model: nn.Module,
    embedding: nn.Embedding,
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    targets_original: np.ndarray,
    target_scale_factor: float,
    normalizer: NormalizationStats,
) -> EpochMetrics:
    model.eval()
    with torch.no_grad():
        q_values = embedding(label_tensor)
        model_inputs = torch.cat([feature_tensor, q_values], dim=1)
        predictions = _ensure_prediction_column(model(model_inputs)).squeeze(1)
    predictions_model_scale = denormalize_targets(predictions.cpu().numpy(), normalizer)
    denormalized_predictions = predictions_model_scale / target_scale_factor
    return EpochMetrics(
        epoch=epoch_number,
        r2=float(r2_score(targets_original, denormalized_predictions)),
        mse=float(mean_squared_error(targets_original * target_scale_factor, predictions_model_scale)),
        mse_original=float(mean_squared_error(targets_original, denormalized_predictions)),
    )


def _resolve_plot_feature_index(plot_feature_index: Optional[int], feature_dim: int) -> int:
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    resolved_index = plot_feature_index if plot_feature_index is not None else min(1, feature_dim - 1)
    if resolved_index < 0 or resolved_index >= feature_dim:
        raise IndexError(f"plot_feature_index {resolved_index} is out of range for feature_dim={feature_dim}.")
    return resolved_index


def _initial_q_vector(label: Any, training_artifacts: TrainingArtifacts, q_dim: int) -> torch.Tensor:
    if label in training_artifacts.label_to_index:
        return training_artifacts.embedding.weight[training_artifacts.label_to_index[label]].detach().clone()
    return torch.randn(q_dim, dtype=torch.float32) * 0.1


__all__ = [
    "CSVColumnConfig",
    "CalibrationArtifacts",
    "EpochMetrics",
    "LatentQConfig",
    "LatentQDataset",
    "LatentQPipelineResult",
    "ModelFactory",
    "NormalizationStats",
    "OutputConfig",
    "TrainingArtifacts",
    "add_common_cli_arguments",
    "assemble_output_frame",
    "build_dataset_from_arrays",
    "build_dataset_from_dataframe",
    "build_q_matrix",
    "calibrate_latent_q_for_test_labels",
    "compute_latent_feature_orthogonality_metrics",
    "denormalize_targets",
    "extract_train_q_matrix",
    "fit_normalization",
    "load_csv_dataset",
    "namespace_to_shared_configs",
    "normalize_features",
    "normalize_targets",
    "parse_index_list",
    "plot_prediction_curve",
    "run_latent_q_pipeline",
    "save_output_frame",
    "save_pipeline_outputs",
    "scale_dataset_targets",
    "split_calibration_and_eval_indices",
    "train_latent_q_model",
]
