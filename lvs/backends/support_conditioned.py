from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lvs.core.pipeline import (
    LatentQConfig,
    NormalizationStats,
    TrainingArtifacts,
    build_dataset_from_arrays,
    calibrate_latent_q_for_test_labels,
    denormalize_targets,
    fit_normalization,
    normalize_features,
    normalize_targets,
    split_support_query_indices,
)


@dataclass(frozen=True)
class SupportModelConfig:
    representation_dim: int = 8
    epochs: int = 200
    lr: float = 1e-3
    support_ratio: float = 0.3
    entity_batch_size: int = 8
    encoder_hidden_sizes: tuple[int, ...] = (128, 128)
    decoder_hidden_sizes: tuple[int, ...] = (256, 128)
    target_coordinate_mode: str = "global"
    target_loss: str = "mse"
    support_scale_floor_fraction: float = 0.05
    support_target_clip: float = 8.0
    smooth_l1_beta: float = 1.0
    standardized_output_bound: float | None = None
    seed: int = 0
    device: str = "cuda:0"


@dataclass
class DirectSupportArtifacts:
    encoder: "DeepSetEncoder"
    decoder: nn.Module
    normalizer: NormalizationStats
    device: torch.device
    train_loss_history: list[float]


@dataclass
class AttentiveSupportArtifacts:
    model: "AttentiveConditionalRegressor"
    normalizer: NormalizationStats
    device: torch.device
    train_loss_history: list[float]
    config: SupportModelConfig
    target_scale_floor: float


@dataclass
class AttentiveSelectorArtifacts:
    global_artifacts: AttentiveSupportArtifacts
    bounded_artifacts: AttentiveSupportArtifacts


@dataclass
class SupportPrediction:
    predictions: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    query_indices: np.ndarray
    representation_labels: np.ndarray
    representations: np.ndarray
    diagnostics_by_label: dict[Any, dict[str, float]]


@dataclass
class QEncoderArtifacts:
    encoder: "DeepSetEncoder"
    device: torch.device
    train_loss_history: list[float]
    reconstruction_loss_history: list[float]
    alignment_loss_history: list[float]


@dataclass
class QEncoderPrediction:
    initial: SupportPrediction
    refined: SupportPrediction
    calibration_seconds: float


@dataclass
class QMultistartPrediction:
    prediction: SupportPrediction
    calibration_seconds: float


class DeepSetEncoder(nn.Module):
    """Permutation-invariant encoder for a set of normalized (x, y) pairs."""

    def __init__(
        self,
        pair_input_dim: int,
        representation_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        resolved = tuple(int(value) for value in hidden_sizes)
        if not resolved or any(value <= 0 for value in resolved):
            raise ValueError("hidden_sizes must contain positive integers")
        layers: list[nn.Module] = []
        previous = pair_input_dim
        for width in resolved:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.pair_encoder = nn.Sequential(*layers)
        self.set_head = nn.Sequential(
            nn.Linear(previous, resolved[-1]),
            nn.ReLU(),
            nn.Linear(resolved[-1], representation_dim),
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor) -> torch.Tensor:
        if support_x.ndim != 2:
            raise ValueError("support_x must be a two-dimensional tensor")
        targets = support_y.reshape(-1, 1)
        if support_x.shape[0] != targets.shape[0] or support_x.shape[0] == 0:
            raise ValueError("support_x and support_y must contain the same non-zero row count")
        encoded = self.pair_encoder(torch.cat([support_x, targets], dim=1))
        return self.set_head(encoded.mean(dim=0))


class AttentiveConditionalRegressor(nn.Module):
    """Query-to-support cross-attention with a permutation-invariant global state."""

    def __init__(
        self,
        feature_dim: int,
        representation_dim: int,
        encoder_hidden_sizes: Sequence[int] = (128, 128),
        decoder_hidden_sizes: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        resolved = tuple(int(value) for value in encoder_hidden_sizes)
        if not resolved or any(value <= 0 for value in resolved):
            raise ValueError("encoder_hidden_sizes must contain positive integers")
        attention_dim = resolved[-1]
        prefix = resolved[:-1]
        self.attention_dim = attention_dim
        self.key_encoder = build_vector_mlp(feature_dim, prefix, attention_dim)
        self.query_encoder = build_vector_mlp(feature_dim, prefix, attention_dim)
        self.value_encoder = build_vector_mlp(feature_dim + 1, prefix, attention_dim)
        self.global_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, representation_dim),
        )
        self.decoder = build_mlp(
            feature_dim + attention_dim + representation_dim,
            decoder_hidden_sizes,
        )

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if support_x.ndim != 2 or query_x.ndim != 2 or support_x.shape[0] == 0:
            raise ValueError("support_x and query_x must be non-empty two-dimensional tensors")
        targets = support_y.reshape(-1, 1)
        if support_x.shape[0] != targets.shape[0]:
            raise ValueError("support_x and support_y must contain the same row count")
        keys = self.key_encoder(support_x)
        values = self.value_encoder(torch.cat([support_x, targets], dim=1))
        queries = self.query_encoder(query_x)
        scores = queries @ keys.transpose(0, 1) / float(self.attention_dim) ** 0.5
        weights = torch.softmax(scores, dim=1)
        attended = weights @ values
        representation = self.global_head(values.mean(dim=0))
        repeated = representation.unsqueeze(0).expand(len(query_x), -1)
        prediction = self.decoder(torch.cat([query_x, attended, repeated], dim=1)).squeeze(1)
        return prediction, representation, weights


def build_mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Module:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        resolved = int(width)
        if resolved <= 0:
            raise ValueError("hidden_sizes must contain positive integers")
        layers.extend((nn.Linear(previous, resolved), nn.ReLU()))
        previous = resolved
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def build_vector_mlp(
    input_dim: int, hidden_sizes: Sequence[int], output_dim: int
) -> nn.Module:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        resolved = int(width)
        if resolved <= 0:
            raise ValueError("hidden_sizes must contain positive integers")
        layers.extend((nn.Linear(previous, resolved), nn.ReLU()))
        previous = resolved
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def _validate_attentive_config(config: SupportModelConfig) -> None:
    if config.target_coordinate_mode not in {"global", "support_robust"}:
        raise ValueError("target_coordinate_mode must be 'global' or 'support_robust'")
    if config.target_loss not in {"mse", "smooth_l1"}:
        raise ValueError("target_loss must be 'mse' or 'smooth_l1'")
    if config.support_scale_floor_fraction <= 0:
        raise ValueError("support_scale_floor_fraction must be positive")
    if config.support_target_clip <= 0:
        raise ValueError("support_target_clip must be positive")
    if config.smooth_l1_beta <= 0:
        raise ValueError("smooth_l1_beta must be positive")
    if config.standardized_output_bound is not None and config.standardized_output_bound <= 0:
        raise ValueError("standardized_output_bound must be positive when provided")


def _training_target_scale_floor(targets: np.ndarray, fraction: float) -> float:
    values = np.asarray(targets, dtype=np.float64)
    center = float(np.median(values))
    robust_scale = 1.4826 * float(np.median(np.abs(values - center)))
    epsilon = float(np.finfo(np.float32).eps * max(1.0, np.max(np.abs(values))))
    return max(fraction * robust_scale, epsilon)


def _support_target_coordinates(
    raw_targets: torch.Tensor,
    support: np.ndarray,
    *,
    scale_floor: float,
    context_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_targets = raw_targets[support]
    center = torch.quantile(support_targets, 0.5)
    mad = torch.quantile(torch.abs(support_targets - center), 0.5)
    scale = (1.4826 * mad).clamp_min(scale_floor)
    standardized = (support_targets - center) / scale
    context = standardized.clamp(-context_clip, context_clip)
    return center, scale, standardized, context


def _bounded_standardized_prediction(
    raw_prediction: torch.Tensor, bound: float | None
) -> torch.Tensor:
    if bound is None:
        return raw_prediction
    return bound * torch.tanh(raw_prediction / bound)


def _attentive_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: SupportModelConfig,
) -> torch.Tensor:
    if config.target_loss == "mse":
        return torch.mean((prediction - target).pow(2))
    return F.smooth_l1_loss(
        prediction,
        target,
        beta=config.smooth_l1_beta,
        reduction="mean",
    )


def train_attentive_cnp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_labels: np.ndarray,
    config: SupportModelConfig,
) -> AttentiveSupportArtifacts:
    _validate_training_arrays(train_x, train_y, train_labels)
    _validate_attentive_config(config)
    device = _resolve_device(config.device)
    _seed_all(config.seed)
    normalizer = fit_normalization(train_x, train_y)
    features = torch.tensor(
        normalize_features(train_x, normalizer), dtype=torch.float32, device=device
    )
    targets = torch.tensor(
        normalize_targets(train_y, normalizer), dtype=torch.float32, device=device
    )
    raw_targets = torch.tensor(train_y, dtype=torch.float32, device=device)
    target_scale_floor = _training_target_scale_floor(
        train_y, config.support_scale_floor_fraction
    )
    model = AttentiveConditionalRegressor(
        train_x.shape[1],
        config.representation_dim,
        config.encoder_hidden_sizes,
        config.decoder_hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    groups = _label_groups(train_labels)
    history: list[float] = []
    for epoch in range(config.epochs):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch, 161803]))
        ordered_labels = list(rng.permutation(np.asarray(list(groups), dtype=object)))
        epoch_losses: list[float] = []
        accumulated: list[torch.Tensor] = []
        optimizer.zero_grad(set_to_none=True)
        for position, label in enumerate(ordered_labels, start=1):
            support, query = _episode_split(groups[label], config.support_ratio, rng)
            if config.target_coordinate_mode == "support_robust":
                center, scale, _, context_targets = _support_target_coordinates(
                    raw_targets,
                    support,
                    scale_floor=target_scale_floor,
                    context_clip=config.support_target_clip,
                )
                query_targets = (raw_targets[query] - center) / scale
            else:
                context_targets = targets[support]
                query_targets = targets[query]
            raw_prediction, _, _ = model(
                features[support], context_targets, features[query]
            )
            prediction = _bounded_standardized_prediction(
                raw_prediction, config.standardized_output_bound
            )
            loss = _attentive_prediction_loss(prediction, query_targets, config)
            accumulated.append(loss)
            epoch_losses.append(float(loss.detach().cpu().item()))
            if len(accumulated) >= config.entity_batch_size or position == len(ordered_labels):
                torch.stack(accumulated).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated.clear()
        history.append(float(np.mean(epoch_losses)))
    return AttentiveSupportArtifacts(
        model=model,
        normalizer=normalizer,
        device=device,
        train_loss_history=history,
        config=config,
        target_scale_floor=target_scale_floor,
    )


def predict_attentive_cnp(
    artifacts: AttentiveSupportArtifacts,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    *,
    support_ratio: float,
    seed: int,
) -> SupportPrediction:
    features = torch.tensor(
        normalize_features(test_x, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    targets_normalized = torch.tensor(
        normalize_targets(test_y, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    raw_targets = torch.tensor(test_y, dtype=torch.float32, device=artifacts.device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    query_indices: list[np.ndarray] = []
    representation_labels: list[Any] = []
    representations: list[np.ndarray] = []
    diagnostics: dict[Any, dict[str, float]] = {}
    artifacts.model.eval()
    with torch.no_grad():
        for raw_label, indices in _label_groups(test_labels).items():
            support, query = split_support_query_indices(
                indices, support_ratio, mode="random", seed=seed, label=raw_label
            )
            coordinate_diagnostics: dict[str, float] = {}
            if artifacts.config.target_coordinate_mode == "support_robust":
                center, scale, support_standardized, context_targets = (
                    _support_target_coordinates(
                        raw_targets,
                        support,
                        scale_floor=artifacts.target_scale_floor,
                        context_clip=artifacts.config.support_target_clip,
                    )
                )
            else:
                center = scale = None
                support_standardized = None
                context_targets = targets_normalized[support]
            raw_prediction, representation, attention = artifacts.model(
                features[support], context_targets, features[query]
            )
            normalized_prediction = _bounded_standardized_prediction(
                raw_prediction, artifacts.config.standardized_output_bound
            )
            if center is not None and scale is not None:
                physical_prediction = center + scale * normalized_prediction
                coordinate_diagnostics = {
                    "support_target_center": float(center.cpu().item()),
                    "support_target_scale": float(scale.cpu().item()),
                    "support_standardized_min": float(
                        support_standardized.min().cpu().item()
                    ),
                    "support_standardized_max": float(
                        support_standardized.max().cpu().item()
                    ),
                    "prediction_standardized_min": float(
                        normalized_prediction.min().cpu().item()
                    ),
                    "prediction_standardized_max": float(
                        normalized_prediction.max().cpu().item()
                    ),
                }
                if artifacts.config.standardized_output_bound is not None:
                    bound = artifacts.config.standardized_output_bound
                    coordinate_diagnostics.update(
                        {
                            "prediction_physical_lower_bound": float(
                                (center - bound * scale).cpu().item()
                            ),
                            "prediction_physical_upper_bound": float(
                                (center + bound * scale).cpu().item()
                            ),
                        }
                    )
                predictions.append(physical_prediction.cpu().numpy())
            else:
                predictions.append(
                    denormalize_targets(
                        normalized_prediction.cpu().numpy(), artifacts.normalizer
                    )
                )
            targets.append(np.asarray(test_y)[query])
            labels.append(np.asarray(test_labels)[query])
            query_indices.append(query)
            representation_labels.append(raw_label)
            representations.append(representation.cpu().numpy())
            entropy = -torch.sum(
                attention * torch.log(attention.clamp_min(1e-12)), dim=1
            )
            diagnostics[raw_label] = {
                "support_rows": float(len(support)),
                "query_rows": float(len(query)),
                "attention_entropy_mean": float(entropy.mean().cpu().item()),
                "attention_max_weight_mean": float(
                    attention.max(dim=1).values.mean().cpu().item()
                ),
                **coordinate_diagnostics,
            }
    return SupportPrediction(
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        labels=np.concatenate(labels),
        query_indices=np.concatenate(query_indices),
        representation_labels=np.asarray(representation_labels),
        representations=np.vstack(representations).astype(np.float32),
        diagnostics_by_label=diagnostics,
    )


def train_attentive_reliability_selector(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_labels: np.ndarray,
    config: SupportModelConfig,
) -> AttentiveSelectorArtifacts:
    """Train the frozen expressive and bounded components used by the selector."""
    global_config = replace(
        config,
        target_coordinate_mode="global",
        target_loss="mse",
        standardized_output_bound=None,
    )
    bounded_config = replace(
        config,
        target_coordinate_mode="support_robust",
        target_loss="smooth_l1",
        standardized_output_bound=8.0,
    )
    return AttentiveSelectorArtifacts(
        global_artifacts=train_attentive_cnp(
            train_x, train_y, train_labels, global_config
        ),
        bounded_artifacts=train_attentive_cnp(
            train_x, train_y, train_labels, bounded_config
        ),
    )


def _predict_attentive_episode(
    artifacts: AttentiveSupportArtifacts,
    features: torch.Tensor,
    normalized_targets: torch.Tensor,
    raw_targets: torch.Tensor,
    support: np.ndarray,
    query: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    coordinate_diagnostics: dict[str, float] = {}
    if artifacts.config.target_coordinate_mode == "support_robust":
        center, scale, support_standardized, context_targets = (
            _support_target_coordinates(
                raw_targets,
                support,
                scale_floor=artifacts.target_scale_floor,
                context_clip=artifacts.config.support_target_clip,
            )
        )
    else:
        center = scale = None
        support_standardized = None
        context_targets = normalized_targets[support]
    raw_prediction, representation, attention = artifacts.model(
        features[support], context_targets, features[query]
    )
    standardized_prediction = _bounded_standardized_prediction(
        raw_prediction, artifacts.config.standardized_output_bound
    )
    if center is not None and scale is not None:
        physical_prediction = center + scale * standardized_prediction
        coordinate_diagnostics = {
            "support_target_center": float(center.cpu().item()),
            "support_target_scale": float(scale.cpu().item()),
            "support_standardized_min": float(
                support_standardized.min().cpu().item()
            ),
            "support_standardized_max": float(
                support_standardized.max().cpu().item()
            ),
            "prediction_standardized_min": float(
                standardized_prediction.min().cpu().item()
            ),
            "prediction_standardized_max": float(
                standardized_prediction.max().cpu().item()
            ),
        }
        if artifacts.config.standardized_output_bound is not None:
            bound = artifacts.config.standardized_output_bound
            coordinate_diagnostics.update(
                {
                    "prediction_physical_lower_bound": float(
                        (center - bound * scale).cpu().item()
                    ),
                    "prediction_physical_upper_bound": float(
                        (center + bound * scale).cpu().item()
                    ),
                }
            )
    else:
        physical_prediction = torch.tensor(
            denormalize_targets(
                standardized_prediction.cpu().numpy(), artifacts.normalizer
            ),
            dtype=torch.float32,
            device=artifacts.device,
        )
    return physical_prediction, representation, attention, coordinate_diagnostics


def predict_attentive_reliability_selector(
    artifacts: AttentiveSelectorArtifacts,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    *,
    support_ratio: float,
    seed: int,
    selector_ratio: float = 0.25,
    selector_min_rows: int = 8,
) -> SupportPrediction:
    """Choose the global or bounded component using support-internal validation."""
    if not 0 < selector_ratio < 1:
        raise ValueError("selector_ratio must be between zero and one")
    if selector_min_rows < 1:
        raise ValueError("selector_min_rows must be positive")
    global_artifacts = artifacts.global_artifacts
    bounded_artifacts = artifacts.bounded_artifacts
    global_features = torch.tensor(
        normalize_features(test_x, global_artifacts.normalizer),
        dtype=torch.float32,
        device=global_artifacts.device,
    )
    bounded_features = torch.tensor(
        normalize_features(test_x, bounded_artifacts.normalizer),
        dtype=torch.float32,
        device=bounded_artifacts.device,
    )
    global_normalized_targets = torch.tensor(
        normalize_targets(test_y, global_artifacts.normalizer),
        dtype=torch.float32,
        device=global_artifacts.device,
    )
    bounded_normalized_targets = torch.tensor(
        normalize_targets(test_y, bounded_artifacts.normalizer),
        dtype=torch.float32,
        device=bounded_artifacts.device,
    )
    global_raw_targets = torch.tensor(
        test_y, dtype=torch.float32, device=global_artifacts.device
    )
    bounded_raw_targets = torch.tensor(
        test_y, dtype=torch.float32, device=bounded_artifacts.device
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    query_indices: list[np.ndarray] = []
    representation_labels: list[Any] = []
    representations: list[np.ndarray] = []
    diagnostics: dict[Any, dict[str, float]] = {}
    global_artifacts.model.eval()
    bounded_artifacts.model.eval()
    with torch.no_grad():
        for raw_label, indices in _label_groups(test_labels).items():
            support, query = split_support_query_indices(
                indices, support_ratio, mode="random", seed=seed, label=raw_label
            )
            if len(support) < 2:
                raise ValueError("Reliability selection needs at least two support rows")
            selection_rows = min(
                max(selector_min_rows, int(np.floor(selector_ratio * len(support)))),
                len(support) - 1,
            )
            internal_ratio = (selection_rows + 0.25) / len(support)
            selection, fit = split_support_query_indices(
                support,
                internal_ratio,
                mode="random",
                seed=seed + 104729,
                label=raw_label,
            )
            global_selection, _, _, _ = _predict_attentive_episode(
                global_artifacts,
                global_features,
                global_normalized_targets,
                global_raw_targets,
                fit,
                selection,
            )
            bounded_selection, _, _, _ = _predict_attentive_episode(
                bounded_artifacts,
                bounded_features,
                bounded_normalized_targets,
                bounded_raw_targets,
                fit,
                selection,
            )
            selection_targets = global_raw_targets[selection]
            global_score = torch.mean(torch.abs(global_selection - selection_targets))
            bounded_score = torch.mean(
                torch.abs(bounded_selection.to(global_artifacts.device) - selection_targets)
            )
            use_bounded = bool((bounded_score <= global_score).cpu().item())
            if use_bounded:
                prediction, representation, attention, coordinate_diagnostics = (
                    _predict_attentive_episode(
                        bounded_artifacts,
                        bounded_features,
                        bounded_normalized_targets,
                        bounded_raw_targets,
                        support,
                        query,
                    )
                )
            else:
                prediction, representation, attention, coordinate_diagnostics = (
                    _predict_attentive_episode(
                        global_artifacts,
                        global_features,
                        global_normalized_targets,
                        global_raw_targets,
                        support,
                        query,
                    )
                )
            if not torch.isfinite(global_score) or not torch.isfinite(bounded_score):
                raise FloatingPointError("Non-finite reliability selector score")
            entropy = -torch.sum(
                attention * torch.log(attention.clamp_min(1e-12)), dim=1
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(np.asarray(test_y)[query])
            labels.append(np.asarray(test_labels)[query])
            query_indices.append(query)
            representation_labels.append(raw_label)
            representations.append(representation.cpu().numpy())
            diagnostics[raw_label] = {
                "support_rows": float(len(support)),
                "query_rows": float(len(query)),
                "selector_fit_rows": float(len(fit)),
                "selector_validation_rows": float(len(selection)),
                "global_selector_mae": float(global_score.cpu().item()),
                "bounded_selector_mae": float(bounded_score.cpu().item()),
                "selector_mae_margin": float(
                    (global_score - bounded_score).cpu().item()
                ),
                "selected_bounded": float(use_bounded),
                "attention_entropy_mean": float(entropy.mean().cpu().item()),
                "attention_max_weight_mean": float(
                    attention.max(dim=1).values.mean().cpu().item()
                ),
                **coordinate_diagnostics,
            }
    return SupportPrediction(
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        labels=np.concatenate(labels),
        query_indices=np.concatenate(query_indices),
        representation_labels=np.asarray(representation_labels),
        representations=np.vstack(representations).astype(np.float32),
        diagnostics_by_label=diagnostics,
    )


def train_deepsets_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_labels: np.ndarray,
    config: SupportModelConfig,
) -> DirectSupportArtifacts:
    _validate_training_arrays(train_x, train_y, train_labels)
    device = _resolve_device(config.device)
    _seed_all(config.seed)
    normalizer = fit_normalization(train_x, train_y)
    features = torch.tensor(
        normalize_features(train_x, normalizer), dtype=torch.float32, device=device
    )
    targets = torch.tensor(
        normalize_targets(train_y, normalizer), dtype=torch.float32, device=device
    )
    encoder = DeepSetEncoder(
        pair_input_dim=train_x.shape[1] + 1,
        representation_dim=config.representation_dim,
        hidden_sizes=config.encoder_hidden_sizes,
    ).to(device)
    decoder = build_mlp(
        train_x.shape[1] + config.representation_dim,
        config.decoder_hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters()], lr=config.lr
    )
    groups = _label_groups(train_labels)
    history: list[float] = []
    for epoch in range(config.epochs):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch, 314159]))
        epoch_losses: list[float] = []
        ordered_labels = list(rng.permutation(np.asarray(list(groups), dtype=object)))
        optimizer.zero_grad(set_to_none=True)
        accumulated: list[torch.Tensor] = []
        for position, label in enumerate(ordered_labels, start=1):
            support, query = _episode_split(groups[label], config.support_ratio, rng)
            context = encoder(features[support], targets[support])
            repeated = context.unsqueeze(0).expand(len(query), -1)
            prediction = decoder(torch.cat([features[query], repeated], dim=1)).squeeze(1)
            loss = torch.mean((prediction - targets[query]).pow(2))
            accumulated.append(loss)
            epoch_losses.append(float(loss.detach().cpu().item()))
            if len(accumulated) >= config.entity_batch_size or position == len(ordered_labels):
                torch.stack(accumulated).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated.clear()
        history.append(float(np.mean(epoch_losses)))
    return DirectSupportArtifacts(
        encoder=encoder,
        decoder=decoder,
        normalizer=normalizer,
        device=device,
        train_loss_history=history,
    )


def predict_deepsets_regressor(
    artifacts: DirectSupportArtifacts,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    *,
    support_ratio: float,
    seed: int,
) -> SupportPrediction:
    normalized_x = torch.tensor(
        normalize_features(test_x, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    normalized_y = torch.tensor(
        normalize_targets(test_y, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    artifacts.encoder.eval()
    artifacts.decoder.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    query_indices: list[np.ndarray] = []
    representation_labels: list[Any] = []
    representations: list[np.ndarray] = []
    diagnostics: dict[Any, dict[str, float]] = {}
    with torch.no_grad():
        for raw_label, indices in _label_groups(test_labels).items():
            support, query = split_support_query_indices(
                indices, support_ratio, mode="random", seed=seed, label=raw_label
            )
            context = artifacts.encoder(normalized_x[support], normalized_y[support])
            repeated = context.unsqueeze(0).expand(len(query), -1)
            normalized_prediction = artifacts.decoder(
                torch.cat([normalized_x[query], repeated], dim=1)
            ).squeeze(1)
            prediction = denormalize_targets(
                normalized_prediction.cpu().numpy(), artifacts.normalizer
            )
            predictions.append(prediction)
            targets.append(np.asarray(test_y)[query])
            labels.append(np.asarray(test_labels)[query])
            query_indices.append(query)
            representation_labels.append(raw_label)
            representations.append(context.cpu().numpy())
            diagnostics[raw_label] = {
                "support_rows": float(len(support)),
                "query_rows": float(len(query)),
            }
    return SupportPrediction(
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        labels=np.concatenate(labels),
        query_indices=np.concatenate(query_indices),
        representation_labels=np.asarray(representation_labels),
        representations=np.vstack(representations).astype(np.float32),
        diagnostics_by_label=diagnostics,
    )


def train_q_support_encoder(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_labels: np.ndarray,
    training_artifacts: TrainingArtifacts,
    config: SupportModelConfig,
    *,
    alignment_weight: float = 0.05,
) -> QEncoderArtifacts:
    _validate_training_arrays(train_x, train_y, train_labels)
    device = training_artifacts.device
    _seed_all(config.seed)
    features = torch.tensor(
        normalize_features(train_x, training_artifacts.normalizer),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        normalize_targets(train_y, training_artifacts.normalizer),
        dtype=torch.float32,
        device=device,
    )
    encoder = DeepSetEncoder(
        pair_input_dim=train_x.shape[1] + 1,
        representation_dim=config.representation_dim,
        hidden_sizes=config.encoder_hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config.lr)
    decoder = training_artifacts.model
    decoder.eval()
    decoder_parameters = tuple(decoder.parameters())
    previous_requires_grad = tuple(parameter.requires_grad for parameter in decoder_parameters)
    for parameter in decoder_parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    train_q = training_artifacts.embedding.weight.detach()
    q_scale = train_q.std(dim=0, unbiased=False).clamp_min(0.05)
    groups = _label_groups(train_labels)
    total_history: list[float] = []
    reconstruction_history: list[float] = []
    alignment_history: list[float] = []
    try:
        for epoch in range(config.epochs):
            rng = np.random.default_rng(np.random.SeedSequence([config.seed, epoch, 271828]))
            ordered_labels = list(rng.permutation(np.asarray(list(groups), dtype=object)))
            epoch_total: list[float] = []
            epoch_reconstruction: list[float] = []
            epoch_alignment: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            accumulated: list[torch.Tensor] = []
            for position, label in enumerate(ordered_labels, start=1):
                support, query = _episode_split(groups[label], config.support_ratio, rng)
                q_value = encoder(features[support], targets[support])
                repeated = q_value.unsqueeze(0).expand(len(query), -1)
                prediction = decoder(torch.cat([features[query], repeated], dim=1)).squeeze(1)
                reconstruction = torch.mean((prediction - targets[query]).pow(2))
                label_key = _label_key(label)
                target_q = train_q[training_artifacts.label_to_index[label_key]]
                alignment = torch.mean(((q_value - target_q) / q_scale).pow(2))
                loss = reconstruction + alignment_weight * alignment
                accumulated.append(loss)
                epoch_total.append(float(loss.detach().cpu().item()))
                epoch_reconstruction.append(float(reconstruction.detach().cpu().item()))
                epoch_alignment.append(float(alignment.detach().cpu().item()))
                if len(accumulated) >= config.entity_batch_size or position == len(ordered_labels):
                    torch.stack(accumulated).mean().backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated.clear()
            total_history.append(float(np.mean(epoch_total)))
            reconstruction_history.append(float(np.mean(epoch_reconstruction)))
            alignment_history.append(float(np.mean(epoch_alignment)))
    finally:
        for parameter, requires_grad in zip(decoder_parameters, previous_requires_grad):
            parameter.requires_grad_(requires_grad)
        for parameter in decoder_parameters:
            parameter.grad = None
    return QEncoderArtifacts(
        encoder=encoder,
        device=device,
        train_loss_history=total_history,
        reconstruction_loss_history=reconstruction_history,
        alignment_loss_history=alignment_history,
    )


def predict_q_encoder(
    encoder_artifacts: QEncoderArtifacts,
    training_artifacts: TrainingArtifacts,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    *,
    support_ratio: float,
    seed: int,
    refine_steps: int = 50,
    refine_lr: float = 0.02,
    trust_region_weight: float = 0.01,
    clip_standard_deviations: float = 3.0,
) -> QEncoderPrediction:
    import time

    normalized_x = torch.tensor(
        normalize_features(test_x, training_artifacts.normalizer),
        dtype=torch.float32,
        device=training_artifacts.device,
    )
    normalized_y = torch.tensor(
        normalize_targets(test_y, training_artifacts.normalizer),
        dtype=torch.float32,
        device=training_artifacts.device,
    )
    encoder_artifacts.encoder.eval()
    decoder = training_artifacts.model
    decoder.eval()
    decoder_parameters = tuple(decoder.parameters())
    previous_requires_grad = tuple(parameter.requires_grad for parameter in decoder_parameters)
    for parameter in decoder_parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    train_q = training_artifacts.embedding.weight.detach()
    q_mean = train_q.mean(dim=0)
    q_std = train_q.std(dim=0, unbiased=False).clamp_min(0.05)
    lower = q_mean - clip_standard_deviations * q_std
    upper = q_mean + clip_standard_deviations * q_std
    initial_predictions: list[np.ndarray] = []
    refined_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    query_indices: list[np.ndarray] = []
    representation_labels: list[Any] = []
    initial_q_values: list[np.ndarray] = []
    refined_q_values: list[np.ndarray] = []
    diagnostics: dict[Any, dict[str, float]] = {}
    started = time.perf_counter()
    try:
        for raw_label, indices in _label_groups(test_labels).items():
            support, query = split_support_query_indices(
                indices, support_ratio, mode="random", seed=seed, label=raw_label
            )
            with torch.no_grad():
                q_initial = encoder_artifacts.encoder(
                    normalized_x[support], normalized_y[support]
                ).clamp(lower, upper)
                repeated = q_initial.unsqueeze(0).expand(len(query), -1)
                initial_normalized = decoder(
                    torch.cat([normalized_x[query], repeated], dim=1)
                ).squeeze(1)
                initial_support_prediction = decoder(
                    torch.cat(
                        [normalized_x[support], q_initial.unsqueeze(0).expand(len(support), -1)],
                        dim=1,
                    )
                ).squeeze(1)
                initial_support_loss = torch.mean(
                    (initial_support_prediction - normalized_y[support]).pow(2)
                )
            q_parameter = nn.Parameter(q_initial.detach().clone())
            optimizer = torch.optim.Adam([q_parameter], lr=refine_lr)
            for _ in range(refine_steps):
                repeated_support = q_parameter.unsqueeze(0).expand(len(support), -1)
                support_prediction = decoder(
                    torch.cat([normalized_x[support], repeated_support], dim=1)
                ).squeeze(1)
                reconstruction = torch.mean(
                    (support_prediction - normalized_y[support]).pow(2)
                )
                trust_region = torch.mean(((q_parameter - q_initial) / q_std).pow(2))
                loss = reconstruction + trust_region_weight * trust_region
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    q_parameter.clamp_(lower, upper)
            q_refined = q_parameter.detach()
            with torch.no_grad():
                repeated = q_refined.unsqueeze(0).expand(len(query), -1)
                refined_normalized = decoder(
                    torch.cat([normalized_x[query], repeated], dim=1)
                ).squeeze(1)
                refined_support_prediction = decoder(
                    torch.cat(
                        [normalized_x[support], q_refined.unsqueeze(0).expand(len(support), -1)],
                        dim=1,
                    )
                ).squeeze(1)
                refined_support_loss = torch.mean(
                    (refined_support_prediction - normalized_y[support]).pow(2)
                )
            initial_predictions.append(
                denormalize_targets(initial_normalized.cpu().numpy(), training_artifacts.normalizer)
            )
            refined_predictions.append(
                denormalize_targets(refined_normalized.cpu().numpy(), training_artifacts.normalizer)
            )
            targets.append(np.asarray(test_y)[query])
            labels.append(np.asarray(test_labels)[query])
            query_indices.append(query)
            representation_labels.append(raw_label)
            initial_q_values.append(q_initial.cpu().numpy())
            refined_q_values.append(q_refined.cpu().numpy())
            diagnostics[raw_label] = {
                "support_rows": float(len(support)),
                "query_rows": float(len(query)),
                "initial_support_loss": float(initial_support_loss.cpu().item()),
                "refined_support_loss": float(refined_support_loss.cpu().item()),
                "q_movement": float(torch.linalg.vector_norm(q_refined - q_initial).cpu().item()),
            }
    finally:
        for parameter, requires_grad in zip(decoder_parameters, previous_requires_grad):
            parameter.requires_grad_(requires_grad)
        for parameter in decoder_parameters:
            parameter.grad = None
    common = {
        "targets": np.concatenate(targets),
        "labels": np.concatenate(labels),
        "query_indices": np.concatenate(query_indices),
        "representation_labels": np.asarray(representation_labels),
        "diagnostics_by_label": diagnostics,
    }
    initial = SupportPrediction(
        predictions=np.concatenate(initial_predictions),
        representations=np.vstack(initial_q_values).astype(np.float32),
        **common,
    )
    refined = SupportPrediction(
        predictions=np.concatenate(refined_predictions),
        representations=np.vstack(refined_q_values).astype(np.float32),
        **common,
    )
    return QEncoderPrediction(
        initial=initial,
        refined=refined,
        calibration_seconds=time.perf_counter() - started,
    )


def predict_q_encoder_multistart(
    encoder_artifacts: QEncoderArtifacts,
    training_artifacts: TrainingArtifacts,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_labels: np.ndarray,
    *,
    config: LatentQConfig,
    clip_standard_deviations: float = 3.0,
) -> QMultistartPrediction:
    """Add a fit-support-only encoder candidate to the standard multistart selector."""
    import time

    normalized_x = torch.tensor(
        normalize_features(test_x, training_artifacts.normalizer),
        dtype=torch.float32,
        device=training_artifacts.device,
    )
    normalized_y = torch.tensor(
        normalize_targets(test_y, training_artifacts.normalizer),
        dtype=torch.float32,
        device=training_artifacts.device,
    )
    train_q = training_artifacts.embedding.weight.detach()
    q_mean = train_q.mean(dim=0)
    q_std = train_q.std(dim=0, unbiased=False).clamp_min(0.05)
    lower = q_mean - clip_standard_deviations * q_std
    upper = q_mean + clip_standard_deviations * q_std
    encoder_artifacts.encoder.eval()

    def provider(_label: Any, fit_indices: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return encoder_artifacts.encoder(
                normalized_x[fit_indices], normalized_y[fit_indices]
            ).clamp(lower, upper)

    test_dataset = build_dataset_from_arrays(test_x, test_labels, test_y)
    started = time.perf_counter()
    calibration = calibrate_latent_q_for_test_labels(
        test_dataset,
        training_artifacts,
        config,
        extra_initial_q_provider=provider,
    )
    calibration_seconds = time.perf_counter() - started
    representation_labels: list[Any] = []
    representations: list[np.ndarray] = []
    diagnostics: dict[Any, dict[str, float]] = {}
    for raw_label, indices in _label_groups(test_labels).items():
        key = _label_key(raw_label)
        support, query = split_support_query_indices(
            indices,
            config.calibration_ratio,
            mode=config.calibration_split_mode,
            seed=config.seed,
            label=key,
        )
        representation_labels.append(raw_label)
        representations.append(calibration.q_by_label[key])
        diagnostics[key] = {
            **calibration.diagnostics_by_label[key],
            "support_rows": float(len(support)),
            "query_rows": float(len(query)),
        }
    prediction = SupportPrediction(
        predictions=calibration.eval_predictions,
        targets=calibration.eval_targets,
        labels=calibration.eval_labels,
        query_indices=calibration.eval_indices,
        representation_labels=np.asarray(representation_labels),
        representations=np.vstack(representations).astype(np.float32),
        diagnostics_by_label=diagnostics,
    )
    return QMultistartPrediction(prediction, calibration_seconds)


def _label_groups(labels: np.ndarray) -> dict[Any, np.ndarray]:
    array = np.asarray(labels)
    return {
        _label_key(label): np.flatnonzero(array == label).astype(np.int64)
        for label in pd.unique(array)
    }


def _episode_split(
    indices: np.ndarray, support_ratio: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 2:
        raise ValueError("Each entity needs at least two rows")
    split = min(max(1, int(np.floor(support_ratio * len(indices)))), len(indices) - 1)
    ordered = rng.permutation(indices)
    return ordered[:split], ordered[split:]


def _label_key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _resolve_device(raw: str) -> torch.device:
    return torch.device(raw if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_training_arrays(
    features: np.ndarray, targets: np.ndarray, labels: np.ndarray
) -> None:
    if np.asarray(features).ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if not (len(features) == len(targets) == len(labels)):
        raise ValueError("features, targets, and labels must have equal row counts")
    if any(len(indices) < 2 for indices in _label_groups(labels).values()):
        raise ValueError("Each entity needs at least two rows")


__all__ = [
    "AttentiveConditionalRegressor",
    "AttentiveSupportArtifacts",
    "AttentiveSelectorArtifacts",
    "DeepSetEncoder",
    "DirectSupportArtifacts",
    "QEncoderArtifacts",
    "QEncoderPrediction",
    "SupportModelConfig",
    "SupportPrediction",
    "build_mlp",
    "predict_attentive_cnp",
    "predict_attentive_reliability_selector",
    "predict_deepsets_regressor",
    "predict_q_encoder",
    "predict_q_encoder_multistart",
    "train_attentive_cnp",
    "train_attentive_reliability_selector",
    "train_deepsets_regressor",
    "train_q_support_encoder",
]
