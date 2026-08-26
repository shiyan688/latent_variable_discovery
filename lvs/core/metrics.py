"""Reusable prediction and latent-space evaluation metrics.

Alignment objects deliberately separate ``fit`` from ``transform`` so callers can
fit on validation labels and evaluate on held-out test labels without leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_EPS = np.finfo(float).eps


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _paired(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_array = _matrix(left, "left")
    right_array = _matrix(right, "right")
    if left_array.shape[0] != right_array.shape[0]:
        raise ValueError("Paired arrays must have the same number of rows.")
    return left_array, right_array


def macro_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Average prediction metrics over labels, giving each label equal weight."""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    label_array = np.asarray(labels).reshape(-1)
    if not (truth.size == prediction.size == label_array.size) or truth.size == 0:
        raise ValueError("y_true, y_pred, and labels must have the same non-zero length.")

    per_label: list[tuple[float, float, float, float]] = []
    for label in np.unique(label_array):
        selected = label_array == label
        label_truth = truth[selected]
        label_prediction = prediction[selected]
        mse = float(mean_squared_error(label_truth, label_prediction))
        mae = float(mean_absolute_error(label_truth, label_prediction))
        scale = float(np.std(label_truth))
        nrmse = float(np.sqrt(mse) / scale) if scale > _EPS else float("nan")
        r2 = float(r2_score(label_truth, label_prediction)) if selected.sum() >= 2 else float("nan")
        per_label.append((mse, mae, nrmse, r2))
    values = np.asarray(per_label, dtype=float)
    finite_nrmse = values[:, 2][np.isfinite(values[:, 2])]
    finite_r2 = values[:, 3][np.isfinite(values[:, 3])]
    return {
        "macro_mse": float(values[:, 0].mean()),
        "macro_rmse": float(np.sqrt(values[:, 0]).mean()),
        "macro_mae": float(values[:, 1].mean()),
        "macro_nrmse": float(finite_nrmse.mean()) if finite_nrmse.size else float("nan"),
        "macro_r2": float(finite_r2.mean()) if finite_r2.size else float("nan"),
    }


def reference_scaled_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    reference_scale: float,
) -> dict[str, float]:
    """Prediction metrics normalized by a scale fixed independently of the test labels.

    ``reference_scale`` should normally be the training-target standard deviation.  This
    avoids the pathological behavior of per-label NRMSE when a held-out response curve
    is (nearly) constant.
    """
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have the same non-zero length.")
    scale = float(reference_scale)
    if not np.isfinite(scale) or scale <= _EPS:
        raise ValueError("reference_scale must be a positive finite number.")
    residual = prediction - truth
    mse = float(np.mean(residual**2))
    return {
        "reference_mse": mse,
        "reference_rmse": float(np.sqrt(mse)),
        "reference_mae": float(np.mean(np.abs(residual))),
        "reference_nrmse": float(np.sqrt(mse) / scale),
    }


@dataclass(frozen=True)
class AffineAlignment:
    """Least-squares map from learned coordinates to reference coordinates."""

    coefficients: np.ndarray
    intercept: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return _matrix(values, "values") @ self.coefficients + self.intercept


def fit_affine_alignment(source: np.ndarray, target: np.ndarray) -> AffineAlignment:
    source_array, target_array = _paired(source, target)
    design = np.column_stack([source_array, np.ones(source_array.shape[0])])
    solution, _, _, _ = np.linalg.lstsq(design, target_array, rcond=None)
    return AffineAlignment(coefficients=solution[:-1], intercept=solution[-1])


def apply_affine_alignment(alignment: AffineAlignment, values: np.ndarray) -> np.ndarray:
    return alignment.transform(values)


@dataclass(frozen=True)
class ProcrustesAlignment:
    source_mean: np.ndarray
    target_mean: np.ndarray
    rotation: np.ndarray
    scale: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ( _matrix(values, "values") - self.source_mean) @ self.rotation * self.scale + self.target_mean


def fit_procrustes_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_scaling: bool = True,
) -> ProcrustesAlignment:
    source_array, target_array = _paired(source, target)
    if source_array.shape[1] != target_array.shape[1]:
        raise ValueError("Procrustes alignment requires equal coordinate dimensions.")
    source_mean = source_array.mean(axis=0)
    target_mean = target_array.mean(axis=0)
    source_centered = source_array - source_mean
    target_centered = target_array - target_mean
    left, singular_values, right_t = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = left @ right_t
    denominator = float(np.sum(source_centered**2))
    scale = float(singular_values.sum() / denominator) if allow_scaling and denominator > _EPS else 1.0
    return ProcrustesAlignment(source_mean, target_mean, rotation, scale)


def procrustes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    allow_scaling: bool = True,
) -> tuple[np.ndarray, float]:
    """Fit and apply Procrustes alignment, returning aligned values and disparity."""
    alignment = fit_procrustes_alignment(source, target, allow_scaling=allow_scaling)
    aligned = alignment.transform(source)
    target_array = _matrix(target, "target")
    denominator = max(float(np.sum((target_array - target_array.mean(axis=0)) ** 2)), _EPS)
    disparity = float(np.sum((aligned - target_array) ** 2) / denominator)
    return aligned, disparity


@dataclass(frozen=True)
class CCAAlignment:
    """CCA projections fitted on one split and reusable on held-out data."""

    left_mean: np.ndarray
    left_scale: np.ndarray
    left_rotations: np.ndarray
    right_mean: np.ndarray
    right_scale: np.ndarray
    right_rotations: np.ndarray

    def apply(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project paired data without refitting the alignment."""
        left_array, right_array = _paired(left, right)
        if left_array.shape[1] != self.left_mean.size:
            raise ValueError("left has a different number of columns than the fitted data.")
        if right_array.shape[1] != self.right_mean.size:
            raise ValueError("right has a different number of columns than the fitted data.")
        left_canonical = (
            (left_array - self.left_mean) / self.left_scale
        ) @ self.left_rotations
        right_canonical = (
            (right_array - self.right_mean) / self.right_scale
        ) @ self.right_rotations
        return left_canonical, right_canonical

    def score(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return per-component correlations using the fixed projections."""
        return _canonical_component_correlations(*self.apply(left, right))


def _canonical_component_correlations(
    left_canonical: np.ndarray,
    right_canonical: np.ndarray,
) -> np.ndarray:
    correlations = []
    for index in range(left_canonical.shape[1]):
        if np.std(left_canonical[:, index]) <= _EPS or np.std(right_canonical[:, index]) <= _EPS:
            correlations.append(0.0)
        else:
            correlations.append(
                float(np.corrcoef(left_canonical[:, index], right_canonical[:, index])[0, 1])
            )
    return np.asarray(correlations)


def fit_cca_alignment(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_components: int | None = None,
) -> CCAAlignment:
    """Fit CCA parameters, typically on validation data only."""
    left_array, right_array = _paired(left, right)
    max_components = min(left_array.shape[1], right_array.shape[1], left_array.shape[0] - 1)
    components = max_components if n_components is None else min(int(n_components), max_components)
    if components < 1:
        raise ValueError("CCA alignment requires at least two rows and one component.")
    model = CCA(n_components=components, max_iter=1000)
    model.fit(left_array, right_array)
    return CCAAlignment(
        left_mean=np.array(model._x_mean, copy=True),
        left_scale=np.array(model._x_std, copy=True),
        left_rotations=np.array(model.x_rotations_, copy=True),
        right_mean=np.array(model._y_mean, copy=True),
        right_scale=np.array(model._y_std, copy=True),
        right_rotations=np.array(model.y_rotations_, copy=True),
    )


def apply_cca_alignment(
    alignment: CCAAlignment,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply validation-fitted CCA projections to another paired split."""
    return alignment.apply(left, right)


def score_cca_alignment(
    alignment: CCAAlignment,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Score another paired split without changing the fitted parameters."""
    return alignment.score(left, right)


def canonical_correlations(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_components: int | None = None,
) -> np.ndarray:
    """Return in-sample CCA correlations by fitting and scoring the same data.

    Use ``fit_cca_alignment`` followed by ``score_cca_alignment`` for
    validation-fit/test-score evaluation without test leakage.
    """
    left_array, right_array = _paired(left, right)
    max_components = min(left_array.shape[1], right_array.shape[1], left_array.shape[0] - 1)
    components = max_components if n_components is None else min(int(n_components), max_components)
    if components < 1:
        return np.empty(0, dtype=float)
    alignment = fit_cca_alignment(left_array, right_array, n_components=components)
    return alignment.score(left_array, right_array)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    array = _matrix(values, "values")
    differences = array[:, None, :] - array[None, :, :]
    matrix = np.sqrt(np.sum(differences**2, axis=2))
    return matrix[np.triu_indices(array.shape[0], k=1)]


def _pairwise_distance_matrix(values: np.ndarray) -> np.ndarray:
    array = _matrix(values, "values")
    differences = array[:, None, :] - array[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def _neighbor_order_and_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = _pairwise_distance_matrix(values)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for row_index in range(order.shape[0]):
        ranks[row_index, order[row_index]] = np.arange(1, order.shape[1] + 1)
    return order, ranks


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def pairwise_distance_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Compare pairwise geometry using normalized stress and correlations."""
    left_array, right_array = _paired(left, right)
    if left_array.shape[0] < 2:
        raise ValueError("At least two rows are required for pairwise distances.")
    left_distances = _pairwise_distances(left_array)
    right_distances = _pairwise_distances(right_array)
    scale = float(np.dot(left_distances, right_distances) / max(np.dot(left_distances, left_distances), _EPS))
    stress = float(np.sqrt(np.sum((scale * left_distances - right_distances) ** 2) / max(np.sum(right_distances**2), _EPS)))

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) <= _EPS or np.std(b) <= _EPS:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "distance_stress": stress,
        "distance_pearson": correlation(left_distances, right_distances),
        "distance_spearman": correlation(_rank(left_distances), _rank(right_distances)),
    }


def knn_overlap(left: np.ndarray, right: np.ndarray, *, k: int = 5) -> float:
    """Mean fraction of shared k-nearest neighbors between two representations."""
    left_array, right_array = _paired(left, right)
    if left_array.shape[0] < 2:
        raise ValueError("At least two rows are required for kNN overlap.")
    resolved_k = min(int(k), left_array.shape[0] - 1)
    if resolved_k < 1:
        raise ValueError("k must be positive.")
    left_distances = np.linalg.norm(left_array[:, None, :] - left_array[None, :, :], axis=2)
    right_distances = np.linalg.norm(right_array[:, None, :] - right_array[None, :, :], axis=2)
    np.fill_diagonal(left_distances, np.inf)
    np.fill_diagonal(right_distances, np.inf)
    left_neighbors = np.argsort(left_distances, axis=1, kind="stable")[:, :resolved_k]
    right_neighbors = np.argsort(right_distances, axis=1, kind="stable")[:, :resolved_k]
    overlaps = [len(set(a).intersection(b)) / resolved_k for a, b in zip(left_neighbors, right_neighbors)]
    return float(np.mean(overlaps))


def trustworthiness_continuity(
    reference: np.ndarray,
    learned: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float]:
    """Return rank-weighted neighborhood trustworthiness and continuity.

    Trustworthiness penalizes false neighbors (folds); continuity penalizes missing
    reference neighbors (tears).  Rows in both matrices must refer to the same labels.
    """
    reference_array, learned_array = _paired(reference, learned)
    sample_count = reference_array.shape[0]
    resolved_k = int(k)
    if sample_count < 3:
        raise ValueError("At least three rows are required for neighborhood metrics.")
    if resolved_k < 1 or resolved_k >= sample_count / 2:
        raise ValueError("k must be positive and smaller than half the sample count.")

    reference_order, reference_ranks = _neighbor_order_and_ranks(reference_array)
    learned_order, learned_ranks = _neighbor_order_and_ranks(learned_array)
    trust_penalty = 0.0
    continuity_penalty = 0.0
    overlap = 0.0
    for index in range(sample_count):
        reference_neighbors = set(reference_order[index, :resolved_k])
        learned_neighbors = set(learned_order[index, :resolved_k])
        overlap += len(reference_neighbors & learned_neighbors) / resolved_k
        for neighbor in learned_neighbors - reference_neighbors:
            trust_penalty += float(reference_ranks[index, neighbor] - resolved_k)
        for neighbor in reference_neighbors - learned_neighbors:
            continuity_penalty += float(learned_ranks[index, neighbor] - resolved_k)

    factor = 2.0 / (sample_count * resolved_k * (2 * sample_count - 3 * resolved_k - 1))
    return {
        "trustworthiness": float(np.clip(1.0 - factor * trust_penalty, 0.0, 1.0)),
        "continuity": float(np.clip(1.0 - factor * continuity_penalty, 0.0, 1.0)),
        "knn_overlap": float(overlap / sample_count),
    }


def neighborhood_preservation_curve(
    reference: np.ndarray,
    learned: np.ndarray,
    *,
    max_k: int = 10,
) -> list[dict[str, float]]:
    """Evaluate neighborhood preservation for every valid ``k`` up to ``max_k``."""
    reference_array, learned_array = _paired(reference, learned)
    largest_valid_k = min(int(max_k), (reference_array.shape[0] - 1) // 2)
    if largest_valid_k < 1:
        raise ValueError("At least three rows are required for a neighborhood curve.")
    rows = []
    for k in range(1, largest_valid_k + 1):
        rows.append({"k": float(k), **trustworthiness_continuity(reference_array, learned_array, k=k)})
    return rows


def local_distance_distortion(
    reference: np.ndarray,
    learned: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, float]:
    """Measure scale-adjusted distortion on reference-space nearest-neighbor edges."""
    reference_array, learned_array = _paired(reference, learned)
    sample_count = reference_array.shape[0]
    resolved_k = min(int(k), sample_count - 1)
    if sample_count < 2 or resolved_k < 1:
        raise ValueError("At least two rows and a positive k are required.")

    reference_distances = _pairwise_distance_matrix(reference_array)
    learned_distances = _pairwise_distance_matrix(learned_array)
    upper = np.triu_indices(sample_count, k=1)
    reference_global = reference_distances[upper]
    learned_global = learned_distances[upper]
    scale = float(
        np.dot(learned_global, reference_global)
        / max(float(np.dot(learned_global, learned_global)), _EPS)
    )

    np.fill_diagonal(reference_distances, np.inf)
    neighbors = np.argsort(reference_distances, axis=1, kind="stable")[:, :resolved_k]
    edge_pairs = {
        tuple(sorted((index, int(neighbor))))
        for index in range(sample_count)
        for neighbor in neighbors[index]
        if index != int(neighbor)
    }
    ratios = []
    for left, right in sorted(edge_pairs):
        denominator = float(reference_distances[left, right])
        if denominator <= _EPS:
            continue
        ratios.append((scale * float(learned_distances[left, right]) + _EPS) / denominator)
    if not ratios:
        return {
            "local_log_distortion_mean": float("nan"),
            "local_log_distortion_median": float("nan"),
            "local_log_distortion_p95": float("nan"),
            "local_collapse_rate": float("nan"),
            "local_tear_rate": float("nan"),
        }
    ratio_array = np.asarray(ratios, dtype=float)
    absolute_log_ratio = np.abs(np.log(ratio_array))
    return {
        "local_log_distortion_mean": float(absolute_log_ratio.mean()),
        "local_log_distortion_median": float(np.median(absolute_log_ratio)),
        "local_log_distortion_p95": float(np.quantile(absolute_log_ratio, 0.95)),
        "local_collapse_rate": float(np.mean(ratio_array < 0.5)),
        "local_tear_rate": float(np.mean(ratio_array > 2.0)),
    }


def grouped_rff_signatures(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_components: int = 64,
    seed: int = 0,
    bandwidth_sample_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic label-level kernel-mean signatures for multivariate rows."""
    array = _matrix(values, "values")
    label_array = np.asarray(labels).reshape(-1)
    if array.shape[0] != label_array.size:
        raise ValueError("values and labels must have the same number of rows.")
    if n_components < 2:
        raise ValueError("n_components must be at least 2.")
    standardized = (array - array.mean(axis=0)) / np.maximum(array.std(axis=0), 1e-8)
    if standardized.shape[0] > bandwidth_sample_size:
        sample_indices = np.linspace(
            0, standardized.shape[0] - 1, bandwidth_sample_size, dtype=int
        )
        bandwidth_sample = standardized[sample_indices]
    else:
        bandwidth_sample = standardized
    sample_distances = _pairwise_distances(bandwidth_sample)
    positive_distances = sample_distances[sample_distances > _EPS]
    bandwidth = float(np.median(positive_distances)) if positive_distances.size else 1.0
    rng = np.random.default_rng(seed)
    frequencies = rng.normal(
        size=(standardized.shape[1], n_components)
    ) / max(bandwidth, 1e-8)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    features = np.sqrt(2.0 / n_components) * np.cos(standardized @ frequencies + phases)
    unique_labels = pd_unique(label_array)
    signatures = np.vstack([features[label_array == label].mean(axis=0) for label in unique_labels])
    return unique_labels, signatures


def pd_unique(values: np.ndarray) -> np.ndarray:
    """Stable unique values without adding a pandas dependency to this module."""
    return np.asarray(list(dict.fromkeys(np.asarray(values).reshape(-1).tolist())), dtype=object)


def effective_rank(values: np.ndarray) -> float:
    """Entropy effective rank of a centered representation."""
    array = _matrix(values, "values")
    singular_values = np.linalg.svd(array - array.mean(axis=0), compute_uv=False)
    weights = singular_values**2
    if weights.sum() <= _EPS:
        return 0.0
    probabilities = weights / weights.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def alignment_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Coordinate metrics for already aligned latent values."""
    truth, prediction = _paired(y_true, y_pred)
    residual = prediction - truth
    variance = float(np.mean((truth - truth.mean(axis=0)) ** 2))
    return {
        "aligned_mse": float(np.mean(residual**2)),
        "aligned_nrmse": float(np.sqrt(np.mean(residual**2) / max(variance, _EPS))),
        "aligned_r2": float(r2_score(truth, prediction, multioutput="variance_weighted")),
    }


__all__ = [
    "AffineAlignment", "CCAAlignment", "ProcrustesAlignment", "alignment_metrics",
    "apply_affine_alignment", "apply_cca_alignment", "canonical_correlations", "effective_rank",
    "fit_affine_alignment", "fit_cca_alignment", "fit_procrustes_alignment", "grouped_rff_signatures",
    "knn_overlap", "local_distance_distortion", "macro_prediction_metrics",
    "neighborhood_preservation_curve", "pairwise_distance_metrics", "procrustes",
    "reference_scaled_prediction_metrics", "score_cca_alignment", "trustworthiness_continuity",
]
