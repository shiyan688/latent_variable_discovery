#!/usr/bin/env python3
"""Run one deterministic cell of the affine-gauge canonical-response benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
EXPECTED_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
FORMAL_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
Q_DIM = 3
TRAIN_ENTITIES = 96
TEST_ENTITIES = 48
GRID_SIZE = 41
PROBE_SIZE = 81
SEEDS = tuple(range(5))

# These ranges are part of the frozen synthetic generator.  The basis functions
# below are deliberately evaluated in physical x, rather than normalized x.
FAMILY_CONFIG: Mapping[str, dict[str, object]] = {
    "polynomial": {"x_min": -1.0, "x_max": 1.0, "coefficient_ranges": ((-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0))},
    "relaxation": {"x_min": 0.0, "x_max": 5.0, "coefficient_ranges": ((-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0))},
    "thermodynamic_chart": {"x_min": 0.0, "x_max": 5.0, "coefficient_ranges": ((-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0))},
}


class SiLUDecoder(nn.Module):
    """Condition-plus-q decoder whose first layer has an explicit q block."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1 + Q_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def family_basis(family: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if family == "polynomial":
        return np.column_stack((np.ones_like(x), x, x**2))
    if family == "relaxation":
        return np.column_stack((np.ones_like(x), np.exp(-2.0 * x), 1.0 / (1.0 + x)))
    if family == "thermodynamic_chart":
        return np.column_stack((
            np.ones_like(x),
            1.0 / (x + 2.0) - 1.0 / 2.5,
            np.log((x + 2.0) / 2.5),
        ))
    raise ValueError(f"unknown family: {family}")


def generate_family_data(family: str, seed: int) -> dict[str, np.ndarray]:
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family}")
    config = FAMILY_CONFIG[family]
    x = np.linspace(float(config["x_min"]), float(config["x_max"]), GRID_SIZE)
    # The cohort is frozen per family.  Training seed controls optimization and
    # calibration only; all five formal seeds therefore share identical data.
    rng = np.random.default_rng(np.random.SeedSequence([20260829, FAMILIES.index(family)]))
    ranges = np.asarray(config["coefficient_ranges"], dtype=np.float64)
    coefficients = rng.uniform(ranges[:, 0], ranges[:, 1], size=(TRAIN_ENTITIES + TEST_ENTITIES, Q_DIM))
    noiseless = (family_basis(family, x)[None, :, :] @ coefficients[:, :, None]).squeeze(-1)
    noise_scale = 0.005 * float(noiseless.std(ddof=0))
    noisy = noiseless + rng.normal(0.0, noise_scale, size=noiseless.shape)
    return {
        "x": x,
        "coefficients": coefficients,
        "noiseless": noiseless,
        "targets": noisy,
        "noise_scale": np.asarray(noise_scale),
    }


def support_query_indices(size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.arange(size, dtype=np.int64)
    support = ordered[ordered % 4 == 0]
    query = ordered[ordered % 4 != 0]
    return support, query


def apply_affine_gauge(model: nn.Module, matrix: np.ndarray, offset: np.ndarray) -> nn.Module:
    """Return the exactly equivalent decoder for q' = matrix @ q + offset."""

    transformed = copy.deepcopy(model)
    first = transformed.network[0]
    assert isinstance(first, nn.Linear)
    matrix = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    with torch.no_grad():
        weight = first.weight.detach().clone()
        q_weight = weight[:, 1:]
        transformed_q_weight = q_weight @ torch.as_tensor(inverse, dtype=weight.dtype)
        transformed_bias = first.bias.detach().clone() - transformed_q_weight @ torch.as_tensor(
            offset, dtype=weight.dtype
        )
        first.weight[:, 1:] = transformed_q_weight
        first.bias.copy_(transformed_bias)
    return transformed


def make_affine_gauges(seed: int, count: int = 25) -> list[dict[str, np.ndarray | float | int]]:
    rng = np.random.default_rng(np.random.SeedSequence([20260829, seed, 2500]))
    gauges: list[dict[str, np.ndarray | float | int]] = []
    for gauge_id in range(count):
        left, _ = np.linalg.qr(rng.normal(size=(Q_DIM, Q_DIM)))
        right, _ = np.linalg.qr(rng.normal(size=(Q_DIM, Q_DIM)))
        singular_values = np.exp(rng.uniform(0.0, np.log(8.0), size=Q_DIM))
        singular_values[0] = 1.0
        matrix = left @ np.diag(singular_values) @ right.T
        offset = rng.normal(0.0, 0.25, size=Q_DIM)
        gauges.append(
            {
                "gauge_id": gauge_id,
                "matrix": matrix,
                "offset": offset,
                "condition_number": float(np.linalg.cond(matrix)),
            }
        )
    assert len(gauges) == 25
    assert max(float(item["condition_number"]) for item in gauges) <= 10.0
    return gauges


def _predict(model: nn.Module, x: np.ndarray, q: np.ndarray, x_mean: float, x_std: float, y_mean: float, y_std: float) -> np.ndarray:
    normalized_x = (np.asarray(x, dtype=np.float64) - x_mean) / x_std
    q_array = np.asarray(q, dtype=np.float64)
    inputs = torch.as_tensor(np.column_stack((normalized_x, np.repeat(q_array[None, :], len(normalized_x), axis=0))))
    with torch.no_grad():
        return (model(inputs).squeeze(1).cpu().numpy() * y_std) + y_mean


def calibrate_q(
    model: nn.Module,
    initial_q: np.ndarray,
    support_x: np.ndarray,
    support_y: np.ndarray,
    x_mean: float,
    x_std: float,
    y_mean: float,
    y_std: float,
    steps: int,
) -> tuple[np.ndarray, float]:
    q = nn.Parameter(torch.as_tensor(initial_q, dtype=torch.float64).clone())
    support_inputs = torch.as_tensor((support_x - x_mean) / x_std, dtype=torch.float64).reshape(-1, 1)
    support_targets = torch.as_tensor((support_y - y_mean) / y_std, dtype=torch.float64).reshape(-1, 1)
    optimizer = torch.optim.Adam([q], lr=0.05)
    for _ in range(steps):
        prediction = model(torch.cat((support_inputs, q.unsqueeze(0).repeat(len(support_inputs), 1)), dim=1))
        loss = torch.mean((prediction - support_targets) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(torch.cat((support_inputs, q.unsqueeze(0).repeat(len(support_inputs), 1)), dim=1))
        loss = float(torch.mean((prediction - support_targets) ** 2).item())
    return q.detach().cpu().numpy(), loss


def _ridge_readout(q: np.ndarray, coefficients: np.ndarray, alpha: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    mean = q.mean(axis=0)
    scale = q.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    standardized = (q - mean) / scale
    design = np.column_stack((np.ones(len(q)), standardized))
    penalty = np.diag([0.0] + [alpha] * Q_DIM)
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients)
    return weights, np.column_stack((mean, scale))


def _ridge_predict(q: np.ndarray, weights: np.ndarray, scaling: np.ndarray) -> np.ndarray:
    standardized = (q - scaling[:, 0]) / scaling[:, 1]
    return np.column_stack((np.ones(len(q)), standardized)) @ weights


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def run_cell(family: str, seed: int, output_root: Path, epochs: int = 1500, calibration_steps: int = 1200, threads: int = 1, smoke: bool = False) -> dict[str, object]:
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError("family or seed is outside the frozen benchmark")
    output_root = output_root.resolve()
    cell_root = output_root / f"{family}_seed{seed}"
    cell_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260829 + seed)
    np.random.seed(20260829 + seed)

    data = generate_family_data(family, seed)
    x = data["x"]
    train_coefficients = data["coefficients"][:TRAIN_ENTITIES]
    test_coefficients = data["coefficients"][TRAIN_ENTITIES:]
    train_targets = data["targets"][:TRAIN_ENTITIES]
    test_targets = data["targets"][TRAIN_ENTITIES:]
    train_x = np.tile(x, TRAIN_ENTITIES)
    train_y = train_targets.reshape(-1)
    x_mean = float(train_x.mean())
    x_std = float(train_x.std(ddof=0))
    y_mean = float(train_y.mean())
    y_std = float(train_y.std(ddof=0))
    model = SiLUDecoder().double()
    embedding = nn.Embedding(TRAIN_ENTITIES, Q_DIM, dtype=torch.float64)
    nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
    train_inputs = torch.as_tensor(np.column_stack(((train_x - x_mean) / x_std, np.repeat(np.arange(TRAIN_ENTITIES), GRID_SIZE))))
    train_targets_tensor = torch.as_tensor(((train_y - y_mean) / y_std).reshape(-1, 1))
    labels = torch.as_tensor(np.repeat(np.arange(TRAIN_ENTITIES), GRID_SIZE), dtype=torch.long)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(embedding.parameters()), lr=1e-3)
    training_start = time.monotonic()
    for _ in range(epochs):
        prediction = model(torch.cat((train_inputs[:, :1], embedding(labels)), dim=1))
        loss = torch.mean((prediction - train_targets_tensor) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    training_seconds = time.monotonic() - training_start
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_q = embedding.weight.detach().cpu().numpy().copy()
    ridge_weights, ridge_scaling = _ridge_readout(train_q, train_coefficients)
    support_positions, query_positions = support_query_indices()
    query_rows: list[dict[str, object]] = []
    coordinate_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    calibration_seconds = 0.0
    selected_starts: list[int] = []
    q_values: list[np.ndarray] = []
    functional_values: list[np.ndarray] = []
    structure_values: list[np.ndarray] = []
    probe_x = np.linspace(float(x.min()), float(x.max()), PROBE_SIZE)
    no_q_coefficients = np.linalg.lstsq(family_basis(family, train_x), train_y, rcond=None)[0]

    for entity_index in range(TEST_ENTITIES):
        entity_x = x
        entity_y = test_targets[entity_index]
        support_x, query_x = entity_x[support_positions], entity_x[query_positions]
        support_y, query_y = entity_y[support_positions], entity_y[query_positions]
        for position in support_positions:
            split_rows.append({"entity_id": entity_index, "position": int(position), "x": float(entity_x[position]), "role": "support"})
        for position in query_positions:
            split_rows.append({"entity_id": entity_index, "position": int(position), "x": float(entity_x[position]), "role": "query"})
        rng = np.random.default_rng(np.random.SeedSequence([20260829, seed, entity_index, 17]))
        starts = [np.zeros(Q_DIM, dtype=np.float64)] + [rng.normal(0.0, 0.1, size=Q_DIM) for _ in range(3)]
        fitted = []
        support_losses = []
        start_time = time.monotonic()
        for initial in starts:
            q_fit, support_loss = calibrate_q(model, initial, support_x, support_y, x_mean, x_std, y_mean, y_std, calibration_steps)
            fitted.append(q_fit)
            support_losses.append(support_loss)
        calibration_seconds += time.monotonic() - start_time
        selected_start = int(np.argmin(support_losses))
        selected_starts.append(selected_start)
        q_fit = fitted[selected_start]
        q_values.append(q_fit)
        decoder_probe = _predict(model, probe_x, q_fit, x_mean, x_std, y_mean, y_std)
        functional_coefficients = np.linalg.lstsq(family_basis(family, probe_x), decoder_probe, rcond=None)[0]
        structure_coefficients = np.linalg.lstsq(family_basis(family, support_x), support_y, rcond=None)[0]
        functional_values.append(functional_coefficients)
        structure_values.append(structure_coefficients)
        predictions = {
            "raw_decoder": _predict(model, query_x, q_fit, x_mean, x_std, y_mean, y_std),
            "decoder_functional": family_basis(family, query_x) @ functional_coefficients,
            "support_structure_req": family_basis(family, query_x) @ structure_coefficients,
            "no_q_global_expression": family_basis(family, query_x) @ no_q_coefficients,
        }
        raw_q_ridge = _ridge_predict(q_fit.reshape(1, -1), ridge_weights, ridge_scaling)[0]
        predictions["raw_q_ridge_diagnostic"] = family_basis(family, query_x) @ raw_q_ridge
        for method, values in predictions.items():
            for row_index, (x_value, target, prediction) in enumerate(zip(query_x, query_y, values)):
                query_rows.append({"entity_id": entity_index, "query_position": int(query_positions[row_index]), "x": float(x_value), "target": float(target), "method": method, "prediction": float(prediction)})
        coordinate_rows.append({
            "entity_id": entity_index,
            "selected_start": selected_start,
            "support_loss": support_losses[selected_start],
            **{f"generating_c{j}": float(test_coefficients[entity_index, j]) for j in range(Q_DIM)},
            **{f"raw_q{j}": float(q_fit[j]) for j in range(Q_DIM)},
            **{f"functional_c{j}": float(functional_coefficients[j]) for j in range(Q_DIM)},
            **{f"structure_c{j}": float(structure_coefficients[j]) for j in range(Q_DIM)},
        })

    q_matrix = np.asarray(q_values)
    functional_matrix = np.asarray(functional_values)
    structure_matrix = np.asarray(structure_values)
    gauge_rows: list[dict[str, object]] = []
    original_query = {entity: _predict(model, x[query_positions], q_matrix[entity], x_mean, x_std, y_mean, y_std) for entity in range(TEST_ENTITIES)}
    original_probe = {entity: _predict(model, probe_x, q_matrix[entity], x_mean, x_std, y_mean, y_std) for entity in range(TEST_ENTITIES)}
    for gauge in make_affine_gauges(seed):
        matrix = np.asarray(gauge["matrix"])
        offset = np.asarray(gauge["offset"])
        transformed_model = apply_affine_gauge(model, matrix, offset)
        transformed_q = (q_matrix @ matrix.T) + offset
        query_change = 0.0
        probe_change = 0.0
        coefficient_change = 0.0
        raw_ridge_change = 0.0
        raw_ridge_prediction_change = 0.0
        q_change = 0.0
        for entity in range(TEST_ENTITIES):
            transformed_query = _predict(transformed_model, x[query_positions], transformed_q[entity], x_mean, x_std, y_mean, y_std)
            transformed_probe = _predict(transformed_model, probe_x, transformed_q[entity], x_mean, x_std, y_mean, y_std)
            transformed_coefficients = np.linalg.lstsq(family_basis(family, probe_x), transformed_probe, rcond=None)[0]
            q_change = max(q_change, float(np.max(np.abs(transformed_q[entity] - q_matrix[entity]))))
            query_change = max(query_change, float(np.max(np.abs(transformed_query - original_query[entity]))))
            probe_change = max(probe_change, float(np.max(np.abs(transformed_probe - original_probe[entity]))))
            coefficient_change = max(coefficient_change, float(np.max(np.abs(transformed_coefficients - functional_matrix[entity]))))
            raw_ridge_change = max(raw_ridge_change, float(np.max(np.abs(_ridge_predict(transformed_q[entity].reshape(1, -1), ridge_weights, ridge_scaling)[0] - _ridge_predict(q_matrix[entity].reshape(1, -1), ridge_weights, ridge_scaling)[0]))))
            transformed_ridge_prediction = family_basis(family, x[query_positions]) @ _ridge_predict(transformed_q[entity].reshape(1, -1), ridge_weights, ridge_scaling)[0]
            original_ridge_prediction = family_basis(family, x[query_positions]) @ _ridge_predict(q_matrix[entity].reshape(1, -1), ridge_weights, ridge_scaling)[0]
            raw_ridge_prediction_change = max(raw_ridge_prediction_change, float(np.max(np.abs(transformed_ridge_prediction - original_ridge_prediction))))
        gauge_rows.append({"gauge_id": int(gauge["gauge_id"]), "condition_number": float(gauge["condition_number"]), "q_coordinate_max_abs_change": q_change, "prediction_max_abs_change": query_change, "probe_prediction_max_abs_change": probe_change, "functional_coefficient_max_abs_change": coefficient_change, "raw_q_ridge_coefficient_max_abs_change": raw_ridge_change, "raw_q_ridge_prediction_max_abs_change": raw_ridge_prediction_change})

    # Repeating calibration with query targets shifted must be exactly identical,
    # because only the support positions are passed to the optimizer.
    perturbed_targets = test_targets.copy()
    perturbed_targets[:, query_positions] += 1_000_000.0
    perturbation_q_max = 0.0
    perturbation_coefficient_max = 0.0
    perturbation_prediction_max = 0.0
    perturbation_start_max = 0.0
    for entity_index in range(TEST_ENTITIES):
        rng = np.random.default_rng(np.random.SeedSequence([20260829, seed, entity_index, 17]))
        starts = [np.zeros(Q_DIM, dtype=np.float64)] + [rng.normal(0.0, 0.1, size=Q_DIM) for _ in range(3)]
        original_losses = []
        perturbed_losses = []
        original_fits = []
        perturbed_fits = []
        for start in starts:
            original_fit, _ = calibrate_q(model, start, x[support_positions], test_targets[entity_index, support_positions], x_mean, x_std, y_mean, y_std, calibration_steps)
            perturbed_fit, _ = calibrate_q(model, start, x[support_positions], perturbed_targets[entity_index, support_positions], x_mean, x_std, y_mean, y_std, calibration_steps)
            original_fits.append(original_fit)
            perturbed_fits.append(perturbed_fit)
            original_losses.append(float(np.mean((_predict(model, x[support_positions], original_fit, x_mean, x_std, y_mean, y_std) - test_targets[entity_index, support_positions]) ** 2)))
            perturbed_losses.append(float(np.mean((_predict(model, x[support_positions], perturbed_fit, x_mean, x_std, y_mean, y_std) - perturbed_targets[entity_index, support_positions]) ** 2)))
        original_selected = int(np.argmin(original_losses))
        perturbed_selected = int(np.argmin(perturbed_losses))
        perturbation_start_max = max(perturbation_start_max, float(abs(original_selected - perturbed_selected)))
        perturbation_q_max = max(perturbation_q_max, float(np.max(np.abs(original_fits[original_selected] - perturbed_fits[perturbed_selected]))))
        original_functional = np.linalg.lstsq(family_basis(family, probe_x), _predict(model, probe_x, original_fits[original_selected], x_mean, x_std, y_mean, y_std), rcond=None)[0]
        perturbed_functional = np.linalg.lstsq(family_basis(family, probe_x), _predict(model, probe_x, perturbed_fits[perturbed_selected], x_mean, x_std, y_mean, y_std), rcond=None)[0]
        perturbation_coefficient_max = max(perturbation_coefficient_max, float(np.max(np.abs(original_functional - perturbed_functional))))
        perturbation_prediction_max = max(perturbation_prediction_max, float(np.max(np.abs(_predict(model, x[query_positions], original_fits[original_selected], x_mean, x_std, y_mean, y_std) - _predict(model, x[query_positions], perturbed_fits[perturbed_selected], x_mean, x_std, y_mean, y_std)))))
    perturbation_max = max(perturbation_start_max, perturbation_q_max, perturbation_coefficient_max, perturbation_prediction_max)

    predictions_frame = pd.DataFrame(query_rows)
    summary_rows = []
    entity_metric_rows = []
    for method, frame in predictions_frame.groupby("method", sort=True):
        summary_rows.append({"method": method, "r2": _r2(frame["target"].to_numpy(), frame["prediction"].to_numpy()), "nrmse": float(np.sqrt(np.mean((frame["target"] - frame["prediction"]) ** 2)) / y_std)})
        for entity_id, entity_frame in frame.groupby("entity_id", sort=True):
            entity_metric_rows.append({"method": method, "entity_id": int(entity_id), "r2": _r2(entity_frame["target"].to_numpy(), entity_frame["prediction"].to_numpy()), "nrmse": float(np.sqrt(np.mean((entity_frame["target"] - entity_frame["prediction"]) ** 2)) / y_std)})
    gauge_frame = pd.DataFrame(gauge_rows)
    summary = {
        "status": "success",
        "scientific_selection_eligible": (not smoke) and epochs == 1500 and calibration_steps == 1200 and output_root == FORMAL_ROOT.resolve(),
        "family": family,
        "seed": seed,
        "train_entities": TRAIN_ENTITIES,
        "test_entities": TEST_ENTITIES,
        "grid_size": GRID_SIZE,
        "probe_size": PROBE_SIZE,
        "epochs": epochs,
        "calibration_steps": calibration_steps,
        "calibration_starts": 4,
        "query_rows": int(len(predictions_frame[predictions_frame["method"] == "raw_decoder"])),
        "training_seconds": training_seconds,
        "calibration_seconds": calibration_seconds,
        "family_summary": summary_rows,
        "maximum_gauge_prediction_change": float(gauge_frame["prediction_max_abs_change"].max()),
        "maximum_gauge_functional_coefficient_change": float(gauge_frame["functional_coefficient_max_abs_change"].max()),
        "maximum_gauge_raw_q_ridge_prediction_change": float(gauge_frame["raw_q_ridge_prediction_max_abs_change"].max()),
        "maximum_gauge_raw_q_coordinate_change": float(gauge_frame["q_coordinate_max_abs_change"].max()),
        "maximum_gauge_condition_number": float(gauge_frame["condition_number"].max()),
        "gauge_count": int(len(gauge_frame)),
        "query_target_input_max_difference": perturbation_max,
        "query_target_perturbation_max_difference": perturbation_max,
        "query_target_perturbation_value": 1_000_000.0,
        "query_target_selected_start_max_difference": perturbation_start_max,
        "query_target_q_max_difference": perturbation_q_max,
        "query_target_coefficient_max_difference": perturbation_coefficient_max,
        "query_target_prediction_max_difference": perturbation_prediction_max,
        "selected_start_counts": {str(i): selected_starts.count(i) for i in range(4)},
    }
    predictions_frame.to_csv(cell_root / "query_predictions.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(cell_root / "entity_coordinates.csv", index=False)
    pd.DataFrame(split_rows).to_csv(cell_root / "support_query_split.csv", index=False)
    gauge_frame.to_csv(cell_root / "gauge_diagnostics.csv", index=False)
    pd.DataFrame(entity_metric_rows).to_csv(cell_root / "entity_metrics.csv", index=False)
    pd.DataFrame({"probe_x": probe_x}).to_csv(cell_root / "fixed_probes.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "embedding": train_q, "family": family, "seed": seed, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "gauges": make_affine_gauges(seed)}, cell_root / "artifact.pt")
    write_json(cell_root / "result.json", summary)
    manifest = {"scope": "gauge_invariant_canonical_response_benchmark_cell", "family": family, "seed": seed, "plan_sha256": sha256(PLAN), "runner_sha256": sha256(Path(__file__)), "epochs": epochs, "calibration_steps": calibration_steps, "cpu_only": True, "query_target_perturbation": 1_000_000.0, "files": {}}
    write_json(cell_root / "manifest.json", manifest)
    manifest["files"] = {path.name: sha256(path) for path in sorted(cell_root.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(cell_root / "manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--calibration-steps", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="Mark this run non-counted; explicit shortened budgets are recommended.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("frozen benchmark plan changed")
    if args.epochs <= 0 or args.calibration_steps <= 0:
        raise ValueError("epochs and calibration-steps must be positive")
    run_cell(args.family, args.seed, args.output_root, args.epochs, args.calibration_steps, args.threads, args.smoke)


if __name__ == "__main__":
    main()
