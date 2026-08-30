#!/usr/bin/env python3
"""Run the frozen gauge-equivariant calibration extension for one source cell.

The source benchmark is treated as an immutable, terminal input.  This runner
never consumes its test q or query predictions: test targets are regenerated
from the frozen synthetic generator and only support targets enter calibration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNNER_PATH = PROJECT_ROOT / "scripts/run_gauge_invariant_canonical_response_benchmark_20260829.py"
SOURCE_PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_AMENDMENT_20260829.md"
SOURCE_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
SOURCE_ANALYSIS = SOURCE_ROOT / "analysis"
FORMAL_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_extension_20260829"
FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
Q_DIM = 3
ADAM_STEPS = 300
GN_STEPS = 30
GAUGE_COUNT = 5
ENTITY_LIMIT = 48
PERTURBATION = 1_000_000.0
EXPECTED_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
EXPECTED_AMENDMENT_SHA256 = "b274f1abaee71990c5a78152d92070262caa6c37572367c60167c7bbe8fbc91f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_source_module() -> Any:
    spec = importlib.util.spec_from_file_location("gauge_source_runner", SOURCE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source runner: {SOURCE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load_source_module()


def verify_source_terminal(family: str, seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Verify the selected source cell and the independent terminal analysis."""
    if sha256(SOURCE_PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("source benchmark plan hash mismatch")
    if sha256(AMENDMENT) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("extension amendment hash mismatch")
    analysis_manifest_path = SOURCE_ANALYSIS / "manifest.json"
    decision_path = SOURCE_ANALYSIS / "decision.json"
    if not analysis_manifest_path.is_file() or not decision_path.is_file():
        raise RuntimeError("source independent analysis is not terminal")
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if analysis_manifest["plan_sha256"] != EXPECTED_PLAN_SHA256:
        raise RuntimeError("source analysis plan hash mismatch")
    if analysis_manifest["runner_sha256"] != sha256(SOURCE_RUNNER_PATH):
        raise RuntimeError("source analysis runner hash mismatch")
    for name, expected in analysis_manifest["files"].items():
        if sha256(SOURCE_ANALYSIS / name) != expected:
            raise RuntimeError(f"source analysis artifact hash mismatch: {name}")
    if decision.get("primary_gates", {}).get("all_15_cells_formal_success") is not True:
        raise RuntimeError("source independent analysis has not verified all 15 terminal cells")
    cell = SOURCE_ROOT / f"{family}_seed{seed}"
    manifest_path = cell / "manifest.json"
    result_path = cell / "result.json"
    artifact_path = cell / "artifact.pt"
    if not all(path.is_file() for path in (manifest_path, result_path, artifact_path)):
        raise RuntimeError(f"source cell is incomplete: {cell}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != EXPECTED_PLAN_SHA256 or manifest["runner_sha256"] != sha256(SOURCE_RUNNER_PATH):
        raise RuntimeError("source cell provenance hash mismatch")
    if result["status"] != "success" or result["scientific_selection_eligible"] is not True:
        raise RuntimeError("source cell is not a formal terminal success")
    if result["epochs"] != 1500 or result["calibration_steps"] != 1200 or result["gauge_count"] != 25:
        raise RuntimeError("source cell frozen budget mismatch")
    for name, expected in manifest["files"].items():
        if sha256(cell / name) != expected:
            raise RuntimeError(f"source cell artifact hash mismatch: {name}")
    return artifact_path, manifest, result


def _normalized_prediction(model: nn.Module, x: np.ndarray, q: np.ndarray, x_mean: float, x_std: float) -> np.ndarray:
    inputs = torch.as_tensor(
        np.column_stack(((np.asarray(x, dtype=np.float64) - x_mean) / x_std,
                         np.repeat(np.asarray(q, dtype=np.float64)[None, :], len(x), axis=0))),
        dtype=torch.float64,
    )
    with torch.no_grad():
        return model(inputs).squeeze(1).cpu().numpy()


def _support_loss(model: nn.Module, support_x: np.ndarray, support_y_norm: np.ndarray, q: np.ndarray, x_mean: float, x_std: float) -> float:
    prediction = _normalized_prediction(model, support_x, q, x_mean, x_std)
    return float(np.mean((prediction - support_y_norm) ** 2))


def calibrate_adam(
    model: nn.Module, initial_q: np.ndarray, support_x: np.ndarray, support_y_norm: np.ndarray,
    x_mean: float, x_std: float, steps: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    q = nn.Parameter(torch.as_tensor(initial_q, dtype=torch.float64).clone())
    support_inputs = torch.as_tensor(((support_x - x_mean) / x_std).reshape(-1, 1), dtype=torch.float64)
    support_targets = torch.as_tensor(support_y_norm.reshape(-1, 1), dtype=torch.float64)
    optimizer = torch.optim.Adam([q], lr=0.05)
    path: list[dict[str, float]] = []
    for iteration in range(steps):
        prediction = model(torch.cat((support_inputs, q.unsqueeze(0).repeat(len(support_inputs), 1)), dim=1))
        loss = torch.mean((prediction - support_targets) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        path.append({"iteration": float(iteration), "loss": float(loss.item()), "step_scale": 1.0})
    final_loss = _support_loss(model, support_x, support_y_norm, q.detach().cpu().numpy(), x_mean, x_std)
    return q.detach().cpu().numpy(), final_loss, path


def _gn_residual_jacobian(
    model: nn.Module, q: np.ndarray, support_x: np.ndarray, support_y_norm: np.ndarray, x_mean: float, x_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    q_tensor = torch.as_tensor(q, dtype=torch.float64).clone().requires_grad_(True)
    support_input = torch.as_tensor(((support_x - x_mean) / x_std), dtype=torch.float64)

    def response(current: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((support_input.reshape(-1, 1), current.unsqueeze(0).repeat(len(support_input), 1)), dim=1)
        return model(inputs).squeeze(1)

    prediction = response(q_tensor)
    jacobian = torch.autograd.functional.jacobian(response, q_tensor, create_graph=False)
    return (prediction.detach().cpu().numpy() - support_y_norm,
            jacobian.detach().cpu().numpy())


def calibrate_gauss_newton(
    model: nn.Module, initial_q: np.ndarray, support_x: np.ndarray, support_y_norm: np.ndarray,
    x_mean: float, x_std: float, steps: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    q = np.asarray(initial_q, dtype=np.float64).copy()
    path: list[dict[str, float]] = []
    line_search = [1.0 / (2**index) for index in range(8)]
    for iteration in range(steps):
        residual, jacobian = _gn_residual_jacobian(model, q, support_x, support_y_norm, x_mean, x_std)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian, tol=1e-12))
        if rank != Q_DIM:
            raise np.linalg.LinAlgError(f"Gauss-Newton support Jacobian is rank deficient at iteration {iteration}")
        delta = np.linalg.solve(jacobian.T @ jacobian, -(jacobian.T @ residual))
        before = float(np.mean(residual**2))
        accepted_scale = 0.0
        after = before
        for scale in line_search:
            candidate = q + scale * delta
            candidate_loss = _support_loss(model, support_x, support_y_norm, candidate, x_mean, x_std)
            if candidate_loss < before:
                q = candidate
                accepted_scale = scale
                after = candidate_loss
                break
        path.append({
            "iteration": float(iteration), "loss": before, "loss_after": after,
            "step_scale": accepted_scale, "jacobian_rank": float(rank),
            "jacobian_sigma_min": float(singular_values[-1]),
            "jacobian_sigma_max": float(singular_values[0]),
            "jacobian_condition": float(singular_values[0] / singular_values[-1]),
        })
    return q, _support_loss(model, support_x, support_y_norm, q, x_mean, x_std), path


def make_basis_interventions(family: str, seed: int, count: int = 10) -> list[np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([20260829, FAMILIES.index(family), seed, 2600]))
    matrices = []
    for _ in range(count):
        left, _ = np.linalg.qr(rng.normal(size=(Q_DIM, Q_DIM)))
        right, _ = np.linalg.qr(rng.normal(size=(Q_DIM, Q_DIM)))
        singular_values = np.exp(rng.uniform(0.0, np.log(8.0), size=Q_DIM))
        matrices.append(left @ np.diag(singular_values) @ right.T)
    if max(np.linalg.cond(matrix) for matrix in matrices) > 10.0:
        raise RuntimeError("basis intervention exceeded frozen condition bound")
    return matrices


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def _chart_functional(model: nn.Module, q: np.ndarray, probe_x: np.ndarray, family: str, x_mean: float, x_std: float, y_mean: float, y_std: float) -> tuple[np.ndarray, np.ndarray]:
    response = _normalized_prediction(model, probe_x, q, x_mean, x_std) * y_std + y_mean
    basis = SOURCE.family_basis(family, probe_x)
    return np.linalg.lstsq(basis, response, rcond=None)[0], response


def _readout_prediction(q: np.ndarray, train_q: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, Q_DIM)
    mean = train_q.mean(axis=0)
    scale = train_q.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(train_q)), (train_q - mean) / scale))
    penalty = np.diag([0.0] + [1e-3] * Q_DIM)
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ coefficients)
    return np.column_stack((np.ones(len(q)), (q - mean) / scale)) @ weights


def run_cell(
    family: str, seed: int, output_root: Path, *, adam_steps: int = ADAM_STEPS, gn_steps: int = GN_STEPS,
    gauge_count: int = GAUGE_COUNT, entity_limit: int = ENTITY_LIMIT, threads: int = 1, smoke: bool = False,
) -> dict[str, object]:
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError("family or seed is outside the frozen benchmark")
    if not (adam_steps > 0 and gn_steps > 0 and 0 < gauge_count <= GAUGE_COUNT and 0 < entity_limit <= ENTITY_LIMIT):
        raise ValueError("extension budgets are outside the declared range")
    source_artifact, source_manifest, source_result = verify_source_terminal(family, seed)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cell_root = output_root / f"{family}_seed{seed}"
    cell_root.mkdir(parents=False, exist_ok=False)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260829 + seed)
    np.random.seed(20260829 + seed)
    artifact = torch.load(source_artifact, map_location="cpu", weights_only=False)
    model = SOURCE.SiLUDecoder().double()
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_q = np.asarray(artifact["embedding"], dtype=np.float64)
    centroid = train_q.mean(axis=0)
    data = SOURCE.generate_family_data(family, seed)
    x = data["x"]
    test_targets = data["targets"][SOURCE.TRAIN_ENTITIES:]
    test_coefficients = data["coefficients"][SOURCE.TRAIN_ENTITIES:]
    train_coefficients = data["coefficients"][:SOURCE.TRAIN_ENTITIES]
    x_mean, x_std = float(artifact["x_mean"]), float(artifact["x_std"])
    y_mean, y_std = float(artifact["y_mean"]), float(artifact["y_std"])
    support_positions, query_positions = SOURCE.support_query_indices()
    entity_count = min(entity_limit, SOURCE.TEST_ENTITIES)
    probe_x = np.linspace(float(x.min()), float(x.max()), SOURCE.PROBE_SIZE)
    basis_probe = SOURCE.family_basis(family, probe_x)
    basis_support = SOURCE.family_basis(family, x[support_positions])
    basis_query = SOURCE.family_basis(family, x[query_positions])
    gauges = artifact["gauges"][:gauge_count]
    charts: list[tuple[int, str, nn.Module, np.ndarray, float]] = [(-1, "original", model, centroid, 1.0)]
    for gauge in gauges:
        matrix, offset = np.asarray(gauge["matrix"]), np.asarray(gauge["offset"])
        transformed = SOURCE.apply_affine_gauge(model, matrix, offset).double().eval()
        charts.append((int(gauge["gauge_id"]), f"gauge_{int(gauge['gauge_id'])}", transformed, matrix @ centroid + offset, float(gauge["condition_number"])))

    calibration_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    q_values: dict[tuple[str, int, int], np.ndarray] = {}
    functional_values: dict[tuple[str, int, int], np.ndarray] = {}
    prediction_values: dict[tuple[str, int, int], np.ndarray] = {}
    probe_response_values: dict[tuple[str, int, int], np.ndarray] = {}
    support_loss_values: dict[tuple[str, int, int], float] = {}
    query_prediction_rows: list[dict[str, object]] = []
    start_time = time.monotonic()
    for gauge_id, chart, chart_model, chart_centroid, chart_condition in charts:
        for entity in range(entity_count):
            support_x = x[support_positions]
            support_y = test_targets[entity, support_positions]
            support_y_norm = (support_y - y_mean) / y_std
            query_x = x[query_positions]
            query_y = test_targets[entity, query_positions]
            for method, budget, calibrator in (("mapped_start_adam", adam_steps, calibrate_adam), ("response_metric_gauss_newton", gn_steps, calibrate_gauss_newton)):
                started = time.monotonic()
                q_fit, support_loss, path = calibrator(chart_model, chart_centroid, support_x, support_y_norm, x_mean, x_std, budget)
                runtime = time.monotonic() - started
                coefficients, probe_response = _chart_functional(chart_model, q_fit, probe_x, family, x_mean, x_std, y_mean, y_std)
                prediction = _normalized_prediction(chart_model, query_x, q_fit, x_mean, x_std) * y_std + y_mean
                key = (method, gauge_id, entity)
                q_values[key], functional_values[key], prediction_values[key] = q_fit, coefficients, prediction
                probe_response_values[key] = probe_response
                support_loss_values[key] = support_loss
                calibration_rows.append({
                    "method": method, "gauge_id": gauge_id, "chart": chart, "entity_id": entity,
                    "gauge_condition_number": chart_condition, "support_loss": support_loss,
                    "query_r2": _r2(query_y, prediction),
                    "query_nrmse": float(np.sqrt(np.mean((query_y - prediction) ** 2)) / y_std),
                    "runtime_seconds": runtime,
                    **{f"q{i}": float(q_fit[i]) for i in range(Q_DIM)},
                    **{f"functional_c{i}": float(coefficients[i]) for i in range(Q_DIM)},
                })
                for position, x_value, target, raw_value, functional_value in zip(query_positions, query_x, query_y, prediction, basis_query @ coefficients):
                    query_prediction_rows.append({
                        "method": method, "gauge_id": gauge_id, "chart": chart, "entity_id": entity,
                        "query_position": int(position), "x": float(x_value), "target": float(target),
                        "prediction": float(raw_value), "functional_prediction": float(functional_value),
                    })
                for row in path:
                    path_rows.append({"method": method, "gauge_id": gauge_id, "chart": chart, "entity_id": entity, **row})
    calibration_seconds = time.monotonic() - start_time

    gauge_rows: list[dict[str, object]] = []
    for method in ("mapped_start_adam", "response_metric_gauss_newton"):
        for gauge_id, chart, _, _, chart_condition in charts:
            query_change = coefficient_change = q_change = loss_change = 0.0
            for entity in range(entity_count):
                original_key = (method, -1, entity)
                current_key = (method, gauge_id, entity)
                query_change = max(query_change, float(np.max(np.abs(prediction_values[current_key] - prediction_values[original_key]))))
                coefficient_change = max(coefficient_change, float(np.max(np.abs(functional_values[current_key] - functional_values[original_key]))))
                q_change = max(q_change, float(np.max(np.abs(q_values[current_key] - q_values[original_key]))))
                loss_change = max(loss_change, abs(support_loss_values[current_key] - support_loss_values[original_key]))
            gauge_rows.append({
                "method": method, "gauge_id": gauge_id, "chart": chart, "condition_number": chart_condition,
                "query_response_max_abs_difference_vs_original": query_change,
                "functional_coordinate_max_abs_difference_vs_original": coefficient_change,
                "raw_q_max_abs_change_vs_original": q_change,
                "support_loss_max_abs_difference_vs_original": loss_change,
            })

    # The query perturbation is applied to a copy, then discarded before support extraction.
    perturbed_targets = test_targets.copy()
    perturbed_targets[:, query_positions] += PERTURBATION
    perturbation_values = {"input": 0.0, "adam_path": 0.0, "gn_path": 0.0, "gn_step": 0.0, "q": 0.0, "functional": 0.0, "prediction": 0.0}
    for gauge_id, chart, chart_model, chart_centroid, _ in charts:
        for entity in range(entity_count):
            support_x = x[support_positions]
            original_support = test_targets[entity, support_positions]
            perturbed_support = perturbed_targets[entity, support_positions]
            perturbation_values["input"] = max(perturbation_values["input"], float(np.max(np.abs(original_support - perturbed_support))))
            support_original = (original_support - y_mean) / y_std
            support_perturbed = (perturbed_support - y_mean) / y_std
            for method, budget, calibrator in (("mapped_start_adam", adam_steps, calibrate_adam), ("response_metric_gauss_newton", gn_steps, calibrate_gauss_newton)):
                q_a, _, path_a = calibrator(chart_model, chart_centroid, support_x, support_original, x_mean, x_std, budget)
                q_b, _, path_b = calibrator(chart_model, chart_centroid, support_x, support_perturbed, x_mean, x_std, budget)
                key = "adam_path" if method == "mapped_start_adam" else "gn_path"
                perturbation_values[key] = max(perturbation_values[key], float(max(abs(a.get("loss", 0.0) - b.get("loss", 0.0)) for a, b in zip(path_a, path_b))))
                if method == "response_metric_gauss_newton":
                    perturbation_values["gn_step"] = max(perturbation_values["gn_step"], float(max(abs(a.get("step_scale", 0.0) - b.get("step_scale", 0.0)) for a, b in zip(path_a, path_b))))
                perturbation_values["q"] = max(perturbation_values["q"], float(np.max(np.abs(q_a - q_b))))
                coeff_a, _ = _chart_functional(chart_model, q_a, probe_x, family, x_mean, x_std, y_mean, y_std)
                coeff_b, _ = _chart_functional(chart_model, q_b, probe_x, family, x_mean, x_std, y_mean, y_std)
                perturbation_values["functional"] = max(perturbation_values["functional"], float(np.max(np.abs(coeff_a - coeff_b))))
                pred_a = _normalized_prediction(chart_model, x[query_positions], q_a, x_mean, x_std) * y_std + y_mean
                pred_b = _normalized_prediction(chart_model, x[query_positions], q_b, x_mean, x_std) * y_std + y_mean
                perturbation_values["prediction"] = max(perturbation_values["prediction"], float(np.max(np.abs(pred_a - pred_b))))
    perturbation_max = max(perturbation_values.values())

    # Basis intervention and projection/bound audits use the response-metric path.
    basis_rows: list[dict[str, object]] = []
    for basis_id, matrix in enumerate(make_basis_interventions(family, seed)):
        transformed_basis = basis_probe @ matrix
        for entity in range(entity_count):
            response = _chart_functional(model, q_values[("response_metric_gauss_newton", -1, entity)], probe_x, family, x_mean, x_std, y_mean, y_std)[1]
            c = np.linalg.lstsq(basis_probe, response, rcond=None)[0]
            c_transformed = np.linalg.lstsq(transformed_basis, response, rcond=None)[0]
            fitted_original = basis_probe @ c
            fitted_transformed = transformed_basis @ c_transformed
            original_distance = np.linalg.norm(response)
            transformed_distance = np.linalg.norm(fitted_transformed)
            reference_response = _chart_functional(model, q_values[("response_metric_gauss_newton", -1, 0)], probe_x, family, x_mean, x_std, y_mean, y_std)[1]
            reference_c = np.linalg.lstsq(basis_probe, reference_response, rcond=None)[0]
            reference_transformed_c = np.linalg.lstsq(transformed_basis, reference_response, rcond=None)[0]
            basis_rows.append({
                "basis_id": basis_id, "entity_id": entity, "basis_condition_number": float(np.linalg.cond(matrix)),
                "coordinate_max_abs_error": float(np.max(np.abs(c_transformed - np.linalg.solve(matrix, c)))),
                "function_distance_abs_error": float(np.linalg.norm(fitted_original - fitted_transformed)),
                "function_norm_distance_abs_error": float(abs(original_distance - transformed_distance)),
                "function_pair_distance_abs_error": float(abs(np.linalg.norm(fitted_original - basis_probe @ reference_c) - np.linalg.norm(fitted_transformed - transformed_basis @ reference_transformed_c))),
            })

    # Compare the frozen raw-q readout with a readout that is explicitly
    # transported through the gauge.  Both are diagnostics; neither is used by
    # either calibration path.
    raw_readout_rows: list[dict[str, object]] = []
    for method in ("mapped_start_adam", "response_metric_gauss_newton"):
        for gauge_id, chart, _, _, _ in charts:
            if gauge_id < 0:
                matrix = np.eye(Q_DIM)
                offset = np.zeros(Q_DIM)
            else:
                selected_gauge = next(gauge for gauge in gauges if int(gauge["gauge_id"]) == gauge_id)
                matrix = np.asarray(selected_gauge["matrix"], dtype=np.float64)
                offset = np.asarray(selected_gauge["offset"], dtype=np.float64)
            inverse = np.linalg.inv(matrix)
            for entity in range(entity_count):
                original_c = _readout_prediction(q_values[(method, -1, entity)], train_q, train_coefficients)[0]
                chart_q = q_values[(method, gauge_id, entity)]
                unaligned_c = _readout_prediction(chart_q, train_q, train_coefficients)[0]
                covariant_c = _readout_prediction(inverse @ (chart_q - offset), train_q, train_coefficients)[0]
                raw_readout_rows.append({
                    "method": method, "gauge_id": gauge_id, "chart": chart, "entity_id": entity,
                    "unaligned_coefficient_max_abs_change_vs_original": float(np.max(np.abs(unaligned_c - original_c))),
                    "covariant_coefficient_max_abs_change_vs_original": float(np.max(np.abs(covariant_c - original_c))),
                    "unaligned_query_response_max_abs_change_vs_original": float(np.max(np.abs(basis_query @ unaligned_c - basis_query @ original_c))),
                    "covariant_query_response_max_abs_change_vs_original": float(np.max(np.abs(basis_query @ covariant_c - basis_query @ original_c))),
                })

    bound_rows: list[dict[str, object]] = []
    narrow_rows: list[dict[str, object]] = []
    noiseless = data["noiseless"][SOURCE.TRAIN_ENTITIES:]
    for entity in range(entity_count):
        true_c = test_coefficients[entity]
        true_probe = basis_probe @ true_c
        true_support = basis_support @ true_c
        true_query = basis_query @ true_c
        support_noise = test_targets[entity, support_positions] - true_support
        query_noise = test_targets[entity, query_positions] - true_query
        support_sigma = float(np.linalg.svd(basis_support, compute_uv=False)[-1])
        query_amplification = float(np.linalg.norm(basis_query, 2) / support_sigma)
        support_structure = np.linalg.lstsq(basis_support, test_targets[entity, support_positions], rcond=None)[0]
        for gauge_id, chart, _, _, _ in charts:
            decoder_c = functional_values[("response_metric_gauss_newton", gauge_id, entity)]
            decoder_response = basis_probe @ decoder_c
            actual_decoder_response = probe_response_values[("response_metric_gauss_newton", gauge_id, entity)]
            projection_residual = float(np.linalg.norm(decoder_response - actual_decoder_response))
            projection_response_error = float(np.linalg.norm(actual_decoder_response - true_probe))
            projection_sigma = float(np.linalg.svd(basis_probe, compute_uv=False)[-1])
            projection_bound = projection_response_error / projection_sigma
            structure_error = float(np.linalg.norm(support_structure - true_c))
            actual_query_error = float(np.linalg.norm(basis_query @ support_structure - test_targets[entity, query_positions]))
            query_bound = query_amplification * float(np.linalg.norm(support_noise)) + float(np.linalg.norm(query_noise))
            bound_rows.extend([
                {"entity_id": entity, "gauge_id": gauge_id, "chart": chart, "estimate": "decoder_functional", "probe_sigma_min_weighted": projection_sigma / np.sqrt(len(probe_x)), "probe_condition_unscaled": float(np.linalg.cond(basis_probe)), "probe_condition_column_scaled": float(np.linalg.cond(basis_probe / np.linalg.norm(basis_probe, axis=0))), "projection_residual": projection_residual, "actual_generating_coefficient_error": float(np.linalg.norm(decoder_c - true_c)), "actual_decoder_probe_response_error": projection_response_error, "projection_response_error": projection_response_error, "projection_bound": projection_bound, "projection_bound_violation": float(np.linalg.norm(decoder_c - true_c) - projection_bound), "bound_ok": bool(np.linalg.norm(decoder_c - true_c) <= projection_bound + 1e-8)},
                {"entity_id": entity, "gauge_id": gauge_id, "chart": chart, "estimate": "support_structure_req", "support_sigma_min": support_sigma, "query_amplification": query_amplification, "support_residual_noise_norm": float(np.linalg.norm(support_noise)), "query_residual_noise_norm": float(np.linalg.norm(query_noise)), "query_error_bound": query_bound, "actual_query_error": actual_query_error, "query_bound_violation": actual_query_error - query_bound, "bound_ok": bool(actual_query_error <= query_bound + 1e-8), "support_condition_number": float(np.linalg.cond(basis_support))},
            ])
        narrow_positions = np.argsort(np.abs(x - np.median(x)))[:3]
        narrow_basis = SOURCE.family_basis(family, x[narrow_positions])
        singular = np.linalg.svd(narrow_basis, compute_uv=False)
        narrow_rank = int(np.linalg.matrix_rank(narrow_basis, tol=1e-12))
        narrow_condition = float(np.linalg.cond(narrow_basis))
        narrow_rows.append({"entity_id": entity, "positions": ",".join(str(int(p)) for p in narrow_positions), "rank": narrow_rank, "sigma_min": float(singular[-1]), "condition_number": narrow_condition, "status": "rank_deficient" if narrow_rank < Q_DIM else ("ill_conditioned" if narrow_condition > 1e4 else "ok"), "veto": False})

    # Response-induced entity geometry is reported in physical response space.
    geometry_rows: list[dict[str, object]] = []
    generating_responses = SOURCE.family_basis(family, probe_x) @ test_coefficients[:entity_count].T
    for method in ("mapped_start_adam", "response_metric_gauss_newton"):
        coefficients = np.asarray([functional_values[(method, -1, entity)] for entity in range(entity_count)])
        for first, second in combinations(range(entity_count), 2):
            geometry_rows.append({"method": method, "entity_first": first, "entity_second": second, "response_distance": float(np.linalg.norm(generating_responses[:, first] - generating_responses[:, second])), "estimated_distance": float(np.linalg.norm(basis_probe @ (coefficients[first] - coefficients[second])))})

    pd.DataFrame(calibration_rows).to_csv(cell_root / "calibration_diagnostics.csv", index=False)
    pd.DataFrame(query_prediction_rows).to_csv(cell_root / "query_predictions.csv", index=False)
    pd.DataFrame(path_rows).to_csv(cell_root / "calibration_paths.csv", index=False)
    pd.DataFrame(gauge_rows).to_csv(cell_root / "gauge_diagnostics.csv", index=False)
    pd.DataFrame(raw_readout_rows).to_csv(cell_root / "raw_readout_diagnostics.csv", index=False)
    pd.DataFrame(basis_rows).to_csv(cell_root / "basis_diagnostics.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(cell_root / "stability_bounds.csv", index=False)
    pd.DataFrame(narrow_rows).to_csv(cell_root / "narrow_support_diagnostics.csv", index=False)
    pd.DataFrame(geometry_rows).to_csv(cell_root / "response_geometry.csv", index=False)
    summary = {
        "status": "success", "scientific_selection_eligible": (not smoke and output_root == FORMAL_ROOT.resolve() and adam_steps == ADAM_STEPS and gn_steps == GN_STEPS and gauge_count == GAUGE_COUNT and entity_count == ENTITY_LIMIT),
        "family": family, "seed": seed, "source_artifact_sha256": sha256(source_artifact), "source_manifest_sha256": sha256(SOURCE_ROOT / f"{family}_seed{seed}" / "manifest.json"), "source_analysis_manifest_sha256": sha256(SOURCE_ANALYSIS / "manifest.json"), "adam_steps": adam_steps, "gn_steps": gn_steps, "gauge_count": gauge_count, "entity_count": entity_count, "calibration_seconds": calibration_seconds,
        "maximum_adam_query_response_difference": float(max(row["query_response_max_abs_difference_vs_original"] for row in gauge_rows if row["method"] == "mapped_start_adam")), "maximum_gn_query_response_difference": float(max(row["query_response_max_abs_difference_vs_original"] for row in gauge_rows if row["method"] == "response_metric_gauss_newton")), "maximum_adam_functional_coordinate_difference": float(max(row["functional_coordinate_max_abs_difference_vs_original"] for row in gauge_rows if row["method"] == "mapped_start_adam")), "maximum_gn_functional_coordinate_difference": float(max(row["functional_coordinate_max_abs_difference_vs_original"] for row in gauge_rows if row["method"] == "response_metric_gauss_newton")), "maximum_raw_q_coordinate_change": float(max(row["raw_q_max_abs_change_vs_original"] for row in gauge_rows)), "maximum_unaligned_raw_readout_response_change": float(max(row["unaligned_query_response_max_abs_change_vs_original"] for row in raw_readout_rows)), "maximum_covariant_raw_readout_response_change": float(max(row["covariant_query_response_max_abs_change_vs_original"] for row in raw_readout_rows)), "maximum_gn_jacobian_condition": float(max(row["jacobian_condition"] for row in path_rows if row["method"] == "response_metric_gauss_newton")), "all_gn_jacobians_full_rank": bool(all(row["jacobian_rank"] == Q_DIM for row in path_rows if row["method"] == "response_metric_gauss_newton")), "all_bound_audits_ok": bool(all(row["bound_ok"] for row in bound_rows)), "query_target_input_max_difference": float(perturbation_values["input"]), "query_target_adam_path_max_difference": float(perturbation_values["adam_path"]), "query_target_gn_path_max_difference": float(perturbation_values["gn_path"]), "query_target_gn_step_max_difference": float(perturbation_values["gn_step"]), "query_target_q_max_difference": float(perturbation_values["q"]), "query_target_functional_max_difference": float(perturbation_values["functional"]), "query_target_prediction_max_difference": float(perturbation_values["prediction"]), "query_target_perturbation_max_difference": float(perturbation_max), "query_target_perturbation_value": PERTURBATION, "narrow_support_failure_count": int(sum(row["status"] != "ok" for row in narrow_rows)), "source_result_status": source_result["status"],
    }
    write_json(cell_root / "result.json", summary)
    manifest = {"scope": "gauge_equivariant_calibration_extension_cell", "family": family, "seed": seed, "amendment_sha256": sha256(AMENDMENT), "source_plan_sha256": sha256(SOURCE_PLAN), "source_runner_sha256": sha256(SOURCE_RUNNER_PATH), "source_artifact_sha256": sha256(source_artifact), "source_manifest_sha256": sha256(SOURCE_ROOT / f"{family}_seed{seed}" / "manifest.json"), "source_analysis_manifest_sha256": sha256(SOURCE_ANALYSIS / "manifest.json"), "adam_steps": adam_steps, "gn_steps": gn_steps, "gauge_count": gauge_count, "entity_count": entity_count, "query_target_perturbation": PERTURBATION, "files": {}}
    write_json(cell_root / "manifest.json", manifest)
    manifest["runner_sha256"] = sha256(Path(__file__))
    manifest["files"] = {path.name: sha256(path) for path in sorted(cell_root.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(cell_root / "manifest.json", manifest)
    print(json.dumps(summary, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    parser.add_argument("--gn-steps", type=int, default=GN_STEPS)
    parser.add_argument("--gauge-count", type=int, default=GAUGE_COUNT)
    parser.add_argument("--entity-limit", type=int, default=ENTITY_LIMIT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.adam_steps = min(args.adam_steps, 2)
        args.gn_steps = min(args.gn_steps, 2)
        args.gauge_count = min(args.gauge_count, 1)
        args.entity_limit = min(args.entity_limit, 2)
    run_cell(args.family, args.seed, args.output_root, adam_steps=args.adam_steps, gn_steps=args.gn_steps, gauge_count=args.gauge_count, entity_limit=args.entity_limit, threads=args.threads, smoke=args.smoke)


if __name__ == "__main__":
    main()
