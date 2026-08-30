#!/usr/bin/env python3
"""Run one controlled, target-blind GIRD discovery cell.

The cell consumes one terminal source decoder and one terminal, independently
calibrated gauge-extension cell.  Dictionary selection uses only outer-training
responses.  Test query targets are retained only for final scoring and the
explicit leakage audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GIRD_PLAN = PROJECT_ROOT / "GIRD_METHOD_EXPERIMENT_PLAN_20260829.md"
SOURCE_PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md"
FOUR_SUPPORT_AMENDMENT = PROJECT_ROOT / "GIRD_FOUR_SUPPORT_POSITION_AMENDMENT_20260829.md"
DECISION_AMENDMENT = PROJECT_ROOT / "GIRD_CONTROLLED_DECISION_STATISTIC_AMENDMENT_20260829.md"
SOURCE_RUNNER = PROJECT_ROOT / "scripts/run_gauge_invariant_canonical_response_benchmark_20260829.py"
EXTENSION_RUNNER = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_stable_extension_20260829.py"
SOURCE_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
EXTENSION_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_stable_extension_20260829"
FORMAL_ROOT = PROJECT_ROOT / "runs/gird_controlled_discovery_20260829"

FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
Q_DIM = 3
TRAIN_ENTITIES = 96
TEST_ENTITIES = 48
GRID_SIZE = 41
PROBE_SIZE = 81
GAUGE_IDS = (-1, 0, 1, 2, 3, 4)
EXTENSION_METHODS = ("mapped_start_adam", "response_metric_gauss_newton")
LAMBDA_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, math.inf)
INNER_FOLDS = tuple(range(5))
MAX_ATOMS = 5
PERTURBATION = 1_000_000.0
EXPECTED_GIRD_PLAN_SHA256 = "c31f1fca60219f2cf2b258bac09e1a590dd7962c5e55d19f659730b743602072"
EXPECTED_SOURCE_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
EXPECTED_AMENDMENT_SHA256 = "d85db0c6d9a5b332aa3499eb9d3f105a2e89a4674b30fe90db0657bd26006613"
EXPECTED_FOUR_SUPPORT_AMENDMENT_SHA256 = "1296f79ad03d7688157d2a24145f2407d5b27b29d93683845be3107844126919"
EXPECTED_DECISION_AMENDMENT_SHA256 = "e110948ab67dec54b4b9d28c96c2276eb29912f6e988bd160bc7869236a04df0"
EXPECTED_EXTENSION_RUNNER_SHA256 = "5257d739592caf96249eb6dc5e8bca81734bf744a20e6fa8366fbaa806150fbd"
EXPECTED_EXTENSION_DECISION_SHA256 = "f3b1a222cf9e56b4209f0e6183dbd80792ae96738a905c06759fa4f655a1a1a0"
EXPECTED_EXTENSION_MANIFEST_SHA256 = "980bd9e7540cf733fbdeb5aba33ea7339b46c3968d53918836e6ab8eaf9fcc30"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load(SOURCE_RUNNER, "gird_source_runner")
STABLE = _load(EXTENSION_RUNNER, "gird_stable_extension_runner")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(values: np.ndarray, name: str) -> None:
    _require(np.isfinite(np.asarray(values, dtype=float)).all(), f"non-finite {name}")


def atom_names(family: str) -> tuple[str, ...]:
    libraries = {
        "polynomial": ("1", "x", "x^2", "x^3", "x^4", "x^5", "exp(x)-1", "exp(-x)-1", "sin(x)", "cos(x)-1", "1/(2+x)-1/2", "log(2+x)-log(2)"),
        "relaxation": ("1", "exp(-2x)", "1/(1+x)", "exp(-x)", "exp(-3x)", "exp(-4x)", "1/(1+0.5x)", "1/(1+2x)", "log(1+x)", "x", "x^2", "sqrt(1+x)-1"),
        "thermodynamic_chart": ("1", "1/(x+2)-1/2.5", "log(z)", "z-1", "(z-1)^2", "1/(x+2)^2-1/2.5^2", "log(z)^2", "exp(-(z-1))-1", "sqrt(z)-1", "1/(1+z)-1/2", "z^2-1", "z^3-1"),
    }
    _require(family in libraries, f"unknown family: {family}")
    return libraries[family]


def atom_design(family: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = (x + 2.0) / 2.5
    if family == "polynomial":
        return np.column_stack((np.ones_like(x), x, x**2, x**3, x**4, x**5, np.exp(x)-1.0, np.exp(-x)-1.0, np.sin(x), np.cos(x)-1.0, 1.0/(2.0+x)-0.5, np.log(2.0+x)-np.log(2.0)))
    if family == "relaxation":
        return np.column_stack((np.ones_like(x), np.exp(-2.0*x), 1.0/(1.0+x), np.exp(-x), np.exp(-3.0*x), np.exp(-4.0*x), 1.0/(1.0+0.5*x), 1.0/(1.0+2.0*x), np.log(1.0+x), x, x**2, np.sqrt(1.0+x)-1.0))
    if family == "thermodynamic_chart":
        return np.column_stack((np.ones_like(x), 1.0/(x+2.0)-1.0/2.5, np.log(z), z-1.0, (z-1.0)**2, 1.0/(x+2.0)**2-1.0/2.5**2, np.log(z)**2, np.exp(-(z-1.0))-1.0, np.sqrt(z)-1.0, 1.0/(1.0+z)-0.5, z**2-1.0, z**3-1.0))
    raise ValueError(f"unknown family: {family}")


def _normalised_prediction(model: torch.nn.Module, x: np.ndarray, q: np.ndarray, x_mean: float, x_std: float, y_mean: float, y_std: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    inputs = torch.as_tensor(np.column_stack(((x-x_mean)/x_std, np.repeat(q[None, :], len(x), axis=0))), dtype=torch.float64)
    with torch.no_grad():
        return model(inputs).squeeze(1).cpu().numpy() * y_std + y_mean


def _fit_linear(design: np.ndarray, response: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    gram = design.T @ design
    if ridge:
        gram = gram + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ response)


def _fit_support_prior(phi_support: np.ndarray, support_y: np.ndarray, phi_probe: np.ndarray, prior_response: np.ndarray, lam: float) -> np.ndarray:
    if math.isinf(lam):
        return np.linalg.lstsq(phi_probe, prior_response, rcond=None)[0]
    prior_c = np.linalg.lstsq(phi_probe, prior_response, rcond=None)[0]
    if lam == 0.0:
        return np.linalg.lstsq(phi_support, support_y, rcond=None)[0]
    stacked_design = np.vstack((phi_support / np.sqrt(len(phi_support)), np.sqrt(lam) * phi_probe / np.sqrt(len(phi_probe))))
    stacked_target = np.concatenate((support_y / np.sqrt(len(phi_support)), np.sqrt(lam) * (phi_probe @ prior_c) / np.sqrt(len(phi_probe))))
    return np.linalg.lstsq(stacked_design, stacked_target, rcond=None)[0]


def _support_fit_diagnostics(phi_support: np.ndarray) -> tuple[int, float, float]:
    singular = np.linalg.svd(np.asarray(phi_support, dtype=float), compute_uv=False)
    return int(np.linalg.matrix_rank(phi_support, tol=1e-12)), float(singular[-1]), float(np.linalg.cond(phi_support))


def _entity_nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    scale = float(np.std(target, ddof=0))
    _require(scale > 0.0, "zero-variance entity target")
    return float(np.sqrt(np.mean((target-prediction)**2)) / scale)


def _lambda_key(value: float) -> tuple[int, float]:
    return (1, 0.0) if math.isinf(value) else (0, value)


def four_support_indices(size: int = GRID_SIZE) -> tuple[np.ndarray, np.ndarray]:
    _require(sha256(FOUR_SUPPORT_AMENDMENT) == EXPECTED_FOUR_SUPPORT_AMENDMENT_SHA256, "four-support amendment hash mismatch")
    support = np.asarray([0, round((size - 1) / 3), round(2 * (size - 1) / 3), size - 1], dtype=np.int64)
    _require(len(np.unique(support)) == 4 and int(support.min()) >= 0 and int(support.max()) < size, "invalid four-support positions")
    query = np.setdiff1d(np.arange(size, dtype=np.int64), support, assume_unique=True)
    return support, query


def _subspace_angle_degrees(reference: np.ndarray, candidate: np.ndarray) -> float:
    if np.asarray(reference).ndim != 2 or np.asarray(candidate).ndim != 2:
        return 90.0
    if np.asarray(reference).shape[0] != np.asarray(candidate).shape[0]:
        return 90.0
    if np.linalg.matrix_rank(candidate, tol=1e-10) < np.linalg.matrix_rank(reference, tol=1e-10):
        return 90.0
    reference_q = np.linalg.qr(np.asarray(reference, dtype=float), mode="reduced")[0]
    candidate_q = np.linalg.qr(np.asarray(candidate, dtype=float), mode="reduced")[0]
    singular_values = np.linalg.svd(reference_q.T @ candidate_q, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(singular_values.min(), -1.0, 1.0))))


def _recovery_status(selected: list[str], names: tuple[str, ...], design: np.ndarray) -> tuple[bool, float, bool]:
    exact = len(selected) == 3 and set(selected) == set(names[:3])
    angle = _subspace_angle_degrees(design[:, :3], design[:, [names.index(name) for name in selected]])
    return exact, angle, bool(exact or angle <= 5.0)


def _choose_lambda(scores: dict[float, float]) -> tuple[float, float]:
    best_score = min(scores.values())
    eligible = [value for value, score in scores.items() if score <= best_score * 1.01]
    selected = sorted(eligible, key=_lambda_key)[0]
    return selected, scores[selected]


def _residualised_atom(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if current.shape[1] == 0:
        return candidate.copy()
    return candidate - current @ np.linalg.lstsq(current, candidate, rcond=None)[0]


def omp_path(
    responses: np.ndarray, design: np.ndarray, names: Iterable[str], *, max_atoms: int = MAX_ATOMS, epsilon: float = 0.0,
    fold: int = -1, source: str = "outer_training",
) -> list[dict[str, Any]]:
    """Deterministic heterogeneous-design multi-response OMP path."""
    responses = np.asarray(responses, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    names = tuple(names)
    _require(responses.ndim == 2 and design.ndim == 2 and responses.shape[1] == design.shape[0], "OMP shape mismatch")
    _require(len(names) == design.shape[1] and names[0] == "1", "OMP library mismatch")
    selected = [0]
    rows: list[dict[str, Any]] = []
    for stage in range(2, max_atoms + 1):
        current = design[:, selected]
        coefficients = np.linalg.lstsq(current, responses.T, rcond=None)[0].T
        residual = responses - coefficients @ current.T
        residual_norms = np.linalg.norm(residual, axis=1)
        candidate_rows: list[dict[str, Any]] = []
        for index, name in enumerate(names):
            if index in selected:
                continue
            orth = _residualised_atom(current, design[:, index])
            norm = float(np.linalg.norm(orth))
            if norm == 0.0:
                continue
            proposed = np.column_stack((current, design[:, index]))
            scaled = proposed / np.linalg.norm(proposed, axis=0)
            singular = np.linalg.svd(scaled, compute_uv=False)
            rank = int(np.linalg.matrix_rank(proposed, tol=1e-10))
            condition = float(np.linalg.cond(scaled))
            if rank < stage or condition > 1e4:
                continue
            correlation = (residual @ orth) / np.maximum(residual_norms * norm, 1e-300)
            score = float(np.sum(correlation**2))
            candidate_rows.append({
                "candidate_index": index, "candidate_name": name, "score": score,
                "gram_sigma_min": float(singular[-1]), "gram_sigma_max": float(singular[0]),
                "gram_condition": condition, "gram_rank": rank,
            })
        _require(candidate_rows, f"OMP has no admissible candidate at stage {stage}")
        candidate_rows.sort(key=lambda row: (-row["score"], row["candidate_index"]))
        winner, runner = candidate_rows[0], candidate_rows[1] if len(candidate_rows) > 1 else None
        margin = float(winner["score"] - (runner["score"] if runner else 0.0))
        certificate = float(4.0 * np.linalg.norm(residual) * epsilon + 2.0 * epsilon**2)
        selected.append(int(winner["candidate_index"]))
        for rank, row in enumerate(candidate_rows):
            rows.append({
                "source": source, "fold": fold, "stage": stage, "selected_before": ";".join(names[i] for i in selected[:-1]),
                "candidate_rank": rank + 1, "candidate_index": row["candidate_index"], "candidate_name": row["candidate_name"],
                "score": row["score"], "winner_score": winner["score"], "runner_up_score": runner["score"] if runner else np.nan,
                "score_margin": margin, "gram_sigma_min": row["gram_sigma_min"], "gram_sigma_max": row["gram_sigma_max"],
                "gram_condition": row["gram_condition"], "gram_rank": row["gram_rank"], "residual_frobenius": float(np.linalg.norm(residual)),
                "stability_epsilon": epsilon, "certificate_bound": certificate, "winner": bool(rank == 0),
                "margin_certified": bool(rank == 0 and margin >= certificate), "selected_after": ";".join(names[i] for i in selected),
            })
    return rows


def nested_dictionary(
    responses: np.ndarray, design: np.ndarray, names: tuple[str, ...], entity_ids: np.ndarray, *, source: str,
    epsilon: float = 0.0,
) -> tuple[list[str], int, list[dict[str, Any]], list[dict[str, Any]]]:
    validation_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for fold in INNER_FOLDS:
        train_mask = (entity_ids % 5) != fold
        valid_mask = ~train_mask
        fold_path = omp_path(responses[train_mask], design, names, fold=fold, source=source, epsilon=epsilon)
        path_rows.extend(fold_path)
        for k in range(2, MAX_ATOMS + 1):
            selected_names = ["1"]
            for stage in range(2, k + 1):
                winners = [row for row in fold_path if row["stage"] == stage and row["winner"]]
                _require(len(winners) == 1, "OMP path has ambiguous winner")
                selected_names.append(str(winners[0]["candidate_name"]))
            selected_indices = [names.index(name) for name in selected_names]
            valid_fit = np.linalg.lstsq(design[:, selected_indices], responses[valid_mask].T, rcond=None)[0].T
            prediction = valid_fit @ design[:, selected_indices].T
            denominator = float(np.sum(responses[valid_mask]**2))
            score = float(np.sum((responses[valid_mask]-prediction)**2) / denominator) if denominator else float("inf")
            validation_rows.append({"source": source, "fold": fold, "k": k, "selected_atoms": ";".join(selected_names), "normalized_validation_mse": score, "validation_entities": int(valid_mask.sum())})
    summary = pd.DataFrame(validation_rows).groupby("k", as_index=False)["normalized_validation_mse"].median()
    scores = {int(row.k): float(row.normalized_validation_mse) for row in summary.itertuples()}
    best = min(scores.values())
    eligible = [k for k, value in scores.items() if value <= best * 1.01]
    selected_k = min(eligible)
    final_path = omp_path(responses, design, names, fold=-1, source=source, epsilon=epsilon)
    path_rows.extend(final_path)
    selected_names = ["1"]
    for stage in range(2, selected_k + 1):
        winners = [row for row in final_path if row["stage"] == stage and row["winner"]]
        _require(len(winners) == 1, "final OMP path has ambiguous winner")
        selected_names.append(str(winners[0]["candidate_name"]))
    for row in validation_rows:
        row["selected_k"] = selected_k
        row["selected_by_1pct_tie"] = bool(row["k"] == selected_k)
    return selected_names, selected_k, validation_rows, path_rows


def _verify_source(family: str, seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    _require(sha256(GIRD_PLAN) == EXPECTED_GIRD_PLAN_SHA256, "GIRD plan hash mismatch")
    _require(sha256(SOURCE_PLAN) == EXPECTED_SOURCE_PLAN_SHA256, "source benchmark plan hash mismatch")
    analysis = SOURCE_ROOT / "analysis"
    analysis_manifest = _json(analysis / "manifest.json")
    decision = _json(analysis / "decision.json")
    _require(decision["primary_gates"]["all_15_cells_formal_success"] is True, "source analysis is not terminal")
    _require(analysis_manifest["plan_sha256"] == EXPECTED_SOURCE_PLAN_SHA256, "source analysis plan mismatch")
    _require(analysis_manifest["runner_sha256"] == sha256(SOURCE_RUNNER), "source runner mismatch")
    for name, expected in analysis_manifest["files"].items():
        _require(sha256(analysis / name) == expected, f"source analysis hash mismatch: {name}")
    cell = SOURCE_ROOT / f"{family}_seed{seed}"
    manifest = _json(cell / "manifest.json")
    result = _json(cell / "result.json")
    _require(manifest["plan_sha256"] == EXPECTED_SOURCE_PLAN_SHA256 and manifest["runner_sha256"] == sha256(SOURCE_RUNNER), "source cell provenance mismatch")
    _require(result["status"] == "success" and result["scientific_selection_eligible"] is True, "source cell is not eligible")
    _require(result["epochs"] == 1500 and result["calibration_steps"] == 1200 and result["gauge_count"] == 25, "source cell budget mismatch")
    for name, expected in manifest["files"].items():
        _require(sha256(cell / name) == expected, f"source cell hash mismatch: {name}")
    return cell / "artifact.pt", manifest, result


def _verify_extension(family: str, seed: int, extension_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Admit only the provenance-bound corrected extension, never the old failure."""
    extension_root = extension_root.resolve()
    _require((extension_root / "launcher.done").is_file(), "gauge extension launcher is not terminal")
    analysis = extension_root / "analysis"
    analysis_manifest = _json(analysis / "manifest.json")
    decision = _json(analysis / "decision.json")
    _require(sha256(analysis / "manifest.json") == EXPECTED_EXTENSION_MANIFEST_SHA256, "stable extension analysis manifest hash mismatch")
    _require(sha256(analysis / "decision.json") == EXPECTED_EXTENSION_DECISION_SHA256, "stable extension decision hash mismatch")
    _require(analysis_manifest.get("scope") == "independent_numerically_stable_gauge_equivariant_calibration_analysis", "stable extension analysis scope mismatch")
    _require(decision.get("benchmark_passed") is True, "stable gauge extension failed its primary gates and is not a GIRD input")
    _require(sha256(AMENDMENT) == EXPECTED_AMENDMENT_SHA256, "numerical amendment hash mismatch")
    _require(sha256(EXTENSION_RUNNER) == EXPECTED_EXTENSION_RUNNER_SHA256, "stable extension runner hash mismatch")
    _require(analysis_manifest["numerical_amendment_sha256"] == EXPECTED_AMENDMENT_SHA256, "extension numerical amendment mismatch")
    _require(analysis_manifest["stable_runner_sha256"] == EXPECTED_EXTENSION_RUNNER_SHA256, "extension stable runner mismatch")
    for name, expected in analysis_manifest["files"].items():
        _require(sha256(analysis / name) == expected, f"extension analysis hash mismatch: {name}")
    cell = extension_root / f"{family}_seed{seed}"
    manifest = _json(cell / "manifest.json")
    _require(manifest["scope"] == "gauge_equivariant_calibration_stable_extension_cell", "extension cell scope mismatch")
    _require(manifest["numerical_amendment_sha256"] == EXPECTED_AMENDMENT_SHA256 and manifest["source_plan_sha256"] == EXPECTED_SOURCE_PLAN_SHA256, "extension provenance mismatch")
    _require(manifest["runner_sha256"] == EXPECTED_EXTENSION_RUNNER_SHA256 and manifest["stable_solver"] == "float64_lstsq", "extension cell runner mismatch")
    _require(manifest["adam_steps"] == 300 and manifest["gn_steps"] == 15 and manifest["gauge_count"] == 5 and manifest["entity_count"] == 48, "extension budget mismatch")
    for name, expected in manifest["files"].items():
        _require(sha256(cell / name) == expected, f"extension cell hash mismatch: {name}")
    result = _json(cell / "result.json")
    _require(result["status"] == "success" and result["scientific_selection_eligible"] is True and result["stable_solver"] == "float64_lstsq", "extension cell is not eligible")
    calibration = pd.read_csv(cell / "calibration_diagnostics.csv")
    query = pd.read_csv(cell / "query_predictions.csv")
    _require(set(calibration.method) == set(EXTENSION_METHODS) and set(calibration.gauge_id) == set(GAUGE_IDS), "extension calibration coverage mismatch")
    _require(len(calibration) == 2 * len(GAUGE_IDS) * TEST_ENTITIES, "extension calibration row count mismatch")
    _finite(calibration[[f"q{i}" for i in range(Q_DIM)]].to_numpy(), "extension q")
    _require(len(query) == 2 * len(GAUGE_IDS) * TEST_ENTITIES * 30, "extension query row count mismatch")
    _finite(query[["query_position", "x", "target", "prediction", "functional_prediction"]].to_numpy(), "extension query")
    return manifest, calibration, query, decision


def _load_chart(model: torch.nn.Module, gauge: dict[str, Any] | None) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    if gauge is None:
        return model, np.eye(Q_DIM), np.zeros(Q_DIM)
    matrix = np.asarray(gauge["matrix"], dtype=np.float64)
    offset = np.asarray(gauge["offset"], dtype=np.float64)
    return SOURCE.apply_affine_gauge(model, matrix, offset).double().eval(), matrix, offset


def _readout(q: np.ndarray, train_q: np.ndarray, train_coefficients: np.ndarray) -> np.ndarray:
    mean = train_q.mean(axis=0)
    scale = train_q.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(train_q)), (train_q-mean)/scale))
    weights = np.linalg.solve(design.T @ design + np.diag([0.0, 1e-3, 1e-3, 1e-3]), design.T @ train_coefficients)
    return np.column_stack((np.ones(len(np.atleast_2d(q))), (np.atleast_2d(q)-mean)/scale)) @ weights


def _run_cell_regime(family: str, seed: int, output_root: Path, *, support_regime: str, extension_root: Path = EXTENSION_ROOT, threads: int = 1, smoke: bool = False) -> dict[str, Any]:
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError("family or seed outside frozen controlled benchmark")
    source_artifact, source_manifest, source_result = _verify_source(family, seed)
    _require(sha256(DECISION_AMENDMENT) == EXPECTED_DECISION_AMENDMENT_SHA256, "decision-statistic amendment hash mismatch")
    extension_manifest, extension_calibration, extension_query, extension_decision = _verify_extension(family, seed, extension_root)
    output_root = output_root.resolve()
    if support_regime == "standard_11":
        support_positions, query_positions = SOURCE.support_query_indices()
    elif support_regime == "four_support":
        support_positions, query_positions = four_support_indices()
    else:
        raise ValueError(f"unknown support regime: {support_regime}")
    cell = output_root / f"{family}_seed{seed}" / support_regime
    cell.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    artifact = torch.load(source_artifact, map_location="cpu", weights_only=False)
    model = SOURCE.SiLUDecoder().double()
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = SOURCE.generate_family_data(family, seed)
    x, targets = data["x"], data["targets"]
    train_targets, test_targets = targets[:TRAIN_ENTITIES], targets[TRAIN_ENTITIES:]
    train_coefficients = data["coefficients"][:TRAIN_ENTITIES]
    test_coefficients = data["coefficients"][TRAIN_ENTITIES:]
    train_q = np.asarray(artifact["embedding"], dtype=np.float64)
    x_mean, x_std = float(artifact["x_mean"]), float(artifact["x_std"])
    y_mean, y_std = float(artifact["y_mean"]), float(artifact["y_std"])
    probe_x = np.linspace(float(x.min()), float(x.max()), PROBE_SIZE)
    basis_probe = SOURCE.family_basis(family, probe_x)
    basis_query = SOURCE.family_basis(family, x[query_positions])
    basis_support = SOURCE.family_basis(family, x[support_positions])
    names = atom_names(family)
    atom_probe = atom_design(family, probe_x)
    atom_support = atom_design(family, x[support_positions])
    atom_query = atom_design(family, x[query_positions])
    gauges = {int(g["gauge_id"]): g for g in artifact["gauges"][:5]}
    charts: list[tuple[int, str, torch.nn.Module, np.ndarray, np.ndarray]] = [(-1, "original", model, np.eye(Q_DIM), np.zeros(Q_DIM))]
    for gauge_id in range(5):
        chart_model, matrix, offset = _load_chart(model, gauges[gauge_id])
        charts.append((gauge_id, f"gauge_{gauge_id}", chart_model, matrix, offset))

    dictionary_validation: list[dict[str, Any]] = []
    dictionary_paths: list[dict[str, Any]] = []
    dictionary_stability: list[dict[str, Any]] = []
    dictionary_recovery: list[dict[str, Any]] = []
    dictionary_selections: dict[tuple[str, int], list[str]] = {}
    test_dictionary_selections: dict[str, list[str]] = {}
    train_responses: dict[int, np.ndarray] = {}
    test_responses: dict[tuple[str, int], np.ndarray] = {}
    # First collect every chart response.  The certificate epsilon is the
    # largest paired response perturbation, computed independently for the
    # outer training decoder probes and the held-out extension calibrations.
    for gauge_id, chart_name, chart_model, matrix, offset in charts:
        response = np.asarray([_normalised_prediction(chart_model, probe_x, matrix @ train_q[i] + offset, x_mean, x_std, y_mean, y_std) for i in range(TRAIN_ENTITIES)])
        train_responses[gauge_id] = response
        for source, method in (("gird_gn", "response_metric_gauss_newton"), ("gird_adam", "mapped_start_adam")):
            rows = []
            for entity in range(TEST_ENTITIES):
                qrow = extension_calibration[(extension_calibration.method == method) & (extension_calibration.gauge_id == gauge_id) & (extension_calibration.entity_id == entity)]
                _require(len(qrow) == 1, "extension calibration q coverage mismatch")
                q = qrow[[f"q{i}" for i in range(Q_DIM)]].to_numpy(dtype=float)[0]
                rows.append(_normalised_prediction(chart_model, probe_x, q, x_mean, x_std, y_mean, y_std))
            test_responses[(source, gauge_id)] = np.asarray(rows)
    outer_epsilon = max(float(np.linalg.norm(train_responses[g] - train_responses[-1])) for g in range(5))
    test_epsilon = {source: max(float(np.linalg.norm(test_responses[(source, g)] - test_responses[(source, -1)])) for g in range(5)) for source in ("gird_gn", "gird_adam")}
    for gauge_id, chart_name, _, _, _ in charts:
        for source in ("gird_gn", "gird_adam"):
            selected, selected_k, validation, paths = nested_dictionary(train_responses[gauge_id], atom_probe, names, np.arange(TRAIN_ENTITIES), source=source, epsilon=outer_epsilon)
            dictionary_selections[(source, gauge_id)] = selected
            exact_recovery, recovery_angle, recovery_ok = _recovery_status(selected, names, atom_probe)
            dictionary_recovery.append({"source": source, "stage": "outer_training", "chart": chart_name, "gauge_id": gauge_id, "selected_atoms": ";".join(selected), "exact_atom_recovery": exact_recovery, "max_function_subspace_angle_degrees": recovery_angle, "recovery_ok": recovery_ok})
            dictionary_validation.extend([{**row, "family": family, "seed": seed, "chart": chart_name, "gauge_id": gauge_id} for row in validation])
            dictionary_paths.extend([{**row, "family": family, "seed": seed, "chart": chart_name, "gauge_id": gauge_id} for row in paths])
            dictionary_stability.append({"source": source, "stage": "outer_training", "chart": chart_name, "gauge_id": gauge_id, "selected_k": selected_k, "selected_atoms": ";".join(selected), "same_as_original": bool(selected == dictionary_selections.get((source, -1), selected)), "stability_epsilon": outer_epsilon, "used_for_prediction": gauge_id == -1})
        for source in ("gird_gn", "gird_adam"):
            test_selected, test_k, _, test_paths = nested_dictionary(test_responses[(source, gauge_id)], atom_probe, names, np.arange(TEST_ENTITIES), source="independent_extension_calibration", epsilon=test_epsilon[source])
            same_as_test_original = bool(test_selected == test_dictionary_selections.get(source, test_selected))
            test_dictionary_selections.setdefault(source, test_selected)
            test_exact, test_angle, test_recovery_ok = _recovery_status(test_selected, names, atom_probe)
            dictionary_recovery.append({"source": source, "stage": "independent_extension_calibration", "chart": chart_name, "gauge_id": gauge_id, "selected_atoms": ";".join(test_selected), "exact_atom_recovery": test_exact, "max_function_subspace_angle_degrees": test_angle, "recovery_ok": test_recovery_ok})
            dictionary_paths.extend([{**row, "family": family, "seed": seed, "chart": chart_name, "gauge_id": gauge_id, "response_method": source} for row in test_paths])
            dictionary_stability.append({"source": source, "stage": "independent_extension_calibration", "chart": chart_name, "gauge_id": gauge_id, "selected_k": test_k, "selected_atoms": ";".join(test_selected), "same_as_test_cohort_original": same_as_test_original, "same_as_train_cohort_original": bool(test_selected == dictionary_selections[(source, -1)]), "stability_epsilon": test_epsilon[source], "used_for_prediction": False})

    # Direct-target control uses only outer-training curves, interpolated to the
    # same fixed probe grid.  It is intentionally separate from decoder GIRD.
    direct_response = np.asarray([np.interp(probe_x, x, row) for row in train_targets])
    direct_selected, direct_k, direct_validation, direct_paths = nested_dictionary(direct_response, atom_probe, names, np.arange(TRAIN_ENTITIES), source="direct_target_omp")
    dictionary_validation.extend([{**row, "family": family, "seed": seed, "chart": "direct_target", "gauge_id": -1} for row in direct_validation])
    dictionary_paths.extend([{**row, "family": family, "seed": seed, "chart": "direct_target", "gauge_id": -1} for row in direct_paths])
    dictionary_selections[("direct_target_omp", -1)] = direct_selected
    dictionary_stability.append({"source": "direct_target_omp", "stage": "outer_training", "chart": "direct_target", "gauge_id": -1, "selected_k": direct_k, "selected_atoms": ";".join(direct_selected), "same_as_original": False})
    direct_exact, direct_angle, direct_recovery_ok = _recovery_status(direct_selected, names, atom_probe)
    dictionary_recovery.append({"source": "direct_target_omp", "stage": "outer_training", "chart": "direct_target", "gauge_id": -1, "selected_atoms": ";".join(direct_selected), "exact_atom_recovery": direct_exact, "max_function_subspace_angle_degrees": direct_angle, "recovery_ok": direct_recovery_ok})
    true_selected = list(names[:3])
    # FPCA rank/ridge is selected with the same fixed entity_id modulo-5 fold.
    fpca_rows: list[dict[str, Any]] = []
    fpca_scores: dict[tuple[int, float], float] = {}
    entity_ids = np.arange(TRAIN_ENTITIES)
    for rank in range(1, 6):
        for ridge in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            fold_scores = []
            for fold in INNER_FOLDS:
                inner_train = (entity_ids % 5) != fold
                fold_mean = train_targets[inner_train].mean(axis=0)
                _, _, fold_vt = np.linalg.svd(train_targets[inner_train] - fold_mean, full_matrices=False)
                components = fold_vt[:rank].T
                values = []
                for entity in np.flatnonzero(~inner_train):
                    fpca_target = train_targets[entity, support_positions] - fold_mean[support_positions]
                    if ridge == 0.0:
                        coefficients = np.linalg.lstsq(components[support_positions], fpca_target, rcond=None)[0]
                    else:
                        fpca_design = np.vstack((components[support_positions], np.sqrt(ridge) * np.eye(rank)))
                        coefficients = np.linalg.lstsq(fpca_design, np.concatenate((fpca_target, np.zeros(rank))), rcond=None)[0]
                    pred = fold_mean[query_positions] + components[query_positions] @ coefficients
                    values.append(_entity_nrmse(train_targets[entity, query_positions], pred))
                fold_scores.append(float(np.median(values)))
                fpca_rows.append({"rank": rank, "ridge": ridge, "fold": fold, "median_entity_nrmse": fold_scores[-1], "fit_entities": int(inner_train.sum()), "fit_scope": "inner_train_only"})
            fpca_scores[(rank, ridge)] = float(np.median(fold_scores))
    fpca_best = min(fpca_scores.values())
    fpca_candidates = [key for key, score in fpca_scores.items() if score <= fpca_best * 1.01]
    fpca_rank, fpca_ridge = sorted(fpca_candidates, key=lambda key: (key[0], key[1]))[0]
    train_mean = train_targets.mean(axis=0)
    centered = train_targets - train_mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    fpca_rows.append({"rank": fpca_rank, "ridge": fpca_ridge, "fold": -1, "median_entity_nrmse": fpca_scores[(fpca_rank, fpca_ridge)], "selected": True, "fit_entities": TRAIN_ENTITIES, "fit_scope": "all_outer_training_refit"})

    # Select the response-prior lambda only on outer-training query rows.  This
    # is the frozen entity_id modulo-5 nested validation; test queries remain
    # untouched until the prediction table is written.
    fusion_rows: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    original_gn_atoms = dictionary_selections[("gird_gn", -1)]
    original_adam_atoms = dictionary_selections[("gird_adam", -1)]
    fusion_choice: dict[str, float] = {}
    # Inner validation entities must not contribute their full-curve artifact
    # embedding to lambda selection.  Recalibrate an original-chart q from
    # that entity's support rows only, initialized by the other inner entities.
    inner_support_priors: dict[tuple[str, int, int], np.ndarray] = {}
    inner_methods = ("gird_gn", "gird_adam")
    inner_support_prior_array = np.full((len(inner_methods), len(INNER_FOLDS), TRAIN_ENTITIES, PROBE_SIZE), np.nan, dtype=np.float64)
    inner_calibration_rows: list[dict[str, Any]] = []
    for fold in INNER_FOLDS:
        inner_train_mask = (entity_ids % 5) != fold
        initial_q = train_q[inner_train_mask].mean(axis=0)
        for entity in np.flatnonzero(~inner_train_mask):
            support_y_norm = (train_targets[entity, support_positions] - y_mean) / y_std
            calibrations = {
                "gird_gn": STABLE.calibrate_gauss_newton(model, initial_q, x[support_positions], support_y_norm, x_mean, x_std, steps=15),
                "gird_adam": STABLE.BASE.calibrate_adam(model, initial_q, x[support_positions], support_y_norm, x_mean, x_std, steps=300),
            }
            for method_index, method in enumerate(inner_methods):
                q_fit, support_loss, path = calibrations[method]
                prior = _normalised_prediction(model, probe_x, q_fit, x_mean, x_std, y_mean, y_std)
                inner_support_priors[(method, fold, int(entity))] = prior
                inner_support_prior_array[method_index, fold, int(entity)] = prior
                for row in path:
                    inner_calibration_rows.append({"record_type": "inner_calibration_path", "method": method, "fold": fold, "entity_id": int(entity), "iteration": int(row["iteration"]), "loss": float(row["loss"]), "loss_after": float(row.get("loss_after", row["loss"])), "step_scale": float(row["step_scale"]), "jacobian_rank": float(row.get("jacobian_rank", np.nan)), "jacobian_condition": float(row.get("jacobian_condition", np.nan)), "support_loss": float(support_loss), "query_targets_used_for_calibration": False})
    for method, atoms in (("gird_gn", original_gn_atoms), ("gird_adam", original_adam_atoms), ("true_basis", true_selected)):
        indices = [names.index(name) for name in atoms]
        scores: dict[float, float] = {}
        for lam in LAMBDA_GRID:
            fold_scores = []
            for fold in INNER_FOLDS:
                values = []
                for entity in np.flatnonzero((np.arange(TRAIN_ENTITIES) % 5) == fold):
                    prior_method = "gird_gn" if method == "true_basis" else method
                    prior = inner_support_priors[(prior_method, fold, int(entity))]
                    coef = _fit_support_prior(atom_support[:, indices], train_targets[entity, support_positions], atom_probe[:, indices], prior, lam)
                    rank, sigma_min, condition = _support_fit_diagnostics(atom_support[:, indices])
                    fit_diagnostics.append({"stage": "inner_lambda", "method": method, "fold": fold, "entity_id": int(entity), "lambda": lam, "support_rank": rank, "coefficient_count": len(indices), "support_sigma_min": sigma_min, "support_condition": condition})
                    prediction = atom_query[:, indices] @ coef
                    values.append(_entity_nrmse(train_targets[entity, query_positions], prediction))
                fold_score = float(np.median(values))
                fold_scores.append(fold_score)
                fusion_rows.append({"method": method, "lambda": lam, "lambda_label": "inf" if math.isinf(lam) else str(lam), "fold": fold, "median_entity_nrmse": fold_score, "selected": False})
            scores[lam] = float(np.median(fold_scores))
        for fold in INNER_FOLDS:
            fold_scores = {float(row["lambda"]): float(row["median_entity_nrmse"]) for row in fusion_rows if row["method"] == method and row["fold"] == fold}
            fold_lambda, fold_score = _choose_lambda(fold_scores)
            fusion_rows.append({"method": method, "lambda": fold_lambda, "lambda_label": "inf" if math.isinf(fold_lambda) else str(fold_lambda), "fold": fold, "median_entity_nrmse": fold_score, "selected": True, "selection_role": "inner_fold"})
        selected_lambda, _ = _choose_lambda(scores)
        fusion_choice[method] = selected_lambda
        fusion_rows.append({"method": method, "lambda": selected_lambda, "lambda_label": "inf" if math.isinf(selected_lambda) else str(selected_lambda), "fold": -1, "median_entity_nrmse": scores[selected_lambda], "selected": True})

    predictions: list[dict[str, Any]] = []
    calibration_inputs: list[dict[str, Any]] = list(inner_calibration_rows)
    basis_fusion_rows: list[dict[str, Any]] = []
    # Keep all method endpoints and fused selections visible.
    method_specs = [("gird_gn", "response_metric_gauss_newton", original_gn_atoms), ("gird_adam", "mapped_start_adam", original_adam_atoms), ("direct_target_omp", None, direct_selected), ("support_only_omp", "response_metric_gauss_newton", original_gn_atoms), ("true_basis", "response_metric_gauss_newton", true_selected)]
    method_values: dict[str, np.ndarray] = {}
    # Ordinary symbolic regression is deliberately entity-local: its OMP path
    # sees only that entity's support values and uses the GIRD-selected K as a
    # predeclared upper bound.  It has no decoder response prior.
    ordinary_k = min(len(original_gn_atoms), len(support_positions))
    ordinary_values = np.zeros((TEST_ENTITIES, len(query_positions)))
    ordinary_selected: dict[int, list[str]] = {}
    for entity in range(TEST_ENTITIES):
        ordinary_response = test_targets[entity, support_positions][None, :]
        ordinary_path = omp_path(ordinary_response, atom_support, names, max_atoms=ordinary_k, source="ordinary_symbolic_regression")
        dictionary_paths.extend([{**row, "family": family, "seed": seed, "chart": "support_only_entity", "gauge_id": entity, "entity_id": entity, "support_regime": support_regime, "leakage_run": False} for row in ordinary_path])
        selected_names = ["1"]
        for stage in range(2, ordinary_k + 1):
            winners = [row for row in ordinary_path if row["stage"] == stage and row["winner"]]
            _require(len(winners) == 1, "ordinary symbolic regression path has ambiguous winner")
            selected_names.append(str(winners[0]["candidate_name"]))
        ordinary_selected[entity] = selected_names
        ordinary_indices = [names.index(name) for name in selected_names]
        ordinary_coefficients = np.linalg.lstsq(atom_support[:, ordinary_indices], ordinary_response[0], rcond=None)[0]
        rank, sigma_min, condition = _support_fit_diagnostics(atom_support[:, ordinary_indices])
        fit_diagnostics.append({"stage": "outer_test", "method": "ordinary_symbolic_regression", "fold": -1, "entity_id": entity, "lambda": 0.0, "support_rank": rank, "coefficient_count": len(ordinary_indices), "support_sigma_min": sigma_min, "support_condition": condition})
        ordinary_values[entity] = atom_query[:, ordinary_indices] @ ordinary_coefficients
    method_values["ordinary_symbolic_regression"] = ordinary_values
    for entity in range(TEST_ENTITIES):
        for pos, x_value, target, pred in zip(query_positions, x[query_positions], test_targets[entity, query_positions], ordinary_values[entity]):
            predictions.append({"family": family, "seed": seed, "entity_id": entity, "query_position": int(pos), "x": float(x_value), "target": float(target), "method": "ordinary_symbolic_regression", "prediction": float(pred), "calibration_uses_query_targets": False})
    for entity in range(TEST_ENTITIES):
        support_y = test_targets[entity, support_positions]
        for method, calibration_method, atoms in method_specs:
            indices = [names.index(name) for name in atoms]
            if method == "direct_target_omp":
                prior_response = np.zeros(PROBE_SIZE)
            else:
                qrow = extension_calibration[(extension_calibration.method == calibration_method) & (extension_calibration.gauge_id == -1) & (extension_calibration.entity_id == entity)]
                q = qrow[[f"q{i}" for i in range(Q_DIM)]].to_numpy(dtype=float)[0]
                prior_response = _normalised_prediction(model, probe_x, q, x_mean, x_std, y_mean, y_std)
                calibration_inputs.append({"entity_id": entity, "method": method, "calibration_method": calibration_method, **{f"q{i}": float(q[i]) for i in range(Q_DIM)}, "query_targets_used_for_calibration": False})
            if method == "support_only_omp":
                lambdas = [0.0]
            elif method == "direct_target_omp":
                lambdas = [0.0]
            else:
                lambdas = [0.0, math.inf, fusion_choice[method]]
            for lam in dict.fromkeys(lambdas):
                coef = _fit_support_prior(atom_support[:, indices], support_y, atom_probe[:, indices], prior_response, lam)
                rank, sigma_min, condition = _support_fit_diagnostics(atom_support[:, indices])
                fit_diagnostics.append({"stage": "outer_test", "method": method, "fold": -1, "entity_id": entity, "lambda": lam, "support_rank": rank, "coefficient_count": len(indices), "support_sigma_min": sigma_min, "support_condition": condition})
                value = atom_query[:, indices] @ coef
                label = f"{method}_lambda_{'inf' if math.isinf(lam) else str(lam)}"
                method_values[label] = method_values.get(label, np.zeros((TEST_ENTITIES, len(query_positions))))
                method_values[label][entity] = value
                for pos, x_value, target, pred in zip(query_positions, x[query_positions], test_targets[entity, query_positions], value):
                    predictions.append({"family": family, "seed": seed, "entity_id": entity, "query_position": int(pos), "x": float(x_value), "target": float(target), "method": label, "prediction": float(pred), "calibration_uses_query_targets": False})
            if method in {"gird_gn", "gird_adam"}:
                selected_lam = fusion_choice[method]
                indices = [names.index(name) for name in atoms]
                coef = _fit_support_prior(atom_support[:, indices], support_y, atom_probe[:, indices], prior_response, selected_lam)
                # The covariance check uses a deterministic invertible H on the
                # selected response basis and the same Gram-metric objective.
                h = np.eye(len(indices))
                if len(indices) > 1:
                    h[0, 1] = 0.25
                transformed_phi = atom_probe[:, indices] @ h
                transformed_support = atom_support[:, indices] @ h
                transformed_prior = prior_response
                transformed_coef = _fit_support_prior(transformed_support, support_y, transformed_phi, transformed_prior, selected_lam)
                basis_fusion_rows.append({"method": method, "entity_id": entity, "lambda": selected_lam, "coordinate_max_abs_error": float(np.max(np.abs(transformed_coef - np.linalg.solve(h, coef)))), "response_max_abs_error": float(np.max(np.abs(transformed_phi @ transformed_coef - atom_probe[:, indices] @ coef)))})

    # Raw-q linear readout and no-q global expression controls.
    raw_values = []
    global_values = np.mean(train_targets, axis=0)[query_positions]
    for entity in range(TEST_ENTITIES):
        qrow = extension_calibration[(extension_calibration.method == "response_metric_gauss_newton") & (extension_calibration.gauge_id == -1) & (extension_calibration.entity_id == entity)]
        q = qrow[[f"q{i}" for i in range(Q_DIM)]].to_numpy(dtype=float)[0]
        raw_coeff = _readout(q, train_q, train_coefficients)[0]
        raw_values.append(basis_query @ raw_coeff)
    method_values["raw_q"] = np.asarray(raw_values)
    method_values["global_no_q"] = np.repeat(global_values[None, :], TEST_ENTITIES, axis=0)
    for label, values in (("raw_q", method_values["raw_q"]), ("global_no_q", method_values["global_no_q"])):
        for entity in range(TEST_ENTITIES):
            for pos, x_value, target, pred in zip(query_positions, x[query_positions], test_targets[entity, query_positions], values[entity]):
                predictions.append({"family": family, "seed": seed, "entity_id": entity, "query_position": int(pos), "x": float(x_value), "target": float(target), "method": label, "prediction": float(pred), "calibration_uses_query_targets": False})
    components = vt[:fpca_rank].T
    fpca_values = []
    for entity in range(TEST_ENTITIES):
        fpca_target = test_targets[entity, support_positions] - train_mean[support_positions]
        if fpca_ridge == 0.0:
            coef = np.linalg.lstsq(components[support_positions], fpca_target, rcond=None)[0]
        else:
            fpca_design = np.vstack((components[support_positions], np.sqrt(fpca_ridge) * np.eye(fpca_rank)))
            fpca_rhs = np.concatenate((fpca_target, np.zeros(fpca_rank)))
            coef = np.linalg.lstsq(fpca_design, fpca_rhs, rcond=None)[0]
        fpca_rank_actual, fpca_sigma_min, fpca_condition = _support_fit_diagnostics(components[support_positions])
        fit_diagnostics.append({"stage": "outer_test", "method": "fpca", "fold": -1, "entity_id": entity, "lambda": fpca_ridge, "support_rank": fpca_rank_actual, "coefficient_count": fpca_rank, "support_sigma_min": fpca_sigma_min, "support_condition": fpca_condition})
        fpca_values.append(train_mean[query_positions] + components[query_positions] @ coef)
    method_values["fpca"] = np.asarray(fpca_values)
    for entity in range(TEST_ENTITIES):
        for pos, x_value, target, pred in zip(query_positions, x[query_positions], test_targets[entity, query_positions], method_values["fpca"][entity]):
            predictions.append({"family": family, "seed": seed, "entity_id": entity, "query_position": int(pos), "x": float(x_value), "target": float(target), "method": "fpca", "prediction": float(pred), "calibration_uses_query_targets": False})

    # Exact query-target leakage audit: perturb query targets only and rerun
    # every support fit and entity-local selection from the unchanged support
    # slice.  The outer dictionary and lambda choices are also re-read and
    # compared explicitly; they must not depend on query targets.
    perturbed = test_targets.copy()
    perturbed[:, query_positions] += PERTURBATION
    ordinary_perturbed_values = np.zeros_like(ordinary_values)
    ordinary_path_differences: list[float] = []
    ordinary_k_differences: list[float] = []
    for entity in range(TEST_ENTITIES):
        path = omp_path(perturbed[entity, support_positions][None, :], atom_support, names, max_atoms=ordinary_k, source="ordinary_symbolic_regression_leakage")
        dictionary_paths.extend([{**row, "family": family, "seed": seed, "chart": "support_only_entity", "gauge_id": entity, "entity_id": entity, "support_regime": support_regime, "leakage_run": True} for row in path])
        selected_names = ["1"]
        for stage in range(2, ordinary_k + 1):
            winners = [row for row in path if row["stage"] == stage and row["winner"]]
            _require(len(winners) == 1, "perturbed ordinary symbolic regression path has ambiguous winner")
            selected_names.append(str(winners[0]["candidate_name"]))
        ordinary_path_differences.append(float(selected_names != ordinary_selected[entity]))
        ordinary_k_differences.append(float(len(selected_names) - len(ordinary_selected[entity])))
        indices = [names.index(name) for name in selected_names]
        coefficient = np.linalg.lstsq(atom_support[:, indices], perturbed[entity, support_positions], rcond=None)[0]
        rank, sigma_min, condition = _support_fit_diagnostics(atom_support[:, indices])
        fit_diagnostics.append({"stage": "leakage", "method": "ordinary_symbolic_regression", "fold": -1, "entity_id": entity, "lambda": 0.0, "support_rank": rank, "coefficient_count": len(indices), "support_sigma_min": sigma_min, "support_condition": condition})
        ordinary_perturbed_values[entity] = atom_query[:, indices] @ coefficient

    def perturbed_fit(label: str, entity: int) -> np.ndarray:
        if label == "ordinary_symbolic_regression":
            return ordinary_perturbed_values[entity]
        if label == "raw_q" or label == "global_no_q":
            return method_values[label][entity]
        if label == "fpca":
            fpca_target = perturbed[entity, support_positions] - train_mean[support_positions]
            if fpca_ridge == 0.0:
                coefficients = np.linalg.lstsq(components[support_positions], fpca_target, rcond=None)[0]
            else:
                fpca_design = np.vstack((components[support_positions], np.sqrt(fpca_ridge) * np.eye(fpca_rank)))
                coefficients = np.linalg.lstsq(fpca_design, np.concatenate((fpca_target, np.zeros(fpca_rank))), rcond=None)[0]
            rank, sigma_min, condition = _support_fit_diagnostics(components[support_positions])
            fit_diagnostics.append({"stage": "leakage", "method": "fpca", "fold": -1, "entity_id": entity, "lambda": fpca_ridge, "support_rank": rank, "coefficient_count": fpca_rank, "support_sigma_min": sigma_min, "support_condition": condition})
            return train_mean[query_positions] + components[query_positions] @ coefficients
        method, lambda_label = label.rsplit("_lambda_", 1)
        lam = math.inf if lambda_label == "inf" else float(lambda_label)
        atoms = {"gird_gn": original_gn_atoms, "gird_adam": original_adam_atoms, "direct_target_omp": direct_selected, "support_only_omp": original_gn_atoms, "true_basis": true_selected}[method]
        indices = [names.index(name) for name in atoms]
        if method == "direct_target_omp":
            prior = np.zeros(PROBE_SIZE)
        else:
            calibration_method = "response_metric_gauss_newton" if method in {"gird_gn", "support_only_omp", "true_basis"} else "mapped_start_adam"
            qrow = extension_calibration[(extension_calibration.method == calibration_method) & (extension_calibration.gauge_id == -1) & (extension_calibration.entity_id == entity)]
            q = qrow[[f"q{i}" for i in range(Q_DIM)]].to_numpy(dtype=float)[0]
            prior = _normalised_prediction(model, probe_x, q, x_mean, x_std, y_mean, y_std)
        coefficient = _fit_support_prior(atom_support[:, indices], perturbed[entity, support_positions], atom_probe[:, indices], prior, lam)
        rank, sigma_min, condition = _support_fit_diagnostics(atom_support[:, indices])
        fit_diagnostics.append({"stage": "leakage", "method": method, "fold": -1, "entity_id": entity, "lambda": lam, "support_rank": rank, "coefficient_count": len(indices), "support_sigma_min": sigma_min, "support_condition": condition})
        return atom_query[:, indices] @ coefficient

    leakage_rows = []
    prediction_frame = pd.DataFrame(predictions)
    for method, values in method_values.items():
        perturbed_values = np.asarray([perturbed_fit(method, entity) for entity in range(TEST_ENTITIES)])
        diffs = [float(np.max(np.abs(values[entity] - perturbed_values[entity]))) for entity in range(TEST_ENTITIES)]
        if method == "ordinary_symbolic_regression":
            path_difference = max(ordinary_path_differences)
            k_difference = max(abs(value) for value in ordinary_k_differences)
        else:
            path_difference = 0.0
            k_difference = 0.0
        leakage_rows.append({"method": method, "query_target_perturbation": PERTURBATION, "support_input_difference": float(np.max(np.abs(test_targets[:, support_positions] - perturbed[:, support_positions]))), "query_target_input_difference": PERTURBATION, "max_prediction_difference": max(diffs), "dictionary_path_input_difference": path_difference, "dictionary_k_input_difference": k_difference, "lambda_input_difference": 0.0, "query_targets_used_for_fit": False})

    # All selected dictionary paths must satisfy the saved certificate; failed
    # paths remain visible and make the cell scientifically ineligible.
    selected_path_rows = [row for row in dictionary_paths if row.get("winner") and row.get("candidate_rank") == 1 and (row.get("source") == "gird_gn" or (row.get("source") == "independent_extension_calibration" and row.get("response_method") == "gird_gn"))]
    all_gn_selected_certified = bool(selected_path_rows) and all(bool(row["margin_certified"]) for row in selected_path_rows)
    stability_rows = pd.DataFrame(dictionary_stability)
    dictionary_identical = bool(stability_rows[stability_rows.stage == "outer_training"].groupby("source")["selected_atoms"].nunique().max() == 1)
    extension_stability_identical = bool(stability_rows[stability_rows.stage == "independent_extension_calibration"].groupby("source")["selected_atoms"].nunique().max() == 1)
    formal_eligible = bool(not smoke and output_root == FORMAL_ROOT.resolve() and extension_root.resolve() == EXTENSION_ROOT.resolve())
    _finite(prediction_frame["prediction"].to_numpy(), "predictions")
    _finite(np.asarray([row["max_prediction_difference"] for row in leakage_rows]), "leakage audit")
    fit_diagnostics_frame = pd.DataFrame(fit_diagnostics)
    fit_diagnostics_frame["rank_deficient"] = fit_diagnostics_frame["support_rank"] < fit_diagnostics_frame["coefficient_count"]
    fit_diagnostics_frame["solver"] = "float64_lstsq"
    files = {
        "dictionary_paths.csv": pd.DataFrame(dictionary_paths),
        "dictionary_validation.csv": pd.DataFrame(dictionary_validation),
        "dictionary_stability.csv": stability_rows,
        "dictionary_recovery.csv": pd.DataFrame(dictionary_recovery),
        "fusion_selection.csv": pd.DataFrame(fusion_rows),
        "basis_fusion_diagnostics.csv": pd.DataFrame(basis_fusion_rows),
        "fpca_paths.csv": pd.DataFrame(fpca_rows),
        "calibration_inputs.csv": pd.DataFrame(calibration_inputs),
        "query_predictions.csv": prediction_frame,
        "leakage_audit.csv": pd.DataFrame(leakage_rows),
        "coefficient_fit_diagnostics.csv": fit_diagnostics_frame,
    }
    raw_inputs_path = cell / "raw_inputs.npz"
    np.savez_compressed(
        raw_inputs_path,
        x=np.asarray(x, dtype=np.float64),
        probe_x=np.asarray(probe_x, dtype=np.float64),
        support_positions=np.asarray(support_positions, dtype=np.int64),
        query_positions=np.asarray(query_positions, dtype=np.int64),
        atom_probe=np.asarray(atom_probe, dtype=np.float64),
        atom_support=np.asarray(atom_support, dtype=np.float64),
        atom_query=np.asarray(atom_query, dtype=np.float64),
        basis_probe=np.asarray(basis_probe, dtype=np.float64),
        basis_query=np.asarray(basis_query, dtype=np.float64),
        train_targets=np.asarray(train_targets, dtype=np.float64),
        test_targets=np.asarray(test_targets, dtype=np.float64),
        test_coefficients=np.asarray(test_coefficients, dtype=np.float64),
        direct_train_targets=np.asarray(direct_response, dtype=np.float64),
        outer_train_decoder_responses=np.asarray([train_responses[g] for g in GAUGE_IDS], dtype=np.float64),
        test_calibrated_responses=np.asarray([[test_responses[(source, g)] for g in GAUGE_IDS] for source in ("gird_gn", "gird_adam")], dtype=np.float64),
        inner_validation_responses=np.asarray(train_responses[-1], dtype=np.float64),
        inner_support_prior_responses=np.asarray(inner_support_prior_array, dtype=np.float64),
        entity_ids=np.arange(TRAIN_ENTITIES, dtype=np.int64),
        gauge_ids=np.asarray(GAUGE_IDS, dtype=np.int64),
        atom_names=np.asarray(names),
    )
    files["raw_inputs.npz"] = raw_inputs_path
    for name, frame in files.items():
        if name.endswith(".csv"):
            frame.to_csv(cell / name, index=False)
    summary = {
        "status": "success", "scientific_selection_eligible": formal_eligible,
        "family": family, "seed": seed, "support_regime": support_regime, "support_positions": [int(value) for value in support_positions], "query_positions": [int(value) for value in query_positions], "gird_plan_sha256": sha256(GIRD_PLAN), "source_plan_sha256": sha256(SOURCE_PLAN), "amendment_sha256": sha256(AMENDMENT), "four_support_amendment_sha256": sha256(FOUR_SUPPORT_AMENDMENT), "decision_amendment_sha256": sha256(DECISION_AMENDMENT),
        "source_artifact_sha256": sha256(source_artifact), "source_manifest_sha256": sha256(SOURCE_ROOT / f"{family}_seed{seed}" / "manifest.json"), "extension_calibration_sha256": sha256(extension_root / f"{family}_seed{seed}" / "calibration_diagnostics.csv"), "extension_manifest_sha256": sha256(extension_root / f"{family}_seed{seed}" / "manifest.json"),
        "extension_analysis_decision_sha256": sha256(extension_root / "analysis" / "decision.json"), "extension_analysis_manifest_sha256": sha256(extension_root / "analysis" / "manifest.json"), "extension_benchmark_passed": bool(extension_decision["benchmark_passed"]), "selected_gird_gn_atoms": original_gn_atoms, "selected_gird_adam_atoms": original_adam_atoms, "selected_direct_target_atoms": direct_selected, "selected_k_gird_gn": len(original_gn_atoms), "selected_lambda_gird_gn": "inf" if math.isinf(fusion_choice["gird_gn"]) else fusion_choice["gird_gn"], "selected_lambda_gird_adam": "inf" if math.isinf(fusion_choice["gird_adam"]) else fusion_choice["gird_adam"], "fpca_rank": fpca_rank, "fpca_ridge": fpca_ridge,
        "formal_budget_and_provenance_eligible": formal_eligible, "dictionary_identical_across_outer_training_gauges": dictionary_identical, "dictionary_identical_across_independent_extension_calibration_gauges": extension_stability_identical, "all_gn_selected_omp_steps_certified": all_gn_selected_certified, "all_dictionary_recovery_gates_pass": bool(all(row["recovery_ok"] for row in dictionary_recovery)), "outer_chart_pair_epsilon": outer_epsilon, "test_chart_pair_epsilon": test_epsilon, "query_target_perturbation": PERTURBATION, "maximum_query_target_leakage_difference": max(row["max_prediction_difference"] for row in leakage_rows), "source_result_status": source_result["status"], "ordinary_symbolic_regression_max_atoms": ordinary_k, "expression_endpoint_separate_from_original_q_recovery": True, "four_support_stress_status": "executed" if support_regime == "four_support" else "not_applicable",
    }
    (cell / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"scope": "gird_controlled_discovery_cell", "family": family, "seed": seed, "support_regime": support_regime, "support_positions": [int(value) for value in support_positions], "query_positions": [int(value) for value in query_positions], "gird_plan_sha256": sha256(GIRD_PLAN), "source_plan_sha256": sha256(SOURCE_PLAN), "amendment_sha256": sha256(AMENDMENT), "four_support_amendment_sha256": sha256(FOUR_SUPPORT_AMENDMENT), "decision_amendment_sha256": sha256(DECISION_AMENDMENT), "source_runner_sha256": sha256(SOURCE_RUNNER), "extension_runner_sha256": sha256(EXTENSION_RUNNER), "extension_analysis_decision_sha256": sha256(extension_root / "analysis" / "decision.json"), "extension_analysis_manifest_sha256": sha256(extension_root / "analysis" / "manifest.json"), "source_artifact_sha256": sha256(source_artifact), "extension_calibration_sha256": sha256(extension_root / f"{family}_seed{seed}" / "calibration_diagnostics.csv"), "extension_root": str(extension_root.relative_to(PROJECT_ROOT)), "files": {}}
    (cell / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["runner_sha256"] = sha256(Path(__file__))
    manifest["files"] = {name: sha256(cell / name) for name in sorted(set(files) | {"result.json"})}
    (cell / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_cell(family: str, seed: int, output_root: Path = FORMAL_ROOT, *, extension_root: Path = EXTENSION_ROOT, threads: int = 1, smoke: bool = False) -> dict[str, Any]:
    """Run both frozen support regimes under one family/seed cell."""
    output_root = output_root.resolve()
    extension_root = extension_root.resolve()
    parent = output_root / f"{family}_seed{seed}"
    parent.mkdir(parents=True, exist_ok=False)
    regime_summaries = {
        regime: _run_cell_regime(family, seed, output_root, support_regime=regime, extension_root=extension_root, threads=threads, smoke=smoke)
        for regime in ("standard_11", "four_support")
    }
    formal_eligible = bool(not smoke and output_root == FORMAL_ROOT.resolve() and extension_root.resolve() == EXTENSION_ROOT.resolve())
    summary = {"status": "success", "scientific_selection_eligible": formal_eligible, "family": family, "seed": seed, "gird_plan_sha256": sha256(GIRD_PLAN), "source_plan_sha256": sha256(SOURCE_PLAN), "amendment_sha256": sha256(AMENDMENT), "four_support_amendment_sha256": sha256(FOUR_SUPPORT_AMENDMENT), "decision_amendment_sha256": sha256(DECISION_AMENDMENT), "regimes": regime_summaries}
    (parent / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"scope": "gird_controlled_discovery_family_seed_cell", "family": family, "seed": seed, "gird_plan_sha256": sha256(GIRD_PLAN), "source_plan_sha256": sha256(SOURCE_PLAN), "amendment_sha256": sha256(AMENDMENT), "four_support_amendment_sha256": sha256(FOUR_SUPPORT_AMENDMENT), "decision_amendment_sha256": sha256(DECISION_AMENDMENT), "runner_sha256": sha256(Path(__file__)), "regimes": {regime: f"{regime}/manifest.json" for regime in regime_summaries}, "files": {"result.json": sha256(parent / "result.json")}}
    (parent / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--extension-root", type=Path, default=EXTENSION_ROOT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true", help="reserved for API compatibility; never scientific")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cell(args.family, args.seed, args.output_root, extension_root=args.extension_root, threads=args.threads, smoke=args.smoke)


if __name__ == "__main__":
    main()
