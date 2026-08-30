#!/usr/bin/env python3
"""Independently audit the frozen GIRD controlled-discovery outputs.

This analyzer never selects a method or changes a protocol.  It verifies all
15 family/seed cells, both frozen support regimes, provenance, saved OMP
paths/certificates, and aggregates query predictions into pooled and
entity-level metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = PROJECT_ROOT / "GIRD_METHOD_EXPERIMENT_PLAN_20260829.md"
SOURCE_PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md"
FOUR_SUPPORT_AMENDMENT = PROJECT_ROOT / "GIRD_FOUR_SUPPORT_POSITION_AMENDMENT_20260829.md"
DECISION_AMENDMENT = PROJECT_ROOT / "GIRD_CONTROLLED_DECISION_STATISTIC_AMENDMENT_20260829.md"
RUNNER = PROJECT_ROOT / "scripts/run_gird_controlled_discovery_20260829.py"
EXTENSION_RUNNER = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_stable_extension_20260829.py"
ROOT = PROJECT_ROOT / "runs/gird_controlled_discovery_20260829"
FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
REGIMES = ("standard_11", "four_support")
EXPECTED_PLAN_SHA256 = "c31f1fca60219f2cf2b258bac09e1a590dd7962c5e55d19f659730b743602072"
EXPECTED_SOURCE_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
EXPECTED_AMENDMENT_SHA256 = "d85db0c6d9a5b332aa3499eb9d3f105a2e89a4674b30fe90db0657bd26006613"
EXPECTED_FOUR_SUPPORT_SHA256 = "1296f79ad03d7688157d2a24145f2407d5b27b29d93683845be3107844126919"
EXPECTED_DECISION_SHA256 = "e110948ab67dec54b4b9d28c96c2276eb29912f6e988bd160bc7869236a04df0"
EXPECTED_EXTENSION_RUNNER_SHA256 = "5257d739592caf96249eb6dc5e8bca81734bf744a20e6fa8366fbaa806150fbd"
EXPECTED_EXTENSION_DECISION_SHA256 = "f3b1a222cf9e56b4209f0e6183dbd80792ae96738a905c06759fa4f655a1a1a0"
EXPECTED_EXTENSION_MANIFEST_SHA256 = "980bd9e7540cf733fbdeb5aba33ea7339b46c3968d53918836e6ab8eaf9fcc30"
PERTURBATION = 1_000_000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(1.0 - np.sum((target - prediction) ** 2) / np.sum((target - target.mean()) ** 2))


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(np.asarray(left)).rank(method="average").to_numpy()
    right_rank = pd.Series(np.asarray(right)).rank(method="average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def subspace_angle_degrees(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape[0] != candidate.shape[0]:
        return 90.0
    if np.linalg.matrix_rank(candidate, tol=1e-10) < np.linalg.matrix_rank(reference, tol=1e-10):
        return 90.0
    reference_q = np.linalg.qr(reference, mode="reduced")[0]
    candidate_q = np.linalg.qr(candidate, mode="reduced")[0]
    singular_values = np.linalg.svd(reference_q.T @ candidate_q, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(singular_values.min(), -1.0, 1.0))))


def lambda_label(value: Any) -> str:
    if value == "inf" or math.isinf(float(value)):
        return "inf"
    return str(float(value))


def recompute_omp(responses: np.ndarray, design: np.ndarray, names: list[str], max_atoms: int, epsilon: float) -> list[dict[str, Any]]:
    responses = np.asarray(responses, dtype=float)
    design = np.asarray(design, dtype=float)
    selected = [0]
    rows = []
    for stage in range(2, max_atoms + 1):
        current = design[:, selected]
        coefficients = np.linalg.lstsq(current, responses.T, rcond=None)[0].T
        residual = responses - coefficients @ current.T
        residual_norms = np.linalg.norm(residual, axis=1)
        candidates = []
        for index, name in enumerate(names):
            if index in selected:
                continue
            orth = design[:, index] - current @ np.linalg.lstsq(current, design[:, index], rcond=None)[0]
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
            candidates.append({"candidate_index": index, "candidate_name": name, "score": float(np.sum(correlation**2)), "gram_sigma_min": float(singular[-1]), "gram_sigma_max": float(singular[0]), "gram_condition": condition, "gram_rank": rank})
        require(candidates, f"independent OMP has no candidate at stage {stage}")
        candidates.sort(key=lambda row: (-row["score"], row["candidate_index"]))
        winner, runner = candidates[0], candidates[1] if len(candidates) > 1 else None
        margin = winner["score"] - (runner["score"] if runner else 0.0)
        certificate = 4.0 * float(np.linalg.norm(residual)) * epsilon + 2.0 * epsilon**2
        selected.append(winner["candidate_index"])
        for rank, row in enumerate(candidates):
            rows.append({"stage": stage, "candidate_index": row["candidate_index"], "candidate_name": row["candidate_name"], "score": row["score"], "score_margin": margin, "gram_sigma_min": row["gram_sigma_min"], "gram_sigma_max": row["gram_sigma_max"], "gram_condition": row["gram_condition"], "gram_rank": row["gram_rank"], "residual_frobenius": float(np.linalg.norm(residual)), "stability_epsilon": epsilon, "certificate_bound": certificate, "winner": rank == 0, "margin_certified": bool(rank == 0 and margin >= certificate), "selected_after": ";".join(names[index] for index in selected)})
    return rows


def fit_prior(phi_support: np.ndarray, support_y: np.ndarray, phi_probe: np.ndarray, prior_response: np.ndarray, lam: float) -> np.ndarray:
    prior_c = np.linalg.lstsq(phi_probe, prior_response, rcond=None)[0]
    if math.isinf(lam):
        return prior_c
    if lam == 0.0:
        return np.linalg.lstsq(phi_support, support_y, rcond=None)[0]
    stacked_design = np.vstack((phi_support / np.sqrt(len(phi_support)), np.sqrt(lam) * phi_probe / np.sqrt(len(phi_probe))))
    stacked_target = np.concatenate((support_y / np.sqrt(len(phi_support)), np.sqrt(lam) * (phi_probe @ prior_c) / np.sqrt(len(phi_probe))))
    return np.linalg.lstsq(stacked_design, stacked_target, rcond=None)[0]


def compare_path_rows(saved: pd.DataFrame, expected: list[dict[str, Any]], label: str) -> None:
    require(len(saved) == len(expected), f"{label}: path row count mismatch")
    saved = saved.sort_values(["fold", "stage", "candidate_index", "candidate_rank"]).reset_index(drop=True)
    expected_frame = pd.DataFrame(expected).sort_values(["fold", "stage", "candidate_index", "candidate_rank"]).reset_index(drop=True)
    for column in ("fold", "stage", "candidate_index", "candidate_rank", "candidate_name", "winner", "selected_after"):
        require(saved[column].astype(str).tolist() == expected_frame[column].astype(str).tolist(), f"{label}: {column} mismatch")
    for column in ("score", "score_margin", "gram_sigma_min", "gram_sigma_max", "gram_condition", "gram_rank", "residual_frobenius", "stability_epsilon", "certificate_bound"):
        require(np.allclose(saved[column].to_numpy(float), expected_frame[column].to_numpy(float), rtol=0.0, atol=1e-11), f"{label}: {column} mismatch")


def compare_validation_rows(saved: pd.DataFrame, expected: list[dict[str, Any]], selected_k: int, label: str) -> None:
    require(len(saved) == len(expected), f"{label}: validation row count mismatch")
    expected_frame = pd.DataFrame(expected)
    expected_frame["selected_k"] = selected_k
    expected_frame["selected_by_1pct_tie"] = expected_frame["k"] == selected_k
    saved = saved.sort_values(["fold", "k"]).reset_index(drop=True)
    expected_frame = expected_frame.sort_values(["fold", "k"]).reset_index(drop=True)
    for column in ("fold", "k", "selected_atoms", "validation_entities", "selected_k", "selected_by_1pct_tie"):
        require(saved[column].astype(str).tolist() == expected_frame[column].astype(str).tolist(), f"{label}: validation {column} mismatch")
    require(
        np.allclose(saved.normalized_validation_mse.to_numpy(float), expected_frame.normalized_validation_mse.to_numpy(float), rtol=0.0, atol=1e-11),
        f"{label}: validation score mismatch",
    )


def path_records(responses: np.ndarray, design: np.ndarray, names: list[str], entity_ids: np.ndarray, max_atoms: int, epsilon: float) -> list[dict[str, Any]]:
    records = []
    for fold in range(5):
        rows = recompute_omp(responses[(entity_ids % 5) != fold], design, names, max_atoms, epsilon)
        counters = {}
        for row in rows:
            counters[row["stage"]] = counters.get(row["stage"], 0) + 1
            records.append({**row, "fold": fold, "candidate_rank": counters[row["stage"]]})
    rows = recompute_omp(responses, design, names, max_atoms, epsilon)
    counters = {}
    for row in rows:
        counters[row["stage"]] = counters.get(row["stage"], 0) + 1
        records.append({**row, "fold": -1, "candidate_rank": counters[row["stage"]]})
    return records


def nested_selection(responses: np.ndarray, design: np.ndarray, names: list[str], entity_ids: np.ndarray, support_positions: np.ndarray, query_positions: np.ndarray, max_atoms: int = 5) -> tuple[list[str], int, list[dict[str, Any]]]:
    validation = []
    for fold in range(5):
        train_mask = (entity_ids % 5) != fold
        path = recompute_omp(responses[train_mask], design, names, max_atoms, 0.0)
        for k in range(2, max_atoms + 1):
            selected = ["1"] + [next(row["candidate_name"] for row in path if row["stage"] == stage and row["winner"]) for stage in range(2, k + 1)]
            indices = [names.index(name) for name in selected]
            coefficients = np.linalg.lstsq(design[:, indices], responses[~train_mask].T, rcond=None)[0].T
            prediction = coefficients @ design[:, indices].T
            validation.append({"fold": fold, "k": k, "selected_atoms": ";".join(selected), "normalized_validation_mse": float(np.sum((responses[~train_mask] - prediction) ** 2) / np.sum(responses[~train_mask] ** 2)), "validation_entities": int((~train_mask).sum())})
    scores = {k: float(np.median([row["normalized_validation_mse"] for row in validation if row["k"] == k])) for k in range(2, max_atoms + 1)}
    best = min(scores.values())
    selected_k = min(k for k, score in scores.items() if score <= best * 1.01)
    final_path = recompute_omp(responses, design, names, max_atoms, 0.0)
    selected_names = ["1"] + [next(row["candidate_name"] for row in final_path if row["stage"] == stage and row["winner"]) for stage in range(2, selected_k + 1)]
    return selected_names, selected_k, validation


def expected_support_positions(regime: str, size: int = 41) -> list[int]:
    return [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40] if regime == "standard_11" else [0, round((size - 1) / 3), round(2 * (size - 1) / 3), size - 1]


def validate_support_positions(actual: list[int], regime: str, size: int = 41) -> None:
    expected = expected_support_positions(regime, size)
    require(actual == expected, f"support position mismatch: expected {expected}, got {actual}")


def validate_lambda_rows(fusion: pd.DataFrame, method: str) -> None:
    selected = fusion[(fusion.method == method) & (fusion.fold == -1) & (fusion.selected.astype(bool))]
    lambda_zero = fusion[(fusion.method == method) & (fusion.fold >= 0) & (fusion["lambda"] == 0.0) & (~fusion.selected.astype(bool))]
    fold_selected = fusion[(fusion.method == method) & (fusion.fold >= 0) & (fusion.selected.astype(bool))]
    require(len(selected) == 1 and len(lambda_zero) == 5 and len(fold_selected) == 5, f"selected/zero lambda fold rows missing: {method}")


def verify_regime(cell: Path, family: str, seed: int, regime: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    regime_cell = cell / regime
    manifest = read_json(regime_cell / "manifest.json")
    result = read_json(regime_cell / "result.json")
    require(manifest.get("scope") == "gird_controlled_discovery_cell", f"regime scope mismatch: {regime_cell}")
    require(result.get("family") == family and result.get("seed") == seed and result.get("support_regime") == regime, f"regime identity mismatch: {regime_cell}")
    require(manifest.get("gird_plan_sha256") == EXPECTED_PLAN_SHA256 and manifest.get("source_plan_sha256") == EXPECTED_SOURCE_PLAN_SHA256, f"GIRD/source plan mismatch: {regime_cell}")
    require(manifest.get("amendment_sha256") == EXPECTED_AMENDMENT_SHA256 and manifest.get("four_support_amendment_sha256") == EXPECTED_FOUR_SUPPORT_SHA256 and manifest.get("decision_amendment_sha256") == EXPECTED_DECISION_SHA256, f"amendment mismatch: {regime_cell}")
    require(manifest.get("runner_sha256") == sha256(RUNNER), f"GIRD runner hash mismatch: {regime_cell}")
    require(sha256(EXTENSION_RUNNER) == EXPECTED_EXTENSION_RUNNER_SHA256 and manifest.get("extension_runner_sha256") == EXPECTED_EXTENSION_RUNNER_SHA256, f"stable extension runner hash mismatch: {regime_cell}")
    require(manifest.get("extension_analysis_decision_sha256") == EXPECTED_EXTENSION_DECISION_SHA256 and manifest.get("extension_analysis_manifest_sha256") == EXPECTED_EXTENSION_MANIFEST_SHA256, f"stable extension analysis binding mismatch: {regime_cell}")
    for name, expected in manifest.get("files", {}).items():
        require((regime_cell / name).is_file() and sha256(regime_cell / name) == expected, f"regime artifact hash mismatch: {regime_cell / name}")
    validate_support_positions(result.get("support_positions"), regime)
    require(result.get("scientific_selection_eligible") is True, f"regime is not formally eligible: {regime_cell}")
    predictions = pd.read_csv(regime_cell / "query_predictions.csv")
    paths = pd.read_csv(regime_cell / "dictionary_paths.csv")
    validation = pd.read_csv(regime_cell / "dictionary_validation.csv")
    stability = pd.read_csv(regime_cell / "dictionary_stability.csv")
    recovery = pd.read_csv(regime_cell / "dictionary_recovery.csv")
    fusion = pd.read_csv(regime_cell / "fusion_selection.csv")
    basis = pd.read_csv(regime_cell / "basis_fusion_diagnostics.csv")
    fpca = pd.read_csv(regime_cell / "fpca_paths.csv")
    calibration = pd.read_csv(regime_cell / "calibration_inputs.csv")
    leakage = pd.read_csv(regime_cell / "leakage_audit.csv")
    fit_diagnostics = pd.read_csv(regime_cell / "coefficient_fit_diagnostics.csv")
    raw_path = regime_cell / "raw_inputs.npz"
    require(raw_path.is_file(), f"raw input bundle missing: {regime_cell}")
    raw = np.load(raw_path, allow_pickle=False)
    required_raw = {"x", "probe_x", "support_positions", "query_positions", "atom_probe", "atom_support", "atom_query", "basis_probe", "basis_query", "train_targets", "test_targets", "test_coefficients", "direct_train_targets", "outer_train_decoder_responses", "test_calibrated_responses", "inner_validation_responses", "inner_support_prior_responses", "entity_ids", "gauge_ids", "atom_names"}
    require(required_raw.issubset(raw.files), f"raw input schema incomplete: {regime_cell}")
    require(raw["support_positions"].tolist() == result["support_positions"] and raw["query_positions"].tolist() == result["query_positions"], f"raw support/query split mismatch: {regime_cell}")
    names = raw["atom_names"].astype(str).tolist()
    require(len(recovery) > 0, f"dictionary recovery rows missing: {regime_cell}")
    for row in recovery.itertuples(index=False):
        selected = str(row.selected_atoms).split(";")
        indices = [names.index(name) for name in selected]
        exact = len(selected) == 3 and set(selected) == set(names[:3])
        angle = subspace_angle_degrees(raw["atom_probe"][:, :3], raw["atom_probe"][:, indices])
        recovered = bool(exact or angle <= 5.0)
        require(bool(row.exact_atom_recovery) == exact, f"recovery exactness mismatch: {regime_cell}/{row.source}/{row.stage}")
        require(np.isclose(float(row.max_function_subspace_angle_degrees), angle, rtol=0.0, atol=1e-10), f"recovery angle mismatch: {regime_cell}/{row.source}/{row.stage}")
        require(bool(row.recovery_ok) == recovered, f"recovery decision mismatch: {regime_cell}/{row.source}/{row.stage}")
    outer_responses = raw["outer_train_decoder_responses"]
    test_responses = raw["test_calibrated_responses"]
    gauge_ids = raw["gauge_ids"].tolist()
    outer_epsilon = max(float(np.linalg.norm(outer_responses[index] - outer_responses[0])) for index in range(1, len(gauge_ids)))
    test_epsilons = [max(float(np.linalg.norm(test_responses[source_index, index] - test_responses[source_index, 0])) for index in range(1, len(gauge_ids))) for source_index in range(2)]
    for source_index, source in enumerate(("gird_gn", "gird_adam")):
        for gauge_index, gauge_id in enumerate(gauge_ids):
            expected = path_records(test_responses[source_index, gauge_index], raw["atom_probe"], names, np.arange(test_responses.shape[2]), 5, test_epsilons[source_index])
            saved = paths[(paths.source == "independent_extension_calibration") & (paths.response_method == source) & (paths.gauge_id == gauge_id)]
            compare_path_rows(saved, expected, f"{regime_cell}/{source}/independent/gauge{gauge_id}")
            expected = path_records(outer_responses[gauge_index], raw["atom_probe"], names, raw["entity_ids"], 5, outer_epsilon)
            saved = paths[(paths.source == source) & (paths.gauge_id == gauge_id)]
            compare_path_rows(saved, expected, f"{regime_cell}/{source}/outer/gauge{gauge_id}")
            selected, selected_k, expected_validation = nested_selection(
                outer_responses[gauge_index], raw["atom_probe"], names, raw["entity_ids"], raw["support_positions"], raw["query_positions"]
            )
            saved_validation = validation[(validation.source == source) & (validation.gauge_id == gauge_id)]
            compare_validation_rows(saved_validation, expected_validation, selected_k, f"{regime_cell}/{source}/outer/gauge{gauge_id}")
            if gauge_id == -1:
                require(selected == list(result[f"selected_{source}_atoms"]), f"{regime_cell}/{source}: selected dictionary mismatch")
    expected = path_records(raw["direct_train_targets"], raw["atom_probe"], names, raw["entity_ids"], 5, 0.0)
    compare_path_rows(paths[(paths.source == "direct_target_omp") & (paths.gauge_id == -1)], expected, f"{regime_cell}/direct-target")
    direct_selected, direct_k, direct_validation = nested_selection(
        raw["direct_train_targets"], raw["atom_probe"], names, raw["entity_ids"], raw["support_positions"], raw["query_positions"]
    )
    compare_validation_rows(
        validation[(validation.source == "direct_target_omp") & (validation.gauge_id == -1)], direct_validation, direct_k, f"{regime_cell}/direct-target"
    )
    require(direct_selected == list(result["selected_direct_target_atoms"]), f"{regime_cell}/direct-target: selected dictionary mismatch")
    ordinary_k = min(int(result["selected_k_gird_gn"]), len(raw["support_positions"]))
    for entity_id in range(raw["test_targets"].shape[0]):
        expected_rows = recompute_omp(raw["test_targets"][entity_id, raw["support_positions"]][None, :], raw["atom_support"], names, ordinary_k, 0.0)
        counters = {}
        expected = []
        for row in expected_rows:
            counters[row["stage"]] = counters.get(row["stage"], 0) + 1
            expected.append({**row, "fold": -1, "candidate_rank": counters[row["stage"]]})
        saved = paths[(paths.source == "ordinary_symbolic_regression") & (paths.entity_id == entity_id)]
        compare_path_rows(saved, expected, f"{regime_cell}/ordinary/entity{entity_id}")
        perturbed = raw["test_targets"][entity_id, raw["support_positions"]][None, :]
        expected_rows = recompute_omp(perturbed, raw["atom_support"], names, ordinary_k, 0.0)
        counters = {}
        expected = []
        for row in expected_rows:
            counters[row["stage"]] = counters.get(row["stage"], 0) + 1
            expected.append({**row, "fold": -1, "candidate_rank": counters[row["stage"]]})
        saved = paths[(paths.source == "ordinary_symbolic_regression_leakage") & (paths.entity_id == entity_id)]
        compare_path_rows(saved, expected, f"{regime_cell}/ordinary-leakage/entity{entity_id}")
    train_targets = raw["train_targets"]
    inner_priors = raw["inner_support_prior_responses"]
    require(inner_priors.shape[:3] == (2, 5, train_targets.shape[0]), f"inner prior method/fold/entity coverage mismatch: {regime_cell}")
    finite_prior_rows = np.isfinite(inner_priors).all(axis=3)
    expected_prior_rows = np.zeros((2, 5, train_targets.shape[0]), dtype=bool)
    for fold in range(5):
        expected_prior_rows[:, fold, (np.arange(train_targets.shape[0]) % 5) == fold] = True
    require(np.array_equal(finite_prior_rows, expected_prior_rows), f"inner prior finite-mask mismatch: {regime_cell}")
    inner_paths = calibration[calibration.record_type == "inner_calibration_path"]
    require(len(inner_paths[inner_paths.method == "gird_gn"]) == train_targets.shape[0] * 15, f"inner GN path coverage mismatch: {regime_cell}")
    require(len(inner_paths[inner_paths.method == "gird_adam"]) == train_targets.shape[0] * 300, f"inner Adam path coverage mismatch: {regime_cell}")
    require(set(inner_paths.query_targets_used_for_calibration.astype(bool)) == {False}, f"inner calibration target leakage: {regime_cell}")
    gn_paths = inner_paths[inner_paths.method == "gird_gn"]
    require(np.isfinite(gn_paths[["loss", "loss_after", "step_scale", "jacobian_rank", "jacobian_condition", "support_loss"]].to_numpy(float)).all(), f"non-finite inner GN path: {regime_cell}")
    require((gn_paths.jacobian_rank == 3.0).all(), f"inner GN rank failure: {regime_cell}")
    atom_support, atom_probe, atom_query = raw["atom_support"], raw["atom_probe"], raw["atom_query"]
    for method, atoms in (("gird_gn", list(result["selected_gird_gn_atoms"])), ("gird_adam", list(result["selected_gird_adam_atoms"])), ("true_basis", names[:3])):
        indices = [names.index(name) for name in atoms]
        expected_rows = []
        fold_scores = {value: [] for value in (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, math.inf)}
        for fold in range(5):
            entities = np.flatnonzero((np.arange(train_targets.shape[0]) % 5) == fold)
            for lam in fold_scores:
                values = []
                for entity_id in entities:
                    prior_index = 0 if method in {"gird_gn", "true_basis"} else 1
                    coefficient = fit_prior(atom_support[:, indices], train_targets[entity_id, raw["support_positions"]], atom_probe[:, indices], inner_priors[prior_index, fold, entity_id], lam)
                    prediction = atom_query[:, indices] @ coefficient
                    target = train_targets[entity_id, raw["query_positions"]]
                    values.append(float(np.sqrt(np.mean((target - prediction) ** 2)) / np.std(target, ddof=0)))
                score = float(np.median(values))
                fold_scores[lam].append(score)
                expected_rows.append({"method": method, "lambda": lam, "fold": fold, "median_entity_nrmse": score, "selected": False})
        deployment_scores = {lam: float(np.median(scores)) for lam, scores in fold_scores.items()}
        deployment_best = min(deployment_scores.values())
        deployed = min((lam for lam, score in deployment_scores.items() if score <= deployment_best * 1.01), key=lambda value: (0 if math.isfinite(value) else 1, value if math.isfinite(value) else 0.0))
        for fold in range(5):
            fold_candidates = {lam: fold_scores[lam][fold] for lam in fold_scores}
            chosen, chosen_score = min(((lam, score) for lam, score in fold_candidates.items() if score <= min(fold_candidates.values()) * 1.01), key=lambda item: (0 if math.isfinite(item[0]) else 1, item[0] if math.isfinite(item[0]) else 0.0))
            expected_rows.append({"method": method, "lambda": chosen, "fold": fold, "median_entity_nrmse": chosen_score, "selected": True})
        expected_rows.append({"method": method, "lambda": deployed, "fold": -1, "median_entity_nrmse": float(np.median(fold_scores[deployed])), "selected": True})
        saved = fusion[fusion.method == method].copy()
        require(len(saved) == len(expected_rows), f"{regime_cell}/{method}: lambda row count mismatch")
        saved = saved.sort_values(["fold", "lambda"]).reset_index(drop=True)
        expected_frame = pd.DataFrame(expected_rows).sort_values(["fold", "lambda"]).reset_index(drop=True)
        require(np.allclose(saved["lambda"].replace({np.inf: 0.0}), expected_frame["lambda"].replace({np.inf: 0.0}), rtol=0.0, atol=0.0), f"{regime_cell}/{method}: lambda grid mismatch")
        require(np.allclose(saved.median_entity_nrmse, expected_frame.median_entity_nrmse, rtol=0.0, atol=1e-10), f"{regime_cell}/{method}: lambda score mismatch")
        require(saved.selected.astype(bool).tolist() == expected_frame.selected.astype(bool).tolist(), f"{regime_cell}/{method}: lambda selection mismatch")
    gn_indices = [names.index(name) for name in result["selected_gird_gn_atoms"]]
    reconstructed = np.linalg.lstsq(raw["atom_probe"][:, gn_indices], test_responses[0, 0].T, rcond=None)[0].T @ raw["atom_probe"][:, gn_indices].T
    functional_coefficients = np.linalg.lstsq(raw["basis_probe"], reconstructed.T, rcond=None)[0].T
    coordinate_rows = [{"seed": seed, "support_regime": regime, "coordinate": index, "spearman": spearman(functional_coefficients[:, index], raw["test_coefficients"][:, index])} for index in range(functional_coefficients.shape[1])]
    query_positions = raw["query_positions"]
    support_positions = raw["support_positions"]
    atom_probe, atom_support, atom_query = raw["atom_probe"], raw["atom_support"], raw["atom_query"]
    test_targets = raw["test_targets"]
    def compare_prediction(label: str, expected: np.ndarray) -> None:
        saved = predictions[predictions.method == label].sort_values(["entity_id", "query_position"]).reset_index(drop=True)
        expected_frame = pd.DataFrame({"entity_id": np.repeat(np.arange(test_targets.shape[0]), len(query_positions)), "query_position": np.tile(query_positions, test_targets.shape[0]), "prediction": expected.reshape(-1)})
        require(len(saved) == len(expected_frame), f"{regime_cell}/{label}: prediction coverage mismatch")
        require(np.allclose(saved.prediction.to_numpy(float), expected_frame.prediction.to_numpy(float), rtol=0.0, atol=1e-10), f"{regime_cell}/{label}: independently refit prediction mismatch")
    def method_prediction(atoms: list[str], prior_rows: np.ndarray, lam: float) -> np.ndarray:
        indices = [names.index(name) for name in atoms]
        values = []
        for entity_id in range(test_targets.shape[0]):
            coefficient = fit_prior(atom_support[:, indices], test_targets[entity_id, support_positions], atom_probe[:, indices], prior_rows[entity_id], lam)
            values.append(atom_query[:, indices] @ coefficient)
        return np.asarray(values)
    gn_atoms = list(result["selected_gird_gn_atoms"])
    adam_atoms = list(result["selected_gird_adam_atoms"])
    direct_atoms = list(result["selected_direct_target_atoms"])
    true_atoms = names[:3]
    gn_prior, adam_prior = test_responses[0, 0], test_responses[1, 0]
    for method, atoms, prior in (("gird_gn", gn_atoms, gn_prior), ("gird_adam", adam_atoms, adam_prior)):
        compare_prediction(f"{method}_lambda_0.0", method_prediction(atoms, prior, 0.0))
        compare_prediction(f"{method}_lambda_inf", method_prediction(atoms, prior, math.inf))
        selected_lambda = float(result[f"selected_lambda_{method}"]) if result[f"selected_lambda_{method}"] != "inf" else math.inf
        compare_prediction(f"{method}_lambda_{lambda_label(result[f'selected_lambda_{method}'])}", method_prediction(atoms, prior, selected_lambda))
    compare_prediction("support_only_omp_lambda_0.0", method_prediction(gn_atoms, np.zeros_like(gn_prior), 0.0))
    compare_prediction("direct_target_omp_lambda_0.0", method_prediction(direct_atoms, np.zeros_like(gn_prior), 0.0))
    compare_prediction("true_basis_lambda_0.0", method_prediction(true_atoms, np.zeros_like(gn_prior), 0.0))
    compare_prediction("true_basis_lambda_inf", method_prediction(true_atoms, gn_prior, math.inf))
    ordinary_values = []
    ordinary_k = min(int(result["selected_k_gird_gn"]), len(support_positions))
    for entity_id in range(test_targets.shape[0]):
        ordinary_path = recompute_omp(test_targets[entity_id, support_positions][None, :], atom_support, names, ordinary_k, 0.0)
        ordinary_atoms = ["1"] + [next(row["candidate_name"] for row in ordinary_path if row["stage"] == stage and row["winner"]) for stage in range(2, ordinary_k + 1)]
        indices = [names.index(name) for name in ordinary_atoms]
        ordinary_values.append(atom_query[:, indices] @ np.linalg.lstsq(atom_support[:, indices], test_targets[entity_id, support_positions], rcond=None)[0])
    compare_prediction("ordinary_symbolic_regression", np.asarray(ordinary_values))
    fpca_rank, fpca_ridge = int(result["fpca_rank"]), float(result["fpca_ridge"])
    train_mean = raw["train_targets"].mean(axis=0)
    _, _, train_vt = np.linalg.svd(raw["train_targets"] - train_mean, full_matrices=False)
    components = train_vt[:fpca_rank].T
    fpca_scores = {}
    for rank in range(1, 6):
        for ridge in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
            fold_scores = []
            for fold in range(5):
                inner_train = (np.arange(train_targets.shape[0]) % 5) != fold
                fold_mean = train_targets[inner_train].mean(axis=0)
                _, _, fold_vt = np.linalg.svd(train_targets[inner_train] - fold_mean, full_matrices=False)
                fold_components = fold_vt[:rank].T
                values = []
                for entity_id in np.flatnonzero(~inner_train):
                    target = train_targets[entity_id, support_positions] - fold_mean[support_positions]
                    if ridge == 0.0:
                        coefficient = np.linalg.lstsq(fold_components[support_positions], target, rcond=None)[0]
                    else:
                        design_fpca = np.vstack((fold_components[support_positions], np.sqrt(ridge) * np.eye(rank)))
                        coefficient = np.linalg.lstsq(design_fpca, np.concatenate((target, np.zeros(rank))), rcond=None)[0]
                    prediction = fold_mean[query_positions] + fold_components[query_positions] @ coefficient
                    target_query = train_targets[entity_id, query_positions]
                    values.append(float(np.sqrt(np.mean((target_query - prediction) ** 2)) / np.std(target_query, ddof=0)))
                fold_scores.append(float(np.median(values)))
            fpca_scores[(rank, ridge)] = float(np.median(fold_scores))
    fpca_best = min(fpca_scores.values())
    fpca_selected = min((key for key, score in fpca_scores.items() if score <= fpca_best * 1.01), key=lambda key: (key[0], key[1]))
    require(fpca_selected == (fpca_rank, fpca_ridge), f"{regime_cell}: FPCA selection mismatch")
    fpca_values = []
    for entity_id in range(test_targets.shape[0]):
        target = test_targets[entity_id, support_positions] - train_mean[support_positions]
        if fpca_ridge == 0.0:
            coefficient = np.linalg.lstsq(components[support_positions], target, rcond=None)[0]
        else:
            design_fpca = np.vstack((components[support_positions], np.sqrt(fpca_ridge) * np.eye(fpca_rank)))
            coefficient = np.linalg.lstsq(design_fpca, np.concatenate((target, np.zeros(fpca_rank))), rcond=None)[0]
        fpca_values.append(train_mean[query_positions] + components[query_positions] @ coefficient)
    compare_prediction("fpca", np.asarray(fpca_values))
    require(len(predictions) > 0 and set(predictions.family) == {family} and set(predictions.seed) == {seed}, f"prediction coverage mismatch: {regime_cell}")
    require(set(predictions.calibration_uses_query_targets.astype(bool)) == {False}, f"query target calibration leak: {regime_cell}")
    require(np.isfinite(predictions[["target", "prediction"]].to_numpy(dtype=float)).all(), f"non-finite predictions: {regime_cell}")
    require(np.isfinite(fit_diagnostics[["support_rank", "coefficient_count", "support_sigma_min", "support_condition"]].to_numpy(dtype=float)).all(), f"non-finite coefficient diagnostics: {regime_cell}")
    require(set(fit_diagnostics.solver) == {"float64_lstsq"} and "rank_deficient" in fit_diagnostics, f"coefficient solver provenance mismatch: {regime_cell}")
    require(np.isfinite(paths[["score", "score_margin", "gram_sigma_min", "gram_sigma_max", "gram_condition", "residual_frobenius", "stability_epsilon", "certificate_bound"]].to_numpy(dtype=float)).all(), f"non-finite OMP paths: {regime_cell}")
    require(np.isfinite(fusion[["lambda", "median_entity_nrmse"]].replace({np.inf: 0.0}).to_numpy(dtype=float)).all(), f"non-finite fusion rows: {regime_cell}")
    require(np.isfinite(leakage[["max_prediction_difference", "support_input_difference", "query_target_input_difference", "dictionary_path_input_difference", "dictionary_k_input_difference", "lambda_input_difference"]].to_numpy(dtype=float)).all(), f"non-finite leakage rows: {regime_cell}")
    require(set(leakage.method) == set(predictions.method), f"leakage method coverage mismatch: {regime_cell}")
    require({"candidate_index", "candidate_name", "score", "score_margin", "gram_sigma_min", "gram_sigma_max", "gram_condition", "gram_rank", "certificate_bound", "margin_certified"}.issubset(paths.columns), f"incomplete OMP path schema: {regime_cell}")
    require(set(paths.source) >= {"gird_gn", "gird_adam", "independent_extension_calibration", "direct_target_omp", "ordinary_symbolic_regression", "ordinary_symbolic_regression_leakage"}, f"OMP source coverage mismatch: {regime_cell}")
    # Canonical selected-method aliases make cross-seed pooled summaries
    # independent of the numeric spelling of the selected lambda.
    aliases = []
    for method in ("gird_gn", "gird_adam"):
        value = result[f"selected_lambda_{method}"]
        label = f"{method}_lambda_{lambda_label(value)}"
        selected = predictions[predictions.method == label].copy()
        require(not selected.empty, f"selected prediction rows missing: {regime_cell}/{label}")
        selected["method"] = f"{method}_selected"
        aliases.append(selected)
    predictions = pd.concat([predictions, *aliases], ignore_index=True)
    return result, predictions, {"paths": paths, "validation": validation, "stability": stability, "recovery": recovery, "fusion": fusion, "basis": basis, "fpca": fpca, "leakage": leakage, "coordinates": pd.DataFrame(coordinate_rows), "fit_diagnostics": fit_diagnostics}, {"manifest": manifest}


def audit_cell(root: Path, family: str, seed: int) -> tuple[dict[str, Any], list[pd.DataFrame], list[dict[str, Any]], list[pd.DataFrame]]:
    cell = root / f"{family}_seed{seed}"
    manifest = read_json(cell / "manifest.json")
    result = read_json(cell / "result.json")
    require(manifest.get("scope") == "gird_controlled_discovery_family_seed_cell", f"cell scope mismatch: {cell}")
    require(result.get("family") == family and result.get("seed") == seed, f"cell identity mismatch: {cell}")
    require(manifest.get("gird_plan_sha256") == EXPECTED_PLAN_SHA256 and manifest.get("source_plan_sha256") == EXPECTED_SOURCE_PLAN_SHA256, f"cell plan mismatch: {cell}")
    require(manifest.get("amendment_sha256") == EXPECTED_AMENDMENT_SHA256 and manifest.get("four_support_amendment_sha256") == EXPECTED_FOUR_SUPPORT_SHA256 and manifest.get("decision_amendment_sha256") == EXPECTED_DECISION_SHA256, f"cell amendment mismatch: {cell}")
    require(manifest.get("runner_sha256") == sha256(RUNNER), f"cell runner mismatch: {cell}")
    require(set(manifest.get("regimes", {})) == set(REGIMES), f"regime coverage mismatch: {cell}")
    regime_data = []
    audit_rows = []
    coordinate_data = []
    for regime in REGIMES:
        regime_result, prediction, frames, _ = verify_regime(cell, family, seed, regime)
        paths, stability, recovery, fusion, basis, fpca, leakage, fit_diagnostics = (
            frames[name] for name in ("paths", "stability", "recovery", "fusion", "basis", "fpca", "leakage", "fit_diagnostics")
        )
        coordinate_data.append(frames["coordinates"].assign(family=family, seed=seed))
        outer = stability[stability.stage == "outer_training"]
        independent = stability[stability.stage == "independent_extension_calibration"]
        selected_paths = paths[(paths.winner.astype(bool)) & (paths.candidate_rank == 1)]
        leakage_ok = bool((leakage.max_prediction_difference <= 1e-12).all() and (leakage.support_input_difference == 0.0).all() and (leakage.query_target_input_difference == PERTURBATION).all() and (leakage.dictionary_path_input_difference == 0.0).all() and (leakage.dictionary_k_input_difference == 0.0).all() and (leakage.lambda_input_difference == 0.0).all())
        covariance_ok = bool((basis.response_max_abs_error <= 1e-8).all() and (basis.coordinate_max_abs_error <= 1e-8).all())
        rank_audit_ok = bool("rank_deficient" in fit_diagnostics and set(fit_diagnostics.solver) == {"float64_lstsq"})
        recovery_gate_ok = bool(recovery.recovery_ok.astype(bool).all())
        four_lambda_ok = True
        for method in ("gird_gn", "gird_adam"):
            method_outer = outer[outer.source == method]
            method_independent = independent[independent.source == method]
            method_outer_stable = bool(not method_outer.empty and method_outer.selected_atoms.nunique() == 1)
            method_test_stable = bool(
                not method_independent.empty
                and method_independent.selected_atoms.nunique() == 1
                and method_independent.same_as_test_cohort_original.astype(bool).all()
            )
            method_paths = selected_paths[(selected_paths.source == method) | ((selected_paths.source == "independent_extension_calibration") & (selected_paths.response_method == method))]
            certificate_ok = bool(not method_paths.empty and method_paths.margin_certified.astype(bool).all())
            validate_lambda_rows(fusion, method)
            selected = fusion[(fusion.method == method) & (fusion.fold == -1) & (fusion.selected.astype(bool))]
            lambda_zero = fusion[(fusion.method == method) & (fusion.fold >= 0) & (fusion["lambda"] == 0.0) & (~fusion.selected.astype(bool))]
            fold_selected = fusion[(fusion.method == method) & (fusion.fold >= 0) & (fusion.selected.astype(bool))]
            selected_lambda = float(selected.iloc[0]["lambda"])
            zero_score = float(lambda_zero.median_entity_nrmse.median())
            selected_score = float(selected.iloc[0].median_entity_nrmse)
            improvement = float(1.0 - selected_score / zero_score)
            if regime == "four_support":
                positive_folds = int(sum(math.isfinite(float(value)) and float(value) > 0.0 for value in fold_selected["lambda"]))
                four_lambda_ok = four_lambda_ok and math.isfinite(selected_lambda) and selected_lambda > 0.0 and positive_folds >= 4 if method == "gird_gn" else four_lambda_ok
            else:
                positive_folds = int(sum(math.isfinite(float(value)) and float(value) > 0.0 for value in fold_selected["lambda"]))
            audit_rows.append({"family": family, "seed": seed, "support_regime": regime, "method": method, "dictionary_stable_outer": method_outer_stable, "dictionary_stable_test": method_test_stable, "selected_paths_certified": certificate_ok, "dictionary_recovery_gate": recovery_gate_ok, "basis_covariance_ok": covariance_ok, "zero_query_target_leakage": leakage_ok, "support_rank_diagnostics_ok": rank_audit_ok, "selected_lambda": selected_lambda, "lambda_zero_inner_nrmse": zero_score, "selected_lambda_inner_nrmse": selected_score, "lambda_improvement_fraction": improvement, "positive_inner_lambda_fold_count": positive_folds, "four_support_finite_lambda": four_lambda_ok if regime == "four_support" and method == "gird_gn" else True, "four_support_improvement_at_least_5pct": bool(improvement >= 0.05) if regime == "four_support" and method == "gird_gn" else True})
        regime_data.append(prediction.assign(family=family, seed=seed, support_regime=regime))
    return result, regime_data, audit_rows, coordinate_data


def aggregate_predictions(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions = pd.concat(frames, ignore_index=True)
    summary_rows = []
    entity_rows = []
    for (family, regime, method), group in all_predictions.groupby(["family", "support_regime", "method"], sort=True):
        pooled = group.groupby(["entity_id", "query_position", "x"], as_index=False).agg(target=("target", "first"), prediction=("prediction", "median"))
        scale = float(pooled.target.std(ddof=0))
        entity_nrmse = []
        for entity_id, entity in pooled.groupby("entity_id", sort=True):
            entity_scale = float(entity.target.std(ddof=0))
            value = float(np.sqrt(np.mean((entity.target - entity.prediction) ** 2)) / entity_scale)
            entity_nrmse.append(value)
            entity_rows.append({"family": family, "support_regime": regime, "method": method, "entity_id": int(entity_id), "r2": r2(entity.target.to_numpy(), entity.prediction.to_numpy()), "nrmse": value})
        summary_rows.append({"family": family, "support_regime": regime, "method": method, "pooled_r2": r2(pooled.target.to_numpy(), pooled.prediction.to_numpy()), "pooled_nrmse": float(np.sqrt(np.mean((pooled.target - pooled.prediction) ** 2)) / scale), "median_entity_nrmse": float(np.median(entity_nrmse))})
    return pd.DataFrame(summary_rows), pd.DataFrame(entity_rows)


def analyze(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    require(sha256(PLAN) == EXPECTED_PLAN_SHA256, "GIRD plan hash mismatch")
    require(sha256(SOURCE_PLAN) == EXPECTED_SOURCE_PLAN_SHA256, "source plan hash mismatch")
    require(sha256(AMENDMENT) == EXPECTED_AMENDMENT_SHA256, "numerical amendment hash mismatch")
    require(sha256(FOUR_SUPPORT_AMENDMENT) == EXPECTED_FOUR_SUPPORT_SHA256, "four-support amendment hash mismatch")
    require(sha256(DECISION_AMENDMENT) == EXPECTED_DECISION_SHA256, "decision-statistic amendment hash mismatch")
    require(root.resolve() == ROOT.resolve(), "analyzer accepts only the frozen formal GIRD root")
    analysis = root / "analysis"
    if analysis.exists():
        raise FileExistsError(f"refusing to overwrite {analysis}")
    all_results = []
    prediction_frames = []
    audit_rows = []
    coordinate_frames = []
    for family in FAMILIES:
        for seed in SEEDS:
            result, frames, rows, coordinates = audit_cell(root, family, seed)
            all_results.append(result)
            prediction_frames.extend(frames)
            audit_rows.extend(rows)
            coordinate_frames.extend(coordinates)
    summary, entity = aggregate_predictions(prediction_frames)
    audits = pd.DataFrame(audit_rows)
    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    four_audit = audits[audits.support_regime == "four_support"]
    coordinate_summary = coordinates.groupby(["family", "support_regime"], as_index=False).spearman.median()
    r2_gates = {}
    standard_baselines = {}
    for family in FAMILIES:
        family_summary = summary[summary.family == family]
        for regime in REGIMES:
            subset = family_summary[family_summary.support_regime == regime].set_index("method")
            r2_gates[f"{family}/{regime}"] = bool(subset.loc["gird_gn_lambda_inf", "pooled_r2"] >= 0.85 and subset.loc["gird_gn_selected", "pooled_r2"] >= 0.85)
        standard = family_summary[family_summary.support_regime == "standard_11"].set_index("method")
        best = min(float(standard.loc[name, "median_entity_nrmse"]) for name in ("true_basis_lambda_0.0", "direct_target_omp_lambda_0.0", "ordinary_symbolic_regression"))
        standard_baselines[family] = {"best_interpretable_median_entity_nrmse": best, "gird_gn_selected_median_entity_nrmse": float(standard.loc["gird_gn_selected", "median_entity_nrmse"]), "within_1_05": bool(float(standard.loc["gird_gn_selected", "median_entity_nrmse"]) <= 1.05 * best)}
    comparisons = {}
    four_test_improvements = {}
    for family in FAMILIES:
        four_summary = summary[(summary.family == family) & (summary.support_regime == "four_support")].set_index("method")
        selected = float(four_summary.loc["gird_gn_selected", "median_entity_nrmse"])
        zero = float(four_summary.loc["gird_gn_lambda_0.0", "median_entity_nrmse"])
        four_test_improvements[family] = {"selected_median_entity_nrmse": selected, "lambda0_median_entity_nrmse": zero, "improvement_fraction": float(1.0 - selected / zero), "at_least_5pct": bool(selected <= 0.95 * zero)}
        comparisons[family] = {"gird_gn_selected_beats_direct_target": bool(selected < float(four_summary.loc["direct_target_omp_lambda_0.0", "median_entity_nrmse"])), "gird_gn_selected_beats_ordinary": bool(selected < float(four_summary.loc["ordinary_symbolic_regression", "median_entity_nrmse"])), "gird_gn_selected_beats_fpca": bool(selected < float(four_summary.loc["fpca", "median_entity_nrmse"]))}
    gates = {
        "all_15_family_seed_cells_present": len(all_results) == 15,
        "all_30_regime_cells_formally_eligible": bool(all(result.get("scientific_selection_eligible") is True for result in all_results) and len(audits) == 60),
        "all_dictionary_chart_stability_audited": bool(audits[audits.method == "gird_gn"].dictionary_stable_outer.all() and audits[audits.method == "gird_gn"].dictionary_stable_test.all()),
        "all_dictionary_recovery_gates_pass": bool(audits.dictionary_recovery_gate.all()),
        "all_gn_selected_omp_steps_certified": bool(audits[audits.method == "gird_gn"].selected_paths_certified.all()),
        "all_basis_covariance_audits_pass": bool(audits.basis_covariance_ok.all()),
        "all_query_target_leakage_audits_zero": bool(audits.zero_query_target_leakage.all()),
        "all_support_rank_diagnostics_explicit_no_fallback": bool(audits.support_rank_diagnostics_ok.all()),
        "four_support_finite_lambda_audited": bool(audits[audits.support_regime == "four_support"].four_support_finite_lambda.all()),
        "four_support_lambda_improves_at_least_5pct": bool(all(value["at_least_5pct"] for value in four_test_improvements.values())),
        "four_support_gird_beats_direct_target": bool(all(values["gird_gn_selected_beats_direct_target"] for values in comparisons.values())),
        "all_decoder_functional_generating_coordinate_medians_at_least_0_90": bool((coordinate_summary.spearman >= 0.90).all()),
        "all_gn_functional_and_fused_pooled_r2_at_least_0_85": bool(all(r2_gates.values())),
        "standard_gn_within_1_05_of_best_interpretable_baseline": bool(all(value["within_1_05"] for value in standard_baselines.values())),
    }
    decision = {"scope": "independent GIRD controlled-discovery analysis", "gird_plan_sha256": EXPECTED_PLAN_SHA256, "source_plan_sha256": EXPECTED_SOURCE_PLAN_SHA256, "numerical_amendment_sha256": EXPECTED_AMENDMENT_SHA256, "four_support_amendment_sha256": EXPECTED_FOUR_SUPPORT_SHA256, "decision_amendment_sha256": EXPECTED_DECISION_SHA256, "gird_runner_sha256": sha256(RUNNER), "stable_extension_runner_sha256": sha256(EXTENSION_RUNNER), "primary_gates": gates, "four_support_comparisons": comparisons, "four_support_test_improvements": four_test_improvements, "coordinate_median_spearman": coordinate_summary.to_dict(orient="records"), "gn_r2_gates": r2_gates, "standard_interpretable_baselines": standard_baselines, "adam_comparator_all_dictionary_chart_stable": bool(audits[audits.method == "gird_adam"].dictionary_stable_outer.all() and audits[audits.method == "gird_adam"].dictionary_stable_test.all()), "adam_comparator_all_selected_omp_steps_certified": bool(audits[audits.method == "gird_adam"].selected_paths_certified.all()), "expression_r2_is_separate_from_original_q_recovery": True, "predictive_superiority_inferred": False, "unique_or_causal_latent_recovery_inferred": False}
    gn_audits = audits[audits.method == "gird_gn"]
    failed_certificates = gn_audits[~gn_audits.selected_paths_certified].sort_values(
        ["family", "seed", "support_regime"]
    )
    protocol_evidence = {
        "scope": "GIRD nested-selection and OMP-certificate evidence",
        "source_analyzer_sha256": sha256(Path(__file__)),
        "source_runner_sha256": sha256(RUNNER),
        "selection_protocol": {
            "outer_training_fold_assignment": "entity_id_modulo_5",
            "fold_count": 5,
            "lambda_rule": "smallest_finite_value_within_1_percent_of_best_median_score",
            "fpca_rank_ridge_uses_same_outer_training_folds": True,
            "seed_aggregation": "pointwise_median_prediction_over_5_seeds_before_metrics",
        },
        "gn_omp_certificates": {
            "total_regime_cells": int(len(gn_audits)),
            "failed_regime_cells": int(len(failed_certificates)),
            "failed_cell_ids": failed_certificates[
                ["family", "seed", "support_regime"]
            ].to_dict(orient="records"),
        },
    }
    analysis.mkdir(parents=True)
    summary.to_csv(analysis / "regime_method_summary.csv", index=False)
    entity.to_csv(analysis / "regime_entity_metrics.csv", index=False)
    coordinates.to_csv(analysis / "coordinate_spearman.csv", index=False)
    audits.to_csv(analysis / "dictionary_fusion_leakage_audit.csv", index=False)
    (analysis / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (analysis / "protocol_evidence.json").write_text(
        json.dumps(protocol_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {"scope": "independent_gird_controlled_discovery_analysis", "run_root": str(root.relative_to(PROJECT_ROOT)), "gird_plan_sha256": EXPECTED_PLAN_SHA256, "source_plan_sha256": EXPECTED_SOURCE_PLAN_SHA256, "numerical_amendment_sha256": EXPECTED_AMENDMENT_SHA256, "four_support_amendment_sha256": EXPECTED_FOUR_SUPPORT_SHA256, "decision_amendment_sha256": EXPECTED_DECISION_SHA256, "gird_runner_sha256": sha256(RUNNER), "stable_extension_runner_sha256": sha256(EXTENSION_RUNNER), "analyzer_sha256": sha256(Path(__file__)), "files": {}}
    manifest["files"] = {path.name: sha256(path) for path in sorted(analysis.iterdir()) if path.name != "manifest.json"}
    (analysis / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
