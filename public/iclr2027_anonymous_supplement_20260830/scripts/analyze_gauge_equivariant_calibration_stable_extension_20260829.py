#!/usr/bin/env python3
"""Independently audit the SVD/lstsq affine-calibration extension.

The analyzer intentionally does not import the old extension analyzer.  It
recomputes coverage, hashes, seed-median predictions, affine differences,
line-search pairing, bounds, and query-target invariance from the new root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
BASE_RUNNER = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_extension_20260829.py"
SOURCE_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
SOURCE_ANALYSIS = SOURCE_ROOT / "analysis"
NUMERICAL_AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md"
STABLE_RUNNER = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_stable_extension_20260829.py"
FAILED_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_extension_20260829"
FAILED_DECISION = FAILED_ROOT / "analysis/decision.json"
FAILED_MANIFEST = FAILED_ROOT / "analysis/manifest.json"
DEFAULT_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_stable_extension_20260829"

FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
METHODS = ("mapped_start_adam", "response_metric_gauss_newton")
GAUGE_IDS = (-1, 0, 1, 2, 3, 4)
Q_DIM = 3
ENTITY_COUNT = 48
GAUGE_COUNT = 5
ADAM_STEPS = 300
GN_STEPS = 15
PERTURBATION = 1_000_000.0
PERTURBATION_DETAILS_FILE = "query_target_perturbation_diagnostics.csv"
FROZEN_QUERY_POSITIONS = tuple(position for position in range(41) if position % 4 != 0)
EXPECTED_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
EXPECTED_NUMERICAL_AMENDMENT_SHA256 = "d85db0c6d9a5b332aa3499eb9d3f105a2e89a4674b30fe90db0657bd26006613"
EXPECTED_BASE_RUNNER_SHA256 = "13f9d21d1525582a2bb874add150bf5679f41642486b62dbbf8c63a2e3286024"
EXPECTED_FAILED_DECISION_SHA256 = "2214a5ff161573d2d9ba767e1d8dd60134ab536500979dd068c59dd4038d49f0"
EXPECTED_FAILED_MANIFEST_SHA256 = "17ce41703ea7d305d598dc14d877d28c29580f057c1f2bd23b620ab5152e4fc4"

CELL_FILES = (
    "result.json", "query_predictions.csv", "calibration_diagnostics.csv",
    "calibration_paths.csv", "gauge_diagnostics.csv", "basis_diagnostics.csv",
    "raw_readout_diagnostics.csv", "stability_bounds.csv",
    "narrow_support_diagnostics.csv", "response_geometry.csv",
    "query_target_perturbation_diagnostics.csv",
)


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


def finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    require(np.isfinite(values).all(), f"non-finite {label}")


def unique(frame: pd.DataFrame, keys: list[str], expected: int, label: str) -> None:
    require(len(frame) == expected, f"{label}: expected {expected} rows, got {len(frame)}")
    require(not frame.duplicated(keys).any(), f"{label}: duplicate keys")


def verify_path_finiteness(paths: pd.DataFrame) -> None:
    finite(paths, ["iteration", "loss", "step_scale"], "calibration paths")
    gn_columns = [
        "loss_after", "delta_norm", "acceptance_tolerance", "jacobian_rank",
        "jacobian_sigma_min", "jacobian_sigma_max", "jacobian_condition",
    ]
    finite(paths[paths.method == "response_metric_gauss_newton"], gn_columns, "GN paths")
    require(
        paths.loc[paths.method == "mapped_start_adam", gn_columns].isna().all().all(),
        "Adam rows unexpectedly populate GN-only path fields",
    )


def verify_perturbation_details(frame: pd.DataFrame, family: str, seed: int) -> dict[str, float]:
    require("target" not in frame.columns, "perturbation audit must not persist query targets")
    require(set(frame.record_type) == {"path", "calibration", "prediction"}, "perturbation record types incomplete")
    source_query = pd.read_csv(SOURCE_ROOT / f"{family}_seed{seed}" / "query_predictions.csv")
    expected_pairs = {
        int(entity): set(zip(group.query_position.astype(int), group.x.astype(float)))
        for entity, group in source_query.groupby("entity_id", sort=True)
    }
    require(
        all(set(group.query_position.astype(int)) == set(FROZEN_QUERY_POSITIONS)
            for _, group in source_query.groupby("entity_id", sort=True)),
        "source query positions do not match the frozen 30-point set",
    )
    path = frame[frame.record_type == "path"]
    calibration = frame[frame.record_type == "calibration"]
    prediction = frame[frame.record_type == "prediction"]
    unique(path, ["method", "gauge_id", "entity_id", "iteration"], len(GAUGE_IDS) * ENTITY_COUNT * (ADAM_STEPS + GN_STEPS), "perturbed paths")
    unique(calibration, ["method", "gauge_id", "entity_id"], len(METHODS) * len(GAUGE_IDS) * ENTITY_COUNT, "perturbed calibration")
    unique(prediction, ["method", "gauge_id", "entity_id", "query_position", "x"], len(METHODS) * len(GAUGE_IDS) * ENTITY_COUNT * 30, "perturbed predictions")
    require(set(prediction.method) == set(METHODS) and set(prediction.gauge_id) == set(GAUGE_IDS), "perturbed prediction coverage")
    for method in METHODS:
        expected_steps = ADAM_STEPS if method == "mapped_start_adam" else GN_STEPS
        for gauge_id in GAUGE_IDS:
            path_group = path[(path.method == method) & (path.gauge_id == gauge_id)]
            require(len(path_group) == ENTITY_COUNT * expected_steps, "perturbed path method/gauge coverage")
            require(set(path_group.iteration.astype(int)) == set(range(expected_steps)), "perturbed path iteration set")
            require(set(path_group.entity_id) == set(range(ENTITY_COUNT)), "perturbed path entity set")
            calibration_group = calibration[(calibration.method == method) & (calibration.gauge_id == gauge_id)]
            require(len(calibration_group) == ENTITY_COUNT, "perturbed calibration method/gauge coverage")
            pred_group = prediction[(prediction.method == method) & (prediction.gauge_id == gauge_id)]
            require(len(pred_group) == ENTITY_COUNT * 30, "perturbed prediction method/gauge coverage")
            for entity_id, entity_group in pred_group.groupby("entity_id", sort=True):
                pairs = set(zip(entity_group.query_position.astype(int), entity_group.x.astype(float)))
                require(pairs == expected_pairs[int(entity_id)], "query position/x changed in perturbation audit")
    path_fields = ("loss", "loss_after", "step_scale", "jacobian_rank", "jacobian_sigma_min", "jacobian_sigma_max", "jacobian_condition", "delta_norm", "acceptance_tolerance")
    maxima: dict[str, float] = {}
    for field in path_fields:
        column = f"{field}_abs_difference"
        original = path[f"{field}_original"].to_numpy(float)
        perturbed = path[f"{field}_perturbed"].to_numpy(float)
        values = path[column].to_numpy(float)
        finite_mask = np.isfinite(original)
        require(np.array_equal(finite_mask, np.isfinite(perturbed)) and np.array_equal(finite_mask, np.isfinite(values)), f"perturbed path finite-mask mismatch: {field}")
        require(np.allclose(np.abs(original[finite_mask] - perturbed[finite_mask]), values[finite_mask], rtol=1e-12, atol=1e-12), f"perturbed path audit mismatch: {field}")
        values = values[finite_mask]
        require(np.isfinite(values).all() and float(values.max(initial=0.0)) == 0.0, f"perturbed path differs: {field}")
        maxima[f"path_{field}"] = float(values.max(initial=0.0))
    for field in ("q0", "q1", "q2", "functional_c0", "functional_c1", "functional_c2"):
        original = calibration[f"{field}_original"].to_numpy(float)
        perturbed = calibration[f"{field}_perturbed"].to_numpy(float)
        values = calibration[f"{field}_abs_difference"].to_numpy(float)
        finite_mask = np.isfinite(original)
        require(np.array_equal(finite_mask, np.isfinite(perturbed)) and np.array_equal(finite_mask, np.isfinite(values)), f"perturbed calibration finite-mask mismatch: {field}")
        require(np.allclose(np.abs(original[finite_mask] - perturbed[finite_mask]), values[finite_mask], rtol=1e-12, atol=1e-12), f"perturbed calibration audit mismatch: {field}")
        values = values[finite_mask]
        require(np.isfinite(values).all() and float(values.max(initial=0.0)) == 0.0, f"perturbed calibration differs: {field}")
        maxima[f"calibration_{field}"] = float(values.max(initial=0.0))
    for field in ("prediction", "functional_prediction"):
        original = prediction[f"{field}_original"].to_numpy(float)
        perturbed = prediction[f"{field}_perturbed"].to_numpy(float)
        values = prediction[f"{field}_abs_difference"].to_numpy(float)
        finite_mask = np.isfinite(original)
        require(np.array_equal(finite_mask, np.isfinite(perturbed)) and np.array_equal(finite_mask, np.isfinite(values)), f"perturbed prediction finite-mask mismatch: {field}")
        require(np.allclose(np.abs(original[finite_mask] - perturbed[finite_mask]), values[finite_mask], rtol=1e-12, atol=1e-12), f"perturbed prediction audit mismatch: {field}")
        values = values[finite_mask]
        require(np.isfinite(values).all() and float(values.max(initial=0.0)) == 0.0, f"perturbed prediction differs: {field}")
        maxima[f"prediction_{field}"] = float(values.max(initial=0.0))
    return maxima


def verify_source_bundle() -> dict[tuple[str, int], dict[str, str]]:
    require(sha256(SOURCE_PLAN) == EXPECTED_PLAN_SHA256, "source plan hash mismatch")
    require(sha256(NUMERICAL_AMENDMENT) == EXPECTED_NUMERICAL_AMENDMENT_SHA256, "numerical amendment hash mismatch")
    require(sha256(BASE_RUNNER) == EXPECTED_BASE_RUNNER_SHA256, "base runner hash mismatch")
    require(sha256(FAILED_DECISION) == EXPECTED_FAILED_DECISION_SHA256, "failed decision hash mismatch")
    require(sha256(FAILED_MANIFEST) == EXPECTED_FAILED_MANIFEST_SHA256, "failed manifest hash mismatch")
    old_decision = read_json(FAILED_DECISION)
    require(old_decision.get("benchmark_passed") is False, "old extension is not the recorded failed result")
    require(old_decision.get("maximum_recalibrated_gn_query_response_difference") == 0.04483535069086719,
            "old failure maximum changed")
    analysis_manifest_path = SOURCE_ANALYSIS / "manifest.json"
    analysis_decision_path = SOURCE_ANALYSIS / "decision.json"
    require(analysis_manifest_path.is_file() and analysis_decision_path.is_file(), "source analysis missing")
    analysis_manifest = read_json(analysis_manifest_path)
    require(analysis_manifest.get("plan_sha256") == EXPECTED_PLAN_SHA256, "source analysis plan mismatch")
    require(analysis_manifest.get("runner_sha256") == sha256(
        PROJECT_ROOT / "scripts/run_gauge_invariant_canonical_response_benchmark_20260829.py"),
            "source analysis runner mismatch")
    require(read_json(analysis_decision_path).get("primary_gates", {}).get("all_15_cells_formal_success") is True,
            "source analysis is not terminal")
    for name, expected in analysis_manifest.get("files", {}).items():
        require((SOURCE_ANALYSIS / name).is_file() and sha256(SOURCE_ANALYSIS / name) == expected,
                f"source analysis artifact mismatch: {name}")
    records = {}
    for family in FAMILIES:
        for seed in SEEDS:
            cell = SOURCE_ROOT / f"{family}_seed{seed}"
            manifest_path, result_path, artifact_path = cell / "manifest.json", cell / "result.json", cell / "artifact.pt"
            require(all(path.is_file() for path in (manifest_path, result_path, artifact_path)), f"source cell incomplete: {cell}")
            manifest, result = read_json(manifest_path), read_json(result_path)
            require(manifest.get("plan_sha256") == EXPECTED_PLAN_SHA256, f"source plan mismatch: {cell}")
            require(result.get("status") == "success" and result.get("scientific_selection_eligible") is True,
                    f"source cell not eligible: {cell}")
            require(result.get("epochs") == 1500 and result.get("calibration_steps") == 1200 and result.get("gauge_count") == 25,
                    f"source budget mismatch: {cell}")
            for name, expected in manifest.get("files", {}).items():
                require((cell / name).is_file() and sha256(cell / name) == expected, f"source artifact mismatch: {cell}/{name}")
            records[(family, seed)] = {
                "artifact": sha256(artifact_path),
                "manifest": sha256(manifest_path),
                "analysis_manifest": sha256(analysis_manifest_path),
            }
    require(len(records) == 15, "source coverage is not 15 cells")
    return records


def verify_cell(root: Path, family: str, seed: int, source: dict[str, str]) -> dict[str, Any]:
    cell = root / f"{family}_seed{seed}"
    manifest_path = cell / "manifest.json"
    require(manifest_path.is_file(), f"stable manifest missing: {cell}")
    manifest = read_json(manifest_path)
    require(manifest.get("scope") == "gauge_equivariant_calibration_stable_extension_cell", f"stable scope mismatch: {cell}")
    require(manifest.get("family") == family and manifest.get("seed") == seed, f"stable identity mismatch: {cell}")
    require(manifest.get("numerical_amendment_sha256") == EXPECTED_NUMERICAL_AMENDMENT_SHA256, f"amendment binding mismatch: {cell}")
    require(manifest.get("source_plan_sha256") == EXPECTED_PLAN_SHA256, f"source-plan binding mismatch: {cell}")
    require(manifest.get("base_runner_sha256") == EXPECTED_BASE_RUNNER_SHA256, f"base-runner binding mismatch: {cell}")
    require(manifest.get("source_artifact_sha256") == source["artifact"], f"source-artifact binding mismatch: {cell}")
    require(manifest.get("source_manifest_sha256") == source["manifest"], f"source-manifest binding mismatch: {cell}")
    require(manifest.get("source_analysis_manifest_sha256") == source["analysis_manifest"], f"source-analysis binding mismatch: {cell}")
    require(manifest.get("failed_extension_decision_sha256") == EXPECTED_FAILED_DECISION_SHA256, f"failed-decision binding mismatch: {cell}")
    require(manifest.get("failed_extension_analysis_manifest_sha256") == EXPECTED_FAILED_MANIFEST_SHA256, f"failed-manifest binding mismatch: {cell}")
    require(manifest.get("runner_sha256") == sha256(STABLE_RUNNER), f"stable runner hash mismatch: {cell}")
    require(manifest.get("stable_solver") == "float64_lstsq", f"stable solver mismatch: {cell}")
    require(manifest.get("adam_steps") == ADAM_STEPS and manifest.get("gn_steps") == GN_STEPS,
            f"stable budget mismatch: {cell}")
    require(manifest.get("gauge_count") == GAUGE_COUNT and manifest.get("entity_count") == ENTITY_COUNT,
            f"stable coverage budget mismatch: {cell}")
    require(manifest.get("query_target_perturbation") == PERTURBATION, f"perturbation mismatch: {cell}")
    require(set(manifest.get("files", {})) == set(CELL_FILES), f"stable file manifest mismatch: {cell}")
    for name, expected in manifest["files"].items():
        require((cell / name).is_file() and sha256(cell / name) == expected, f"stable artifact mismatch: {cell}/{name}")
    result = read_json(cell / "result.json")
    require(result.get("status") == "success" and result.get("scientific_selection_eligible") is True,
            f"stable cell not formally eligible: {cell}")
    require(result.get("family") == family and result.get("seed") == seed, f"stable result identity mismatch: {cell}")
    require(result.get("adam_steps") == ADAM_STEPS and result.get("gn_steps") == GN_STEPS,
            f"stable result budget mismatch: {cell}")
    require(result.get("gauge_count") == GAUGE_COUNT and result.get("entity_count") == ENTITY_COUNT,
            f"stable result coverage mismatch: {cell}")
    require(result.get("source_result_status") == "success", f"stable source result mismatch: {cell}")
    require(result.get("scope") == "gauge_equivariant_calibration_stable_extension_cell", f"stable result scope mismatch: {cell}")
    require(result.get("stable_solver") == "float64_lstsq", f"stable result solver mismatch: {cell}")
    frames = {name: pd.read_csv(cell / name) for name in CELL_FILES if name.endswith(".csv")}
    calibration, query, paths = frames["calibration_diagnostics.csv"], frames["query_predictions.csv"], frames["calibration_paths.csv"]
    gauge, basis = frames["gauge_diagnostics.csv"], frames["basis_diagnostics.csv"]
    readout, bounds = frames["raw_readout_diagnostics.csv"], frames["stability_bounds.csv"]
    narrow, geometry = frames["narrow_support_diagnostics.csv"], frames["response_geometry.csv"]
    unique(calibration, ["method", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "calibration")
    unique(query, ["method", "gauge_id", "entity_id", "query_position", "x"], 2 * 6 * ENTITY_COUNT * 30, "query")
    unique(paths, ["method", "gauge_id", "entity_id", "iteration"], 6 * ENTITY_COUNT * (ADAM_STEPS + GN_STEPS), "paths")
    unique(gauge, ["method", "gauge_id"], 2 * 6, "gauge diagnostics")
    unique(basis, ["basis_id", "entity_id"], 10 * ENTITY_COUNT, "basis")
    unique(readout, ["method", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "readout")
    unique(bounds, ["estimate", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "bounds")
    unique(narrow, ["entity_id"], ENTITY_COUNT, "narrow support")
    unique(geometry, ["method", "entity_first", "entity_second"], 2 * (ENTITY_COUNT * (ENTITY_COUNT - 1) // 2), "geometry")
    require(set(calibration.method) == set(METHODS) and set(calibration.gauge_id) == set(GAUGE_IDS), f"calibration coverage: {cell}")
    require(set(paths.method) == set(METHODS) and set(paths.gauge_id) == set(GAUGE_IDS), f"path coverage: {cell}")
    require(set(query.method) == set(METHODS) and set(query.gauge_id) == set(GAUGE_IDS), f"query coverage: {cell}")
    require(set(gauge.method) == set(METHODS) and set(gauge.gauge_id) == set(GAUGE_IDS), f"gauge categorical coverage: {cell}")
    require(set(readout.method) == set(METHODS) and set(readout.gauge_id) == set(GAUGE_IDS), f"readout categorical coverage: {cell}")
    for method in METHODS:
        for gauge_id in GAUGE_IDS:
            calibration_group = calibration[(calibration.method == method) & (calibration.gauge_id == gauge_id)]
            query_group = query[(query.method == method) & (query.gauge_id == gauge_id)]
            path_group = paths[(paths.method == method) & (paths.gauge_id == gauge_id)]
            expected_steps = ADAM_STEPS if method == "mapped_start_adam" else GN_STEPS
            require(len(calibration_group) == ENTITY_COUNT, f"calibration group coverage: {cell}/{method}/{gauge_id}")
            require(len(query_group) == ENTITY_COUNT * 30, f"query group coverage: {cell}/{method}/{gauge_id}")
            require(len(path_group) == ENTITY_COUNT * expected_steps, f"path group coverage: {cell}/{method}/{gauge_id}")
            require(set(calibration_group.entity_id) == set(range(ENTITY_COUNT)), f"calibration entity group: {cell}/{method}/{gauge_id}")
            require(set(query_group.entity_id) == set(range(ENTITY_COUNT)), f"query entity group: {cell}/{method}/{gauge_id}")
            require(set(path_group.entity_id) == set(range(ENTITY_COUNT)), f"path entity group: {cell}/{method}/{gauge_id}")
            for entity_id in range(ENTITY_COUNT):
                require(set(path_group[path_group.entity_id == entity_id].iteration.astype(int)) == set(range(expected_steps)), f"path iteration entity set: {cell}/{method}/{gauge_id}/{entity_id}")
                require(len(query_group[query_group.entity_id == entity_id]) == 30, f"query entity rows: {cell}/{method}/{gauge_id}/{entity_id}")
    source_query = pd.read_csv(SOURCE_ROOT / f"{family}_seed{seed}" / "query_predictions.csv")
    require(
        all(set(group.query_position.astype(int)) == set(FROZEN_QUERY_POSITIONS)
            for _, group in source_query.groupby("entity_id", sort=True)),
        f"source query positions do not match frozen set: {cell}",
    )
    for entity_id, source_group in source_query.groupby("entity_id", sort=True):
        expected_query_pairs = set(zip(source_group.query_position.astype(int), source_group.x.astype(float)))
        for method in METHODS:
            for gauge_id in GAUGE_IDS:
                current_group = query[(query.method == method) & (query.gauge_id == gauge_id) & (query.entity_id == entity_id)]
                require(set(zip(current_group.query_position.astype(int), current_group.x.astype(float))) == expected_query_pairs, f"query position/x set mismatch: {cell}/{method}/{gauge_id}/{entity_id}")
    for basis_id in range(10):
        require(set(basis[basis.basis_id == basis_id].entity_id) == set(range(ENTITY_COUNT)), f"basis entity set: {cell}/{basis_id}")
    require(set(narrow.entity_id) == set(range(ENTITY_COUNT)), f"narrow entity set: {cell}")
    source_narrow = pd.read_csv(FAILED_ROOT / f"{family}_seed{seed}/narrow_support_diagnostics.csv")
    require(list(narrow.columns) == list(source_narrow.columns), f"narrow schema columns changed: {cell}")
    require(
        narrow.sort_values("entity_id").reset_index(drop=True).equals(
            source_narrow.sort_values("entity_id").reset_index(drop=True)
        ),
        f"narrow rank/condition/status/veto diagnostics changed: {cell}",
    )
    expected_pairs = {(first, second) for first in range(ENTITY_COUNT) for second in range(first + 1, ENTITY_COUNT)}
    observed_pairs = {(int(first), int(second)) for first, second in zip(geometry.entity_first, geometry.entity_second)}
    require(observed_pairs == expected_pairs, f"geometry unordered pair set: {cell}")
    for method in METHODS:
        gauge_group = gauge[gauge.method == method]
        require(len(gauge_group) == len(GAUGE_IDS) and set(gauge_group.gauge_id) == set(GAUGE_IDS), f"gauge method coverage: {cell}/{method}")
        readout_group = readout[readout.method == method]
        require(len(readout_group) == len(GAUGE_IDS) * ENTITY_COUNT and set(readout_group.gauge_id) == set(GAUGE_IDS), f"readout method coverage: {cell}/{method}")
        geometry_group = geometry[geometry.method == method]
        geometry_pairs = {(int(first), int(second)) for first, second in zip(geometry_group.entity_first, geometry_group.entity_second)}
        require(len(geometry_group) == len(expected_pairs) and geometry_pairs == expected_pairs, f"geometry method coverage: {cell}/{method}")
    for estimate in ("decoder_functional", "support_structure_req"):
        bounds_group = bounds[bounds.estimate == estimate]
        require(len(bounds_group) == len(GAUGE_IDS) * ENTITY_COUNT and set(bounds_group.gauge_id) == set(GAUGE_IDS), f"bounds categorical coverage: {cell}/{estimate}")
        for gauge_id in GAUGE_IDS:
            require(set(bounds_group[bounds_group.gauge_id == gauge_id].entity_id) == set(range(ENTITY_COUNT)), f"bounds entity coverage: {cell}/{estimate}/{gauge_id}")
    perturbation_maxima = verify_perturbation_details(frames[PERTURBATION_DETAILS_FILE], family, seed)
    finite(calibration, ["support_loss", "query_r2", "query_nrmse", "runtime_seconds"] + [f"q{i}" for i in range(Q_DIM)] + [f"functional_c{i}" for i in range(Q_DIM)], "calibration")
    finite(query, ["query_position", "x", "target", "prediction", "functional_prediction"], "query")
    verify_path_finiteness(paths)
    finite(gauge, ["condition_number", "query_response_max_abs_difference_vs_original", "functional_coordinate_max_abs_difference_vs_original", "raw_q_max_abs_change_vs_original", "support_loss_max_abs_difference_vs_original"], "gauge")
    finite(basis, ["basis_condition_number", "coordinate_max_abs_error", "function_distance_abs_error", "function_norm_distance_abs_error", "function_pair_distance_abs_error"], "basis")
    finite(readout, ["unaligned_coefficient_max_abs_change_vs_original", "covariant_coefficient_max_abs_change_vs_original", "unaligned_query_response_max_abs_change_vs_original", "covariant_query_response_max_abs_change_vs_original"], "readout")
    finite(geometry, ["response_distance", "estimated_distance"], "geometry")
    require(set(bounds.estimate) == {"decoder_functional", "support_structure_req"}, f"bound estimates: {cell}")
    for estimate, columns in {
        "decoder_functional": ["probe_sigma_min_weighted", "probe_condition_unscaled", "probe_condition_column_scaled", "projection_residual", "actual_generating_coefficient_error", "actual_decoder_probe_response_error", "projection_response_error", "projection_bound", "projection_bound_violation"],
        "support_structure_req": ["support_sigma_min", "query_amplification", "support_residual_noise_norm", "query_residual_noise_norm", "query_error_bound", "actual_query_error", "query_bound_violation", "support_condition_number"],
    }.items():
        finite(bounds[bounds.estimate == estimate], columns, f"{estimate} bounds")
    require(bounds.bound_ok.astype(bool).all(), f"bound violation: {cell}")
    require((narrow.veto.astype(bool) == False).all(), f"narrow-support veto: {cell}")
    target_spread = query.groupby(["method", "gauge_id", "entity_id", "query_position", "x"])["target"].agg(lambda v: float(np.ptp(v))).max()
    require(float(target_spread) == 0.0, f"query target mismatch: {cell}")
    require(result.get("all_gn_jacobians_full_rank") is True and result.get("all_bound_audits_ok") is True, f"cell gate summary mismatch: {cell}")
    return {"family": family, "seed": seed, "cell": cell, "result": result, "perturbation_maxima": perturbation_maxima, **frames}


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    require(denominator > 0.0, "zero target variance")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def pointwise_median(query: pd.DataFrame, family: str, method: str) -> pd.DataFrame:
    subset = query[(query.family == family) & (query.seed.isin(SEEDS)) & (query.method == method) & (query.gauge_id == -1) & (query.chart == "original")]
    keys = ["entity_id", "query_position", "x"]
    counts = subset.groupby(keys).size()
    require(len(counts) == ENTITY_COUNT * 30 and (counts == len(SEEDS)).all(), f"seed coverage mismatch: {family}/{method}")
    require(float(subset.groupby(keys)["target"].agg(lambda v: float(np.ptp(v))).max()) == 0.0, f"target mismatch: {family}/{method}")
    return subset.groupby(keys, as_index=False).agg(target=("target", "first"), prediction=("prediction", "median"), functional_prediction=("functional_prediction", "median"))


def aggregate_prediction(query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, entities = [], []
    for family in FAMILIES:
        for method in METHODS:
            median = pointwise_median(query, family, method)
            for representation, column in (("raw", "prediction"), ("functional", "functional_prediction")):
                target, prediction = median.target.to_numpy(float), median[column].to_numpy(float)
                summary.append({"family": family, "method": method, "representation": representation, "pooled_physical_r2": r2(target, prediction), "pooled_physical_rmse": float(np.sqrt(np.mean((target - prediction) ** 2)))})
                for entity_id, group in median.groupby("entity_id", sort=True):
                    et, ep = group.target.to_numpy(float), group[column].to_numpy(float)
                    entities.append({"family": family, "method": method, "representation": representation, "entity_id": int(entity_id), "r2": r2(et, ep), "rmse": float(np.sqrt(np.mean((et - ep) ** 2))), "nrmse_target_std": float(np.sqrt(np.mean((et - ep) ** 2)) / et.std(ddof=0))})
    return pd.DataFrame(summary), pd.DataFrame(entities)


def recompute_gauge_differences(query: pd.DataFrame, calibration: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    rows, max_response, max_coordinate = [], 0.0, 0.0
    keys = ["entity_id", "query_position", "x"]
    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                original = query[(query.family == family) & (query.seed == seed) & (query.method == method) & (query.gauge_id == -1)][keys + ["prediction", "functional_prediction"]]
                orig_c = calibration[(calibration.family == family) & (calibration.seed == seed) & (calibration.method == method) & (calibration.gauge_id == -1)].set_index("entity_id").sort_index()
                for gauge_id in range(GAUGE_COUNT):
                    current = query[(query.family == family) & (query.seed == seed) & (query.method == method) & (query.gauge_id == gauge_id)][keys + ["prediction", "functional_prediction"]]
                    merged = original.merge(current, on=keys, suffixes=("_original", "_gauge"), validate="one_to_one")
                    response = float(np.max(np.abs(merged.prediction_gauge - merged.prediction_original)))
                    function_response = float(np.max(np.abs(merged.functional_prediction_gauge - merged.functional_prediction_original)))
                    cur_c = calibration[(calibration.family == family) & (calibration.seed == seed) & (calibration.method == method) & (calibration.gauge_id == gauge_id)].set_index("entity_id").sort_index()
                    coordinate = float(np.max(np.abs(cur_c[[f"functional_c{i}" for i in range(Q_DIM)]].to_numpy() - orig_c[[f"functional_c{i}" for i in range(Q_DIM)]].to_numpy())))
                    raw = float(np.max(np.abs(cur_c[[f"q{i}" for i in range(Q_DIM)]].to_numpy() - orig_c[[f"q{i}" for i in range(Q_DIM)]].to_numpy())))
                    condition = float(cur_c["gauge_condition_number"].iloc[0])
                    rows.append({"family": family, "seed": seed, "method": method, "gauge_id": gauge_id, "condition_number": condition, "query_response_max_abs_difference": response, "functional_prediction_max_abs_difference": function_response, "functional_coordinate_max_abs_difference": coordinate, "raw_q_max_abs_change": raw})
                    if method == "response_metric_gauss_newton":
                        max_response, max_coordinate = max(max_response, response), max(max_coordinate, coordinate)
    return pd.DataFrame(rows), max_response, max_coordinate


def pair_step_scales(paths: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    gn = paths[paths.method == "response_metric_gauss_newton"].copy()
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            for gauge_id in range(GAUGE_COUNT):
                original = gn[(gn.family == family) & (gn.seed == seed) & (gn.gauge_id == -1)][["entity_id", "iteration", "step_scale"]]
                current = gn[(gn.family == family) & (gn.seed == seed) & (gn.gauge_id == gauge_id)][["entity_id", "iteration", "step_scale"]]
                merged = original.merge(current, on=["entity_id", "iteration"], suffixes=("_original", "_gauge"), validate="one_to_one")
                require(len(merged) == ENTITY_COUNT * GN_STEPS, f"step-scale pair coverage: {family}/{seed}/{gauge_id}")
                exact = bool(np.array_equal(merged.step_scale_original.to_numpy(), merged.step_scale_gauge.to_numpy()))
                rows.append({"family": family, "seed": seed, "gauge_id": gauge_id, "rows": len(merged), "accepted_step_fraction_original": float((merged.step_scale_original > 0).mean()), "accepted_step_fraction_gauge": float((merged.step_scale_gauge > 0).mean()), "exact_step_scale_match": exact, "max_abs_step_scale_difference": float(np.max(np.abs(merged.step_scale_original - merged.step_scale_gauge)))})
    frame = pd.DataFrame(rows)
    return frame, bool(frame.exact_step_scale_match.all())


def method_summary(calibration: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for method in METHODS:
            c = calibration[(calibration.family == family) & (calibration.method == method) & (calibration.gauge_id == -1)]
            p = paths[(paths.family == family) & (paths.method == method) & (paths.gauge_id == -1)]
            final_loss_column = "loss_after" if "loss_after" in p.columns else "loss"
            rows.append({
                "family": family,
                "method": method,
                "entities": len(c),
                "median_original_support_loss": float(c.support_loss.median()),
                "median_runtime_seconds": float(c.runtime_seconds.median()),
                "total_runtime_seconds": float(c.runtime_seconds.sum()),
                "accepted_step_fraction": float((p.step_scale > 0).mean()),
                "median_final_path_loss": float(p.groupby(["seed", "entity_id"])[final_loss_column].last().median()),
                "median_max_jacobian_condition": float(p.groupby("entity_id").jacobian_condition.max().median()) if method == "response_metric_gauss_newton" else float("nan"),
            })
    return pd.DataFrame(rows)


def condition_stratified(stable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                new = stable[(stable.family == family) & (stable.seed == seed) & (stable.method == method)]
                old = pd.read_csv(FAILED_ROOT / f"{family}_seed{seed}/gauge_diagnostics.csv")
                old = old[old.method == method].set_index("gauge_id")
                for _, row in new[new.gauge_id >= 0].iterrows():
                    cond = float(row.condition_number)
                    stratum = "condition_le_10" if cond <= 10 else ("10_lt_condition_le_100" if cond <= 100 else "condition_gt_100")
                    old_value = float(old.loc[int(row.gauge_id), "query_response_max_abs_difference_vs_original"])
                    stable_value = float(row.query_response_max_abs_difference)
                    rows.append({"family": family, "seed": seed, "method": method, "gauge_id": int(row.gauge_id), "condition_number": cond, "condition_stratum": stratum, "failed_extension_max_response_difference": old_value, "stable_max_response_difference": stable_value, "improvement_factor": old_value / stable_value if stable_value > 0 else float("inf")})
    return pd.DataFrame(rows)


def summarize_bounds(bounds: pd.DataFrame) -> tuple[bool, float, float]:
    projection = bounds[bounds.estimate == "decoder_functional"]
    structure = bounds[bounds.estimate == "support_structure_req"]
    p_violation, q_violation = float(projection.projection_bound_violation.max()), float(structure.query_bound_violation.max())
    return bool(bounds.bound_ok.astype(bool).all() and p_violation <= 1e-8 and q_violation <= 1e-8), p_violation, q_violation


def summarize_jacobians(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for method in METHODS:
            subset = paths[(paths.family == family) & (paths.method == method) & (paths.gauge_id == -1)]
            row = {"family": family, "method": method, "path_rows": len(subset), "accepted_step_fraction": float((subset.step_scale > 0).mean()), "median_final_path_loss": float(subset.groupby(["seed", "entity_id"])["loss_after" if method == "response_metric_gauss_newton" else "loss"].last().median())}
            if method == "response_metric_gauss_newton":
                row.update({"sigma_min_min": float(subset.jacobian_sigma_min.min()), "sigma_min_median": float(subset.jacobian_sigma_min.median()), "condition_median": float(subset.jacobian_condition.median()), "condition_max": float(subset.jacobian_condition.max())})
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_readout(readout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            for method in METHODS:
                subset = readout[(readout.family == family) & (readout.seed == seed) & (readout.method == method)]
                rows.append({"family": family, "seed": seed, "method": method, "max_unaligned_response_change": float(subset.unaligned_query_response_max_abs_change_vs_original.max()), "max_covariant_response_change": float(subset.covariant_query_response_max_abs_change_vs_original.max()), "max_unaligned_coefficient_change": float(subset.unaligned_coefficient_max_abs_change_vs_original.max()), "max_covariant_coefficient_change": float(subset.covariant_coefficient_max_abs_change_vs_original.max())})
    return pd.DataFrame(rows)


def summarize_basis(basis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            subset = basis[(basis.family == family) & (basis.seed == seed)]
            rows.append({"family": family, "seed": seed, "basis_rows": len(subset), "coordinate_max_abs_error": float(subset.coordinate_max_abs_error.max()), "fitted_response_max_abs_error": float(subset.function_distance_abs_error.max()), "pair_response_max_abs_error": float(subset.function_pair_distance_abs_error.max()), "basis_condition_max": float(subset.basis_condition_number.max())})
    return pd.DataFrame(rows)


def summarize_narrow(narrow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            subset = narrow[(narrow.family == family) & (narrow.seed == seed)]
            rows.append({"family": family, "seed": seed, "entities": len(subset), "rank_min": int(subset["rank"].min()), "sigma_min_min": float(subset.sigma_min.min()), "condition_max": float(subset.condition_number.max()), "rank_deficient_entities": int((subset.status == "rank_deficient").sum()), "ill_conditioned_entities": int((subset.status == "ill_conditioned").sum())})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def analyze_root(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(root == DEFAULT_ROOT.resolve(), "formal analysis requires the exact stable extension root")
    analysis = root / "analysis"
    require(not analysis.exists(), f"refusing to overwrite analysis: {analysis}")
    source = verify_source_bundle()
    expected = {f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS}
    actual = {path.name for path in root.iterdir() if path.is_dir() and path.name not in {"analysis", "launcher_logs"}}
    require(actual == expected, "stable root does not contain exactly 15 cells")
    records = [verify_cell(root, family, seed, source[(family, seed)]) for family in FAMILIES for seed in SEEDS]
    query = pd.concat([r["query_predictions.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    calibration = pd.concat([r["calibration_diagnostics.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    paths = pd.concat([r["calibration_paths.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    basis = pd.concat([r["basis_diagnostics.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    readout = pd.concat([r["raw_readout_diagnostics.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    bounds = pd.concat([r["stability_bounds.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    narrow = pd.concat([r["narrow_support_diagnostics.csv"].assign(family=r["family"], seed=r["seed"]) for r in records], ignore_index=True)
    prediction, entity = aggregate_prediction(query)
    gauge_summary, max_response, max_coordinate = recompute_gauge_differences(query, calibration)
    step_summary, step_passed = pair_step_scales(paths)
    condition_summary = condition_stratified(gauge_summary)
    method_summary_frame = method_summary(calibration, paths)
    jacobian_summary = summarize_jacobians(paths)
    readout_summary = summarize_readout(readout)
    basis_summary = summarize_basis(basis)
    narrow_summary = summarize_narrow(narrow)
    bounds_passed, projection_violation, query_violation = summarize_bounds(bounds)
    gn = prediction[(prediction.method == "response_metric_gauss_newton") & (prediction.representation == "functional")].set_index("family").pooled_physical_r2
    gn_paths = paths[paths.method == "response_metric_gauss_newton"]
    require(bool((gn_paths.jacobian_rank == Q_DIM).all() and (gn_paths.jacobian_sigma_min > 0).all()), "GN Jacobian rank gate failed")
    require(bool((basis.coordinate_max_abs_error.max() <= 1e-8) and (basis.function_distance_abs_error.max() <= 1e-8) and (basis.function_pair_distance_abs_error.max() <= 1e-8)), "basis gate failed")
    perturbation_fields = ("query_target_input_max_difference", "query_target_adam_path_max_difference", "query_target_gn_path_max_difference", "query_target_gn_step_max_difference", "query_target_q_max_difference", "query_target_functional_max_difference", "query_target_prediction_max_difference", "query_target_perturbation_max_difference")
    result_frame = pd.DataFrame([r["result"] for r in records])
    raw_perturbation_maxima = {
        field: max(float(record["perturbation_maxima"][field]) for record in records)
        for field in records[0]["perturbation_maxima"]
    }
    raw_perturbation_passed = all(value == 0.0 for value in raw_perturbation_maxima.values())
    perturbation_passed = bool(
        all(float(result_frame[field].max()) == 0.0 for field in perturbation_fields)
        and (result_frame.query_target_perturbation_value == PERTURBATION).all()
        and raw_perturbation_passed
    )
    old_max = float(read_json(FAILED_DECISION)["maximum_recalibrated_gn_query_response_difference"])
    improvement = old_max / max_response if max_response > 0 else float("inf")
    gates = {
        "all_15_cells_formal_hash_and_eligibility_verified": len(records) == 15,
        "all_stable_gn_jacobians_full_rank": bool((gn_paths.jacobian_rank == Q_DIM).all() and (gn_paths.jacobian_sigma_min > 0).all()),
        "maximum_stable_gn_query_response_difference_at_most_1e_6": max_response <= 1e-6,
        "maximum_stable_gn_functional_coordinate_difference_at_most_1e_6": max_coordinate <= 1e-6,
        "all_family_stable_gn_functional_expression_pooled_physical_r2_at_least_0_85": bool((gn >= 0.85).all()),
        "basis_coordinate_fitted_and_pair_response_errors_at_most_1e_8": bool(basis.coordinate_max_abs_error.max() <= 1e-8 and basis.function_distance_abs_error.max() <= 1e-8 and basis.function_pair_distance_abs_error.max() <= 1e-8),
        "both_deterministic_bound_audits_have_zero_violations": bounds_passed,
        "exact_query_target_invariance": perturbation_passed,
        "paired_gn_line_search_step_scales_exactly_identical": step_passed,
        "stable_response_difference_improves_failed_extension_by_at_least_100x": improvement >= 100.0,
    }
    decision = {
        "scope": "independent numerically stable gauge-equivariant calibration extension",
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "numerical_amendment_sha256": EXPECTED_NUMERICAL_AMENDMENT_SHA256,
        "base_runner_sha256": EXPECTED_BASE_RUNNER_SHA256,
        "failed_extension_decision_sha256": EXPECTED_FAILED_DECISION_SHA256,
        "failed_extension_analysis_manifest_sha256": EXPECTED_FAILED_MANIFEST_SHA256,
        "stable_runner_sha256": sha256(STABLE_RUNNER),
        "primary_gates": gates,
        "benchmark_passed": all(gates.values()),
        "gn_original_chart_functional_pooled_physical_r2": {family: float(gn[family]) for family in FAMILIES},
        "maximum_stable_gn_query_response_difference": max_response,
        "maximum_stable_gn_functional_coordinate_difference": max_coordinate,
        "failed_extension_maximum_response_difference": old_max,
        "stable_response_improvement_factor": improvement,
        "basis_max_coordinate_error": float(basis.coordinate_max_abs_error.max()),
        "basis_max_fitted_response_error": float(basis.function_distance_abs_error.max()),
        "basis_max_pair_response_error": float(basis.function_pair_distance_abs_error.max()),
        "maximum_projection_bound_violation": projection_violation,
        "maximum_query_bound_violation": query_violation,
        "raw_query_target_perturbation_maxima": raw_perturbation_maxima,
        "predictive_superiority_inferred": False,
        "unique_or_causal_latent_recovery_inferred": False,
    }
    analysis.mkdir(parents=True)
    prediction.to_csv(analysis / "family_prediction_summary.csv", index=False)
    entity.to_csv(analysis / "per_entity_metrics.csv", index=False)
    gauge_summary.to_csv(analysis / "gauge_invariance_summary.csv", index=False)
    step_summary.to_csv(analysis / "gn_step_scale_pairing.csv", index=False)
    condition_summary.to_csv(analysis / "condition_stratified_comparison.csv", index=False)
    pd.DataFrame([{"field": field, "maximum_abs_difference": value} for field, value in sorted(raw_perturbation_maxima.items())]).to_csv(analysis / "query_target_perturbation_audit_summary.csv", index=False)
    method_summary_frame.to_csv(analysis / "method_summary.csv", index=False)
    jacobian_summary.to_csv(analysis / "jacobian_summary.csv", index=False)
    readout_summary.to_csv(analysis / "raw_readout_summary.csv", index=False)
    basis_summary.to_csv(analysis / "basis_summary.csv", index=False)
    narrow_summary.to_csv(analysis / "narrow_support_summary.csv", index=False)
    write_text = lambda path, value: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text(analysis / "decision.json", decision)
    (analysis / "STABLE_EXTENSION_RESULTS.md").write_text(
        "# Numerically stable gauge-equivariant calibration: independent analysis\n\n"
        f"Overall frozen gate: **{'PASS' if decision['benchmark_passed'] else 'FAIL'}**.\n\n"
        "The primary method is direct float64 least-squares Gauss--Newton; the ordinary mapped-start Adam path is diagnostic only.\n\n"
        "## Gates\n\n" + markdown_table(pd.DataFrame([{"gate": k, "result": "PASS" if v else "FAIL"} for k, v in gates.items()])) +
        "\n\n## Five-seed pointwise-median functional prediction\n\n" + markdown_table(prediction[prediction.method == "response_metric_gauss_newton"]) +
        "\n\n## Support loss, runtime, and accepted-step diagnostics\n\n" + markdown_table(method_summary_frame) +
        "\n\n## Raw-readout and Jacobian diagnostics\n\n" + markdown_table(jacobian_summary) +
        "\n\n" + markdown_table(readout_summary) +
        "\n\n## Condition-stratified old-versus-stable response differences\n\n" + markdown_table(condition_summary) + "\n",
        encoding="utf-8",
    )
    manifest = {"scope": "independent_numerically_stable_gauge_equivariant_calibration_analysis", "run_root": str(root.relative_to(PROJECT_ROOT)), "source_plan_sha256": EXPECTED_PLAN_SHA256, "numerical_amendment_sha256": EXPECTED_NUMERICAL_AMENDMENT_SHA256, "base_runner_sha256": EXPECTED_BASE_RUNNER_SHA256, "failed_extension_decision_sha256": EXPECTED_FAILED_DECISION_SHA256, "failed_extension_analysis_manifest_sha256": EXPECTED_FAILED_MANIFEST_SHA256, "stable_runner_sha256": sha256(STABLE_RUNNER), "analyzer_sha256": sha256(Path(__file__)), "files": {}}
    for path in sorted(analysis.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    write_text(analysis / "manifest.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    analyze_root(parser.parse_args().run_root)


if __name__ == "__main__":
    main()
