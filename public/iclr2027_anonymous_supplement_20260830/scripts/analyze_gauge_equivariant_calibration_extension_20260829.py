#!/usr/bin/env python3
"""Independently verify and summarize the frozen gauge-calibration extension.

The extension runner emits one terminal cell per family/seed.  This analyzer
does not import or execute that runner's calculations.  It verifies the source
benchmark and every extension artifact, then recomputes all aggregate metrics
from the CSV files.  In particular, prediction R2 is computed after a
pointwise median over the five seeds; cell-level R2 values are never averaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
SOURCE_RUNNER = PROJECT_ROOT / "scripts/run_gauge_invariant_canonical_response_benchmark_20260829.py"
SOURCE_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
SOURCE_ANALYSIS = SOURCE_ROOT / "analysis"
AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_AMENDMENT_20260829.md"
EXTENSION_RUNNER = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_extension_20260829.py"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_extension_20260829"

FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
METHODS = ("mapped_start_adam", "response_metric_gauss_newton")
Q_DIM = 3
ENTITY_COUNT = 48
GAUGE_IDS = (-1, 0, 1, 2, 3, 4)
GAUGE_COUNT = 5
ADAM_STEPS = 300
GN_STEPS = 30
PERTURBATION = 1_000_000.0
EXPECTED_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"
EXPECTED_AMENDMENT_SHA256 = "b274f1abaee71990c5a78152d92070262caa6c37572367c60167c7bbe8fbc91f"

CELL_FILES = (
    "result.json",
    "query_predictions.csv",
    "calibration_diagnostics.csv",
    "calibration_paths.csv",
    "gauge_diagnostics.csv",
    "basis_diagnostics.csv",
    "raw_readout_diagnostics.csv",
    "stability_bounds.csv",
    "narrow_support_diagnostics.csv",
    "response_geometry.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), f"non-finite {label}")


def _unique(frame: pd.DataFrame, keys: list[str], expected: int, label: str) -> None:
    _require(len(frame) == expected, f"{label}: expected {expected} rows, got {len(frame)}")
    _require(not frame.duplicated(keys).any(), f"{label}: duplicate key rows")


def verify_source_bundle() -> dict[tuple[str, int], dict[str, Any]]:
    """Verify all 15 source cells and the source independent analysis."""
    _require(sha256(SOURCE_PLAN) == EXPECTED_PLAN_SHA256, "source plan hash mismatch")
    _require(sha256(AMENDMENT) == EXPECTED_AMENDMENT_SHA256, "amendment hash mismatch")
    analysis_manifest_path = SOURCE_ANALYSIS / "manifest.json"
    decision_path = SOURCE_ANALYSIS / "decision.json"
    _require(analysis_manifest_path.is_file() and decision_path.is_file(), "source analysis is not terminal")
    analysis_manifest = _read_json(analysis_manifest_path)
    decision = _read_json(decision_path)
    _require(analysis_manifest.get("plan_sha256") == EXPECTED_PLAN_SHA256, "source analysis plan hash mismatch")
    _require(analysis_manifest.get("runner_sha256") == sha256(SOURCE_RUNNER), "source analysis runner hash mismatch")
    _require(analysis_manifest.get("analyzer_sha256") == sha256(
        PROJECT_ROOT / "scripts/analyze_gauge_invariant_canonical_response_benchmark_20260829.py"
    ), "source analysis analyzer hash mismatch")
    for name, expected in analysis_manifest.get("files", {}).items():
        path = SOURCE_ANALYSIS / name
        _require(path.is_file() and sha256(path) == expected, f"source analysis artifact hash mismatch: {name}")
    _require(decision.get("primary_gates", {}).get("all_15_cells_formal_success") is True,
             "source analyzer did not verify all 15 cells")

    source_records: dict[tuple[str, int], dict[str, Any]] = {}
    for family in FAMILIES:
        for seed in SEEDS:
            cell = SOURCE_ROOT / f"{family}_seed{seed}"
            manifest_path = cell / "manifest.json"
            result_path = cell / "result.json"
            artifact_path = cell / "artifact.pt"
            _require(all(path.is_file() for path in (manifest_path, result_path, artifact_path)),
                     f"source cell incomplete: {cell}")
            manifest = _read_json(manifest_path)
            result = _read_json(result_path)
            _require(manifest.get("plan_sha256") == EXPECTED_PLAN_SHA256, f"source plan mismatch: {cell}")
            _require(manifest.get("runner_sha256") == sha256(SOURCE_RUNNER), f"source runner mismatch: {cell}")
            _require(result.get("status") == "success" and result.get("scientific_selection_eligible") is True,
                     f"source cell is not formally eligible: {cell}")
            _require(result.get("epochs") == 1500 and result.get("calibration_steps") == 1200,
                     f"source budget mismatch: {cell}")
            _require(result.get("gauge_count") == 25, f"source gauge count mismatch: {cell}")
            for name, expected in manifest.get("files", {}).items():
                path = cell / name
                _require(path.is_file() and sha256(path) == expected, f"source artifact hash mismatch: {path}")
            source_records[(family, seed)] = {
                "artifact_sha256": sha256(artifact_path),
                "manifest_sha256": sha256(manifest_path),
                "analysis_manifest_sha256": sha256(analysis_manifest_path),
            }
    _require(len(source_records) == 15, "source cell coverage is not 15")
    return source_records


def _verify_csv_schema(cell: Path) -> dict[str, pd.DataFrame]:
    frames = {name: pd.read_csv(cell / name) for name in CELL_FILES if name.endswith(".csv")}
    calibration = frames["calibration_diagnostics.csv"]
    query = frames["query_predictions.csv"]
    paths = frames["calibration_paths.csv"]
    gauge = frames["gauge_diagnostics.csv"]
    basis = frames["basis_diagnostics.csv"]
    readout = frames["raw_readout_diagnostics.csv"]
    bounds = frames["stability_bounds.csv"]
    narrow = frames["narrow_support_diagnostics.csv"]
    geometry = frames["response_geometry.csv"]

    _unique(calibration, ["method", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "calibration")
    _require(set(calibration["method"]) == set(METHODS), "calibration method coverage mismatch")
    _require(set(calibration["gauge_id"]) == set(GAUGE_IDS), "calibration gauge coverage mismatch")
    _require(set(calibration["chart"]) == {"original", "gauge_0", "gauge_1", "gauge_2", "gauge_3", "gauge_4"}, "calibration chart coverage mismatch")
    _require(set(calibration["entity_id"]) == set(range(ENTITY_COUNT)), "calibration entity coverage mismatch")
    _finite(calibration, ["support_loss", "query_r2", "query_nrmse", "runtime_seconds"] +
            [f"q{i}" for i in range(Q_DIM)] + [f"functional_c{i}" for i in range(Q_DIM)], "calibration")

    query_keys = ["method", "gauge_id", "entity_id", "query_position", "x"]
    _unique(query, query_keys, 2 * 6 * ENTITY_COUNT * 30, "query predictions")
    _require(set(query["method"]) == set(METHODS), "query method coverage mismatch")
    _require(set(query["gauge_id"]) == set(GAUGE_IDS), "query gauge coverage mismatch")
    _require(set(query["chart"]) == {"original", "gauge_0", "gauge_1", "gauge_2", "gauge_3", "gauge_4"}, "query chart coverage mismatch")
    _finite(query, ["query_position", "x", "target", "prediction", "functional_prediction"], "query predictions")

    path_keys = ["method", "gauge_id", "entity_id", "iteration"]
    expected_paths = 6 * ENTITY_COUNT * (ADAM_STEPS + GN_STEPS)
    _unique(paths, path_keys, expected_paths, "calibration paths")
    _require(set(paths["method"]) == set(METHODS), "path method coverage mismatch")
    _finite(paths, ["iteration", "loss", "step_scale"], "calibration paths")
    for method, steps in (("mapped_start_adam", ADAM_STEPS), ("response_metric_gauss_newton", GN_STEPS)):
        method_paths = paths[paths.method == method]
        _require(len(method_paths) == 6 * ENTITY_COUNT * steps, f"path budget mismatch: {method}")
        _require(set(method_paths["iteration"]) == set(range(steps)), f"path iterations mismatch: {method}")
    _finite(paths[paths.method == "response_metric_gauss_newton"],
            ["loss_after", "jacobian_rank", "jacobian_sigma_min", "jacobian_sigma_max", "jacobian_condition"],
            "GN calibration paths")

    _unique(gauge, ["method", "gauge_id"], 2 * 6, "gauge diagnostics")
    _require(set(gauge["gauge_id"]) == set(GAUGE_IDS), "gauge diagnostic coverage mismatch")
    _finite(gauge, ["condition_number", "query_response_max_abs_difference_vs_original",
                    "functional_coordinate_max_abs_difference_vs_original", "raw_q_max_abs_change_vs_original",
                    "support_loss_max_abs_difference_vs_original"], "gauge diagnostics")
    _unique(readout, ["method", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "raw readout")
    _finite(readout, ["unaligned_coefficient_max_abs_change_vs_original",
                      "covariant_coefficient_max_abs_change_vs_original",
                      "unaligned_query_response_max_abs_change_vs_original",
                      "covariant_query_response_max_abs_change_vs_original"], "raw readout")
    _unique(basis, ["basis_id", "entity_id"], 10 * ENTITY_COUNT, "basis diagnostics")
    _require(set(basis["basis_id"]) == set(range(10)), "basis intervention coverage mismatch")
    _require(set(basis["entity_id"]) == set(range(ENTITY_COUNT)), "basis entity coverage mismatch")
    _finite(basis, ["basis_condition_number", "coordinate_max_abs_error", "function_distance_abs_error",
                     "function_norm_distance_abs_error", "function_pair_distance_abs_error"], "basis diagnostics")
    _unique(bounds, ["estimate", "gauge_id", "entity_id"], 2 * 6 * ENTITY_COUNT, "stability bounds")
    _require(set(bounds["estimate"]) == {"decoder_functional", "support_structure_req"}, "bound estimate mismatch")
    projection_columns = ["probe_sigma_min_weighted", "probe_condition_unscaled", "probe_condition_column_scaled",
                          "projection_residual", "actual_generating_coefficient_error",
                          "actual_decoder_probe_response_error", "projection_response_error", "projection_bound",
                          "projection_bound_violation"]
    support_columns = ["support_sigma_min", "query_amplification", "support_residual_noise_norm",
                       "query_residual_noise_norm", "query_error_bound", "actual_query_error",
                       "query_bound_violation", "support_condition_number"]
    _finite(bounds[bounds.estimate == "decoder_functional"], projection_columns, "projection stability bounds")
    _finite(bounds[bounds.estimate == "support_structure_req"], support_columns, "support stability bounds")
    _require(set(bounds["bound_ok"].dropna().unique()) <= {True, False}, "bound_ok is not boolean")
    _require(bounds["bound_ok"].astype(bool).all(), "a stability bound is marked false")
    _unique(narrow, ["entity_id"], ENTITY_COUNT, "narrow support")
    _require(set(narrow["entity_id"]) == set(range(ENTITY_COUNT)), "narrow-support entity coverage mismatch")
    _require(set(narrow["veto"].dropna().unique()) <= {True, False}, "narrow veto is not boolean")
    _finite(narrow, ["rank", "sigma_min", "condition_number"], "narrow support")
    _unique(geometry, ["method", "entity_first", "entity_second"], 2 * (ENTITY_COUNT * (ENTITY_COUNT - 1) // 2), "response geometry")
    _require(set(geometry["method"]) == set(METHODS), "geometry method coverage mismatch")
    _finite(geometry, ["response_distance", "estimated_distance"], "response geometry")

    required_query = {"method", "chart", "gauge_id", "entity_id", "query_position", "x", "target", "prediction", "functional_prediction"}
    _require(required_query <= set(query.columns), "query schema is incomplete")
    _require({"jacobian_rank", "jacobian_sigma_min", "jacobian_condition"} <= set(paths.columns), "path schema is incomplete")
    # Targets must be identical across methods and gauges within a cell.
    target_spread = query.groupby(["entity_id", "query_position", "x"])["target"].agg(lambda values: float(np.ptp(values))).max()
    _require(float(target_spread) == 0.0, "query targets differ across method/chart rows")
    return frames


def verify_extension_cell(run_root: Path, family: str, seed: int, source: dict[str, Any]) -> dict[str, Any]:
    cell = run_root / f"{family}_seed{seed}"
    manifest_path = cell / "manifest.json"
    _require(manifest_path.is_file(), f"extension cell manifest missing: {cell}")
    manifest = _read_json(manifest_path)
    _require(manifest.get("scope") == "gauge_equivariant_calibration_extension_cell", f"scope mismatch: {cell}")
    _require(manifest.get("family") == family and manifest.get("seed") == seed, f"cell identity mismatch: {cell}")
    _require(manifest.get("amendment_sha256") == EXPECTED_AMENDMENT_SHA256, f"amendment provenance mismatch: {cell}")
    _require(manifest.get("source_plan_sha256") == EXPECTED_PLAN_SHA256, f"source plan provenance mismatch: {cell}")
    _require(manifest.get("source_runner_sha256") == sha256(SOURCE_RUNNER), f"source runner provenance mismatch: {cell}")
    _require(manifest.get("source_artifact_sha256") == source["artifact_sha256"], f"source artifact binding mismatch: {cell}")
    _require(manifest.get("source_manifest_sha256") == source["manifest_sha256"], f"source manifest binding mismatch: {cell}")
    _require(manifest.get("source_analysis_manifest_sha256") == source["analysis_manifest_sha256"], f"source analysis binding mismatch: {cell}")
    _require(manifest.get("runner_sha256") == sha256(EXTENSION_RUNNER), f"extension runner hash mismatch: {cell}")
    _require(manifest.get("adam_steps") == ADAM_STEPS and manifest.get("gn_steps") == GN_STEPS,
             f"extension step budget mismatch: {cell}")
    _require(manifest.get("gauge_count") == GAUGE_COUNT and manifest.get("entity_count") == ENTITY_COUNT,
             f"extension coverage budget mismatch: {cell}")
    _require(manifest.get("query_target_perturbation") == PERTURBATION, f"perturbation mismatch: {cell}")
    expected_manifest_files = set(CELL_FILES)
    _require(set(manifest.get("files", {})) == expected_manifest_files, f"extension file manifest mismatch: {cell}")
    for name, expected in manifest["files"].items():
        path = cell / name
        _require(path.is_file() and sha256(path) == expected, f"extension artifact hash mismatch: {path}")
    result = _read_json(cell / "result.json")
    _require(result.get("status") == "success" and result.get("scientific_selection_eligible") is True,
             f"extension cell is not formally eligible: {cell}")
    _require(result.get("family") == family and result.get("seed") == seed, f"result identity mismatch: {cell}")
    _require(result.get("adam_steps") == ADAM_STEPS and result.get("gn_steps") == GN_STEPS,
             f"result budget mismatch: {cell}")
    _require(result.get("gauge_count") == GAUGE_COUNT and result.get("entity_count") == ENTITY_COUNT,
             f"result coverage mismatch: {cell}")
    _require(result.get("source_result_status") == "success", f"source result status mismatch: {cell}")
    frames = _verify_csv_schema(cell)
    record = {"family": family, "seed": seed, "cell": str(cell), "result": result, "manifest": manifest}
    record.update(frames)
    return record


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((target - target.mean()) ** 2))
    _require(denominator > 0.0, "R2 target has zero variance")
    return float(1.0 - np.sum((target - prediction) ** 2) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    statistic = float(spearmanr(left, right).statistic)
    _require(np.isfinite(statistic), "non-finite Spearman statistic")
    return statistic


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small deterministic Markdown table without optional packages."""
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pointwise_median(query: pd.DataFrame, family: str, method: str) -> pd.DataFrame:
    subset = query[(query.family == family) & (query.method == method) &
                   (query.gauge_id == -1) & (query.chart == "original")]
    keys = ["entity_id", "query_position", "x"]
    counts = subset.groupby(keys).size()
    _require(len(counts) == ENTITY_COUNT * 30 and (counts == len(SEEDS)).all(),
             f"pointwise seed coverage mismatch: {family}/{method}")
    target_spread = subset.groupby(keys)["target"].agg(lambda values: float(np.ptp(values))).max()
    _require(float(target_spread) == 0.0, f"pointwise target mismatch: {family}/{method}")
    return (subset.groupby(keys, as_index=False)
            .agg(target=("target", "first"), prediction=("prediction", "median"),
                 functional_prediction=("functional_prediction", "median")))


def aggregate_predictions(query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    entity_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for method in METHODS:
            median = _pointwise_median(query, family, method)
            for representation, column in (("raw", "prediction"), ("functional", "functional_prediction")):
                target = median.target.to_numpy(dtype=float)
                prediction = median[column].to_numpy(dtype=float)
                target_std = float(target.std(ddof=0))
                rows.append({
                    "family": family, "method": method, "representation": representation,
                    "pooled_physical_r2": _r2(target, prediction),
                    "pooled_physical_rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
                    "pooled_target_std": target_std,
                })
                for entity_id, entity in median.groupby("entity_id", sort=True):
                    entity_target = entity.target.to_numpy(dtype=float)
                    entity_prediction = entity[column].to_numpy(dtype=float)
                    entity_std = float(entity_target.std(ddof=0))
                    entity_rows.append({
                        "family": family, "method": method, "representation": representation,
                        "entity_id": int(entity_id), "r2": _r2(entity_target, entity_prediction),
                        "rmse": float(np.sqrt(np.mean((entity_target - entity_prediction) ** 2))),
                        "nrmse_target_std": float(np.sqrt(np.mean((entity_target - entity_prediction) ** 2)) / entity_std),
                    })
    return pd.DataFrame(rows), pd.DataFrame(entity_rows)


def summarize_calibration(calibration: pd.DataFrame, query_summary: pd.DataFrame, entity_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for method in METHODS:
            original = calibration[(calibration.family == family) & (calibration.method == method) & (calibration.gauge_id == -1)]
            query = query_summary[(query_summary.family == family) & (query_summary.method == method)]
            raw_entities = entity_metrics[(entity_metrics.family == family) & (entity_metrics.method == method) & (entity_metrics.representation == "raw")]
            functional_entities = entity_metrics[(entity_metrics.family == family) & (entity_metrics.method == method) & (entity_metrics.representation == "functional")]
            raw_row = query[query.representation == "raw"].iloc[0]
            functional_row = query[query.representation == "functional"].iloc[0]
            rows.append({
                "family": family, "method": method,
                "median_original_support_loss": float(original.support_loss.median()),
                "median_original_runtime_seconds": float(original.runtime_seconds.median()),
                "pooled_raw_physical_r2": float(raw_row.pooled_physical_r2),
                "pooled_functional_physical_r2": float(functional_row.pooled_physical_r2),
                "raw_entity_median_r2": float(raw_entities.r2.median()),
                "functional_entity_median_r2": float(functional_entities.r2.median()),
                "raw_entity_nrmse_p95": float(raw_entities.nrmse_target_std.quantile(0.95)),
                "functional_entity_nrmse_p95": float(functional_entities.nrmse_target_std.quantile(0.95)),
                "raw_entity_nrmse_max": float(raw_entities.nrmse_target_std.max()),
                "functional_entity_nrmse_max": float(functional_entities.nrmse_target_std.max()),
                "raw_entities_nrmse_gt_10": int((raw_entities.nrmse_target_std > 10.0).sum()),
                "functional_entities_nrmse_gt_10": int((functional_entities.nrmse_target_std > 10.0).sum()),
            })
    return pd.DataFrame(rows)


def summarize_gauge_invariance(query: pd.DataFrame, calibration: pd.DataFrame, gauge: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float]:
    rows: list[dict[str, Any]] = []
    max_gn_prediction = 0.0
    max_gn_functional_prediction = 0.0
    max_gn_coordinate = 0.0
    keys = ["entity_id", "query_position", "x"]
    for family in FAMILIES:
        for method in METHODS:
            for seed in SEEDS:
                original = query[(query.family == family) & (query.seed == seed) & (query.method == method) & (query.gauge_id == -1)][keys + ["prediction", "functional_prediction"]]
                for gauge_id in range(GAUGE_COUNT):
                    current = query[(query.family == family) & (query.seed == seed) & (query.method == method) & (query.gauge_id == gauge_id)][keys + ["prediction", "functional_prediction"]]
                    merged = original.merge(current, on=keys, suffixes=("_original", "_gauge"), validate="one_to_one")
                    prediction_change = float(np.max(np.abs(merged.prediction_gauge - merged.prediction_original)))
                    functional_change = float(np.max(np.abs(merged.functional_prediction_gauge - merged.functional_prediction_original)))
                    original_c = calibration[(calibration.family == family) & (calibration.seed == seed) & (calibration.method == method) & (calibration.gauge_id == -1)].set_index("entity_id").sort_index()
                    current_c = calibration[(calibration.family == family) & (calibration.seed == seed) & (calibration.method == method) & (calibration.gauge_id == gauge_id)].set_index("entity_id").sort_index()
                    coordinate_change = float(np.max(np.abs(current_c[[f"functional_c{i}" for i in range(Q_DIM)]].to_numpy() - original_c[[f"functional_c{i}" for i in range(Q_DIM)]].to_numpy())))
                    raw_change = float(np.max(np.abs(current_c[[f"q{i}" for i in range(Q_DIM)]].to_numpy() - original_c[[f"q{i}" for i in range(Q_DIM)]].to_numpy())))
                    rows.append({
                        "family": family, "seed": seed, "method": method, "gauge_id": gauge_id,
                        "query_response_max_abs_difference": prediction_change,
                        "functional_prediction_max_abs_difference": functional_change,
                        "functional_coordinate_max_abs_difference": coordinate_change,
                        "raw_q_max_abs_change": raw_change,
                    })
                    if method == "response_metric_gauss_newton":
                        max_gn_prediction = max(max_gn_prediction, prediction_change)
                        max_gn_functional_prediction = max(max_gn_functional_prediction, functional_change)
                        max_gn_coordinate = max(max_gn_coordinate, coordinate_change)
    _require(len(rows) == len(FAMILIES) * len(SEEDS) * len(METHODS) * GAUGE_COUNT, "gauge summary coverage mismatch")
    # Cross-check the runner's terminal summaries without using them as the primary calculation.
    _require(float(gauge.query_response_max_abs_difference_vs_original.max()) >= max_gn_prediction,
             "gauge diagnostic does not cover recomputed response change")
    _require(float(gauge.functional_coordinate_max_abs_difference_vs_original.max()) >= max_gn_coordinate,
             "gauge diagnostic does not cover recomputed coordinate change")
    return pd.DataFrame(rows), max_gn_prediction, max_gn_functional_prediction, max_gn_coordinate


def summarize_jacobians(paths: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    gn = paths[paths.method == "response_metric_gauss_newton"].copy()
    _require(len(gn) == len(FAMILIES) * len(SEEDS) * len(GAUGE_IDS) * ENTITY_COUNT * GN_STEPS,
             "GN path aggregate coverage mismatch")
    full_rank = bool((gn.jacobian_rank == Q_DIM).all() & (gn.jacobian_sigma_min > 0.0).all())
    rows = []
    for family in FAMILIES:
        subset = gn[gn.family == family]
        rows.append({
            "family": family, "path_rows": len(subset), "all_full_rank": bool((subset.jacobian_rank == Q_DIM).all()),
            "sigma_min_min": float(subset.jacobian_sigma_min.min()),
            "sigma_min_median": float(subset.jacobian_sigma_min.median()),
            "condition_median": float(subset.jacobian_condition.median()),
            "condition_max": float(subset.jacobian_condition.max()),
            "accepted_step_fraction": float((subset.step_scale > 0.0).mean()),
        })
    return pd.DataFrame(rows), full_rank


def summarize_basis(basis: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for (family, seed, basis_id), subset in basis.groupby(["family", "seed", "basis_id"], sort=True):
        rows.append({
            "family": family, "seed": int(seed), "basis_id": int(basis_id), "entities": len(subset),
            "condition_number": float(subset.basis_condition_number.iloc[0]),
            "coordinate_max_abs_error": float(subset.coordinate_max_abs_error.max()),
            "function_distance_max_abs_error": float(subset.function_distance_abs_error.max()),
            "function_pair_distance_max_abs_error": float(subset.function_pair_distance_abs_error.max()),
            "function_norm_distance_max_abs_error_diagnostic": float(subset.function_norm_distance_abs_error.max()),
        })
    maxima = {
        "coordinate": float(basis.coordinate_max_abs_error.max()),
        "fitted_response": float(basis.function_distance_abs_error.max()),
        "pair_response": float(basis.function_pair_distance_abs_error.max()),
    }
    _require(len(rows) == len(FAMILIES) * len(SEEDS) * 10, "basis summary coverage mismatch")
    return pd.DataFrame(rows), maxima


def summarize_bounds(bounds: pd.DataFrame) -> tuple[pd.DataFrame, bool, float, float]:
    rows = []
    for estimate, subset in bounds.groupby("estimate", sort=True):
        row = {
            "estimate": estimate, "rows": len(subset), "all_bound_ok": bool(subset.bound_ok.astype(bool).all()),
            "max_projection_bound_violation": float(subset.get("projection_bound_violation", pd.Series([0.0])).max()),
            "max_query_bound_violation": float(subset.get("query_bound_violation", pd.Series([0.0])).max()),
            "max_actual_error": float(subset.get("actual_query_error", pd.Series([0.0])).max()),
        }
        if estimate == "decoder_functional":
            row.update({
                "probe_sigma_min_min": float(subset.probe_sigma_min_weighted.min()),
                "probe_condition_unscaled_max": float(subset.probe_condition_unscaled.max()),
                "probe_condition_column_scaled_max": float(subset.probe_condition_column_scaled.max()),
            })
        else:
            row.update({
                "support_sigma_min_min": float(subset.support_sigma_min.min()),
                "query_amplification_max": float(subset.query_amplification.max()),
                "support_condition_number_max": float(subset.support_condition_number.max()),
            })
        rows.append(row)
    projection = bounds[bounds.estimate == "decoder_functional"]
    structure = bounds[bounds.estimate == "support_structure_req"]
    projection_violation = float(projection.projection_bound_violation.max())
    query_violation = float(structure.query_bound_violation.max())
    passed = bool(bounds.bound_ok.astype(bool).all() and projection_violation <= 1e-8 and query_violation <= 1e-8)
    return pd.DataFrame(rows), passed, projection_violation, query_violation


def summarize_geometry(geometry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, seed, method), subset in geometry.groupby(["family", "seed", "method"], sort=True):
        rows.append({
            "family": family, "seed": int(seed), "method": method,
            "pair_count": len(subset), "response_distance_median": float(subset.response_distance.median()),
            "estimated_distance_median": float(subset.estimated_distance.median()),
            "response_vs_estimated_distance_spearman": _spearman(subset.response_distance.to_numpy(), subset.estimated_distance.to_numpy()),
        })
    _require(len(rows) == len(FAMILIES) * len(SEEDS) * len(METHODS), "geometry summary coverage mismatch")
    return pd.DataFrame(rows)


def summarize_readout(readout: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for (family, seed, method), subset in readout.groupby(["family", "seed", "method"], sort=True):
        rows.append({
            "family": family, "seed": int(seed), "method": method,
            "max_unaligned_response_change": float(subset.unaligned_query_response_max_abs_change_vs_original.max()),
            "max_covariant_response_change": float(subset.covariant_query_response_max_abs_change_vs_original.max()),
            "max_unaligned_coefficient_change": float(subset.unaligned_coefficient_max_abs_change_vs_original.max()),
            "max_covariant_coefficient_change": float(subset.covariant_coefficient_max_abs_change_vs_original.max()),
        })
    maxima = {}
    for method in METHODS:
        subset = readout[readout.method == method]
        maxima[f"{method}_unaligned"] = float(subset.unaligned_query_response_max_abs_change_vs_original.max())
        maxima[f"{method}_covariant"] = float(subset.covariant_query_response_max_abs_change_vs_original.max())
    return pd.DataFrame(rows), maxima


def summarize_narrow(narrow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, seed), subset in narrow.groupby(["family", "seed"], sort=True):
        rows.append({
            "family": family, "seed": int(seed), "entities": len(subset),
            "rank_min": int(subset["rank"].min()), "sigma_min_min": float(subset.sigma_min.min()),
            "condition_max": float(subset.condition_number.max()),
            "rank_deficient_entities": int((subset.status == "rank_deficient").sum()),
            "ill_conditioned_entities": int((subset.status == "ill_conditioned").sum()),
            "veto_any": bool(subset.veto.astype(bool).any()),
        })
    _require(len(rows) == len(FAMILIES) * len(SEEDS), "narrow-support summary coverage mismatch")
    _require(not narrow.veto.astype(bool).any(), "narrow-support diagnostic veto was set")
    return pd.DataFrame(rows)


def summarize_perturbation(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, bool]:
    fields = ("query_target_input_max_difference", "query_target_adam_path_max_difference",
              "query_target_gn_path_max_difference", "query_target_gn_step_max_difference",
              "query_target_q_max_difference", "query_target_functional_max_difference",
              "query_target_prediction_max_difference", "query_target_perturbation_max_difference")
    rows = []
    for result in results:
        row = {"family": result["family"], "seed": result["seed"], "perturbation_value": result.get("query_target_perturbation_value")}
        row.update({field: result.get(field) for field in fields})
        rows.append(row)
    frame = pd.DataFrame(rows)
    passed = bool(all(float(frame[field].max()) == 0.0 for field in fields) and
                  (frame.perturbation_value == PERTURBATION).all())
    return frame, passed


def _markdown(decision: dict[str, Any], family_summary: pd.DataFrame, basis_summary: pd.DataFrame,
              geometry_summary: pd.DataFrame, narrow_summary: pd.DataFrame) -> str:
    gates = decision["primary_gates"]
    lines = [
        "# Gauge-equivariant calibration extension: independent analysis",
        "",
        "This report is generated only after all 15 formal cells and their source hashes pass validation.",
        "Prediction metrics below use the original chart and a pointwise median over the five seeds before pooled physical-unit R² is computed; cell R² values are not averaged.",
        "",
        "## Decision",
        "",
        f"- Overall frozen gate: **{'PASS' if decision['benchmark_passed'] else 'FAIL'}**.",
        "- This experiment tests affine-equivariant support calibration for the declared decoder/full-rank regime. It does not test nonlinear-gauge universality, automatic basis discovery, unique causal recovery, or predictive superiority.",
        "",
        "## Primary gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, value in gates.items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines += ["", "## Original-chart prediction summary", "", _markdown_table(family_summary),
              "", "The NRMSE tail columns are descriptive diagnostics, normalized by each entity's target standard deviation.",
              "", "## Basis and geometry diagnostics", "", _markdown_table(basis_summary),
              "", "Response-induced geometry (Spearman between generating and estimated probe-response pair distances):", "",
              _markdown_table(geometry_summary), "", "## Narrow support", "", _markdown_table(narrow_summary),
              "", "Narrow-support rank/conditioning is diagnostic only and never vetoes the main experiment.", ""]
    return "\n".join(lines)


def analyze_root(run_root: Path) -> dict[str, Any]:
    """Run the complete independent analysis, refusing to overwrite output."""
    run_root = run_root.resolve()
    analysis_root = run_root / "analysis"
    _require(not analysis_root.exists(), f"refusing to overwrite analysis root: {analysis_root}")
    source_records = verify_source_bundle()
    expected_cells = {f"{family}_seed{seed}" for family in FAMILIES for seed in SEEDS}
    auxiliary_directories = {"analysis", "launcher_logs"}
    actual_cells = {path.name for path in run_root.iterdir()
                    if path.is_dir() and path.name not in auxiliary_directories}
    _require(actual_cells == expected_cells, "extension run root does not contain exactly the 15 expected cells")
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        for seed in SEEDS:
            records.append(verify_extension_cell(run_root, family, seed, source_records[(family, seed)]))
    _require(len(records) == 15, "extension formal coverage is not 15")

    results = [record["result"] for record in records]
    calibration = pd.concat([record["calibration_diagnostics.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    query = pd.concat([record["query_predictions.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    paths = pd.concat([record["calibration_paths.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    gauge = pd.concat([record["gauge_diagnostics.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    basis = pd.concat([record["basis_diagnostics.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    readout = pd.concat([record["raw_readout_diagnostics.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    bounds = pd.concat([record["stability_bounds.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    narrow = pd.concat([record["narrow_support_diagnostics.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)
    geometry = pd.concat([record["response_geometry.csv"].assign(family=record["family"], seed=record["seed"]) for record in records], ignore_index=True)

    prediction_summary, entity_metrics = aggregate_predictions(query)
    family_summary = summarize_calibration(calibration, prediction_summary, entity_metrics)
    gauge_summary, max_gn_prediction, max_gn_functional_prediction, max_gn_coordinate = summarize_gauge_invariance(query, calibration, gauge)
    jacobian_summary, all_gn_full_rank = summarize_jacobians(paths)
    basis_summary, basis_max = summarize_basis(basis)
    bounds_summary, bounds_passed, projection_violation, query_violation = summarize_bounds(bounds)
    perturbation_summary, perturbation_passed = summarize_perturbation(results)
    geometry_summary = summarize_geometry(geometry)
    readout_summary, readout_maxima = summarize_readout(readout)
    narrow_summary = summarize_narrow(narrow)
    adam_diagnostics = family_summary[family_summary.method == "mapped_start_adam"].copy()
    gauge_maxima = (gauge_summary[gauge_summary.method == "mapped_start_adam"]
                    .groupby("family", as_index=False)
                    .agg(adam_max_gauge_query_response_difference=("query_response_max_abs_difference", "max"),
                         adam_max_gauge_functional_coordinate_difference=("functional_coordinate_max_abs_difference", "max"),
                         adam_max_gauge_raw_q_change=("raw_q_max_abs_change", "max")))
    adam_diagnostics = adam_diagnostics.merge(gauge_maxima, on="family", validate="one_to_one")

    gn_prediction = prediction_summary[(prediction_summary.method == "response_metric_gauss_newton") & (prediction_summary.representation == "raw")].set_index("family")["pooled_physical_r2"]
    gn_functional = prediction_summary[(prediction_summary.method == "response_metric_gauss_newton") & (prediction_summary.representation == "functional")].set_index("family")["pooled_physical_r2"]
    gates = {
        "all_15_cells_formal_hash_and_eligibility_verified": len(records) == 15,
        "all_response_metric_gn_jacobians_full_rank": all_gn_full_rank,
        "maximum_recalibrated_gn_query_response_difference_at_most_1e_6": max_gn_prediction <= 1e-6,
        "maximum_recalibrated_gn_functional_coordinate_difference_at_most_1e_6": max_gn_coordinate <= 1e-6,
        "all_family_gn_functional_expression_pooled_physical_r2_at_least_0_85": bool((gn_functional >= 0.85).all()),
        "basis_coordinate_error_at_most_1e_8": basis_max["coordinate"] <= 1e-8,
        "basis_fitted_response_error_at_most_1e_8": basis_max["fitted_response"] <= 1e-8,
        "basis_pair_response_error_at_most_1e_8": basis_max["pair_response"] <= 1e-8,
        "both_deterministic_bound_audits_have_zero_violations": bounds_passed,
        "exact_query_target_invariance": perturbation_passed,
    }
    decision = {
        "scope": "independent gauge-equivariant calibration extension",
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "extension_runner_sha256": sha256(EXTENSION_RUNNER),
        "source_analysis_manifest_sha256": sha256(SOURCE_ANALYSIS / "manifest.json"),
        "primary_gates": gates,
        "benchmark_passed": all(gates.values()),
        "gn_original_chart_raw_pooled_physical_r2": {family: float(gn_prediction[family]) for family in FAMILIES},
        "gn_original_chart_functional_pooled_physical_r2": {family: float(gn_functional[family]) for family in FAMILIES},
        "maximum_recalibrated_gn_query_response_difference": max_gn_prediction,
        "maximum_recalibrated_gn_functional_coordinate_difference": max_gn_coordinate,
        "maximum_recalibrated_gn_functional_prediction_difference": max_gn_functional_prediction,
        "basis_primary_max_errors": basis_max,
        "maximum_projection_bound_violation": projection_violation,
        "maximum_query_bound_violation": query_violation,
        "maximum_unaligned_raw_readout_response_change_adam": readout_maxima["mapped_start_adam_unaligned"],
        "maximum_covariant_raw_readout_response_change_adam": readout_maxima["mapped_start_adam_covariant"],
        "maximum_unaligned_raw_readout_response_change_gn": readout_maxima["response_metric_gauss_newton_unaligned"],
        "maximum_covariant_raw_readout_response_change_gn": readout_maxima["response_metric_gauss_newton_covariant"],
        "covariant_raw_readout_diagnostic_passed_gn": readout_maxima["response_metric_gauss_newton_covariant"] <= 1e-6,
        "unaligned_raw_readout_failure_observed_gn": readout_maxima["response_metric_gauss_newton_unaligned"] > 1e-6,
        "narrow_support_veto": bool(narrow.veto.astype(bool).any()),
        "predictive_superiority_inferred": False,
        "unique_or_causal_latent_recovery_inferred": False,
    }

    analysis_root.mkdir(parents=True)
    prediction_summary.to_csv(analysis_root / "family_prediction_summary.csv", index=False)
    entity_metrics.to_csv(analysis_root / "per_entity_metrics.csv", index=False)
    family_summary.to_csv(analysis_root / "method_family_summary.csv", index=False)
    adam_diagnostics.to_csv(analysis_root / "adam_diagnostics.csv", index=False)
    gauge_summary.to_csv(analysis_root / "gauge_invariance_summary.csv", index=False)
    jacobian_summary.to_csv(analysis_root / "jacobian_summary.csv", index=False)
    basis_summary.to_csv(analysis_root / "basis_summary.csv", index=False)
    bounds_summary.to_csv(analysis_root / "stability_bounds_summary.csv", index=False)
    perturbation_summary.to_csv(analysis_root / "query_target_invariance.csv", index=False)
    geometry_summary.to_csv(analysis_root / "response_geometry_summary.csv", index=False)
    readout_summary.to_csv(analysis_root / "raw_readout_summary.csv", index=False)
    narrow_summary.to_csv(analysis_root / "narrow_support_summary.csv", index=False)
    write_json(analysis_root / "decision.json", decision)
    (analysis_root / "EXTENSION_RESULTS.md").write_text(
        _markdown(decision, family_summary, basis_summary, geometry_summary, narrow_summary), encoding="utf-8"
    )
    manifest = {
        "scope": "independent_gauge_equivariant_calibration_extension_analysis",
        "run_root": str(run_root.relative_to(PROJECT_ROOT)) if run_root.is_relative_to(PROJECT_ROOT) else str(run_root),
        "source_plan_sha256": EXPECTED_PLAN_SHA256,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "extension_runner_sha256": sha256(EXTENSION_RUNNER),
        "analyzer_sha256": sha256(Path(__file__)),
        "files": {},
    }
    for path in sorted(analysis_root.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    write_json(analysis_root / "manifest.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main() -> None:
    analyze_root(parse_args().run_root)


if __name__ == "__main__":
    main()
