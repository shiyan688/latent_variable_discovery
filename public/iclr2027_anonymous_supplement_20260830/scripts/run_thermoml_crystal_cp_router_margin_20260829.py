#!/usr/bin/env python3
"""Run the frozen v4 maximum-margin ThermoML crystal-Cp router audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_thermoml_crystal_cp_transition_structure_20260829 as v3


AMENDMENT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_ROUTER_MARGIN_AMENDMENT_20260829.md"
V3_RUNNER_PATH = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_transition_structure_20260829.py"
V3_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_transition_structure_development_20260829"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_router_margin_development_20260829"

EXPECTED_AMENDMENT_SHA256 = "07194d41108d177405d63682135dd9f1bbf2e419d7d72894dfcf81d4ee4920ae"
EXPECTED_V3_RUNNER_SHA256 = "df48cb6674de34949f54cb768c7ff85dd03e606ffd3a28e7d578ca24609426e1"
EXPECTED_V3_RESULT_SHA256 = "aec86dba8c9dc33f3942b473f08f8b2bf9479e84dcef496b1fe688af7e08ebaa"
EXPECTED_V3_DECISION_SHA256 = "8aa9ec409020d681bcbdceecd43e6fa39de337a687f7799f5070f761db817764"
EXPECTED_V3_ANALYSIS_DECISION_SHA256 = "1d32cb2ed88509af08b21e81acf845f6811064d9fcf9bd4381c1bd337c8b5d13"
EXPECTED_V3_ANALYSIS_MANIFEST_SHA256 = "25d82c1c66735646db1ceacf55845598090310f516e841898f85e419569643db"
EXPECTED_FOLD_GAMMAS = (200, 100, 100, 100, 50)
EXPECTED_FOLD_DEGREES = (1, 2, 2, 2, 2)
EXPECTED_FOLD_ATOM = "inverse_sqrt"
EXPECTED_OOF_R2 = 0.8593003362
EXPECTED_NEGATIVE_PREDICTIONS = 748
EXPECTED_FINAL = (100, 2, "inverse_sqrt", 0.0003)
REPRODUCTION_ATOL = 5e-10
MARGIN_TIE_ATOL = 1e-12


def sha256(path: Path) -> str:
    return v3.sha256(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_bindings() -> None:
    for path, expected in (
        (AMENDMENT_PATH, EXPECTED_AMENDMENT_SHA256),
        (V3_RUNNER_PATH, EXPECTED_V3_RUNNER_SHA256),
        (V3_ROOT / "result.json", EXPECTED_V3_RESULT_SHA256),
        (V3_ROOT / "decision.json", EXPECTED_V3_DECISION_SHA256),
        (V3_ROOT / "analysis/decision.json", EXPECTED_V3_ANALYSIS_DECISION_SHA256),
        (V3_ROOT / "analysis/manifest.json", EXPECTED_V3_ANALYSIS_MANIFEST_SHA256),
    ):
        _require(path.is_file(), f"required frozen v4 input is missing: {path}")
        _require(sha256(path) == expected, f"frozen v4 binding mismatch: {path}")
    v3._check_paths()


def margin_fields(rows: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    by_entity = {str(row["entity_id"]): float(row["stage_ratio"]) for row in rows}
    ordered = sorted(by_entity.items())
    ratios = np.asarray([ratio for _, ratio in ordered], float)
    lower = ratios[ratios <= gamma]
    upper = ratios[ratios > gamma]
    routing = ",".join(f"{entity_id}:{int(ratio > gamma)}" for entity_id, ratio in ordered)
    return {
        "nearest_lower_training_ratio": float(lower.max()) if len(lower) else np.nan,
        "nearest_upper_training_ratio": float(upper.min()) if len(upper) else np.nan,
        "log_margin": float(np.min(np.abs(np.log1p(ratios) - np.log1p(gamma)))),
        "routing_vector_sha256": hashlib.sha256(routing.encode()).hexdigest(),
    }


def select_max_margin(
    rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    scope: str,
    fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, candidates = v3._select(rows, base_rows, scope, fold)
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_candidate.setdefault(row["candidate_id"], []).append(row)
    for candidate in candidates:
        candidate.update(margin_fields(rows_by_candidate[candidate["candidate_id"]], float(candidate["gamma"])))
    tied = [candidate for candidate in candidates if candidate["one_percent_tie"]]
    first_four = min(_first_four_key(row) for row in tied)
    finalists = [row for row in tied if _first_four_key(row) == first_four]
    maximum_margin = max(row["log_margin"] for row in finalists)
    margin_tied = [row for row in finalists if abs(row["log_margin"] - maximum_margin) <= MARGIN_TIE_ATOL]
    selected = max(margin_tied, key=lambda row: row["gamma"])
    for candidate in candidates:
        candidate["selected"] = candidate["candidate_id"] == selected["candidate_id"]
        candidate["tie_rule"] = "within_1_percent_then_fewer_routed_then_smaller_degree_then_atom_order_then_larger_delta_then_max_log_margin_then_larger_gamma"
    return selected, candidates


def _first_four_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["routed_entity_count"],
        row["degree"],
        v3.ATOM_ORDER[row["atom"]],
        -row["delta"],
    )


def _perturbation_audit(data: pd.DataFrame, infos: dict[str, dict[str, Any]], selections: dict[int, dict[str, Any]], output_root: Path) -> pd.DataFrame:
    records = []
    for fold in v3.FOLDS:
        copied = data.copy()
        mask = copied["fold"].eq(fold) & copied["spread_role"].eq("query")
        copied.loc[mask, "cp_j_per_mol_k"] += 1_000_000.0
        copy_path = output_root / f"query_target_perturbed_fold_{fold}.csv"
        copied.to_csv(copy_path, index=False)
        rerun_infos = v3._load_infos(v3.load_development_curves(copy_path))
        train_ids = [entity_id for entity_id, info in rerun_infos.items() if info["fold"] != fold]
        rows, bases = v3._candidate_rows(rerun_infos, train_ids, "perturbation_selection", fold)
        perturbed_selection, _ = select_max_margin(rows, bases, "perturbation_selection", fold)
        _require(perturbed_selection["candidate_id"] == selections[fold]["candidate_id"], f"perturbed selection changed in fold {fold}")
        for entity_id, original in infos.items():
            if original["fold"] != fold:
                continue
            perturbed = rerun_infos[entity_id]
            old = v3._selected_fit(original, selections[fold])
            new = v3._selected_fit(perturbed, perturbed_selection)
            records.append({
                "fold": fold, "entity_id": entity_id,
                "candidate_id_original": selections[fold]["candidate_id"],
                "candidate_id_perturbed": perturbed_selection["candidate_id"],
                "candidate_equal": True,
                "routed_original": old["routed"], "routed_perturbed": new["routed"],
                "stage_ratio_original": original["stage_ratio"], "stage_ratio_perturbed": perturbed["stage_ratio"],
                "stage_ratio_abs_difference": abs(original["stage_ratio"] - perturbed["stage_ratio"]),
                "coefficient_max_abs_difference": float(np.max(np.abs(old["fit"]["coefficients"] - new["fit"]["coefficients"]))),
                "prediction_max_abs_difference": float(np.max(np.abs(old["fit"]["prediction"] - new["fit"]["prediction"]))),
                "query_target_perturbation": 1_000_000.0,
                "query_targets_used_for_fit": False,
            })
    frame = pd.DataFrame(records)
    frame.to_csv(output_root / "query_target_perturbation.csv", index=False)
    return frame


def _final_package(infos: dict[str, dict[str, Any]], output_root: Path) -> dict[str, Any]:
    rows, bases = v3._candidate_rows(infos, sorted(infos), "all_development", -1)
    selection, candidates = select_max_margin(rows, bases, "all_development", -1)
    points, coefficients = [], []
    for info in infos.values():
        selected = v3._selected_fit(info, selection)
        coefficients.append(v3._coefficient_row(info, selection, "transition_selected", "final_package", -1))
        for source_row_id, position, temperature, target, prediction in zip(info["query_source_row_id"], info["query_position"], info["query_temperature"], info["query_target"], selected["fit"]["prediction"], strict=True):
            points.append({"scope": "final_package", "entity_id": info["entity_id"], "doi": info["doi"], "fold": info["fold"], "candidate_id": selection["candidate_id"], "routed": selected["routed"], "stage_ratio": info["stage_ratio"], "source_row_id": int(source_row_id), "position": int(position), "temperature_k": float(temperature), "cp_j_per_mol_k": float(target), "prediction_cp_j_per_mol_k": float(prediction)})
    package = {
        "scope": "ThermoML crystal-Cp v4 all-development maximum-margin package",
        "selected_candidate_id": selection["candidate_id"],
        "selected_gamma": selection["gamma"], "selected_degree": selection["degree"],
        "selected_atom": selection["atom"], "selected_delta": selection["delta"],
        "nearest_lower_training_ratio": selection["nearest_lower_training_ratio"],
        "nearest_upper_training_ratio": selection["nearest_upper_training_ratio"],
        "log_margin": selection["log_margin"], "routing_vector_sha256": selection["routing_vector_sha256"],
        "selected_candidate_metrics": selection, "confirmation_targets_opened": False,
    }
    pd.DataFrame(candidates).to_csv(output_root / "final_package_candidate_metrics.csv", index=False)
    pd.DataFrame(points).to_csv(output_root / "final_package_point_predictions.csv", index=False)
    pd.DataFrame(coefficients).to_csv(output_root / "final_package_coefficients.csv", index=False)
    (output_root / "final_package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    return package


def _verify_declared(selections: dict[int, dict[str, Any]], package: dict[str, Any], pooled_r2: float, negative_count: int) -> None:
    _require(tuple(int(selections[fold]["gamma"]) for fold in v3.FOLDS) == EXPECTED_FOLD_GAMMAS, "v4 fold gamma reproduction mismatch")
    _require(tuple(int(selections[fold]["degree"]) for fold in v3.FOLDS) == EXPECTED_FOLD_DEGREES, "v4 fold degree reproduction mismatch")
    _require(all(selections[fold]["atom"] == EXPECTED_FOLD_ATOM for fold in v3.FOLDS), "v4 fold atom reproduction mismatch")
    _require(abs(pooled_r2 - EXPECTED_OOF_R2) <= REPRODUCTION_ATOL, "v4 OOF R2 reproduction mismatch")
    _require(negative_count == EXPECTED_NEGATIVE_PREDICTIONS, "v4 negative-prediction reproduction mismatch")
    observed_final = (int(package["selected_gamma"]), int(package["selected_degree"]), str(package["selected_atom"]), float(package["selected_delta"]))
    _require(observed_final == EXPECTED_FINAL, "v4 all-development package reproduction mismatch")


def run_experiment(data_path: str | Path = v3.DATA_PATH, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    data_path, output_root = Path(data_path).resolve(), Path(output_root).resolve()
    _require(not output_root.exists(), f"formal v4 output root must be absent: {output_root}")
    _require(data_path == v3.DATA_PATH.resolve(), "v4 runner is bound to sealed development data")
    started = time.perf_counter()
    _check_bindings()
    data = v3.load_development_curves(data_path)
    v2_binding = v3.bind_v2_baseline(data)
    output_root.mkdir(parents=True, exist_ok=False)
    infos = v3._load_infos(data)
    selections, candidate_rows = {}, []
    for fold in v3.FOLDS:
        train_ids = [entity_id for entity_id, info in infos.items() if info["fold"] != fold]
        rows, bases = v3._candidate_rows(infos, train_ids, "outer_training", fold)
        selections[fold], candidates = select_max_margin(rows, bases, "outer_training", fold)
        candidate_rows.extend(candidates)
    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame.to_csv(output_root / "candidate_metrics.csv", index=False)
    pd.DataFrame(list(selections.values())).to_csv(output_root / "fold_selections.csv", index=False)
    points, metrics, coefficients = v3._emit_predictions(infos, selections, output_root)
    perturbation = _perturbation_audit(data, infos, selections, output_root)
    package = _final_package(infos, output_root)
    transition_points = points.loc[points["method"].eq("transition_selected")]
    target = transition_points["cp_j_per_mol_k"].to_numpy(float)
    prediction = transition_points["prediction_cp_j_per_mol_k"].to_numpy(float)
    pooled_r2 = float(1.0 - np.square(target - prediction).sum() / np.square(target - target.mean()).sum())
    negative_count = int(np.count_nonzero(prediction < 0.0))
    _verify_declared(selections, package, pooled_r2, negative_count)
    finite_coefficients = all(np.isfinite([row[f"coefficient_{index}"] for index in range(int(row["terms"]))]).all() for _, row in coefficients.iterrows())
    decision = v3._write_decision(metrics, candidate_frame, selections, perturbation, pooled_r2, len(transition_points), bool(finite_coefficients and np.isfinite(prediction).all()), output_root)
    decision["scope"] = "ThermoML crystal-Cp v4 maximum-margin OOF development decision"
    decision["v4_declared_reproduction_passed"] = True
    (output_root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "success", "scope": "ThermoML crystal-Cp v4 maximum-margin development OOF",
        "entities": v3.EXPECTED_ENTITIES, "dois": v3.EXPECTED_DOIS, "rows": v3.EXPECTED_ROWS,
        "oof_query_rows": v3.EXPECTED_OOF_QUERY_ROWS, "candidate_grid_size": len(v3.GAMMA_GRID) * len(v3.DEGREE_GRID) * len(v3.ATOM_GRID) * len(v3.DELTA_GRID),
        "v2_binding": v2_binding, "selected_fold_candidates": list(selections.values()),
        "final_package_candidate_id": package["selected_candidate_id"],
        "oof_transition_pooled_physical_r2": pooled_r2, "negative_prediction_count": negative_count,
        "decision_passed": decision["passed"], "v4_declared_reproduction_passed": True,
        "query_targets_used_for_fit": False, "confirmation_targets_opened": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in sorted(output_root.iterdir()) if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}}
    manifest = {
        "scope": result["scope"], "router_margin_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "v3_runner_sha256": EXPECTED_V3_RUNNER_SHA256, "v3_result_sha256": EXPECTED_V3_RESULT_SHA256,
        "v3_decision_sha256": EXPECTED_V3_DECISION_SHA256, "v3_analysis_decision_sha256": EXPECTED_V3_ANALYSIS_DECISION_SHA256,
        "v3_analysis_manifest_sha256": EXPECTED_V3_ANALYSIS_MANIFEST_SHA256,
        "data_csv_sha256": v3.EXPECTED_DATA_CSV_SHA256, "runner_sha256": sha256(Path(__file__).resolve()),
        "files": files, "confirmation_targets_opened": False, "python": sys.version, "platform": platform.platform(),
    }
    (output_root / "artifact_hashes.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=v3.DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
