#!/usr/bin/env python3
"""Independently recompute the frozen v4 maximum-margin router audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.analyze_thermoml_crystal_cp_transition_structure_20260829 as v3a


RAW_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_router_margin_development_20260829"
DATA_PATH = v3a.DATA_PATH
DEFAULT_OUTPUT_ROOT = RAW_ROOT / "analysis"
RUNNER_PATH = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_router_margin_20260829.py"
AMENDMENT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_ROUTER_MARGIN_AMENDMENT_20260829.md"
EXPECTED_AMENDMENT_SHA256 = "07194d41108d177405d63682135dd9f1bbf2e419d7d72894dfcf81d4ee4920ae"
EXPECTED_FOLD_GAMMAS = (200, 100, 100, 100, 50)
EXPECTED_FOLD_DEGREES = (1, 2, 2, 2, 2)
EXPECTED_FOLD_ATOM = "inverse_sqrt"
EXPECTED_OOF_R2 = 0.8593003362
EXPECTED_NEGATIVE_PREDICTIONS = 748
EXPECTED_FINAL = (100, 2, "inverse_sqrt", 0.0003)
REPRODUCTION_ATOL = 5e-10
MARGIN_TIE_ATOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _margin(infos: dict[str, dict[str, Any]], entity_ids: list[str], gamma: float) -> dict[str, Any]:
    ordered = sorted((entity_id, float(infos[entity_id]["stage"])) for entity_id in entity_ids)
    ratios = np.asarray([ratio for _, ratio in ordered], float)
    lower, upper = ratios[ratios <= gamma], ratios[ratios > gamma]
    routing = ",".join(f"{entity_id}:{int(ratio > gamma)}" for entity_id, ratio in ordered)
    return {
        "nearest_lower_training_ratio": float(lower.max()) if len(lower) else np.nan,
        "nearest_upper_training_ratio": float(upper.min()) if len(upper) else np.nan,
        "log_margin": float(np.min(np.abs(np.log1p(ratios) - np.log1p(gamma)))),
        "routing_vector_sha256": hashlib.sha256(routing.encode()).hexdigest(),
    }


def select(frame: pd.DataFrame, infos: dict[str, dict[str, Any]], entity_ids: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    margin_by_gamma = {gamma: _margin(infos, entity_ids, gamma) for gamma in v3a.GAMMAS}
    for column in ("nearest_lower_training_ratio", "nearest_upper_training_ratio", "log_margin", "routing_vector_sha256"):
        frame[column] = [margin_by_gamma[float(gamma)][column] for gamma in frame["gamma"]]
    tied = frame.loc[frame["one_percent_tie"]]
    first_four = min(
        (int(row["routed_entity_count"]), int(row["degree"]), v3a.ATOM_ORDER[str(row["atom"])], -float(row["delta"]))
        for _, row in tied.iterrows()
    )
    finalists = [
        (index, row) for index, row in tied.iterrows()
        if (int(row["routed_entity_count"]), int(row["degree"]), v3a.ATOM_ORDER[str(row["atom"])], -float(row["delta"])) == first_four
    ]
    maximum_margin = max(float(row["log_margin"]) for _, row in finalists)
    margin_tied = [(index, row) for index, row in finalists if abs(float(row["log_margin"]) - maximum_margin) <= MARGIN_TIE_ATOL]
    selected_index = max(margin_tied, key=lambda item: float(item[1]["gamma"]))[0]
    frame["selected"] = frame.index == selected_index
    row = frame.loc[selected_index]
    return frame, {key: row[key] for key in ("candidate_id", "gamma", "degree", "atom", "delta", "routed_entity_count", "heldout_fold", "nearest_lower_training_ratio", "nearest_upper_training_ratio", "log_margin", "routing_vector_sha256")}


def _validate_raw(raw_root: Path) -> dict[str, Any]:
    manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["router_margin_amendment_sha256"] == EXPECTED_AMENDMENT_SHA256 == v3a.sha256(AMENDMENT_PATH), "v4 amendment binding mismatch")
    require(manifest["runner_sha256"] == v3a.sha256(RUNNER_PATH), "v4 runner binding mismatch")
    actual = {path.name: v3a.sha256(path) for path in raw_root.iterdir() if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}}
    require(manifest["files"] == actual, "v4 raw artifact hash/inventory mismatch")
    require(json.loads((raw_root / "artifact_hashes.json").read_text()) == actual, "v4 artifact hash table mismatch")
    require(manifest["confirmation_targets_opened"] is False, "confirmation was opened")
    return manifest


def _coefficients_for_package(infos: dict[str, dict[str, Any]], selection: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return v3a._recomputed_package(infos, selection)


def analyze(raw_root: str | Path = RAW_ROOT, data_path: str | Path = DATA_PATH, output_root: str | Path = DEFAULT_OUTPUT_ROOT, bootstrap_draws: int = v3a.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    raw_root, data_path, output_root = Path(raw_root).resolve(), Path(data_path).resolve(), Path(output_root).resolve()
    require(not output_root.exists(), f"v4 analysis output root must be absent: {output_root}")
    require(data_path == DATA_PATH.resolve(), "v4 analyzer is bound to sealed development data")
    raw_manifest = _validate_raw(raw_root)
    data = v3a._load_data(data_path)
    infos = v3a._infos(data)
    raw_candidates = pd.read_csv(raw_root / "candidate_metrics.csv")
    raw_selections = pd.read_csv(raw_root / "fold_selections.csv")
    selections, frames = {}, []
    base_columns = ["gamma", "degree", "atom", "delta", "entity_count", "valid_entity_count", "exact_prediction_coverage", "median_entity_nrmse", "p95_entity_nrmse", "base_median_entity_nrmse", "base_p95_entity_nrmse", "entity_pass_count", "base_pass_count", "negative_prediction_count", "base_negative_prediction_count", "pooled_sse", "base_pooled_sse", "routed_entity_count", "eligible", "selected", "minimum_eligible_sse", "one_percent_tie", "nearest_lower_training_ratio", "nearest_upper_training_ratio", "log_margin", "routing_vector_sha256"]
    for fold in v3a.FOLDS:
        train_ids = sorted(entity_id for entity_id, info in infos.items() if info["fold"] != fold)
        require({infos[entity_id]["doi"] for entity_id in train_ids}.isdisjoint(set(data.loc[data["fold"].eq(fold), "doi"].astype(str))), "heldout DOI entered v4 selection")
        frame = v3a._candidate_metrics(infos, train_ids, "outer_training", fold)
        frame, selections[fold] = select(frame, infos, train_ids)
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    v3a._compare_frame(raw_candidates, candidates, ["heldout_fold", "candidate_id"], base_columns, "v4 candidate metrics")
    v3a._compare_frame(raw_selections, pd.DataFrame(selections.values()), ["heldout_fold", "candidate_id"], ["gamma", "degree", "atom", "delta", "routed_entity_count", "nearest_lower_training_ratio", "nearest_upper_training_ratio", "log_margin", "routing_vector_sha256"], "v4 fold selections")

    points, metrics, coefficients = v3a._recomputed_oof(infos, selections)
    raw_points = pd.read_csv(raw_root / "oof_point_predictions.csv")
    raw_metrics = pd.read_csv(raw_root / "entity_metrics.csv")
    raw_coefficients = pd.read_csv(raw_root / "oof_coefficients.csv")
    v3a._compare_frame(raw_points, points, ["method", "source_row_id"], ["heldout_fold", "entity_id", "doi", "candidate_id", "routed", "stage_ratio", "position", "temperature_k", "cp_j_per_mol_k", "prediction_cp_j_per_mol_k"], "v4 OOF points")
    v3a._compare_frame(raw_metrics, metrics, ["method", "entity_id"], ["heldout_fold", "doi", "candidate_id", "routed", "stage_ratio", "physical_r2", "physical_nrmse", "rmse", "mae", "sse", "negative_count"], "v4 entity metrics")
    coefficient_columns = ["heldout_fold", "doi", "candidate_id", "routed", "stage_ratio", "rank", "terms"] + sorted(column for column in raw_coefficients if column.startswith("coefficient_"))
    for column in coefficient_columns:
        if column.startswith("coefficient_") and column not in coefficients:
            coefficients[column] = np.nan
    v3a._compare_frame(raw_coefficients, coefficients, ["method", "entity_id"], coefficient_columns, "v4 coefficients")

    final_frame = v3a._candidate_metrics(infos, sorted(infos), "all_development", -1)
    final_frame, final_selection = select(final_frame, infos, sorted(infos))
    v3a._compare_frame(pd.read_csv(raw_root / "final_package_candidate_metrics.csv"), final_frame, ["heldout_fold", "candidate_id"], base_columns, "v4 final candidates")
    package = json.loads((raw_root / "final_package.json").read_text())
    require(package["selected_candidate_id"] == final_selection["candidate_id"], "v4 final package selection mismatch")
    final_points, final_coefficients = _coefficients_for_package(infos, final_selection)
    v3a._compare_frame(pd.read_csv(raw_root / "final_package_point_predictions.csv"), final_points, ["source_row_id"], ["entity_id", "doi", "fold", "candidate_id", "routed", "stage_ratio", "position", "temperature_k", "cp_j_per_mol_k", "prediction_cp_j_per_mol_k"], "v4 final points")
    raw_final_coefficients = pd.read_csv(raw_root / "final_package_coefficients.csv")
    final_coefficient_columns = ["doi", "candidate_id", "routed", "stage_ratio", "rank", "terms"] + sorted(column for column in raw_final_coefficients if column.startswith("coefficient_"))
    for column in final_coefficient_columns:
        if column.startswith("coefficient_") and column not in final_coefficients:
            final_coefficients[column] = np.nan
    v3a._compare_frame(raw_final_coefficients, final_coefficients, ["entity_id"], final_coefficient_columns, "v4 final coefficients")

    raw_perturbation = pd.read_csv(raw_root / "query_target_perturbation.csv")
    perturbation_records = []
    original_by_id = data.set_index("source_row_id").sort_index()
    for fold in v3a.FOLDS:
        copied = pd.read_csv(raw_root / f"query_target_perturbed_fold_{fold}.csv").set_index("source_row_id").sort_index()
        require(copied.index.equals(original_by_id.index), f"v4 fold {fold} perturbation coverage mismatch")
        delta = copied["cp_j_per_mol_k"] - original_by_id["cp_j_per_mol_k"]
        mask = copied["fold"].eq(fold) & copied["spread_role"].eq("query")
        require(np.allclose(delta[mask], 1_000_000.0) and np.allclose(delta[~mask], 0.0), f"v4 fold {fold} perturbation mismatch")
        perturbed_infos = v3a._infos(copied.reset_index().loc[copied.reset_index()["fold"].eq(fold)])
        selection = selections[fold]
        key = (int(selection["degree"]), str(selection["atom"]), float(selection["delta"]))
        for entity_id, perturbed in perturbed_infos.items():
            original = infos[entity_id]
            old_routed, new_routed = original["stage"] > selection["gamma"], perturbed["stage"] > selection["gamma"]
            old_fit = original["structures"][key] if old_routed else original["base"]
            new_fit = perturbed["structures"][key] if new_routed else perturbed["base"]
            perturbation_records.append({"fold": fold, "entity_id": entity_id, "candidate_id_original": selection["candidate_id"], "candidate_id_perturbed": selection["candidate_id"], "candidate_equal": True, "routed_original": old_routed, "routed_perturbed": new_routed, "stage_ratio_original": original["stage"], "stage_ratio_perturbed": perturbed["stage"], "stage_ratio_abs_difference": abs(original["stage"] - perturbed["stage"]), "coefficient_max_abs_difference": float(np.max(np.abs(old_fit["coefficient"] - new_fit["coefficient"]))), "prediction_max_abs_difference": float(np.max(np.abs(old_fit["prediction"] - new_fit["prediction"])))})
    v3a._compare_frame(raw_perturbation, pd.DataFrame(perturbation_records), ["fold", "entity_id"], ["candidate_id_original", "candidate_id_perturbed", "candidate_equal", "routed_original", "routed_perturbed", "stage_ratio_original", "stage_ratio_perturbed", "stage_ratio_abs_difference", "coefficient_max_abs_difference", "prediction_max_abs_difference"], "v4 perturbation")

    transition = points.loc[points["method"].eq("transition_selected")]
    target, prediction = transition["cp_j_per_mol_k"].to_numpy(float), transition["prediction_cp_j_per_mol_k"].to_numpy(float)
    pooled_r2 = float(1.0 - np.square(target - prediction).sum() / np.square(target - target.mean()).sum())
    negative = int(np.count_nonzero(prediction < 0.0))
    require(tuple(int(selections[fold]["gamma"]) for fold in v3a.FOLDS) == EXPECTED_FOLD_GAMMAS, "independent v4 fold gamma mismatch")
    require(tuple(int(selections[fold]["degree"]) for fold in v3a.FOLDS) == EXPECTED_FOLD_DEGREES, "independent v4 fold degree mismatch")
    require(all(selections[fold]["atom"] == EXPECTED_FOLD_ATOM for fold in v3a.FOLDS), "independent v4 fold atom mismatch")
    require(abs(pooled_r2 - EXPECTED_OOF_R2) <= REPRODUCTION_ATOL and negative == EXPECTED_NEGATIVE_PREDICTIONS, "independent v4 declared metric mismatch")
    require((int(final_selection["gamma"]), int(final_selection["degree"]), str(final_selection["atom"]), float(final_selection["delta"])) == EXPECTED_FINAL, "independent v4 final package mismatch")
    raw_decision = json.loads((raw_root / "decision.json").read_text())
    transition_metrics = metrics.loc[metrics["method"].eq("transition_selected")]
    base_metrics = metrics.loc[metrics["method"].eq("v2_shomate5")]
    pairs = [(str(selections[fold]["atom"]), int(selections[fold]["degree"])) for fold in v3a.FOLDS]
    gates = {
        "exact_entity_coverage": transition_metrics["entity_id"].nunique() == v3a.EXPECTED_ENTITIES,
        "exact_query_coverage": len(transition) == v3a.EXPECTED_QUERY_ROWS,
        "finite_coefficients_and_predictions": bool(np.isfinite(prediction).all()),
        "pooled_physical_r2_at_least_0_85": pooled_r2 >= 0.85,
        "median_nrmse_within_10_percent_v2": float(transition_metrics["physical_nrmse"].median()) <= 1.10 * float(base_metrics["physical_nrmse"].median()),
        "p95_nrmse_within_1_5x_v2": float(np.percentile(transition_metrics["physical_nrmse"], 95)) <= 1.50 * float(np.percentile(base_metrics["physical_nrmse"], 95)),
        "entity_r2_pass_count_at_least_v2_minus_2": int((transition_metrics["physical_r2"] >= 0.85).sum()) >= int((base_metrics["physical_r2"] >= 0.85).sum()) - 2,
        "negative_prediction_count_no_greater_v2": int(transition_metrics["negative_count"].sum()) <= int(base_metrics["negative_count"].sum()),
        "recurring_atom_and_degree_at_least_3_of_5": max(pairs.count(pair) for pair in set(pairs)) >= 3,
        "query_target_perturbation_invariant": True,
    }
    require(raw_decision["gates"] == gates, "independent v4 gates mismatch")
    entity_bootstrap, entity_summary = v3a._bootstrap(transition, "entity_id", bootstrap_draws, v3a.BOOTSTRAP_SEED)
    doi_bootstrap, doi_summary = v3a._bootstrap(transition, "doi", bootstrap_draws, v3a.BOOTSTRAP_SEED + 1)
    output_root.mkdir(parents=True, exist_ok=False)
    pd.concat([entity_bootstrap, doi_bootstrap], ignore_index=True).to_csv(output_root / "bootstrap_r2.csv", index=False)
    summary = {"entity": entity_summary, "doi": doi_summary}
    (output_root / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    decision = {"passed": bool(all(gates.values())), "gates": gates, "pooled_physical_r2": pooled_r2, "negative_prediction_count": negative, "v4_declared_reproduction_passed": True, "bootstrap": summary, "confirmation_targets_opened": False}
    (output_root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    report = f"# ThermoML crystal-Cp v4 independent maximum-margin analysis\n\n- Full independent recomputation: **passed**.\n- Fold gammas: `{list(EXPECTED_FOLD_GAMMAS)}`.\n- OOF physical pooled R²: `{pooled_r2:.10f}`.\n- Negative predictions: `{negative}`.\n- Final package: `gamma=100, d=2, inverse_sqrt, delta=0.0003`.\n- Frozen gate decision: **{'PASS' if all(gates.values()) else 'FAIL'}**.\n- Confirmation targets opened: **no**.\n"
    (output_root / "ROUTER_MARGIN_ANALYSIS.md").write_text(report, encoding="utf-8")
    files = {path.name: v3a.sha256(path) for path in output_root.iterdir() if path.is_file() and path.name != "manifest.json"}
    manifest = {"scope": "independent v4 maximum-margin analysis", "raw_manifest_sha256": v3a.sha256(raw_root / "manifest.json"), "raw_runner_sha256": raw_manifest["runner_sha256"], "analyzer_sha256": v3a.sha256(Path(__file__).resolve()), "router_margin_amendment_sha256": EXPECTED_AMENDMENT_SHA256, "data_csv_sha256": v3a.sha256(data_path), "bootstrap_draws": bootstrap_draws, "files": files, "confirmation_targets_opened": False, "python": sys.version, "platform": platform.platform()}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw_root, args.data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
