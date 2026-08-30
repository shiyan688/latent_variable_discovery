#!/usr/bin/env python3
"""Independently aggregate the 25 frozen crystal-Cp neural/GIRD cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lvs.core.metrics import (
    effective_rank,
    local_distance_distortion,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_neural_gird_development_20260829"
ANALYSIS_ROOT = ROOT / "analysis"
RUNNER = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_neural_gird_cell_20260829.py"
LAUNCHER = PROJECT_ROOT / "scripts/launch_thermoml_crystal_cp_neural_gird_4gpu_20260829.py"
ADAPTER = PROJECT_ROOT / "runs/thermoml_crystal_cp_neural_gird_setup_20260829/sealed_v4_basis_adapter.json"
DATA = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
EXPECTED_RUNNER_SHA256 = "d296d75b56dc5857cba8c4c53d25ed610d57e7cb91f0622154ce2e44a3c7d723"
EXPECTED_ADAPTER_SHA256 = "0b402d6c6b7fe7b474e11046ec0c158e12bf1f5def5695d4bc0658119a30ee80"
EXPECTED_DATA_SHA256 = "f73d3c676932304c8e5c21e79e7bc9c678e20c84db8d60b59a8e60feee400e4e"
EXPECTED_CLARIFICATION_SHA256 = "7e04a98a50f381da296e96c93ebc1c717dd80b12687b35b53695ef2985343aff"
EXPECTED_V4_AMENDMENT_SHA256 = "07194d41108d177405d63682135dd9f1bbf2e419d7d72894dfcf81d4ee4920ae"
EXPECTED_V4_DECISION_SHA256 = "09c9cb17cf64cc5e6f5bc3b9958ed07ed7fda56245b545961c4312647503e0e8"
EXPECTED_V4_PACKAGE_SHA256 = "cc52ecd255fa30955474a6ff370cfb66d43208f6ed3e0b6b4dac3c6f117e4c61"
EXPECTED_V4_FOLDS_SHA256 = "84032cb98a5d619a2aad8b50dda012f6c8919da6c2f739215657b4187c6ae7fb"
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
REGIMES = ("spread", "prefix", "four_support")
OFFICIAL_VARIANTS = {"spread": "spread_offset0", "prefix": "prefix", "four_support": "four_support"}
LAMBDA_METHODS = (
    "gird_lambda_0", "gird_lambda_0.0001", "gird_lambda_0.001", "gird_lambda_0.01",
    "gird_lambda_0.1", "gird_lambda_1", "gird_lambda_10", "gird_lambda_100", "gird_lambda_inf",
)
ALWAYS_METHODS = (
    "raw_decoder_adam", "decoder_functional", "support_structure_re_q",
    "direct_target_dictionary", *LAMBDA_METHODS,
)
REQUIRED_CELL_FILES = (
    "manifest.json", "cell_summary.json", "training_history.csv", "checkpoint.pt",
    "query_predictions.csv", "support_query_split.csv", "query_target_perturbation.csv",
    "adam_calibration.csv", "decoder_probes.csv", "canonical_coordinates.csv",
    "gird_strata.csv", "gird_coefficients.csv", "lambda_selection.json",
    "dictionary_selection.json", "dictionary_paths.csv", "dictionary_inputs.csv",
    "gauge_diagnostics.csv", "gauge_dictionary_selection.csv", "gauge_fixed_lambda_diagnostics.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    return float(1.0 - np.square(target - prediction).sum() / denominator)


def prediction_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entity_rows = []
    method_rows = []
    for (regime, method), frame in predictions.groupby(["regime", "method"], sort=True):
        local_entities = []
        for entity_id, entity in frame.groupby("entity_id", sort=True):
            target = entity["cp_j_per_mol_k"].to_numpy(float)
            prediction = entity["prediction_cp_j_per_mol_k"].to_numpy(float)
            error = prediction - target
            scale = float(np.std(target, ddof=0))
            row = {
                "regime": regime,
                "method": method,
                "entity_id": entity_id,
                "doi": str(entity["doi"].iloc[0]),
                "physical_r2": r2(target, prediction),
                "physical_nrmse": float(np.sqrt(np.mean(error**2)) / scale) if scale > 0.0 else None,
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "maximum_absolute_error": float(np.max(np.abs(error))),
                "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
            }
            entity_rows.append(row)
            local_entities.append(row)
        entity_frame = pd.DataFrame(local_entities)
        target = frame["cp_j_per_mol_k"].to_numpy(float)
        prediction = frame["prediction_cp_j_per_mol_k"].to_numpy(float)
        method_rows.append(
            {
                "regime": regime,
                "method": method,
                "query_rows": len(frame),
                "entities": frame["entity_id"].nunique(),
                "dois": frame["doi"].nunique(),
                "pooled_physical_r2": r2(target, prediction),
                "median_entity_nrmse": float(entity_frame["physical_nrmse"].median()),
                "p95_entity_nrmse": float(entity_frame["physical_nrmse"].quantile(0.95)),
                "maximum_entity_nrmse": float(entity_frame["physical_nrmse"].max()),
                "median_entity_r2": float(entity_frame["physical_r2"].median()),
                "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
            }
        )
    return pd.DataFrame(method_rows), pd.DataFrame(entity_rows)


def pointwise_median(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "source_row_id", "entity_id", "doi", "regime", "position", "temperature_k",
        "cp_j_per_mol_k", "method",
    ]
    counts = predictions.groupby(keys, dropna=False)["seed"].nunique()
    require(int(counts.min()) == len(SEEDS) and int(counts.max()) == len(SEEDS), "method/query row lacks five seeds")
    return (
        predictions.groupby(keys, as_index=False, dropna=False)["prediction_cp_j_per_mol_k"]
        .median()
        .sort_values(["regime", "method", "entity_id", "position"], kind="stable")
    )


def distance_correlation(left: np.ndarray, right: np.ndarray) -> float:
    def centered(values: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
        return distances - distances.mean(axis=0) - distances.mean(axis=1)[:, None] + distances.mean()

    a = centered(np.asarray(left, dtype=float))
    b = centered(np.asarray(right, dtype=float))
    covariance = float(np.mean(a * b))
    denominator = math.sqrt(max(float(np.mean(a * a) * np.mean(b * b)), 0.0))
    return float(math.sqrt(max(covariance / denominator, 0.0))) if denominator > 0.0 else 0.0


def representation_diagnostics(reference: np.ndarray, learned: np.ndarray) -> dict[str, float]:
    curve = neighborhood_preservation_curve(reference, learned, max_k=10)
    local = local_distance_distortion(reference, learned, k=min(5, len(reference) - 1))
    distance = pairwise_distance_metrics(reference, learned)
    return {
        "continuity_auc": float(np.mean([row["continuity"] for row in curve])),
        "trustworthiness_auc": float(np.mean([row["trustworthiness"] for row in curve])),
        "knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve])),
        "distance_spearman": distance["distance_spearman"],
        "distance_correlation": distance_correlation(reference, learned),
        "effective_rank": effective_rank(learned),
        **local,
    }


def _distance_vector(values: np.ndarray) -> np.ndarray:
    matrix = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    return matrix[np.triu_indices(len(values), 1)]


def basis_design(terms: list[dict[str, Any]], temperature: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    columns = []
    for term in terms:
        kind = term["kind"]
        if kind == "constant":
            column = np.ones_like(values)
        elif kind == "scaled_temperature_power":
            column = (values / float(term["scale"])) ** int(term["power"])
        elif kind == "inverse_scaled_temperature_power":
            column = (float(term["scale"]) / values) ** int(term["power"])
        elif kind == "normalized_u_power":
            column = ((values - t_min) / (t_max - t_min)) ** int(term["power"])
        elif kind == "upper_boundary_atom":
            u = (values - t_min) / (t_max - t_min)
            remainder = float(term["delta"]) + 1.0 - u
            if term["atom"] == "inverse_sqrt":
                column = 1.0 / np.sqrt(remainder)
            elif term["atom"] == "log":
                column = -np.log(remainder)
            elif term["atom"] == "inverse":
                column = 1.0 / remainder
            else:
                column = 1.0 / remainder**2
        else:
            raise ValueError(f"unknown basis term {kind}")
        columns.append(column)
    return np.column_stack(columns)


def projection_fidelity(adapter: dict[str, Any], coordinates: pd.DataFrame, probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, coefficient_rows in coordinates.loc[coordinates["method"].eq("decoder_functional")].groupby(
        ["fold", "seed", "entity_id", "regime"], sort=True
    ):
        fold, seed, entity_id, regime = key
        probe = probes.loc[
            probes["fold"].eq(fold)
            & probes["seed"].eq(seed)
            & probes["entity_id"].eq(entity_id)
            & probes["support_variant"].eq(OFFICIAL_VARIANTS[regime])
            & probes["method"].eq("adam")
        ].sort_values("probe_index", kind="stable")
        require(len(probe) == 41, "decoder projection probe coverage changed")
        basis_id = str(coefficient_rows["basis_id"].iloc[0])
        specification = adapter["folds"][str(int(fold))]
        branches = {specification["background"]["basis_id"]: specification["background"], specification["transition"]["basis_id"]: specification["transition"]}
        require(basis_id in branches, "canonical coordinate basis is not sealed")
        branch = branches[basis_id]
        coefficients = coefficient_rows.set_index("coordinate_name")["coordinate_value"]
        ordered = np.asarray([coefficients[term["name"]] for term in branch["terms"]], dtype=float)
        temperature = probe["temperature_k"].to_numpy(float)
        target = probe["decoder_cp_j_per_mol_k"].to_numpy(float)
        prediction = basis_design(branch["terms"], temperature, temperature.min(), temperature.max()) @ ordered
        rows.append(
            {
                "fold": fold, "seed": seed, "entity_id": entity_id, "regime": regime,
                "basis_id": basis_id, "projection_physical_r2": r2(target, prediction),
                "projection_rmse": float(np.sqrt(np.mean(np.square(target - prediction)))),
            }
        )
    return pd.DataFrame(rows)


def omp_margin_certificates(inputs: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, regime), group in inputs.loc[
        inputs["source"].eq("decoder_gn") & inputs["gauge_id"].eq(-1)
    ].groupby(["fold", "regime"], sort=True):
        matrices = []
        for (_, support_variant), current in group.groupby(["seed", "support_variant"], sort=True):
            ordered = current.sort_values(["entity_id", "probe_index"], kind="stable")
            matrices.append((int(current["seed"].iloc[0]), str(support_variant), ordered[["entity_id", "probe_index"]], ordered["response"].to_numpy(float)))
        epsilon = 0.0
        for left, right in combinations(matrices, 2):
            require(left[2].reset_index(drop=True).equals(right[2].reset_index(drop=True)), "OMP response vectors are not aligned")
            epsilon = max(epsilon, float(np.linalg.norm(left[3] - right[3])))
        selected = paths.loc[
            paths["fold"].eq(fold)
            & paths["source"].eq(f"decoder_gn:{regime}")
            & paths["inner_fold"].eq(-1)
            & paths["winner"].eq(True)
        ]
        for row in selected.itertuples():
            bound = 4.0 * float(row.residual_frobenius) * epsilon + 2.0 * epsilon**2
            rows.append(
                {
                    "fold": int(fold), "seed": int(row.seed), "regime": regime, "stage": int(row.stage),
                    "selected_atom": row.candidate_name, "epsilon": epsilon,
                    "residual_frobenius": float(row.residual_frobenius), "score_margin": float(row.score_margin),
                    "sufficient_bound": bound, "certificate_status": "PASS" if float(row.score_margin) > bound else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def four_support_decision(
    summaries: list[dict[str, Any]], strata: pd.DataFrame, predictions: pd.DataFrame, entity_metrics: pd.DataFrame,
    gauge_passed: bool, leakage_passed: bool, dictionary_passed: bool,
) -> dict[str, Any]:
    minimum_votes = min(int(summary["four_support_contributing_lambda_folds"]) for summary in summaries)
    statuses = {summary["four_support_conditional_status"] for summary in summaries}
    consensus = strata.loc[strata["regime"].eq("four_support")].groupby(["entity_id", "doi"])["stratum"].agg(["nunique", "first"])
    prior = consensus.loc[consensus["nunique"].eq(1) & consensus["first"].eq("prior_eligible")]
    evidence_available = minimum_votes >= 3 and statuses == {"READY_FOR_ANALYSIS"} and len(prior) >= 20 and prior.reset_index()["doi"].nunique() >= 5
    if not evidence_available:
        return {
            "status": "NOT_TESTED", "minimum_contributing_folds": minimum_votes,
            "prior_eligible_entities": int(len(prior)), "prior_eligible_dois": int(prior.reset_index()["doi"].nunique()),
        }
    ids = set(prior.reset_index()["entity_id"])
    current = entity_metrics.loc[entity_metrics["regime"].eq("four_support") & entity_metrics["entity_id"].isin(ids)]
    medians = current.groupby("method")["physical_nrmse"].median().to_dict()
    required = {"conditional_gird", "gird_lambda_0", "direct_target_dictionary"}
    require(required <= set(medians), "conditional comparison methods are incomplete")
    support_ids = set(
        consensus.loc[consensus["nunique"].eq(1) & consensus["first"].eq("support_identified")]
        .reset_index()["entity_id"]
    )
    conditional_points = predictions.loc[
        predictions["regime"].eq("four_support") & predictions["method"].eq("conditional_gird")
        & predictions["entity_id"].isin(support_ids),
        ["source_row_id", "prediction_cp_j_per_mol_k"],
    ].rename(columns={"prediction_cp_j_per_mol_k": "conditional"})
    zero_points = predictions.loc[
        predictions["regime"].eq("four_support") & predictions["method"].eq("gird_lambda_0")
        & predictions["entity_id"].isin(support_ids),
        ["source_row_id", "prediction_cp_j_per_mol_k"],
    ].rename(columns={"prediction_cp_j_per_mol_k": "zero"})
    support_comparison = conditional_points.merge(zero_points, on="source_row_id", validate="one_to_one")
    gates = {
        "dictionary_and_certificate": dictionary_passed,
        "gauge": gauge_passed,
        "leakage": leakage_passed,
        "five_percent_gain_over_lambda0": medians["conditional_gird"] <= 0.95 * medians["gird_lambda_0"],
        "beats_direct_target_dictionary": medians["conditional_gird"] < medians["direct_target_dictionary"],
        "support_identified_equals_lambda0": (
            len(support_comparison) == len(conditional_points) == len(zero_points)
            and (support_comparison.empty or float(np.max(np.abs(support_comparison["conditional"] - support_comparison["zero"]))) == 0.0)
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": {key: bool(value) for key, value in gates.items()},
        "median_entity_nrmse": {key: float(value) for key, value in medians.items()},
    }


def analyze(root: Path = ROOT, analysis_root: Path = ANALYSIS_ROOT) -> dict[str, Any]:
    root = root.resolve()
    analysis_root = analysis_root.resolve()
    require(root == ROOT.resolve(), "aggregate analyzer is bound to the exact formal root")
    require(analysis_root == ANALYSIS_ROOT.resolve(), "aggregate analyzer is bound to the exact analysis root")
    require(not analysis_root.exists(), "analysis root must be absent")
    require(sha256(RUNNER) == EXPECTED_RUNNER_SHA256, "formal runner changed")
    require(sha256(ADAPTER) == EXPECTED_ADAPTER_SHA256, "sealed basis adapter changed")
    require(sha256(DATA) == EXPECTED_DATA_SHA256, "development data changed")
    launcher_status = json.loads((root / "launcher_status.json").read_text(encoding="utf-8"))
    require(launcher_status.get("state") == "completed_all" and len(launcher_status.get("completed", [])) == 25, "launcher is not terminal")
    require(launcher_status.get("analyzer_sha256") == sha256(Path(__file__).resolve()), "launcher did not freeze this analyzer")
    require(launcher_status.get("launcher_sha256") == sha256(LAUNCHER), "launcher hash changed")
    ledger = [json.loads(line) for line in (root / "launcher_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    require(sum(row.get("event") == "cell_started" for row in ledger) == 25, "launcher start ledger coverage changed")
    terminal_events = [row for row in ledger if row.get("event") == "cell_terminal"]
    require(len(terminal_events) == 25 and all(row["return_code"] == 0 for row in terminal_events), "launcher terminal ledger changed")
    adapter = json.loads(ADAPTER.read_text(encoding="utf-8"))

    frames: dict[str, list[pd.DataFrame]] = {
        "prediction": [], "coordinate": [], "probe": [], "adam": [], "strata": [],
        "gauge": [], "gauge_fixed": [], "dictionary_input": [], "dictionary_path": [],
    }
    summaries = []
    input_hashes = {}
    devices = set()
    leakage_passed = True
    lambda_selections = {}
    dictionary_selections = {}
    for fold in FOLDS:
        for seed in SEEDS:
            cell = root / f"fold{fold}_seed{seed}"
            for name in REQUIRED_CELL_FILES:
                require((cell / name).is_file(), f"missing cell artifact: {cell / name}")
            manifest = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((cell / "cell_summary.json").read_text(encoding="utf-8"))
            require(manifest["runner_sha256"] == EXPECTED_RUNNER_SHA256, "cell runner hash mismatch")
            require(manifest["basis_adapter_sha256"] == EXPECTED_ADAPTER_SHA256, "cell adapter hash mismatch")
            require(manifest["implementation_clarification_sha256"] == EXPECTED_CLARIFICATION_SHA256, "cell clarification mismatch")
            require(manifest["router_margin_amendment_sha256"] == EXPECTED_V4_AMENDMENT_SHA256, "cell v4 amendment mismatch")
            require(manifest["v4_analysis_decision_sha256"] == EXPECTED_V4_DECISION_SHA256, "cell v4 decision mismatch")
            require(manifest["v4_final_package_sha256"] == EXPECTED_V4_PACKAGE_SHA256, "cell v4 package mismatch")
            require(manifest["v4_fold_selections_sha256"] == EXPECTED_V4_FOLDS_SHA256, "cell v4 folds mismatch")
            require(manifest["scientific_selection_eligible"] is True and manifest["confirmation_targets_opened"] is False, "ineligible cell in formal root")
            require(summary["status"] == "success" and summary["epochs_completed"] == 1000, "cell is not terminal")
            actual_files = {
                path.name: sha256(path) for path in sorted(cell.iterdir())
                if path.is_file() and path.name not in {"manifest.json", "manifest.running.json"}
            }
            require(actual_files == manifest["files"], "cell artifact inventory/hash mismatch")
            devices.add(manifest["device"])
            summaries.append(summary)
            lambda_selections[(fold, seed)] = json.loads((cell / "lambda_selection.json").read_text(encoding="utf-8"))
            dictionary_selections[(fold, seed)] = json.loads((cell / "dictionary_selection.json").read_text(encoding="utf-8"))
            for name in REQUIRED_CELL_FILES:
                input_hashes[str((cell / name).relative_to(PROJECT_ROOT))] = sha256(cell / name)

            leakage = pd.read_csv(cell / "query_target_perturbation.csv")
            base_columns = [
                "q_max_abs_difference", "raw_prediction_max_abs_difference", "stage_ratio_abs_difference",
                "structure_coefficient_max_abs_difference", "structure_prediction_max_abs_difference",
                "fixed_grid_coefficient_max_abs_difference", "fixed_grid_prediction_max_abs_difference",
                "selected_support_loss_max_abs_difference",
            ]
            leakage_passed &= not leakage["gird_stratum_changed"].any() and float(np.abs(leakage[base_columns].to_numpy(float)).max()) == 0.0
            ready = leakage["conditional_status"].eq("READY_FOR_ANALYSIS")
            if ready.any():
                leakage_passed &= float(np.abs(leakage.loc[ready, ["gird_lambda_abs_difference", "gird_coefficient_max_abs_difference", "gird_prediction_max_abs_difference"]].to_numpy(float)).max()) == 0.0

            for key, filename in (
                ("prediction", "query_predictions.csv"), ("coordinate", "canonical_coordinates.csv"),
                ("probe", "decoder_probes.csv"), ("adam", "adam_calibration.csv"),
                ("strata", "gird_strata.csv"), ("gauge", "gauge_diagnostics.csv"),
                ("gauge_fixed", "gauge_fixed_lambda_diagnostics.csv"),
                ("dictionary_input", "dictionary_inputs.csv"), ("dictionary_path", "dictionary_paths.csv"),
            ):
                frame = pd.read_csv(cell / filename)
                frame["fold"] = fold
                frame["seed"] = seed
                frames[key].append(frame)
    require(devices == {"cuda"}, f"formal cells must all be CUDA, observed {devices}")
    combined = {key: pd.concat(values, ignore_index=True) for key, values in frames.items()}

    data = pd.read_csv(DATA)
    role_columns = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}
    expected_query_ids = {
        regime: set(data.loc[data[column].eq("query"), "source_row_id"].astype(int))
        for regime, column in role_columns.items()
    }
    for seed in SEEDS:
        for regime in REGIMES:
            for method in ALWAYS_METHODS:
                observed = set(
                    combined["prediction"].loc[
                        combined["prediction"]["seed"].eq(seed)
                        & combined["prediction"]["regime"].eq(regime)
                        & combined["prediction"]["method"].eq(method),
                        "source_row_id",
                    ].astype(int)
                )
                require(observed == expected_query_ids[regime], f"query coverage mismatch for seed={seed} {regime} {method}")

    fixed_predictions = combined["prediction"].loc[combined["prediction"]["method"].isin(ALWAYS_METHODS)]
    require(np.isfinite(fixed_predictions[["cp_j_per_mol_k", "prediction_cp_j_per_mol_k"]].to_numpy(float)).all(), "non-finite fixed prediction")
    median = pointwise_median(fixed_predictions)
    optional = combined["prediction"].loc[~combined["prediction"]["method"].isin(ALWAYS_METHODS)]
    for method in sorted(set(optional["method"])):
        method_frame = optional.loc[optional["method"].eq(method)]
        expected_regimes = ("spread", "prefix") if method == "selected_gird" else ("four_support",)
        complete = all(
            set(method_frame.loc[method_frame["seed"].eq(seed) & method_frame["regime"].eq(regime), "source_row_id"].astype(int))
            == expected_query_ids[regime]
            for seed in SEEDS
            for regime in expected_regimes
        )
        if complete:
            median = pd.concat([median, pointwise_median(method_frame)], ignore_index=True)
    for source, alias in (("gird_lambda_0", "support_omp"), ("gird_lambda_inf", "decoder_only_dictionary")):
        aliased = median.loc[median["method"].eq(source)].copy()
        aliased["method"] = alias
        median = pd.concat([median, aliased], ignore_index=True)
    method_summary, entity_metrics = prediction_metrics(median)

    fidelity = projection_fidelity(adapter, combined["coordinate"], combined["probe"])
    functional_primary = method_summary.loc[
        method_summary["regime"].eq("spread") & method_summary["method"].eq("decoder_functional")
    ].iloc[0]

    continuity_rows = []
    geometry_rows = []
    offset_rows = []
    function_vectors: dict[tuple[int, int, str], tuple[list[str], np.ndarray]] = {}
    q_vectors: dict[tuple[int, int, str], tuple[list[str], np.ndarray]] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            for regime, variant in OFFICIAL_VARIANTS.items():
                adam = combined["adam"].loc[
                    combined["adam"]["fold"].eq(fold) & combined["adam"]["seed"].eq(seed)
                    & combined["adam"]["cohort"].eq("outer_test") & combined["adam"]["support_variant"].eq(variant)
                ].sort_values("entity_id", kind="stable")
                entities = adam["entity_id"].astype(str).tolist()
                q = adam[[f"q{index}" for index in range(4)]].to_numpy(float)
                functions = []
                for entity in entities:
                    current = combined["prediction"].loc[
                        combined["prediction"]["fold"].eq(fold) & combined["prediction"]["seed"].eq(seed)
                        & combined["prediction"]["regime"].eq(regime) & combined["prediction"]["method"].eq("decoder_functional")
                        & combined["prediction"]["entity_id"].astype(str).eq(entity)
                    ].sort_values("temperature_k", kind="stable")
                    u = (current["temperature_k"] - current["temperature_k"].min()) / (current["temperature_k"].max() - current["temperature_k"].min())
                    functions.append(np.interp(np.linspace(0.0, 1.0, 41), u, current["prediction_cp_j_per_mol_k"]))
                function = np.asarray(functions)
                q_vectors[(fold, seed, regime)] = (entities, q)
                function_vectors[(fold, seed, regime)] = (entities, function)
                continuity_rows.append({"fold": fold, "seed": seed, "regime": regime, **representation_diagnostics(function, q)})
            offset_q = {}
            offset_response = {}
            for offset in range(4):
                variant = f"spread_offset{offset}"
                q_frame = combined["adam"].loc[
                    combined["adam"]["fold"].eq(fold) & combined["adam"]["seed"].eq(seed)
                    & combined["adam"]["cohort"].eq("outer_test") & combined["adam"]["support_variant"].eq(variant)
                ].sort_values("entity_id", kind="stable")
                offset_q[offset] = q_frame[[f"q{index}" for index in range(4)]].to_numpy(float)
                response_frame = combined["probe"].loc[
                    combined["probe"]["fold"].eq(fold) & combined["probe"]["seed"].eq(seed)
                    & combined["probe"]["cohort"].eq("outer_test") & combined["probe"]["support_variant"].eq(variant)
                    & combined["probe"]["method"].eq("stable_gn")
                ].sort_values(["entity_id", "probe_index"], kind="stable")
                offset_response[offset] = response_frame.pivot(
                    index="entity_id", columns="probe_index", values="decoder_cp_j_per_mol_k"
                ).sort_index().to_numpy(float)
            for left_offset, right_offset in combinations(range(4), 2):
                scale = np.maximum(offset_response[left_offset].std(axis=1), 1e-12)
                response_nrmse = np.sqrt(
                    np.mean(np.square(offset_response[left_offset] - offset_response[right_offset]), axis=1)
                ) / scale
                offset_rows.append(
                    {
                        "fold": fold, "seed": seed, "left_offset": left_offset, "right_offset": right_offset,
                        "raw_q_distance_spearman": pairwise_distance_metrics(
                            offset_q[left_offset], offset_q[right_offset]
                        )["distance_spearman"],
                        "median_decoder_response_nrmse": float(np.median(response_nrmse)),
                        "maximum_decoder_response_nrmse": float(np.max(response_nrmse)),
                    }
                )
    for fold in FOLDS:
        for regime in REGIMES:
            for left_seed, right_seed in combinations(SEEDS, 2):
                left_entities, left_q = q_vectors[(fold, left_seed, regime)]
                right_entities, right_q = q_vectors[(fold, right_seed, regime)]
                left_function_entities, left_function = function_vectors[(fold, left_seed, regime)]
                right_function_entities, right_function = function_vectors[(fold, right_seed, regime)]
                require(left_entities == right_entities == left_function_entities == right_function_entities, "geometry entity alignment changed")
                geometry_rows.extend(
                    {
                        "fold": fold, "regime": regime, "left_seed": left_seed, "right_seed": right_seed,
                        "representation": name,
                        "distance_spearman": float(spearmanr(_distance_vector(left), _distance_vector(right)).statistic),
                    }
                    for name, left, right in (("raw_q", left_q, right_q), ("function", left_function, right_function))
                )
    continuity = pd.DataFrame(continuity_rows)
    geometry = pd.DataFrame(geometry_rows)
    offset_stability = pd.DataFrame(offset_rows)
    geometry_medians = geometry.groupby("representation")["distance_spearman"].median().to_dict()
    bridge_gates = {
        "decoder_functional_pooled_r2_at_least_0_85": float(functional_primary["pooled_physical_r2"]) >= 0.85,
        "minimum_projection_r2_at_least_0_95": float(fidelity["projection_physical_r2"].min()) >= 0.95,
        "finite_functional_coefficients": bool(np.isfinite(combined["coordinate"].loc[combined["coordinate"]["method"].eq("decoder_functional"), "coordinate_value"].to_numpy(float)).all()),
        "function_geometry_at_least_raw_q": geometry_medians["function"] >= geometry_medians["raw_q"],
    }

    stable_gauge = combined["gauge"].loc[combined["gauge"]["method"].eq("stable_gn")]
    gauge_fixed = combined["gauge_fixed"]
    require(set(stable_gauge["gauge_id"].astype(int)) == set(range(8)), "eight-gauge coverage changed")
    require(
        gauge_fixed.groupby(["fold", "seed", "gauge_id", "entity_id", "regime"])["lambda"].nunique().eq(9).all(),
        "fixed-lambda gauge coverage changed",
    )
    ready_gauge = stable_gauge["conditional_status"].eq("READY_FOR_ANALYSIS")
    gauge_passed = (
        stable_gauge["identical_dictionary"].astype(bool).all()
        and stable_gauge["gird_stratum_equal"].astype(bool).all()
        and float(stable_gauge["prediction_max_abs_difference"].max()) <= 1e-6
        and float(stable_gauge["functional_coefficient_max_abs_difference"].max()) <= 1e-6
        and float(gauge_fixed["prediction_max_abs_difference"].max()) <= 1e-6
        and (
            not ready_gauge.any()
            or (
                stable_gauge.loc[ready_gauge, "gird_lambda_equal"].astype(bool).all()
                and float(stable_gauge.loc[ready_gauge, "gird_prediction_max_abs_difference"].max()) <= 1e-6
            )
        )
    )
    gauge_summary = pd.DataFrame(
        [{
            "identical_dictionary_all": bool(stable_gauge["identical_dictionary"].astype(bool).all()),
            "maximum_decoder_prediction_difference": float(stable_gauge["prediction_max_abs_difference"].max()),
            "maximum_functional_coefficient_difference": float(stable_gauge["functional_coefficient_max_abs_difference"].max()),
            "maximum_fixed_lambda_prediction_difference": float(gauge_fixed["prediction_max_abs_difference"].max()),
            "passed": bool(gauge_passed),
        }]
    )

    require(
        combined["dictionary_path"]["certificate_status"].eq("PENDING_AGGREGATE").all(),
        "cell-level OMP certificate status changed",
    )
    certificates = omp_margin_certificates(combined["dictionary_input"], combined["dictionary_path"])
    require(len(certificates) > 0, "OMP certificates were not recomputed")
    certificate_passed = certificates["certificate_status"].eq("PASS").all()
    recurrence_passed = True
    recurrence_rows = []
    for seed in SEEDS:
        motifs = [tuple(dictionary_selections[(fold, seed)]["decoder_gn:spread"]["atoms"]) for fold in FOLDS]
        counts = pd.Series(motifs).value_counts()
        recurrence = int(counts.max())
        recurrence_rows.append({"seed": seed, "maximum_fold_recurrence": recurrence, "passed": recurrence >= 4})
        recurrence_passed &= recurrence >= 4
    paths_selected = combined["dictionary_path"].loc[
        combined["dictionary_path"]["source"].eq("decoder_gn:spread")
        & combined["dictionary_path"]["inner_fold"].eq(-1)
        & combined["dictionary_path"]["winner"].eq(True)
    ]
    dictionary_sizes_valid = all(
        2 <= len(selection["decoder_gn:spread"]["atoms"]) <= 5
        for selection in dictionary_selections.values()
    )
    dictionary_passed = bool(
        recurrence_passed
        and certificate_passed
        and dictionary_sizes_valid
        and paths_selected["gram_condition"].le(1e4).all()
        and paths_selected["path_status"].eq("OK").all()
    )

    spread_metrics = entity_metrics.loc[entity_metrics["regime"].eq("spread")]
    spread_medians = spread_metrics.groupby("method")["physical_nrmse"].median().to_dict()
    best_interpretable = min(spread_medians["support_structure_re_q"], spread_medians["direct_target_dictionary"])
    primary_gird_gate = spread_medians.get("selected_gird", math.inf) <= 1.05 * best_interpretable
    conditional = four_support_decision(
        summaries, combined["strata"], median, entity_metrics, gauge_passed, leakage_passed, dictionary_passed
    )
    gird_summary = {
        "dictionary_passed": dictionary_passed,
        "omp_margin_certificates_all_passed": bool(certificate_passed),
        "dictionary_recurrence_passed": bool(recurrence_passed),
        "gauge_passed": bool(gauge_passed),
        "leakage_passed": bool(leakage_passed),
        "primary_spread_no_more_than_five_percent_worse": bool(primary_gird_gate),
        "conditional_four_support": conditional,
    }

    decision = {
        "scope": "independent 25-cell ThermoML crystal-Cp neural/GIRD aggregate",
        "cells_verified": 25,
        "learned_bridge_status": "PASS" if all(bridge_gates.values()) else "FAIL",
        "learned_bridge_gates": {key: bool(value) for key, value in bridge_gates.items()},
        "gird_status": (
            "NOT_TESTED"
            if conditional["status"] == "NOT_TESTED"
            else conditional["status"]
            if dictionary_passed and gauge_passed and leakage_passed and primary_gird_gate
            else "FAIL"
        ),
        "gird": gird_summary,
        "confirmation_targets_opened": False,
    }
    analysis_root.mkdir(parents=True)
    median.to_csv(analysis_root / "five_seed_pointwise_median_predictions.csv", index=False)
    method_summary.to_csv(analysis_root / "method_summary.csv", index=False)
    entity_metrics.to_csv(analysis_root / "entity_metrics.csv", index=False)
    fidelity.to_csv(analysis_root / "decoder_projection_fidelity.csv", index=False)
    continuity.to_csv(analysis_root / "q_function_continuity.csv", index=False)
    geometry.to_csv(analysis_root / "cross_seed_geometry_stability.csv", index=False)
    offset_stability.to_csv(analysis_root / "support_offset_stability.csv", index=False)
    gauge_summary.to_csv(analysis_root / "gauge_summary.csv", index=False)
    certificates.to_csv(analysis_root / "omp_margin_certificates.csv", index=False)
    pd.DataFrame(recurrence_rows).to_csv(analysis_root / "dictionary_recurrence.csv", index=False)
    write_json(analysis_root / "gird_decision.json", gird_summary)
    write_json(analysis_root / "decision.json", decision)
    files = {
        path.name: sha256(path) for path in sorted(analysis_root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "scope": decision["scope"],
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "files": files,
        "confirmation_targets_opened": False,
    }
    write_json(analysis_root / "manifest.json", manifest)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.root / "analysis"), sort_keys=True))


if __name__ == "__main__":
    main()
