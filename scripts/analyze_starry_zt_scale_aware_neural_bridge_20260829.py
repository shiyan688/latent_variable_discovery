#!/usr/bin/env python3
"""Aggregate the frozen 15-cell scale-aware Starry ZT bridge repair."""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT / "runs/starry_zt_scale_aware_neural_bridge_20260829"
ANALYSIS_ROOT = ROOT / "analysis"
PLAN = PROJECT_ROOT / "STARRY_ZT_SCALE_AWARE_NEURAL_BRIDGE_PLAN_20260829.md"
RUNNER = PROJECT_ROOT / "scripts/run_starry_zt_scale_aware_neural_bridge_20260829.py"
EXPECTED_PLAN_SHA256 = "82655e3a6e6e68776aae9442e9eb47bd7bc99e5e2c677941654cad8055ca1a95"
EXPECTED_RUNNER_SHA256 = "ef5f0bff2324b84942332beedf4c255412fa59e3dcedd8c48715613d56eef702"
FAMILIES = (
    "raw_decoder",
    "raw_q_ridge_req",
    "functional_degree1",
    "functional_degree2",
    "functional_degree3",
    "functional_degree4",
    "structure_req",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def standardized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array - array.mean(axis=0)) / array.std(axis=0)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)).statistic)
    if not np.isfinite(value):
        raise ValueError("non-finite stability correlation")
    return value


def entity_bootstrap_interval(frame: pd.DataFrame, seed: int = 20260829) -> list[float]:
    rows = []
    for _, entity in frame.groupby("label", sort=True):
        target = entity["target"].to_numpy(float)
        prediction = entity["prediction"].to_numpy(float)
        rows.append(
            (
                len(entity),
                float(target.sum()),
                float(np.square(target).sum()),
                float(np.square(target - prediction).sum()),
            )
        )
    statistics = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(statistics), size=(10_000, len(statistics)))
    totals = statistics[sampled].sum(axis=1)
    denominator = totals[:, 2] - np.square(totals[:, 1]) / totals[:, 0]
    values = 1.0 - totals[:, 3] / denominator
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("frozen plan hash changed")
    if sha256(RUNNER) != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("formal runner hash changed")
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=False)

    cell_summaries = []
    prediction_frames = []
    coordinate_frames = []
    input_hashes = {}
    for fold in range(5):
        for seed in range(3):
            cell_root = ROOT / f"fold{fold}_seed{seed}"
            manifest_path = cell_root / "manifest.json"
            summary_path = cell_root / "cell_summary.json"
            prediction_path = cell_root / "query_predictions.csv"
            coordinate_path = cell_root / "entity_coordinates.csv"
            split_path = cell_root / "support_query_split.csv"
            for path in (manifest_path, summary_path, prediction_path, coordinate_path, split_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
                input_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
            manifest = read_json(manifest_path)
            summary = read_json(summary_path)
            if manifest["plan_sha256"] != EXPECTED_PLAN_SHA256:
                raise ValueError(f"plan mismatch in {cell_root}")
            if manifest["runner_sha256"] != EXPECTED_RUNNER_SHA256:
                raise ValueError(f"runner mismatch in {cell_root}")
            if not manifest["scientific_selection_eligible"] or manifest["temporal_confirmation_opened"]:
                raise ValueError(f"invalid evidence boundary in {cell_root}")
            if summary["status"] != "success" or not summary["scientific_selection_eligible"]:
                raise ValueError(f"incomplete cell {cell_root}")
            cell_summaries.append(summary)
            predictions = pd.read_csv(prediction_path)
            coordinates = pd.read_csv(coordinate_path)
            if set(predictions["family"].unique()) != set(FAMILIES):
                raise ValueError(f"family mismatch in {cell_root}")
            predictions["fold"] = fold
            predictions["seed"] = seed
            coordinate_frames.append(coordinates)
            prediction_frames.append(predictions)

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_coordinates = pd.concat(coordinate_frames, ignore_index=True)
    if not np.isfinite(all_predictions[["target", "prediction"]].to_numpy()).all():
        raise ValueError("non-finite query prediction")
    truth_counts = all_predictions.groupby(["source_row_id", "family"])["target"].nunique()
    if int(truth_counts.max()) != 1:
        raise ValueError("query target differs across seeds")

    median_predictions = (
        all_predictions.groupby(
            ["source_row_id", "label", "temperature", "target", "family"],
            as_index=False,
        )["prediction"]
        .median()
        .sort_values(["family", "label", "temperature"], kind="stable")
    )
    family_rows = []
    per_entity_rows = []
    for family, frame in median_predictions.groupby("family", sort=True):
        bootstrap_interval = entity_bootstrap_interval(frame)
        family_rows.append(
            {
                "family": family,
                "pooled_r2": float(r2_score(frame["target"], frame["prediction"])),
                "physical_rmse": float(mean_squared_error(frame["target"], frame["prediction"]) ** 0.5),
                "entity_bootstrap_r2_low": bootstrap_interval[0],
                "entity_bootstrap_r2_high": bootstrap_interval[1],
            }
        )
        for label, entity in frame.groupby("label", sort=True):
            target = entity["target"].to_numpy(float)
            prediction = entity["prediction"].to_numpy(float)
            scale = float(np.std(target))
            per_entity_rows.append(
                {
                    "family": family,
                    "label": label,
                    "r2": float(r2_score(target, prediction)),
                    "reference_nrmse": float(mean_squared_error(target, prediction) ** 0.5 / scale),
                }
            )
    family_summary = pd.DataFrame(family_rows)
    per_entity = pd.DataFrame(per_entity_rows)

    seed_rows = []
    for seed in range(3):
        for family, frame in all_predictions.loc[all_predictions["seed"].eq(seed)].groupby("family", sort=True):
            seed_rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "pooled_r2": float(r2_score(frame["target"], frame["prediction"])),
                    "physical_rmse": float(mean_squared_error(frame["target"], frame["prediction"]) ** 0.5),
                }
            )
    seed_summary = pd.DataFrame(seed_rows)

    stability_rows = []
    raw_columns = [f"raw_q{index}" for index in range(4)]
    functional_columns = [f"functional_q{index}" for index in range(3)]
    for fold in range(5):
        fold_coordinates = all_coordinates.loc[all_coordinates["fold"].eq(fold)]
        for left_seed, right_seed in combinations(range(3), 2):
            left = fold_coordinates.loc[fold_coordinates["seed"].eq(left_seed)].sort_values("label")
            right = fold_coordinates.loc[fold_coordinates["seed"].eq(right_seed)].sort_values("label")
            if not left["label"].to_numpy().tolist() == right["label"].to_numpy().tolist():
                raise ValueError("coordinate entity mismatch")
            raw_left = standardized(left[raw_columns].to_numpy(float))
            raw_right = standardized(right[raw_columns].to_numpy(float))
            functional_left = standardized(left[functional_columns].to_numpy(float))
            functional_right = standardized(right[functional_columns].to_numpy(float))
            stability_rows.append(
                {
                    "fold": fold,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "raw_unaligned_coordinate_spearman": finite_spearman(raw_left, raw_right),
                    "functional_named_coordinate_spearman": finite_spearman(functional_left, functional_right),
                    "raw_distance_geometry_spearman": finite_spearman(pdist(raw_left), pdist(raw_right)),
                    "functional_distance_geometry_spearman": finite_spearman(
                        pdist(functional_left), pdist(functional_right)
                    ),
                }
            )
    stability = pd.DataFrame(stability_rows)

    pivot = per_entity.pivot(index="label", columns="family", values="reference_nrmse")
    maximum_functional_to_structure_ratio = float(
        (pivot["functional_degree2"] / pivot["structure_req"]).max()
    )
    response_degree2 = [
        float(summary["decoder_response_projection_r2"]["2"])
        for summary in cell_summaries
    ]
    query_target_input_max_difference = max(
        float(summary["query_target_input_max_difference"])
        for summary in cell_summaries
    )
    r2_by_family = family_summary.set_index("family")["pooled_r2"].to_dict()
    stability_gain = float(
        stability["functional_distance_geometry_spearman"].median()
        - stability["raw_distance_geometry_spearman"].median()
    )
    named_stability_gain = float(
        stability["functional_named_coordinate_spearman"].median()
        - stability["raw_unaligned_coordinate_spearman"].median()
    )
    degree2_entity = per_entity.loc[per_entity["family"].eq("functional_degree2")]
    gates = {
        "all_15_cells_complete": len(cell_summaries) == 15,
        "exact_query_target_invariance": query_target_input_max_difference == 0.0,
        "functional_degree2_physical_r2_at_least_0_85": r2_by_family["functional_degree2"] >= 0.85,
        "functional_degree2_decoder_fidelity_at_least_0_95": min(response_degree2) >= 0.95,
        "structure_req_physical_r2_at_least_0_85": r2_by_family["structure_req"] >= 0.85,
        "no_functional_entity_above_ten_times_structure_nrmse": maximum_functional_to_structure_ratio <= 10.0,
        "functional_geometry_more_stable_than_raw_geometry": stability_gain > 0.0,
    }
    scale_repair_gate_names = tuple(
        name for name in gates if name != "functional_geometry_more_stable_than_raw_geometry"
    )
    decision = {
        "scope": "Starry ZT development scale-aware neural-to-canonical q bridge repair",
        "cells": len(cell_summaries),
        "entities": int(median_predictions["label"].nunique()),
        "query_rows": int(
            median_predictions.loc[median_predictions["family"].eq("structure_req")].shape[0]
        ),
        "family_summary": family_rows,
        "minimum_degree2_decoder_response_r2": min(response_degree2),
        "median_degree2_decoder_response_r2": float(np.median(response_degree2)),
        "maximum_functional_to_structure_entity_nrmse_ratio": maximum_functional_to_structure_ratio,
        "query_target_input_max_difference": query_target_input_max_difference,
        "median_raw_distance_geometry_spearman": float(stability["raw_distance_geometry_spearman"].median()),
        "median_functional_distance_geometry_spearman": float(
            stability["functional_distance_geometry_spearman"].median()
        ),
        "functional_minus_raw_geometry_stability": stability_gain,
        "median_raw_unaligned_coordinate_spearman": float(
            stability["raw_unaligned_coordinate_spearman"].median()
        ),
        "median_functional_named_coordinate_spearman": float(
            stability["functional_named_coordinate_spearman"].median()
        ),
        "functional_minus_raw_named_coordinate_stability": named_stability_gain,
        "functional_degree2_median_entity_r2": float(degree2_entity["r2"].median()),
        "functional_degree2_entities_r2_at_least_0_85": int(
            degree2_entity["r2"].ge(0.85).sum()
        ),
        "functional_degree2_entities_above_ten_times_structure_nrmse": int(
            (pivot["functional_degree2"] / pivot["structure_req"]).gt(10.0).sum()
        ),
        "gates": gates,
        "scale_aware_tail_repair_supported": all(gates[name] for name in scale_repair_gate_names),
        "full_neural_to_canonical_bridge_supported": all(gates.values()),
        "neural_to_canonical_bridge_supported": all(gates.values()),
        "temporal_confirmation_reused": False,
        "predictive_superiority_inferred": False,
    }

    all_predictions.to_csv(ANALYSIS_ROOT / "all_seed_predictions.csv", index=False)
    median_predictions.to_csv(ANALYSIS_ROOT / "median_query_predictions.csv", index=False)
    all_coordinates.to_csv(ANALYSIS_ROOT / "all_entity_coordinates.csv", index=False)
    family_summary.to_csv(ANALYSIS_ROOT / "family_summary.csv", index=False)
    seed_summary.to_csv(ANALYSIS_ROOT / "seed_summary.csv", index=False)
    per_entity.to_csv(ANALYSIS_ROOT / "per_entity_metrics.csv", index=False)
    stability.to_csv(ANALYSIS_ROOT / "cross_seed_stability.csv", index=False)
    write_json(ANALYSIS_ROOT / "decision.json", decision)
    write_json(
        ANALYSIS_ROOT / "manifest.json",
        {
            "plan_sha256": EXPECTED_PLAN_SHA256,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "analyzer_sha256": sha256(Path(__file__)),
            "inputs": input_hashes,
        },
    )
    print(json.dumps(decision))


if __name__ == "__main__":
    main()
