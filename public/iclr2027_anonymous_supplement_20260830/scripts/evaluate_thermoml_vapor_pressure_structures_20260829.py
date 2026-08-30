#!/usr/bin/env python3
"""Development-only evaluation of frozen ThermoML vapor-pressure structures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN = PROJECT_ROOT / "THERMOML_VAPOR_PRESSURE_PLAN_20260829.md"
AMENDMENT = PROJECT_ROOT / "THERMOML_EXPRESSION_SUCCESS_AMENDMENT_20260829.md"
DATA_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_development_data_20260829"
DATA_MANIFEST = DATA_ROOT / "manifest.json"
DATA_CSV = DATA_ROOT / "development_curves.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_structure_development_20260829"
EXPECTED_PLAN_SHA256 = "8793f712b6a32aa514906ffb13ae7169d0de8556f9bda342b1202d94b0bb2deb"
EXPECTED_AMENDMENT_SHA256 = "18e15a3bad4e5fc00464e1c4829062bbc5e7fed477d96781d1629b6884526faf"
EXPECTED_DATA_MANIFEST_SHA256 = "f69a3afff8e658a658e06ee5f2966e32e9f86d082966a4df2b22548ee476ac86"
EXPECTED_DATA_CSV_SHA256 = "9ebc8ea5a8b870cb98cc829c1700d4ebdad806c043014a0a5051ada8629411b6"
CORRECTIONS = ("v_log", "v_T", "v_inv2")
FAMILIES = ("coarse_cc",) + CORRECTIONS
R_GAS = 8.31446261815324


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.square(target - prediction).sum())
    total = float(np.square(target - target.mean()).sum())
    if total == 0.0:
        raise ValueError("R2 target variance is zero")
    return 1.0 - residual / total


def design(temperature: np.ndarray, reference: float, family: str) -> np.ndarray:
    inverse_offset = 1.0 / temperature - 1.0 / reference
    columns = [np.ones(len(temperature)), inverse_offset]
    if family == "v_log":
        columns.append(np.log(temperature / reference))
    elif family == "v_T":
        columns.append((temperature - reference) / reference)
    elif family == "v_inv2":
        columns.append(np.square(inverse_offset))
    elif family != "coarse_cc":
        raise ValueError(f"unknown family: {family}")
    return np.column_stack(columns)


def fit_support(
    support: pd.DataFrame, reference: float, family: str
) -> np.ndarray:
    matrix = design(support["temperature_k"].to_numpy(float), reference, family)
    target = np.log(support["pressure_kpa"].to_numpy(float))
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def group_bootstrap_interval(
    frame: pd.DataFrame, group_column: str, seed: int
) -> list[float]:
    rows = []
    for _, group in frame.groupby(group_column, sort=True):
        target = group["pressure_kpa"].to_numpy(float)
        prediction = group["prediction_kpa"].to_numpy(float)
        rows.append(
            (
                len(group),
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
        raise RuntimeError("frozen ThermoML plan hash changed")
    if sha256(AMENDMENT) != EXPECTED_AMENDMENT_SHA256:
        raise RuntimeError("expression-success amendment hash changed")
    if sha256(DATA_MANIFEST) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("development materialization manifest changed")
    if sha256(DATA_CSV) != EXPECTED_DATA_CSV_SHA256:
        raise RuntimeError("development curve data changed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    data = pd.read_csv(DATA_CSV)
    if data["entity_id"].nunique() != 282 or data["doi"].nunique() != 142:
        raise ValueError("development identity counts changed")
    if set(data["fold"].unique()) != set(range(5)):
        raise ValueError("development folds changed")
    if data.groupby("doi")["fold"].nunique().max() != 1:
        raise ValueError("DOI crosses development folds")
    if not data["pressure_kpa"].gt(0).all():
        raise ValueError("nonpositive development pressure")

    prediction_rows = []
    coefficient_rows = []
    query_target_input_max_difference = 0.0
    for fold in range(5):
        training = data.loc[~data["fold"].eq(fold)]
        heldout = data.loc[data["fold"].eq(fold)]
        reference = float(training["temperature_k"].median())
        for entity_id, entity in heldout.groupby("entity_id", sort=True):
            support = entity.loc[entity["role"].eq("support")].sort_values("temperature_k")
            query = entity.loc[entity["role"].eq("query")].sort_values("temperature_k")
            for family in FAMILIES:
                coefficients = fit_support(support, reference, family)
                log_prediction = design(
                    query["temperature_k"].to_numpy(float), reference, family
                ) @ coefficients
                prediction = np.exp(log_prediction)
                if not np.isfinite(prediction).all() or not np.all(prediction > 0.0):
                    raise ValueError(f"non-finite or nonpositive prediction for {entity_id} {family}")

                perturbed = entity.copy()
                perturbed.loc[perturbed["role"].eq("query"), "pressure_kpa"] *= 1.0e6
                perturbed_support = perturbed.loc[perturbed["role"].eq("support")].sort_values(
                    "temperature_k"
                )
                perturbed_coefficients = fit_support(perturbed_support, reference, family)
                perturbed_prediction = design(
                    query["temperature_k"].to_numpy(float), reference, family
                ) @ perturbed_coefficients
                query_target_input_max_difference = max(
                    query_target_input_max_difference,
                    float(np.max(np.abs(coefficients - perturbed_coefficients))),
                    float(np.max(np.abs(log_prediction - perturbed_prediction))),
                )

                padded = np.full(3, np.nan)
                padded[: len(coefficients)] = coefficients
                coefficient_rows.append(
                    {
                        "fold": fold,
                        "entity_id": entity_id,
                        "doi": entity["doi"].iloc[0],
                        "common_name": entity["common_name"].iloc[0],
                        "formula": entity["formula"].iloc[0],
                        "family": family,
                        "temperature_reference_k": reference,
                        "support_rows": len(support),
                        "query_rows": len(query),
                        "q0_log_pressure_at_reference": padded[0],
                        "q1_inverse_temperature_k": padded[1],
                        "effective_enthalpy_j_mol": -R_GAS * padded[1],
                        "q2_correction": padded[2],
                    }
                )
                for row, prediction_log, prediction_kpa in zip(
                    query.itertuples(index=False), log_prediction, prediction, strict=True
                ):
                    prediction_rows.append(
                        {
                            "source_row_id": row.source_row_id,
                            "fold": fold,
                            "entity_id": entity_id,
                            "doi": row.doi,
                            "temperature_k": row.temperature_k,
                            "pressure_kpa": row.pressure_kpa,
                            "log_pressure": float(np.log(row.pressure_kpa)),
                            "family": family,
                            "prediction_kpa": float(prediction_kpa),
                            "prediction_log_pressure": float(prediction_log),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    if len(predictions) != 7_240 * len(FAMILIES):
        raise ValueError("query prediction count changed")
    if query_target_input_max_difference != 0.0:
        raise ValueError("query target changed support-derived result")

    family_rows = []
    entity_rows = []
    for family, frame in predictions.groupby("family", sort=True):
        for entity_id, entity in frame.groupby("entity_id", sort=True):
            target = entity["pressure_kpa"].to_numpy(float)
            prediction = entity["prediction_kpa"].to_numpy(float)
            target_log = entity["log_pressure"].to_numpy(float)
            prediction_log = entity["prediction_log_pressure"].to_numpy(float)
            scale = float(target.std())
            if scale == 0.0:
                raise ValueError(f"zero entity pressure scale: {entity_id}")
            entity_rows.append(
                {
                    "family": family,
                    "entity_id": entity_id,
                    "doi": entity["doi"].iloc[0],
                    "physical_r2": r2(target, prediction),
                    "log_r2": r2(target_log, prediction_log),
                    "physical_nrmse": float(np.sqrt(np.mean(np.square(target - prediction))) / scale),
                }
            )
        entity_metrics = pd.DataFrame(entity_rows).loc[lambda x: x["family"].eq(family)]
        family_rows.append(
            {
                "family": family,
                "pooled_physical_r2": r2(
                    frame["pressure_kpa"].to_numpy(float),
                    frame["prediction_kpa"].to_numpy(float),
                ),
                "pooled_log_r2": r2(
                    frame["log_pressure"].to_numpy(float),
                    frame["prediction_log_pressure"].to_numpy(float),
                ),
                "median_entity_physical_r2": float(entity_metrics["physical_r2"].median()),
                "fraction_entities_physical_r2_at_least_0_85": float(
                    np.mean(entity_metrics["physical_r2"] >= 0.85)
                ),
                "median_entity_physical_nrmse": float(entity_metrics["physical_nrmse"].median()),
                "maximum_entity_physical_nrmse": float(entity_metrics["physical_nrmse"].max()),
                "entity_bootstrap_physical_r2_95_interval": group_bootstrap_interval(
                    frame, "entity_id", 20260829
                ),
                "doi_bootstrap_physical_r2_95_interval": group_bootstrap_interval(
                    frame, "doi", 20260830
                ),
            }
        )

    family_summary = pd.DataFrame(family_rows)
    candidates = family_summary.loc[family_summary["family"].isin(CORRECTIONS)].copy()
    best_nrmse = float(candidates["median_entity_physical_nrmse"].min())
    tied = set(
        candidates.loc[
            candidates["median_entity_physical_nrmse"] <= best_nrmse * 1.01, "family"
        ]
    )
    selected_family = next(family for family in CORRECTIONS if family in tied)
    selected = family_summary.loc[family_summary["family"].eq(selected_family)].iloc[0]
    interpretation = {
        "v_log": "Clausius--Clapeyron plus a logarithmic-temperature curvature term consistent with an integrated heat-capacity correction",
        "v_T": "Clausius--Clapeyron plus a local first-order temperature correction",
        "v_inv2": "Clausius--Clapeyron plus reciprocal-temperature curvature",
    }[selected_family]
    expression_passed = bool(
        selected["pooled_physical_r2"] >= 0.85
        and query_target_input_max_difference == 0.0
    )
    decision = {
        "scope": "ThermoML development-only DOI-grouped unseen-entity support-query structure selection",
        "selected_family": selected_family,
        "selected_expression": f"ln(P/1 kPa)=q0+q1*(1/T-1/T_ref)+q2*{selected_family}",
        "stage_wise_interpretation": interpretation,
        "selection_metric": "lowest median entity physical NRMSE; within 1% use frozen order v_log, v_T, v_inv2",
        "selected_pooled_physical_r2": float(selected["pooled_physical_r2"]),
        "selected_pooled_log_r2": float(selected["pooled_log_r2"]),
        "selected_median_entity_physical_r2": float(selected["median_entity_physical_r2"]),
        "selected_fraction_entities_physical_r2_at_least_0_85": float(
            selected["fraction_entities_physical_r2_at_least_0_85"]
        ),
        "query_target_input_max_difference": query_target_input_max_difference,
        "expression_endpoint_passed_on_development": expression_passed,
        "robustness_battery_complete": False,
        "pending_robustness_items": ["support PCHIP tail comparison", "support Antoine baseline"],
        "confirmation_targets_opened": False,
        "predictive_superiority_inferred": False,
        "family_summary": family_summary.to_dict(orient="records"),
    }

    predictions.to_csv(OUTPUT_ROOT / "oof_query_predictions.csv", index=False)
    coefficients.to_csv(OUTPUT_ROOT / "oof_expression_coefficients.csv", index=False)
    pd.DataFrame(entity_rows).to_csv(OUTPUT_ROOT / "per_entity_metrics.csv", index=False)
    family_summary.to_csv(OUTPUT_ROOT / "family_summary.csv", index=False)
    (OUTPUT_ROOT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "development_curves_sha256": EXPECTED_DATA_CSV_SHA256,
        "evaluator_sha256": sha256(Path(__file__)),
        "confirmation_targets_opened": False,
        "files": {
            path.name: sha256(path)
            for path in sorted(OUTPUT_ROOT.iterdir())
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
