#!/usr/bin/env python3
"""Frozen strong baselines for ThermoML vapor-pressure development curves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = PROJECT_ROOT / "THERMOML_BASELINE_PROTOCOL_20260829.md"
DATA_CSV = (
    PROJECT_ROOT
    / "runs/thermoml_vapor_pressure_development_data_20260829/development_curves.csv"
)
STRUCTURE_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_structure_development_20260829"
STRUCTURE_MANIFEST = STRUCTURE_ROOT / "manifest.json"
STRUCTURE_DECISION = STRUCTURE_ROOT / "decision.json"
STRUCTURE_PREDICTIONS = STRUCTURE_ROOT / "oof_query_predictions.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_baselines_development_20260829"
EXPECTED_PROTOCOL_SHA256 = "f27ecd820f448fa02111cb2d2a78393821e14a6bd2711ee1671f77e10009e44a"
EXPECTED_DATA_CSV_SHA256 = "9ebc8ea5a8b870cb98cc829c1700d4ebdad806c043014a0a5051ada8629411b6"
EXPECTED_STRUCTURE_MANIFEST_SHA256 = "88a17aa99a58a9c265b6b6cace7fbd89e94f1544853e2f5b4669c07b36cd2485"
EXPECTED_STRUCTURE_DECISION_SHA256 = "928ee5bc4f21c156a00a49737dca0729b8e9a52f7fb23a165a870b7727adaa26"


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


def selected_design(temperature: np.ndarray, reference: float) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(temperature)),
            1.0 / temperature - 1.0 / reference,
            np.log(temperature / reference),
        ]
    )


def antoine_prediction(support: pd.DataFrame, query_temperature: np.ndarray) -> np.ndarray:
    support_c = support["temperature_k"].to_numpy(float) - 273.15
    query_c = query_temperature - 273.15
    target = np.log10(support["pressure_kpa"].to_numpy(float))
    lower = float(-support_c.min() + 1.0e-6)
    upper = 2000.0
    grid = np.linspace(lower, upper, 257)

    def fit_at(c_value: float) -> tuple[float, np.ndarray]:
        matrix = np.column_stack([np.ones(len(support_c)), 1.0 / (support_c + c_value)])
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ coefficients
        return float(np.square(residual).sum()), coefficients

    losses = np.asarray([fit_at(value)[0] for value in grid])
    best_index = int(np.argmin(losses))
    left = grid[max(0, best_index - 1)]
    right = grid[min(len(grid) - 1, best_index + 1)]
    optimized = minimize_scalar(
        lambda value: fit_at(float(value))[0], bounds=(left, right), method="bounded"
    )
    c_value = float(optimized.x)
    _, coefficients = fit_at(c_value)
    prediction_log10 = np.column_stack(
        [np.ones(len(query_c)), 1.0 / (query_c + c_value)]
    ) @ coefficients
    return np.power(10.0, prediction_log10)


def main() -> None:
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("baseline protocol changed")
    if sha256(DATA_CSV) != EXPECTED_DATA_CSV_SHA256:
        raise RuntimeError("development data changed")
    if sha256(STRUCTURE_MANIFEST) != EXPECTED_STRUCTURE_MANIFEST_SHA256:
        raise RuntimeError("structure result manifest changed")
    if sha256(STRUCTURE_DECISION) != EXPECTED_STRUCTURE_DECISION_SHA256:
        raise RuntimeError("structure decision changed")
    decision = json.loads(STRUCTURE_DECISION.read_text(encoding="utf-8"))
    if decision["selected_family"] != "v_log":
        raise ValueError("baseline evaluator is sealed to selected v_log")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    data = pd.read_csv(DATA_CSV)
    selected = pd.read_csv(STRUCTURE_PREDICTIONS)
    selected = selected.loc[selected["family"].eq("v_log")].copy()
    selected["family"] = "selected_v_log_req"
    prediction_frames = [selected]

    rows = []
    for fold in range(5):
        training = data.loc[~data["fold"].eq(fold)]
        heldout = data.loc[data["fold"].eq(fold)]
        reference = float(training["temperature_k"].median())
        global_coefficients = np.linalg.lstsq(
            selected_design(training["temperature_k"].to_numpy(float), reference),
            np.log(training["pressure_kpa"].to_numpy(float)),
            rcond=None,
        )[0]
        for entity_id, entity in heldout.groupby("entity_id", sort=True):
            support = entity.loc[entity["role"].eq("support")].sort_values("temperature_k")
            query = entity.loc[entity["role"].eq("query")].sort_values("temperature_k")
            support_temperature = support["temperature_k"].to_numpy(float)
            support_log = np.log(support["pressure_kpa"].to_numpy(float))
            query_temperature = query["temperature_k"].to_numpy(float)
            nearest_indices = np.abs(
                query_temperature[:, None] - support_temperature[None, :]
            ).argmin(axis=1)
            methods = {
                "support_nearest_log": np.exp(support_log[nearest_indices]),
                "support_linear_log": np.exp(
                    interp1d(
                        support_temperature,
                        support_log,
                        kind="linear",
                        fill_value="extrapolate",
                    )(query_temperature)
                ),
                "support_pchip_log": np.exp(
                    PchipInterpolator(support_temperature, support_log, extrapolate=True)(
                        query_temperature
                    )
                ),
                "support_antoine": antoine_prediction(support, query_temperature),
                "no_q_global_selected_formula": np.exp(
                    selected_design(query_temperature, reference) @ global_coefficients
                ),
            }
            for family, prediction in methods.items():
                if not np.isfinite(prediction).all() or not np.all(prediction > 0.0):
                    raise ValueError(f"invalid prediction for {entity_id} {family}")
                for row, value in zip(query.itertuples(index=False), prediction, strict=True):
                    rows.append(
                        {
                            "source_row_id": row.source_row_id,
                            "fold": fold,
                            "entity_id": entity_id,
                            "doi": row.doi,
                            "temperature_k": row.temperature_k,
                            "pressure_kpa": row.pressure_kpa,
                            "log_pressure": float(np.log(row.pressure_kpa)),
                            "family": family,
                            "prediction_kpa": float(value),
                            "prediction_log_pressure": float(np.log(value)),
                        }
                    )
    prediction_frames.append(pd.DataFrame(rows))
    predictions = pd.concat(prediction_frames, ignore_index=True)

    summary_rows = []
    entity_rows = []
    for family, frame in predictions.groupby("family", sort=True):
        family_entity_rows = []
        for entity_id, entity in frame.groupby("entity_id", sort=True):
            target = entity["pressure_kpa"].to_numpy(float)
            prediction = entity["prediction_kpa"].to_numpy(float)
            target_log = entity["log_pressure"].to_numpy(float)
            prediction_log = entity["prediction_log_pressure"].to_numpy(float)
            metric = {
                "family": family,
                "entity_id": entity_id,
                "doi": entity["doi"].iloc[0],
                "physical_r2": r2(target, prediction),
                "log_r2": r2(target_log, prediction_log),
                "physical_nrmse": float(
                    np.sqrt(np.mean(np.square(target - prediction))) / target.std()
                ),
            }
            family_entity_rows.append(metric)
            entity_rows.append(metric)
        entity_metrics = pd.DataFrame(family_entity_rows)
        summary_rows.append(
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
                "median_entity_physical_nrmse": float(
                    entity_metrics["physical_nrmse"].median()
                ),
                "maximum_entity_physical_nrmse": float(
                    entity_metrics["physical_nrmse"].max()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        "median_entity_physical_nrmse", kind="stable"
    )
    per_entity = pd.DataFrame(entity_rows)
    selected_entity = per_entity.loc[
        per_entity["family"].eq("selected_v_log_req"), ["entity_id", "physical_nrmse"]
    ].rename(columns={"physical_nrmse": "selected_nrmse"})
    pchip_entity = per_entity.loc[
        per_entity["family"].eq("support_pchip_log"), ["entity_id", "physical_nrmse"]
    ].rename(columns={"physical_nrmse": "pchip_nrmse"})
    ratios = selected_entity.merge(pchip_entity, on="entity_id", validate="one_to_one")
    ratios["selected_to_pchip_nrmse_ratio"] = ratios["selected_nrmse"] / ratios["pchip_nrmse"]
    maximum_ratio = float(ratios["selected_to_pchip_nrmse_ratio"].max())
    result = {
        "scope": "ThermoML development-only strong baseline comparison",
        "selected_expression_endpoint_remains_passed": True,
        "maximum_entity_selected_to_pchip_nrmse_ratio": maximum_ratio,
        "legacy_ten_times_pchip_diagnostic_passed": bool(maximum_ratio <= 10.0),
        "predictive_superiority_inferred": False,
        "confirmation_targets_opened": False,
        "family_summary": summary.to_dict(orient="records"),
    }

    predictions.to_csv(OUTPUT_ROOT / "oof_query_predictions.csv", index=False)
    per_entity.to_csv(OUTPUT_ROOT / "per_entity_metrics.csv", index=False)
    ratios.to_csv(OUTPUT_ROOT / "selected_to_pchip_entity_ratios.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "family_summary.csv", index=False)
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "data_csv_sha256": EXPECTED_DATA_CSV_SHA256,
        "structure_manifest_sha256": EXPECTED_STRUCTURE_MANIFEST_SHA256,
        "structure_decision_sha256": EXPECTED_STRUCTURE_DECISION_SHA256,
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
