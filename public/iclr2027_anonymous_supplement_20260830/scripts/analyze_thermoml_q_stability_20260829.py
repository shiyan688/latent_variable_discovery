#!/usr/bin/env python3
"""Support-offset and physical-coordinate stability for ThermoML v_log q."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN = PROJECT_ROOT / "THERMOML_Q_STABILITY_PLAN_20260829.md"
DATA_CSV = (
    PROJECT_ROOT
    / "runs/thermoml_vapor_pressure_development_data_20260829/development_curves.csv"
)
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_q_stability_development_20260829"
EXPECTED_PLAN_SHA256 = "b99d58d317667be7a9bf27bb3ad5264ca380625066729f400bf649bcada2200a"
EXPECTED_DATA_SHA256 = "9ebc8ea5a8b870cb98cc829c1700d4ebdad806c043014a0a5051ada8629411b6"
R_GAS = 8.31446261815324


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design(temperature: np.ndarray, reference: float) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    return np.column_stack(
        [
            np.ones(len(values)),
            1.0 / values - 1.0 / reference,
            np.log(values / reference),
        ]
    )


def standardized(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    scale = array.std(axis=0)
    if np.any(scale == 0.0):
        raise ValueError("constant coordinate prevents distance standardization")
    return (array - array.mean(axis=0)) / scale


def main() -> None:
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("q stability plan changed")
    if sha256(DATA_CSV) != EXPECTED_DATA_SHA256:
        raise RuntimeError("development data changed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(DATA_CSV)
    data["log_pressure"] = np.log(data["pressure_kpa"].to_numpy(float))
    reference_by_fold = {
        fold: float(data.loc[~data["fold"].eq(fold), "temperature_k"].median())
        for fold in range(5)
    }
    coefficient_rows = []
    prediction_rows = []
    response_rows = []
    for entity_id, entity in data.groupby("entity_id", sort=True):
        ordered = entity.sort_values("temperature_k", kind="stable").reset_index(drop=True)
        fold = int(ordered["fold"].iloc[0])
        reference = reference_by_fold[fold]
        normalized_temperature = (
            ordered["temperature_k"].to_numpy(float)
            - float(ordered["temperature_k"].min())
        ) / (
            float(ordered["temperature_k"].max())
            - float(ordered["temperature_k"].min())
        )
        response_rows.append(
            np.interp(
                np.linspace(0.0, 1.0, 41),
                normalized_temperature,
                ordered["log_pressure"].to_numpy(float),
            )
        )
        for offset in range(4):
            support_mask = np.arange(len(ordered)) % 4 == offset
            support = ordered.loc[support_mask]
            query = ordered.loc[~support_mask]
            support_design = design(
                support["temperature_k"].to_numpy(float), reference
            )
            coefficients = np.linalg.lstsq(
                support_design,
                support["log_pressure"].to_numpy(float),
                rcond=None,
            )[0]
            column_norms = np.linalg.norm(support_design, axis=0)
            scaled_condition_number = float(
                np.linalg.cond(support_design / column_norms)
            )
            prediction_log = design(
                query["temperature_k"].to_numpy(float), reference
            ) @ coefficients
            prediction_kpa = np.exp(prediction_log)
            coefficient_rows.append(
                {
                    "fold": fold,
                    "entity_id": entity_id,
                    "doi": ordered["doi"].iloc[0],
                    "common_name": ordered["common_name"].iloc[0],
                    "formula": ordered["formula"].iloc[0],
                    "offset": offset,
                    "support_rows": len(support),
                    "query_rows": len(query),
                    "temperature_reference_k": reference,
                    "q0_log_pressure_at_reference": float(coefficients[0]),
                    "q1_inverse_temperature_k": float(coefficients[1]),
                    "q2_log_temperature": float(coefficients[2]),
                    "effective_delta_cp_j_mol_k": float(R_GAS * coefficients[2]),
                    "effective_delta_hvap_kj_mol": float(
                        R_GAS
                        * (coefficients[2] * reference - coefficients[1])
                        / 1000.0
                    ),
                    "scaled_design_condition_number": scaled_condition_number,
                }
            )
            for row, log_value, physical_value in zip(
                query.itertuples(index=False),
                prediction_log,
                prediction_kpa,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "source_row_id": row.source_row_id,
                        "entity_id": entity_id,
                        "doi": row.doi,
                        "offset": offset,
                        "temperature_k": row.temperature_k,
                        "pressure_kpa": row.pressure_kpa,
                        "log_pressure": row.log_pressure,
                        "prediction_kpa": float(physical_value),
                        "prediction_log_pressure": float(log_value),
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)
    predictions = pd.DataFrame(prediction_rows)
    offset_rows = []
    for offset, frame in predictions.groupby("offset", sort=True):
        offset_rows.append(
            {
                "offset": int(offset),
                "pooled_physical_r2": float(
                    r2_score(frame["pressure_kpa"], frame["prediction_kpa"])
                ),
                "pooled_log_r2": float(
                    r2_score(
                        frame["log_pressure"], frame["prediction_log_pressure"]
                    )
                ),
            }
        )
    offset_summary = pd.DataFrame(offset_rows)

    physical_columns = [
        "q0_log_pressure_at_reference",
        "effective_delta_hvap_kj_mol",
        "effective_delta_cp_j_mol_k",
    ]
    stability_rows = []
    for left_offset, right_offset in combinations(range(4), 2):
        left = coefficients.loc[coefficients["offset"].eq(left_offset)].sort_values(
            "entity_id"
        )
        right = coefficients.loc[coefficients["offset"].eq(right_offset)].sort_values(
            "entity_id"
        )
        if left["entity_id"].tolist() != right["entity_id"].tolist():
            raise ValueError("support-offset entity mismatch")
        row = {"left_offset": left_offset, "right_offset": right_offset}
        for column in physical_columns:
            row[f"{column}_spearman"] = float(
                spearmanr(left[column], right[column]).statistic
            )
        row["physical_q_distance_spearman"] = float(
            spearmanr(
                pdist(standardized(left[physical_columns].to_numpy(float))),
                pdist(standardized(right[physical_columns].to_numpy(float))),
            ).statistic
        )
        stability_rows.append(row)
    stability = pd.DataFrame(stability_rows)

    offset0 = coefficients.loc[coefficients["offset"].eq(0)].sort_values("entity_id")
    response = np.asarray(response_rows)
    response_distance = pdist(standardized(response))
    q_distance = pdist(standardized(offset0[physical_columns].to_numpy(float)))
    response_geometry_spearman = float(
        spearmanr(q_distance, response_distance).statistic
    )
    coordinate_distribution = {
        column: {
            "median": float(offset0[column].median()),
            "q05": float(offset0[column].quantile(0.05)),
            "q95": float(offset0[column].quantile(0.95)),
            "min": float(offset0[column].min()),
            "max": float(offset0[column].max()),
        }
        for column in physical_columns
    }
    condition_distribution = {
        "median": float(coefficients["scaled_design_condition_number"].median()),
        "q95": float(coefficients["scaled_design_condition_number"].quantile(0.95)),
        "maximum": float(coefficients["scaled_design_condition_number"].max()),
    }
    result = {
        "scope": "ThermoML development selected-expression support-offset q stability",
        "selected_expression": "ln(P/1 kPa)=q0+q1*(1/T-1/T_ref)+q2*ln(T/T_ref)",
        "offset_summary": offset_summary.to_dict(orient="records"),
        "median_physical_q_distance_spearman_across_offsets": float(
            stability["physical_q_distance_spearman"].median()
        ),
        "minimum_physical_q_distance_spearman_across_offsets": float(
            stability["physical_q_distance_spearman"].min()
        ),
        "q_response_curve_distance_spearman": response_geometry_spearman,
        "coordinate_distribution_offset0": coordinate_distribution,
        "scaled_design_condition_number": condition_distribution,
        "confirmation_targets_opened": False,
        "interpretation_scope": "effective stage-wise coordinates, not definitive thermodynamic measurements",
    }
    coefficients.to_csv(OUTPUT_ROOT / "offset_expression_coefficients.csv", index=False)
    predictions.to_csv(OUTPUT_ROOT / "offset_query_predictions.csv", index=False)
    offset_summary.to_csv(OUTPUT_ROOT / "offset_prediction_summary.csv", index=False)
    stability.to_csv(OUTPUT_ROOT / "offset_q_stability.csv", index=False)
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "plan_sha256": EXPECTED_PLAN_SHA256,
                "data_sha256": EXPECTED_DATA_SHA256,
                "analyzer_sha256": sha256(Path(__file__)),
                "confirmation_targets_opened": False,
                "files": {
                    path.name: sha256(path)
                    for path in sorted(OUTPUT_ROOT.iterdir())
                    if path.is_file() and path.name != "manifest.json"
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
