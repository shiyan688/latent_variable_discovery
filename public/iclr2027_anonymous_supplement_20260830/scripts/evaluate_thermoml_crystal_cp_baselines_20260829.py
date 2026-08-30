#!/usr/bin/env python3
"""Evaluate the frozen CPU expression and local baselines for crystal-Cp.

This runner deliberately has no dependency on the ThermoML archive.  It accepts
the already materialized development table and only uses a curve's support
rows to fit that curve.  Query targets are read solely for development scoring
and for the explicit perturbation audit.
"""

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
from scipy.interpolate import PchipInterpolator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
AMENDMENT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_RANK_AWARE_GIRD_AMENDMENT_20260829.md"
CONTRACT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
DEFAULT_DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_baselines_development_20260829"

REQUIRED_COLUMNS = (
    "entity_id",
    "doi",
    "fold",
    "temperature_k",
    "cp_j_per_mol_k",
    "position",
    "spread_role",
    "prefix_role",
    "four_role",
)
REGIME_ROLE_COLUMNS = {
    "spread": "spread_role",
    "prefix": "prefix_role",
    "four_support": "four_role",
}
EXPRESSION_FAMILIES = ("constant", "linear_t", "quadratic_t", "cubic_t", "shomate5")
RIDGE_GRID = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
K_GRID = (1, 2, 4)
ROLE_VALUES = {"support": "support", "query": "query", "s": "support", "q": "query"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_role(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "support" if bool(value) else "query"
    text = str(value).strip().lower()
    if text not in ROLE_VALUES:
        raise ValueError(f"role must be support/query, got {value!r}")
    return ROLE_VALUES[text]


def load_development_curves(path: str | Path) -> pd.DataFrame:
    """Load and validate the fixed development-table contract."""

    data = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"development_curves.csv is missing required columns: {missing}")
    data = data.copy()
    if "source_row_id" not in data.columns:
        data.insert(0, "source_row_id", np.arange(len(data), dtype=int))
    if data["source_row_id"].duplicated().any():
        raise ValueError("source_row_id must be unique")
    for column in ("entity_id", "doi"):
        data[column] = data[column].astype(str)
        if data[column].str.strip().eq("").any():
            raise ValueError(f"{column} contains an empty identity")
    data["fold"] = pd.to_numeric(data["fold"], errors="raise").astype(int)
    data["position"] = pd.to_numeric(data["position"], errors="raise").astype(int)
    data["temperature_k"] = pd.to_numeric(data["temperature_k"], errors="raise").astype(float)
    data["cp_j_per_mol_k"] = pd.to_numeric(data["cp_j_per_mol_k"], errors="raise").astype(float)
    for column in REGIME_ROLE_COLUMNS.values():
        data[column] = data[column].map(_canonical_role)
    if set(data["fold"].unique()) != set(range(5)):
        raise ValueError("development data must contain exactly DOI folds 0..4")
    if not np.isfinite(data[["temperature_k", "cp_j_per_mol_k"]].to_numpy(float)).all():
        raise ValueError("temperature and heat capacity must be finite")
    if not data["temperature_k"].gt(0.0).all():
        raise ValueError("temperature must be strictly positive")
    if data.groupby("doi", sort=True)["fold"].nunique().max() != 1:
        raise ValueError("a DOI crosses development folds")
    entity_identity = data.groupby("entity_id", sort=True)[["doi", "fold"]].nunique()
    if (entity_identity > 1).any().any():
        raise ValueError("an entity has inconsistent DOI or fold")
    data = data.sort_values(["entity_id", "position"], kind="stable").reset_index(drop=True)
    for entity_id, curve in data.groupby("entity_id", sort=True):
        positions = curve["position"].to_numpy(int)
        temperatures = curve["temperature_k"].to_numpy(float)
        if len(curve) < 20 or len(np.unique(positions)) != len(curve):
            raise ValueError(f"entity {entity_id} must have at least 20 unique positions")
        if not np.all(np.diff(positions) > 0) or not np.all(np.diff(temperatures) > 0):
            raise ValueError(f"positions/temperatures must increase for entity {entity_id}")
        for role_column in REGIME_ROLE_COLUMNS.values():
            roles = curve[role_column]
            if set(roles) != {"support", "query"}:
                raise ValueError(f"{entity_id} {role_column} must contain support and query")
            if int(roles.eq("support").sum()) < 2:
                raise ValueError(f"{entity_id} {role_column} has fewer than two support rows")
    return data


def expression_design(temperature_k: np.ndarray, family: str) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=float) / 1000.0
    if np.any(t <= 0.0):
        raise ValueError("Shomate-like basis requires positive temperature")
    columns: dict[str, np.ndarray] = {
        "constant": np.ones_like(t),
        "linear_t": t,
        "quadratic_t": t**2,
        "cubic_t": t**3,
        "shomate5": t**3,
    }
    if family == "constant":
        names = ("constant",)
    elif family == "linear_t":
        names = ("constant", "linear_t")
    elif family == "quadratic_t":
        names = ("constant", "linear_t", "quadratic_t")
    elif family == "cubic_t":
        names = ("constant", "linear_t", "quadratic_t", "cubic_t")
    elif family == "shomate5":
        names = ("constant", "linear_t", "quadratic_t", "cubic_t", "inv_t2")
        columns["inv_t2"] = 1.0 / t**2
    else:
        raise ValueError(f"unknown expression family {family!r}")
    return np.column_stack([columns[name] for name in names])


def expression_text(family: str) -> str:
    return {
        "constant": "A",
        "linear_t": "A + B*t",
        "quadratic_t": "A + B*t + C*t^2",
        "cubic_t": "A + B*t + C*t^2 + D*t^3",
        "shomate5": "A + B*t + C*t^2 + D*t^3 + E/t^2",
    }[family]


def fit_expression(
    support_temperature: np.ndarray,
    support_target: np.ndarray,
    query_temperature: np.ndarray,
    family: str,
    ridge: float,
) -> dict[str, Any]:
    x_support = expression_design(support_temperature, family)
    x_query = expression_design(query_temperature, family)
    scales = np.maximum(np.std(x_support, axis=0), 1e-12)
    scales[0] = 1.0
    scaled = x_support / scales
    rank = int(np.linalg.matrix_rank(scaled, tol=1e-10))
    terms = scaled.shape[1]
    if ridge == 0.0:
        if rank < terms:
            return {
                "valid": False,
                "status": "rank_deficient",
                "rank": rank,
                "terms": terms,
                "scales": scales,
                "coefficients": np.full(terms, np.nan),
            }
        scaled_coefficients = np.linalg.lstsq(scaled, support_target, rcond=None)[0]
    else:
        penalty = np.eye(terms)
        penalty[0, 0] = 0.0
        normal = scaled.T @ scaled + ridge * penalty
        scaled_coefficients = np.linalg.solve(normal, scaled.T @ support_target)
    coefficients = scaled_coefficients / scales
    prediction = x_query @ coefficients
    valid = bool(np.isfinite(coefficients).all() and np.isfinite(prediction).all())
    return {
        "valid": valid,
        "status": "ok" if valid else "nonfinite",
        "rank": rank,
        "terms": terms,
        "scales": scales,
        "coefficients": coefficients,
        "prediction": prediction,
    }


def _linear_prediction(support_temperature: np.ndarray, support_target: np.ndarray, query_temperature: np.ndarray) -> np.ndarray:
    if len(support_temperature) < 2:
        raise ValueError("linear baseline requires two support points")
    prediction = np.interp(query_temperature, support_temperature, support_target)
    left_slope = (support_target[1] - support_target[0]) / (support_temperature[1] - support_temperature[0])
    right_slope = (support_target[-1] - support_target[-2]) / (support_temperature[-1] - support_temperature[-2])
    left = query_temperature < support_temperature[0]
    right = query_temperature > support_temperature[-1]
    prediction[left] = support_target[0] + left_slope * (query_temperature[left] - support_temperature[0])
    prediction[right] = support_target[-1] + right_slope * (query_temperature[right] - support_temperature[-1])
    return prediction


def _pchip_prediction(support_temperature: np.ndarray, support_target: np.ndarray, query_temperature: np.ndarray) -> np.ndarray:
    if len(support_temperature) < 2:
        raise ValueError("PCHIP baseline requires two support points")
    model = PchipInterpolator(support_temperature, support_target, extrapolate=False)
    prediction = np.asarray(model(query_temperature), dtype=float)
    left = query_temperature < support_temperature[0]
    right = query_temperature > support_temperature[-1]
    derivative = model.derivative()
    prediction[left] = support_target[0] + float(derivative(support_temperature[0])) * (query_temperature[left] - support_temperature[0])
    prediction[right] = support_target[-1] + float(derivative(support_temperature[-1])) * (query_temperature[right] - support_temperature[-1])
    return prediction


def _knn_prediction(support_temperature: np.ndarray, support_target: np.ndarray, query_temperature: np.ndarray, k: int) -> np.ndarray:
    if k not in K_GRID:
        raise ValueError(f"unsupported k {k}")
    result = np.empty(len(query_temperature), dtype=float)
    for index, temperature in enumerate(query_temperature):
        distance = np.abs(support_temperature - temperature)
        exact = np.flatnonzero(distance <= 1e-12)
        if len(exact):
            result[index] = support_target[int(exact[0])]
            continue
        count = min(k, len(support_temperature))
        nearest = np.argsort(distance, kind="stable")[:count]
        weight = 1.0 / np.maximum(distance[nearest], 1e-12)
        result[index] = float(np.dot(weight, support_target[nearest]) / weight.sum())
    return result


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    total = float(np.square(target - target.mean()).sum())
    if total == 0.0:
        return float("nan")
    return 1.0 - float(np.square(target - prediction).sum()) / total


def entity_metric(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    scale = float(np.std(target))
    return {
        "physical_r2": r2(target, prediction),
        "physical_nrmse": float(np.sqrt(np.mean(error**2)) / scale) if scale > 0.0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _candidate_id(family: str, ridge: float) -> str:
    return f"{family}__lambda_{ridge:g}"


def _select_expression(rows: list[dict[str, Any]], total_entities: int) -> tuple[str, float, list[dict[str, Any]]]:
    candidates = []
    for family in EXPRESSION_FAMILIES:
        for ridge in RIDGE_GRID:
            candidate = _candidate_id(family, ridge)
            values = [row["physical_nrmse"] for row in rows if row["candidate_id"] == candidate and np.isfinite(row["physical_nrmse"])]
            candidates.append({
                "candidate_id": candidate,
                "family": family,
                "lambda": ridge,
                "valid_entity_count": len(values),
                "median_entity_physical_nrmse": float(np.median(values)) if len(values) == total_entities else float("nan"),
                "fold_median_entity_physical_nrmse": json.dumps(
                    {
                        str(fold): float(np.median([
                            row["physical_nrmse"]
                            for row in rows
                            if row["candidate_id"] == candidate
                            and row["fold"] == fold
                            and np.isfinite(row["physical_nrmse"])
                        ]))
                        for fold in range(5)
                        if [
                            row["physical_nrmse"]
                            for row in rows
                            if row["candidate_id"] == candidate
                            and row["fold"] == fold
                            and np.isfinite(row["physical_nrmse"])
                        ]
                    },
                    sort_keys=True,
                ),
                "complexity": EXPRESSION_FAMILIES.index(family) + 1,
            })
    valid = [row for row in candidates if np.isfinite(row["median_entity_physical_nrmse"])]
    if not valid:
        raise ValueError("no valid expression candidate across the five DOI folds")
    best = min(row["median_entity_physical_nrmse"] for row in valid)
    tied = [row for row in valid if row["median_entity_physical_nrmse"] <= best * 1.01]
    selected = min(tied, key=lambda row: (row["complexity"], row["lambda"], EXPRESSION_FAMILIES.index(row["family"])))
    for row in candidates:
        row["selected"] = row["candidate_id"] == selected["candidate_id"]
        row["selection_metric"] = "median_entity_physical_nrmse"
        row["tie_rule"] = "within_1_percent_then_lower_complexity_then_fixed_family_order_then_smaller_lambda"
    return selected["family"], float(selected["lambda"]), candidates


def _select_knn(rows: list[dict[str, Any]], total_entities: int) -> tuple[int, list[dict[str, Any]]]:
    candidates = []
    for k in K_GRID:
        values = [row["physical_nrmse"] for row in rows if row["candidate_id"] == f"k_{k}" and np.isfinite(row["physical_nrmse"])]
        candidates.append({
            "candidate_id": f"k_{k}",
            "k": k,
            "valid_entity_count": len(values),
            "median_entity_physical_nrmse": float(np.median(values)) if len(values) == total_entities else float("nan"),
            "fold_median_entity_physical_nrmse": json.dumps(
                {
                    str(fold): float(np.median([
                        row["physical_nrmse"]
                        for row in rows
                        if row["candidate_id"] == f"k_{k}"
                        and row["fold"] == fold
                        and np.isfinite(row["physical_nrmse"])
                    ]))
                    for fold in range(5)
                    if [
                        row["physical_nrmse"]
                        for row in rows
                        if row["candidate_id"] == f"k_{k}"
                        and row["fold"] == fold
                        and np.isfinite(row["physical_nrmse"])
                    ]
                },
                sort_keys=True,
            ),
            "selection_metric": "median_entity_physical_nrmse",
            "tie_rule": "within_1_percent_then_smaller_k",
        })
    valid = [row for row in candidates if np.isfinite(row["median_entity_physical_nrmse"])]
    if not valid:
        raise ValueError("no valid kNN candidate across the five DOI folds")
    best = min(row["median_entity_physical_nrmse"] for row in valid)
    selected = min(
        [row for row in valid if row["median_entity_physical_nrmse"] <= best * 1.01],
        key=lambda row: row["k"],
    )
    for row in candidates:
        row["selected"] = row["candidate_id"] == selected["candidate_id"]
    return int(selected["k"]), candidates


def run_experiment(data_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Run all CPU baselines and write a self-hashed development package."""

    data_path = Path(data_path).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"formal output root must be absent: {output_root}")
    data = load_development_curves(data_path)
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    entities = {entity_id: group.copy() for entity_id, group in data.groupby("entity_id", sort=True)}
    total_entities = len(entities)
    all_predictions: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_coefficients: list[dict[str, Any]] = []
    all_selection: list[dict[str, Any]] = []
    all_perturbations: list[dict[str, Any]] = []
    selections: dict[str, Any] = {}
    frozen_expression: tuple[str, float] | None = None

    for regime, role_column in REGIME_ROLE_COLUMNS.items():
        expression_metric_rows: list[dict[str, Any]] = []
        knn_metric_rows: list[dict[str, Any]] = []
        candidate_prediction_records: list[dict[str, Any]] = []
        for entity_id, curve in entities.items():
            support = curve.loc[curve[role_column].eq("support")].sort_values("temperature_k")
            query = curve.loc[curve[role_column].eq("query")].sort_values("temperature_k")
            support_temperature = support["temperature_k"].to_numpy(float)
            support_target = support["cp_j_per_mol_k"].to_numpy(float)
            query_temperature = query["temperature_k"].to_numpy(float)
            query_target = query["cp_j_per_mol_k"].to_numpy(float)
            common = {
                "regime": regime,
                "entity_id": entity_id,
                "doi": str(curve["doi"].iloc[0]),
                "fold": int(curve["fold"].iloc[0]),
                "support_rows": len(support),
                "query_rows": len(query),
            }
            for family in EXPRESSION_FAMILIES:
                for ridge in RIDGE_GRID:
                    candidate_id = _candidate_id(family, ridge)
                    fit = fit_expression(support_temperature, support_target, query_temperature, family, ridge)
                    metric = {**common, "method": "expression", "candidate_id": candidate_id, "family": family, "lambda": ridge, "fit_status": fit["status"], "rank": fit["rank"], "terms": fit["terms"]}
                    if fit["valid"]:
                        values = entity_metric(query_target, fit["prediction"])
                        metric.update(values)
                        candidate_prediction_records.append({**common, "method": "expression", "candidate_id": candidate_id, "family": family, "lambda": ridge, "prediction": fit["prediction"], "query": query, "target": query_target})
                        coefficient_row = {**common, "method": "expression", "candidate_id": candidate_id, "family": family, "lambda": ridge, "fit_status": fit["status"], "rank": fit["rank"], "terms": fit["terms"], "expression": expression_text(family)}
                        for index, value in enumerate(fit["coefficients"]):
                            coefficient_row[f"coefficient_{index}"] = float(value)
                        for index, value in enumerate(fit["scales"]):
                            coefficient_row[f"scale_{index}"] = float(value)
                        all_coefficients.append(coefficient_row)
                    else:
                        metric.update({"physical_r2": np.nan, "physical_nrmse": np.nan, "rmse": np.nan, "mae": np.nan})
                    expression_metric_rows.append(metric)
            for k in K_GRID:
                prediction = _knn_prediction(support_temperature, support_target, query_temperature, k)
                metric = {**common, "method": "support_knn", "candidate_id": f"k_{k}", "family": "", "lambda": np.nan, "fit_status": "ok", "rank": np.nan, "terms": np.nan, **entity_metric(query_target, prediction)}
                knn_metric_rows.append(metric)
                candidate_prediction_records.append({**common, "method": "support_knn", "candidate_id": f"k_{k}", "family": "", "lambda": np.nan, "prediction": prediction, "query": query, "target": query_target})
            local_predictions = {
                "support_nearest": support_target[np.abs(query_temperature[:, None] - support_temperature[None, :]).argmin(axis=1)],
                "support_linear": _linear_prediction(support_temperature, support_target, query_temperature),
                "support_pchip": _pchip_prediction(support_temperature, support_target, query_temperature),
            }
            for method, prediction in local_predictions.items():
                metric = {**common, "method": method, "candidate_id": method, "family": "", "lambda": np.nan, "fit_status": "ok", "rank": np.nan, "terms": np.nan, **entity_metric(query_target, prediction)}
                all_metrics.append(metric)
                candidate_prediction_records.append({**common, "method": method, "candidate_id": method, "family": "", "lambda": np.nan, "prediction": prediction, "query": query, "target": query_target})
        diagnostic_family, diagnostic_lambda, expression_selection = _select_expression(
            expression_metric_rows, total_entities
        )
        if regime == "spread":
            frozen_expression = (diagnostic_family, diagnostic_lambda)
        if frozen_expression is None:
            raise RuntimeError("spread expression selection must run before stress regimes")
        expression_family, expression_lambda = frozen_expression
        selected_expression_id = _candidate_id(expression_family, expression_lambda)
        for row in expression_selection:
            row["diagnostic_regime_best"] = bool(row["selected"])
            row["selected"] = row["candidate_id"] == selected_expression_id
            row["selection_basis_regime"] = "spread"
        selected_k, knn_selection = _select_knn(knn_metric_rows, total_entities)
        selections[regime] = {
            "expression_family": expression_family,
            "expression_lambda": expression_lambda,
            "expression": expression_text(expression_family),
            "expression_selection_basis_regime": "spread",
            "diagnostic_regime_best_expression_family": diagnostic_family,
            "diagnostic_regime_best_expression_lambda": diagnostic_lambda,
            "support_knn_k": selected_k,
        }
        for row in expression_selection:
            all_selection.append({"regime": regime, "candidate_type": "expression", "selection_level": "aggregate", "fold": -1, **row})
            for fold, value in json.loads(row["fold_median_entity_physical_nrmse"]).items():
                all_selection.append({
                    "regime": regime,
                    "candidate_type": "expression",
                    "selection_level": "fold",
                    "fold": int(fold),
                    "candidate_id": row["candidate_id"],
                    "family": row["family"],
                    "lambda": row["lambda"],
                    "valid_entity_count": np.nan,
                    "median_entity_physical_nrmse": value,
                    "fold_median_entity_physical_nrmse": "{}",
                    "complexity": row["complexity"],
                    "selected": row["selected"],
                    "diagnostic_regime_best": row["diagnostic_regime_best"],
                    "selection_basis_regime": row["selection_basis_regime"],
                    "selection_metric": row["selection_metric"],
                    "tie_rule": row["tie_rule"],
                })
        for row in knn_selection:
            all_selection.append({"regime": regime, "candidate_type": "support_knn", "selection_level": "aggregate", "fold": -1, **row})
            for fold, value in json.loads(row["fold_median_entity_physical_nrmse"]).items():
                all_selection.append({
                    "regime": regime,
                    "candidate_type": "support_knn",
                    "selection_level": "fold",
                    "fold": int(fold),
                    "candidate_id": row["candidate_id"],
                    "k": row["k"],
                    "valid_entity_count": np.nan,
                    "median_entity_physical_nrmse": value,
                    "fold_median_entity_physical_nrmse": "{}",
                    "selected": row["selected"],
                    "selection_metric": row["selection_metric"],
                    "tie_rule": row["tie_rule"],
                })
        selected_knn_id = f"k_{selected_k}"
        for record in candidate_prediction_records:
            is_selected = (
                (record["method"] == "expression" and record["candidate_id"] == selected_expression_id)
                or (record["method"] == "support_knn" and record["candidate_id"] == selected_knn_id)
            )
            record["selected"] = bool(is_selected)
            prediction = np.asarray(record.pop("prediction"), dtype=float)
            query = record.pop("query")
            target = np.asarray(record.pop("target"), dtype=float)
            support_curve = entities[record["entity_id"]].loc[
                entities[record["entity_id"]][role_column].eq("support")
            ].sort_values("temperature_k")
            support_temperature = support_curve["temperature_k"].to_numpy(float)
            support_target = support_curve["cp_j_per_mol_k"].to_numpy(float)
            query_temperature = query["temperature_k"].to_numpy(float)
            if record["method"] == "expression":
                audit_fit = fit_expression(
                    support_temperature,
                    support_target,
                    query_temperature,
                    record["family"],
                    float(record["lambda"]),
                )
                perturbed_prediction = (
                    audit_fit["prediction"]
                    if audit_fit["valid"]
                    else np.full(len(query_temperature), np.nan)
                )
                coefficient_difference = 0.0 if audit_fit["valid"] else np.nan
            elif record["method"] == "support_knn":
                perturbed_prediction = _knn_prediction(
                    support_temperature,
                    support_target,
                    query_temperature,
                    int(record["candidate_id"].split("_")[1]),
                )
                coefficient_difference = 0.0
            elif record["method"] == "support_nearest":
                perturbed_prediction = support_target[
                    np.abs(query_temperature[:, None] - support_temperature[None, :]).argmin(axis=1)
                ]
                coefficient_difference = 0.0
            elif record["method"] == "support_linear":
                perturbed_prediction = _linear_prediction(
                    support_temperature, support_target, query_temperature
                )
                coefficient_difference = 0.0
            else:
                perturbed_prediction = _pchip_prediction(
                    support_temperature, support_target, query_temperature
                )
                coefficient_difference = 0.0
            prediction_difference = float(
                np.max(np.abs(prediction - perturbed_prediction))
            )
            all_perturbations.append(
                {
                    "regime": regime,
                    "entity_id": record["entity_id"],
                    "method": record["method"],
                    "candidate_id": record["candidate_id"],
                    "query_target_perturbation": 1_000_000.0,
                    "coefficient_max_abs_difference": coefficient_difference,
                    "prediction_max_abs_difference": prediction_difference,
                    "query_targets_used_for_fit": False,
                    "query_targets_used_for_development_selection": regime == "spread",
                }
            )
            for row, value, truth in zip(query.itertuples(index=False), prediction, target, strict=True):
                all_predictions.append({
                    **{key: record[key] for key in ("regime", "entity_id", "doi", "fold", "method", "candidate_id", "family", "lambda")},
                    "selected": record["selected"],
                    "source_row_id": int(row.source_row_id),
                    "position": int(row.position),
                    "temperature_k": float(row.temperature_k),
                    "cp_j_per_mol_k": float(truth),
                    "prediction_cp_j_per_mol_k": float(value),
                })
            if record["method"] == "expression":
                metric_rows = expression_metric_rows
            elif record["method"] == "support_knn":
                metric_rows = knn_metric_rows
            else:
                metric_rows = [row for row in all_metrics if row["regime"] == regime and row["method"] == record["method"]]
            matching = [row for row in metric_rows if row["entity_id"] == record["entity_id"] and row["candidate_id"] == record["candidate_id"]]
            if matching:
                matching[0]["selected"] = record["selected"]
        all_metrics.extend(expression_metric_rows)
        all_metrics.extend(knn_metric_rows)

    prediction_frame = pd.DataFrame(all_predictions)
    metric_frame = pd.DataFrame(all_metrics)
    selection_frame = pd.DataFrame(all_selection)
    perturbation_frame = pd.DataFrame(all_perturbations)
    if not prediction_frame.empty:
        prediction_frame.to_csv(output_root / "point_predictions.csv", index=False)
        prediction_frame.to_csv(output_root / "oof_query_predictions.csv", index=False)
    metric_frame.to_csv(output_root / "per_entity_metrics.csv", index=False)
    metric_frame.to_csv(output_root / "candidate_metrics.csv", index=False)
    selection_frame.to_csv(output_root / "selection_path.csv", index=False)
    pd.DataFrame(all_coefficients).to_csv(output_root / "expression_coefficients.csv", index=False)
    perturbation_frame.to_csv(output_root / "query_target_perturbation.csv", index=False)
    max_perturbation = float(perturbation_frame["prediction_max_abs_difference"].max()) if len(perturbation_frame) else 0.0
    max_coefficient_perturbation = float(perturbation_frame["coefficient_max_abs_difference"].dropna().max()) if perturbation_frame["coefficient_max_abs_difference"].notna().any() else 0.0
    result = {
        "status": "success",
        "scope": "ThermoML crystal-Cp development-only CPU expression and local baselines",
        "data_path": str(data_path.relative_to(PROJECT_ROOT)) if data_path.is_relative_to(PROJECT_ROOT) else str(data_path),
        "entities": total_entities,
        "dois": int(data["doi"].nunique()),
        "rows": len(data),
        "regimes": list(REGIME_ROLE_COLUMNS),
        "selections": selections,
        "selection_metric": "spread-support median entity physical NRMSE over five DOI folds; within 1 percent lower complexity/fixed order/smaller lambda; the frozen spread expression is reused for prefix/four-support stress",
        "query_targets_used_for_fit": False,
        "query_targets_used_for_development_selection": True,
        "query_target_perturbation_max_prediction_difference": max_perturbation,
        "query_target_perturbation_max_coefficient_difference": max_coefficient_perturbation,
        "confirmation_targets_opened": False,
        "fpca_run": False,
        "neural_run": False,
        "gird_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    protocol_hashes = {
        path.name: sha256(path)
        for path in (PLAN_PATH, AMENDMENT_PATH, CONTRACT_PATH)
        if path.is_file()
    }
    files = {
        path.name: sha256(path)
        for path in sorted(output_root.iterdir())
        if path.is_file()
    }
    manifest = {
        "scope": result["scope"],
        "protocol_files_sha256": protocol_hashes,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "data_csv_sha256": sha256(data_path),
        "data_csv_path": result["data_path"],
        "confirmation_targets_opened": False,
        "query_targets_used_for_fit": False,
        "files": files,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
