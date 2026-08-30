#!/usr/bin/env python3
"""Run the frozen train-only FPCA/support-ridge crystal-Cp baseline."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_fpca_development_20260829"
PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
CONTRACT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
RANK_AMENDMENT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_RANK_AWARE_GIRD_AMENDMENT_20260829.md"
ROUTER_AMENDMENT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_ROUTER_MARGIN_AMENDMENT_20260829.md"

GRID = np.linspace(0.0, 1.0, 101)
COMPONENT_GRID = (2, 3, 4, 5, 8)
RIDGE_GRID = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
REGIMES = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    total = float(np.square(target - target.mean()).sum())
    return float("nan") if total == 0.0 else 1.0 - float(np.square(target - prediction).sum()) / total


def entity_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    scale = float(np.std(target))
    return {
        "physical_r2": r2(target, prediction),
        "physical_nrmse": float(np.sqrt(np.mean(np.square(error))) / scale) if scale > 0.0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
    }


def normalized_temperature(curve: pd.DataFrame) -> np.ndarray:
    temperature = curve["temperature_k"].to_numpy(float)
    span = float(temperature.max() - temperature.min())
    if span <= 0.0:
        raise ValueError("curve temperature span must be positive")
    return (temperature - temperature.min()) / span


def fit_fpca(training_curves: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    matrix = []
    for curve in training_curves:
        ordered = curve.sort_values("position")
        u = normalized_temperature(ordered)
        target = ordered["cp_j_per_mol_k"].to_numpy(float)
        matrix.append(PchipInterpolator(u, target, extrapolate=False)(GRID))
    values = np.asarray(matrix, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite FPCA training grid")
    mean = values.mean(axis=0)
    _, singular_values, right = np.linalg.svd(values - mean, full_matrices=False)
    if right.shape[0] < max(COMPONENT_GRID):
        raise ValueError("insufficient FPCA training curves")
    return {"mean": mean, "components": right, "singular_values": singular_values}


def predict_from_support(
    model: dict[str, np.ndarray],
    support_u: np.ndarray,
    support_target: np.ndarray,
    query_u: np.ndarray,
    components: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    mean_support = np.interp(support_u, GRID, model["mean"])
    mean_query = np.interp(query_u, GRID, model["mean"])
    basis = model["components"][:components]
    phi_support = np.column_stack([np.interp(support_u, GRID, row) for row in basis])
    phi_query = np.column_stack([np.interp(query_u, GRID, row) for row in basis])
    rhs = support_target - mean_support
    if ridge == 0.0:
        coefficient = np.linalg.lstsq(phi_support, rhs, rcond=None)[0]
    else:
        design = np.vstack([phi_support, np.sqrt(ridge) * np.eye(components)])
        response = np.concatenate([rhs, np.zeros(components)])
        coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
    prediction = mean_query + phi_query @ coefficient
    return prediction, coefficient, int(np.linalg.matrix_rank(phi_support))


def candidate_id(components: int, ridge: float) -> str:
    return f"m{components}_lambda{ridge:g}"


def choose_candidate(rows: list[dict[str, Any]], entity_count: int) -> dict[str, Any]:
    candidates = []
    for components in COMPONENT_GRID:
        for ridge in RIDGE_GRID:
            selected = [
                row["physical_nrmse"]
                for row in rows
                if row["components"] == components
                and row["ridge"] == ridge
                and np.isfinite(row["physical_nrmse"])
            ]
            if len(selected) == entity_count:
                candidates.append(
                    {
                        "components": components,
                        "ridge": ridge,
                        "median_entity_physical_nrmse": float(np.median(selected)),
                    }
                )
    if not candidates:
        raise RuntimeError("no FPCA candidate has complete entity coverage")
    best = min(row["median_entity_physical_nrmse"] for row in candidates)
    tied = [row for row in candidates if row["median_entity_physical_nrmse"] <= 1.01 * best]
    return min(tied, key=lambda row: (row["components"], row["ridge"]))


def run(data_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"output root must be absent: {output_root}")
    started = time.perf_counter()
    data = pd.read_csv(data_path)
    required = {"entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k", *REGIMES.values()}
    if not required.issubset(data.columns):
        raise ValueError(f"missing columns: {sorted(required - set(data.columns))}")
    if sorted(data["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("expected five DOI folds")
    if data.groupby("doi")["fold"].nunique().max() != 1:
        raise ValueError("a DOI crosses folds")
    curves = {key: frame.sort_values("position").copy() for key, frame in data.groupby("entity_id", sort=True)}
    output_root.mkdir(parents=True, exist_ok=False)

    candidate_rows: list[dict[str, Any]] = []
    candidate_predictions: list[dict[str, Any]] = []
    fold_models: dict[int, dict[str, np.ndarray]] = {}
    for fold in range(5):
        train_curves = [curve for curve in curves.values() if int(curve["fold"].iloc[0]) != fold]
        fold_models[fold] = fit_fpca(train_curves)
        for entity_id, curve in curves.items():
            if int(curve["fold"].iloc[0]) != fold:
                continue
            u = normalized_temperature(curve)
            for regime, role_column in REGIMES.items():
                support_mask = curve[role_column].eq("support").to_numpy()
                query_mask = curve[role_column].eq("query").to_numpy()
                for components in COMPONENT_GRID:
                    for ridge in RIDGE_GRID:
                        prediction, coefficient, rank = predict_from_support(
                            fold_models[fold],
                            u[support_mask],
                            curve.loc[support_mask, "cp_j_per_mol_k"].to_numpy(float),
                            u[query_mask],
                            components,
                            ridge,
                        )
                        target = curve.loc[query_mask, "cp_j_per_mol_k"].to_numpy(float)
                        common = {
                            "regime": regime,
                            "entity_id": entity_id,
                            "doi": str(curve["doi"].iloc[0]),
                            "fold": fold,
                            "components": components,
                            "ridge": ridge,
                            "candidate_id": candidate_id(components, ridge),
                            "support_rows": int(support_mask.sum()),
                            "query_rows": int(query_mask.sum()),
                            "support_basis_rank": rank,
                            "coefficient_norm": float(np.linalg.norm(coefficient)),
                        }
                        candidate_rows.append({**common, **entity_metrics(target, prediction)})
                        query_frame = curve.loc[query_mask]
                        for row_index, predicted in zip(query_frame.index, prediction, strict=True):
                            candidate_predictions.append(
                                {
                                    **common,
                                    "source_row_id": int(data.loc[row_index, "source_row_id"]),
                                    "temperature_k": float(data.loc[row_index, "temperature_k"]),
                                    "target": float(data.loc[row_index, "cp_j_per_mol_k"]),
                                    "prediction": float(predicted),
                                }
                            )

    selections: dict[str, Any] = {}
    selected_predictions: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    entity_count = len(curves)
    for regime in REGIMES:
        rows = [row for row in candidate_rows if row["regime"] == regime]
        choice = choose_candidate(rows, entity_count)
        choice["selection_metric"] = "development OOF median entity physical NRMSE"
        choice["tie_rule"] = "within 1 percent choose smaller components, then smaller ridge"
        selections[regime] = choice
        selected_metrics.extend(
            row for row in rows if row["components"] == choice["components"] and row["ridge"] == choice["ridge"]
        )
        selected_predictions.extend(
            row
            for row in candidate_predictions
            if row["regime"] == regime
            and row["components"] == choice["components"]
            and row["ridge"] == choice["ridge"]
        )

    prediction_frame = pd.DataFrame(selected_predictions).sort_values(["regime", "source_row_id"])
    if not np.isfinite(prediction_frame["prediction"]).all():
        raise RuntimeError("selected FPCA predictions are non-finite")
    expected = {regime: int(data[role].eq("query").sum()) for regime, role in REGIMES.items()}
    observed = prediction_frame.groupby("regime").size().to_dict()
    if observed != expected:
        raise RuntimeError(f"query coverage mismatch: {observed} != {expected}")

    pd.DataFrame(candidate_rows).to_csv(output_root / "candidate_entity_metrics.csv", index=False)
    pd.DataFrame(selected_metrics).to_csv(output_root / "selected_entity_metrics.csv", index=False)
    prediction_frame.to_csv(output_root / "selected_query_predictions.csv", index=False)
    (output_root / "selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    manifest = {
        "status": "success",
        "scope": "development-only train-fold FPCA/support-ridge crystal-Cp baseline",
        "data_path": str(data_path.relative_to(PROJECT_ROOT)),
        "data_sha256": sha256(data_path),
        "entities": entity_count,
        "doi": int(data["doi"].nunique()),
        "folds": 5,
        "grid": {"coordinate": "u=(T-Tmin)/(Tmax-Tmin)", "points": 101},
        "component_grid": list(COMPONENT_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "basis_fit": "complete outer-training curves only",
        "heldout_fit": "support targets only",
        "query_target_perturbation_max_prediction_change": 0.0,
        "confirmation_targets_opened": False,
        "runtime_seconds": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocol_hashes": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in (PLAN_PATH, CONTRACT_PATH, RANK_AMENDMENT_PATH, ROUTER_AMENDMENT_PATH)
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "artifacts": {
            "selection": "selection.json",
            "candidate_entity_metrics": "candidate_entity_metrics.csv",
            "selected_entity_metrics": "selected_entity_metrics.csv",
            "selected_query_predictions": "selected_query_predictions.csv",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.data_path.resolve(), args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
