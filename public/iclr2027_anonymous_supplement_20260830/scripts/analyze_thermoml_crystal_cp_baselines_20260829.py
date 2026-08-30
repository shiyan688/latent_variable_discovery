#!/usr/bin/env python3
"""Independently audit and summarize the crystal-Cp CPU baseline package.

The evaluator writes raw point predictions and a provenance manifest.  This
module deliberately recomputes all reported metrics from the raw prediction
table and the materialized development curves; it does not trust the
evaluator's per-entity metrics or result summary.  It is development-only and
has no code path that opens the ThermoML archive or confirmation cohort.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "scripts/evaluate_thermoml_crystal_cp_baselines_20260829.py"
DEFAULT_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_baselines_development_20260829"
PLAN = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
AMENDMENT = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_RANK_AWARE_GIRD_AMENDMENT_20260829.md"
CONTRACT = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
EXPECTED_PROTOCOL_HASHES = {
    PLAN.name: "2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618",
    AMENDMENT.name: "fffb406998900ff38131ee58bd9d98364ea05c3a334bf4b639c0456696c77639",
    CONTRACT.name: "ec37eff5ab2c5847735e4b3d8db4098fd4db2bcbf67792e4b54a4fb8ba43ea15",
}
EXPECTED_COUNTS = {"entities": 247, "dois": 159, "rows": 23_742}
REGIMES = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}
EXPRESSION_FAMILIES = ("constant", "linear_t", "quadratic_t", "cubic_t", "shomate5")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260829
RAW_FILES = {
    "candidate_metrics.csv",
    "expression_coefficients.csv",
    "oof_query_predictions.csv",
    "per_entity_metrics.csv",
    "point_predictions.csv",
    "query_target_perturbation.csv",
    "result.json",
    "selection_path.csv",
    "manifest.json",
}
POINT_COLUMNS = [
    "regime", "entity_id", "doi", "fold", "method", "candidate_id",
    "family", "lambda", "selected", "source_row_id", "position",
    "temperature_k", "cp_j_per_mol_k", "prediction_cp_j_per_mol_k",
]
DATA_COLUMNS = [
    "entity_id", "doi", "fold", "temperature_k", "cp_j_per_mol_k",
    "position", "spread_role", "prefix_role", "four_role",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_id(family: str, ridge: float) -> str:
    return f"{family}__lambda_{ridge:g}"


def _bool_column(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value is True or str(value).strip().lower() == "true")


def load_and_verify_package(root: Path, strict_counts: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Verify provenance and return only independently needed raw tables."""

    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / "manifest.json"
    result_path = root / "result.json"
    manifest = _json(manifest_path)
    result = _json(result_path)
    if manifest.get("scope") != result.get("scope"):
        raise ValueError("manifest/result scope mismatch")
    if result.get("status") != "success":
        raise ValueError("raw result is not successful")
    if result.get("confirmation_targets_opened") is not False:
        raise ValueError("confirmation target access is not false")
    if manifest.get("confirmation_targets_opened") is not False:
        raise ValueError("manifest says confirmation targets were opened")
    if manifest.get("query_targets_used_for_fit") is not False:
        raise ValueError("manifest says query targets entered fitting")
    if manifest.get("runner_sha256") != sha256(RUNNER):
        raise ValueError("runner hash mismatch")
    for name, expected in EXPECTED_PROTOCOL_HASHES.items():
        if manifest.get("protocol_files_sha256", {}).get(name) != expected:
            raise ValueError(f"protocol hash mismatch in manifest: {name}")
        if sha256(PROJECT_ROOT / name) != expected:
            raise ValueError(f"protocol file changed: {name}")
    data_path = PROJECT_ROOT / manifest["data_csv_path"]
    if not data_path.is_file() or manifest.get("data_csv_sha256") != sha256(data_path):
        raise ValueError("development data hash mismatch")
    listed = manifest.get("files", {})
    if set(listed) != RAW_FILES - {"manifest.json"}:
        raise ValueError("raw manifest file inventory mismatch")
    for name, digest in listed.items():
        path = root / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"raw file hash mismatch: {name}")
    if result.get("data_path") != manifest.get("data_csv_path"):
        raise ValueError("result/manifest data path mismatch")

    data_header = pd.read_csv(data_path, nrows=0).columns.tolist()
    data_usecols = DATA_COLUMNS + (["source_row_id"] if "source_row_id" in data_header else [])
    data = pd.read_csv(data_path, usecols=data_usecols)
    if len(data) != EXPECTED_COUNTS["rows"] and strict_counts:
        raise ValueError(f"unexpected development data row count: {len(data)}")
    for column in ("entity_id", "doi"):
        data[column] = data[column].astype(str)
    if "source_row_id" not in data:
        data["source_row_id"] = np.arange(len(data), dtype=np.int64)
    else:
        data["source_row_id"] = pd.to_numeric(data["source_row_id"], errors="raise").astype(np.int64)
    if data["source_row_id"].duplicated().any():
        raise ValueError("development source-row ids are not unique")
    if not np.isfinite(data[["temperature_k", "cp_j_per_mol_k"]].to_numpy(float)).all():
        raise ValueError("development values are non-finite")
    if not data.temperature_k.gt(0).all():
        raise ValueError("development temperatures are not positive")
    entity_count = int(data.entity_id.nunique())
    doi_count = int(data.doi.nunique())
    if strict_counts and (entity_count, doi_count, len(data)) != tuple(EXPECTED_COUNTS.values()):
        raise ValueError("frozen development counts do not match")
    if result.get("entities") != entity_count or result.get("dois") != doi_count or result.get("rows") != len(data):
        raise ValueError("result/data count mismatch")
    if set(result.get("regimes", [])) != set(REGIMES):
        raise ValueError("regime inventory mismatch")

    points = pd.read_csv(root / "point_predictions.csv", usecols=POINT_COLUMNS)
    for column in ("entity_id", "doi", "method", "candidate_id", "regime"):
        points[column] = points[column].astype(str)
    points["selected"] = _bool_column(points["selected"])
    for column in ("cp_j_per_mol_k", "prediction_cp_j_per_mol_k", "temperature_k"):
        points[column] = pd.to_numeric(points[column], errors="raise")
    points["source_row_id"] = pd.to_numeric(points["source_row_id"], errors="raise").astype(np.int64)
    points["fold"] = pd.to_numeric(points["fold"], errors="raise").astype(int)
    if points.duplicated(["regime", "entity_id", "method", "candidate_id", "source_row_id"]).any():
        raise ValueError("duplicate raw point prediction key")
    return data, points, manifest, result


def verify_query_coverage(data: pd.DataFrame, points: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_by_regime: dict[str, set[tuple[str, int]]] = {}
    for regime, role_column in REGIMES.items():
        query = data.loc[data[role_column].eq("query"), ["entity_id", "source_row_id"]]
        expected = set(map(tuple, query.to_records(index=False)))
        expected_by_regime[regime] = expected
        if not expected:
            raise ValueError(f"empty query set: {regime}")
        checks[regime] = {"expected_query_rows": len(expected), "methods": {}}
        subset = points.loc[points.regime.eq(regime)]
        for (method, candidate_id), group in subset.groupby(["method", "candidate_id"], sort=True):
            keys = set(map(tuple, group[["entity_id", "source_row_id"]].to_records(index=False)))
            if not keys <= expected:
                raise ValueError(f"prediction contains non-query rows: {regime}/{method}/{candidate_id}")
            checks[regime]["methods"][f"{method}/{candidate_id}"] = {
                "rows": len(group),
                "entities": int(group.entity_id.nunique()),
                "expected_rows": len(expected),
                "coverage_complete": keys == expected,
                "missing_rows": len(expected - keys),
            }
    return checks


def verify_selection(data: pd.DataFrame, points: pd.DataFrame, result: dict[str, Any]) -> dict[str, Any]:
    selections = result.get("selections", {})
    if set(selections) != set(REGIMES):
        raise ValueError("selection result does not cover all regimes")
    spread = selections["spread"]
    selected_id = _candidate_id(spread["expression_family"], float(spread["expression_lambda"]))
    checks = {"selected_candidate_id": selected_id, "expression_reused": True, "regimes": {}}
    for regime in REGIMES:
        choice = selections[regime]
        current_id = _candidate_id(choice["expression_family"], float(choice["expression_lambda"]))
        same = current_id == selected_id and choice.get("expression_selection_basis_regime") == "spread"
        checks["regimes"][regime] = {
            "reported_candidate_id": current_id,
            "reuses_spread_expression": bool(same),
            "raw_selected_rows": int(points.loc[
                points.regime.eq(regime) & points.method.eq("expression") & points.candidate_id.eq(selected_id) & points.selected,
            ].shape[0]),
        }
        if not same:
            checks["expression_reused"] = False
    if not checks["expression_reused"]:
        raise ValueError("spread expression is not reused by all regimes")
    for regime, role_column in REGIMES.items():
        expected = int(data.loc[data[role_column].eq("query"), "source_row_id"].nunique())
        row = checks["regimes"][regime]
        row["expected_query_rows"] = expected
        row["selected_expression_coverage_complete"] = row["raw_selected_rows"] == expected
        row["coverage_failure"] = None if row["selected_expression_coverage_complete"] else (
            f"selected {selected_id} has {row['raw_selected_rows']}/{expected} valid query rows; no value was imputed"
        )
    return checks


def verify_zero_perturbation(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(root / "query_target_perturbation.csv")
    if frame.empty:
        raise ValueError("missing perturbation audit rows")
    if _bool_column(frame["query_targets_used_for_fit"]).any():
        raise ValueError("query target entered a fit according to perturbation audit")
    prediction_max = float(pd.to_numeric(frame["prediction_max_abs_difference"], errors="raise").max())
    coefficient_values = pd.to_numeric(frame["coefficient_max_abs_difference"], errors="coerce")
    coefficient_max = float(coefficient_values.max()) if coefficient_values.notna().any() else 0.0
    if prediction_max != 0.0 or coefficient_max != 0.0:
        raise ValueError("query-target perturbation is nonzero")
    if float(result.get("query_target_perturbation_max_prediction_difference", np.nan)) != 0.0:
        raise ValueError("result perturbation summary is nonzero")
    if float(result.get("query_target_perturbation_max_coefficient_difference", np.nan)) != 0.0:
        raise ValueError("result coefficient perturbation summary is nonzero")
    return {
        "rows": int(len(frame)),
        "prediction_max_abs_difference": prediction_max,
        "coefficient_max_abs_difference": coefficient_max,
        "query_targets_used_for_fit": False,
        "query_targets_used_for_development_selection": bool(_bool_column(frame["query_targets_used_for_development_selection"]).any()),
    }


def _entity_stats(group: pd.DataFrame) -> pd.DataFrame:
    x = group.copy()
    x["finite"] = np.isfinite(x["prediction_cp_j_per_mol_k"].to_numpy(float)) & np.isfinite(x["cp_j_per_mol_k"].to_numpy(float))
    x = x.loc[x.finite].copy()
    if x.empty:
        return pd.DataFrame(columns=["entity_id", "doi", "n", "sse", "mae_sum", "y_sum", "y2_sum", "physical_r2", "physical_nrmse", "rmse", "mae", "negative_predictions"])
    x["error"] = x.prediction_cp_j_per_mol_k - x.cp_j_per_mol_k
    rows = []
    for (entity_id, doi), g in x.groupby(["entity_id", "doi"], sort=True):
        n = len(g)
        y = g.cp_j_per_mol_k.to_numpy(float)
        e = g.error.to_numpy(float)
        sse = float(np.dot(e, e))
        y_sum = float(y.sum())
        y2_sum = float(np.dot(y, y))
        total = y2_sum - y_sum * y_sum / n
        scale = float(np.std(y))
        rows.append({
            "entity_id": entity_id, "doi": doi, "n": n, "sse": sse,
            "mae_sum": float(np.abs(e).sum()), "y_sum": y_sum, "y2_sum": y2_sum,
            "physical_r2": float(1.0 - sse / total) if total > 0 else np.nan,
            "physical_nrmse": float(np.sqrt(sse / n) / scale) if scale > 0 else np.nan,
            "rmse": float(np.sqrt(sse / n)), "mae": float(np.abs(e).mean()),
            "negative_predictions": int((g.prediction_cp_j_per_mol_k.to_numpy(float) < 0).sum()),
        })
    return pd.DataFrame(rows)


def summarize_predictions(data: pd.DataFrame, points: pd.DataFrame, selection: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    selected_id = selection["selected_candidate_id"]
    expected_rows = {
        regime: int(data.loc[data[role].eq("query"), "source_row_id"].nunique())
        for regime, role in REGIMES.items()
    }
    for (regime, method, candidate_id), group in points.groupby(["regime", "method", "candidate_id"], sort=True):
        stats = _entity_stats(group)
        n_rows = int(np.isfinite(group.prediction_cp_j_per_mol_k.to_numpy(float)).sum())
        target = group.cp_j_per_mol_k.to_numpy(float)
        prediction = group.prediction_cp_j_per_mol_k.to_numpy(float)
        finite = np.isfinite(target) & np.isfinite(prediction)
        y = target[finite]
        yh = prediction[finite]
        sse = float(np.square(yh - y).sum())
        total = float(np.square(y - y.mean()).sum()) if len(y) else np.nan
        pooled_r2 = float(1.0 - sse / total) if total > 0 else np.nan
        q = stats.physical_nrmse.dropna().to_numpy(float)
        selected = bool(method == "expression" and candidate_id == selected_id)
        top = stats.sort_values(["sse", "entity_id"], ascending=[False, True], kind="stable").reset_index(drop=True)
        total_sse = float(stats.sse.sum())
        for rank, item in top.head(20).iterrows():
            top_rows.append({
                "regime": regime, "method": method, "candidate_id": candidate_id,
                "selected_expression": selected, "rank": int(rank + 1),
                "entity_id": item.entity_id, "doi": item.doi, "sse": float(item.sse),
                "fraction_of_sse": float(item.sse / total_sse) if total_sse > 0 else np.nan,
                "cumulative_top_fraction": float(top.head(rank + 1).sse.sum() / total_sse) if total_sse > 0 else np.nan,
            })
        cumulative = [float(top.head(k).sse.sum() / total_sse) if total_sse > 0 else np.nan for k in (1, 3, 5)]
        finite_entities = int(len(q))
        rows.append({
            "regime": regime, "method": method, "candidate_id": candidate_id,
            "selected": selected, "expression_family": group.family.iloc[0],
            "lambda": group["lambda"].iloc[0], "expected_query_rows": expected_rows[regime],
            "prediction_rows": int(len(group)), "finite_prediction_rows": n_rows,
            "coverage_complete": bool(len(group) == expected_rows[regime] and n_rows == expected_rows[regime]),
            "entity_count": int(group.entity_id.nunique()), "finite_entity_count": finite_entities,
            "physical_r2": pooled_r2, "rmse": float(np.sqrt(sse / len(y))) if len(y) else np.nan,
            "mae": float(np.abs(yh - y).mean()) if len(y) else np.nan,
            "median_entity_nrmse": float(np.median(q)) if len(q) else np.nan,
            "p90_entity_nrmse": float(np.quantile(q, .90)) if len(q) else np.nan,
            "p95_entity_nrmse": float(np.quantile(q, .95)) if len(q) else np.nan,
            "max_entity_nrmse": float(np.max(q)) if len(q) else np.nan,
            "entity_r2_ge_0_85_count": int((stats.physical_r2 >= .85).sum()),
            "entity_r2_finite_count": int(stats.physical_r2.notna().sum()),
            "negative_prediction_count": int((prediction[finite] < 0).sum()),
            "negative_prediction_fraction": float((prediction[finite] < 0).mean()) if len(y) else np.nan,
            "sse": sse, "sse_top1_fraction": cumulative[0], "sse_top3_fraction": cumulative[1],
            "sse_top5_fraction": cumulative[2],
        })
    summary = pd.DataFrame(rows)
    return summary, pd.DataFrame(top_rows)


def paired_expression_pchip(summary: pd.DataFrame, points: pd.DataFrame, selection: dict[str, Any]) -> pd.DataFrame:
    selected_id = selection["selected_candidate_id"]
    rows = []
    for regime in REGIMES:
        expr = points.loc[points.regime.eq(regime) & points.method.eq("expression") & points.candidate_id.eq(selected_id), ["entity_id", "doi", "cp_j_per_mol_k", "prediction_cp_j_per_mol_k"]]
        pchip = points.loc[points.regime.eq(regime) & points.method.eq("support_pchip"), ["entity_id", "doi", "cp_j_per_mol_k", "prediction_cp_j_per_mol_k"]]
        e = _entity_stats(expr).rename(columns={"physical_nrmse": "expression_nrmse", "physical_r2": "expression_r2", "sse": "expression_sse"})
        p = _entity_stats(pchip).rename(columns={"physical_nrmse": "pchip_nrmse", "physical_r2": "pchip_r2", "sse": "pchip_sse"})
        merged = e[["entity_id", "doi", "expression_nrmse", "expression_r2", "expression_sse"]].merge(
            p[["entity_id", "pchip_nrmse", "pchip_r2", "pchip_sse"]], on="entity_id", how="outer"
        )
        valid = merged.expression_nrmse.notna() & merged.pchip_nrmse.notna()
        delta = merged.loc[valid, "expression_nrmse"] - merged.loc[valid, "pchip_nrmse"]
        wins = int((delta < 0).sum()); ties = int(np.isclose(delta, 0.0, rtol=0, atol=1e-15).sum()); losses = int((delta > 0).sum())
        rows.append({
            "regime": regime, "expression_candidate_id": selected_id,
            "paired_entities": int(valid.sum()), "expression_wins": wins,
            "ties": ties, "pchip_wins": losses,
            "expression_win_fraction": float(wins / valid.sum()) if valid.sum() else np.nan,
            "median_expression_minus_pchip_nrmse": float(np.median(delta)) if len(delta) else np.nan,
            "mean_expression_minus_pchip_nrmse": float(np.mean(delta)) if len(delta) else np.nan,
            "coverage_failure": int((~valid).sum()),
        })
    return pd.DataFrame(rows)


def _pooled_r2_from_units(units: pd.DataFrame, indices: np.ndarray) -> float:
    selected = units.iloc[indices]
    n = float(selected.n.sum())
    y_sum = float(selected.y_sum.sum())
    total = float(selected.y2_sum.sum() - y_sum * y_sum / n)
    sse = float(selected.sse.sum())
    return float(1.0 - sse / total) if total > 0 else np.nan


def bootstrap_selected_vs_pchip(points: pd.DataFrame, selection: dict[str, Any]) -> dict[str, Any]:
    selected_id = selection["selected_candidate_id"]
    expr = points.loc[points.regime.eq("spread") & points.method.eq("expression") & points.candidate_id.eq(selected_id)]
    pchip = points.loc[points.regime.eq("spread") & points.method.eq("support_pchip")]
    e = _entity_stats(expr).rename(columns={"sse": "expr_sse"})
    p = _entity_stats(pchip).rename(columns={"sse": "pchip_sse"})
    units = e[["entity_id", "doi", "n", "y_sum", "y2_sum", "expr_sse"]].merge(
        p[["entity_id", "pchip_sse"]], on=["entity_id"], how="inner"
    )
    if len(units) != expr.entity_id.nunique() or len(units) == 0:
        raise ValueError("selected spread expression/PCHIP bootstrap coverage incomplete")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    entity_indices = rng.integers(0, len(units), size=(BOOTSTRAP_DRAWS, len(units)))
    n = units.n.to_numpy(float)[entity_indices].sum(axis=1)
    y_sum = units.y_sum.to_numpy(float)[entity_indices].sum(axis=1)
    total = units.y2_sum.to_numpy(float)[entity_indices].sum(axis=1) - y_sum * y_sum / n
    expr_r2 = 1.0 - units.expr_sse.to_numpy(float)[entity_indices].sum(axis=1) / total
    pchip_r2 = 1.0 - units.pchip_sse.to_numpy(float)[entity_indices].sum(axis=1) / total
    entity_delta = expr_r2 - pchip_r2

    doi_units = units.groupby("doi", sort=True)[["n", "y_sum", "y2_sum", "expr_sse", "pchip_sse"]].sum().reset_index()
    doi_indices = rng.integers(0, len(doi_units), size=(BOOTSTRAP_DRAWS, len(doi_units)))
    n_d = doi_units.n.to_numpy(float)[doi_indices].sum(axis=1)
    y_sum_d = doi_units.y_sum.to_numpy(float)[doi_indices].sum(axis=1)
    total_d = doi_units.y2_sum.to_numpy(float)[doi_indices].sum(axis=1) - y_sum_d * y_sum_d / n_d
    expr_d = 1.0 - doi_units.expr_sse.to_numpy(float)[doi_indices].sum(axis=1) / total_d
    pchip_d = 1.0 - doi_units.pchip_sse.to_numpy(float)[doi_indices].sum(axis=1) / total_d
    doi_delta = expr_d - pchip_d

    def describe(a: np.ndarray) -> dict[str, Any]:
        valid = np.isfinite(a)
        return {
            "valid_draws": int(valid.sum()), "invalid_draws": int((~valid).sum()),
            "q025": float(np.quantile(a[valid], .025)) if valid.any() else np.nan,
            "q50": float(np.quantile(a[valid], .50)) if valid.any() else np.nan,
            "q975": float(np.quantile(a[valid], .975)) if valid.any() else np.nan,
        }
    return {
        "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
        "selected_expression_candidate_id": selected_id,
        "entity_units": int(len(units)), "doi_units": int(len(doi_units)),
        "entity_bootstrap": {"expression_r2": describe(expr_r2), "pchip_r2": describe(pchip_r2), "r2_difference_expression_minus_pchip": describe(entity_delta)},
        "doi_bootstrap": {"expression_r2": describe(expr_d), "pchip_r2": describe(pchip_d), "r2_difference_expression_minus_pchip": describe(doi_delta)},
    }


def make_decision(summary: pd.DataFrame, selection: dict[str, Any], coverage: dict[str, Any], perturbation: dict[str, Any], paired: pd.DataFrame, bootstrap: dict[str, Any]) -> dict[str, Any]:
    selected_id = selection["selected_candidate_id"]
    spread = summary.loc[summary.regime.eq("spread") & summary.candidate_id.eq(selected_id) & summary.method.eq("expression")]
    if len(spread) != 1:
        raise ValueError("selected spread expression summary missing")
    row = spread.iloc[0]
    finite_pass = bool(row.finite_prediction_rows == row.expected_query_rows and row.coverage_complete)
    leakage_pass = bool(perturbation["prediction_max_abs_difference"] == 0.0 and perturbation["coefficient_max_abs_difference"] == 0.0 and not perturbation["query_targets_used_for_fit"])
    endpoint = {
        "name": "v2_expression_endpoint",
        "pooled_physical_r2": float(row.physical_r2),
        "r2_threshold": .85, "pooled_r2_pass": bool(row.physical_r2 >= .85),
        "nonconstant_pass": selection["selected_candidate_id"].split("__", 1)[0] != "constant",
        "finite_and_complete_pass": finite_pass,
        "leakage_pass": leakage_pass,
        "status": "PASS" if bool(row.physical_r2 >= .85 and selection["selected_candidate_id"].split("__", 1)[0] != "constant" and finite_pass and leakage_pass) else "FAIL",
        "coverage_note": selection["regimes"]["spread"]["coverage_failure"],
    }
    return {
        "status": "success", "scope": "independent development-only ThermoML crystal-Cp baseline analysis",
        "v2_expression_endpoint": endpoint,
        "selected_spread_expression": selected_id,
        "selection_and_coverage": selection, "query_coverage": coverage,
        "zero_perturbation": perturbation,
        "paired_selected_expression_vs_pchip": paired.to_dict(orient="records"),
        "bootstrap": bootstrap,
        "four_support_selected_expression_coverage_failure": selection["regimes"]["four_support"]["coverage_failure"],
        "no_confirmation_access": True,
    }


def write_report(analysis_root: Path, decision: dict[str, Any], summary: pd.DataFrame, paired: pd.DataFrame, top: pd.DataFrame, bootstrap: dict[str, Any]) -> None:
    selected = decision["selected_spread_expression"]
    table = summary.loc[(summary.selected) | summary.method.isin(["support_pchip", "support_nearest", "support_linear", "support_knn"])]
    lines = [
        "# ThermoML crystal-Cp development baseline analysis",
        "",
        "This report is an independent recomputation from the raw development prediction table. The confirmation cohort was not opened. Absolute physical-unit errors are reported together with entity-normalized tails; incomplete candidates are not imputed.",
        "",
        f"Selected spread expression: `{selected}`. The same expression choice is used for prefix and four-support stress by the frozen protocol.",
        f"v2 expression endpoint: **{decision['v2_expression_endpoint']['status']}** (development OOF pooled physical R² = {decision['v2_expression_endpoint']['pooled_physical_r2']:.6f}; threshold 0.85).",
        "",
        "## Main method table",
        "",
        "| Regime | Method/candidate | Coverage | Pooled R² | RMSE | MAE | Median entity NRMSE | p95 entity NRMSE | Entity R² ≥ .85 | Negative predictions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.sort_values(["regime", "selected", "method", "candidate_id"], ascending=[True, False, True, True]).iterrows():
        def fmt(value: Any) -> str:
            return "NA" if pd.isna(value) else f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
        coverage = f"{int(row.finite_prediction_rows)}/{int(row.expected_query_rows)}"
        lines.append(f"| {row.regime} | {row.method}/{row.candidate_id} | {coverage} | {fmt(row.physical_r2)} | {fmt(row.rmse)} | {fmt(row.mae)} | {fmt(row.median_entity_nrmse)} | {fmt(row.p95_entity_nrmse)} | {int(row.entity_r2_ge_0_85_count)}/{int(row.entity_r2_finite_count)} | {int(row.negative_prediction_count)} |")
    lines += [
        "",
        "The complete candidate-by-regime table is in `method_regime_summary.csv`; it includes every fixed expression/ridge candidate and every local baseline.",
        "",
        "## Selected expression versus PCHIP",
        "",
        "| Regime | Paired entities | Expression wins | Ties | PCHIP wins | Median Δ entity NRMSE (expression − PCHIP) | Coverage failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in paired.iterrows():
        lines.append(f"| {row.regime} | {int(row.paired_entities)} | {int(row.expression_wins)} | {int(row.ties)} | {int(row.pchip_wins)} | {row.median_expression_minus_pchip_nrmse:.6g} | {int(row.coverage_failure)} |")
    lines += [
        "",
        "## Bootstrap",
        "",
        f"The spread selected expression/PCHIP comparison uses {BOOTSTRAP_DRAWS:,} entity and DOI bootstrap draws with fixed seed {BOOTSTRAP_SEED}. Percentile intervals and valid/invalid draw counts are in `bootstrap_selected_expression_vs_pchip.json`.",
        "",
        "## SSE concentration",
        "",
        "Top-20 entity SSE contributors are in `top_sse_contributors.csv`; `sse_top1_fraction`, `sse_top3_fraction`, and `sse_top5_fraction` are in the summary table. These diagnostics expose whether pooled R² is dominated by a small number of curves.",
        "",
        "## Protocol and failure notes",
        "",
        "- Query-target perturbation maximum prediction and coefficient changes are exactly zero; query targets were not used for fitting.",
        f"- Four-support selected `{selected}` coverage is explicitly recorded as: {decision['four_support_selected_expression_coverage_failure'] or 'complete'}.",
        "- No missing selected-expression query prediction is filled, and no confirmation target is accessed.",
    ]
    (analysis_root / "BASELINE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(root: str | Path = DEFAULT_ROOT, *, strict_counts: bool = True, analysis_root: str | Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root).resolve()
    target = Path(analysis_root).resolve() if analysis_root is not None else root / "analysis"
    if target.exists():
        raise FileExistsError(f"analysis output root must be absent: {target}")
    data, points, raw_manifest, raw_result = load_and_verify_package(root, strict_counts=strict_counts)
    coverage = verify_query_coverage(data, points)
    selection = verify_selection(data, points, raw_result)
    perturbation = verify_zero_perturbation(root, raw_result)
    summary, top = summarize_predictions(data, points, selection)
    paired = paired_expression_pchip(summary, points, selection)
    bootstrap = bootstrap_selected_vs_pchip(points, selection)
    decision = make_decision(summary, selection, coverage, perturbation, paired, bootstrap)
    target.mkdir(parents=True, exist_ok=False)
    summary.to_csv(target / "method_regime_summary.csv", index=False)
    paired.to_csv(target / "selected_expression_vs_pchip.csv", index=False)
    top.to_csv(target / "top_sse_contributors.csv", index=False)
    (target / "bootstrap_selected_expression_vs_pchip.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    (target / "verification.json").write_text(json.dumps({"query_coverage": coverage, "selection": selection, "zero_perturbation": perturbation}, indent=2) + "\n", encoding="utf-8")
    decision["runtime_seconds"] = time.perf_counter() - started
    (target / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    write_report(target, decision, summary, paired, top, bootstrap)
    files = {path.name: sha256(path) for path in sorted(target.iterdir()) if path.is_file()}
    analysis_manifest = {
        "scope": decision["scope"], "raw_root": str(root.relative_to(PROJECT_ROOT)) if root.is_relative_to(PROJECT_ROOT) else str(root),
        "raw_manifest_sha256": sha256(root / "manifest.json"), "raw_result_sha256": sha256(root / "result.json"),
        "raw_data_csv_sha256": raw_manifest["data_csv_sha256"], "raw_runner_sha256": raw_manifest["runner_sha256"],
        "raw_files_sha256": raw_manifest["files"], "protocol_files_sha256": EXPECTED_PROTOCOL_HASHES,
        "analyzer_sha256": sha256(Path(__file__).resolve()), "files": files,
        "confirmation_targets_opened": False, "python": sys.version, "platform": platform.platform(),
    }
    (target / "manifest.json").write_text(json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--allow-fixture-counts", action="store_true", help="skip the frozen 247/159/23742 count gate for tests only")
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, strict_counts=not args.allow_fixture_counts, analysis_root=args.analysis_root), indent=2))


if __name__ == "__main__":
    main()
