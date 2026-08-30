#!/usr/bin/env python3
"""Independently verify and aggregate the frozen affine-gauge benchmark."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN = PROJECT_ROOT / "GAUGE_INVARIANT_CANONICAL_RESPONSE_BENCHMARK_PLAN_20260829.md"
RUNNER = PROJECT_ROOT / "scripts/run_gauge_invariant_canonical_response_benchmark_20260829.py"
RUN_ROOT = PROJECT_ROOT / "runs/gauge_invariant_canonical_response_benchmark_20260829"
ANALYSIS_ROOT = RUN_ROOT / "analysis"
FAMILIES = ("polynomial", "relaxation", "thermodynamic_chart")
SEEDS = tuple(range(5))
SEEDED_METHODS = ("raw_decoder", "decoder_functional", "raw_q_ridge_diagnostic")
FIXED_METHODS = ("support_structure_req", "no_q_global_expression")
EXPECTED_PLAN_SHA256 = "ba2a587bd6f7a2945b118c2316ae8f52e0dce9663abfb2fe03f81a084720ada6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return float(1.0 - residual / total)


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("non-finite Spearman statistic")
    return value


def verify_cell(family: str, seed: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell = RUN_ROOT / f"{family}_seed{seed}"
    manifest = json.loads((cell / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((cell / "result.json").read_text(encoding="utf-8"))
    if manifest["plan_sha256"] != EXPECTED_PLAN_SHA256 or sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise ValueError("frozen plan hash mismatch")
    if manifest["runner_sha256"] != sha256(RUNNER):
        raise ValueError("runner hash mismatch")
    for name, expected in manifest["files"].items():
        if sha256(cell / name) != expected:
            raise ValueError(f"artifact hash mismatch: {cell / name}")
    if result["status"] != "success" or result["scientific_selection_eligible"] is not True:
        raise ValueError(f"cell is not a formal success: {family}/seed{seed}")
    if result["epochs"] != 1500 or result["calibration_steps"] != 1200:
        raise ValueError("formal budget mismatch")
    if manifest["cpu_only"] is not True or result["gauge_count"] != 25:
        raise ValueError("device or gauge-count mismatch")
    prediction = pd.read_csv(cell / "query_predictions.csv").assign(family=family, seed=seed)
    coordinate = pd.read_csv(cell / "entity_coordinates.csv").assign(family=family, seed=seed)
    gauge = pd.read_csv(cell / "gauge_diagnostics.csv").assign(family=family, seed=seed)
    if len(coordinate) != 48 or len(gauge) != 25:
        raise ValueError("coordinate or gauge coverage mismatch")
    expected_methods = set(SEEDED_METHODS + FIXED_METHODS)
    if set(prediction["method"]) != expected_methods:
        raise ValueError("method coverage mismatch")
    if prediction.groupby("method").size().to_dict() != {method: 1440 for method in expected_methods}:
        raise ValueError("query-row coverage mismatch")
    if not np.isfinite(prediction[["target", "prediction"]].to_numpy()).all():
        raise ValueError("non-finite prediction")
    return result, prediction, coordinate, gauge


def aggregate_family(family: str, predictions: pd.DataFrame, coordinates: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    keys = ["entity_id", "query_position", "x"]
    seed_zero = predictions[predictions.seed == 0]
    reference_targets = seed_zero[keys + ["target"]].drop_duplicates().sort_values(keys).reset_index(drop=True)
    for seed in SEEDS[1:]:
        current = predictions[predictions.seed == seed][keys + ["target"]].drop_duplicates().sort_values(keys).reset_index(drop=True)
        if not reference_targets.equals(current):
            raise ValueError(f"targets differ across seeds for {family}")

    aggregate_parts = []
    for method in SEEDED_METHODS:
        frame = predictions[predictions.method == method]
        aggregate_parts.append(frame.groupby(keys, as_index=False).agg(target=("target", "first"), prediction=("prediction", "median")).assign(method=method))
    for method in FIXED_METHODS:
        frame = predictions[predictions.method == method]
        spread = frame.groupby(keys)["prediction"].agg(lambda values: float(np.ptp(values))).max()
        if float(spread) > 1e-12:
            raise ValueError(f"fixed method changed across seeds: {family}/{method}")
        aggregate_parts.append(frame[frame.seed == 0][keys + ["target", "prediction"]].assign(method=method))
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    scale = float(aggregate[aggregate.method == "raw_decoder"].target.std(ddof=0))
    summary_rows: list[dict[str, object]] = []
    entity_rows: list[dict[str, object]] = []
    for method, frame in aggregate.groupby("method", sort=True):
        summary_rows.append({
            "family": family,
            "method": method,
            "pooled_r2": r2(frame.target.to_numpy(), frame.prediction.to_numpy()),
            "pooled_nrmse": float(np.sqrt(np.mean((frame.target - frame.prediction) ** 2)) / scale),
        })
        for entity_id, entity in frame.groupby("entity_id", sort=True):
            entity_rows.append({
                "family": family,
                "method": method,
                "entity_id": int(entity_id),
                "r2": r2(entity.target.to_numpy(), entity.prediction.to_numpy()),
                "nrmse": float(np.sqrt(np.mean((entity.target - entity.prediction) ** 2)) / scale),
            })

    coordinate_median = coordinates.groupby("entity_id", as_index=False).median(numeric_only=True)
    coordinate_spearman = {
        f"c{index}": finite_spearman(
            coordinate_median[f"generating_c{index}"].to_numpy(),
            coordinate_median[f"functional_c{index}"].to_numpy(),
        )
        for index in range(3)
    }
    coordinate_spearman["median"] = float(np.median(list(coordinate_spearman.values())))

    geometry_rows: list[dict[str, object]] = []
    for first, second in combinations(SEEDS, 2):
        left = coordinates[coordinates.seed == first].sort_values("entity_id")
        right = coordinates[coordinates.seed == second].sort_values("entity_id")
        for coordinate_type, prefix in (("raw", "raw_q"), ("functional", "functional_c")):
            left_distance = pdist(left[[f"{prefix}{i}" for i in range(3)]].to_numpy())
            right_distance = pdist(right[[f"{prefix}{i}" for i in range(3)]].to_numpy())
            geometry_rows.append({
                "family": family,
                "seed_first": first,
                "seed_second": second,
                "coordinate_type": coordinate_type,
                "distance_spearman": finite_spearman(left_distance, right_distance),
            })
    return summary_rows, entity_rows, coordinate_spearman, geometry_rows


def main() -> None:
    if ANALYSIS_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ANALYSIS_ROOT}")
    all_results: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coordinate_frames: list[pd.DataFrame] = []
    gauge_frames: list[pd.DataFrame] = []
    for family in FAMILIES:
        for seed in SEEDS:
            result, prediction, coordinate, gauge = verify_cell(family, seed)
            all_results.append(result)
            prediction_frames.append(prediction)
            coordinate_frames.append(coordinate)
            gauge_frames.append(gauge)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    gauges = pd.concat(gauge_frames, ignore_index=True)
    summaries: list[dict[str, object]] = []
    entities: list[dict[str, object]] = []
    geometries: list[dict[str, object]] = []
    coordinate_summary: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        family_summary, family_entities, correlations, family_geometry = aggregate_family(
            family,
            predictions[predictions.family == family],
            coordinates[coordinates.family == family],
        )
        summaries.extend(family_summary)
        entities.extend(family_entities)
        geometries.extend(family_geometry)
        coordinate_summary[family] = correlations

    summary_frame = pd.DataFrame(summaries)
    functional_r2 = summary_frame[summary_frame.method == "decoder_functional"].set_index("family")["pooled_r2"].to_dict()
    structure_r2 = summary_frame[summary_frame.method == "support_structure_req"].set_index("family")["pooled_r2"].to_dict()
    gates = {
        "all_15_cells_formal_success": len(all_results) == 15,
        "maximum_gauge_prediction_change_at_most_1e_5": float(gauges.prediction_max_abs_change.max()) <= 1e-5,
        "maximum_gauge_functional_coefficient_change_at_most_1e_5": float(gauges.functional_coefficient_max_abs_change.max()) <= 1e-5,
        "all_functional_family_r2_at_least_0_85": all(value >= 0.85 for value in functional_r2.values()),
        "all_structure_family_r2_at_least_0_85": all(value >= 0.85 for value in structure_r2.values()),
        "all_functional_generating_median_spearman_at_least_0_90": all(value["median"] >= 0.90 for value in coordinate_summary.values()),
        "exact_query_target_invariance": max(float(result["query_target_input_max_difference"]) for result in all_results) == 0.0,
    }
    decision = {
        "scope": "controlled gauge-invariant canonical response benchmark",
        "primary_gates": gates,
        "benchmark_passed": all(gates.values()),
        "functional_family_pooled_r2": functional_r2,
        "structure_family_pooled_r2": structure_r2,
        "functional_vs_generating_coordinate_spearman": coordinate_summary,
        "maximum_gauge_prediction_change": float(gauges.prediction_max_abs_change.max()),
        "maximum_gauge_functional_coefficient_change": float(gauges.functional_coefficient_max_abs_change.max()),
        "maximum_gauge_raw_q_coordinate_change": float(gauges.q_coordinate_max_abs_change.max()),
        "maximum_frozen_raw_q_readout_prediction_change": float(gauges.raw_q_ridge_prediction_max_abs_change.max()),
        "query_target_input_max_difference": max(float(result["query_target_input_max_difference"]) for result in all_results),
        "predictive_superiority_inferred": False,
        "unique_or_causal_latent_recovery_inferred": False,
    }

    ANALYSIS_ROOT.mkdir(parents=True)
    summary_frame.to_csv(ANALYSIS_ROOT / "family_summary.csv", index=False)
    pd.DataFrame(entities).to_csv(ANALYSIS_ROOT / "per_entity_metrics.csv", index=False)
    pd.DataFrame(geometries).to_csv(ANALYSIS_ROOT / "cross_seed_geometry.csv", index=False)
    pd.DataFrame([
        {"family": family, **values} for family, values in coordinate_summary.items()
    ]).to_csv(ANALYSIS_ROOT / "coordinate_recovery.csv", index=False)
    write_json(ANALYSIS_ROOT / "decision.json", decision)
    manifest = {
        "plan_sha256": sha256(PLAN),
        "runner_sha256": sha256(RUNNER),
        "analyzer_sha256": sha256(Path(__file__)),
        "files": {},
    }
    for path in sorted(ANALYSIS_ROOT.iterdir()):
        if path.name != "manifest.json":
            manifest["files"][path.name] = sha256(path)
    write_json(ANALYSIS_ROOT / "manifest.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
