#!/usr/bin/env python3
"""Independently analyze the 25 crystal-Cp no-q temperature-MLP cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CACHE = PROJECT_ROOT / "runs/_runtime_cache"
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUNTIME_CACHE / "xdg"))

import numpy as np
import pandas as pd
import torch

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.backends.torch_mlp import build_torch_model_factory


PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
CONTRACT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
DATA_MANIFEST_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/manifest.json"
RUNNER_PATH = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_no_q_temperature_mlp_20260829.py"
RAW_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_no_q_temperature_mlp_development_20260829"
DEFAULT_OUTPUT_ROOT = RAW_ROOT / "analysis"

EXPECTED_PLAN_SHA256 = "2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618"
EXPECTED_CONTRACT_SHA256 = "ec37eff5ab2c5847735e4b3d8db4098fd4db2bcbf67792e4b54a4fb8ba43ea15"
EXPECTED_DATA_SHA256 = "f73d3c676932304c8e5c21e79e7bc9c678e20c84db8d60b59a8e60feee400e4e"
EXPECTED_DATA_MANIFEST_SHA256 = "d88172109a7a244195f13dfa01516cfe6cb16cd038ca797dfd61d2d50694208a"
EXPECTED_RUNNER_SHA256 = "5b2d919be05c97fccc8c83474970dd5fa934ca45a1fe21a589fc9b08f3c2db0b"
EXPECTED_ENTITIES = 247
EXPECTED_DOIS = 159
EXPECTED_ROWS = 23_742
EXPECTED_QUERY_ROWS = {"spread": 17_704, "prefix": 17_897, "four_support": 22_754}
REGIME_ROLES = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}
EXPECTED_EPOCHS = 1000
EXPECTED_DEVICE = "cuda"
EXPECTED_SELECTION_ELIGIBLE = True
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260829
CHECKPOINT_RECOMPUTE_STANDARDIZED_RTOL = 1e-6
CHECKPOINT_RECOMPUTE_STANDARDIZED_ATOL = 5e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    return 1.0 - float(np.square(target - prediction).sum()) / denominator if denominator else np.nan


def _bootstrap(frame: pd.DataFrame, unit: str, draws: int, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for unit_id, group in frame.groupby(unit, sort=True):
        target = group["cp_j_per_mol_k"].to_numpy(float)
        prediction = group["prediction_cp_j_per_mol_k"].to_numpy(float)
        rows.append((str(unit_id), len(group), float(target.sum()), float(np.square(target).sum()), float(np.square(target - prediction).sum())))
    aggregates = pd.DataFrame(rows, columns=["unit_id", "n", "sum_y", "sum_y2", "sse"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(aggregates), size=(draws, len(aggregates)))
    totals = aggregates[["n", "sum_y", "sum_y2", "sse"]].to_numpy(float)[sampled].sum(axis=1)
    denominator = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
    values = 1.0 - totals[:, 3] / denominator
    valid = np.isfinite(values) & (denominator > 0.0)
    result = pd.DataFrame({"unit": unit, "draw": np.arange(draws), "physical_r2": values, "valid": valid})
    summary = {"unit": unit, "seed": seed, "draws": draws, "valid_draws": int(valid.sum()), "invalid_draws": int((~valid).sum()), "percentile_2_5": float(np.percentile(values[valid], 2.5)), "percentile_97_5": float(np.percentile(values[valid], 97.5))}
    return result, summary


def _metric_rows(median: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, entity_id), group in median.groupby(["regime", "entity_id"], sort=True):
        target = group["cp_j_per_mol_k"].to_numpy(float)
        prediction = group["prediction_cp_j_per_mol_k"].to_numpy(float)
        error = prediction - target
        scale = float(np.std(target))
        rows.append({"regime": regime, "entity_id": entity_id, "doi": str(group["doi"].iloc[0]), "query_rows": len(group), "physical_r2": _r2(target, prediction), "physical_nrmse": float(np.sqrt(np.mean(error**2)) / scale) if scale else np.nan, "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))), "maximum_absolute_error": float(np.max(np.abs(error))), "negative_prediction_count": int(np.count_nonzero(prediction < 0.0))})
    return pd.DataFrame(rows)


def _strata(data: pd.DataFrame, median: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored_frames = []
    for regime, role_column in REGIME_ROLES.items():
        bounds = data.loc[data[role_column].eq("support")].groupby("entity_id")["temperature_k"].agg(["min", "max"])
        frame = median.loc[median["regime"].eq(regime)].join(bounds, on="entity_id")
        frame["temperature_stratum"] = np.where(frame["temperature_k"] < frame["min"], "below_support", np.where(frame["temperature_k"] > frame["max"], "above_support", "within_support_range"))
        scored_frames.append(frame)
    scored = pd.concat(scored_frames, ignore_index=True)
    rows = []
    for (regime, name), group in scored.groupby(["regime", "temperature_stratum"], sort=True):
        target, prediction = group["cp_j_per_mol_k"].to_numpy(float), group["prediction_cp_j_per_mol_k"].to_numpy(float)
        error = prediction - target
        rows.append({"regime": regime, "temperature_stratum": name, "query_rows": len(group), "physical_r2": _r2(target, prediction), "rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))), "p95_absolute_error": float(np.percentile(np.abs(error), 95)), "maximum_absolute_error": float(np.max(np.abs(error)))})
    return scored.drop(columns=["min", "max"]), pd.DataFrame(rows)


def analyze(raw_root: str | Path = RAW_ROOT, data_path: str | Path = DATA_PATH, output_root: str | Path = DEFAULT_OUTPUT_ROOT, bootstrap_draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    raw_root, data_path, output_root = Path(raw_root).resolve(), Path(data_path).resolve(), Path(output_root).resolve()
    require(not output_root.exists(), f"analysis output root must be absent: {output_root}")
    require(data_path == DATA_PATH.resolve(), "analyzer is bound to sealed development data")
    for path, expected in ((PLAN_PATH, EXPECTED_PLAN_SHA256), (CONTRACT_PATH, EXPECTED_CONTRACT_SHA256), (DATA_PATH, EXPECTED_DATA_SHA256), (DATA_MANIFEST_PATH, EXPECTED_DATA_MANIFEST_SHA256), (RUNNER_PATH, EXPECTED_RUNNER_SHA256)):
        require(path.is_file() and sha256(path) == expected, f"analyzer input binding mismatch: {path}")
    data = pd.read_csv(data_path).sort_values(["entity_id", "position"], kind="stable").reset_index(drop=True)
    require(len(data) == EXPECTED_ROWS and data["entity_id"].nunique() == EXPECTED_ENTITIES and data["doi"].nunique() == EXPECTED_DOIS, "sealed data coverage mismatch")
    all_predictions, resources, input_hashes = [], [], {}
    devices = set()
    maximum_recomputed_prediction_difference = 0.0
    maximum_recomputed_standardized_prediction_difference = 0.0
    for fold in FOLDS:
        train = data.loc[data["fold"].ne(fold)]
        heldout = data.loc[data["fold"].eq(fold)].sort_values("source_row_id", kind="stable")
        heldout_queries = {
            regime: heldout.loc[heldout[role_column].eq("query")].sort_values("source_row_id", kind="stable")
            for regime, role_column in REGIME_ROLES.items()
        }
        for seed in SEEDS:
            cell = raw_root / f"fold{fold}_seed{seed}"
            manifest_path, summary_path = cell / "manifest.json", cell / "cell_summary.json"
            required_paths = [manifest_path, summary_path, cell / "query_predictions.csv", cell / "training_history.csv", cell / "checkpoint.pt", cell / "terminal_ledger.json"] + [cell / f"query_target_perturbed_{regime}.csv" for regime in REGIME_ROLES]
            for path in required_paths:
                require(path.is_file(), f"missing cell artifact: {path}")
                input_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
            manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
            require(manifest["runner_sha256"] == EXPECTED_RUNNER_SHA256 and manifest["plan_sha256"] == EXPECTED_PLAN_SHA256 and manifest["execution_contract_sha256"] == EXPECTED_CONTRACT_SHA256, f"cell binding mismatch: {cell}")
            require(manifest["scientific_selection_eligible"] is EXPECTED_SELECTION_ELIGIBLE, f"cell eligibility mismatch: {cell}")
            require(not any(manifest[key] for key in ("entity_id_input", "support_input", "q_input", "query_target_input", "confirmation_targets_opened")), f"no-q input boundary mismatch: {cell}")
            require(manifest["normalizer_fit_scope"] == "outer_training_rows_only" and manifest["deterministic_epoch_full_training_row_pass"], f"training scope mismatch: {cell}")
            require(manifest["device"] == EXPECTED_DEVICE and summary["device"] == EXPECTED_DEVICE, f"device mismatch: {cell}")
            devices.add(manifest["device"])
            for name, expected_hash in manifest["files"].items():
                require(sha256(cell / name) == expected_hash, f"cell artifact hash mismatch: {cell / name}")
            history = pd.read_csv(cell / "training_history.csv")
            expected_updates = EXPECTED_EPOCHS * int(np.ceil(len(train) / 256))
            require(len(history) == EXPECTED_EPOCHS and history["rows_visited"].eq(len(train)).all() and int(history["updates_completed"].iloc[-1]) == expected_updates, f"history/update mismatch: {cell}")
            require(summary["status"] == "success" and summary["epochs_completed"] == EXPECTED_EPOCHS and summary["optimizer_updates"] == expected_updates and summary["backward_calls"] == expected_updates, f"terminal budget mismatch: {cell}")
            require(summary["query_target_perturbation_max_prediction_difference_by_regime"] == {regime: 0.0 for regime in REGIME_ROLES}, f"query perturbation changed prediction: {cell}")
            checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            require(tuple(checkpoint["hidden_widths"]) == (256, 128) and tuple(checkpoint["input_features"]) == ("temperature_k",), f"checkpoint architecture/input mismatch: {cell}")
            require(np.isclose(checkpoint["temperature_mean"], train["temperature_k"].mean()) and np.isclose(checkpoint["temperature_std"], train["temperature_k"].std(ddof=0)) and np.isclose(checkpoint["target_mean"], train["cp_j_per_mol_k"].mean()) and np.isclose(checkpoint["target_std"], train["cp_j_per_mol_k"].std(ddof=0)), f"outer-training normalizer mismatch: {cell}")
            predictions = pd.read_csv(cell / "query_predictions.csv").sort_values(["regime", "source_row_id"], kind="stable")
            require(set(predictions["regime"]) == set(REGIME_ROLES), f"regime coverage mismatch: {cell}")
            for regime, expected_query in heldout_queries.items():
                observed = predictions.loc[predictions["regime"].eq(regime)]
                require(np.array_equal(observed["source_row_id"].to_numpy(int), expected_query["source_row_id"].to_numpy(int)), f"{regime} heldout query coverage mismatch: {cell}")
            model = build_torch_model_factory((256, 128))(1)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            features = torch.tensor(((heldout["temperature_k"].to_numpy(np.float32) - checkpoint["temperature_mean"]) / checkpoint["temperature_std"])[:, None])
            with torch.no_grad():
                recomputed_standardized_all = model(features).squeeze(1).numpy().astype(float)
            recomputed_all = recomputed_standardized_all * checkpoint["target_std"] + checkpoint["target_mean"]
            recomputed_by_row = pd.Series(recomputed_all, index=heldout["source_row_id"].to_numpy(int))
            recomputed_standardized_by_row = pd.Series(recomputed_standardized_all, index=heldout["source_row_id"].to_numpy(int))
            original = data.set_index("source_row_id").sort_index()
            for regime, role_column in REGIME_ROLES.items():
                observed = predictions.loc[predictions["regime"].eq(regime)]
                recomputed = recomputed_by_row.loc[observed["source_row_id"].to_numpy(int)].to_numpy(float)
                observed_prediction = observed["prediction_cp_j_per_mol_k"].to_numpy(float)
                difference = float(np.max(np.abs(recomputed - observed_prediction)))
                maximum_recomputed_prediction_difference = max(maximum_recomputed_prediction_difference, difference)
                recomputed_standardized = recomputed_standardized_by_row.loc[observed["source_row_id"].to_numpy(int)].to_numpy(float)
                observed_standardized = (observed_prediction - checkpoint["target_mean"]) / checkpoint["target_std"]
                standardized_difference = float(np.max(np.abs(recomputed_standardized - observed_standardized)))
                maximum_recomputed_standardized_prediction_difference = max(
                    maximum_recomputed_standardized_prediction_difference,
                    standardized_difference,
                )
                require(
                    np.allclose(
                        recomputed_standardized,
                        observed_standardized,
                        rtol=CHECKPOINT_RECOMPUTE_STANDARDIZED_RTOL,
                        atol=CHECKPOINT_RECOMPUTE_STANDARDIZED_ATOL,
                    ),
                    f"{regime} checkpoint prediction mismatch: {cell}",
                )
                copied = pd.read_csv(cell / f"query_target_perturbed_{regime}.csv").set_index("source_row_id").sort_index()
                delta = copied["cp_j_per_mol_k"] - original["cp_j_per_mol_k"]
                mask = copied["fold"].eq(fold) & copied[role_column].eq("query")
                require(np.allclose(delta[mask], 1_000_000.0) and np.allclose(delta[~mask], 0.0), f"{regime} query perturbation copy mismatch: {cell}")
            predictions["seed"] = seed
            all_predictions.append(predictions)
            resources.append({"fold": fold, "seed": seed, "device": summary["device"], "training_rows": summary["training_rows"], "query_rows_by_regime": json.dumps(summary["query_rows_by_regime"], sort_keys=True), "epochs": summary["epochs_completed"], "optimizer_updates": summary["optimizer_updates"], "backward_calls": summary["backward_calls"], "training_seconds": summary["training_seconds"], "peak_gpu_memory_allocated_bytes": summary["peak_gpu_memory_allocated_bytes"], "peak_gpu_memory_reserved_bytes": summary["peak_gpu_memory_reserved_bytes"], "peak_process_memory_bytes": summary["peak_process_memory_bytes"]})
    require(devices == {EXPECTED_DEVICE}, "mixed execution devices")
    predictions = pd.concat(all_predictions, ignore_index=True)
    counts = predictions.groupby(["regime", "source_row_id"])["seed"].nunique()
    observed_query_rows = counts.groupby("regime").size().to_dict()
    require(observed_query_rows == EXPECTED_QUERY_ROWS and counts.eq(5).all(), "each regime/source query must have exactly five seed predictions")
    identity = ["regime", "source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k"]
    median = predictions.groupby(identity, as_index=False)["prediction_cp_j_per_mol_k"].median().sort_values(["regime", "source_row_id"], kind="stable")
    require(median.groupby("regime").size().to_dict() == EXPECTED_QUERY_ROWS, "per-regime median OOF coverage mismatch")
    scored, stratum_metrics = _strata(data, median)
    entity_metrics = _metric_rows(median)
    bootstrap_frames = []
    bootstrap_summary = {}
    metrics_by_regime = {}
    for regime_index, regime in enumerate(REGIME_ROLES):
        frame = median.loc[median["regime"].eq(regime)]
        target, prediction = frame["cp_j_per_mol_k"].to_numpy(float), frame["prediction_cp_j_per_mol_k"].to_numpy(float)
        error = prediction - target
        regime_entities = entity_metrics.loc[entity_metrics["regime"].eq(regime)]
        entity_bootstrap, entity_summary = _bootstrap(frame, "entity_id", bootstrap_draws, BOOTSTRAP_SEED + 10 * regime_index)
        doi_bootstrap, doi_summary = _bootstrap(frame, "doi", bootstrap_draws, BOOTSTRAP_SEED + 10 * regime_index + 1)
        entity_bootstrap.insert(0, "regime", regime)
        doi_bootstrap.insert(0, "regime", regime)
        bootstrap_frames.extend([entity_bootstrap, doi_bootstrap])
        bootstrap_summary[regime] = {"entity": entity_summary, "doi": doi_summary}
        metrics_by_regime[regime] = {
            "query_rows": len(frame), "pooled_physical_r2": _r2(target, prediction),
            "pooled_rmse": float(np.sqrt(np.mean(error**2))), "pooled_mae": float(np.mean(np.abs(error))),
            "median_entity_physical_r2": float(regime_entities["physical_r2"].median()),
            "median_entity_physical_nrmse": float(regime_entities["physical_nrmse"].median()),
            "p90_entity_physical_nrmse": float(np.percentile(regime_entities["physical_nrmse"].dropna(), 90)),
            "p95_entity_physical_nrmse": float(np.percentile(regime_entities["physical_nrmse"].dropna(), 95)),
            "maximum_entity_physical_nrmse": float(regime_entities["physical_nrmse"].max()),
            "negative_prediction_count": int(np.count_nonzero(prediction < 0.0)),
            "entity_bootstrap_physical_r2": entity_summary, "doi_bootstrap_physical_r2": doi_summary,
        }
    gates = {"exact_entity_coverage_each_regime": bool(entity_metrics.groupby("regime")["entity_id"].nunique().eq(EXPECTED_ENTITIES).all()), "exact_query_coverage_each_regime": observed_query_rows == EXPECTED_QUERY_ROWS, "five_seed_pointwise_median_by_regime_and_source_row": bool(counts.eq(5).all()), "finite_predictions": bool(np.isfinite(median["prediction_cp_j_per_mol_k"]).all()), "query_target_perturbation_invariant_each_regime": True, "all_cells_terminal": len(resources) == 25}
    decision = {
        "scope": "crystal-Cp no_q_temperature_mlp development-only OOF",
        "gates": gates, "passed": bool(all(gates.values())), "device": EXPECTED_DEVICE,
        "query_rows_by_regime": observed_query_rows, "metrics_by_regime": metrics_by_regime,
        "maximum_checkpoint_prediction_abs_difference": maximum_recomputed_prediction_difference,
        "maximum_checkpoint_standardized_prediction_abs_difference": maximum_recomputed_standardized_prediction_difference,
        "checkpoint_recompute_tolerance": {
            "standardized_rtol": CHECKPOINT_RECOMPUTE_STANDARDIZED_RTOL,
            "standardized_atol": CHECKPOINT_RECOMPUTE_STANDARDIZED_ATOL,
            "comparison": "outer-train-standardized CUDA-saved float32 predictions versus CPU checkpoint replay",
        },
        "raw_q_metrics_applicable": False, "confirmation_targets_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output_root / "all_seed_query_predictions.csv", index=False)
    scored.to_csv(output_root / "median_query_predictions.csv", index=False)
    entity_metrics.to_csv(output_root / "entity_metrics.csv", index=False)
    stratum_metrics.to_csv(output_root / "temperature_stratum_metrics.csv", index=False)
    pd.DataFrame(resources).to_csv(output_root / "cell_resources.csv", index=False)
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(output_root / "bootstrap_r2.csv", index=False)
    (output_root / "bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2) + "\n", encoding="utf-8")
    (output_root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    report_lines = ["# Crystal-Cp no-q temperature MLP development analysis", "", "- Cells: `5 DOI folds × 5 seeds = 25`; pointwise median is computed independently by `(regime, source_row_id)`.", f"- Exact OOF query coverage: `{observed_query_rows}`."]
    for regime in REGIME_ROLES:
        values = metrics_by_regime[regime]
        report_lines.append(f"- `{regime}`: pooled physical R² `{values['pooled_physical_r2']:.6f}`, RMSE `{values['pooled_rmse']:.6f}`, MAE `{values['pooled_mae']:.6f}`.")
    report_lines.extend(["- The model receives temperature only: no entity ID, support, q, or query target.", "- Confirmation targets opened: **no**.", ""])
    report = "\n".join(report_lines)
    (output_root / "NO_Q_TEMPERATURE_MLP_ANALYSIS.md").write_text(report, encoding="utf-8")
    files = {path.name: sha256(path) for path in output_root.iterdir() if path.is_file() and path.name != "manifest.json"}
    manifest = {"scope": decision["scope"], "runner_sha256": EXPECTED_RUNNER_SHA256, "analyzer_sha256": sha256(Path(__file__).resolve()), "plan_sha256": EXPECTED_PLAN_SHA256, "execution_contract_sha256": EXPECTED_CONTRACT_SHA256, "data_sha256": EXPECTED_DATA_SHA256, "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256, "bootstrap_draws": bootstrap_draws, "bootstrap_seed_entity": BOOTSTRAP_SEED, "bootstrap_seed_doi": BOOTSTRAP_SEED + 1, "checkpoint_recompute_tolerance": decision["checkpoint_recompute_tolerance"], "input_hashes": input_hashes, "files": files, "confirmation_targets_opened": False, "python": sys.version, "platform": platform.platform()}
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
