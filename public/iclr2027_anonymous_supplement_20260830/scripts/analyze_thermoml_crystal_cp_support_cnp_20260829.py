#!/usr/bin/env python3
"""Independently analyze the 25 development support-CNP cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_thermoml_crystal_cp_support_cnp_20260829 import (  # noqa: E402
    CONTRACT_PATH,
    DATA_MANIFEST_PATH,
    DATA_PATH,
    ENCODER_WIDTHS,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_DATA_MANIFEST_SHA256,
    EXPECTED_DATA_SHA256,
    EXPECTED_DOIS,
    EXPECTED_ENTITIES,
    EXPECTED_PLAN_SHA256,
    EXPECTED_ROWS,
    FORMAL_ROOT,
    QUERY_HEAD_WIDTHS,
    REGIMES,
    SupportCNP,
    sha256,
)

PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
RUNNER_PATH = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_support_cnp_20260829.py"
DEFAULT_OUTPUT_ROOT = FORMAL_ROOT / "analysis"
EXPECTED_QUERY_ROWS = {"spread": 17_704, "prefix": 17_897, "four_support": 22_754}
ROLE_COLUMNS = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20260829
CHECKPOINT_REPLAY_RTOL, CHECKPOINT_REPLAY_ATOL = 1e-6, 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _checkpoint_replay_differences(
    recomputed_standardized: np.ndarray,
    expected_physical: np.ndarray,
    stats: dict[str, float],
) -> tuple[float, float, bool]:
    expected_standardized = (
        np.asarray(expected_physical, dtype=float) - stats["target_mean"]
    ) / stats["target_std"]
    recomputed_standardized = np.asarray(recomputed_standardized, dtype=float)
    standardized_difference = float(
        np.max(np.abs(recomputed_standardized - expected_standardized))
    )
    physical_difference = float(
        np.max(
            np.abs(
                recomputed_standardized * stats["target_std"]
                + stats["target_mean"]
                - expected_physical
            )
        )
    )
    passed = bool(
        np.allclose(
            recomputed_standardized,
            expected_standardized,
            rtol=CHECKPOINT_REPLAY_RTOL,
            atol=CHECKPOINT_REPLAY_ATOL,
        )
    )
    return standardized_difference, physical_difference, passed


def _same_integer_ids(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            np.sort(np.asarray(left, dtype=int)),
            np.sort(np.asarray(right, dtype=int)),
        )
    )


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    return 1.0 - float(np.square(target - prediction).sum()) / denominator if denominator else float("nan")


def _bootstrap(frame: pd.DataFrame, unit: str, seed: int) -> dict[str, Any]:
    pieces = []
    for unit_id, group in frame.groupby(unit, sort=True):
        y = group.cp_j_per_mol_k.to_numpy(float)
        p = group.prediction_cp_j_per_mol_k.to_numpy(float)
        pieces.append((str(unit_id), len(y), float(y.sum()), float(np.square(y).sum()), float(np.square(y - p).sum())))
    table = np.asarray([item[1:] for item in pieces], dtype=float)
    sampled = np.random.default_rng(seed).integers(0, len(table), size=(BOOTSTRAP_DRAWS, len(table)))
    totals = table[sampled].sum(axis=1)
    denominator = totals[:, 2] - totals[:, 1] ** 2 / totals[:, 0]
    values = 1.0 - totals[:, 3] / denominator
    valid = np.isfinite(values) & (denominator > 0)
    return {"unit": unit, "seed": seed, "draws": BOOTSTRAP_DRAWS, "valid_draws": int(valid.sum()), "invalid_draws": int((~valid).sum()), "percentile_2_5": float(np.percentile(values[valid], 2.5)), "percentile_97_5": float(np.percentile(values[valid], 97.5))}


def _metric(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, entity), group in frame.groupby(["regime", "entity_id"], sort=True):
        y, p = group.cp_j_per_mol_k.to_numpy(float), group.prediction_cp_j_per_mol_k.to_numpy(float)
        scale = float(np.std(y))
        rows.append({"regime": regime, "entity_id": entity, "doi": str(group.doi.iloc[0]), "query_rows": len(group), "physical_r2": _r2(y, p), "physical_nrmse": float(np.sqrt(np.mean((p - y) ** 2)) / scale) if scale else np.nan, "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "mae": float(np.mean(np.abs(p - y))), "p95_absolute_error": float(np.percentile(np.abs(p - y), 95)), "maximum_absolute_error": float(np.max(np.abs(p - y))), "negative_prediction_count": int(np.count_nonzero(p < 0))})
    return pd.DataFrame(rows)


def _stability(frame: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, entity, source), group in frame.groupby(["regime", "entity_id", "source_row_id"], sort=True):
        values = group.prediction_cp_j_per_mol_k.to_numpy(float)
        rows.append({"regime": regime, "entity_id": entity, "source_row_id": int(source), "seed_prediction_std": float(np.std(values)), "seed_prediction_range": float(np.ptp(values))})
    result = pd.DataFrame(rows)
    by_entity = []
    for (regime, entity), group in frame.groupby(["regime", "entity_id"], sort=True):
        pivot = group.pivot(index="source_row_id", columns="seed", values="prediction_cp_j_per_mol_k").sort_index()
        correlations = []
        for left in pivot.columns:
            for right in pivot.columns:
                if int(left) < int(right):
                    correlations.append(float(pivot[left].corr(pivot[right], method="spearman")))
        by_entity.append({"regime": regime, "entity_id": entity, "seed_pair_spearman_median": float(np.nanmedian(correlations)), "seed_pair_count": len(correlations)})
    offset_rows = []
    for (entity, seed), group in offsets.groupby(["entity_id", "seed"], sort=True):
        pivot = group.pivot(index="source_row_id", columns="spread_offset", values="prediction_cp_j_per_mol_k").sort_index()
        require(set(pivot.columns) == {0, 1, 2, 3}, f"offset stability lacks all four offsets for {entity} seed {seed}")
        require(pivot.notna().all().all(), f"offset stability has missing common points for {entity} seed {seed}")
        for left in range(4):
            for right in range(left + 1, 4):
                difference = pivot[left].to_numpy(float) - pivot[right].to_numpy(float)
                offset_rows.append({"kind": "within_seed_offset_pair", "entity_id": entity, "seed": int(seed), "offset_left": left, "offset_right": right, "common_query_projection_rows": len(pivot), "spearman": float(pivot[left].corr(pivot[right], method="spearman")), "rmse": float(np.sqrt(np.mean(difference ** 2))), "mean_absolute_difference": float(np.mean(np.abs(difference))), "prediction_range": float(np.ptp(np.concatenate((pivot[left].to_numpy(float), pivot[right].to_numpy(float)))))})
    result["kind"] = "across_seed_point"
    return pd.concat([result, pd.DataFrame(by_entity), pd.DataFrame(offset_rows)], ignore_index=True, sort=False)


def analyze(raw_root: str | Path = FORMAL_ROOT, data_path: str | Path = DATA_PATH, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    raw_root, data_path, output_root = Path(raw_root).resolve(), Path(data_path).resolve(), Path(output_root).resolve()
    require(not output_root.exists(), f"analysis output root must be absent: {output_root}")
    require(data_path == DATA_PATH.resolve(), "analyzer is bound to sealed development data")
    for path, expected in ((PLAN_PATH, EXPECTED_PLAN_SHA256), (CONTRACT_PATH, EXPECTED_CONTRACT_SHA256), (DATA_PATH, EXPECTED_DATA_SHA256), (DATA_MANIFEST_PATH, EXPECTED_DATA_MANIFEST_SHA256)):
        require(path.is_file() and sha256(path) == expected, f"sealed input mismatch: {path}")
    data = pd.read_csv(data_path).sort_values(["entity_id", "position"], kind="stable").reset_index(drop=True)
    require((len(data), data.entity_id.nunique(), data.doi.nunique()) == (EXPECTED_ROWS, EXPECTED_ENTITIES, EXPECTED_DOIS), "data coverage mismatch")
    all_rows, regime_rows, all_offset_rows, resources = [], [], [], []
    checkpoint_replay_standardized_max = 0.0
    checkpoint_replay_physical_max = 0.0
    runner_hash = sha256(RUNNER_PATH)
    for fold in range(5):
        heldout = data.loc[data.fold.eq(fold)]
        for seed in range(5):
            cell = raw_root / f"fold{fold}_seed{seed}"
            for name in ("manifest.json", "cell_summary.json", "terminal_ledger.json", "query_predictions.csv", "spread_offset_predictions.csv", "training_history.csv", "checkpoint.pt", "query_target_perturbed_spread.csv", "query_target_perturbed_prefix.csv", "query_target_perturbed_four_support.csv"):
                require((cell / name).is_file(), f"missing artifact: {cell / name}")
            manifest = json.loads((cell / "manifest.json").read_text())
            summary = json.loads((cell / "cell_summary.json").read_text())
            ledger = json.loads((cell / "terminal_ledger.json").read_text())
            for name, expected_hash in manifest["files"].items():
                require(sha256(cell / name) == expected_hash, f"cell artifact hash mismatch: {cell / name}")
            require(manifest["runner_sha256"] == runner_hash and manifest["plan_sha256"] == EXPECTED_PLAN_SHA256 and manifest["execution_contract_sha256"] == EXPECTED_CONTRACT_SHA256, f"cell provenance mismatch: {cell}")
            require(manifest["scientific_selection_eligible"] is True and summary["status"] == "success" and ledger["status"] == "terminal_success", f"cell not terminal: {cell}")
            require(tuple(manifest["encoder_widths"]) == ENCODER_WIDTHS and tuple(manifest["query_head_widths"]) == QUERY_HEAD_WIDTHS and manifest["entity_batch_size"] == 16 and manifest["epochs"] == 1000, f"CNP configuration mismatch: {cell}")
            require(manifest["normalizer_fit_scope"] == "outer_training_rows_only" and manifest["query_target_input"] is False and manifest["confirmation_targets_opened"] is False, f"information boundary mismatch: {cell}")
            require(summary["query_target_perturbation_max_prediction_difference_by_regime"] == {key: 0.0 for key in REGIMES}, f"query perturbation changed prediction: {cell}")
            original = data.set_index("source_row_id").sort_index()
            for regime in REGIMES:
                perturbed = pd.read_csv(cell / f"query_target_perturbed_{regime}.csv").set_index("source_row_id").sort_index()
                delta = perturbed.cp_j_per_mol_k - original.cp_j_per_mol_k
                role_column = ROLE_COLUMNS[regime]
                mask = original.fold.eq(fold) & original[role_column].eq("query")
                require(np.allclose(delta[mask], 1_000_000.0) and np.allclose(delta[~mask], 0.0), f"perturbation copy mismatch: {cell} {regime}")
            train_entities = int(data.loc[data.fold.ne(fold)].entity_id.nunique())
            expected_updates = 1000 * int(math.ceil(train_entities / 16))
            history = pd.read_csv(cell / "training_history.csv")
            require(len(history) == 1000 and int(history.optimizer_updates_completed.iloc[-1]) == expected_updates and int(summary["optimizer_updates"]) == expected_updates and int(summary["backward_calls"]) == expected_updates, f"update count mismatch: {cell}")
            query = pd.read_csv(cell / "query_predictions.csv")
            offsets = pd.read_csv(cell / "spread_offset_predictions.csv")
            require(set(offsets.spread_offset) == {0, 1, 2, 3} and len(offsets) == 4 * len(heldout), f"offset prediction coverage mismatch: {cell}")
            require(offsets.groupby("source_row_id").spread_offset.nunique().eq(4).all() and np.isfinite(offsets.prediction_cp_j_per_mol_k.to_numpy(float)).all(), f"offset prediction stability input mismatch: {cell}")
            offsets["seed"] = seed
            all_offset_rows.append(offsets)
            require(set(query.regime) == set(REGIMES) and query["spread_offset"].isin([0, 1, 2, 3]).all(), f"regime table mismatch: {cell}")
            for regime in REGIMES:
                expected = heldout.loc[heldout[ROLE_COLUMNS[regime]].eq("query"), "source_row_id"].to_numpy(int)
                for offset in ([0, 1, 2, 3] if regime == "spread" else [0]):
                    actual = query.loc[(query.regime == regime) & (query.spread_offset == offset), "source_row_id"].to_numpy(int)
                    if offset == 0:
                        require(_same_integer_ids(actual, expected), f"query coverage mismatch: {cell} {regime}")
                    expected_offset = 0
                    for _, curve in heldout.groupby("entity_id", sort=True):
                        support = curve.loc[curve.position.to_numpy(int) % 4 == offset] if regime == "spread" else curve.loc[curve[ROLE_COLUMNS[regime]].eq("support")]
                        expected_offset += len(curve) - len(support)
                    require(len(actual) == expected_offset, f"offset coverage mismatch: {cell} {regime} {offset}")
                regime_frame = query.loc[(query.regime == regime) & (query.spread_offset == 0)].copy()
                regime_frame["seed"] = seed
                regime_rows.append(regime_frame)
            primary = query.loc[(query.regime == "spread") & (query.spread_offset == 0)].copy()
            primary["seed"] = seed
            all_rows.append(primary)
            resources.append({"fold": fold, "seed": seed, "device": summary["device"], "training_seconds": summary["training_seconds"], "peak_gpu_memory_allocated_bytes": summary["peak_gpu_memory_allocated_bytes"], "peak_gpu_memory_reserved_bytes": summary["peak_gpu_memory_reserved_bytes"], "peak_process_memory_bytes": summary["peak_process_memory_bytes"], "optimizer_updates": summary["optimizer_updates"], "backward_calls": summary["backward_calls"]})
            checkpoint = torch.load(cell / "checkpoint.pt", map_location="cpu", weights_only=True)
            model = SupportCNP()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            subset = heldout.sort_values("source_row_id", kind="stable")
            stats = checkpoint["normalizer"]
            # Checkpoint reconstruction on every official held-out support/query regime.
            with torch.no_grad():
                for regime in REGIMES:
                    role_column = ROLE_COLUMNS[regime]
                    recomputed = {}
                    for entity, curve in subset.groupby("entity_id", sort=True):
                        support = curve.loc[curve[role_column].eq("support")].sort_values("position")
                        qf = curve.loc[curve[role_column].eq("query")].sort_values("source_row_id")
                        sx = torch.tensor(((support.temperature_k.to_numpy(np.float32) - stats["temperature_mean"]) / stats["temperature_std"]))[:, None]
                        sy = torch.tensor((support.cp_j_per_mol_k.to_numpy(np.float32) - stats["target_mean"]) / stats["target_std"])
                        qx = torch.tensor(((qf.temperature_k.to_numpy(np.float32) - stats["temperature_mean"]) / stats["temperature_std"]))[:, None]
                        values = model(sx, sy, qx).numpy()
                        recomputed.update(zip(qf.source_row_id.to_numpy(int), values.tolist()))
                    expected_prediction = query.loc[(query.regime == regime) & (query.spread_offset == 0)].set_index("source_row_id")["prediction_cp_j_per_mol_k"]
                    expected_prediction = expected_prediction.loc[sorted(recomputed)].to_numpy(float)
                    standardized_difference, physical_difference, replay_passed = _checkpoint_replay_differences(
                        np.asarray([recomputed[key] for key in sorted(recomputed)]),
                        expected_prediction,
                        stats,
                    )
                    checkpoint_replay_standardized_max = max(
                        checkpoint_replay_standardized_max, standardized_difference
                    )
                    checkpoint_replay_physical_max = max(
                        checkpoint_replay_physical_max, physical_difference
                    )
                    require(replay_passed, f"checkpoint {regime} mismatch: {cell}")
    frame = pd.concat(all_rows, ignore_index=True)
    counts = frame.groupby("source_row_id").seed.nunique()
    require(len(counts) == EXPECTED_QUERY_ROWS["spread"] and counts.eq(5).all(), "OOF seed coverage mismatch")
    identity = ["source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k", "regime", "spread_offset"]
    median = frame.groupby(identity, as_index=False).prediction_cp_j_per_mol_k.median().sort_values("source_row_id", kind="stable")
    metrics = _metric(median)
    regime_frame = pd.concat(regime_rows, ignore_index=True)
    regime_identity = ["source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k", "regime", "spread_offset"]
    regime_median = regime_frame.groupby(regime_identity, as_index=False).prediction_cp_j_per_mol_k.median().sort_values(["regime", "source_row_id"], kind="stable")
    regime_metrics = _metric(regime_median)
    regime_gates = {}
    for regime in REGIMES:
        rows = regime_frame.loc[regime_frame.regime.eq(regime)]
        source_counts = rows.groupby("source_row_id").seed.nunique()
        expected_rows = EXPECTED_QUERY_ROWS[regime]
        regime_gates[regime] = {
            "exact_source_row_coverage": len(source_counts) == expected_rows,
            "five_seed_per_query_point": bool(len(source_counts) == expected_rows and source_counts.eq(5).all()),
            "finite_predictions": bool(np.isfinite(rows.prediction_cp_j_per_mol_k.to_numpy(float)).all()),
            "exact_entity_coverage": int(rows.entity_id.nunique()) == EXPECTED_ENTITIES,
            "exact_median_entity_coverage": int(regime_median.loc[regime_median.regime.eq(regime)].entity_id.nunique()) == EXPECTED_ENTITIES,
        }
    target, prediction = median.cp_j_per_mol_k.to_numpy(float), median.prediction_cp_j_per_mol_k.to_numpy(float)
    bootstrap = {"entity": _bootstrap(median, "entity_id", BOOTSTRAP_SEED), "doi": _bootstrap(median, "doi", BOOTSTRAP_SEED + 1)}
    validity_gates = {"all_regime_gates": bool(all(all(values.values()) for values in regime_gates.values())), "spread_query_coverage": len(median) == EXPECTED_QUERY_ROWS["spread"], "spread_five_seed_pointwise_median": bool(counts.eq(5).all()), "spread_finite_predictions": bool(np.isfinite(prediction).all()), "spread_entity_coverage": metrics.entity_id.nunique() == EXPECTED_ENTITIES, "all_cells_terminal": len(resources) == 25, "checkpoint_replay_within_standardized_tolerance": True, "query_target_perturbation_invariant": True}
    decision = {"scope": "ThermoML crystal-Cp support_CNP development OOF", "cells": 25, "validity_gates": validity_gates, "regime_gates": regime_gates, "exact_entity_coverage": metrics.entity_id.nunique() == EXPECTED_ENTITIES, "exact_query_coverage": len(median) == EXPECTED_QUERY_ROWS["spread"], "query_rows_by_regime": {regime: int(len(regime_median.loc[regime_median.regime.eq(regime)])) for regime in REGIMES}, "five_seed_pointwise_median": bool(counts.eq(5).all()), "finite_predictions": bool(np.isfinite(prediction).all()), "checkpoint_replay_tolerance": {"coordinate": "outer_train_standardized_target", "rtol": CHECKPOINT_REPLAY_RTOL, "atol": CHECKPOINT_REPLAY_ATOL}, "checkpoint_replay_max_standardized_difference": checkpoint_replay_standardized_max, "checkpoint_replay_max_physical_difference": checkpoint_replay_physical_max, "query_target_perturbation_invariant": True, "all_cells_terminal": len(resources) == 25, "pooled_physical_r2": _r2(target, prediction), "pooled_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))), "pooled_mae": float(np.mean(np.abs(prediction - target))), "median_entity_physical_r2": float(metrics.physical_r2.median()), "median_entity_physical_nrmse": float(metrics.physical_nrmse.median()), "p95_entity_physical_nrmse": float(np.nanpercentile(metrics.physical_nrmse, 95)), "maximum_entity_physical_nrmse": float(np.nanmax(metrics.physical_nrmse)), "negative_prediction_count": int(np.count_nonzero(prediction < 0)), "entity_bootstrap_physical_r2": bootstrap["entity"], "doi_bootstrap_physical_r2": bootstrap["doi"], "regime_metrics": {regime: {"pooled_physical_r2": _r2(group.cp_j_per_mol_k.to_numpy(float), group.prediction_cp_j_per_mol_k.to_numpy(float)), "median_entity_physical_nrmse": float(regime_metrics.loc[regime_metrics.regime.eq(regime), "physical_nrmse"].median())} for regime, group in regime_median.groupby("regime", sort=True)}, "passed": bool(all(validity_gates.values())), "confirmation_targets_opened": False}
    output_root.mkdir(parents=True)
    frame.to_csv(output_root / "spread_seed_query_predictions.csv", index=False)
    median.to_csv(output_root / "spread_median_query_predictions.csv", index=False)
    metrics.to_csv(output_root / "spread_entity_metrics.csv", index=False)
    regime_median.to_csv(output_root / "regime_median_query_predictions.csv", index=False)
    regime_metrics.to_csv(output_root / "regime_entity_metrics.csv", index=False)
    pd.DataFrame(resources).to_csv(output_root / "cell_resources.csv", index=False)
    stability = _stability(frame, pd.concat(all_offset_rows, ignore_index=True))
    stability.to_csv(output_root / "function_space_stability.csv", index=False)
    (output_root / "bootstrap_summary.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    (output_root / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    (output_root / "SUPPORT_CNP_RESULTS.md").write_text(f"# Crystal-Cp support-CNP development analysis\n\n- Five DOI folds × five seeds; pointwise seed median on spread queries.\n- Pooled physical R²: `{decision['pooled_physical_r2']:.6f}`.\n- Median entity NRMSE: `{decision['median_entity_physical_nrmse']:.6f}`; p95: `{decision['p95_entity_physical_nrmse']:.6f}`.\n- Entity-bootstrap 95% R² interval: `{bootstrap['entity']['percentile_2_5']:.6f}`–`{bootstrap['entity']['percentile_97_5']:.6f}`; DOI-bootstrap: `{bootstrap['doi']['percentile_2_5']:.6f}`–`{bootstrap['doi']['percentile_97_5']:.6f}`.\n- Architecture: DeepSets `(128,128)` encoder and `(256,128)` query head; no explicit q or entity ID.\n- Confirmation response values opened: **no**.\n")
    files = {path.name: sha256(path) for path in output_root.iterdir() if path.is_file() and path.name != "manifest.json"}
    (output_root / "manifest.json").write_text(json.dumps({"scope": decision["scope"], "runner_sha256": runner_hash, "analyzer_sha256": sha256(Path(__file__).resolve()), "plan_sha256": EXPECTED_PLAN_SHA256, "execution_contract_sha256": EXPECTED_CONTRACT_SHA256, "data_sha256": EXPECTED_DATA_SHA256, "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256, "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED, "checkpoint_replay_tolerance": decision["checkpoint_replay_tolerance"], "files": files, "confirmation_targets_opened": False}, indent=2) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw_root, args.data, args.output_root), indent=2))


if __name__ == "__main__":
    main()
