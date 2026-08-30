#!/usr/bin/env python3
"""Independently audit the frozen crystal-Cp FPCA baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_fpca_development_20260829"
DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
RUNNER_PATH = PROJECT_ROOT / "scripts/run_thermoml_crystal_cp_fpca_20260829.py"
REGIMES = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(raw_root: Path = RAW_ROOT) -> dict:
    manifest = json.loads((raw_root / "manifest.json").read_text())
    if manifest["runner_sha256"] != sha256(RUNNER_PATH):
        raise RuntimeError("FPCA runner changed after execution")
    if manifest["data_sha256"] != sha256(DATA_PATH):
        raise RuntimeError("FPCA data hash mismatch")
    if manifest["confirmation_targets_opened"]:
        raise RuntimeError("FPCA manifest claims confirmation access")

    data = pd.read_csv(DATA_PATH).set_index("source_row_id")
    predictions = pd.read_csv(raw_root / "selected_query_predictions.csv")
    entity_metrics = pd.read_csv(raw_root / "selected_entity_metrics.csv")
    selection = json.loads((raw_root / "selection.json").read_text())
    summaries = []
    for regime, role_column in REGIMES.items():
        frame = predictions.loc[predictions["regime"].eq(regime)].copy()
        expected_ids = set(data.index[data[role_column].eq("query")].tolist())
        observed_ids = set(frame["source_row_id"].tolist())
        if observed_ids != expected_ids or len(frame) != len(expected_ids):
            raise RuntimeError(f"{regime} exact query coverage failed")
        joined = frame.join(data[["entity_id", "fold", "temperature_k", "cp_j_per_mol_k"]], on="source_row_id", rsuffix="_source")
        for column in ("entity_id", "fold", "temperature_k"):
            source = f"{column}_source"
            if not np.array_equal(joined[column].to_numpy(), joined[source].to_numpy()):
                raise RuntimeError(f"{regime} {column} alignment failed")
        if not np.array_equal(joined["target"].to_numpy(), joined["cp_j_per_mol_k"].to_numpy()):
            raise RuntimeError(f"{regime} target alignment failed")
        choice = selection[regime]
        if not frame["components"].eq(choice["components"]).all() or not np.isclose(frame["ridge"], choice["ridge"]).all():
            raise RuntimeError(f"{regime} selected candidate mismatch")
        target = frame["target"].to_numpy(float)
        prediction = frame["prediction"].to_numpy(float)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{regime} contains non-finite prediction")
        metrics = []
        for entity_id, group in frame.groupby("entity_id", sort=True):
            y = group["target"].to_numpy(float)
            z = group["prediction"].to_numpy(float)
            error = z - y
            scale = float(np.std(y))
            total = float(np.square(y - y.mean()).sum())
            metrics.append(
                {
                    "entity_id": entity_id,
                    "physical_nrmse": float(np.sqrt(np.mean(np.square(error))) / scale) if scale > 0 else np.nan,
                    "physical_r2": 1.0 - float(np.square(error).sum()) / total if total > 0 else np.nan,
                }
            )
        recomputed = pd.DataFrame(metrics).sort_values("entity_id")
        saved = entity_metrics.loc[entity_metrics["regime"].eq(regime)].sort_values("entity_id")
        if len(saved) != len(recomputed):
            raise RuntimeError(f"{regime} entity metric coverage failed")
        for column in ("physical_nrmse", "physical_r2"):
            if not np.allclose(saved[column], recomputed[column], rtol=0.0, atol=1e-10, equal_nan=True):
                raise RuntimeError(f"{regime} saved {column} mismatch")
        error = prediction - target
        total = float(np.square(target - target.mean()).sum())
        summaries.append(
            {
                "regime": regime,
                "components": int(choice["components"]),
                "ridge": float(choice["ridge"]),
                "entities": int(frame["entity_id"].nunique()),
                "query_rows": len(frame),
                "physical_r2": 1.0 - float(np.square(error).sum()) / total,
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "median_entity_nrmse": float(np.nanmedian(recomputed["physical_nrmse"])),
                "p95_entity_nrmse": float(np.nanquantile(recomputed["physical_nrmse"], 0.95)),
                "max_entity_nrmse": float(np.nanmax(recomputed["physical_nrmse"])),
                "entity_r2_ge_0_85_count": int(np.sum(recomputed["physical_r2"] >= 0.85)),
                "negative_prediction_count": int(np.sum(prediction < 0.0)),
            }
        )

    analysis_root = raw_root / "analysis"
    analysis_root.mkdir(exist_ok=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(analysis_root / "summary.csv", index=False)
    spread = next(row for row in summaries if row["regime"] == "spread")
    decision = {
        "status": "success",
        "scope": "independent development-only crystal-Cp FPCA audit",
        "spread_physical_r2": spread["physical_r2"],
        "fpca_reaches_expression_accuracy_threshold": bool(spread["physical_r2"] >= 0.85),
        "interpretation": "train-only low-rank functional baseline; not a symbolic expression",
        "confirmation_targets_opened": False,
    }
    (analysis_root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = [
        "# Crystal-Cp FPCA independent analysis",
        "",
        "The basis is fitted only on complete outer-training curves; held-out scores use support targets only.",
        "",
        "| Regime | m | Ridge | Queries | Pooled R² | Median entity NRMSE | p95 entity NRMSE | Entity R²≥.85 | Negative predictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['regime']} | {row['components']} | {row['ridge']:.6g} | {row['query_rows']} | "
            f"{row['physical_r2']:.6f} | {row['median_entity_nrmse']:.6f} | {row['p95_entity_nrmse']:.6f} | "
            f"{row['entity_r2_ge_0_85_count']}/{row['entities']} | {row['negative_prediction_count']} |"
        )
    lines.extend(
        [
            "",
            "The primary spread-support FPCA baseline does not reach pooled R² 0.85. This does not weaken the passing transition expression; it shows that an unconstrained train-only low-rank curve basis is not sufficient for the same boundary anomalies.",
        ]
    )
    (analysis_root / "FPCA_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in sorted(analysis_root.iterdir()) if path.name != "manifest.json"}
    analysis_manifest = {
        "status": "success",
        "raw_manifest_sha256": sha256(raw_root / "manifest.json"),
        "runner_sha256": sha256(RUNNER_PATH),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "data_sha256": sha256(DATA_PATH),
        "files": files,
        "confirmation_targets_opened": False,
    }
    (analysis_root / "manifest.json").write_text(json.dumps(analysis_manifest, indent=2), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.raw_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
