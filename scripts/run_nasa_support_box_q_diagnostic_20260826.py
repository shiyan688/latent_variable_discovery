#!/usr/bin/env python3
"""Project support-calibrated NASA q into a support-matched coordinate box."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


PLAN_PATH = PROJECT_ROOT / "NASA_SUPPORT_BOX_Q_DIAGNOSTIC_PLAN_20260826.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _box_q(
    anchors: pd.DataFrame,
    unconstrained: pd.DataFrame,
    q_columns: list[str],
) -> tuple[pd.DataFrame, float]:
    lower = anchors[q_columns].min().to_numpy(float)
    upper = anchors[q_columns].max().to_numpy(float)
    original = unconstrained[q_columns].to_numpy(float)
    projected = np.clip(original, lower, upper)
    output = unconstrained[["label", "split"]].copy()
    output[q_columns] = projected
    for index, column in enumerate(q_columns):
        output[f"original_{column}"] = original[:, index]
        output[f"clipped_{column}"] = projected[:, index] != original[:, index]
    output["clip_l2"] = np.linalg.norm(projected - original, axis=1)
    violation = max(
        float(np.maximum(lower - projected, 0.0).max()),
        float(np.maximum(projected - upper, 0.0).max()),
    )
    return output, violation


def _predict(
    frame: pd.DataFrame,
    feature_columns: list[str],
    q_frame: pd.DataFrame,
    source: Any,
    config: Any,
) -> pd.DataFrame:
    features = torch.tensor(
        (frame[feature_columns].to_numpy(np.float32) - source.normalizer.feature_mean)
        / source.normalizer.feature_std,
        dtype=torch.float32,
        device=source.device,
    )
    query = matched._query_indices(frame, config)
    labels = frame.label.to_numpy()
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    rows = []
    with torch.no_grad():
        for q_row in q_frame.itertuples(index=False):
            indices = query[labels[query] == q_row.label]
            q_value = torch.tensor(
                [getattr(q_row, column) for column in q_columns],
                dtype=torch.float32,
                device=source.device,
            )
            inputs = torch.cat(
                [features[indices], q_value.unsqueeze(0).repeat(len(indices), 1)], dim=1
            )
            normalized = source.model(inputs).squeeze(1)
            prediction = (
                source.normalizer.target_mean
                + source.normalizer.target_std * normalized.cpu().numpy()
            )
            for index, value in zip(indices, prediction):
                rows.append(
                    {
                        "row_index": int(index),
                        "label": q_row.label,
                        "target": float(frame.iloc[index].target),
                        "prediction": float(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)


def run_cell(
    source_result: Path,
    record: dict[str, Any],
    matched_cell_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    result, checkpoint, source, config = matched._load_source(source_result, device)
    feature_columns = list(checkpoint["feature_columns"])
    validation = pd.read_csv(matched._resolve(record["test_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    support_matched_q = pd.read_csv(matched_cell_dir / "support_matched_q.csv")
    anchors = support_matched_q.loc[support_matched_q.split == "meta_fit"].copy()
    unconstrained = support_matched_q.loc[
        support_matched_q.split == "structure_validation"
    ].copy()
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    box_q, max_violation = _box_q(anchors, unconstrained, q_columns)
    functional = matched._functional_coordinates(box_q, source, config.q_dim)
    box_q = box_q.merge(functional, on=["label", "split"], validate="one_to_one")
    predictions = _predict(validation, feature_columns, box_q, source, config)

    train_reference_scale = float(
        pd.read_csv(matched._resolve(record["train_csv"])).target.std(ddof=0)
    )
    reference_nrmse = float(
        np.sqrt(np.mean((predictions.prediction - predictions.target) ** 2))
        / train_reference_scale
    )
    matched_summary = json.loads((matched_cell_dir / "cell_summary.json").read_text())
    original_nrmse = float(matched_summary["recalibrated_validation_reference_nrmse"])
    clipped_columns = [f"clipped_{column}" for column in q_columns]
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(source_result.relative_to(PROJECT_ROOT)),
        "anchor_labels": int(anchors.label.nunique()),
        "structure_validation_labels": int(box_q.label.nunique()),
        "upstream_query_target_leakage_max_q_difference": float(
            matched_summary["query_target_leakage_max_q_difference"]
        ),
        "coordinate_box_max_violation": max_violation,
        "coordinate_clip_fraction": float(box_q[clipped_columns].to_numpy().mean()),
        "clip_l2_median": float(box_q.clip_l2.median()),
        "raw_q_validation_max_abs_z": matched._max_abs_z(
            anchors, box_q, q_columns
        ),
        "functional_validation_max_abs_z": matched._max_abs_z(
            anchors, box_q, list(matched.FUNCTIONAL_COLUMNS)
        ),
        "box_reference_nrmse": reference_nrmse,
        "unconstrained_reference_nrmse": original_nrmse,
        "prediction_nrmse_ratio": reference_nrmse / original_nrmse,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    box_q.to_csv(output_dir / "box_q.csv", index=False)
    predictions.to_csv(output_dir / "query_predictions.csv", index=False)
    (output_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--method", choices=matched.METHODS, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.q_root.resolve()
    matched_root = args.matched_root.resolve()
    output_root = args.output_root.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if seeds != (0, 1, 2, 3, 4):
        raise ValueError("the frozen diagnostic requires seeds 0,1,2,3,4")
    records = matched._prepared_records(q_root)
    method_root = output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
        "plan_sha256": _sha256(PLAN_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "q_root": str(q_root.relative_to(PROJECT_ROOT)),
        "matched_root": str(matched_root.relative_to(PROJECT_ROOT)),
        "method": args.method,
        "datasets": list(matched.DATASETS),
        "seeds": list(seeds),
        "q_interface": "coordinate-wise clip of support-calibrated q to support-matched meta-fit q ranges",
        "query_targets_used_for_q": False,
        "planned": 15,
    }
    manifest_path = method_root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {method_root} without --resume")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    device = torch.device(args.device)
    completed = []
    status_path = method_root / "status.jsonl"
    for dataset in matched.DATASETS:
        for seed in seeds:
            output_dir = method_root / dataset / f"seed{seed}"
            summary_path = output_dir / "cell_summary.json"
            if args.resume and summary_path.exists():
                existing = json.loads(summary_path.read_text())
                if existing.get("status") == "success":
                    completed.append(existing)
                    continue
            summary = run_cell(
                matched._source_result(q_root, dataset, args.method, seed),
                records[dataset],
                matched_root / args.method / dataset / f"seed{seed}",
                output_dir,
                device,
            )
            completed.append(summary)
            with status_path.open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(
                f"[{len(completed)}/15] {dataset} seed{seed} {args.method} "
                f"functional_z={summary['functional_validation_max_abs_z']:.4g} "
                f"nrmse_ratio={summary['prediction_nrmse_ratio']:.4g}",
                flush=True,
            )
    pd.DataFrame(completed).sort_values(["dataset", "seed"]).to_csv(
        method_root / "cell_summary.csv", index=False
    )
    (method_root / "status.json").write_text(
        json.dumps(
            {"state": "completed_all", "planned": 15, "success": 15, "failed": 0},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
