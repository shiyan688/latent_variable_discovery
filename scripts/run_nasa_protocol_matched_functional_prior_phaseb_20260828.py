#!/usr/bin/env python3
"""Validate the frozen protocol-matched functional prior on held-out batteries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.core.pipeline import calibrate_latent_q_for_test_labels
import scripts.run_nasa_functional_response_prior_meta_20260827 as response_prior
import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


PLAN_PATH = PROJECT_ROOT / "NASA_PROTOCOL_MATCHED_FUNCTIONAL_PRIOR_PHASEB_PLAN_20260828.md"
METHOD = "prefix_q_continuity_step1"
WEIGHTS = (0.0, 0.01)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_result(q_root: Path, dataset: str, seed: int) -> Path:
    matches = sorted((q_root / dataset / METHOD).glob(f"seed{seed}_q4_*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one source result, found {matches}")
    return matches[0]


def run_cell(
    source_result: Path,
    record: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    functional_prior_subspace_rank: int,
) -> dict[str, Any]:
    result, checkpoint, source, config = matched._load_source(source_result, device)
    feature_columns = list(checkpoint["feature_columns"])
    train = pd.read_csv(matched._resolve(record["train_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    validation = pd.read_csv(matched._resolve(record["test_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    reference_scale = float(train.target.std(ddof=0))
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    q_frames = []
    prediction_frames = []
    score_rows = []
    leakage_differences = []

    for weight in WEIGHTS:
        weighted_config = replace(
            config,
            calibration_q_prior_weight=0.0,
            calibration_functional_prior_weight=weight,
            calibration_functional_prior_subspace_rank=functional_prior_subspace_rank,
        )
        weight_q = []
        predictions = []
        targets = []
        for label, label_frame in validation.groupby("label", sort=False):
            label_frame = label_frame.reset_index(drop=True)
            probe_features = response_prior._protocol_matched_probe_features(label_frame)
            calibrated = calibrate_latent_q_for_test_labels(
                matched._dataset(label_frame, feature_columns),
                source,
                weighted_config,
                functional_prior_features=probe_features,
            )
            q_frame = matched._q_frame(
                calibrated, "structure_validation", config.q_dim
            )
            q_frame["prior_weight"] = weight
            weight_q.append(q_frame)
            predictions.append(calibrated.eval_predictions)
            targets.append(calibrated.eval_targets)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "label": calibrated.eval_labels,
                        "target": calibrated.eval_targets,
                        "prediction": calibrated.eval_predictions,
                        "prior_weight": weight,
                    }
                )
            )

            perturbed = label_frame.copy()
            perturbed.loc[
                matched._query_indices(label_frame, weighted_config), "target"
            ] += 123.456
            perturbed_calibrated = calibrate_latent_q_for_test_labels(
                matched._dataset(perturbed, feature_columns),
                source,
                weighted_config,
                functional_prior_features=probe_features,
            )
            perturbed_q = matched._q_frame(
                perturbed_calibrated, "structure_validation", config.q_dim
            )
            leakage_differences.append(
                float(
                    np.abs(
                        q_frame[q_columns].to_numpy(float)
                        - perturbed_q[q_columns].to_numpy(float)
                    ).max()
                )
            )

        weight_q_frame = pd.concat(weight_q, ignore_index=True)
        responses = response_prior._protocol_matched_responses(
            weight_q_frame, source, config.q_dim, validation
        )
        weight_q_frame = weight_q_frame.merge(
            responses, on=["label", "split"], validate="one_to_one"
        )
        q_frames.append(weight_q_frame)
        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        score_rows.append(
            {
                "prior_weight": weight,
                "validation_reference_nrmse": float(
                    np.sqrt(np.mean((prediction - target) ** 2)) / reference_scale
                ),
            }
        )

    q_values = pd.concat(q_frames, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    baseline = float(
        scores.loc[scores.prior_weight == 0.0, "validation_reference_nrmse"].iloc[0]
    )
    selected = float(
        scores.loc[scores.prior_weight == 0.01, "validation_reference_nrmse"].iloc[0]
    )
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(source_result.relative_to(PROJECT_ROOT)),
        "structure_validation_labels": int(q_values.label.nunique()),
        "prior_weights_scored": int(scores.prior_weight.nunique()),
        "validation_q_rows": int(len(q_values)),
        "query_target_leakage_max_q_difference": float(
            np.max(leakage_differences)
        ),
        "baseline_weight0_validation_reference_nrmse": baseline,
        "selected_weight": 0.01,
        "selected_validation_reference_nrmse": selected,
        "selected_to_baseline_nrmse_ratio": selected / baseline,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    q_values.to_csv(output_dir / "validation_q.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "query_predictions.csv", index=False
    )
    scores.to_csv(output_dir / "weight_scores.csv", index=False)
    (output_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, default=PLAN_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--functional-prior-subspace-rank", type=int, default=2)
    parser.add_argument(
        "--datasets", nargs="+", choices=matched.DATASETS, default=list(matched.DATASETS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.q_root.resolve()
    output_root = args.output_root.resolve()
    plan_path = args.plan_path.resolve()
    datasets = tuple(args.datasets)
    seeds = tuple(args.seeds)
    records = matched._prepared_records(q_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": str(plan_path.relative_to(PROJECT_ROOT)),
        "plan_sha256": _sha256(plan_path),
        "runner_sha256": _sha256(Path(__file__)),
        "q_root": str(q_root.relative_to(PROJECT_ROOT)),
        "method": METHOD,
        "datasets": list(datasets),
        "seeds": list(seeds),
        "prior_weights": list(WEIGHTS),
        "selected_weight": 0.01,
        "functional_prior_subspace_rank": args.functional_prior_subspace_rank,
        "functional_prior_protocol": "first-observed",
        "planned": len(datasets) * len(seeds),
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {output_root} without --resume")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    device = torch.device(args.device)
    completed = []
    status_path = output_root / "status.jsonl"
    planned = len(datasets) * len(seeds)
    for dataset in datasets:
        for seed in seeds:
            output_dir = output_root / dataset / f"seed{seed}"
            summary_path = output_dir / "cell_summary.json"
            if args.resume and summary_path.exists():
                existing = json.loads(summary_path.read_text())
                if existing.get("status") == "success":
                    completed.append(existing)
                    continue
            summary = run_cell(
                _source_result(q_root, dataset, seed),
                records[dataset],
                output_dir,
                device,
                args.functional_prior_subspace_rank,
            )
            completed.append(summary)
            with status_path.open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(
                f"[{len(completed)}/{planned}] {dataset} seed{seed} "
                f"ratio={summary['selected_to_baseline_nrmse_ratio']:.4g} "
                f"leakage={summary['query_target_leakage_max_q_difference']:.3g}",
                flush=True,
            )

    pd.DataFrame(completed).sort_values(["dataset", "seed"]).to_csv(
        output_root / "cell_summary.csv", index=False
    )
    (output_root / "status.json").write_text(
        json.dumps(
            {
                "state": "completed_all",
                "planned": planned,
                "success": planned,
                "failed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
