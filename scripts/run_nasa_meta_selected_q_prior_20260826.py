#!/usr/bin/env python3
"""Select a soft q prior inside meta-fit support, then calibrate held-out NASA q."""

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
import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


PLAN_PATH = PROJECT_ROOT / "NASA_META_SELECTED_Q_PRIOR_PLAN_20260826.md"
METHODS = ("prefix_q_continuity_step1", "prefix_q_mse_step1")
PRIOR_WEIGHTS = (0.0, 0.001, 0.01, 0.1, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_result(q_root: Path, dataset: str, method: str, seed: int) -> Path:
    matches = sorted((q_root / dataset / method).glob(f"seed{seed}_q4_*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one source result, found {matches}")
    return matches[0]


def _meta_prior_grid(
    train: pd.DataFrame,
    feature_columns: list[str],
    checkpoint: dict[str, Any],
    source: Any,
    config: Any,
    train_reference_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_weights = source.embedding.weight.detach()
    label_to_index = checkpoint["label_to_index"]
    q_rows = []
    score_rows = []
    for prior_weight in PRIOR_WEIGHTS:
        entity_selection_losses = []
        entity_query_nrmses = []
        weighted_config = replace(config, calibration_q_prior_weight=prior_weight)
        for label, label_frame in train.groupby("label", sort=False):
            leave_out = int(label_to_index[label])
            keep = torch.arange(
                original_weights.shape[0], device=source.device
            ) != leave_out
            loo_source = matched._artifacts_with_embedding(source, original_weights[keep])
            calibrated = calibrate_latent_q_for_test_labels(
                matched._dataset(label_frame.reset_index(drop=True), feature_columns),
                loo_source,
                weighted_config,
            )
            q_frame = matched._q_frame(calibrated, "meta_fit", config.q_dim)
            q_frame["prior_weight"] = prior_weight
            q_rows.append(q_frame)
            entity_selection_losses.append(
                float(q_frame.iloc[0].calibration_selection_loss)
            )
            entity_query_nrmses.append(
                float(
                    np.sqrt(
                        np.mean(
                            (calibrated.eval_predictions - calibrated.eval_targets) ** 2
                        )
                    )
                    / train_reference_scale
                )
            )
        score_rows.append(
            {
                "prior_weight": prior_weight,
                "support_selection_loss_median": float(
                    np.median(entity_selection_losses)
                ),
                "support_selection_loss_mean": float(
                    np.mean(entity_selection_losses)
                ),
                "meta_query_nrmse_median_diagnostic_only": float(
                    np.median(entity_query_nrmses)
                ),
            }
        )
    return pd.concat(q_rows, ignore_index=True), pd.DataFrame(score_rows)


def run_cell(
    source_result: Path,
    record: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    result, checkpoint, source, config = matched._load_source(source_result, device)
    feature_columns = list(checkpoint["feature_columns"])
    train = pd.read_csv(matched._resolve(record["train_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    validation = pd.read_csv(matched._resolve(record["test_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    train_reference_scale = float(train.target.std(ddof=0))
    all_meta_q, prior_scores = _meta_prior_grid(
        train,
        feature_columns,
        checkpoint,
        source,
        config,
        train_reference_scale,
    )
    selected_score = prior_scores.sort_values(
        ["support_selection_loss_median", "prior_weight"], kind="stable"
    ).iloc[0]
    selected_weight = float(selected_score.prior_weight)
    selected_meta_q = all_meta_q.loc[
        all_meta_q.prior_weight == selected_weight
    ].copy()
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    prior_source = matched._artifacts_with_embedding(
        source,
        torch.tensor(
            selected_meta_q[q_columns].to_numpy(np.float32),
            dtype=torch.float32,
            device=source.device,
        ),
    )
    selected_config = replace(config, calibration_q_prior_weight=selected_weight)
    calibrated = calibrate_latent_q_for_test_labels(
        matched._dataset(validation, feature_columns), prior_source, selected_config
    )
    validation_q = matched._q_frame(
        calibrated, "structure_validation", config.q_dim
    )
    validation_q["prior_weight"] = selected_weight

    perturbed = validation.copy()
    perturbed.loc[matched._query_indices(validation, config), "target"] += 123.456
    perturbed_calibrated = calibrate_latent_q_for_test_labels(
        matched._dataset(perturbed, feature_columns), prior_source, selected_config
    )
    perturbed_q = matched._q_frame(
        perturbed_calibrated, "structure_validation", config.q_dim
    )
    leakage = validation_q[["label", *q_columns]].merge(
        perturbed_q[["label", *q_columns]],
        on="label",
        suffixes=("", "_perturbed"),
        validate="one_to_one",
    )
    leakage_difference = float(
        np.abs(
            leakage[q_columns].to_numpy(float)
            - leakage[
                [f"{column}_perturbed" for column in q_columns]
            ].to_numpy(float)
        ).max()
    )

    selected_q = pd.concat([selected_meta_q, validation_q], ignore_index=True)
    functional = matched._functional_coordinates(selected_q, source, config.q_dim)
    selected_q = selected_q.merge(
        functional, on=["label", "split"], validate="one_to_one"
    )
    selected_meta_functional = selected_q.loc[selected_q.split == "meta_fit"]
    validation_functional = selected_q.loc[
        selected_q.split == "structure_validation"
    ]
    validation_nrmse = float(
        np.sqrt(np.mean((calibrated.eval_predictions - calibrated.eval_targets) ** 2))
        / train_reference_scale
    )
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(source_result.relative_to(PROJECT_ROOT)),
        "meta_fit_labels": int(selected_meta_q.label.nunique()),
        "structure_validation_labels": int(validation_q.label.nunique()),
        "prior_weights_scored": int(len(prior_scores)),
        "selected_prior_weight": selected_weight,
        "selected_support_loss_median": float(
            selected_score.support_selection_loss_median
        ),
        "query_target_leakage_max_q_difference": leakage_difference,
        "raw_q_validation_max_abs_z": matched._max_abs_z(
            selected_meta_q, validation_q, q_columns
        ),
        "functional_validation_max_abs_z": matched._max_abs_z(
            selected_meta_functional,
            validation_functional,
            list(matched.FUNCTIONAL_COLUMNS),
        ),
        "selected_prior_validation_reference_nrmse": validation_nrmse,
        "prefix_q_no_prior_reference_nrmse": float(
            result["prediction"]["reference_nrmse"]
        ),
        "selected_to_prefix_nrmse_ratio": validation_nrmse
        / float(result["prediction"]["reference_nrmse"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_scores.to_csv(output_dir / "meta_prior_scores.csv", index=False)
    all_meta_q.to_csv(output_dir / "all_meta_q_candidates.csv", index=False)
    selected_q.to_csv(output_dir / "selected_support_matched_q.csv", index=False)
    pd.DataFrame(
        {
            "row_index": calibrated.eval_indices,
            "label": calibrated.eval_labels,
            "target": calibrated.eval_targets,
            "prediction": calibrated.eval_predictions,
        }
    ).to_csv(output_dir / "query_predictions.csv", index=False)
    (output_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.q_root.resolve()
    output_root = args.output_root.resolve()
    records = matched._prepared_records(q_root)
    method_root = output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
        "plan_sha256": _sha256(PLAN_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "q_root": str(q_root.relative_to(PROJECT_ROOT)),
        "method": args.method,
        "datasets": list(matched.DATASETS),
        "seeds": [0, 1, 2, 3, 4],
        "prior_weights": list(PRIOR_WEIGHTS),
        "prior_selection": "median support-internal selection loss across eight LOO meta-fit entities",
        "query_targets_used_for_prior_selection": False,
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
        for seed in range(5):
            output_dir = method_root / dataset / f"seed{seed}"
            summary_path = output_dir / "cell_summary.json"
            if args.resume and summary_path.exists():
                existing = json.loads(summary_path.read_text())
                if existing.get("status") == "success":
                    completed.append(existing)
                    continue
            summary = run_cell(
                _source_result(q_root, dataset, args.method, seed),
                records[dataset],
                output_dir,
                device,
            )
            completed.append(summary)
            with status_path.open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(
                f"[{len(completed)}/15] {dataset} seed{seed} {args.method} "
                f"lambda={summary['selected_prior_weight']:.4g} "
                f"functional_z={summary['functional_validation_max_abs_z']:.4g} "
                f"nrmse={summary['selected_prior_validation_reference_nrmse']:.4g}",
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
