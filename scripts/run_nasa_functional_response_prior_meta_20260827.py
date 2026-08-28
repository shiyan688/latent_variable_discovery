#!/usr/bin/env python3
"""Screen a gauge-invariant decoder-response prior on NASA meta-fit entities."""

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


PLAN_PATH = PROJECT_ROOT / "NASA_FUNCTIONAL_RESPONSE_PRIOR_META_PLAN_20260827.md"
METHOD = "prefix_q_continuity_step1"
PRIOR_WEIGHTS = (0.0, 0.001, 0.01, 0.1, 1.0)
PROBE_CYCLES = (1.0, 10.0, 20.0, 28.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_result(q_root: Path, dataset: str, seed: int) -> Path:
    matches = sorted((q_root / dataset / METHOD).glob(f"seed{seed}_q4_*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one source result, found {matches}")
    return matches[0]


def _probe_responses(
    q_frame: pd.DataFrame,
    source: Any,
    q_dim: int,
) -> pd.DataFrame:
    normalized = (
        matched.REFERENCE_CONDITIONS - source.normalizer.feature_mean
    ) / source.normalizer.feature_std
    conditions = torch.tensor(
        normalized, dtype=torch.float32, device=source.device
    )
    rows = []
    with torch.no_grad():
        for row in q_frame.itertuples(index=False):
            q_value = torch.tensor(
                [getattr(row, f"q{index + 1}") for index in range(q_dim)],
                dtype=torch.float32,
                device=source.device,
            )
            prediction = source.model(
                torch.cat(
                    [conditions, q_value.unsqueeze(0).repeat(len(conditions), 1)],
                    dim=1,
                )
            ).squeeze(1)
            physical = (
                source.normalizer.target_mean
                + source.normalizer.target_std * prediction.cpu().numpy()
            )
            rows.append(
                {
                    "label": row.label,
                    "split": row.split,
                    "response_cycle1": float(physical[0]),
                    "response_cycle10": float(physical[1]),
                    "response_cycle20": float(physical[2]),
                    "response_cycle28": float(physical[3]),
                }
            )
    return pd.DataFrame(rows)


def _protocol_matched_probe_features(label_frame: pd.DataFrame) -> np.ndarray:
    first = label_frame.sort_values("discharge_index", kind="stable").iloc[0]
    protocol = [
        float(first.ambient_temperature),
        float(first.load_current_amp),
        float(first.cutoff_voltage),
    ]
    return np.asarray([[cycle, *protocol] for cycle in PROBE_CYCLES], dtype=np.float32)


def _protocol_matched_responses(
    q_frame: pd.DataFrame,
    source: Any,
    q_dim: int,
    train: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    by_label = {label: frame for label, frame in train.groupby("label", sort=False)}
    with torch.no_grad():
        for row in q_frame.itertuples(index=False):
            features = _protocol_matched_probe_features(by_label[row.label])
            normalized = (
                features - source.normalizer.feature_mean
            ) / source.normalizer.feature_std
            conditions = torch.tensor(
                normalized, dtype=torch.float32, device=source.device
            )
            q_value = torch.tensor(
                [getattr(row, f"q{index + 1}") for index in range(q_dim)],
                dtype=torch.float32,
                device=source.device,
            )
            prediction = source.model(
                torch.cat(
                    [conditions, q_value.unsqueeze(0).repeat(len(conditions), 1)],
                    dim=1,
                )
            ).squeeze(1)
            physical = (
                source.normalizer.target_mean
                + source.normalizer.target_std * prediction.cpu().numpy()
            )
            rows.append(
                {
                    "label": row.label,
                    "split": row.split,
                    "capacity_cycle1": float(physical[0]),
                    "early_fade_rate": float((physical[0] - physical[1]) / 9.0),
                    "response_cycle1": float(physical[0]),
                    "response_cycle10": float(physical[1]),
                    "response_cycle20": float(physical[2]),
                    "response_cycle28": float(physical[3]),
                }
            )
    return pd.DataFrame(rows)


def run_cell(
    source_result: Path,
    record: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    prior_weights: tuple[float, ...],
    functional_prior_subspace_rank: int,
    functional_prior_protocol: str,
) -> dict[str, Any]:
    result, checkpoint, source, config = matched._load_source(source_result, device)
    feature_columns = list(checkpoint["feature_columns"])
    train = pd.read_csv(matched._resolve(record["train_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    reference_scale = float(train.target.std(ddof=0))
    original_weights = source.embedding.weight.detach()
    label_to_index = checkpoint["label_to_index"]
    candidate_frames = []
    score_rows = []
    leakage_differences = []

    for prior_weight in prior_weights:
        weighted_config = replace(
            config,
            calibration_q_prior_weight=0.0,
            calibration_functional_prior_weight=prior_weight,
            calibration_functional_prior_subspace_rank=functional_prior_subspace_rank,
        )
        weight_q_frames = []
        selection_losses = []
        meta_query_nrmses = []
        for label, label_frame in train.groupby("label", sort=False):
            label_frame = label_frame.reset_index(drop=True)
            functional_prior_features = (
                matched.REFERENCE_CONDITIONS
                if functional_prior_protocol == "fixed"
                else _protocol_matched_probe_features(label_frame)
            )
            leave_out = int(label_to_index[label])
            keep = torch.arange(
                original_weights.shape[0], device=source.device
            ) != leave_out
            loo_source = matched._artifacts_with_embedding(
                source, original_weights[keep]
            )
            calibrated = calibrate_latent_q_for_test_labels(
                matched._dataset(label_frame, feature_columns),
                loo_source,
                weighted_config,
                functional_prior_features=functional_prior_features,
            )
            q_frame = matched._q_frame(calibrated, "meta_fit", config.q_dim)
            selection_losses.append(
                float(q_frame.iloc[0].calibration_selection_loss)
            )
            meta_query_nrmses.append(
                float(
                    np.sqrt(
                        np.mean(
                            (calibrated.eval_predictions - calibrated.eval_targets) ** 2
                        )
                    )
                    / reference_scale
                )
            )

            perturbed = label_frame.copy()
            perturbed.loc[
                matched._query_indices(label_frame, weighted_config), "target"
            ] += 123.456
            perturbed_calibrated = calibrate_latent_q_for_test_labels(
                matched._dataset(perturbed, feature_columns),
                loo_source,
                weighted_config,
                functional_prior_features=functional_prior_features,
            )
            q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
            perturbed_q = matched._q_frame(
                perturbed_calibrated, "meta_fit", config.q_dim
            )
            leakage_differences.append(
                float(
                    np.abs(
                        q_frame[q_columns].to_numpy(float)
                        - perturbed_q[q_columns].to_numpy(float)
                    ).max()
                )
            )
            weight_q_frames.append(q_frame)

        weight_q = pd.concat(weight_q_frames, ignore_index=True)
        if functional_prior_protocol == "fixed":
            functional = matched._functional_coordinates(
                weight_q, source, config.q_dim
            )
            probes = _probe_responses(weight_q, source, config.q_dim)
            weight_q = weight_q.merge(
                functional, on=["label", "split"], validate="one_to_one"
            ).merge(probes, on=["label", "split"], validate="one_to_one")
        else:
            responses = _protocol_matched_responses(
                weight_q, source, config.q_dim, train
            )
            weight_q = weight_q.merge(
                responses, on=["label", "split"], validate="one_to_one"
            )
        weight_q["prior_weight"] = prior_weight
        candidate_frames.append(weight_q)
        score_rows.append(
            {
                "prior_weight": prior_weight,
                "support_selection_loss_median": float(
                    np.median(selection_losses)
                ),
                "support_selection_loss_mean": float(np.mean(selection_losses)),
                "meta_query_nrmse_median": float(np.median(meta_query_nrmses)),
                "meta_query_nrmse_mean": float(np.mean(meta_query_nrmses)),
            }
        )

    candidates = pd.concat(candidate_frames, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(source_result.relative_to(PROJECT_ROOT)),
        "meta_fit_labels": int(candidates.label.nunique()),
        "prior_weights_scored": int(len(scores)),
        "candidate_q_rows": int(len(candidates)),
        "query_target_leakage_max_q_difference": float(
            np.max(leakage_differences)
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_dir / "meta_q_candidates.csv", index=False)
    scores.to_csv(output_dir / "prior_scores.csv", index=False)
    (output_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan-path", type=Path, default=PLAN_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", choices=matched.DATASETS, default=list(matched.DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--prior-weights", nargs="+", type=float, default=list(PRIOR_WEIGHTS))
    parser.add_argument("--functional-prior-subspace-rank", type=int, default=0)
    parser.add_argument(
        "--functional-prior-protocol",
        choices=("fixed", "first-observed"),
        default="fixed",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.q_root.resolve()
    output_root = args.output_root.resolve()
    plan_path = args.plan_path.resolve()
    datasets = tuple(args.datasets)
    seeds = tuple(args.seeds)
    prior_weights = tuple(args.prior_weights)
    if not set(prior_weights).issubset(PRIOR_WEIGHTS):
        raise ValueError(f"prior weights must be selected from {PRIOR_WEIGHTS}")
    records = matched._prepared_records(q_root)
    planned = len(datasets) * len(seeds)
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
        "prior_weights": list(prior_weights),
        "functional_prior_features": matched.REFERENCE_CONDITIONS.tolist(),
        "functional_prior_subspace_rank": args.functional_prior_subspace_rank,
        "functional_prior_protocol": args.functional_prior_protocol,
        "structure_validation_read": False,
        "planned": planned,
    }
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {output_root} without --resume")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    completed = []
    status_path = output_root / "status.jsonl"
    device = torch.device(args.device)
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
                prior_weights,
                args.functional_prior_subspace_rank,
                args.functional_prior_protocol,
            )
            completed.append(summary)
            with status_path.open("a") as handle:
                handle.write(json.dumps(summary) + "\n")
            print(
                f"[{len(completed)}/{planned}] {dataset} seed{seed} "
                f"weights={len(prior_weights)} leakage="
                f"{summary['query_target_leakage_max_q_difference']:.3g}",
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
