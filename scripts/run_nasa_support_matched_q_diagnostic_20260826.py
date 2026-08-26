#!/usr/bin/env python3
"""Recalibrate NASA meta-fit q from prefix support and diagnose the interface."""

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

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.pipeline import (
    LatentQConfig,
    NormalizationStats,
    TrainingArtifacts,
    build_dataset_from_arrays,
    calibrate_latent_q_for_test_labels,
    split_support_query_indices,
)


PLAN_PATH = PROJECT_ROOT / "NASA_SUPPORT_MATCHED_Q_DIAGNOSTIC_PLAN_20260826.md"
DATASETS = tuple(f"nasa_battery_capacity_reviewer_clean_inner{index}" for index in range(3))
METHODS = ("joint_continuity_step1", "joint_mse_step1")
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")
REFERENCE_CONDITIONS = np.asarray(
    [
        [1.0, 24.0, 2.0, 2.5],
        [10.0, 24.0, 2.0, 2.5],
        [20.0, 24.0, 2.0, 2.5],
        [28.0, 24.0, 2.0, 2.5],
    ],
    dtype=np.float32,
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_result(q_root: Path, dataset: str, method: str, seed: int) -> Path:
    matches = sorted((q_root / dataset / method).glob(f"seed{seed}_q4_*/result.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one source result, found {matches}")
    return matches[0]


def _prepared_records(q_root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((q_root / "experiment_manifest.json").read_text())
    summaries = manifest["prepared_summaries"]
    if len(summaries) != 1:
        raise ValueError("expected exactly one prepared-data summary")
    records = json.loads(_resolve(summaries[0]).read_text())
    return {record["name"]: record for record in records}


def _load_source(
    result_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], TrainingArtifacts, LatentQConfig]:
    result = json.loads(result_path.read_text())
    checkpoint = torch.load(
        _resolve(result["artifacts"]["training_checkpoint"]),
        map_location=device,
        weights_only=False,
    )
    q_dim = int(result["job"]["q_dim"])
    model = build_torch_model_factory(tuple(checkpoint["hidden_sizes"]))(
        len(checkpoint["feature_columns"]) + q_dim
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    label_to_index = checkpoint["label_to_index"]
    embedding = torch.nn.Embedding(len(label_to_index), q_dim).to(device)
    embedding.load_state_dict(checkpoint["embedding_state_dict"])
    normalizer_payload = checkpoint["normalizer"]
    normalizer = NormalizationStats(
        feature_mean=np.asarray(normalizer_payload["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(normalizer_payload["feature_std"], dtype=np.float32),
        target_mean=float(normalizer_payload["target_mean"]),
        target_std=float(normalizer_payload["target_std"]),
    )
    artifacts = TrainingArtifacts(
        model=model,
        embedding=embedding,
        normalizer=normalizer,
        label_to_index={},
        device=device,
        train_history=[],
    )
    config = replace(
        LatentQConfig(**result["latent_config"]),
        device=str(device),
        verbose=False,
    )
    return result, checkpoint, artifacts, config


def _dataset(frame: pd.DataFrame, feature_columns: list[str]):
    return build_dataset_from_arrays(
        frame[feature_columns].to_numpy(np.float32),
        frame["label"].to_numpy(),
        frame["target"].to_numpy(np.float32),
        feature_names=feature_columns,
    )


def _artifacts_with_embedding(
    source: TrainingArtifacts,
    weights: torch.Tensor,
) -> TrainingArtifacts:
    embedding = torch.nn.Embedding(weights.shape[0], weights.shape[1]).to(source.device)
    embedding.weight.data.copy_(weights)
    return TrainingArtifacts(
        model=source.model,
        embedding=embedding,
        normalizer=source.normalizer,
        label_to_index={},
        device=source.device,
        train_history=[],
    )


def _q_frame(calibrated: Any, split: str, q_dim: int) -> pd.DataFrame:
    rows = []
    for label, q_value in calibrated.q_by_label.items():
        row: dict[str, Any] = {"label": label, "split": split}
        row.update({f"q{index + 1}": float(q_value[index]) for index in range(q_dim)})
        row.update(
            {
                f"calibration_{key}": float(value)
                for key, value in calibrated.diagnostics_by_label[label].items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _calibrate_meta_fit_loo(
    frame: pd.DataFrame,
    feature_columns: list[str],
    source: TrainingArtifacts,
    checkpoint: dict[str, Any],
    config: LatentQConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    original_weights = source.embedding.weight.detach()
    label_to_index = checkpoint["label_to_index"]
    q_frames = []
    predictions = []
    targets = []
    for label, label_frame in frame.groupby("label", sort=False):
        leave_out = int(label_to_index[label])
        keep = torch.arange(original_weights.shape[0], device=source.device) != leave_out
        loo_artifacts = _artifacts_with_embedding(source, original_weights[keep])
        calibrated = calibrate_latent_q_for_test_labels(
            _dataset(label_frame.reset_index(drop=True), feature_columns),
            loo_artifacts,
            config,
        )
        q_frames.append(_q_frame(calibrated, "meta_fit", config.q_dim))
        predictions.append(calibrated.eval_predictions)
        targets.append(calibrated.eval_targets)
    return pd.concat(q_frames, ignore_index=True), np.concatenate(predictions), np.concatenate(targets)


def _functional_coordinates(
    q_frame: pd.DataFrame,
    source: TrainingArtifacts,
    q_dim: int,
) -> pd.DataFrame:
    normalized_conditions = (
        REFERENCE_CONDITIONS - source.normalizer.feature_mean
    ) / source.normalizer.feature_std
    rows = []
    with torch.no_grad():
        conditions = torch.tensor(
            normalized_conditions,
            dtype=torch.float32,
            device=source.device,
        )
        for row in q_frame.itertuples(index=False):
            q_value = torch.tensor(
                [getattr(row, f"q{index + 1}") for index in range(q_dim)],
                dtype=torch.float32,
                device=source.device,
            )
            inputs = torch.cat(
                [conditions, q_value.unsqueeze(0).repeat(len(conditions), 1)], dim=1
            )
            prediction = source.model(inputs).squeeze(1).cpu().numpy()
            prediction = (
                source.normalizer.target_mean
                + source.normalizer.target_std * prediction
            )
            rows.append(
                {
                    "label": row.label,
                    "split": row.split,
                    "capacity_cycle1": float(prediction[0]),
                    "early_fade_rate": float((prediction[0] - prediction[1]) / 9.0),
                }
            )
    return pd.DataFrame(rows)


def _support_jacobians(
    frame: pd.DataFrame,
    q_frame: pd.DataFrame,
    feature_columns: list[str],
    source: TrainingArtifacts,
    config: LatentQConfig,
) -> pd.DataFrame:
    rows = []
    for parameter in source.model.parameters():
        parameter.requires_grad_(False)
    for label, label_frame in frame.groupby("label", sort=False):
        label_frame = label_frame.reset_index(drop=True)
        indices = np.arange(len(label_frame), dtype=np.int64)
        support, _ = split_support_query_indices(
            indices,
            config.calibration_ratio,
            mode=config.calibration_split_mode,
            seed=config.seed,
            label=label,
        )
        features = (
            label_frame[feature_columns].to_numpy(np.float32)
            - source.normalizer.feature_mean
        ) / source.normalizer.feature_std
        support_features = torch.tensor(
            features[support], dtype=torch.float32, device=source.device
        )
        selected = q_frame.loc[q_frame.label == label].iloc[0]
        q_value = torch.tensor(
            [selected[f"q{index + 1}"] for index in range(config.q_dim)],
            dtype=torch.float32,
            device=source.device,
            requires_grad=True,
        )

        def predict(candidate_q: torch.Tensor) -> torch.Tensor:
            repeated = candidate_q.unsqueeze(0).repeat(len(support_features), 1)
            return source.model(torch.cat([support_features, repeated], dim=1)).squeeze(1)

        jacobian = torch.autograd.functional.jacobian(
            predict, q_value, vectorize=True
        ).detach().cpu().numpy()
        singular = np.linalg.svd(jacobian, compute_uv=False)
        floor = max(float(singular[0]) * 1e-8, 1e-12)
        rows.append(
            {
                "label": label,
                "jacobian_smax": float(singular[0]),
                "jacobian_smin": float(singular[-1]),
                "jacobian_condition_floor1e8": float(singular[0] / max(float(singular[-1]), floor)),
                "jacobian_effective_rank_1e4": int(np.sum(singular > singular[0] * 1e-4)),
                "support_rows": int(len(support)),
            }
        )
    return pd.DataFrame(rows)


def _max_abs_z(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
) -> float:
    mean = train[columns].mean().to_numpy(float)
    std = train[columns].std(ddof=0).to_numpy(float)
    std = np.maximum(std, 1e-8)
    return float(np.abs((validation[columns].to_numpy(float) - mean) / std).max())


def _nearest_standardized_distance(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
) -> np.ndarray:
    mean = train[columns].mean().to_numpy(float)
    std = np.maximum(train[columns].std(ddof=0).to_numpy(float), 1e-8)
    train_values = (train[columns].to_numpy(float) - mean) / std
    validation_values = (validation[columns].to_numpy(float) - mean) / std
    distances = np.linalg.norm(
        validation_values[:, None, :] - train_values[None, :, :], axis=2
    )
    return distances.min(axis=1)


def _query_indices(frame: pd.DataFrame, config: LatentQConfig) -> np.ndarray:
    parts = []
    labels = frame.label.to_numpy()
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        _, query = split_support_query_indices(
            indices,
            config.calibration_ratio,
            mode=config.calibration_split_mode,
            seed=config.seed,
            label=label,
        )
        parts.append(query)
    return np.concatenate(parts)


def run_cell(
    result_path: Path,
    record: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    result, checkpoint, source, config = _load_source(result_path, device)
    feature_columns = list(checkpoint["feature_columns"])
    train = pd.read_csv(_resolve(record["train_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)
    validation = pd.read_csv(_resolve(record["test_csv"])).sort_values(
        ["label", "discharge_index"], kind="stable"
    ).reset_index(drop=True)

    train_q, train_predictions, train_targets = _calibrate_meta_fit_loo(
        train, feature_columns, source, checkpoint, config
    )
    validation_dataset = _dataset(validation, feature_columns)
    validation_calibrated = calibrate_latent_q_for_test_labels(
        validation_dataset, source, config
    )
    validation_q = _q_frame(
        validation_calibrated, "structure_validation", config.q_dim
    )

    perturbed = validation.copy()
    perturbed.loc[_query_indices(validation, config), "target"] += 123.456
    perturbed_calibrated = calibrate_latent_q_for_test_labels(
        _dataset(perturbed, feature_columns), source, config
    )
    leakage_difference = max(
        float(
            np.max(
                np.abs(
                    validation_calibrated.q_by_label[label]
                    - perturbed_calibrated.q_by_label[label]
                )
            )
        )
        for label in validation_calibrated.q_by_label
    )

    all_q = pd.concat([train_q, validation_q], ignore_index=True)
    functional = _functional_coordinates(all_q, source, config.q_dim)
    all_q = all_q.merge(functional, on=["label", "split"], validate="one_to_one")
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    matched_train = all_q.loc[all_q.split == "meta_fit"].copy()
    matched_validation = all_q.loc[all_q.split == "structure_validation"].copy()
    matched_validation["matched_q_nearest_distance"] = _nearest_standardized_distance(
        matched_train, matched_validation, q_columns
    )

    original_train = pd.read_csv(_resolve(result["artifacts"]["train_label_q"]))
    original_test = pd.read_csv(_resolve(result["artifacts"]["test_label_q"]))
    train_comparison = matched_train.merge(
        original_train.loc[:, ["label", *q_columns]],
        on="label",
        suffixes=("", "_full_curve"),
        validate="one_to_one",
    )
    train_comparison["q_displacement_from_full_curve"] = np.linalg.norm(
        train_comparison[q_columns].to_numpy(float)
        - train_comparison[[f"{column}_full_curve" for column in q_columns]].to_numpy(float),
        axis=1,
    )
    validation_comparison = matched_validation.merge(
        original_test.loc[:, ["label", *q_columns]],
        on="label",
        suffixes=("", "_saved"),
        validate="one_to_one",
    )
    test_q_reproduction_max_abs = float(
        np.abs(
            validation_comparison[q_columns].to_numpy(float)
            - validation_comparison[[f"{column}_saved" for column in q_columns]].to_numpy(float)
        ).max()
    )

    train_jacobian = _support_jacobians(
        train, matched_train, feature_columns, source, config
    ).assign(split="meta_fit")
    validation_jacobian = _support_jacobians(
        validation, matched_validation, feature_columns, source, config
    ).assign(split="structure_validation")
    jacobians = pd.concat([train_jacobian, validation_jacobian], ignore_index=True)
    all_q = all_q.merge(
        jacobians, on=["label", "split"], validate="one_to_one"
    )

    train_reference_scale = float(train.target.std(ddof=0))
    train_reference_nrmse = float(
        np.sqrt(np.mean((train_predictions - train_targets) ** 2)) / train_reference_scale
    )
    validation_reference_nrmse = float(
        np.sqrt(
            np.mean(
                (validation_calibrated.eval_predictions - validation_calibrated.eval_targets) ** 2
            )
        )
        / train_reference_scale
    )
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(result_path.relative_to(PROJECT_ROOT)),
        "meta_fit_labels": int(len(matched_train)),
        "structure_validation_labels": int(len(matched_validation)),
        "query_target_leakage_max_q_difference": leakage_difference,
        "test_q_reproduction_max_abs": test_q_reproduction_max_abs,
        "saved_validation_reference_nrmse": float(result["prediction"]["reference_nrmse"]),
        "recalibrated_validation_reference_nrmse": validation_reference_nrmse,
        "validation_reference_nrmse_abs_difference": abs(
            validation_reference_nrmse - float(result["prediction"]["reference_nrmse"])
        ),
        "meta_fit_prefix_reference_nrmse": train_reference_nrmse,
        "matched_raw_q_validation_max_abs_z": _max_abs_z(
            matched_train, matched_validation, q_columns
        ),
        "matched_functional_validation_max_abs_z": _max_abs_z(
            matched_train, matched_validation, list(FUNCTIONAL_COLUMNS)
        ),
        "matched_q_nearest_distance_median": float(
            matched_validation.matched_q_nearest_distance.median()
        ),
        "meta_fit_q_displacement_from_full_curve_median": float(
            train_comparison.q_displacement_from_full_curve.median()
        ),
        "meta_fit_candidate_dispersion_median": float(
            matched_train.calibration_candidate_q_dispersion.median()
        ),
        "validation_candidate_dispersion_median": float(
            matched_validation.calibration_candidate_q_dispersion.median()
        ),
        "support_jacobian_smin_median": float(jacobians.jacobian_smin.median()),
        "support_jacobian_condition_median": float(
            jacobians.jacobian_condition_floor1e8.median()
        ),
        "support_jacobian_effective_rank_median": float(
            jacobians.jacobian_effective_rank_1e4.median()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    all_q.to_csv(output_dir / "support_matched_q.csv", index=False)
    jacobians.to_csv(output_dir / "support_jacobians.csv", index=False)
    train_comparison.to_csv(output_dir / "meta_fit_full_vs_prefix_q.csv", index=False)
    validation_comparison.to_csv(
        output_dir / "validation_saved_vs_recalibrated_q.csv", index=False
    )
    (output_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.q_root.resolve()
    output_root = args.output_root.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if seeds != (0, 1, 2, 3, 4):
        raise ValueError("the frozen diagnostic requires seeds 0,1,2,3,4")
    records = _prepared_records(q_root)
    device = torch.device(args.device)
    cells = [
        (dataset, seed, _source_result(q_root, dataset, args.method, seed))
        for dataset in DATASETS
        for seed in seeds
    ]
    method_root = output_root / args.method
    method_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
        "plan_sha256": _sha256(PLAN_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "q_root": str(q_root.relative_to(PROJECT_ROOT)),
        "method": args.method,
        "datasets": list(DATASETS),
        "seeds": list(seeds),
        "device": args.device,
        "meta_fit_q_information": "prefix support with leave-one-entity-out train-q prior",
        "structure_validation_q_information": "prefix support with all meta-fit train-q prior",
        "query_targets_used_for_q": False,
        "planned": len(cells),
    }
    manifest_path = method_root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to reuse {method_root} without --resume")
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2))

    completed = []
    status_path = method_root / "status.jsonl"
    for dataset, seed, result_path in cells:
        cell_dir = method_root / dataset / f"seed{seed}"
        summary_path = cell_dir / "cell_summary.json"
        if args.resume and summary_path.exists():
            existing = json.loads(summary_path.read_text())
            if existing.get("status") == "success":
                completed.append(existing)
                continue
        summary = run_cell(result_path, records[dataset], cell_dir, device)
        completed.append(summary)
        with status_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(
            f"[{len(completed)}/{len(cells)}] {dataset} seed{seed} {args.method} "
            f"raw_z={summary['matched_raw_q_validation_max_abs_z']:.4g} "
            f"functional_z={summary['matched_functional_validation_max_abs_z']:.4g}",
            flush=True,
        )
    pd.DataFrame(completed).sort_values(["dataset", "seed"]).to_csv(
        method_root / "cell_summary.csv", index=False
    )
    (method_root / "status.json").write_text(
        json.dumps(
            {
                "state": "completed_all",
                "planned": len(cells),
                "success": len(completed),
                "failed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
