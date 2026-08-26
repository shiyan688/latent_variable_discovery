#!/usr/bin/env python3
"""Run fair support-conditioned real-data baselines and encoder-to-q refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from lvs.backends.support_conditioned import (
    SupportModelConfig,
    SupportPrediction,
    predict_attentive_cnp,
    predict_attentive_reliability_selector,
    predict_deepsets_regressor,
    predict_q_encoder,
    predict_q_encoder_multistart,
    train_attentive_cnp,
    train_attentive_reliability_selector,
    train_deepsets_regressor,
    train_q_support_encoder,
)
from lvs.backends.torch_mlp import build_torch_model_factory, parse_hidden_sizes
from lvs.core.loss_presets import get_loss_preset
from lvs.core.metrics import (
    effective_rank,
    grouped_rff_signatures,
    local_distance_distortion,
    macro_prediction_metrics,
    neighborhood_preservation_curve,
    pairwise_distance_metrics,
    reference_scaled_prediction_metrics,
)
from lvs.core.pipeline import LatentQConfig, build_dataset_from_arrays, train_latent_q_model

DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_encoder_pilot_20260811" / "new_methods"
METHODS = (
    "deepsets_direct",
    "attentive_cnp",
    "attentive_supportnorm_mse",
    "attentive_supportnorm_huber",
    "attentive_supportnorm_huber_bound",
    "attentive_reliability_selector",
    "encoder_q_refine",
    "encoder_q_multistart",
)
ATTENTIVE_METHODS = {
    "attentive_cnp",
    "attentive_supportnorm_mse",
    "attentive_supportnorm_huber",
    "attentive_supportnorm_huber_bound",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-summary", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--q-dim", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--decoder-epochs", type=int, default=200)
    parser.add_argument("--encoder-epochs", type=int, default=200)
    parser.add_argument("--support-ratio", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--entity-batch-size", type=int, default=8)
    parser.add_argument("--hidden-sizes", default="256,128")
    parser.add_argument("--encoder-hidden-sizes", default="128,128")
    parser.add_argument("--refine-steps", type=int, default=50)
    parser.add_argument("--refine-lr", type=float, default=0.02)
    parser.add_argument("--trust-region-weight", type=float, default=0.01)
    parser.add_argument("--clip-standard-deviations", type=float, default=3.0)
    parser.add_argument("--alignment-weight", type=float, default=0.05)
    parser.add_argument("--cal-steps", type=int, default=200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--cal-num-starts", type=int, default=4)
    parser.add_argument("--cal-selection-ratio", type=float, default=0.25)
    parser.add_argument("--cal-selection-min-rows", type=int, default=24)
    parser.add_argument("--cal-refine-steps", type=int, default=50)
    parser.add_argument("--support-scale-floor-fraction", type=float, default=0.05)
    parser.add_argument("--support-target-clip", type=float, default=8.0)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--standardized-output-bound", type=float, default=8.0)
    parser.add_argument("--selector-ratio", type=float, default=0.25)
    parser.add_argument("--selector-min-rows", type=int, default=8)
    parser.add_argument("--max-train-per-label", type=int, default=256)
    parser.add_argument("--max-test-per-label", type=int, default=256)
    parser.add_argument("--subsample-seed", type=int, default=20260808)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-artifacts", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _load_record(summary_path: Path, dataset_name: str) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records = payload.get("datasets", payload) if isinstance(payload, dict) else payload
    if isinstance(records, dict):
        records = list(records.values())
    for record in records:
        if record.get("name") == dataset_name or record.get("dataset") == dataset_name:
            return record
    raise KeyError(f"Dataset {dataset_name!r} is absent from {summary_path}")


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cap_rows_per_label(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if maximum <= 0:
        return frame.reset_index(drop=True)
    pieces = []
    for index, (_, group) in enumerate(frame.groupby("label", sort=False)):
        if len(group) > maximum:
            group = group.sample(n=maximum, random_state=seed + index)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def _prediction_metrics(
    prediction: SupportPrediction, train_y: np.ndarray
) -> dict[str, float]:
    return {
        **macro_prediction_metrics(
            prediction.targets, prediction.predictions, prediction.labels
        ),
        **reference_scaled_prediction_metrics(
            prediction.targets,
            prediction.predictions,
            reference_scale=float(np.std(train_y)),
        ),
    }


def _representation_metrics(
    prediction: SupportPrediction,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    signature_seed: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    query_frame = test_frame.iloc[prediction.query_indices].reset_index(drop=True)
    response_labels, response_signatures = grouped_rff_signatures(
        np.column_stack(
            [
                query_frame[feature_columns].to_numpy(float),
                query_frame["target"].to_numpy(float),
            ]
        ),
        query_frame["label"].to_numpy(),
        n_components=64,
        seed=signature_seed,
    )
    representation_map = {
        label: prediction.representations[index]
        for index, label in enumerate(prediction.representation_labels)
    }
    representations = np.vstack([representation_map[label] for label in response_labels])
    if len(response_labels) < 3:
        return {"effective_rank": effective_rank(representations)}, []
    curve = neighborhood_preservation_curve(
        response_signatures,
        representations,
        max_k=min(10, (len(response_labels) - 1) // 2),
    )
    metrics = {
        "response_continuity_auc": float(np.mean([row["continuity"] for row in curve])),
        "response_trustworthiness_auc": float(
            np.mean([row["trustworthiness"] for row in curve])
        ),
        "response_knn_overlap_auc": float(np.mean([row["knn_overlap"] for row in curve])),
        **{
            f"response_{key}": value
            for key, value in pairwise_distance_metrics(
                representations, response_signatures
            ).items()
        },
        **{
            f"response_{key}": value
            for key, value in local_distance_distortion(
                response_signatures,
                representations,
                k=min(5, len(response_labels) - 1),
            ).items()
        },
        "effective_rank": effective_rank(representations),
    }
    return metrics, curve


def _latent_config(args: argparse.Namespace) -> LatentQConfig:
    use_multistart = args.method == "encoder_q_multistart"
    common = dict(
        q_dim=args.q_dim,
        epochs=args.decoder_epochs,
        batch_size=args.batch_size,
        lr=1e-3,
        calibration_steps=args.cal_steps if use_multistart else args.refine_steps,
        calibration_lr=args.cal_lr if use_multistart else args.refine_lr,
        calibration_ratio=args.support_ratio,
        calibration_split_mode="random",
        calibration_init_mode="prior_random" if use_multistart else "legacy_random",
        calibration_num_starts=args.cal_num_starts if use_multistart else 1,
        calibration_selection_ratio=args.cal_selection_ratio if use_multistart else 0.0,
        calibration_selection_min_rows=(
            args.cal_selection_min_rows if use_multistart else 2
        ),
        calibration_refine_steps=args.cal_refine_steps if use_multistart else 0,
        calibration_refine_only_after_selection=use_multistart,
        seed=args.seed,
        device=args.device,
        verbose=False,
        early_stop_enabled=False,
        optimization_schedule="joint",
        joint_steps_per_cycle=2,
        theta_steps_per_cycle=1,
        q_steps_per_cycle=1,
    )
    return LatentQConfig(**common, **get_loss_preset("continuity").config_kwargs())


def run_job(args: argparse.Namespace) -> Path:
    if not 0 < args.support_ratio < 1:
        raise ValueError("support_ratio must be between zero and one")
    record = _load_record(args.prepared_summary, args.dataset)
    feature_columns = list(record["feature_columns"])
    train_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["train_csv"])),
        args.max_train_per_label,
        args.subsample_seed,
    )
    test_frame = _cap_rows_per_label(
        pd.read_csv(_resolve_path(record["test_csv"])),
        args.max_test_per_label,
        args.subsample_seed + 10000,
    )
    required = {"label", "target", *feature_columns}
    for name, frame in (("train", train_frame), ("test", test_frame)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} data are missing columns: {sorted(missing)}")
    job = {
        "dataset": args.dataset,
        "prepared_summary": str(args.prepared_summary),
        "method": args.method,
        "seed": args.seed,
        "q_dim": args.q_dim,
        "decoder_epochs": args.decoder_epochs,
        "encoder_epochs": args.encoder_epochs,
        "support_ratio": args.support_ratio,
        "batch_size": args.batch_size,
        "entity_batch_size": args.entity_batch_size,
        "hidden_sizes": args.hidden_sizes,
        "encoder_hidden_sizes": args.encoder_hidden_sizes,
        "refine_steps": args.refine_steps,
        "refine_lr": args.refine_lr,
        "trust_region_weight": args.trust_region_weight,
        "clip_standard_deviations": args.clip_standard_deviations,
        "alignment_weight": args.alignment_weight,
        "cal_steps": args.cal_steps,
        "cal_lr": args.cal_lr,
        "cal_num_starts": args.cal_num_starts,
        "cal_selection_ratio": args.cal_selection_ratio,
        "cal_selection_min_rows": args.cal_selection_min_rows,
        "cal_refine_steps": args.cal_refine_steps,
        "support_scale_floor_fraction": args.support_scale_floor_fraction,
        "support_target_clip": args.support_target_clip,
        "smooth_l1_beta": args.smooth_l1_beta,
        "standardized_output_bound": args.standardized_output_bound,
        "selector_ratio": args.selector_ratio,
        "selector_min_rows": args.selector_min_rows,
        "max_train_per_label": args.max_train_per_label,
        "max_test_per_label": args.max_test_per_label,
        "subsample_seed": args.subsample_seed,
    }
    run_dir = (
        args.output_root
        / args.dataset
        / args.method
        / f"seed{args.seed}_q{args.q_dim}_{stable_hash(job)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if args.resume and result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("status") == "success" and existing.get("job") == job:
            return result_path
    train_x = train_frame[feature_columns].to_numpy(np.float32)
    train_y = train_frame["target"].to_numpy(np.float32)
    train_labels = train_frame["label"].to_numpy()
    test_x = test_frame[feature_columns].to_numpy(np.float32)
    test_y = test_frame["target"].to_numpy(np.float32)
    test_labels = test_frame["label"].to_numpy()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    encoder_hidden_sizes = parse_hidden_sizes(args.encoder_hidden_sizes)
    support_relative = args.method.startswith("attentive_supportnorm_")
    robust_loss = args.method in {
        "attentive_supportnorm_huber",
        "attentive_supportnorm_huber_bound",
    }
    bounded_output = args.method == "attentive_supportnorm_huber_bound"
    support_config = SupportModelConfig(
        representation_dim=args.q_dim,
        epochs=args.encoder_epochs,
        lr=1e-3,
        support_ratio=args.support_ratio,
        entity_batch_size=args.entity_batch_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        decoder_hidden_sizes=hidden_sizes,
        target_coordinate_mode="support_robust" if support_relative else "global",
        target_loss="smooth_l1" if robust_loss else "mse",
        support_scale_floor_fraction=args.support_scale_floor_fraction,
        support_target_clip=args.support_target_clip,
        smooth_l1_beta=args.smooth_l1_beta,
        standardized_output_bound=(
            args.standardized_output_bound if bounded_output else None
        ),
        seed=args.seed,
        device=args.device,
    )
    started = time.perf_counter()
    initial_prediction: dict[str, float] | None = None
    initial_spatial: dict[str, float] | None = None
    initial_curve: list[dict[str, float]] = []
    calibration_seconds = 0.0
    training_diagnostics: dict[str, Any] = {}
    state_payload: dict[str, Any]
    if args.method == "attentive_reliability_selector":
        selector_artifacts = train_attentive_reliability_selector(
            train_x, train_y, train_labels, support_config
        )
        prediction = predict_attentive_reliability_selector(
            selector_artifacts,
            test_x,
            test_y,
            test_labels,
            support_ratio=args.support_ratio,
            seed=args.seed,
            selector_ratio=args.selector_ratio,
            selector_min_rows=args.selector_min_rows,
        )
        selector_rows = list(prediction.diagnostics_by_label.values())
        training_diagnostics = {
            "global_train_loss_last": (
                selector_artifacts.global_artifacts.train_loss_history[-1]
            ),
            "bounded_train_loss_last": (
                selector_artifacts.bounded_artifacts.train_loss_history[-1]
            ),
            "global_train_loss_history": (
                selector_artifacts.global_artifacts.train_loss_history
            ),
            "bounded_train_loss_history": (
                selector_artifacts.bounded_artifacts.train_loss_history
            ),
            "selected_bounded_fraction": float(
                np.mean([row["selected_bounded"] for row in selector_rows])
            ),
            "median_global_selector_mae": float(
                np.median([row["global_selector_mae"] for row in selector_rows])
            ),
            "median_bounded_selector_mae": float(
                np.median([row["bounded_selector_mae"] for row in selector_rows])
            ),
            "attention_entropy_mean": float(
                np.mean([row["attention_entropy_mean"] for row in selector_rows])
            ),
            "attention_max_weight_mean": float(
                np.mean([row["attention_max_weight_mean"] for row in selector_rows])
            ),
        }
        state_payload = {
            "global_attentive_model": (
                selector_artifacts.global_artifacts.model.state_dict()
            ),
            "bounded_attentive_model": (
                selector_artifacts.bounded_artifacts.model.state_dict()
            ),
            "global_normalizer": asdict(
                selector_artifacts.global_artifacts.normalizer
            ),
            "bounded_normalizer": asdict(
                selector_artifacts.bounded_artifacts.normalizer
            ),
            "global_support_config": asdict(
                selector_artifacts.global_artifacts.config
            ),
            "bounded_support_config": asdict(
                selector_artifacts.bounded_artifacts.config
            ),
            "bounded_target_scale_floor": (
                selector_artifacts.bounded_artifacts.target_scale_floor
            ),
        }
    elif args.method == "deepsets_direct" or args.method in ATTENTIVE_METHODS:
        if args.method == "deepsets_direct":
            artifacts = train_deepsets_regressor(
                train_x, train_y, train_labels, support_config
            )
            prediction = predict_deepsets_regressor(
                artifacts,
                test_x,
                test_y,
                test_labels,
                support_ratio=args.support_ratio,
                seed=args.seed,
            )
            state_payload = {
                "encoder": artifacts.encoder.state_dict(),
                "decoder": artifacts.decoder.state_dict(),
                "normalizer": asdict(artifacts.normalizer),
            }
        else:
            artifacts = train_attentive_cnp(
                train_x, train_y, train_labels, support_config
            )
            prediction = predict_attentive_cnp(
                artifacts,
                test_x,
                test_y,
                test_labels,
                support_ratio=args.support_ratio,
                seed=args.seed,
            )
            attention_rows = list(prediction.diagnostics_by_label.values())
            state_payload = {
                "attentive_model": artifacts.model.state_dict(),
                "normalizer": asdict(artifacts.normalizer),
                "support_config": asdict(artifacts.config),
                "target_scale_floor": artifacts.target_scale_floor,
            }
        training_diagnostics = {
            "train_loss_last": artifacts.train_loss_history[-1],
            "train_loss_history": artifacts.train_loss_history,
        }
        if args.method in ATTENTIVE_METHODS:
            training_diagnostics.update(
                {
                    "target_coordinate_mode": artifacts.config.target_coordinate_mode,
                    "target_loss": artifacts.config.target_loss,
                    "target_scale_floor": artifacts.target_scale_floor,
                    "standardized_output_bound": artifacts.config.standardized_output_bound,
                    "attention_entropy_mean": float(
                        np.mean([row["attention_entropy_mean"] for row in attention_rows])
                    ),
                    "attention_max_weight_mean": float(
                        np.mean([row["attention_max_weight_mean"] for row in attention_rows])
                    ),
                }
            )
            standardized_extrema = [
                max(
                    abs(row["prediction_standardized_min"]),
                    abs(row["prediction_standardized_max"]),
                )
                for row in attention_rows
                if "prediction_standardized_min" in row
            ]
            if standardized_extrema:
                training_diagnostics["prediction_standardized_abs_max"] = float(
                    max(standardized_extrema)
                )
    else:
        latent_config = _latent_config(args)
        train_dataset = build_dataset_from_arrays(
            train_x,
            train_labels,
            train_y,
            feature_names=feature_columns,
        )
        latent = train_latent_q_model(
            train_dataset,
            build_torch_model_factory(hidden_sizes),
            latent_config,
        )
        encoder = train_q_support_encoder(
            train_x,
            train_y,
            train_labels,
            latent,
            support_config,
            alignment_weight=args.alignment_weight,
        )
        training_diagnostics = {
            "latent_config": asdict(latent_config),
            "decoder_train_loss_last": latent.train_history[-1].mse,
            "encoder_train_loss_last": encoder.train_loss_history[-1],
            "encoder_reconstruction_loss_last": encoder.reconstruction_loss_history[-1],
            "encoder_alignment_loss_last": encoder.alignment_loss_history[-1],
            "encoder_train_loss_history": encoder.train_loss_history,
        }
        if args.method == "encoder_q_refine":
            q_prediction = predict_q_encoder(
                encoder,
                latent,
                test_x,
                test_y,
                test_labels,
                support_ratio=args.support_ratio,
                seed=args.seed,
                refine_steps=args.refine_steps,
                refine_lr=args.refine_lr,
                trust_region_weight=args.trust_region_weight,
                clip_standard_deviations=args.clip_standard_deviations,
            )
            prediction = q_prediction.refined
            calibration_seconds = q_prediction.calibration_seconds
            initial_prediction = _prediction_metrics(q_prediction.initial, train_y)
            initial_spatial, initial_curve = _representation_metrics(
                q_prediction.initial,
                test_frame,
                feature_columns,
                signature_seed=args.subsample_seed,
            )
            diagnostics = list(q_prediction.refined.diagnostics_by_label.values())
            training_diagnostics.update(
                {
                    "mean_q_movement": float(
                        np.mean([row["q_movement"] for row in diagnostics])
                    ),
                    "median_initial_support_loss": float(
                        np.median([row["initial_support_loss"] for row in diagnostics])
                    ),
                    "median_refined_support_loss": float(
                        np.median([row["refined_support_loss"] for row in diagnostics])
                    ),
                }
            )
        else:
            multistart = predict_q_encoder_multistart(
                encoder,
                latent,
                test_x,
                test_y,
                test_labels,
                config=latent_config,
                clip_standard_deviations=args.clip_standard_deviations,
            )
            prediction = multistart.prediction
            calibration_seconds = multistart.calibration_seconds
            diagnostics = list(prediction.diagnostics_by_label.values())
            training_diagnostics.update(
                {
                    "selected_encoder_candidate_fraction": float(
                        np.mean(
                            [row["selected_extra_candidate"] for row in diagnostics]
                        )
                    ),
                    "inner_selection_fraction": float(
                        np.mean([row["inner_selection_used"] for row in diagnostics])
                    ),
                    "candidate_q_dispersion_mean": float(
                        np.mean([row["candidate_q_dispersion"] for row in diagnostics])
                    ),
                }
            )
        state_payload = {
            "support_encoder": encoder.encoder.state_dict(),
            "latent_decoder": latent.model.state_dict(),
            "train_embedding": latent.embedding.state_dict(),
            "normalizer": asdict(latent.normalizer),
        }
    prediction_metrics = _prediction_metrics(prediction, train_y)
    spatial, curve = _representation_metrics(
        prediction,
        test_frame,
        feature_columns,
        signature_seed=args.subsample_seed,
    )
    artifacts_payload: dict[str, str] = {}
    if args.save_artifacts:
        query_frame = test_frame.iloc[prediction.query_indices].copy()
        query_frame["prediction"] = prediction.predictions
        prediction_path = run_dir / "query_predictions.csv"
        query_frame.to_csv(prediction_path, index=False)
        representation_frame = pd.DataFrame(
            prediction.representations,
            columns=[f"representation_{index + 1}" for index in range(args.q_dim)],
        )
        representation_frame.insert(0, "label", prediction.representation_labels)
        representation_path = run_dir / "test_representations.csv"
        representation_frame.to_csv(representation_path, index=False)
        curve_path = run_dir / "continuity_curve.csv"
        pd.DataFrame(curve).to_csv(curve_path, index=False)
        state_path = run_dir / "model_state.pt"
        torch.save(state_payload, state_path)
        artifacts_payload = {
            "query_predictions": str(prediction_path),
            "test_representations": str(representation_path),
            "continuity_curve": str(curve_path),
            "model_state": str(state_path),
        }
        if initial_curve:
            initial_curve_path = run_dir / "initial_continuity_curve.csv"
            pd.DataFrame(initial_curve).to_csv(initial_curve_path, index=False)
            artifacts_payload["initial_continuity_curve"] = str(initial_curve_path)
    payload = {
        "status": "success",
        "job": job,
        "dataset": {
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_labels": int(pd.Series(train_labels).nunique()),
            "test_labels": int(pd.Series(test_labels).nunique()),
            "support_rows": int(
                sum(row["support_rows"] for row in prediction.diagnostics_by_label.values())
            ),
            "query_rows": int(len(prediction.query_indices)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None,
        },
        "prediction": prediction_metrics,
        "initial_prediction": initial_prediction,
        "spatial": spatial,
        "initial_spatial": initial_spatial,
        "training": training_diagnostics,
        "support_diagnostics": prediction.diagnostics_by_label,
        "calibration_seconds": calibration_seconds,
        "artifacts": artifacts_payload,
        "wall_time_seconds": time.perf_counter() - started,
    }
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    return result_path


def main() -> None:
    print(run_job(parse_args()))


if __name__ == "__main__":
    main()
