#!/usr/bin/env python3
"""Run one frozen fold/seed cell of the Starry ZT neural canonical-q bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.pipeline import (
    CSVColumnConfig,
    LatentQConfig,
    _ensure_prediction_column,
    _optimize_calibration_q,
    build_dataset_from_dataframe,
    denormalize_targets,
    normalize_features,
    normalize_targets,
    train_latent_q_model,
)


DATA_ROOT = PROJECT_ROOT / "data/application_reviewer_clean/starry_te/zt"
PLAN = PROJECT_ROOT / "STARRY_ZT_NEURAL_CANONICAL_BRIDGE_PLAN_20260829.md"
FORMAL_ROOT = PROJECT_ROOT / "runs/starry_zt_neural_canonical_bridge_20260829"
EXPECTED_PLAN_SHA256 = "1b7d1d41052694004f05167634359ee377939ac1b6f959672e07d41a8fb32323"
FEATURES = [
    "temperature",
    "comp_n_elements",
    "comp_entropy",
    "comp_max_fraction",
    "comp_mean_z",
    "comp_std_z",
    "comp_min_z",
    "comp_max_z",
    "comp_mean_period",
    "comp_mean_group",
]
Q_DIM = 4
ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, required=True, choices=range(3))
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def polynomial_coefficients(
    frame: pd.DataFrame, temperature_mean: float, temperature_scale: float, degree: int
) -> np.ndarray:
    tau = (frame["temperature"].to_numpy(float) - temperature_mean) / temperature_scale
    design = np.column_stack([tau**power for power in range(degree + 1)])
    return np.linalg.lstsq(design, frame["target"].to_numpy(float), rcond=None)[0]


def polynomial_prediction(
    temperatures: np.ndarray,
    temperature_mean: float,
    temperature_scale: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    tau = (np.asarray(temperatures, dtype=float) - temperature_mean) / temperature_scale
    design = np.column_stack([tau**power for power in range(len(coefficients))])
    return design @ coefficients


def support_query_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = frame.sort_values("temperature", kind="stable").index.to_numpy()
    support = ordered[np.arange(len(ordered)) % 4 == 0]
    query = ordered[np.arange(len(ordered)) % 4 != 0]
    return support, query


def predict_model(
    model: nn.Module,
    features: np.ndarray,
    q: torch.Tensor,
    artifacts,
) -> np.ndarray:
    normalized = torch.tensor(
        normalize_features(features, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    repeated_q = q.unsqueeze(0).repeat(len(normalized), 1)
    with torch.no_grad():
        prediction = _ensure_prediction_column(
            model(torch.cat([normalized, repeated_q], dim=1))
        ).squeeze(1)
    return denormalize_targets(prediction.cpu().numpy(), artifacts.normalizer)


def main() -> None:
    args = parse_args()
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("frozen plan hash changed")
    output_root = args.output_root.resolve()
    if not args.smoke and output_root != FORMAL_ROOT.resolve():
        raise ValueError("formal cells must use the frozen output root")
    cell_root = output_root / f"fold{args.fold}_seed{args.seed}"
    cell_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    train_path = DATA_ROOT / "train.csv"
    test_path = DATA_ROOT / "test.csv"
    data = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    if data.shape[0] != 5216 or data["label"].nunique() != 80:
        raise ValueError("reviewer-clean Starry ZT cohort changed")
    data["source_row_id"] = np.arange(len(data))
    labels = sorted(data["label"].unique().tolist())
    fold_by_label = {label: index % 5 for index, label in enumerate(labels)}
    data["fold"] = data["label"].map(fold_by_label)
    outer_train = data.loc[~data["fold"].eq(args.fold)].reset_index(drop=True)
    outer_test = data.loc[data["fold"].eq(args.fold)].reset_index(drop=True)
    temperature_mean = float(outer_train["temperature"].mean())
    temperature_scale = float(outer_train["temperature"].std())

    epochs = 2 if args.smoke else 1000
    calibration_steps = 5 if args.smoke else 1200
    calibration_starts = 2 if args.smoke else 4
    config = LatentQConfig(
        q_dim=Q_DIM,
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        calibration_steps=calibration_steps,
        calibration_lr=0.05,
        seed=20260829 + 100 * args.fold + args.seed,
        device="cpu",
        verbose=False,
        early_stop_enabled=False,
        latent_feature_orthogonality_weight=0.05,
        latent_feature_orthogonality_type="hsic",
        latent_feature_stats_mode="rich_rff_kme",
        latent_curve_continuity_weight=0.05,
        latent_curve_continuity_grid_size=64,
        calibration_q_prior_weight=0.01,
        latent_q_l2_weight=0.001,
        normalize_global_regularizers_per_epoch=True,
        prediction_loss_type="label_balanced_mse",
    )
    manifest = {
        "scope": "starry_zt_neural_canonical_bridge_cell",
        "scientific_selection_eligible": not args.smoke,
        "fold": args.fold,
        "seed": args.seed,
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "train_csv_sha256": sha256(train_path),
        "test_csv_sha256": sha256(test_path),
        "outer_train_entities": int(outer_train["label"].nunique()),
        "outer_test_entities": int(outer_test["label"].nunique()),
        "epochs": epochs,
        "calibration_steps": calibration_steps,
        "calibration_starts": calibration_starts,
        "threads": args.threads,
        "temporal_confirmation_opened": False,
    }
    write_json(cell_root / "manifest.running.json", manifest)

    selected_columns = ["label", *FEATURES, "target"]
    train_dataset = build_dataset_from_dataframe(
        outer_train[selected_columns],
        CSVColumnConfig(
            feature_cols=tuple(range(1, 1 + len(FEATURES))),
            label_col=0,
            target_col=-1,
            has_header=True,
        ),
    )
    start_time = time.monotonic()
    artifacts = train_latent_q_model(
        train_dataset,
        build_torch_model_factory((256, 128)),
        config,
    )
    training_seconds = time.monotonic() - start_time
    model = artifacts.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    test_features = outer_test[FEATURES].to_numpy(np.float32)
    normalized_test_features = torch.tensor(
        normalize_features(test_features, artifacts.normalizer),
        dtype=torch.float32,
        device=artifacts.device,
    )
    normalized_test_targets = torch.tensor(
        normalize_targets(outer_test["target"].to_numpy(np.float32), artifacts.normalizer).reshape(-1, 1),
        dtype=torch.float32,
        device=artifacts.device,
    )
    q_prior_mean = artifacts.embedding.weight.detach().mean(dim=0)
    q_prior_std = artifacts.embedding.weight.detach().std(dim=0, unbiased=False).clamp_min(0.05)
    mse_loss = nn.MSELoss()

    train_q_rows = []
    train_coefficients = []
    for label in sorted(outer_train["label"].unique()):
        index = artifacts.label_to_index[label]
        raw_q = artifacts.embedding.weight[index].detach().cpu().numpy()
        coefficients = polynomial_coefficients(
            outer_train.loc[outer_train["label"].eq(label)],
            temperature_mean,
            temperature_scale,
            degree=2,
        )
        train_q_rows.append((label, *raw_q))
        train_coefficients.append((label, *coefficients))
    train_q_frame = pd.DataFrame(train_q_rows, columns=["label", "raw_q0", "raw_q1", "raw_q2", "raw_q3"]).set_index("label")
    train_coefficient_frame = pd.DataFrame(train_coefficients, columns=["label", "coef0", "coef1", "coef2"]).set_index("label")
    ridge_scores = []
    for alpha in ALPHAS:
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        score = float(
            cross_val_score(
                estimator,
                train_q_frame,
                train_coefficient_frame,
                cv=LeaveOneOut(),
                scoring="neg_mean_squared_error",
            ).mean()
        )
        ridge_scores.append((score, alpha))
    selected_alpha = max(ridge_scores)[1]
    raw_q_coefficient_map = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha)).fit(
        train_q_frame, train_coefficient_frame
    )

    prediction_frames = []
    coordinate_rows = []
    split_rows = []
    response_true_by_degree = {degree: [] for degree in range(1, 5)}
    response_prediction_by_degree = {degree: [] for degree in range(1, 5)}
    query_target_input_max_difference = 0.0
    calibration_seconds = 0.0
    for label, entity in outer_test.groupby("label", sort=True):
        support_indices, query_indices = support_query_indices(entity)
        for role, indices in (("support", support_indices), ("query", query_indices)):
            for _, split_row in outer_test.loc[indices].iterrows():
                split_rows.append(
                    {
                        "fold": args.fold,
                        "seed": args.seed,
                        "label": label,
                        "source_row_id": int(split_row["source_row_id"]),
                        "temperature": float(split_row["temperature"]),
                        "role": role,
                    }
                )
        label_seed = np.random.SeedSequence([config.seed, int(label), 314159])
        rng = np.random.default_rng(label_seed)
        initial_candidates = [q_prior_mean.detach().cpu()]
        for _ in range(calibration_starts - 1):
            draw = rng.normal(size=Q_DIM).astype(np.float32)
            initial_candidates.append(
                q_prior_mean.detach().cpu() + q_prior_std.detach().cpu() * torch.tensor(draw)
            )
        fitted_candidates = []
        support_losses = []
        calibration_start = time.monotonic()
        for initial_q in initial_candidates:
            candidate = _optimize_calibration_q(
                initial_q,
                steps=calibration_steps,
                indices=support_indices,
                feature_tensor=normalized_test_features,
                target_tensor=normalized_test_targets,
                model=model,
                mse_loss=mse_loss,
                q_prior_mean=q_prior_mean,
                q_prior_std=q_prior_std,
                functional_prior_features=None,
                functional_prior_mean=None,
                functional_prior_std=None,
                functional_prior_components=None,
                config=config,
            )
            fitted_candidates.append(candidate)
            with torch.no_grad():
                repeated_q = candidate.unsqueeze(0).repeat(len(support_indices), 1)
                support_prediction = _ensure_prediction_column(
                    model(
                        torch.cat(
                            [normalized_test_features[support_indices], repeated_q], dim=1
                        )
                    )
                )
                support_losses.append(
                    float(mse_loss(support_prediction, normalized_test_targets[support_indices]).item())
                )
        selected_start = int(np.argmin(support_losses))
        calibrated_q = fitted_candidates[selected_start]
        calibration_seconds += time.monotonic() - calibration_start

        perturbed_targets = normalized_test_targets.clone()
        perturbed_targets[query_indices] += 1_000_000.0
        perturbed_q = _optimize_calibration_q(
            initial_candidates[selected_start],
            steps=calibration_steps,
            indices=support_indices,
            feature_tensor=normalized_test_features,
            target_tensor=perturbed_targets,
            model=model,
            mse_loss=mse_loss,
            q_prior_mean=q_prior_mean,
            q_prior_std=q_prior_std,
            functional_prior_features=None,
            functional_prior_mean=None,
            functional_prior_std=None,
            functional_prior_components=None,
            config=config,
        )
        query_target_input_max_difference = max(
            query_target_input_max_difference,
            float(torch.max(torch.abs(perturbed_q - calibrated_q)).item()),
        )

        query = outer_test.loc[query_indices]
        support = outer_test.loc[support_indices]
        raw_decoder_prediction = predict_model(
            model,
            query[FEATURES].to_numpy(np.float32),
            calibrated_q,
            artifacts,
        )
        raw_q_coefficients = raw_q_coefficient_map.predict(
            pd.DataFrame(
                calibrated_q.detach().cpu().numpy().reshape(1, -1),
                columns=train_q_frame.columns,
            )
        )[0]
        raw_q_formula_prediction = polynomial_prediction(
            query["temperature"].to_numpy(float),
            temperature_mean,
            temperature_scale,
            raw_q_coefficients,
        )
        structure_coefficients = polynomial_coefficients(
            support, temperature_mean, temperature_scale, degree=2
        )
        structure_prediction = polynomial_prediction(
            query["temperature"].to_numpy(float),
            temperature_mean,
            temperature_scale,
            structure_coefficients,
        )

        grid_temperatures = np.linspace(
            float(entity["temperature"].min()),
            float(entity["temperature"].max()),
            41,
        )
        grid_features = np.repeat(entity[FEATURES].iloc[[0]].to_numpy(np.float32), 41, axis=0)
        grid_features[:, 0] = grid_temperatures
        decoder_response = predict_model(model, grid_features, calibrated_q, artifacts)
        functional_coefficients = {}
        for degree in range(1, 5):
            grid_frame = pd.DataFrame(
                {"temperature": grid_temperatures, "target": decoder_response}
            )
            coefficients = polynomial_coefficients(
                grid_frame, temperature_mean, temperature_scale, degree
            )
            functional_coefficients[degree] = coefficients
            projected_response = polynomial_prediction(
                grid_temperatures, temperature_mean, temperature_scale, coefficients
            )
            response_true_by_degree[degree].append(decoder_response)
            response_prediction_by_degree[degree].append(projected_response)
            values = polynomial_prediction(
                query["temperature"].to_numpy(float),
                temperature_mean,
                temperature_scale,
                coefficients,
            )
            scored = query[["source_row_id", "label", "temperature", "target"]].copy()
            scored["prediction"] = values
            scored["family"] = f"functional_degree{degree}"
            prediction_frames.append(scored)

        for family, values in {
            "raw_decoder": raw_decoder_prediction,
            "raw_q_ridge_req": raw_q_formula_prediction,
            "structure_req": structure_prediction,
        }.items():
            scored = query[["source_row_id", "label", "temperature", "target"]].copy()
            scored["prediction"] = values
            scored["family"] = family
            prediction_frames.append(scored)
        row = {
            "fold": args.fold,
            "seed": args.seed,
            "label": label,
            "support_rows": len(support),
            "query_rows": len(query),
            "selected_start": selected_start,
            "support_loss": support_losses[selected_start],
            "ridge_alpha": selected_alpha,
        }
        row.update({f"raw_q{index}": float(value) for index, value in enumerate(calibrated_q.cpu().numpy())})
        row.update({f"functional_q{index}": float(value) for index, value in enumerate(functional_coefficients[2])})
        row.update({f"structure_q{index}": float(value) for index, value in enumerate(structure_coefficients)})
        row.update({f"raw_q_mapped_q{index}": float(value) for index, value in enumerate(raw_q_coefficients)})
        coordinate_rows.append(row)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    family_rows = []
    for family, frame in predictions.groupby("family", sort=True):
        family_rows.append(
            {
                "family": family,
                "r2": float(r2_score(frame["target"], frame["prediction"])),
                "rmse": float(mean_squared_error(frame["target"], frame["prediction"]) ** 0.5),
            }
        )
    response_fidelity = {}
    for degree in range(1, 5):
        observed = np.concatenate(response_true_by_degree[degree])
        projected = np.concatenate(response_prediction_by_degree[degree])
        response_fidelity[str(degree)] = float(r2_score(observed, projected))

    training_history = pd.DataFrame(
        [
            {
                "epoch": item.epoch,
                "r2": item.r2,
                "mse": item.mse,
                **{f"loss_{key}": value for key, value in item.loss_components.items()},
            }
            for item in artifacts.train_history
        ]
    )
    predictions.to_csv(cell_root / "query_predictions.csv", index=False)
    pd.DataFrame(split_rows).to_csv(cell_root / "support_query_split.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(cell_root / "entity_coordinates.csv", index=False)
    train_q_frame.join(train_coefficient_frame).reset_index().to_csv(
        cell_root / "train_entity_coordinates.csv", index=False
    )
    training_history.to_csv(cell_root / "training_history.csv", index=False)
    pd.DataFrame(family_rows).to_csv(cell_root / "family_summary.csv", index=False)
    torch.save(
        {
            "model_state_dict": artifacts.model.state_dict(),
            "embedding_state_dict": artifacts.embedding.state_dict(),
            "label_to_index": artifacts.label_to_index,
            "normalizer": {
                "feature_mean": artifacts.normalizer.feature_mean,
                "feature_std": artifacts.normalizer.feature_std,
                "target_mean": artifacts.normalizer.target_mean,
                "target_std": artifacts.normalizer.target_std,
            },
            "config": config,
            "features": FEATURES,
        },
        cell_root / "checkpoint.pt",
    )
    summary = {
        "status": "success",
        "scientific_selection_eligible": not args.smoke,
        "fold": args.fold,
        "seed": args.seed,
        "training_seconds": training_seconds,
        "calibration_seconds": calibration_seconds,
        "query_rows": int(predictions.loc[predictions["family"].eq("structure_req")].shape[0]),
        "family_summary": family_rows,
        "decoder_response_projection_r2": response_fidelity,
        "selected_raw_q_ridge_alpha": selected_alpha,
        "query_target_input_max_difference": query_target_input_max_difference,
        "epochs_completed": len(artifacts.train_history),
    }
    write_json(cell_root / "cell_summary.json", summary)
    manifest["files"] = {
        path.name: sha256(path)
        for path in sorted(cell_root.iterdir())
        if path.is_file() and path.name != "manifest.running.json"
    }
    write_json(cell_root / "manifest.json", manifest)
    (cell_root / "manifest.running.json").unlink()
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
