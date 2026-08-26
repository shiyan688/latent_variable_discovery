#!/usr/bin/env python3
"""Calibrate held-out NASA q as a convex mixture of support-matched meta-fit q."""

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

from lvs.core.pipeline import split_support_query_indices
import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


PLAN_PATH = PROJECT_ROOT / "NASA_CONVEX_SUPPORT_Q_DIAGNOSTIC_PLAN_20260826.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit_selection_indices(
    support: np.ndarray,
    selection_ratio: float,
    min_rows: int,
    seed: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if selection_ratio <= 0 or len(support) < min_rows:
        return support, support, False
    fit, selection = split_support_query_indices(
        support,
        1.0 - selection_ratio,
        mode="random",
        seed=seed,
        label=f"{label}:convex-selection",
    )
    return fit, selection, True


def _prediction_loss(
    logits: torch.Tensor,
    indices: np.ndarray,
    anchors: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    q_value = torch.softmax(logits, dim=0) @ anchors
    selected = features[indices]
    repeated = q_value.unsqueeze(0).repeat(len(selected), 1)
    prediction = model(torch.cat([selected, repeated], dim=1)).squeeze(1)
    return torch.mean((prediction - targets[indices]) ** 2)


def _optimize_logits(
    initial: torch.Tensor,
    steps: int,
    indices: np.ndarray,
    anchors: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    model: torch.nn.Module,
    lr: float,
) -> torch.Tensor:
    logits = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([logits], lr=lr)
    for _ in range(steps):
        loss = _prediction_loss(logits, indices, anchors, features, targets, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits.detach()


def _calibrate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    anchors_frame: pd.DataFrame,
    source: Any,
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    anchors = torch.tensor(
        anchors_frame[q_columns].to_numpy(np.float32),
        dtype=torch.float32,
        device=source.device,
    )
    features = torch.tensor(
        (frame[feature_columns].to_numpy(np.float32) - source.normalizer.feature_mean)
        / source.normalizer.feature_std,
        dtype=torch.float32,
        device=source.device,
    )
    targets = torch.tensor(
        (frame.target.to_numpy(np.float32) - source.normalizer.target_mean)
        / source.normalizer.target_std,
        dtype=torch.float32,
        device=source.device,
    )
    for parameter in source.model.parameters():
        parameter.requires_grad_(False)

    q_rows = []
    weight_rows = []
    prediction_rows = []
    labels = frame.label.to_numpy()
    for label in pd.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        support, query = split_support_query_indices(
            label_indices,
            config.calibration_ratio,
            mode=config.calibration_split_mode,
            seed=config.seed,
            label=label,
        )
        fit, selection, used_inner = _fit_selection_indices(
            support,
            config.calibration_selection_ratio,
            config.calibration_selection_min_rows,
            config.seed,
            str(label),
        )
        with torch.no_grad():
            anchor_losses = []
            for anchor in anchors:
                repeated = anchor.unsqueeze(0).repeat(len(fit), 1)
                prediction = source.model(
                    torch.cat([features[fit], repeated], dim=1)
                ).squeeze(1)
                anchor_losses.append(float(torch.mean((prediction - targets[fit]) ** 2)))
        best_anchor = int(np.argmin(anchor_losses))
        best_logits = torch.full(
            (len(anchors),), -4.0, dtype=torch.float32, device=source.device
        )
        best_logits[best_anchor] = 4.0
        seed_sequence = np.random.SeedSequence(
            [int(config.seed), sum(str(label).encode("utf-8")), 314159]
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed_sequence.generate_state(1, dtype=np.uint64)[0] % np.uint64(2**63 - 1))
        )
        starts = [
            torch.zeros(len(anchors), dtype=torch.float32),
            best_logits.cpu(),
            torch.randn(len(anchors), generator=generator, dtype=torch.float32),
            torch.randn(len(anchors), generator=generator, dtype=torch.float32),
        ]
        fitted = [
            _optimize_logits(
                start.to(source.device),
                config.calibration_steps,
                fit,
                anchors,
                features,
                targets,
                source.model,
                config.calibration_lr,
            )
            for start in starts
        ]
        with torch.no_grad():
            selection_losses = [
                float(
                    _prediction_loss(
                        logits,
                        selection,
                        anchors,
                        features,
                        targets,
                        source.model,
                    )
                )
                for logits in fitted
            ]
        selected_start = int(np.argmin(selection_losses))
        logits = fitted[selected_start]
        if config.calibration_refine_steps > 0 and (
            not config.calibration_refine_only_after_selection or used_inner
        ):
            logits = _optimize_logits(
                logits,
                config.calibration_refine_steps,
                support,
                anchors,
                features,
                targets,
                source.model,
                config.calibration_lr,
            )
        with torch.no_grad():
            weights = torch.softmax(logits, dim=0)
            q_value = weights @ anchors
            repeated = q_value.unsqueeze(0).repeat(len(query), 1)
            normalized_prediction = source.model(
                torch.cat([features[query], repeated], dim=1)
            ).squeeze(1)
            prediction = (
                source.normalizer.target_mean
                + source.normalizer.target_std * normalized_prediction.cpu().numpy()
            )
        weight_values = weights.cpu().numpy()
        entropy = float(-np.sum(weight_values * np.log(np.maximum(weight_values, 1e-12))))
        q_row: dict[str, Any] = {
            "label": label,
            "split": "structure_validation",
            "selected_start": selected_start,
            "selection_loss": selection_losses[selected_start],
            "weight_entropy": entropy,
            "effective_anchors": float(np.exp(entropy)),
            "max_anchor_weight": float(weight_values.max()),
        }
        q_row.update(
            {f"q{index + 1}": float(q_value[index]) for index in range(config.q_dim)}
        )
        q_rows.append(q_row)
        for anchor_label, weight in zip(anchors_frame.label, weight_values):
            weight_rows.append(
                {"label": label, "anchor_label": anchor_label, "weight": float(weight)}
            )
        for index, value in zip(query, prediction):
            prediction_rows.append(
                {
                    "row_index": int(index),
                    "label": label,
                    "target": float(frame.iloc[index].target),
                    "prediction": float(value),
                }
            )
    return pd.DataFrame(q_rows), pd.DataFrame(weight_rows), pd.DataFrame(prediction_rows)


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
    matched_q = pd.read_csv(matched_cell_dir / "support_matched_q.csv")
    anchors = matched_q.loc[matched_q.split == "meta_fit"].copy()
    convex_q, weights, predictions = _calibrate(
        validation, feature_columns, anchors, source, config
    )
    functional = matched._functional_coordinates(convex_q, source, config.q_dim)
    convex_q = convex_q.merge(
        functional, on=["label", "split"], validate="one_to_one"
    )

    perturbed = validation.copy()
    perturbed.loc[matched._query_indices(validation, config), "target"] += 123.456
    perturbed_q, _, _ = _calibrate(
        perturbed, feature_columns, anchors, source, config
    )
    q_columns = [f"q{index + 1}" for index in range(config.q_dim)]
    leakage = convex_q.loc[:, ["label", *q_columns]].merge(
        perturbed_q.loc[:, ["label", *q_columns]],
        on="label",
        suffixes=("", "_perturbed"),
        validate="one_to_one",
    )
    leakage_difference = float(
        np.abs(
            leakage[q_columns].to_numpy(float)
            - leakage[[f"{column}_perturbed" for column in q_columns]].to_numpy(float)
        ).max()
    )

    train_reference_scale = float(
        pd.read_csv(matched._resolve(record["train_csv"])).target.std(ddof=0)
    )
    reference_nrmse = float(
        np.sqrt(np.mean((predictions.prediction - predictions.target) ** 2))
        / train_reference_scale
    )
    matched_summary = json.loads((matched_cell_dir / "cell_summary.json").read_text())
    summary = {
        "status": "success",
        "dataset": result["job"]["dataset"],
        "method": result["job"]["method"],
        "seed": int(result["job"]["seed"]),
        "source_result": str(source_result.relative_to(PROJECT_ROOT)),
        "anchor_labels": int(anchors.label.nunique()),
        "structure_validation_labels": int(convex_q.label.nunique()),
        "query_target_leakage_max_q_difference": leakage_difference,
        "convex_weight_sum_max_abs_error": float(
            np.abs(weights.groupby("label").weight.sum().to_numpy() - 1.0).max()
        ),
        "convex_min_weight": float(weights.weight.min()),
        "raw_q_validation_max_abs_z": matched._max_abs_z(
            anchors, convex_q, q_columns
        ),
        "functional_validation_max_abs_z": matched._max_abs_z(
            anchors, convex_q, list(matched.FUNCTIONAL_COLUMNS)
        ),
        "convex_reference_nrmse": reference_nrmse,
        "unconstrained_reference_nrmse": float(
            matched_summary["recalibrated_validation_reference_nrmse"]
        ),
        "prediction_nrmse_ratio": reference_nrmse
        / float(matched_summary["recalibrated_validation_reference_nrmse"]),
        "effective_anchors_median": float(convex_q.effective_anchors.median()),
        "max_anchor_weight_median": float(convex_q.max_anchor_weight.median()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    convex_q.to_csv(output_dir / "convex_q.csv", index=False)
    weights.to_csv(output_dir / "convex_weights.csv", index=False)
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
        "q_interface": "support-optimized convex mixture of eight support-matched meta-fit q anchors",
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
