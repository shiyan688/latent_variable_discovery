#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from q_optimize_torch import parse_hidden_sizes  # noqa: E402
from latent_q_pipeline import CSVColumnConfig, LatentQConfig, OutputConfig  # noqa: E402
from q_optimize_torch import run_torch_latent_q_from_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run torch-only ablations on prepared application datasets.")
    parser.add_argument("--prepared-summary", type=Path, default=PROJECT_ROOT / "data" / "application" / "prepared_datasets.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "application_torch_ablation")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to include.")
    parser.add_argument("--q-dims", default="1,2")
    parser.add_argument("--orth-weights", default="0,0.05")
    parser.add_argument(
        "--orth-types",
        default="pearson",
        help=(
            "Comma-separated orthogonality loss types. Supported: pearson, hsic, nhsic, "
            "distance_correlation, adversarial, propensity."
        ),
    )
    parser.add_argument(
        "--orth-stats-modes",
        default="mean_std",
        help="Comma-separated acquisition embedding modes: mean_std, rich, rff_kme, rich_rff_kme.",
    )
    parser.add_argument("--continuity-weights", default="0,0.05")
    parser.add_argument("--cal-q-prior-weights", default="0")
    parser.add_argument("--latent-q-l2-weights", default="0")
    parser.add_argument("--prediction-loss-types", default="mse")
    parser.add_argument("--latent-q-whitening-weights", default="0")
    parser.add_argument("--latent-jacobian-disentanglement-weights", default="0")
    parser.add_argument("--latent-q-canonicalization-modes", default="none")
    parser.add_argument("--latent-q-smoothness-weights", default="0")
    parser.add_argument("--latent-q-smoothness-epsilon", type=float, default=0.05)
    parser.add_argument("--hidden-sizes-list", default="128,64;256,128")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--epochs-list",
        default=None,
        help="Optional comma-separated epoch values to sweep. Overrides --epochs when provided.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cal-steps", type=int, default=1200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument("--continuity-grid-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.prepared_summary.read_text(encoding="utf-8"))
    if args.datasets:
        keep = set(args.datasets)
        records = [record for record in records if record["name"] in keep]
    run_root = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    q_dims = _parse_int_list(args.q_dims)
    epochs_options = _parse_int_list(args.epochs_list) if args.epochs_list else [args.epochs]
    orth_weights = _parse_float_list(args.orth_weights)
    orth_types = [value.strip() for value in args.orth_types.split(",") if value.strip()]
    orth_stats_modes = [value.strip() for value in args.orth_stats_modes.split(",") if value.strip()]
    continuity_weights = _parse_float_list(args.continuity_weights)
    hidden_sizes_options = [parse_hidden_sizes(value) for value in args.hidden_sizes_list.split(";") if value.strip()]
    summary_rows: list[dict[str, Any]] = []

    cal_q_prior_weights = _parse_float_list(args.cal_q_prior_weights)
    latent_q_l2_weights = _parse_float_list(args.latent_q_l2_weights)
    prediction_loss_types = [value.strip() for value in args.prediction_loss_types.split(",") if value.strip()]
    latent_q_whitening_weights = _parse_float_list(args.latent_q_whitening_weights)
    latent_jacobian_disentanglement_weights = _parse_float_list(args.latent_jacobian_disentanglement_weights)
    latent_q_canonicalization_modes = [
        value.strip() for value in args.latent_q_canonicalization_modes.split(",") if value.strip()
    ]
    latent_q_smoothness_weights = _parse_float_list(args.latent_q_smoothness_weights)
    for (
        record,
        epochs,
        q_dim,
        orth_weight,
        orth_type,
        orth_stats_mode,
        continuity_weight,
        cal_q_prior_weight,
        latent_q_l2_weight,
        prediction_loss_type,
        latent_q_whitening_weight,
        latent_jacobian_disentanglement_weight,
        latent_q_canonicalization_mode,
        latent_q_smoothness_weight,
        hidden_sizes,
    ) in itertools.product(
        records,
        epochs_options,
        q_dims,
        orth_weights,
        orth_types,
        orth_stats_modes,
        continuity_weights,
        cal_q_prior_weights,
        latent_q_l2_weights,
        prediction_loss_types,
        latent_q_whitening_weights,
        latent_jacobian_disentanglement_weights,
        latent_q_canonicalization_modes,
        latent_q_smoothness_weights,
        hidden_sizes_options,
    ):
        effective_orth_type = "none" if orth_weight <= 0 else orth_type
        tag = (
            f"{record['name']}_ep{epochs}_q{q_dim}_orth{orth_weight:g}_{effective_orth_type}_{orth_stats_mode}_"
            f"cont{continuity_weight:g}_calprior{cal_q_prior_weight:g}_ql2{latent_q_l2_weight:g}_"
            f"{prediction_loss_type}_white{latent_q_whitening_weight:g}_jac{latent_jacobian_disentanglement_weight:g}_"
            f"canon{latent_q_canonicalization_mode}_smooth{latent_q_smoothness_weight:g}_"
            f"h{'-'.join(str(size) for size in hidden_sizes)}"
        )
        dataset_dir = run_root / tag
        latent_dir = dataset_dir / "latent_q"
        latent_dir.mkdir(parents=True, exist_ok=True)
        feature_columns = list(record["feature_columns"])
        config = LatentQConfig(
            q_dim=q_dim,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            calibration_steps=args.cal_steps,
            calibration_lr=args.cal_lr,
            calibration_ratio=args.cal_ratio,
            seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
            latent_feature_orthogonality_weight=orth_weight,
            latent_feature_orthogonality_type=orth_type,
            latent_feature_stats_mode=orth_stats_mode,
            latent_curve_continuity_weight=continuity_weight,
            latent_curve_continuity_grid_size=args.continuity_grid_size,
            calibration_q_prior_weight=cal_q_prior_weight,
            latent_q_l2_weight=latent_q_l2_weight,
            prediction_loss_type=prediction_loss_type,
            latent_q_whitening_weight=latent_q_whitening_weight,
            latent_jacobian_disentanglement_weight=latent_jacobian_disentanglement_weight,
            latent_q_canonicalization_mode=latent_q_canonicalization_mode,
            latent_q_smoothness_weight=latent_q_smoothness_weight,
            latent_q_smoothness_epsilon=args.latent_q_smoothness_epsilon,
        )
        output_config = OutputConfig(
            output_dir=latent_dir,
            train_output_name="train_with_q.csv",
            test_output_name="test_with_q.csv",
            plot_output_name="fit_vs_real_latent_q_torch.png",
            save_csv=True,
            save_plot=True,
            plot_feature_index=0,
            plot_title=f"{tag} latent-q fit",
        )
        try:
            result = run_torch_latent_q_from_csv(
                train_csv=record["train_csv"],
                test_csv=record["test_csv"],
                column_config=CSVColumnConfig(
                    feature_cols=tuple(range(1, 1 + len(feature_columns))),
                    label_col=0,
                    target_col=-1,
                    has_header=True,
                ),
                config=config,
                output_config=output_config,
                hidden_sizes=hidden_sizes,
            )
            metrics = dict(result.metrics)
            status = "success"
            error = None
        except Exception as exc:
            metrics = {}
            status = "error"
            error = str(exc)

        metadata = {
            "dataset": record,
            "status": status,
            "error": error,
            "metrics": metrics,
            "config": {
                "q_dim": q_dim,
                "orth_weight": orth_weight,
                "orth_type": orth_type,
                "orth_stats_mode": orth_stats_mode,
                "continuity_weight": continuity_weight,
                "cal_q_prior_weight": cal_q_prior_weight,
                "latent_q_l2_weight": latent_q_l2_weight,
                "prediction_loss_type": prediction_loss_type,
                "latent_q_whitening_weight": latent_q_whitening_weight,
                "latent_jacobian_disentanglement_weight": latent_jacobian_disentanglement_weight,
                "latent_q_canonicalization_mode": latent_q_canonicalization_mode,
                "latent_q_smoothness_weight": latent_q_smoothness_weight,
                "latent_q_smoothness_epsilon": args.latent_q_smoothness_epsilon,
                "hidden_sizes": list(hidden_sizes),
                "epochs": epochs,
                "batch_size": args.batch_size,
                "cal_steps": args.cal_steps,
                "cal_lr": args.cal_lr,
                "device": args.device,
            },
        }
        (dataset_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "dataset": record["name"],
            "status": status,
            "error": error,
            "q_dim": q_dim,
            "epochs": epochs,
            "orth_weight": orth_weight,
            "orth_type": orth_type,
            "orth_stats_mode": orth_stats_mode,
            "continuity_weight": continuity_weight,
            "cal_q_prior_weight": cal_q_prior_weight,
            "latent_q_l2_weight": latent_q_l2_weight,
            "prediction_loss_type": prediction_loss_type,
            "latent_q_whitening_weight": latent_q_whitening_weight,
            "latent_jacobian_disentanglement_weight": latent_jacobian_disentanglement_weight,
            "latent_q_canonicalization_mode": latent_q_canonicalization_mode,
            "latent_q_smoothness_weight": latent_q_smoothness_weight,
            "latent_q_smoothness_epsilon": args.latent_q_smoothness_epsilon,
            "hidden_sizes": ",".join(str(size) for size in hidden_sizes),
            "test_r2": metrics.get("test_r2"),
            "test_mse": metrics.get("test_mse"),
            "train_r2_last_epoch": metrics.get("train_r2_last_epoch"),
            "latent_feature_corr_mean_abs": metrics.get("latent_feature_corr_mean_abs"),
            "latent_feature_corr_max_abs": metrics.get("latent_feature_corr_max_abs"),
            "epochs_completed": metrics.get("epochs_completed"),
            "run_dir": str(dataset_dir),
        }
        for key, value in metrics.items():
            if key.startswith(("train_q", "test_q")):
                row[key] = value
        summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(run_root / "summary.csv", index=False)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    (run_root / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_root / 'summary.csv'}")


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    main()
