#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_q_pipeline import CSVColumnConfig, LatentQConfig, OutputConfig
from q_optimize_torch import parse_hidden_sizes, run_torch_latent_q_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent-q torch fitting on prepared application datasets.")
    parser.add_argument("--prepared-summary", type=Path, default=PROJECT_ROOT / "data" / "application" / "prepared_datasets.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "application_real")
    parser.add_argument("--q-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cal-steps", type=int, default=1200)
    parser.add_argument("--cal-lr", type=float, default=0.05)
    parser.add_argument("--cal-ratio", type=float, default=0.3)
    parser.add_argument("--orth-weight", type=float, default=0.05)
    parser.add_argument(
        "--orth-type",
        choices=("pearson", "hsic", "nhsic", "distance_correlation", "adversarial", "propensity"),
        default="pearson",
    )
    parser.add_argument(
        "--orth-stats-mode",
        choices=("mean_std", "rich", "rff_kme", "rich_rff_kme"),
        default="mean_std",
    )
    parser.add_argument("--continuity-weight", type=float, default=0.0)
    parser.add_argument("--continuity-grid-size", type=int, default=64)
    parser.add_argument("--latent-q-l2-weight", type=float, default=0.0)
    parser.add_argument("--prediction-loss-type", choices=("mse", "label_balanced_mse"), default="mse")
    parser.add_argument("--hidden-sizes", default="128,64")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = json.loads(args.prepared_summary.read_text(encoding="utf-8"))
    run_root = args.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_torch_qdim{args.q_dim}"
    run_root.mkdir(parents=True, exist_ok=True)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    summary: list[dict[str, Any]] = []

    for record in records:
        dataset_name = record["name"]
        dataset_dir = run_root / dataset_name
        latent_dir = dataset_dir / "latent_q"
        latent_dir.mkdir(parents=True, exist_ok=True)
        feature_columns = list(record["feature_columns"])
        feature_indices = ",".join(str(index) for index in range(1, 1 + len(feature_columns)))
        config = LatentQConfig(
            q_dim=args.q_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            calibration_steps=args.cal_steps,
            calibration_lr=args.cal_lr,
            calibration_ratio=args.cal_ratio,
            seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
            latent_feature_orthogonality_weight=args.orth_weight,
            latent_feature_orthogonality_type=args.orth_type,
            latent_feature_stats_mode=args.orth_stats_mode,
            latent_curve_continuity_weight=args.continuity_weight,
            latent_curve_continuity_grid_size=args.continuity_grid_size,
            latent_q_l2_weight=args.latent_q_l2_weight,
            prediction_loss_type=args.prediction_loss_type,
        )
        output_config = OutputConfig(
            output_dir=latent_dir,
            train_output_name="train_with_q.csv",
            test_output_name="test_with_q.csv",
            plot_output_name="fit_vs_real_latent_q_torch.png",
            save_csv=True,
            save_plot=True,
            plot_feature_index=0,
            plot_title=f"{dataset_name} latent-q fit",
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

        run_metadata = {
            "dataset": record,
            "status": status,
            "error": error,
            "metrics": metrics,
            "config": {
                "q_dim": args.q_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "cal_steps": args.cal_steps,
                "cal_lr": args.cal_lr,
                "cal_ratio": args.cal_ratio,
                "orth_weight": args.orth_weight,
                "orth_type": args.orth_type,
                "orth_stats_mode": args.orth_stats_mode,
                "continuity_weight": args.continuity_weight,
                "continuity_grid_size": args.continuity_grid_size,
                "latent_q_l2_weight": args.latent_q_l2_weight,
                "prediction_loss_type": args.prediction_loss_type,
                "hidden_sizes": list(hidden_sizes),
                "device": args.device,
                "feature_indices": feature_indices,
            },
        }
        (dataset_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        row = {
            "dataset": dataset_name,
            "status": status,
            "error": error,
            "test_r2": metrics.get("test_r2"),
            "test_mse": metrics.get("test_mse"),
            "train_r2_last_epoch": metrics.get("train_r2_last_epoch"),
            "latent_feature_corr_mean_abs": metrics.get("latent_feature_corr_mean_abs"),
            "latent_feature_corr_max_abs": metrics.get("latent_feature_corr_max_abs"),
            "epochs_completed": metrics.get("epochs_completed"),
            "run_dir": str(dataset_dir),
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    pd.DataFrame(summary).to_csv(run_root / "summary.csv", index=False)
    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved run summary: {run_root / 'summary.csv'}")

if __name__ == "__main__":
    main()
