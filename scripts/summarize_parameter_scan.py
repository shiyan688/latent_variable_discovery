#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reviewer-clean latent-q parameter scan results.")
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "application_reviewer_clean_parameter_scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "runs" / "application_reviewer_clean_parameter_scan" / "leaderboard.csv",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_root = args.scan_root if args.scan_root.is_absolute() else PROJECT_ROOT / args.scan_root
    summaries = sorted(scan_root.glob("*/calratio_*/*/summary.csv"))
    if not summaries:
        raise SystemExit(f"No summary.csv files found under {scan_root}")

    frames = []
    for path in summaries:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        parts = path.relative_to(scan_root).parts
        frame["loss_group"] = parts[0]
        frame["calibration_ratio"] = float(parts[1].replace("calratio_", ""))
        frame["summary_path"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit("Summary files exist but contain no rows.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["status"].eq("success")].copy()
    combined = combined.sort_values("test_r2", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    columns = [
        "dataset",
        "test_r2",
        "test_mse",
        "train_r2_last_epoch",
        "q_dim",
        "calibration_ratio",
        "cal_q_prior_weight",
        "loss_group",
        "orth_weight",
        "continuity_weight",
        "latent_feature_corr_mean_abs",
        "run_dir",
    ]
    columns = [column for column in columns if column in combined.columns]
    print(f"Loaded {len(combined)} successful rows from {len(summaries)} summary files.")
    print(f"Saved leaderboard: {args.output}")
    print("\nBest per dataset:")
    print(combined.groupby("dataset", as_index=False).head(1)[columns].to_string(index=False))
    print(f"\nTop {args.top_k} overall:")
    print(combined.head(args.top_k)[columns].to_string(index=False))


if __name__ == "__main__":
    main()
