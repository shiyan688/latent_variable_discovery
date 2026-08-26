#!/usr/bin/env python3
"""Print complete-block summaries from an in-progress discovery analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def _complete_blocks(
    frame: pd.DataFrame, block_columns: list[str], expected_methods: int
) -> tuple[pd.DataFrame, int]:
    sizes = frame.groupby(block_columns).size()
    complete = sizes[sizes.eq(expected_methods)].index
    if len(complete) == 0:
        return frame.iloc[0:0].copy(), 0
    balanced = frame.set_index(block_columns).loc[complete].reset_index()
    return balanced, len(complete)


def main() -> None:
    args = parse_args()
    synthetic = pd.read_csv(args.analysis_dir / "synthetic_all_runs.csv")
    real = pd.read_csv(args.analysis_dir / "real_all_runs.csv")
    synthetic, synthetic_blocks = _complete_blocks(
        synthetic, ["expression_id", "seed"], expected_methods=10
    )
    real, real_blocks = _complete_blocks(real, ["dataset", "seed"], expected_methods=9)
    synthetic["prediction_rank"] = synthetic.groupby(
        ["expression_id", "seed"]
    )["reference_nrmse"].rank()
    synthetic_prediction = synthetic.groupby("method").agg(
        blocks=("reference_nrmse", "size"),
        reference_nrmse=("reference_nrmse", "mean"),
        mean_rank=("prediction_rank", "mean"),
    ).sort_values("mean_rank")
    synthetic_latent = synthetic[
        synthetic["method"].str.startswith(("joint_", "alternating_"))
    ].copy()
    synthetic_latent["continuity_rank"] = synthetic_latent.groupby(
        ["expression_id", "seed"]
    )["continuity_auc"].rank(ascending=False)
    synthetic_continuity = synthetic_latent.groupby("method").agg(
        aligned_nrmse=("aligned_nrmse", "mean"),
        continuity_auc=("continuity_auc", "mean"),
        trustworthiness_auc=("trustworthiness_auc", "mean"),
        distortion_p95=("local_log_distortion_p95", "mean"),
        mean_rank=("continuity_rank", "mean"),
    ).sort_values("mean_rank")
    synthetic_winners = synthetic.loc[
        synthetic.groupby(["expression_id", "seed"])["reference_nrmse"].idxmin(),
        ["expression_id", "seed", "method", "reference_nrmse"],
    ]
    real["prediction_rank"] = real.groupby(["dataset", "seed"])[
        "reference_nrmse"
    ].rank()
    real_prediction = real.groupby("method").agg(
        blocks=("reference_nrmse", "size"),
        reference_nrmse=("reference_nrmse", "mean"),
        mean_rank=("prediction_rank", "mean"),
    ).sort_values("mean_rank")
    real_latent = real[real["method"].str.startswith(("joint_", "alternating_"))].copy()
    real_latent["continuity_rank"] = real_latent.groupby(["dataset", "seed"])[
        "response_continuity_auc"
    ].rank(ascending=False)
    real_continuity = real_latent.groupby("method").agg(
        continuity_auc=("response_continuity_auc", "mean"),
        trustworthiness_auc=("response_trustworthiness_auc", "mean"),
        distance_spearman=("response_distance_spearman", "mean"),
        distortion_p95=("response_local_log_distortion_p95", "mean"),
        mean_rank=("continuity_rank", "mean"),
    ).sort_values("mean_rank")
    real_winners = real.loc[
        real.groupby(["dataset", "seed"])["reference_nrmse"].idxmin(),
        ["dataset", "seed", "method", "reference_nrmse"],
    ]
    print(
        f"COMPLETE_BLOCKS synthetic={synthetic_blocks} real={real_blocks} "
        f"balanced_rows synthetic={len(synthetic)} real={len(real)}"
    )
    print("\nSYNTHETIC_PREDICTION\n", synthetic_prediction.to_string())
    print("\nSYNTHETIC_CONTINUITY\n", synthetic_continuity.to_string())
    print("\nSYNTHETIC_WINNERS\n", synthetic_winners.to_string(index=False))
    print("\nREAL_PREDICTION\n", real_prediction.to_string())
    print("\nREAL_CONTINUITY\n", real_continuity.to_string())
    print("\nREAL_WINNERS\n", real_winners.to_string(index=False))


if __name__ == "__main__":
    main()
