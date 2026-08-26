#!/usr/bin/env python3
"""Create paired, task-stratified reports for calibration strategy studies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BASELINE = "legacy_k1_s200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--additional-input-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--study", choices=("synthetic", "real"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, clean.size, size=(samples, clean.size))
    medians = np.median(clean[indices], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _wilcoxon_pvalue(differences: np.ndarray) -> float:
    clean = np.asarray(differences, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0 or np.allclose(clean, 0.0):
        return 1.0
    return float(wilcoxon(clean, alternative="two-sided").pvalue)


def _study_spec(study: str) -> tuple[list[str], str, list[tuple[str, str]]]:
    if study == "synthetic":
        return (
            ["expression_id", "method", "seed", "q_dim"],
            "expression_id",
            [
                ("reference_nrmse", "lower"),
                ("label_reference_nrmse_p95", "lower"),
                ("aligned_nrmse", "lower"),
                ("continuity_auc", "higher"),
                ("local_log_distortion_p95", "lower"),
            ],
        )
    return (
        ["dataset", "method", "seed", "q_dim"],
        "dataset",
        [
            ("reference_nrmse", "lower"),
            ("label_reference_nrmse_p95", "lower"),
            ("response_continuity_auc", "higher"),
            ("response_local_log_distortion_p95", "lower"),
        ],
    )


def analyze(args: argparse.Namespace) -> pd.DataFrame:
    sources = [
        args.input_root / "all_strategy_runs.csv",
        *[root / "all_strategy_runs.csv" for root in args.additional_input_root],
    ]
    frame = pd.concat([pd.read_csv(source) for source in sources], ignore_index=True)
    block_columns, group_column, requested_metrics = _study_spec(args.study)
    metrics = [(name, direction) for name, direction in requested_metrics if name in frame]
    missing_blocks = set(block_columns) - set(frame)
    if missing_blocks:
        raise ValueError(f"Missing block columns: {sorted(missing_blocks)}")
    baseline = frame[frame["strategy"] == BASELINE]
    if baseline.empty:
        raise ValueError(f"No {BASELINE!r} rows found in {sources}.")

    rows: list[dict[str, object]] = []
    strategies = sorted(set(frame["strategy"]) - {BASELINE})
    group_values: list[object] = ["ALL", *sorted(frame[group_column].unique())]
    for strategy_index, strategy in enumerate(strategies):
        paired = baseline.merge(
            frame[frame["strategy"] == strategy],
            on=block_columns,
            suffixes=("_baseline", "_comparison"),
        )
        for group_index, group in enumerate(group_values):
            selected = paired if group == "ALL" else paired[paired[group_column] == group]
            for metric_index, (metric, direction) in enumerate(metrics):
                baseline_values = selected[f"{metric}_baseline"].to_numpy(float)
                comparison_values = selected[f"{metric}_comparison"].to_numpy(float)
                raw_difference = comparison_values - baseline_values
                oriented_difference = (
                    raw_difference if direction == "lower" else -raw_difference
                )
                relative_change = raw_difference / np.maximum(
                    np.abs(baseline_values), 1e-12
                )
                oriented_relative_change = (
                    relative_change if direction == "lower" else -relative_change
                )
                ci_low, ci_high = _bootstrap_median_ci(
                    oriented_relative_change,
                    samples=args.bootstrap_samples,
                    seed=args.seed
                    + 10_000 * strategy_index
                    + 100 * group_index
                    + metric_index,
                )
                rows.append(
                    {
                        "study": args.study,
                        "group": group,
                        "strategy": strategy,
                        "metric": metric,
                        "direction": direction,
                        "paired_blocks": int(len(selected)),
                        "baseline_mean": float(np.mean(baseline_values)),
                        "comparison_mean": float(np.mean(comparison_values)),
                        "mean_raw_difference": float(np.mean(raw_difference)),
                        "median_raw_difference": float(np.median(raw_difference)),
                        "median_oriented_relative_change_pct": float(
                            100.0 * np.median(oriented_relative_change)
                        ),
                        "median_oriented_relative_ci95_low_pct": 100.0 * ci_low,
                        "median_oriented_relative_ci95_high_pct": 100.0 * ci_high,
                        "win_rate": float(np.mean(oriented_difference < 0)),
                        "wilcoxon_pvalue": _wilcoxon_pvalue(raw_difference),
                    }
                )
    return pd.DataFrame(rows)


def _percent(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.2f}%"


def write_report(args: argparse.Namespace, effects: pd.DataFrame) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effects.to_csv(args.output_dir / "paired_effects.csv", index=False)
    primary = effects[effects["metric"] == "reference_nrmse"].copy()
    lines = [
        "# Calibration strategy paired analysis",
        "",
        f"- Study: `{args.study}`",
        "- Inputs: "
        + ", ".join(
            f"`{root / 'all_strategy_runs.csv'}`"
            for root in [args.input_root, *args.additional_input_root]
        ),
        f"- Baseline: `{BASELINE}`",
        "- Relative change is direction-oriented: negative is better.",
        "- Wilcoxon p-values are exploratory and unadjusted.",
        "",
        "## Prediction NRMSE",
        "",
        "| Group | Strategy | n | Median relative change | 95% bootstrap CI | Win rate | Wilcoxon p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            "| {group} | {strategy} | {n} | {effect} | [{low}, {high}] | "
            "{wins} | {p:.4g} |".format(
                group=row.group,
                strategy=row.strategy,
                n=row.paired_blocks,
                effect=_percent(row.median_oriented_relative_change_pct),
                low=_percent(row.median_oriented_relative_ci95_low_pct),
                high=_percent(row.median_oriented_relative_ci95_high_pct),
                wins=_percent(100.0 * row.win_rate),
                p=row.wilcoxon_pvalue,
            )
        )
    (args.output_dir / "analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    effects = analyze(args)
    write_report(args, effects)


if __name__ == "__main__":
    main()
