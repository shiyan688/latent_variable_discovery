#!/usr/bin/env python3
"""Analyze the confirmatory PDEBench Burgers latent-variable study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ADAPTIVE = "latent_adaptive_k4_min24"
LEGACY = "latent_legacy_k1"
LOWER_IS_BETTER = {
    "reference_nrmse": True,
    "label_reference_nrmse_p95": True,
    "aligned_nrmse": True,
    "distance_spearman": False,
    "continuity_auc": False,
    "local_log_distortion_p95": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _wilcoxon_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) < 2 or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def paired_comparison(
    frame: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
    keys: list[str],
    lower_is_better: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = frame[frame["strategy"].isin([left, right])]
    wide = selected.pivot_table(index=keys, columns="strategy", values=metric, aggfunc="first")
    wide = wide.dropna(subset=[left, right])
    left_values = wide[left].to_numpy(float)
    right_values = wide[right].to_numpy(float)
    raw_difference = left_values - right_values
    oriented = right_values - left_values if lower_is_better else left_values - right_values
    denominator = np.maximum(np.abs(right_values), 1e-12)
    relative = 100.0 * oriented / denominator
    ci_low, ci_high = _bootstrap_median_ci(
        relative,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "pairs": int(len(wide)),
        "left_median": float(np.median(left_values)),
        "right_median": float(np.median(right_values)),
        "median_oriented_relative_improvement_percent": float(np.median(relative)),
        "bootstrap_95_ci_percent": [ci_low, ci_high],
        "win_rate": float(np.mean(oriented > 0.0)),
        "wilcoxon_p": _wilcoxon_p(raw_difference),
    }


def _format_effect(effect: dict[str, Any]) -> str:
    low, high = effect["bootstrap_95_ci_percent"]
    return (
        f"{effect['left']} vs {effect['right']} on {effect['metric']}: "
        f"{effect['pairs']} pairs; median oriented improvement "
        f"{effect['median_oriented_relative_improvement_percent']:.2f}% "
        f"(95% bootstrap CI {low:.2f}% to {high:.2f}%); "
        f"wins {100.0 * effect['win_rate']:.1f}%; Wilcoxon p={effect['wilcoxon_p']:.4g}."
    )


def _dimension_table(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[frame["strategy"] == ADAPTIVE]
    metrics = [column for column in LOWER_IS_BETTER if column in selected]
    return selected.groupby("q_dim", as_index=False)[metrics].agg("median")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _choose_dimension(dimension_table: pd.DataFrame) -> tuple[int, float]:
    best = float(dimension_table["reference_nrmse"].min())
    candidates = dimension_table[
        dimension_table["reference_nrmse"] <= 1.05 * best
    ]["q_dim"]
    return int(candidates.min()), best


def _plot(frame: pd.DataFrame, output_path: Path) -> None:
    latent = frame[frame["strategy"].isin([LEGACY, ADAPTIVE])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    panels = (
        ("reference_nrmse", "Query reference NRMSE", True),
        ("label_reference_nrmse_p95", "Trajectory p95 NRMSE", True),
        ("continuity_auc", "Initial-condition continuity AUC", False),
    )
    colors = {LEGACY: "#8C8C8C", ADAPTIVE: "#2878B5"}
    labels = {LEGACY: "Legacy K=1", ADAPTIVE: "Adaptive K=4"}
    markers = {"joint_mse": "o", "alternating_mse": "s"}
    for axis, (metric, title, _) in zip(axes.flat[:3], panels):
        for strategy in (LEGACY, ADAPTIVE):
            for method in ("joint_mse", "alternating_mse"):
                selected = latent[
                    (latent["strategy"] == strategy) & (latent["method"] == method)
                ]
                summary = selected.groupby("q_dim")[metric].agg(["mean", "std"]).reset_index()
                if summary.empty:
                    continue
                axis.errorbar(
                    summary["q_dim"],
                    summary["mean"],
                    yerr=summary["std"].fillna(0.0),
                    color=colors[strategy],
                    marker=markers[method],
                    linestyle="-" if strategy == ADAPTIVE else "--",
                    linewidth=1.5,
                    capsize=3,
                    label=f"{labels[strategy]}, {method.replace('_mse', '')}",
                )
        axis.set_title(title)
        axis.set_xlabel("Latent dimension")
        axis.grid(alpha=0.22)
    axes.flat[0].set_ylabel("Lower is better")
    axes.flat[2].set_ylabel("Higher is better")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[0].legend(handles, legend_labels, fontsize=7, frameon=False)

    baseline_axis = axes.flat[3]
    baseline_order = [
        "pooled_mlp_no_latent",
        "support_mean",
        "support_knn4",
        ADAPTIVE,
        "full_ic_pca_mlp_reference",
    ]
    baseline = frame[
        (frame["q_dim"] == 8)
        & (frame["method"] == "joint_mse")
        & frame["strategy"].isin(baseline_order)
    ]
    values = baseline.groupby("strategy")["reference_nrmse"].agg(["mean", "std"])
    present = [name for name in baseline_order if name in values.index]
    display = [
        "No latent MLP",
        "Support mean",
        "Support KNN-4",
        "Latent adaptive",
        "Full-IC PCA-MLP*",
    ]
    display = [display[baseline_order.index(name)] for name in present]
    positions = np.arange(len(present))
    baseline_axis.bar(
        positions,
        values.loc[present, "mean"],
        yerr=values.loc[present, "std"].fillna(0.0),
        color=["#B8B8B8", "#D9A441", "#E07B39", "#2878B5", "#5B8C5A"][: len(present)],
        capsize=3,
    )
    baseline_axis.set_xticks(positions, display, rotation=25, ha="right", fontsize=8)
    baseline_axis.set_ylabel("Query reference NRMSE")
    baseline_axis.set_title("q=8 joint: baselines and reference")
    baseline_axis.grid(axis="y", alpha=0.22)
    fig.suptitle("PDEBench Burgers sparse-support latent study", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    all_runs_path = args.output_root / "all_runs.csv"
    frame = pd.read_csv(all_runs_path)
    latent = frame[frame["strategy"].isin([ADAPTIVE, LEGACY])]
    expected_latent_rows = 3 * 2 * 3 * 2
    if len(latent) != expected_latent_rows:
        raise RuntimeError(
            f"Expected {expected_latent_rows} latent rows, found {len(latent)}; analysis aborted."
        )

    calibration_effects = [
        paired_comparison(
            frame,
            left=ADAPTIVE,
            right=LEGACY,
            metric=metric,
            keys=["q_dim", "method", "seed"],
            lower_is_better=lower_is_better,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index,
        )
        for index, (metric, lower_is_better) in enumerate(LOWER_IS_BETTER.items())
    ]
    method_effect = paired_comparison(
        frame[frame["strategy"] == ADAPTIVE].assign(strategy=frame["method"]),
        left="alternating_mse",
        right="joint_mse",
        metric="reference_nrmse",
        keys=["q_dim", "seed"],
        lower_is_better=True,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 100,
    )
    baseline_effects = []
    for index, baseline in enumerate(
        ("pooled_mlp_no_latent", "support_mean", "support_knn4")
    ):
        selected = frame[(frame["q_dim"] == 8) & (frame["method"] == "joint_mse")]
        baseline_effects.append(
            paired_comparison(
                selected,
                left=ADAPTIVE,
                right=baseline,
                metric="reference_nrmse",
                keys=["seed"],
                lower_is_better=True,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed + 200 + index,
            )
        )
    dimensions = _dimension_table(frame)
    chosen_dimension, best_nrmse = _choose_dimension(dimensions)
    figure_path = args.output_root / "pdebench_burgers_diagnostics.png"
    _plot(frame, figure_path)

    analysis = {
        "status": "success",
        "latent_rows": int(len(latent)),
        "calibration_effects": calibration_effects,
        "method_effect": method_effect,
        "baseline_effects": baseline_effects,
        "dimension_medians": dimensions.to_dict(orient="records"),
        "selected_dimension_by_5pct_rule": chosen_dimension,
        "best_dimension_median_reference_nrmse": best_nrmse,
        "figure": str(figure_path),
    }
    temporary = (args.output_root / "analysis.json.tmp")
    temporary.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    temporary.replace(args.output_root / "analysis.json")

    lines = [
        "# PDEBench Burgers latent-study report",
        "",
        "The report covers 18 jobs (three q dimensions, two training schedules, and three seeds).",
        "The full-initial-condition PCA-MLP receives extra information and is not ranked as a fair sparse-support competitor.",
        "",
        "## Frozen calibration versus legacy",
        "",
        *[f"- {_format_effect(effect)}" for effect in calibration_effects],
        "",
        "## Training schedule",
        "",
        f"- {_format_effect(method_effect)}",
        "",
        "## Same-information baselines at q=8, joint training",
        "",
        *[f"- {_format_effect(effect)}" for effect in baseline_effects],
        "",
        "## Dimension choice",
        "",
        f"The best median query NRMSE is {best_nrmse:.6g}. Under the predeclared within-5% rule, q={chosen_dimension} is selected.",
        "",
        _markdown_table(dimensions),
        "",
        "## Artifacts",
        "",
        f"- Diagnostic figure: `{figure_path.name}` and `{figure_path.with_suffix('.pdf').name}`.",
        "- Machine-readable analysis: `analysis.json`.",
        "- Per-run table: `all_runs.csv`.",
        "",
    ]
    report_path = args.output_root / "analysis_report.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text("\n".join(lines), encoding="utf-8")
    temporary_report.replace(report_path)
    print(report_path)


if __name__ == "__main__":
    main()
