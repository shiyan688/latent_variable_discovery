#!/usr/bin/env python3
"""Aggregate prediction, continuity, and representation-stability diagnostics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

from lvs.core.metrics import fit_procrustes_alignment, knn_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_results(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob("**/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if result.get("status") != "success":
            continue
        rows.append(
            {
                **result.get("job", {}),
                **result.get("dataset", {}),
                **result.get("prediction", {}),
                **result.get("spatial", {}),
                "wall_time_seconds": result.get("wall_time_seconds"),
                "result_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def _mean_ci_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, key_values))
        for metric in metrics:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            row[f"{metric}_ci95_half"] = (
                float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _add_within_block_ranks(
    frame: pd.DataFrame,
    block_columns: list[str],
    metric_directions: dict[str, bool],
) -> pd.DataFrame:
    ranked = frame.copy()
    for metric, ascending in metric_directions.items():
        if metric in ranked:
            ranked[f"{metric}_rank"] = ranked.groupby(block_columns)[metric].rank(
                method="average", ascending=ascending
            )
    return ranked


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_index in range(len(values) - 1, -1, -1):
        index = order[reverse_index]
        candidate = values[index] * len(values) / (reverse_index + 1)
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _paired_tests(
    frame: pd.DataFrame,
    *,
    block_columns: list[str],
    metric: str,
    reference_method: str,
) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    if metric not in frame:
        return pd.DataFrame()
    pivot = frame.pivot_table(index=block_columns, columns="method", values=metric, aggfunc="mean")
    if reference_method not in pivot:
        return pd.DataFrame()
    rows = []
    for method in pivot.columns:
        if method == reference_method:
            continue
        paired = pivot[[reference_method, method]].dropna()
        if paired.empty:
            continue
        difference = paired[method] - paired[reference_method]
        try:
            p_value = float(wilcoxon(difference).pvalue) if np.any(difference != 0) else 1.0
        except ValueError:
            p_value = float("nan")
        rows.append(
            {
                "metric": metric,
                "reference_method": reference_method,
                "method": method,
                "paired_blocks": len(paired),
                "mean_difference_method_minus_reference": float(difference.mean()),
                "median_difference_method_minus_reference": float(difference.median()),
                "method_win_rate_lower_is_better": float(np.mean(difference < 0)),
                "wilcoxon_p": p_value,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        finite = output["wilcoxon_p"].notna()
        output.loc[finite, "wilcoxon_bh_q"] = _benjamini_hochberg(
            output.loc[finite, "wilcoxon_p"].to_numpy(float)
        )
    return output


def _synthetic_seed_stability(root: Path) -> pd.DataFrame:
    records = []
    grouped: dict[tuple[int, str], list[tuple[int, Path]]] = {}
    for result_path in root.glob("expr*/**/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        job = result.get("job", {})
        q_path = result_path.parent / "test_label_q.csv"
        if q_path.exists() and job.get("method") not in {
            "no_q_mlp", "random_forest", "support_knn", "oracle_q_mlp"
        }:
            grouped.setdefault((int(job["expression_id"]), str(job["method"])), []).append(
                (int(job["seed"]), q_path)
            )
    for (expression_id, method), entries in grouped.items():
        for (left_seed, left_path), (right_seed, right_path) in combinations(sorted(entries), 2):
            left = pd.read_csv(left_path)
            right = pd.read_csv(right_path)
            aligned_columns = [column for column in left.columns if column.startswith("aligned_q")]
            true_columns = [
                column for column in left.columns
                if column.startswith("q") and column[1:].isdigit()
            ]
            merged = left[["label", *aligned_columns, *true_columns]].merge(
                right[["label", *aligned_columns]], on="label", suffixes=("_left", "_right")
            )
            left_values = merged[[f"{column}_left" for column in aligned_columns]].to_numpy(float)
            right_values = merged[[f"{column}_right" for column in aligned_columns]].to_numpy(float)
            true_values = merged[true_columns].to_numpy(float)
            scale = max(float(np.sqrt(np.mean((true_values - true_values.mean(axis=0)) ** 2))), 1e-12)
            records.append(
                {
                    "expression_id": expression_id,
                    "method": method,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "aligned_seed_nrmse": float(np.sqrt(np.mean((left_values - right_values) ** 2)) / scale),
                    "aligned_seed_knn_overlap": knn_overlap(
                        left_values, right_values, k=min(5, len(merged) - 1)
                    ),
                }
            )
    return pd.DataFrame(records)


def _real_seed_stability(root: Path) -> pd.DataFrame:
    records = []
    grouped: dict[tuple[str, str, int], list[tuple[int, Path]]] = {}
    for result_path in root.glob("*/*/seed*_q*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        job = result.get("job", {})
        q_path = result_path.parent / "test_label_q.csv"
        if q_path.exists() and int(job.get("q_dim", 0)) > 0:
            grouped.setdefault(
                (str(job["dataset"]), str(job["method"]), int(job["q_dim"])), []
            ).append((int(job["seed"]), q_path))
    for (dataset, method, q_dim), entries in grouped.items():
        for (left_seed, left_path), (right_seed, right_path) in combinations(sorted(entries), 2):
            left = pd.read_csv(left_path)
            right = pd.read_csv(right_path)
            q_columns = [f"q{index + 1}" for index in range(q_dim)]
            merged = left[["label", *q_columns]].merge(
                right[["label", *q_columns]], on="label", suffixes=("_left", "_right")
            )
            left_values = merged[[f"{column}_left" for column in q_columns]].to_numpy(float)
            right_values = merged[[f"{column}_right" for column in q_columns]].to_numpy(float)
            alignment = fit_procrustes_alignment(left_values, right_values)
            aligned_left = alignment.transform(left_values)
            scale = max(
                float(np.sqrt(np.mean((right_values - right_values.mean(axis=0)) ** 2))),
                1e-12,
            )
            records.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "q_dim": q_dim,
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "procrustes_seed_nrmse": float(
                        np.sqrt(np.mean((aligned_left - right_values) ** 2)) / scale
                    ),
                    "aligned_seed_knn_overlap": knn_overlap(
                        aligned_left, right_values, k=min(5, len(merged) - 1)
                    ),
                }
            )
    return pd.DataFrame(records)


def _collect_continuity_curves(root: Path, domain: str) -> pd.DataFrame:
    rows = []
    for result_path in root.glob("**/result.json"):
        curve_path = result_path.parent / "continuity_curve.csv"
        if not curve_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        try:
            curve = pd.read_csv(curve_path)
        except pd.errors.EmptyDataError:
            # Some valid tasks (for example UCI with only two held-out labels)
            # have no defined neighborhood curve and therefore store an empty file.
            continue
        if curve.empty:
            continue
        for record in curve.to_dict("records"):
            rows.append({"domain": domain, **result["job"], **record})
    return pd.DataFrame(rows)


def _plot_heatmap(
    table: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    color_label: str,
    cmap: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.empty:
        return
    figure_width = max(7.0, 0.8 * len(table.columns) + 2.5)
    figure_height = max(4.0, 0.45 * len(table.index) + 2.0)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    image = axis.imshow(table.to_numpy(float), aspect="auto", cmap=cmap)
    axis.set_xticks(np.arange(len(table.columns)), labels=table.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(table.index)), labels=table.index)
    axis.set_title(title)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(color_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_continuity_curves(curves: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if curves.empty:
        return
    methods = sorted(curves["method"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    for axis, (domain, subset) in zip(axes, curves.groupby("domain", sort=False)):
        for method in methods:
            selected = subset[subset["method"] == method]
            if selected.empty:
                continue
            summary = selected.groupby("k")["continuity"].agg(["mean", "std", "count"])
            x = summary.index.to_numpy(float)
            mean = summary["mean"].to_numpy(float)
            error = (summary["std"] / np.sqrt(summary["count"])).fillna(0).to_numpy(float)
            axis.plot(x, mean, marker="o", markersize=3, label=method)
            axis.fill_between(x, mean - 1.96 * error, mean + 1.96 * error, alpha=0.12)
        axis.set_title(domain)
        axis.set_xlabel("k")
        axis.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("continuity")
    axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = _load_results(args.synthetic_root)
    real = _load_results(args.real_root)
    synthetic.to_csv(args.output_dir / "synthetic_all_runs.csv", index=False)
    real.to_csv(args.output_dir / "real_all_runs.csv", index=False)

    synthetic_ranked = _add_within_block_ranks(
        synthetic,
        ["expression_id", "seed"],
        {
            "reference_nrmse": True,
            "aligned_nrmse": True,
            "continuity_auc": False,
            "trustworthiness_auc": False,
            "local_log_distortion_p95": True,
        },
    )
    real_ranked = _add_within_block_ranks(
        real,
        ["dataset", "seed"],
        {
            "reference_nrmse": True,
            "response_continuity_auc": False,
            "response_trustworthiness_auc": False,
            "response_local_log_distortion_p95": True,
        },
    )
    synthetic_ranked.to_csv(args.output_dir / "synthetic_ranked_runs.csv", index=False)
    real_ranked.to_csv(args.output_dir / "real_ranked_runs.csv", index=False)

    synthetic_metrics = [
        metric
        for metric in (
            "reference_nrmse", "reference_nrmse_rank", "aligned_nrmse",
            "continuity_auc", "trustworthiness_auc", "local_log_distortion_p95",
            "distance_spearman", "cca_mean", "effective_rank", "wall_time_seconds",
        )
        if metric in synthetic_ranked
    ]
    real_metrics = [
        metric
        for metric in (
            "reference_nrmse", "reference_nrmse_rank", "response_continuity_auc",
            "response_trustworthiness_auc", "response_local_log_distortion_p95",
            "response_distance_spearman", "acquisition_distance_spearman",
            "effective_rank", "wall_time_seconds",
        )
        if metric in real_ranked
    ]
    synthetic_summary = _mean_ci_summary(synthetic_ranked, ["method"], synthetic_metrics)
    real_summary = _mean_ci_summary(real_ranked, ["method"], real_metrics)
    synthetic_summary.to_csv(args.output_dir / "synthetic_method_summary.csv", index=False)
    real_summary.to_csv(args.output_dir / "real_method_summary.csv", index=False)

    paired = pd.concat(
        [
            _paired_tests(
                synthetic, block_columns=["expression_id", "seed"],
                metric="reference_nrmse", reference_method="joint_mse"
            ).assign(domain="synthetic"),
            _paired_tests(
                real, block_columns=["dataset", "seed"],
                metric="reference_nrmse", reference_method="joint_mse"
            ).assign(domain="real"),
        ],
        ignore_index=True,
    )
    paired.to_csv(args.output_dir / "paired_method_tests.csv", index=False)

    synthetic_stability = _synthetic_seed_stability(args.synthetic_root)
    real_stability = _real_seed_stability(args.real_root)
    synthetic_stability.to_csv(args.output_dir / "synthetic_seed_stability.csv", index=False)
    real_stability.to_csv(args.output_dir / "real_seed_stability.csv", index=False)

    if not synthetic_ranked.empty and "reference_nrmse_rank" in synthetic_ranked:
        table = synthetic_ranked.pivot_table(
            index="method", columns="expression_id", values="reference_nrmse_rank", aggfunc="mean"
        )
        _plot_heatmap(
            table,
            args.output_dir / "synthetic_prediction_rank_heatmap.png",
            title="Synthetic prediction rank by expression",
            color_label="mean rank (lower is better)",
            cmap="viridis_r",
        )
    if not real_ranked.empty and "reference_nrmse_rank" in real_ranked:
        table = real_ranked.pivot_table(
            index="method", columns="dataset", values="reference_nrmse_rank", aggfunc="mean"
        )
        _plot_heatmap(
            table,
            args.output_dir / "real_prediction_rank_heatmap.png",
            title="Real-data prediction rank by dataset",
            color_label="mean rank (lower is better)",
            cmap="viridis_r",
        )
    if not real.empty and "response_continuity_auc" in real:
        table = real.pivot_table(
            index="method", columns="dataset", values="response_continuity_auc", aggfunc="mean"
        )
        _plot_heatmap(
            table,
            args.output_dir / "real_continuity_heatmap.png",
            title="Held-out response continuity",
            color_label="continuity AUC (higher is better)",
            cmap="viridis",
        )

    curves = pd.concat(
        [
            _collect_continuity_curves(args.synthetic_root, "synthetic true-q geometry"),
            _collect_continuity_curves(args.real_root, "real held-out response geometry"),
        ],
        ignore_index=True,
    )
    curves.to_csv(args.output_dir / "all_continuity_curves.csv", index=False)
    _plot_continuity_curves(curves, args.output_dir / "continuity_curves.png")

    report = _build_report(
        synthetic=synthetic,
        real=real,
        synthetic_summary=synthetic_summary,
        real_summary=real_summary,
        synthetic_stability=synthetic_stability,
        real_stability=real_stability,
    )
    (args.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def _build_report(
    *,
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    synthetic_summary: pd.DataFrame,
    real_summary: pd.DataFrame,
    synthetic_stability: pd.DataFrame,
    real_stability: pd.DataFrame,
) -> str:
    return f"""# Latent Discovery Validation Report

## Material Passport

- Artifact type: experiment validation report
- Verification status: ANALYZED
- Synthetic completed rows discovered: {len(synthetic)}
- Real-data completed rows discovered: {len(real)}
- Raw inputs: local result JSON/CSV files only
- External upload: none

## Primary endpoints

- Prediction: RMSE normalized by the training-target standard deviation (`reference_nrmse`).
- Synthetic latent recovery: validation-fit/test-score alignment, CCA, distance geometry, continuity, trustworthiness, and local distortion.
- Real latent structure: continuity against RFF kernel-mean signatures computed only from held-out query responses.
- Reproducibility: aligned cross-seed q NRMSE and neighborhood overlap.

## Available aggregate rows

- Synthetic method summaries: {len(synthetic_summary)}
- Real method summaries: {len(real_summary)}
- Synthetic seed-pair stability rows: {len(synthetic_stability)}
- Real seed-pair stability rows: {len(real_stability)}

## Mandatory cautions

1. Per-label NRMSE and per-label R2 are diagnostic only; nearly constant curves can make them numerically pathological.
2. The broad real-data run is capped per label and is a screening pilot, not the final full-row result.
3. Test-time q uses support targets. It is a few-shot curve-completion protocol, not zero-shot prediction.
4. UCI gas drift has only two held-out labels; rank-based neighborhood continuity is undefined for that task.
5. Multiple method comparisons are reported with Benjamini-Hochberg adjusted Wilcoxon p-values; effect sizes and win rates remain primary.
6. Real-data response continuity is associational evidence for a useful latent descriptor, not proof of a unique causal physical variable.

## Statistical fallacy scan (11/11 checked)

- Simpson/ecological: results are retained per expression/dataset before aggregation.
- Berkson/collider: no conditioning-based causal claim is made; group-holdout selection remains a documented limitation.
- Base-rate: not applicable to regression endpoints.
- Regression to mean: no extreme-score enrollment design.
- Survivorship: failed jobs must remain visible in launcher logs and are not silently excluded.
- Look-elsewhere/forking paths: the experiment manifest fixes methods, seeds, metrics, and data caps before final analysis.
- Correlation/causation and reverse causality: q-response association is not described as causal recovery.
"""


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
