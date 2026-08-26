#!/usr/bin/env python3
"""Create paired prediction/geometry tables for the controlled loss ablation."""
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

BASELINE = "joint_mse"
METHOD_ORDER = (
    "joint_mse",
    "joint_lb_mse",
    "joint_hsic",
    "joint_continuity",
    "joint_q_l2",
    "joint_calprior",
    "joint_hsic_cont",
    "joint_all_mse",
    "joint_fixed",
    "joint_dynamic",
)
DISPLAY_NAMES = {
    "joint_mse": "MSE",
    "joint_lb_mse": "Label-balanced",
    "joint_hsic": "+ HSIC",
    "joint_continuity": "+ continuity",
    "joint_q_l2": "+ q-L2",
    "joint_calprior": "+ calibration prior",
    "joint_hsic_cont": "+ HSIC + continuity",
    "joint_all_mse": "All constraints (MSE)",
    "joint_fixed": "All + label-balanced",
    "joint_dynamic": "All + adaptive weights",
}
METRICS = {
    "reference_nrmse": True,
    "continuity_auc": False,
    "local_log_distortion_p95": True,
    "effective_rank": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _record(path: Path, kind: str) -> dict[str, Any] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "success":
        return None
    job = payload["job"]
    prediction = payload.get("prediction", {})
    spatial = payload.get("spatial", {})
    real = kind == "real"
    dataset = job["dataset"] if real else f"expr{int(job['expression_id']):03d}"
    prefix = "response_" if real else ""
    return {
        "kind": kind,
        "dataset": dataset,
        "method": job["method"],
        "loss_preset": job.get("loss_preset"),
        "seed": job["seed"],
        "q_dim": job.get("q_dim"),
        "reference_nrmse": prediction.get("reference_nrmse"),
        "macro_nrmse": prediction.get("macro_nrmse"),
        "continuity_auc": spatial.get(f"{prefix}continuity_auc"),
        "trustworthiness_auc": spatial.get(f"{prefix}trustworthiness_auc"),
        "knn_overlap_auc": spatial.get(f"{prefix}knn_overlap_auc"),
        "local_log_distortion_p95": spatial.get(
            f"{prefix}local_log_distortion_p95"
        ),
        "distance_spearman": spatial.get(f"{prefix}distance_spearman"),
        "effective_rank": spatial.get("effective_rank"),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        "result_path": str(path),
    }


def load_runs(synthetic_root: Path, real_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root, kind in ((synthetic_root, "synthetic"), (real_root, "real")):
        for path in sorted(root.glob("**/result.json")):
            row = _record(path, kind)
            if row is not None and row["method"] in METHOD_ORDER:
                rows.append(row)
    return pd.DataFrame(rows)


def _wilcoxon(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or np.allclose(finite, 0.0):
        return 1.0
    return float(wilcoxon(finite, zero_method="wilcox").pvalue)


def _bh_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    for rank_index in range(len(array) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, array[original_index] * len(array) / rank)
        adjusted[original_index] = running
    return pd.Series(adjusted, index=values.index)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_kind in ("synthetic", "real"):
        kind_frame = frame[frame["kind"] == scope_kind]
        scopes = [("all", kind_frame), *[
            (dataset, kind_frame[kind_frame["dataset"] == dataset])
            for dataset in sorted(kind_frame["dataset"].unique())
        ]]
        for scope, selected in scopes:
            for method in METHOD_ORDER[1:]:
                pair_frame = selected[selected["method"].isin([BASELINE, method])]
                for metric, lower_is_better in METRICS.items():
                    wide = pair_frame.pivot_table(
                        index=["dataset", "seed"],
                        columns="method",
                        values=metric,
                        aggfunc="first",
                    )
                    if BASELINE not in wide or method not in wide:
                        continue
                    wide = wide.dropna(subset=[BASELINE, method])
                    candidate = wide[method].to_numpy(float)
                    baseline = wide[BASELINE].to_numpy(float)
                    raw_delta = candidate - baseline
                    oriented = -raw_delta if lower_is_better else raw_delta
                    rows.append(
                        {
                            "kind": scope_kind,
                            "scope": scope,
                            "method": method,
                            "metric": metric,
                            "pairs": len(wide),
                            "candidate_median": np.median(candidate),
                            "baseline_median": np.median(baseline),
                            "median_delta_candidate_minus_baseline": np.median(raw_delta),
                            "median_oriented_improvement": np.median(oriented),
                            "win_rate": np.mean(oriented > 0),
                            "wilcoxon_p": _wilcoxon(raw_delta),
                        }
                    )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["bh_q_within_kind_scope_metric"] = output.groupby(
            ["kind", "scope", "metric"]
        )["wilcoxon_p"].transform(_bh_adjust)
    return output


def method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [*METRICS, "macro_nrmse", "wall_time_seconds"]
    return (
        frame.groupby(["kind", "dataset", "method"], as_index=False)[metrics]
        .agg(["count", "mean", "median", "std"])
        .pipe(_flatten_columns)
    )


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.5g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _plot(effects: pd.DataFrame, output_path: Path) -> None:
    selected = effects[(effects["scope"] == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {"synthetic": "#2878B5", "real": "#D95F02"}
    for axis, kind in zip(axes, ("synthetic", "real")):
        subset = selected[selected["kind"] == kind]
        prediction = subset[subset["metric"] == "reference_nrmse"].set_index("method")
        continuity = subset[subset["metric"] == "continuity_auc"].set_index("method")
        methods = [method for method in METHOD_ORDER[1:] if method in prediction.index]
        x = [prediction.loc[method, "median_delta_candidate_minus_baseline"] for method in methods]
        y = [continuity.loc[method, "median_delta_candidate_minus_baseline"] for method in methods]
        axis.axvline(0, color="#999999", linewidth=0.8)
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.scatter(x, y, color=colors[kind], s=45)
        for method, x_value, y_value in zip(methods, x, y):
            axis.annotate(DISPLAY_NAMES[method], (x_value, y_value), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set_xlabel("Δ reference NRMSE vs MSE (lower is better)")
        axis.set_ylabel("Δ continuity AUC vs MSE (higher is better)")
        axis.set_title(kind.capitalize())
        axis.grid(alpha=0.2)
    fig.suptitle("Loss-component prediction–geometry trade-off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = load_runs(args.synthetic_root, args.real_root)
    expected = 2 * 3 * len(METHOD_ORDER) * 3
    if not args.allow_incomplete and len(frame) != expected:
        raise RuntimeError(f"Expected {expected} completed rows, found {len(frame)}.")
    summary = method_summary(frame)
    effects = paired_effects(frame)
    frame.to_csv(args.output_root / "loss_ablation_all_runs.csv", index=False)
    summary.to_csv(args.output_root / "loss_ablation_dataset_summary.csv", index=False)
    effects.to_csv(args.output_root / "loss_ablation_paired_effects.csv", index=False)
    figure_path = args.output_root / "loss_ablation_tradeoff.png"
    if not effects.empty:
        _plot(effects, figure_path)

    compact = summary[[
        "kind",
        "dataset",
        "method",
        "reference_nrmse_median",
        "continuity_auc_median",
        "local_log_distortion_p95_median",
        "effective_rank_median",
    ]]
    effect_columns = [
        "kind",
        "method",
        "metric",
        "pairs",
        "median_delta_candidate_minus_baseline",
        "win_rate",
        "wilcoxon_p",
        "bh_q_within_kind_scope_metric",
    ]
    if effects.empty:
        global_effects = pd.DataFrame(columns=effect_columns)
    else:
        global_effects = effects[
            (effects["scope"] == "all")
            & effects["metric"].isin(
                ["reference_nrmse", "continuity_auc", "local_log_distortion_p95"]
            )
        ][effect_columns]
    report = [
        "# Controlled loss-component ablation",
        "",
        "All candidates use the same data, model, joint optimization schedule, promoted K=4 calibration, and seeds. Only the named loss component or loss weighting changes. Negative Δ NRMSE/distortion is favorable; positive Δ continuity is favorable.",
        "",
        "## Per-dataset medians",
        "",
        _markdown_table(compact),
        "",
        "## Paired effects against joint MSE",
        "",
        _markdown_table(global_effects),
        "",
        "## Artifacts",
        "",
        f"- Trade-off figure: `{figure_path.name}` and `{figure_path.with_suffix('.pdf').name}`.",
        "- Machine-readable run, summary, and paired-effect CSV files are in this directory.",
        "",
    ]
    report_path = args.output_root / "LOSS_COMPONENT_ABLATION_REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(report), encoding="utf-8")
    temporary.replace(report_path)
    print(report_path)


if __name__ == "__main__":
    main()
