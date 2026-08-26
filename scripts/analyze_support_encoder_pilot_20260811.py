#!/usr/bin/env python3
"""Reconcile and summarize every raw result from the support-encoder pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_encoder_pilot_20260811"
METHOD_ORDER = (
    "encoder_q_refine",
    "deepsets_direct",
    "joint_continuity",
    "no_q_mlp",
    "random_forest",
    "support_knn",
)
DATASET_ORDER = (
    "nasa_battery_capacity",
    "nasa_cmapss_fd001_sensor_response",
    "starry_te_seebeck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _flatten(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    job = payload["job"]
    prediction = payload.get("prediction", {})
    spatial = payload.get("spatial", {})
    initial_prediction = payload.get("initial_prediction") or {}
    initial_spatial = payload.get("initial_spatial") or {}
    return {
        "dataset": job["dataset"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "q_dim": int(job.get("q_dim", 0)),
        "macro_nrmse": prediction.get("macro_nrmse"),
        "reference_nrmse": prediction.get("reference_nrmse"),
        "macro_rmse": prediction.get("macro_rmse"),
        "macro_mae": prediction.get("macro_mae"),
        "response_continuity_auc": spatial.get("response_continuity_auc"),
        "response_trustworthiness_auc": spatial.get("response_trustworthiness_auc"),
        "response_knn_overlap_auc": spatial.get("response_knn_overlap_auc"),
        "response_distance_stress": spatial.get("response_distance_stress"),
        "response_local_collapse_rate": spatial.get("response_local_collapse_rate"),
        "response_local_tear_rate": spatial.get("response_local_tear_rate"),
        "effective_rank": spatial.get("effective_rank"),
        "initial_macro_nrmse": initial_prediction.get("macro_nrmse"),
        "initial_reference_nrmse": initial_prediction.get("reference_nrmse"),
        "initial_response_continuity_auc": initial_spatial.get("response_continuity_auc"),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        "result_path": str(path),
    }


def _load_results(root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted([*(root / "new_methods").glob("**/result.json"), *(root / "anchors").glob("**/result.json")]):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                errors.append(f"non-success result: {path}")
                continue
            rows.append(_flatten(path, payload))
        except Exception as exc:  # keep reconciliation failures visible
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, errors
    duplicate = frame.duplicated(["dataset", "method", "seed"], keep=False)
    if duplicate.any():
        for row in frame.loc[duplicate, ["dataset", "method", "seed", "result_path"]].to_dict("records"):
            errors.append(f"duplicate cell: {row}")
    return frame.sort_values(["dataset", "method", "seed"]).reset_index(drop=True), errors


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "macro_nrmse",
        "reference_nrmse",
        "macro_rmse",
        "macro_mae",
        "response_continuity_auc",
        "response_trustworthiness_auc",
        "response_knn_overlap_auc",
        "response_distance_stress",
        "response_local_collapse_rate",
        "response_local_tear_rate",
        "effective_rank",
        "wall_time_seconds",
    ]
    grouped = frame.groupby(["dataset", "method"], sort=False)
    pieces = [grouped.size().rename("n").reset_index()]
    for metric in metrics:
        statistics = grouped[metric].agg(["median", "mean", "std", "min", "max"]).reset_index()
        statistics = statistics.rename(
            columns={
                name: f"{metric}_{name}"
                for name in ("median", "mean", "std", "min", "max")
            }
        )
        pieces.append(statistics)
    output = pieces[0]
    for piece in pieces[1:]:
        output = output.merge(piece, on=["dataset", "method"], how="left")
    return output


def _paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("macro_nrmse", "reference_nrmse", "response_continuity_auc")
    anchor = frame[frame["method"] == "joint_continuity"].set_index(["dataset", "seed"])
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == "joint_continuity":
            continue
        candidate = frame[frame["method"] == method].set_index(["dataset", "seed"])
        common = candidate.index.intersection(anchor.index)
        for dataset in DATASET_ORDER:
            keys = [key for key in common if key[0] == dataset]
            row: dict[str, Any] = {"dataset": dataset, "method": method, "paired_n": len(keys)}
            for metric in metrics:
                if not keys or metric not in candidate or metric not in anchor:
                    row[f"delta_{metric}_median"] = np.nan
                    row[f"wins_{metric}"] = 0
                    continue
                candidate_values = candidate.loc[keys, metric].to_numpy(float)
                anchor_values = anchor.loc[keys, metric].to_numpy(float)
                finite = np.isfinite(candidate_values) & np.isfinite(anchor_values)
                delta = candidate_values[finite] - anchor_values[finite]
                row[f"delta_{metric}_median"] = float(np.median(delta)) if len(delta) else np.nan
                lower_is_better = metric != "response_continuity_auc"
                row[f"wins_{metric}"] = int(np.sum(delta < 0 if lower_is_better else delta > 0))
            rows.append(row)
    return pd.DataFrame(rows)


def _format(value: Any, digits: int = 3) -> str:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(resolved) else f"{resolved:.{digits}f}"


def _report(frame: pd.DataFrame, summary: pd.DataFrame, effects: pd.DataFrame, errors: list[str], expected: int) -> str:
    lines = [
        "# Support-conditioned latent-variable pilot results",
        "",
        "> Exploratory development evidence only; do not promote to confirmatory claims without a frozen replication.",
        "",
        "## Completion and integrity",
        "",
        f"- Reconciled successful cells: {len(frame)}/{expected}.",
        f"- Raw-result reconciliation issues: {len(errors)}.",
        "- All values below are computed directly from raw `result.json` files.",
    ]
    if errors:
        lines.extend(["", "Issues:", "", *[f"- {item}" for item in errors]])
    lines.extend(["", "## Per-dataset results (median across seeds)", ""])
    for dataset in DATASET_ORDER:
        lines.extend([
            f"### {dataset}",
            "",
            "| Method | n | Macro NRMSE ↓ | Ref. NRMSE median [min, max] ↓ | Continuity ↑ | Collapse ↓ | Tear ↓ | Eff. rank | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        subset = summary[summary["dataset"] == dataset].set_index("method")
        for method in METHOD_ORDER:
            if method not in subset.index:
                lines.append(f"| {method} | 0 | — | — | — | — | — | — | — |")
                continue
            row = subset.loc[method]
            lines.append(
                f"| {method} | {int(row['n'])} | {_format(row['macro_nrmse_median'])} | "
                f"{_format(row['reference_nrmse_median'])} [{_format(row['reference_nrmse_min'])}, {_format(row['reference_nrmse_max'])}] | "
                f"{_format(row['response_continuity_auc_median'])} | "
                f"{_format(row['response_local_collapse_rate_median'])} | {_format(row['response_local_tear_rate_median'])} | "
                f"{_format(row['effective_rank_median'])} | {_format(row['wall_time_seconds_median'], 1)} |"
            )
        lines.append("")
    lines.extend([
        "## Paired differences from `joint_continuity`",
        "",
        "Negative prediction deltas are better; positive continuity deltas are better. Wins are counted over paired seeds.",
        "",
        "| Dataset | Method | n | Δ macro NRMSE | wins | Δ ref. NRMSE | wins | Δ continuity | wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in effects.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['method']} | {int(row['paired_n'])} | "
            f"{_format(row['delta_macro_nrmse_median'])} | {int(row['wins_macro_nrmse'])} | "
            f"{_format(row['delta_reference_nrmse_median'])} | {int(row['wins_reference_nrmse'])} | "
            f"{_format(row['delta_response_continuity_auc_median'])} | {int(row['wins_response_continuity_auc'])} |"
        )
    encoder = frame[frame["method"] == "encoder_q_refine"]
    lines.extend([
        "",
        "## Encoder initialization to support-only refinement",
        "",
        "This isolates what the 50 constrained q-refinement steps did to the encoder's own initialization. Prediction wins count seeds with lower refined reference NRMSE; continuity wins count higher refined AUC.",
        "",
        "| Dataset | n | Initial ref. NRMSE | Refined ref. NRMSE | prediction wins | Initial continuity | Refined continuity | continuity wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASET_ORDER:
        subset = encoder[encoder["dataset"] == dataset]
        prediction_finite = subset[["initial_reference_nrmse", "reference_nrmse"]].dropna()
        continuity_finite = subset[["initial_response_continuity_auc", "response_continuity_auc"]].dropna()
        lines.append(
            f"| {dataset} | {len(subset)} | {_format(subset['initial_reference_nrmse'].median())} | "
            f"{_format(subset['reference_nrmse'].median())} | "
            f"{int((prediction_finite['reference_nrmse'] < prediction_finite['initial_reference_nrmse']).sum())}/{len(prediction_finite)} | "
            f"{_format(subset['initial_response_continuity_auc'].median())} | "
            f"{_format(subset['response_continuity_auc'].median())} | "
            f"{int((continuity_finite['response_continuity_auc'] > continuity_finite['initial_response_continuity_auc']).sum())}/{len(continuity_finite)} |"
        )
    lines.extend([
        "",
        "## Exploratory readout",
        "",
        "- The learned q initializer is not a drop-in replacement for multistart calibration: it does not deliver seed-consistent prediction gains over `joint_continuity`. Its clearest benefit is smoother, less collapsed representation geometry.",
        "- Refinement strongly improves the encoder's own battery initialization, is nearly inert on C-MAPSS, and cannot rescue the shared Starry seed-2 failure. Lower support loss therefore does not guarantee held-out-query generalization.",
        "- The global-mean DeepSets/CNP baseline is too weak, especially on Starry. A query-to-support attentive conditional model is the next justified learned support baseline because it can express the local behavior that makes support kNN strong.",
        "- Starry macro NRMSE is dominated by near-constant label scales. Reference-scaled NRMSE and the displayed seed ranges make the failure tail easier to interpret, but the extreme seed must remain visible.",
        "- The next q experiment should add the encoder output as one candidate inside the existing support-internal multistart selector, preserving random/prior fallbacks, rather than replacing them.",
    ])
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Treat any pattern as a candidate mechanism, not a paper claim. A method should be advanced only when its prediction gain is seed-consistent and is not purchased by representation collapse, tearing, or unstable dataset-specific failure.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frame, errors = _load_results(args.campaign_root)
    expected = len(DATASET_ORDER) * len(METHOD_ORDER) * 3
    if frame.empty:
        raise SystemExit("No successful result.json files found")
    summary = _summary(frame)
    effects = _paired_effects(frame)
    frame.to_csv(args.campaign_root / "all_results.csv", index=False)
    summary.to_csv(args.campaign_root / "summary_by_dataset_method.csv", index=False)
    effects.to_csv(args.campaign_root / "paired_effects_vs_joint_continuity.csv", index=False)
    report = _report(frame, summary, effects, errors, expected)
    report_path = args.campaign_root / "SUPPORT_ENCODER_PILOT_RESULTS.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    print(report_path)


if __name__ == "__main__":
    main()
