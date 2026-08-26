#!/usr/bin/env python3
"""Consolidate the time-budgeted campaign into detailed, paired result tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_ROOTS = (
    PROJECT_ROOT / "runs" / "loss_component_ablation_synthetic_20260809",
    PROJECT_ROOT / "runs" / "loss_component_ablation_real_20260809",
    PROJECT_ROOT / "runs" / "pdebench_burgers_latent_20260809",
)
LOWER_IS_BETTER = {
    "reference_nrmse": True,
    "continuity_auc": False,
    "local_log_distortion_p95": True,
    "effective_rank": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-running", action="store_true")
    return parser.parse_args()


def _planned_roots(campaign_root: Path) -> list[Path]:
    roots = set(CURRENT_ROOTS)
    plan_path = campaign_root / "planned_tasks.jsonl"
    for line in plan_path.read_text(encoding="utf-8").splitlines():
        try:
            roots.add(Path(json.loads(line)["output_root"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(roots)


def _family(path: Path) -> str:
    value = str(path)
    if "pdebench_burgers_latent_20260809" in value:
        return "pde_core"
    if "extended_pdebench_support" in value:
        return "pde_support"
    if "extended_pdebench_seeds" in value:
        return "pde_core"
    if "loss_component_ablation" in value or "extended_loss_seed" in value:
        return "component_core"
    if "extended_loss_qdim" in value:
        return "qdim"
    if "extended_loss_dose" in value:
        return "loss_dose"
    if "extended_loss_support" in value:
        return "support"
    return "other"


def _config_fields(payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("latent_config") or {}
    return {
        "prediction_loss_type": config.get("prediction_loss_type"),
        "orthogonality_type": config.get("latent_feature_orthogonality_type"),
        "orthogonality_weight": config.get("latent_feature_orthogonality_weight"),
        "continuity_weight": config.get("latent_curve_continuity_weight"),
        "q_l2_weight": config.get("latent_q_l2_weight"),
        "calibration_prior_weight": config.get("calibration_q_prior_weight"),
        "loss_weighting": config.get("loss_weighting"),
    }


def _loss_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    job = payload["job"]
    real = "dataset" in job
    spatial = payload.get("spatial", {})
    prefix = "response_" if real else ""
    prediction = payload.get("prediction", {})
    return {
        "family": _family(path),
        "domain": "real" if real else "synthetic",
        "dataset": job["dataset"] if real else f"expr{int(job['expression_id']):03d}",
        "method": job["method"],
        "strategy": "latent" if (payload.get("latent_config") is not None) else "baseline",
        "loss_preset": job.get("loss_preset"),
        "seed": job["seed"],
        "q_dim": job.get("q_dim"),
        "support_ratio": job.get("support_ratio"),
        "reference_nrmse": prediction.get("reference_nrmse"),
        "macro_nrmse": prediction.get("macro_nrmse"),
        "macro_rmse": prediction.get("macro_rmse"),
        "macro_r2": prediction.get("macro_r2"),
        "continuity_auc": spatial.get(f"{prefix}continuity_auc"),
        "trustworthiness_auc": spatial.get(f"{prefix}trustworthiness_auc"),
        "knn_overlap_auc": spatial.get(f"{prefix}knn_overlap_auc"),
        "distance_spearman": spatial.get(f"{prefix}distance_spearman"),
        "local_log_distortion_p95": spatial.get(f"{prefix}local_log_distortion_p95"),
        "local_collapse_rate": spatial.get(f"{prefix}local_collapse_rate"),
        "local_tear_rate": spatial.get(f"{prefix}local_tear_rate"),
        "effective_rank": spatial.get("effective_rank"),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        **_config_fields(payload),
        "result_path": str(path),
    }


def _pde_rows(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    job = payload["job"]
    rows = []
    for strategy, values in payload["strategies"].items():
        prediction = values.get("prediction", {})
        spatial = values.get("spatial", {})
        rows.append(
            {
                "family": _family(path),
                "domain": "pde",
                "dataset": "PDEBench_Burgers_Nu0.02",
                "method": job["method"],
                "strategy": strategy,
                "loss_preset": "mse",
                "seed": job["seed"],
                "q_dim": job["q_dim"],
                "support_ratio": job["support_ratio"],
                "reference_nrmse": prediction.get("reference_nrmse"),
                "macro_nrmse": prediction.get("macro_nrmse"),
                "macro_rmse": prediction.get("macro_rmse"),
                "macro_r2": prediction.get("macro_r2"),
                "label_reference_nrmse_p95": prediction.get("label_reference_nrmse_p95"),
                "continuity_auc": spatial.get("continuity_auc"),
                "trustworthiness_auc": spatial.get("trustworthiness_auc"),
                "knn_overlap_auc": spatial.get("knn_overlap_auc"),
                "distance_spearman": spatial.get("distance_spearman"),
                "local_log_distortion_p95": spatial.get("local_log_distortion_p95"),
                "local_collapse_rate": spatial.get("local_collapse_rate"),
                "local_tear_rate": spatial.get("local_tear_rate"),
                "effective_rank": spatial.get("effective_rank"),
                "calibration_seconds": values.get("calibration_seconds"),
                "result_path": str(path),
            }
        )
    return rows


def load_rows(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("**/result.json"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("status") != "success":
                continue
            job = payload.get("job", {})
            if job.get("problem", "").startswith("pdebench"):
                rows.extend(_pde_rows(path, payload))
            elif "dataset" in job or "expression_id" in job:
                rows.append(_loss_row(path, payload))
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in (
            "reference_nrmse",
            "macro_nrmse",
            "continuity_auc",
            "trustworthiness_auc",
            "knn_overlap_auc",
            "distance_spearman",
            "local_log_distortion_p95",
            "local_collapse_rate",
            "local_tear_rate",
            "effective_rank",
            "wall_time_seconds",
            "calibration_seconds",
        )
        if column in frame
    ]
    group = [
        "family",
        "domain",
        "dataset",
        "method",
        "strategy",
        "q_dim",
        "support_ratio",
    ]
    output = frame.groupby(group, dropna=False, as_index=False)[metrics].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    output.columns = [
        "_".join(str(value) for value in column if value != "")
        if isinstance(column, tuple)
        else str(column)
        for column in output.columns
    ]
    return output


def _wilcoxon(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox").pvalue)


def _bh(series: pd.Series) -> pd.Series:
    values = series.to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    for index in range(len(values) - 1, -1, -1):
        source = order[index]
        running = min(running, values[source] * len(values) / (index + 1))
        adjusted[source] = running
    return pd.Series(adjusted, index=series.index)


def paired_effects(frame: pd.DataFrame) -> pd.DataFrame:
    loss = frame[frame["domain"].isin(["synthetic", "real"])].copy()
    baseline = loss[loss["method"] == "joint_mse"].drop_duplicates(
        ["domain", "dataset", "seed", "q_dim", "support_ratio"], keep="first"
    )
    rows: list[dict[str, Any]] = []
    for family in sorted(loss["family"].unique()):
        candidates = loss[(loss["family"] == family) & (loss["method"] != "joint_mse")]
        for (domain, method), selected in candidates.groupby(["domain", "method"]):
            for metric, lower_is_better in LOWER_IS_BETTER.items():
                left = selected.dropna(subset=[metric])
                right = baseline.dropna(subset=[metric])
                merged = left.merge(
                    right[
                        ["domain", "dataset", "seed", "q_dim", "support_ratio", metric]
                    ],
                    on=["domain", "dataset", "seed", "q_dim", "support_ratio"],
                    suffixes=("_candidate", "_baseline"),
                )
                if merged.empty:
                    continue
                delta = (
                    merged[f"{metric}_candidate"].to_numpy(float)
                    - merged[f"{metric}_baseline"].to_numpy(float)
                )
                oriented = -delta if lower_is_better else delta
                rows.append(
                    {
                        "family": family,
                        "domain": domain,
                        "method": method,
                        "metric": metric,
                        "pairs": len(delta),
                        "median_delta_candidate_minus_joint_mse": np.median(delta),
                        "mean_delta_candidate_minus_joint_mse": np.mean(delta),
                        "win_rate": np.mean(oriented > 0),
                        "wilcoxon_p": _wilcoxon(delta),
                    }
                )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["bh_q"] = output.groupby(["family", "domain", "metric"])[
            "wilcoxon_p"
        ].transform(_bh)
    return output


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    lines = [header, separator]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    status = json.loads((args.campaign_root / "campaign_status.json").read_text())
    if not args.allow_running and status["state"] not in {
        "completed_budget",
        "completed_all",
        "completed_with_failures",
    }:
        raise RuntimeError(f"Campaign is still {status['state']}; use --allow-running for diagnostics.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = load_rows(_planned_roots(args.campaign_root))
    summary = _summary(frame)
    effects = paired_effects(frame)
    frame.to_csv(args.output_root / "extended_all_result_rows.csv", index=False)
    summary.to_csv(args.output_root / "extended_group_summary.csv", index=False)
    effects.to_csv(args.output_root / "extended_paired_effects.csv", index=False)

    task_status_path = args.campaign_root / "task_status.jsonl"
    task_rows = []
    if task_status_path.exists():
        task_rows = [json.loads(line) for line in task_status_path.read_text().splitlines() if line]
    task_frame = pd.DataFrame(task_rows)
    if not task_frame.empty:
        task_frame.to_csv(args.output_root / "extended_task_ledger.csv", index=False)
        progress = task_frame.groupby("family", as_index=False).agg(
            completed=("returncode", lambda values: int(np.sum(np.asarray(values) == 0))),
            failed=("returncode", lambda values: int(np.sum(np.asarray(values) != 0))),
            median_seconds=("elapsed_seconds", "median"),
            total_gpu_hours=("elapsed_seconds", lambda values: float(np.sum(values) / 3600)),
        )
    else:
        progress = pd.DataFrame(
            columns=["family", "completed", "failed", "median_seconds", "total_gpu_hours"]
        )
    progress.to_csv(args.output_root / "extended_family_progress.csv", index=False)

    compact = summary[[
        "family",
        "domain",
        "dataset",
        "method",
        "strategy",
        "q_dim",
        "support_ratio",
        "reference_nrmse_count",
        "reference_nrmse_median",
        "continuity_auc_median",
        "local_log_distortion_p95_median",
        "effective_rank_median",
    ]]
    lines = [
        "# Extended 15-hour campaign report",
        "",
        f"Campaign state: `{status['state']}`. Consolidated result rows: {len(frame)}.",
        "",
        "## Execution coverage",
        "",
        _markdown_table(progress),
        "",
        "## Every-dataset result table",
        "",
        _markdown_table(compact),
        "",
        "## Machine-readable artifacts",
        "",
        "- `extended_all_result_rows.csv`: one row per loss run or PDE strategy.",
        "- `extended_group_summary.csv`: count/mean/median/std/min/max by exact protocol cell.",
        "- `extended_paired_effects.csv`: paired deltas, win rates, Wilcoxon p values, and BH q values.",
        "- `extended_task_ledger.csv`: GPU, runtime, return code, and timeout status for every dispatched task.",
        "",
    ]
    report_path = args.output_root / "EXTENDED_15H_RESULTS.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    print(report_path)


if __name__ == "__main__":
    main()
