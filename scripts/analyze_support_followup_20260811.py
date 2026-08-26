#!/usr/bin/env python3
"""Reconcile the 18 follow-up cells and compare them with prior matched controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_followup_20260811"
PRIOR_CSV = PROJECT_ROOT / "runs" / "iclr_support_encoder_pilot_20260811" / "all_results.csv"
DATASETS = ("nasa_battery_capacity", "nasa_cmapss_fd001_sensor_response", "starry_te_seebeck")
METHODS = ("encoder_q_multistart", "encoder_q_refine", "joint_continuity", "attentive_cnp", "deepsets_direct", "support_knn", "random_forest", "no_q_mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _new_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    job, prediction, spatial, training = payload["job"], payload["prediction"], payload.get("spatial", {}), payload.get("training", {})
    return {"dataset": job["dataset"], "method": job["method"], "seed": int(job["seed"]),
        "q_dim": int(job.get("q_dim", 0)), "macro_nrmse": prediction.get("macro_nrmse"),
        "reference_nrmse": prediction.get("reference_nrmse"), "macro_rmse": prediction.get("macro_rmse"),
        "macro_mae": prediction.get("macro_mae"), "response_continuity_auc": spatial.get("response_continuity_auc"),
        "response_local_collapse_rate": spatial.get("response_local_collapse_rate"),
        "response_local_tear_rate": spatial.get("response_local_tear_rate"), "effective_rank": spatial.get("effective_rank"),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        "selected_encoder_candidate_fraction": training.get("selected_encoder_candidate_fraction"),
        "attention_max_weight_mean": training.get("attention_max_weight_mean"), "result_path": str(path)}


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    rows, errors = [], []
    for path in sorted((args.campaign_root / "new_methods").glob("**/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "success": errors.append(f"non-success: {path}")
            else: rows.append(_new_row(path, payload))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    new = pd.DataFrame(rows)
    if not new.empty and new.duplicated(["dataset", "method", "seed"]).any():
        errors.append("duplicate follow-up dataset/method/seed cell")
    prior = pd.read_csv(PRIOR_CSV)
    for column in ("selected_encoder_candidate_fraction", "attention_max_weight_mean"):
        prior[column] = np.nan
    columns = sorted(set(new.columns) | set(prior.columns))
    combined = pd.concat([new.reindex(columns=columns), prior.reindex(columns=columns)], ignore_index=True)
    combined = combined[combined["method"].isin(METHODS)]
    return combined.sort_values(["dataset", "method", "seed"]), errors


def paired(frame: pd.DataFrame, candidate: str, anchor: str) -> pd.DataFrame:
    left = frame[frame.method == candidate].set_index(["dataset", "seed"])
    right = frame[frame.method == anchor].set_index(["dataset", "seed"])
    rows = []
    for dataset in DATASETS:
        keys = [(dataset, seed) for seed in (0, 1, 2) if (dataset, seed) in left.index and (dataset, seed) in right.index]
        ref_delta = np.array([left.loc[key, "reference_nrmse"] - right.loc[key, "reference_nrmse"] for key in keys], float)
        cont_delta = np.array([left.loc[key, "response_continuity_auc"] - right.loc[key, "response_continuity_auc"] for key in keys], float)
        rows.append({"dataset": dataset, "candidate": candidate, "anchor": anchor, "n": len(keys),
            "reference_delta_median": float(np.nanmedian(ref_delta)), "reference_wins": int(np.sum(ref_delta < 0)),
            "continuity_delta_median": float(np.nanmedian(cont_delta)) if np.isfinite(cont_delta).any() else np.nan,
            "continuity_wins": int(np.sum(cont_delta > 0))})
    return pd.DataFrame(rows)


def fmt(value: Any) -> str:
    try: value = float(value)
    except (TypeError, ValueError): return "—"
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def report(frame: pd.DataFrame, effects: pd.DataFrame, errors: list[str]) -> str:
    new = frame[frame.method.isin(("attentive_cnp", "encoder_q_multistart"))]
    lines = ["# Targeted support-conditioning follow-up", "", "> Exploratory development evidence only.", "",
        "## Completion and integrity", "", f"- New successful cells reconciled: {len(new)}/18.",
        f"- Reconciliation issues: {len(errors)}.", "- Prior controls are reused from the matched 54-cell pilot; they were not rerun.", "",
        "## Per-dataset results", ""]
    for dataset in DATASETS:
        lines += [f"### {dataset}", "", "| Method | n | Reference NRMSE median [min, max] ↓ | Continuity ↑ | Collapse ↓ | Tear ↓ | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            part = frame[(frame.dataset == dataset) & (frame.method == method)]
            if part.empty: continue
            lines.append(f"| {method} | {len(part)} | {fmt(part.reference_nrmse.median())} [{fmt(part.reference_nrmse.min())}, {fmt(part.reference_nrmse.max())}] | {fmt(part.response_continuity_auc.median())} | {fmt(part.response_local_collapse_rate.median())} | {fmt(part.response_local_tear_rate.median())} | {fmt(part.wall_time_seconds.median())} |")
        lines.append("")
    lines += ["## Paired effects", "", "Negative NRMSE deltas favor the candidate; positive continuity deltas favor the candidate.", "",
        "| Dataset | Candidate | Anchor | Δ ref. NRMSE | wins | Δ continuity | wins |", "|---|---|---|---:|---:|---:|---:|"]
    for _, row in effects.iterrows():
        lines.append(f"| {row.dataset} | {row.candidate} | {row.anchor} | {fmt(row.reference_delta_median)} | {int(row.reference_wins)}/{int(row.n)} | {fmt(row.continuity_delta_median)} | {int(row.continuity_wins)}/{int(row.n)} |")
    q = new[new.method == "encoder_q_multistart"]
    a = new[new.method == "attentive_cnp"]
    lines += ["", "## Mechanism diagnostics", "", "| Dataset | Encoder candidate selected | Attention max weight |", "|---|---:|---:|"]
    for dataset in DATASETS:
        lines.append(f"| {dataset} | {fmt(q[q.dataset == dataset].selected_encoder_candidate_fraction.median())} | {fmt(a[a.dataset == dataset].attention_max_weight_mean.median())} |")
    lines += ["", "## Gate reminder", "", "Passing this exploratory gate selects a candidate for a separately frozen confirmation; it is not itself a confirmatory claim.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frame, errors = load(args)
    effects = pd.concat([paired(frame, "encoder_q_multistart", "joint_continuity"),
        paired(frame, "attentive_cnp", "deepsets_direct"), paired(frame, "attentive_cnp", "support_knn")], ignore_index=True)
    frame.to_csv(args.campaign_root / "combined_results.csv", index=False)
    effects.to_csv(args.campaign_root / "paired_effects.csv", index=False)
    text = report(frame, effects, errors)
    path = args.campaign_root / "SUPPORT_FOLLOWUP_RESULTS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
