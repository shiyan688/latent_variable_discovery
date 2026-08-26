#!/usr/bin/env python3
"""Reconcile the support-relative robustness pilot and apply its frozen gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_support_robustness_20260811"
PRIOR_CSV = (
    PROJECT_ROOT / "runs" / "iclr_support_followup_20260811" / "combined_results.csv"
)
DATASETS = (
    "nasa_battery_capacity",
    "nasa_cmapss_fd001_sensor_response",
    "starry_te_seebeck",
)
NEW_METHODS = (
    "attentive_supportnorm_mse",
    "attentive_supportnorm_huber",
    "attentive_supportnorm_huber_bound",
)
ANCHORS = (
    "attentive_cnp",
    "deepsets_direct",
    "joint_continuity",
    "support_knn",
    "random_forest",
)
METHODS = (*NEW_METHODS, *ANCHORS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _prediction_audit(payload: dict[str, Any]) -> dict[str, Any]:
    prediction_path = Path(payload["artifacts"]["query_predictions"])
    frame = pd.read_csv(prediction_path)
    finite = bool(
        np.isfinite(frame["prediction"].to_numpy(float)).all()
        and np.isfinite(frame["target"].to_numpy(float)).all()
    )
    bound_violations = 0
    method = payload["job"]["method"]
    if method == "attentive_supportnorm_huber_bound":
        diagnostics = payload.get("support_diagnostics", {})
        for label, group in frame.groupby("label", sort=False):
            row = diagnostics[str(label)]
            lower = float(row["prediction_physical_lower_bound"])
            upper = float(row["prediction_physical_upper_bound"])
            values = group["prediction"].to_numpy(float)
            tolerance = 1e-6 * max(1.0, abs(lower), abs(upper))
            bound_violations += int(
                np.sum((values < lower - tolerance) | (values > upper + tolerance))
            )
    return {
        "predictions_finite": finite,
        "prediction_abs_max": float(np.max(np.abs(frame["prediction"].to_numpy(float)))),
        "bound_violations": bound_violations,
    }


def _new_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    job = payload["job"]
    prediction = payload["prediction"]
    spatial = payload.get("spatial", {})
    training = payload.get("training", {})
    audit = _prediction_audit(payload)
    reference_nrmse = float(prediction["reference_nrmse"])
    return {
        "dataset": job["dataset"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "q_dim": int(job.get("q_dim", 0)),
        "macro_nrmse": prediction.get("macro_nrmse"),
        "reference_nrmse": reference_nrmse,
        "macro_rmse": prediction.get("macro_rmse"),
        "macro_mae": prediction.get("macro_mae"),
        "response_continuity_auc": spatial.get("response_continuity_auc"),
        "response_trustworthiness_auc": spatial.get("response_trustworthiness_auc"),
        "response_local_collapse_rate": spatial.get("response_local_collapse_rate"),
        "response_local_tear_rate": spatial.get("response_local_tear_rate"),
        "effective_rank": spatial.get("effective_rank"),
        "wall_time_seconds": payload.get("wall_time_seconds"),
        "target_scale_floor": training.get("target_scale_floor"),
        "prediction_standardized_abs_max": training.get(
            "prediction_standardized_abs_max"
        ),
        "catastrophic": int(reference_nrmse > 1.0),
        "result_path": str(path),
        **audit,
    }


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((args.campaign_root / "new_methods").glob("**/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                errors.append(f"non-success result: {path}")
            else:
                rows.append(_new_row(path, payload))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    new = pd.DataFrame(rows)
    if not new.empty and new.duplicated(["dataset", "method", "seed"]).any():
        errors.append("duplicate new dataset/method/seed cell")
    prior = pd.read_csv(PRIOR_CSV)
    for column in (
        "response_trustworthiness_auc",
        "target_scale_floor",
        "prediction_standardized_abs_max",
        "catastrophic",
        "predictions_finite",
        "prediction_abs_max",
        "bound_violations",
    ):
        if column not in prior:
            prior[column] = np.nan
    prior["catastrophic"] = (prior["reference_nrmse"] > 1.0).astype(int)
    columns = sorted(set(new.columns) | set(prior.columns))
    combined = pd.concat(
        [new.reindex(columns=columns), prior.reindex(columns=columns)],
        ignore_index=True,
    )
    combined = combined[combined["method"].isin(METHODS)]
    return combined.sort_values(["dataset", "method", "seed"]), errors


def paired(frame: pd.DataFrame, candidate: str, anchor: str) -> pd.DataFrame:
    left = frame[frame.method == candidate].set_index(["dataset", "seed"])
    right = frame[frame.method == anchor].set_index(["dataset", "seed"])
    rows = []
    for dataset in DATASETS:
        keys = [
            (dataset, seed)
            for seed in (0, 1, 2)
            if (dataset, seed) in left.index and (dataset, seed) in right.index
        ]
        ref_delta = np.asarray(
            [
                left.loc[key, "reference_nrmse"]
                - right.loc[key, "reference_nrmse"]
                for key in keys
            ],
            dtype=float,
        )
        continuity_delta = np.asarray(
            [
                left.loc[key, "response_continuity_auc"]
                - right.loc[key, "response_continuity_auc"]
                for key in keys
            ],
            dtype=float,
        )
        rows.append(
            {
                "dataset": dataset,
                "candidate": candidate,
                "anchor": anchor,
                "n": len(keys),
                "reference_delta_median": float(np.nanmedian(ref_delta)),
                "reference_wins": int(np.sum(ref_delta < 0)),
                "continuity_delta_median": (
                    float(np.nanmedian(continuity_delta))
                    if np.isfinite(continuity_delta).any()
                    else np.nan
                ),
                "continuity_wins": int(np.sum(continuity_delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def _gate_decisions(
    frame: pd.DataFrame, errors: list[str]
) -> dict[str, Any]:
    new = frame[frame.method.isin(NEW_METHODS)].copy()
    primary_finite = bool(
        len(new) == 27
        and np.isfinite(new["reference_nrmse"].to_numpy(float)).all()
        and np.isfinite(new["macro_nrmse"].to_numpy(float)).all()
        and new["predictions_finite"].fillna(False).astype(bool).all()
    )
    validity = len(errors) == 0 and primary_finite
    bounded_starry = new[
        (new.method == "attentive_supportnorm_huber_bound")
        & (new.dataset == "starry_te_seebeck")
    ]
    starry_safety = bool(
        len(bounded_starry) == 3
        and int(bounded_starry["catastrophic"].sum()) == 0
        and int(bounded_starry["bound_violations"].sum()) == 0
    )
    advancement: dict[str, Any] = {}
    anchor = frame[frame.method == "attentive_cnp"]
    for method in NEW_METHODS:
        candidate = new[new.method == method]
        wins: dict[str, int] = {}
        median_ratio: dict[str, float] = {}
        for dataset in DATASETS:
            cand = candidate[candidate.dataset == dataset].set_index("seed")
            base = anchor[anchor.dataset == dataset].set_index("seed")
            paired_seeds = sorted(set(cand.index) & set(base.index))
            wins[dataset] = int(
                sum(
                    cand.loc[seed, "reference_nrmse"]
                    < base.loc[seed, "reference_nrmse"]
                    for seed in paired_seeds
                )
            )
            median_ratio[dataset] = float(
                cand["reference_nrmse"].median()
                / base["reference_nrmse"].median()
            )
        improved = [dataset for dataset in DATASETS if wins[dataset] >= 2]
        remaining = [dataset for dataset in DATASETS if dataset not in improved]
        no_catastrophe = bool(
            len(candidate) == 9 and int(candidate["catastrophic"].sum()) == 0
        )
        remaining_safe = all(median_ratio[dataset] <= 1.1 for dataset in remaining)
        advancement[method] = {
            "pass": bool(
                validity
                and len(improved) >= 2
                and no_catastrophe
                and remaining_safe
            ),
            "paired_wins": wins,
            "median_ratio_to_attentive_cnp": median_ratio,
            "datasets_with_at_least_two_wins": improved,
            "no_catastrophic_failure": no_catastrophe,
            "remaining_dataset_within_10_percent": remaining_safe,
        }
    return {
        "validity_gate": validity,
        "starry_safety_gate_bounded": starry_safety,
        "reconciliation_issues": errors,
        "advancement": advancement,
    }


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.4g}"


def report(
    frame: pd.DataFrame,
    effects: pd.DataFrame,
    gates: dict[str, Any],
) -> str:
    new = frame[frame.method.isin(NEW_METHODS)]
    lines = [
        "# Support-relative robustness pilot",
        "",
        "> Frozen exploratory development evidence; not a confirmatory claim.",
        "",
        "## Completion and integrity",
        "",
        f"- New successful cells reconciled: {len(new)}/27.",
        f"- Validity gate: **{'PASS' if gates['validity_gate'] else 'FAIL'}**.",
        f"- Bounded-variant Starry safety gate: **{'PASS' if gates['starry_safety_gate_bounded'] else 'FAIL'}**.",
        f"- Reconciliation issues: {len(gates['reconciliation_issues'])}.",
        "- Prior controls were reused from the matched earlier pilots and not rerun.",
        "",
        "## Per-dataset results",
        "",
    ]
    for dataset in DATASETS:
        lines += [
            f"### {dataset}",
            "",
            "| Method | n | Reference NRMSE median [min, max] ↓ | Cat. >1 | Continuity ↑ | Collapse ↓ | Std. pred. | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            part = frame[(frame.dataset == dataset) & (frame.method == method)]
            if part.empty:
                continue
            lines.append(
                f"| {method} | {len(part)} | {fmt(part.reference_nrmse.median())} "
                f"[{fmt(part.reference_nrmse.min())}, {fmt(part.reference_nrmse.max())}] | "
                f"{int(part.catastrophic.fillna(0).sum())} | "
                f"{fmt(part.response_continuity_auc.median())} | "
                f"{fmt(part.response_local_collapse_rate.median())} | "
                f"{fmt(part.prediction_standardized_abs_max.median())} | "
                f"{fmt(part.wall_time_seconds.median())} |"
            )
        lines.append("")
    lines += [
        "## Sequential paired effects",
        "",
        "Negative NRMSE deltas favor the candidate; positive continuity deltas favor it.",
        "",
        "| Dataset | Candidate | Anchor | Δ ref. NRMSE | wins | Δ continuity | wins |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in effects.iterrows():
        lines.append(
            f"| {row.dataset} | {row.candidate} | {row.anchor} | "
            f"{fmt(row.reference_delta_median)} | {int(row.reference_wins)}/{int(row.n)} | "
            f"{fmt(row.continuity_delta_median)} | {int(row.continuity_wins)}/{int(row.n)} |"
        )
    lines += [
        "",
        "## Starry seed-level safety",
        "",
        "| Method | Seed | Reference NRMSE | Max |prediction| | Bound violations |",
        "|---|---:|---:|---:|---:|",
    ]
    starry = new[new.dataset == "starry_te_seebeck"].sort_values(["method", "seed"])
    for _, row in starry.iterrows():
        lines.append(
            f"| {row.method} | {int(row.seed)} | {fmt(row.reference_nrmse)} | "
            f"{fmt(row.prediction_abs_max)} | {int(row.bound_violations)} |"
        )
    lines += [
        "",
        "## Frozen gate decisions",
        "",
        "| Variant | General advancement | Datasets with ≥2/3 wins | No catastrophic failure | Remaining dataset ≤10% regression |",
        "|---|---:|---|---:|---:|",
    ]
    for method, row in gates["advancement"].items():
        improved = ", ".join(row["datasets_with_at_least_two_wins"]) or "none"
        lines.append(
            f"| {method} | {'PASS' if row['pass'] else 'FAIL'} | {improved} | "
            f"{'yes' if row['no_catastrophic_failure'] else 'no'} | "
            f"{'yes' if row['remaining_dataset_within_10_percent'] else 'no'} |"
        )
    lines += [
        "",
        "A Starry safety pass alone supports an anti-extrapolation ablation, not a generally improved method. A general gate pass would only select a frozen candidate for separate confirmation.",
        "",
    ]
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    frame, errors = load(args)
    effects = pd.concat(
        [
            paired(frame, "attentive_supportnorm_mse", "attentive_cnp"),
            paired(
                frame,
                "attentive_supportnorm_huber",
                "attentive_supportnorm_mse",
            ),
            paired(
                frame,
                "attentive_supportnorm_huber_bound",
                "attentive_supportnorm_huber",
            ),
            paired(
                frame,
                "attentive_supportnorm_huber_bound",
                "support_knn",
            ),
        ],
        ignore_index=True,
    )
    gates = _gate_decisions(frame, errors)
    frame.to_csv(args.campaign_root / "combined_results.csv", index=False)
    effects.to_csv(args.campaign_root / "paired_effects.csv", index=False)
    _write_json_atomic(args.campaign_root / "gate_decisions.json", gates)
    text = report(frame, effects, gates)
    path = args.campaign_root / "SUPPORT_ROBUSTNESS_RESULTS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
