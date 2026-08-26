#!/usr/bin/env python3
"""Reconcile and gate the frozen attentive reliability-selector pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "iclr_attentive_selector_20260811"
PRIOR_CSV = (
    PROJECT_ROOT / "runs" / "iclr_support_robustness_20260811" / "combined_results.csv"
)
DATASETS = (
    "nasa_battery_capacity",
    "nasa_cmapss_fd001_sensor_response",
    "starry_te_seebeck",
)
SELECTOR = "attentive_reliability_selector"
GLOBAL = "attentive_cnp"
BOUNDED = "attentive_supportnorm_huber_bound"
METHODS = (SELECTOR, GLOBAL, BOUNDED, "joint_continuity", "support_knn", "random_forest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _new_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    prediction_path = Path(payload["artifacts"]["query_predictions"])
    predictions = pd.read_csv(prediction_path)
    diagnostics = list(payload["support_diagnostics"].values())
    scores = np.asarray(
        [
            value
            for row in diagnostics
            for value in (row["global_selector_mae"], row["bounded_selector_mae"])
        ],
        dtype=float,
    )
    raw = predictions[["target", "prediction"]].to_numpy(float)
    job = payload["job"]
    prediction = payload["prediction"]
    spatial = payload.get("spatial", {})
    training = payload.get("training", {})
    reference_nrmse = float(prediction["reference_nrmse"])
    return {
        "dataset": job["dataset"],
        "method": job["method"],
        "seed": int(job["seed"]),
        "q_dim": int(job["q_dim"]),
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
        "selected_bounded_fraction": training.get("selected_bounded_fraction"),
        "prediction_abs_max": float(np.max(np.abs(predictions.prediction.to_numpy(float)))),
        "predictions_finite": bool(np.isfinite(raw).all()),
        "selector_scores_finite": bool(np.isfinite(scores).all()),
        "catastrophic": int(reference_nrmse > 1.0),
        "result_path": str(path),
    }


def load(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((args.campaign_root / "new_method").glob("**/result.json")):
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
        errors.append("duplicate selector dataset/method/seed cell")
    prior = pd.read_csv(PRIOR_CSV)
    for column in (
        "selected_bounded_fraction",
        "predictions_finite",
        "selector_scores_finite",
        "prediction_abs_max",
        "catastrophic",
    ):
        if column not in prior:
            prior[column] = np.nan
    prior["catastrophic"] = (prior.reference_nrmse > 1.0).astype(int)
    columns = sorted(set(new.columns) | set(prior.columns))
    combined = pd.concat(
        [new.reindex(columns=columns), prior.reindex(columns=columns)],
        ignore_index=True,
    )
    combined = combined[combined.method.isin(METHODS)]
    return combined.sort_values(["dataset", "method", "seed"]), errors


def paired(frame: pd.DataFrame, anchor: str) -> pd.DataFrame:
    candidate = frame[frame.method == SELECTOR].set_index(["dataset", "seed"])
    baseline = frame[frame.method == anchor].set_index(["dataset", "seed"])
    rows = []
    for dataset in DATASETS:
        keys = [
            (dataset, seed)
            for seed in (0, 1, 2)
            if (dataset, seed) in candidate.index and (dataset, seed) in baseline.index
        ]
        delta = np.asarray(
            [
                candidate.loc[key, "reference_nrmse"]
                - baseline.loc[key, "reference_nrmse"]
                for key in keys
            ],
            dtype=float,
        )
        rows.append(
            {
                "dataset": dataset,
                "candidate": SELECTOR,
                "anchor": anchor,
                "n": len(keys),
                "reference_delta_median": float(np.median(delta)),
                "reference_wins": int(np.sum(delta < 0)),
            }
        )
    return pd.DataFrame(rows)


def gate(frame: pd.DataFrame, errors: list[str]) -> dict[str, Any]:
    selector = frame[frame.method == SELECTOR]
    validity = bool(
        len(errors) == 0
        and len(selector) == 9
        and np.isfinite(selector.reference_nrmse.to_numpy(float)).all()
        and np.isfinite(selector.macro_nrmse.to_numpy(float)).all()
        and selector.predictions_finite.fillna(False).astype(bool).all()
        and selector.selector_scores_finite.fillna(False).astype(bool).all()
    )
    starry = selector[selector.dataset == "starry_te_seebeck"]
    safety = bool(
        len(starry) == 3
        and int(starry.catastrophic.sum()) == 0
        and bool((starry.reference_nrmse < 1.0).all())
    )
    comparisons: dict[str, dict[str, float | bool]] = {}
    all_within = True
    for dataset in DATASETS:
        selected = selector[selector.dataset == dataset].reference_nrmse.median()
        global_median = frame[
            (frame.dataset == dataset) & (frame.method == GLOBAL)
        ].reference_nrmse.median()
        bounded_median = frame[
            (frame.dataset == dataset) & (frame.method == BOUNDED)
        ].reference_nrmse.median()
        best_component = min(global_median, bounded_median)
        ratio = float(selected / best_component)
        within = bool(ratio <= 1.1)
        all_within = all_within and within
        comparisons[dataset] = {
            "selector_median": float(selected),
            "best_component_median": float(best_component),
            "ratio": ratio,
            "within_10_percent": within,
        }
    no_catastrophe = bool(
        len(selector) == 9 and int(selector.catastrophic.sum()) == 0
    )
    bounded_fraction = {
        dataset: float(
            selector[selector.dataset == dataset].selected_bounded_fraction.median()
        )
        for dataset in DATASETS
    }
    routing_pattern = bool(
        bounded_fraction["starry_te_seebeck"]
        > bounded_fraction["nasa_battery_capacity"]
        and bounded_fraction["starry_te_seebeck"]
        > bounded_fraction["nasa_cmapss_fd001_sensor_response"]
    )
    return {
        "validity_gate": validity,
        "safety_gate": safety,
        "general_advancement_gate": bool(
            validity and safety and no_catastrophe and all_within
        ),
        "no_catastrophic_failure": no_catastrophe,
        "dataset_comparisons": comparisons,
        "median_bounded_selection_fraction": bounded_fraction,
        "routing_pattern_supported": routing_pattern,
        "reconciliation_issues": errors,
    }


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.4g}"


def report(frame: pd.DataFrame, effects: pd.DataFrame, gates: dict[str, Any]) -> str:
    selector = frame[frame.method == SELECTOR]
    lines = [
        "# Attentive reliability selector pilot",
        "",
        "> Frozen exploratory development evidence; not a confirmatory claim.",
        "",
        "## Completion and gates",
        "",
        f"- Successful selector cells: {len(selector)}/9.",
        f"- Validity gate: **{'PASS' if gates['validity_gate'] else 'FAIL'}**.",
        f"- Starry safety gate: **{'PASS' if gates['safety_gate'] else 'FAIL'}**.",
        f"- General advancement gate: **{'PASS' if gates['general_advancement_gate'] else 'FAIL'}**.",
        f"- Reconciliation issues: {len(gates['reconciliation_issues'])}.",
        "",
        "## Per-dataset results",
        "",
    ]
    for dataset in DATASETS:
        lines += [
            f"### {dataset}",
            "",
            "| Method | n | Reference NRMSE median [min, max] ↓ | Cat. >1 | Continuity ↑ | Collapse ↓ | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
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
                f"{fmt(part.wall_time_seconds.median())} |"
            )
        lines.append("")
    lines += [
        "## Paired selector effects",
        "",
        "Negative deltas favor the selector.",
        "",
        "| Dataset | Anchor | Δ reference NRMSE | wins |",
        "|---|---|---:|---:|",
    ]
    for _, row in effects.iterrows():
        lines.append(
            f"| {row.dataset} | {row.anchor} | {fmt(row.reference_delta_median)} | "
            f"{int(row.reference_wins)}/{int(row.n)} |"
        )
    lines += [
        "",
        "## Selector mechanism and frozen comparison",
        "",
        "| Dataset | Bounded selection fraction | Selector / better-component median | Within 10% |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        comparison = gates["dataset_comparisons"][dataset]
        lines.append(
            f"| {dataset} | {fmt(gates['median_bounded_selection_fraction'][dataset])} | "
            f"{fmt(comparison['ratio'])} | {'yes' if comparison['within_10_percent'] else 'no'} |"
        )
    lines += [
        "",
        f"The predeclared cross-dataset routing pattern was **{'supported' if gates['routing_pattern_supported'] else 'not supported'}**.",
        "",
        "A gate pass selects the method for held-out confirmation; these nine development cells are not themselves confirmatory.",
        "",
    ]
    return "\n".join(lines)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    frame, errors = load(args)
    effects = pd.concat([paired(frame, GLOBAL), paired(frame, BOUNDED)], ignore_index=True)
    gates = gate(frame, errors)
    frame.to_csv(args.campaign_root / "combined_results.csv", index=False)
    effects.to_csv(args.campaign_root / "paired_effects.csv", index=False)
    write_json_atomic(args.campaign_root / "gate_decisions.json", gates)
    text = report(frame, effects, gates)
    path = args.campaign_root / "ATTENTIVE_SELECTOR_RESULTS.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
