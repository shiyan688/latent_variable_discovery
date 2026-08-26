#!/usr/bin/env python3
"""Run the frozen support-envelope projected-q confirmation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

import run_iclr_real_discovery as real
import run_q_knn_reliability_selector_20260822 as base

DEFAULT_ROOT = PROJECT_ROOT / "runs" / "support_envelope_projected_q_confirm_20260824"
PLAN_PATH = PROJECT_ROOT / "SUPPORT_ENVELOPE_PROJECTED_Q_PLAN_20260824.md"
PROJECTION_MULTIPLIER = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-job")
    run.add_argument("--prepared-summary", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    base._experiment_args(run)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--gpus", default="0,1,6,7")
    launch.add_argument("--seeds", default=",".join(str(seed) for seed in range(40, 50)))
    launch.add_argument("--poll-seconds", type=float, default=15.0)
    launch.add_argument("--single-job-timeout-minutes", type=float, default=240.0)
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    launch.add_argument("--dry-run", action="store_true")
    base._experiment_args(launch)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def run_job(args: argparse.Namespace) -> Path:
    result_path = base.run_job(args)
    payload = json.loads(result_path.read_text())
    if payload.get("projection", {}).get("multiplier") == PROJECTION_MULTIPLIER:
        return result_path

    record = real._load_record(args.prepared_summary, args.dataset)
    train = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["train_csv"])),
        args.max_train_per_label,
        args.subsample_seed,
    )
    test = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["test_csv"])),
        args.max_test_per_label,
        args.subsample_seed + 10000,
    )
    test_y = test.target.to_numpy(np.float32)
    test_labels = test.label.to_numpy()
    support, query = real._support_query_indices(test_labels, args.support_ratio, args.seed)
    query_frame = pd.read_csv(payload["artifacts"]["query_predictions"])
    np.testing.assert_allclose(query_frame.target.to_numpy(), test_y[query])
    np.testing.assert_array_equal(query_frame.label.to_numpy(), test_labels[query])

    train_scale = max(float(np.std(train.target.to_numpy(np.float32))), 1e-8)
    raw_prediction = query_frame.q_prediction.to_numpy(float)
    projected_prediction = raw_prediction.copy()
    envelope_rows = []
    for label in pd.unique(test_labels):
        support_values = test_y[support][test_labels[support] == label]
        lower = float(support_values.min() - PROJECTION_MULTIPLIER * train_scale)
        upper = float(support_values.max() + PROJECTION_MULTIPLIER * train_scale)
        selected = test_labels[query] == label
        projected_prediction[selected] = np.clip(raw_prediction[selected], lower, upper)
        envelope_rows.append(
            {
                "label": label,
                "support_rows": int(len(support_values)),
                "support_min": float(support_values.min()),
                "support_max": float(support_values.max()),
                "train_target_std": train_scale,
                "lower_bound": lower,
                "upper_bound": upper,
                "query_rows": int(selected.sum()),
                "projection_active_rows": int(
                    np.sum(projected_prediction[selected] != raw_prediction[selected])
                ),
            }
        )
    bounds = pd.DataFrame(envelope_rows)
    violations = 0
    for row in envelope_rows:
        selected = test_labels[query] == row["label"]
        violations += int(
            np.sum(
                (projected_prediction[selected] < row["lower_bound"] - 1e-7)
                | (projected_prediction[selected] > row["upper_bound"] + 1e-7)
            )
        )
    assert violations == 0

    query_frame["projected_q_prediction"] = projected_prediction
    query_frame["projection_active"] = projected_prediction != raw_prediction
    query_frame.to_csv(payload["artifacts"]["query_predictions"], index=False)
    bounds_path = result_path.parent / "support_envelope_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    payload["metrics"]["projected_q"] = base._metrics(
        query_frame.target.to_numpy(float),
        projected_prediction,
        query_frame.label.to_numpy(),
        train.target.to_numpy(np.float32),
    )
    payload["projection"] = {
        "name": "support_envelope_projected_q",
        "multiplier": PROJECTION_MULTIPLIER,
        "formula": "[support_min - multiplier * train_target_std, support_max + multiplier * train_target_std]",
        "active_fraction": float(np.mean(projected_prediction != raw_prediction)),
        "bound_violations": violations,
        "query_leakage_upper_bound": payload["query_leakage_probe_max_abs_difference"],
    }
    payload["artifacts"]["support_envelope_bounds"] = str(bounds_path)
    base._write_json_atomic(result_path, payload)
    return result_path


def _markdown(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.6g}" if isinstance(value, float) else str(value) for value in row
            )
            + " |"
        )
    return "\n".join(rows)


def summarize(root: Path) -> None:
    paths = list(root.glob("results/*/seed*/result.json"))
    payloads = [json.loads(path.read_text()) for path in paths]
    cells = {(payload["job"]["dataset"], payload["job"]["seed"]) for payload in payloads}
    expected = {(dataset, seed) for dataset, _ in base.DATASETS for seed in range(40, 50)}
    if cells != expected:
        raise RuntimeError(f"formal cell mismatch: found {len(cells)}, expected {len(expected)}")
    rows = []
    for payload in payloads:
        for method, metrics in payload["metrics"].items():
            rows.append(
                {
                    "dataset": payload["job"]["dataset"],
                    "seed": payload["job"]["seed"],
                    "method": method,
                    **metrics,
                    "projection_active_fraction": payload["projection"]["active_fraction"],
                    "bound_violations": payload["projection"]["bound_violations"],
                    "leakage": payload["projection"]["query_leakage_upper_bound"],
                }
            )
    all_runs = pd.DataFrame(rows).sort_values(["dataset", "seed", "method"])
    all_runs.to_csv(root / "all_runs.csv", index=False)
    summary = (
        all_runs.groupby(["dataset", "method"])
        .agg(
            median_nrmse=("reference_nrmse", "median"),
            p90_nrmse=("reference_nrmse", lambda values: values.quantile(0.9)),
            max_nrmse=("reference_nrmse", "max"),
            catastrophic_runs=("reference_nrmse", lambda values: int((values > 1).sum())),
            median_active_fraction=("projection_active_fraction", "median"),
        )
        .reset_index()
    )
    summary.to_csv(root / "method_summary.csv", index=False)
    effects = []
    projected = all_runs[all_runs.method == "projected_q"]
    for dataset, _ in base.DATASETS:
        for anchor in ("latent_q", "support_knn", "selector"):
            paired = projected[projected.dataset == dataset].merge(
                all_runs[(all_runs.dataset == dataset) & (all_runs.method == anchor)],
                on="seed",
                suffixes=("_projected", "_anchor"),
            )
            delta = paired.reference_nrmse_projected - paired.reference_nrmse_anchor
            effects.append(
                {
                    "dataset": dataset,
                    "anchor": anchor,
                    "wins": int((delta < 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "median_delta": float(delta.median()),
                    "wilcoxon_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0,
                }
            )
    effects_frame = pd.DataFrame(effects)
    effects_frame["wilcoxon_bh_q"] = base._bh_adjust(effects_frame.wilcoxon_p.tolist())
    effects_frame.to_csv(root / "paired_effects.csv", index=False)
    stability = base._seed_stability(root)
    stability.to_csv(root / "q_seed_stability.csv", index=False)

    lookup = summary.set_index(["dataset", "method"])
    nasa = "nasa_battery_capacity"
    starry = [dataset for dataset, _ in base.DATASETS if dataset != nasa]
    gates = {
        "integrity": bool(
            len(paths) == 40
            and len(all_runs) == 200
            and all_runs.leakage.max() <= 1e-7
            and all_runs.bound_violations.sum() == 0
            and np.isfinite(all_runs.reference_nrmse).all()
        ),
        "nasa_retention": bool(
            lookup.loc[(nasa, "projected_q"), "median_nrmse"]
            <= 1.05 * lookup.loc[(nasa, "latent_q"), "median_nrmse"]
            and lookup.loc[(nasa, "projected_q"), "catastrophic_runs"] == 0
        ),
        "starry_safety": bool(
            all(
                lookup.loc[(dataset, "projected_q"), "catastrophic_runs"] == 0
                and lookup.loc[(dataset, "projected_q"), "median_nrmse"]
                <= lookup.loc[(dataset, "latent_q"), "median_nrmse"]
                for dataset in starry
            )
        ),
        "pooled_improvement": bool(
            projected.reference_nrmse.median()
            < all_runs[all_runs.method == "latent_q"].reference_nrmse.median()
        ),
    }
    audit = {
        "results": len(paths),
        "unique_cells": len(cells),
        "method_rows": len(all_runs),
        "max_leakage_upper_bound": float(all_runs.leakage.max()),
        "bound_violations": int(all_runs.bound_violations.sum()),
        "gates": gates,
        "advancement": "PASS" if all(gates.values()) else "FAIL",
    }
    base._write_json_atomic(root / "terminal_audit.json", audit)
    report = (
        "# Support-envelope projected-q confirmation\n\n"
        "## Material Passport\n\n"
        "- Origin Skill: experiment-agent\n"
        "- Verification Status: VERIFIED TERMINAL\n"
        "- Version Label: support_envelope_projected_q_v1\n\n"
        "## Per-dataset results\n\n"
        + _markdown(summary)
        + "\n\n## Paired effects\n\n"
        + _markdown(effects_frame)
        + "\n\n## Predeclared gates\n\n```json\n"
        + json.dumps(audit, indent=2)
        + "\n```\n"
    )
    (root / "SUPPORT_ENVELOPE_PROJECTED_Q_RESULTS.md").write_text(report)


def main() -> None:
    args = parse_args()
    if args.command == "run-job":
        print(run_job(args))
    elif args.command == "summarize":
        summarize(args.output_root)
    else:
        base.DEFAULT_ROOT = DEFAULT_ROOT
        base.PLAN_PATH = PLAN_PATH
        base.__file__ = __file__
        base.launch(args)


if __name__ == "__main__":
    main()
