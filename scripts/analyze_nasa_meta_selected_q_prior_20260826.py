#!/usr/bin/env python3
"""Aggregate and gate the frozen NASA meta-selected soft-q-prior diagnostic."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


METHODS = ("prefix_q_continuity_step1", "prefix_q_mse_step1")
CONTINUITY = "prefix_q_continuity_step1"
OLD_METHOD = {
    "prefix_q_continuity_step1": "joint_continuity_step1",
    "prefix_q_mse_step1": "joint_mse_step1",
}
PRIOR_WEIGHTS = {0.0, 0.001, 0.01, 0.1, 1.0}
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("non-finite cross-seed Spearman")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-baseline-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    frames = []
    q_frames = []
    score_frames = []
    prediction_frames = []
    for method in METHODS:
        status = json.loads((root / method / "status.json").read_text())
        expected = {"state": "completed_all", "planned": 15, "success": 15, "failed": 0}
        if status != expected:
            raise ValueError(f"nonterminal status for {method}: {status}")
        cells = pd.read_csv(root / method / "cell_summary.csv")
        if len(cells) != 15 or (cells.status != "success").any():
            raise ValueError(f"expected 15 successful cells for {method}")
        frames.append(cells)
        for path in sorted((root / method).glob("**/selected_support_matched_q.csv")):
            cell = json.loads((path.parent / "cell_summary.json").read_text())
            keys = {
                "dataset": cell["dataset"],
                "method": cell["method"],
                "seed": cell["seed"],
            }
            q_frames.append(pd.read_csv(path).assign(**keys))
            score_frames.append(
                pd.read_csv(path.parent / "meta_prior_scores.csv").assign(**keys)
            )
            prediction_frames.append(
                pd.read_csv(path.parent / "query_predictions.csv").assign(**keys)
            )
    cells = pd.concat(frames, ignore_index=True).sort_values(
        ["method", "dataset", "seed"]
    )
    q_all = pd.concat(q_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(cells) != 30 or len(q_all) != 390 or len(scores) != 150:
        raise ValueError("expected 30 cells, 13 selected q and 5 prior scores per cell")

    old = pd.read_csv(args.old_baseline_root.resolve() / "all_cells.csv")[
        ["dataset", "method", "seed", "recalibrated_validation_reference_nrmse"]
    ].rename(
        columns={"recalibrated_validation_reference_nrmse": "old_reference_nrmse"}
    )
    cells["old_method"] = cells.method.map(OLD_METHOD)
    cells = cells.merge(
        old,
        left_on=["dataset", "old_method", "seed"],
        right_on=["dataset", "method", "seed"],
        suffixes=("", "_old"),
        validate="one_to_one",
    ).drop(columns="method_old")
    cells["selected_to_old_nrmse_ratio"] = (
        cells.selected_prior_validation_reference_nrmse / cells.old_reference_nrmse
    )

    q_columns = ["q1", "q2", "q3", "q4"]
    q_pair_rows = []
    functional_pair_rows = []
    for (dataset, method), group in q_all.loc[q_all.split == "meta_fit"].groupby(
        ["dataset", "method"]
    ):
        by_seed = {seed: frame for seed, frame in group.groupby("seed")}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            merged = by_seed[seed_a].merge(
                by_seed[seed_b],
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            q_pair_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "spearman": _finite_spearman(
                        pdist(merged[[f"{column}_a" for column in q_columns]]),
                        pdist(merged[[f"{column}_b" for column in q_columns]]),
                    ),
                }
            )
            for coordinate in FUNCTIONAL_COLUMNS:
                functional_pair_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "coordinate": coordinate,
                        "spearman": _finite_spearman(
                            merged[f"{coordinate}_a"].to_numpy(),
                            merged[f"{coordinate}_b"].to_numpy(),
                        ),
                    }
                )
    q_pairs = pd.DataFrame(q_pair_rows)
    functional_pairs = pd.DataFrame(functional_pair_rows)
    q_stability = (
        q_pairs.groupby(["dataset", "method"], as_index=False)
        .spearman.median()
        .rename(columns={"spearman": "median_spearman"})
    )
    functional_stability = (
        functional_pairs.groupby(
            ["dataset", "method", "coordinate"], as_index=False
        )
        .spearman.median()
        .rename(columns={"spearman": "median_spearman"})
    )
    continuity_q = q_stability.loc[q_stability.method == CONTINUITY]
    continuity_functional = functional_stability.loc[
        functional_stability.method == CONTINUITY
    ]

    numeric_columns = [
        "query_target_leakage_max_q_difference",
        "raw_q_validation_max_abs_z",
        "functional_validation_max_abs_z",
        "selected_prior_validation_reference_nrmse",
        "prefix_q_no_prior_reference_nrmse",
        "old_reference_nrmse",
        "selected_to_old_nrmse_ratio",
    ]
    continuity = cells.loc[cells.method == CONTINUITY]
    functional_gate_rows = continuity_functional.groupby("coordinate").agg(
        median_of_split_medians=("median_spearman", "median"),
        min_split_median=("median_spearman", "min"),
    )
    score_group_sizes = scores.groupby(["dataset", "method", "seed"]).size()
    gates = {
        "gate_1_integrity": bool(
            np.isfinite(cells[numeric_columns].to_numpy(float)).all()
            and np.isfinite(q_all[q_columns].to_numpy(float)).all()
            and np.isfinite(predictions[["target", "prediction"]].to_numpy(float)).all()
            and (cells.meta_fit_labels == 8).all()
            and (cells.structure_validation_labels == 5).all()
            and (cells.prior_weights_scored == 5).all()
            and (score_group_sizes == 5).all()
            and set(scores.prior_weight) == PRIOR_WEIGHTS
            and set(cells.selected_prior_weight).issubset(PRIOR_WEIGHTS)
            and cells.query_target_leakage_max_q_difference.max() == 0.0
        ),
        "gate_2_prediction_retention": bool(
            continuity.selected_prior_validation_reference_nrmse.median()
            <= 1.05 * continuity.old_reference_nrmse.median()
            and int((continuity.selected_to_old_nrmse_ratio <= 1.10).sum()) >= 10
        ),
        "gate_3_interface_safety": bool(
            continuity.raw_q_validation_max_abs_z.median() <= 3.0
            and continuity.functional_validation_max_abs_z.median() <= 3.0
            and int((continuity.functional_validation_max_abs_z <= 6.0).sum()) >= 12
        ),
        "gate_4_representation_stability": bool(
            continuity_q.median_spearman.min() >= 0.80
            and (functional_gate_rows.median_of_split_medians >= 0.70).all()
            and (functional_gate_rows.min_split_median >= 0.50).all()
        ),
    }
    gates["advance_to_bounded_symbolic_stage_c2"] = bool(all(gates.values()))

    summary = (
        cells.groupby("method", as_index=False)
        .agg(
            cells=("seed", "size"),
            selected_nrmse_median=("selected_prior_validation_reference_nrmse", "median"),
            prefix_nrmse_median=("prefix_q_no_prior_reference_nrmse", "median"),
            old_nrmse_median=("old_reference_nrmse", "median"),
            old_ratio_median=("selected_to_old_nrmse_ratio", "median"),
            retained_cells=("selected_to_old_nrmse_ratio", lambda values: int((values <= 1.10).sum())),
            raw_z_median=("raw_q_validation_max_abs_z", "median"),
            raw_z_max=("raw_q_validation_max_abs_z", "max"),
            functional_z_median=("functional_validation_max_abs_z", "median"),
            functional_z_max=("functional_validation_max_abs_z", "max"),
        )
        .sort_values("method")
    )
    weight_counts = (
        cells.groupby(["method", "selected_prior_weight"]).size().reset_index(name="cells")
    )

    cells.to_csv(root / "all_cells.csv", index=False)
    q_all.to_csv(root / "all_selected_q.csv", index=False)
    scores.to_csv(root / "all_meta_prior_scores.csv", index=False)
    predictions.to_csv(root / "all_query_predictions.csv", index=False)
    q_pairs.to_csv(root / "cross_seed_q_pairs.csv", index=False)
    q_stability.to_csv(root / "cross_seed_q_stability.csv", index=False)
    functional_pairs.to_csv(root / "cross_seed_functional_pairs.csv", index=False)
    functional_stability.to_csv(root / "cross_seed_functional_stability.csv", index=False)
    summary.to_csv(root / "method_summary.csv", index=False)
    weight_counts.to_csv(root / "selected_prior_weight_counts.csv", index=False)
    (root / "gate_decision.json").write_text(json.dumps(gates, indent=2))

    report_rows = []
    for row in summary.itertuples(index=False):
        report_rows.append(
            [
                row.method,
                f"{row.selected_nrmse_median:.4g}",
                f"{row.prefix_nrmse_median:.4g}",
                f"{row.old_nrmse_median:.4g}",
                f"{row.old_ratio_median:.4g}",
                f"{row.retained_cells}/15",
                f"{row.raw_z_median:.4g} / {row.raw_z_max:.4g}",
                f"{row.functional_z_median:.4g} / {row.functional_z_max:.4g}",
            ]
        )
    stability_rows = [
        ["q distance", f"{continuity_q.median_spearman.median():.4g}", f"{continuity_q.median_spearman.min():.4g}"]
    ]
    for coordinate, row in functional_gate_rows.iterrows():
        stability_rows.append(
            [coordinate, f"{row.median_of_split_medians:.4g}", f"{row.min_split_median:.4g}"]
        )
    report = [
        "# NASA meta-selected soft q-prior diagnostic",
        "",
        f"**冻结判定：** {'ADVANCE' if gates['advance_to_bounded_symbolic_stage_c2'] else 'DO NOT ADVANCE'}",
        "",
        "prior weight 只由八个 meta-fit 实体各自前 30% support 内部的 selection loss 选择；后续目标和 structure-validation query 不参与选择。",
        "",
        "## 预测与接口",
        "",
        *_table(
            ["方法", "selected", "prefix λ=0", "旧方法", "对旧 ratio", "保持", "raw z 中位/最大", "functional z 中位/最大"],
            report_rows,
        ),
        "",
        "## Continuity selected meta-fit 稳定性",
        "",
        *_table(["对象", "split 中位的中位", "最差 split 中位"], stability_rows),
        "",
        "## 选择的 prior weight",
        "",
        *_table(
            ["方法", "weight", "cells"],
            [[row.method, f"{row.selected_prior_weight:g}", str(row.cells)] for row in weight_counts.itertuples(index=False)],
        ),
        "",
        "## Gate",
        "",
        *_table(["Gate", "结果"], [[key, "PASS" if value else "FAIL"] for key, value in gates.items()]),
        "",
        "本轮仍是已暴露 inner splits 上的开发诊断；即使通过也只允许独立冻结 bounded symbolic Stage C2。",
    ]
    (root / "META_SELECTED_Q_PRIOR_REPORT.md").write_text("\n".join(report) + "\n")
    (root / "status.json").write_text(
        json.dumps(
            {"state": "completed_all", "planned": 30, "success": 30, "failed": 0, "advance": gates["advance_to_bounded_symbolic_stage_c2"]},
            indent=2,
        )
    )
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
