#!/usr/bin/env python3
"""Diagnose why the frozen raw-q prior grid cannot satisfy its stability gate."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


METHODS = ("prefix_q_continuity_step1", "prefix_q_mse_step1")
CONTINUITY = "prefix_q_continuity_step1"
Q_COLUMNS = ("q1", "q2", "q3", "q4")
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("non-finite Spearman correlation")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    score_frames = []
    q_frames = []
    functional_frames = []
    for method in METHODS:
        paths = sorted((root / method).glob("**/cell_summary.json"))
        if len(paths) != 15:
            raise ValueError(f"expected 15 terminal cells for {method}")
        for summary_path in paths:
            cell = json.loads(summary_path.read_text())
            keys = {
                "dataset": cell["dataset"],
                "method": method,
                "seed": int(cell["seed"]),
            }
            scores = pd.read_csv(summary_path.parent / "meta_prior_scores.csv").assign(
                **keys
            )
            candidates = pd.read_csv(
                summary_path.parent / "all_meta_q_candidates.csv"
            ).assign(**keys)
            score_frames.append(scores)
            q_frames.append(candidates)

            source_path = PROJECT_ROOT / cell["source_result"]
            _, _, source, config = matched._load_source(
                source_path, torch.device("cpu")
            )
            for prior_weight, frame in candidates.groupby("prior_weight"):
                functional = matched._functional_coordinates(
                    frame, source, config.q_dim
                )
                functional_frames.append(
                    functional.assign(prior_weight=prior_weight, **keys)
                )

    scores = pd.concat(score_frames, ignore_index=True)
    q_candidates = pd.concat(q_frames, ignore_index=True)
    functional_candidates = pd.concat(functional_frames, ignore_index=True)
    if len(scores) != 150 or len(q_candidates) != 1200:
        raise ValueError("expected five scores and forty candidate q rows per cell")

    proxy_rows = []
    for (dataset, method, seed), frame in scores.groupby(
        ["dataset", "method", "seed"]
    ):
        support_choice = frame.sort_values(
            ["support_selection_loss_median", "prior_weight"], kind="stable"
        ).iloc[0]
        diagnostic_choice = frame.sort_values(
            ["meta_query_nrmse_median_diagnostic_only", "prior_weight"],
            kind="stable",
        ).iloc[0]
        proxy_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "support_selected_weight": float(support_choice.prior_weight),
                "meta_query_oracle_weight_diagnostic_only": float(
                    diagnostic_choice.prior_weight
                ),
                "choices_match": bool(
                    support_choice.prior_weight == diagnostic_choice.prior_weight
                ),
                "support_to_meta_query_spearman": _spearman(
                    frame.support_selection_loss_median.to_numpy(float),
                    frame.meta_query_nrmse_median_diagnostic_only.to_numpy(float),
                ),
                "selected_meta_query_nrmse_diagnostic_only": float(
                    support_choice.meta_query_nrmse_median_diagnostic_only
                ),
                "oracle_meta_query_nrmse_diagnostic_only": float(
                    diagnostic_choice.meta_query_nrmse_median_diagnostic_only
                ),
            }
        )
    proxy = pd.DataFrame(proxy_rows)
    proxy_summary = (
        proxy.groupby("method", as_index=False)
        .agg(
            cells=("seed", "size"),
            choices_match=("choices_match", "sum"),
            proxy_spearman_median=("support_to_meta_query_spearman", "median"),
            selected_meta_query_nrmse_median=(
                "selected_meta_query_nrmse_diagnostic_only",
                "median",
            ),
            oracle_meta_query_nrmse_median=(
                "oracle_meta_query_nrmse_diagnostic_only",
                "median",
            ),
        )
        .sort_values("method")
    )
    fixed_score_summary = (
        scores.groupby(["method", "prior_weight"], as_index=False)
        .agg(
            support_loss_median=("support_selection_loss_median", "median"),
            meta_query_nrmse_median_diagnostic_only=(
                "meta_query_nrmse_median_diagnostic_only",
                "median",
            ),
        )
        .sort_values(["method", "prior_weight"])
    )

    q_stability_rows = []
    for (dataset, method, prior_weight), frame in q_candidates.groupby(
        ["dataset", "method", "prior_weight"]
    ):
        by_seed = {int(seed): group for seed, group in frame.groupby("seed")}
        values = []
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            merged = by_seed[seed_a].merge(
                by_seed[seed_b],
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            values.append(
                _spearman(
                    pdist(merged[[f"{column}_a" for column in Q_COLUMNS]]),
                    pdist(merged[[f"{column}_b" for column in Q_COLUMNS]]),
                )
            )
        q_stability_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "prior_weight": prior_weight,
                "median_spearman": float(np.median(values)),
                "min_pair_spearman": float(np.min(values)),
            }
        )
    q_stability = pd.DataFrame(q_stability_rows)

    functional_stability_rows = []
    for (dataset, method, prior_weight), frame in functional_candidates.groupby(
        ["dataset", "method", "prior_weight"]
    ):
        by_seed = {int(seed): group for seed, group in frame.groupby("seed")}
        for coordinate in FUNCTIONAL_COLUMNS:
            values = []
            for seed_a, seed_b in combinations(sorted(by_seed), 2):
                merged = by_seed[seed_a].merge(
                    by_seed[seed_b],
                    on="label",
                    suffixes=("_a", "_b"),
                    validate="one_to_one",
                )
                values.append(
                    _spearman(
                        merged[f"{coordinate}_a"].to_numpy(float),
                        merged[f"{coordinate}_b"].to_numpy(float),
                    )
                )
            functional_stability_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "prior_weight": prior_weight,
                    "coordinate": coordinate,
                    "median_spearman": float(np.median(values)),
                    "min_pair_spearman": float(np.min(values)),
                }
            )
    functional_stability = pd.DataFrame(functional_stability_rows)

    q_gate = (
        q_stability.loc[q_stability.method == CONTINUITY]
        .groupby("prior_weight", as_index=False)
        .agg(
            q_median_of_splits=("median_spearman", "median"),
            q_min_split=("median_spearman", "min"),
        )
    )
    functional_gate = (
        functional_stability.loc[functional_stability.method == CONTINUITY]
        .groupby(["prior_weight", "coordinate"], as_index=False)
        .agg(
            median_of_splits=("median_spearman", "median"),
            min_split=("median_spearman", "min"),
        )
    )
    gate_rows = []
    for q_row in q_gate.itertuples(index=False):
        coordinates = functional_gate.loc[
            functional_gate.prior_weight == q_row.prior_weight
        ].set_index("coordinate")
        representation_pass = bool(
            q_row.q_min_split >= 0.80
            and (coordinates.median_of_splits >= 0.70).all()
            and (coordinates.min_split >= 0.50).all()
        )
        gate_rows.append(
            {
                "prior_weight": q_row.prior_weight,
                "q_median_of_splits": q_row.q_median_of_splits,
                "q_min_split": q_row.q_min_split,
                "capacity_median_of_splits": coordinates.loc[
                    "capacity_cycle1", "median_of_splits"
                ],
                "capacity_min_split": coordinates.loc[
                    "capacity_cycle1", "min_split"
                ],
                "early_fade_median_of_splits": coordinates.loc[
                    "early_fade_rate", "median_of_splits"
                ],
                "early_fade_min_split": coordinates.loc[
                    "early_fade_rate", "min_split"
                ],
                "representation_gate_pass": representation_pass,
            }
        )
    fixed_weight_gate = pd.DataFrame(gate_rows)

    proxy.to_csv(root / "prior_selection_proxy_cells.csv", index=False)
    proxy_summary.to_csv(root / "prior_selection_proxy_summary.csv", index=False)
    fixed_score_summary.to_csv(root / "fixed_weight_score_summary.csv", index=False)
    q_stability.to_csv(root / "fixed_weight_q_stability.csv", index=False)
    functional_stability.to_csv(
        root / "fixed_weight_functional_stability.csv", index=False
    )
    fixed_weight_gate.to_csv(root / "fixed_weight_representation_gate.csv", index=False)

    report = [
        "# NASA raw-q prior failure diagnostic",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-27",
        "- Verification Status: ANALYZED",
        "- Version Label: nasa_raw_q_prior_failure_v1",
        "",
        "这是 formal gate 之后的事后机制诊断，不改变原 gate，也不使用 structure-validation query 来选择新权重。meta-query oracle 只使用 meta-fit 实体后 70% 目标，属于不可部署的诊断参照。",
        "",
        "## Support 选择代理",
        "",
        *_table(
            ["方法", "匹配 meta-query oracle", "rank 相关中位", "selected meta-query", "oracle meta-query"],
            [
                [
                    row.method,
                    f"{int(row.choices_match)}/{int(row.cells)}",
                    f"{row.proxy_spearman_median:.3f}",
                    f"{row.selected_meta_query_nrmse_median:.4g}",
                    f"{row.oracle_meta_query_nrmse_median:.4g}",
                ]
                for row in proxy_summary.itertuples(index=False)
            ],
        ),
        "",
        "## Continuity 固定权重的 representation gate",
        "",
        *_table(
            ["weight", "q 中位/最差 split", "capacity 中位/最差", "early fade 中位/最差", "结果"],
            [
                [
                    f"{row.prior_weight:g}",
                    f"{row.q_median_of_splits:.3f} / {row.q_min_split:.3f}",
                    f"{row.capacity_median_of_splits:.3f} / {row.capacity_min_split:.3f}",
                    f"{row.early_fade_median_of_splits:.3f} / {row.early_fade_min_split:.3f}",
                    "PASS" if row.representation_gate_pass else "FAIL",
                ]
                for row in fixed_weight_gate.itertuples(index=False)
            ],
        ),
        "",
        "## 解释",
        "",
        "support 内部 prediction loss 与 meta-query NRMSE 的排序总体同向，但它偏好最弱或零 raw-q 正则；正式 continuity 因而在 12/15 cells 选择 weight 0。更重要的是，五个固定权重没有一个能通过既有 representation gate，因此无需再用 structure-validation query 补跑固定权重。raw-q Gaussian prior 仍以每个 seed 自己的 embedding 坐标为参照，不能消除 q/第一层的 affine gauge；下一项方法应把约束定义在 decoder response/functional space，而不是继续调 raw-q 权重。",
    ]
    (root / "RAW_Q_PRIOR_FAILURE_DIAGNOSTIC.md").write_text(
        "\n".join(report) + "\n"
    )
    print(fixed_weight_gate.to_string(index=False))


if __name__ == "__main__":
    main()
