#!/usr/bin/env python3
"""Aggregate the frozen NASA functional-response-prior meta-only screen."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


Q_COLUMNS = ("q1", "q2", "q3", "q4")
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")
RESPONSE_COLUMNS = (
    "response_cycle1",
    "response_cycle10",
    "response_cycle20",
    "response_cycle28",
)
PRIOR_WEIGHTS = {0.0, 0.001, 0.01, 0.1, 1.0}


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
    parser.add_argument(
        "--representation-gate",
        choices=("raw-q", "functional-response"),
        default="raw-q",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    status = json.loads((root / "status.json").read_text())
    if status != {
        "state": "completed_all",
        "planned": 15,
        "success": 15,
        "failed": 0,
    }:
        raise ValueError(f"nonterminal status: {status}")

    cell_frames = []
    score_frames = []
    candidate_frames = []
    paths = sorted(root.glob("**/cell_summary.json"))
    if len(paths) != 15:
        raise ValueError("expected 15 cell summaries")
    for path in paths:
        cell = json.loads(path.read_text())
        keys = {"dataset": cell["dataset"], "seed": int(cell["seed"])}
        cell_frames.append(pd.DataFrame([cell]))
        score_frames.append(pd.read_csv(path.parent / "prior_scores.csv").assign(**keys))
        candidate_frames.append(
            pd.read_csv(path.parent / "meta_q_candidates.csv").assign(**keys)
        )
    cells = pd.concat(cell_frames, ignore_index=True).sort_values(["dataset", "seed"])
    scores = pd.concat(score_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    numeric = [
        "query_target_leakage_max_q_difference",
        "meta_fit_labels",
        "prior_weights_scored",
        "candidate_q_rows",
    ]
    integrity = bool(
        len(cells) == 15
        and len(scores) == 75
        and len(candidates) == 600
        and np.isfinite(cells[numeric].to_numpy(float)).all()
        and np.isfinite(scores.select_dtypes(include=[np.number]).to_numpy(float)).all()
        and np.isfinite(
            candidates[
                list(Q_COLUMNS) + list(FUNCTIONAL_COLUMNS) + list(RESPONSE_COLUMNS)
            ].to_numpy(float)
        ).all()
        and (cells.meta_fit_labels == 8).all()
        and (cells.prior_weights_scored == 5).all()
        and (cells.candidate_q_rows == 40).all()
        and cells.query_target_leakage_max_q_difference.max() == 0.0
        and set(scores.prior_weight) == PRIOR_WEIGHTS
        and set(candidates.prior_weight) == PRIOR_WEIGHTS
    )

    q_rows = []
    response_rows = []
    functional_rows = []
    for (dataset, prior_weight), frame in candidates.groupby(
        ["dataset", "prior_weight"]
    ):
        by_seed = {int(seed): group for seed, group in frame.groupby("seed")}
        q_values = []
        response_values = []
        functional_values = {coordinate: [] for coordinate in FUNCTIONAL_COLUMNS}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            merged = by_seed[seed_a].merge(
                by_seed[seed_b],
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            q_values.append(
                _spearman(
                    pdist(merged[[f"{column}_a" for column in Q_COLUMNS]]),
                    pdist(merged[[f"{column}_b" for column in Q_COLUMNS]]),
                )
            )
            response_values.append(
                _spearman(
                    pdist(
                        merged[
                            [f"{column}_a" for column in RESPONSE_COLUMNS]
                        ]
                    ),
                    pdist(
                        merged[
                            [f"{column}_b" for column in RESPONSE_COLUMNS]
                        ]
                    ),
                )
            )
            for coordinate in FUNCTIONAL_COLUMNS:
                functional_values[coordinate].append(
                    _spearman(
                        merged[f"{coordinate}_a"].to_numpy(float),
                        merged[f"{coordinate}_b"].to_numpy(float),
                    )
                )
        q_rows.append(
            {
                "dataset": dataset,
                "prior_weight": prior_weight,
                "median_spearman": float(np.median(q_values)),
                "min_pair_spearman": float(np.min(q_values)),
            }
        )
        response_rows.append(
            {
                "dataset": dataset,
                "prior_weight": prior_weight,
                "median_spearman": float(np.median(response_values)),
                "min_pair_spearman": float(np.min(response_values)),
            }
        )
        for coordinate, values in functional_values.items():
            functional_rows.append(
                {
                    "dataset": dataset,
                    "prior_weight": prior_weight,
                    "coordinate": coordinate,
                    "median_spearman": float(np.median(values)),
                    "min_pair_spearman": float(np.min(values)),
                }
            )
    q_stability = pd.DataFrame(q_rows)
    response_stability = pd.DataFrame(response_rows)
    functional_stability = pd.DataFrame(functional_rows)

    score_summary = (
        scores.groupby("prior_weight", as_index=False)
        .agg(
            support_loss_median=("support_selection_loss_median", "median"),
            meta_query_nrmse_median=("meta_query_nrmse_median", "median"),
        )
        .sort_values("prior_weight")
    )
    q_gate = (
        q_stability.groupby("prior_weight", as_index=False)
        .agg(
            q_median_of_splits=("median_spearman", "median"),
            q_min_split=("median_spearman", "min"),
        )
    )
    response_gate = (
        response_stability.groupby("prior_weight", as_index=False)
        .agg(
            response_median_of_splits=("median_spearman", "median"),
            response_min_split=("median_spearman", "min"),
        )
    )
    baseline_response_by_split = response_stability.loc[
        response_stability.prior_weight == 0.0
    ].set_index("dataset").median_spearman
    functional_gate = (
        functional_stability.groupby(["prior_weight", "coordinate"], as_index=False)
        .agg(
            median_of_splits=("median_spearman", "median"),
            min_split=("median_spearman", "min"),
        )
    )
    rows = []
    baseline = float(
        score_summary.loc[
            score_summary.prior_weight == 0.0, "meta_query_nrmse_median"
        ].iloc[0]
    )
    for score in score_summary.itertuples(index=False):
        q_row = q_gate.loc[q_gate.prior_weight == score.prior_weight].iloc[0]
        response_row = response_gate.loc[
            response_gate.prior_weight == score.prior_weight
        ].iloc[0]
        response_by_split = response_stability.loc[
            response_stability.prior_weight == score.prior_weight
        ].set_index("dataset").median_spearman
        coordinates = functional_gate.loc[
            functional_gate.prior_weight == score.prior_weight
        ].set_index("coordinate")
        prediction_eligible = bool(score.meta_query_nrmse_median <= 1.05 * baseline)
        response_geometry_retained = bool(
            (response_by_split >= baseline_response_by_split - 0.05).all()
        )
        functional_coordinates_eligible = bool(
            (coordinates.median_of_splits >= 0.70).all()
            and (coordinates.min_split >= 0.50).all()
        )
        if args.representation_gate == "raw-q":
            representation_eligible = bool(
                q_row.q_min_split >= 0.80 and functional_coordinates_eligible
            )
        else:
            representation_eligible = bool(
                response_geometry_retained and functional_coordinates_eligible
            )
        rows.append(
            {
                "prior_weight": score.prior_weight,
                "support_loss_median": score.support_loss_median,
                "meta_query_nrmse_median": score.meta_query_nrmse_median,
                "prediction_eligible": prediction_eligible,
                "q_median_of_splits": q_row.q_median_of_splits,
                "q_min_split": q_row.q_min_split,
                "response_median_of_splits": response_row.response_median_of_splits,
                "response_min_split": response_row.response_min_split,
                "response_geometry_retained": response_geometry_retained,
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
                "representation_eligible": representation_eligible,
                "eligible": bool(prediction_eligible and representation_eligible),
            }
        )
    eligibility = pd.DataFrame(rows)
    eligible = eligibility.loc[eligibility.eligible].sort_values(
        ["meta_query_nrmse_median", "prior_weight"], kind="stable"
    )
    selected_weight = (
        None if not integrity or eligible.empty else float(eligible.iloc[0].prior_weight)
    )
    selection = {
        "integrity": integrity,
        "baseline_weight0_meta_query_nrmse_median": baseline,
        "selected_weight": selected_weight,
        "authorize_phase_b_validation": bool(integrity and selected_weight is not None),
        "selection_uses_structure_validation": False,
        "representation_gate": args.representation_gate,
    }

    cells.to_csv(root / "all_cells.csv", index=False)
    scores.to_csv(root / "all_prior_scores.csv", index=False)
    candidates.to_csv(root / "all_meta_q_candidates.csv", index=False)
    q_stability.to_csv(root / "fixed_weight_q_stability.csv", index=False)
    response_stability.to_csv(
        root / "fixed_weight_response_stability.csv", index=False
    )
    functional_stability.to_csv(
        root / "fixed_weight_functional_stability.csv", index=False
    )
    eligibility.to_csv(root / "weight_eligibility.csv", index=False)
    (root / "selected_weight.json").write_text(json.dumps(selection, indent=2))

    report = [
        "# NASA functional-response prior meta-only screen",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-27",
        "- Verification Status: ANALYZED",
        f"- Version Label: nasa_functional_response_prior_meta_v1_{args.representation_gate}",
        "",
        f"**Phase-B decision:** {'AUTHORIZE' if selection['authorize_phase_b_validation'] else 'STOP'}",
        "",
        "本屏只使用八个 meta-fit batteries；structure-validation 数据未被读取。候选必须同时保留 later-cycle meta-query prediction 并通过预先声明的 representation gate。",
        "",
        *_table(
            ["weight", "meta-query", "pred", "q 中位/最差", "response 中位/最差", "capacity 中位/最差", "fade 中位/最差", "repr", "eligible"],
            [
                [
                    f"{row.prior_weight:g}",
                    f"{row.meta_query_nrmse_median:.4g}",
                    "PASS" if row.prediction_eligible else "FAIL",
                    f"{row.q_median_of_splits:.3f}/{row.q_min_split:.3f}",
                    f"{row.response_median_of_splits:.3f}/{row.response_min_split:.3f}",
                    f"{row.capacity_median_of_splits:.3f}/{row.capacity_min_split:.3f}",
                    f"{row.early_fade_median_of_splits:.3f}/{row.early_fade_min_split:.3f}",
                    "PASS" if row.representation_eligible else "FAIL",
                    "YES" if row.eligible else "NO",
                ]
                for row in eligibility.itertuples(index=False)
            ],
        ),
        "",
        f"selected weight: {selected_weight if selected_weight is not None else 'none'}",
    ]
    (root / "FUNCTIONAL_RESPONSE_PRIOR_META_REPORT.md").write_text(
        "\n".join(report) + "\n"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
