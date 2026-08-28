#!/usr/bin/env python3
"""Analyze the frozen NASA protocol-matched functional-prior Phase B."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_nasa_support_matched_q_diagnostic_20260826 as matched


Q_COLUMNS = ("q1", "q2", "q3", "q4")
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")
RESPONSE_COLUMNS = (
    "response_cycle1",
    "response_cycle10",
    "response_cycle20",
    "response_cycle28",
)
WEIGHTS = {0.0, 0.01}


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError("non-finite Spearman correlation")
    return value


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    status = json.loads((root / "status.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    paths = sorted(root.glob("**/cell_summary.json"))
    if len(paths) != 15:
        raise ValueError(f"expected 15 cells, found {len(paths)}")

    cells = []
    q_frames = []
    score_frames = []
    for path in paths:
        cell = json.loads(path.read_text())
        keys = {"dataset": cell["dataset"], "seed": int(cell["seed"])}
        cells.append(cell)
        q_frames.append(pd.read_csv(path.parent / "validation_q.csv").assign(**keys))
        score_frames.append(pd.read_csv(path.parent / "weight_scores.csv").assign(**keys))
    cells = pd.DataFrame(cells).sort_values(["dataset", "seed"])
    q_values = pd.concat(q_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)

    numeric = [
        "query_target_leakage_max_q_difference",
        "baseline_weight0_validation_reference_nrmse",
        "selected_validation_reference_nrmse",
        "selected_to_baseline_nrmse_ratio",
    ]
    integrity = bool(
        status == {"state": "completed_all", "planned": 15, "success": 15, "failed": 0}
        and len(cells) == 15
        and len(q_values) == 150
        and len(scores) == 30
        and np.isfinite(cells[numeric].to_numpy(float)).all()
        and np.isfinite(
            q_values[
                list(Q_COLUMNS) + list(FUNCTIONAL_COLUMNS) + list(RESPONSE_COLUMNS)
            ].to_numpy(float)
        ).all()
        and (cells.structure_validation_labels == 5).all()
        and (cells.prior_weights_scored == 2).all()
        and (cells.validation_q_rows == 10).all()
        and (cells.selected_weight == 0.01).all()
        and cells.query_target_leakage_max_q_difference.max() == 0.0
        and set(q_values.prior_weight) == WEIGHTS
        and set(scores.prior_weight) == WEIGHTS
    )

    stability_rows = []
    for (dataset, prior_weight), frame in q_values.groupby(
        ["dataset", "prior_weight"]
    ):
        by_seed = {int(seed): group for seed, group in frame.groupby("seed")}
        values = {"raw_q_distance": [], "response_distance": []}
        values.update({coordinate: [] for coordinate in FUNCTIONAL_COLUMNS})
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            merged = by_seed[seed_a].merge(
                by_seed[seed_b],
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            values["raw_q_distance"].append(
                _spearman(
                    pdist(merged[[f"{column}_a" for column in Q_COLUMNS]]),
                    pdist(merged[[f"{column}_b" for column in Q_COLUMNS]]),
                )
            )
            values["response_distance"].append(
                _spearman(
                    pdist(merged[[f"{column}_a" for column in RESPONSE_COLUMNS]]),
                    pdist(merged[[f"{column}_b" for column in RESPONSE_COLUMNS]]),
                )
            )
            for coordinate in FUNCTIONAL_COLUMNS:
                values[coordinate].append(
                    _spearman(
                        merged[f"{coordinate}_a"].to_numpy(float),
                        merged[f"{coordinate}_b"].to_numpy(float),
                    )
                )
        for endpoint, correlations in values.items():
            stability_rows.append(
                {
                    "dataset": dataset,
                    "prior_weight": prior_weight,
                    "endpoint": endpoint,
                    "median_spearman": float(np.median(correlations)),
                    "worst_pair_spearman": float(np.min(correlations)),
                }
            )
    stability = pd.DataFrame(stability_rows)

    q_root = PROJECT_ROOT / manifest["q_root"]
    records = matched._prepared_records(q_root)
    descriptor_frames = []
    for dataset, record in records.items():
        validation = pd.read_csv(matched._resolve(record["test_csv"]))
        rows = []
        for label, frame in validation.groupby("label", sort=False):
            early = frame.loc[frame.discharge_index <= 10]
            slope, intercept = np.polyfit(early.discharge_index, early.target, 1)
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "empirical_capacity_cycle1": float(slope + intercept),
                    "empirical_early_fade_rate": float(-slope),
                }
            )
        descriptor_frames.append(pd.DataFrame(rows))
    descriptors = pd.concat(descriptor_frames, ignore_index=True)
    empirical_rows = []
    for (dataset, seed, prior_weight), frame in q_values.groupby(
        ["dataset", "seed", "prior_weight"]
    ):
        merged = frame.merge(
            descriptors.loc[descriptors.dataset == dataset],
            on=["dataset", "label"],
            validate="one_to_one",
        )
        empirical_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "prior_weight": prior_weight,
                "capacity_spearman": _spearman(
                    merged.capacity_cycle1.to_numpy(float),
                    merged.empirical_capacity_cycle1.to_numpy(float),
                ),
                "early_fade_spearman": _spearman(
                    merged.early_fade_rate.to_numpy(float),
                    merged.empirical_early_fade_rate.to_numpy(float),
                ),
            }
        )
    empirical = pd.DataFrame(empirical_rows)

    prediction_ratio = float(
        cells.selected_validation_reference_nrmse.median()
        / cells.baseline_weight0_validation_reference_nrmse.median()
    )
    retained_cells = int((cells.selected_to_baseline_nrmse_ratio <= 1.10).sum())
    prediction_gate = bool(prediction_ratio <= 1.05 and retained_cells >= 10)

    selected_stability = stability.loc[stability.prior_weight == 0.01]
    functional_summary = (
        selected_stability.loc[
            selected_stability.endpoint.isin(FUNCTIONAL_COLUMNS)
        ]
        .groupby("endpoint")
        .median_spearman.agg(["median", "min"])
    )
    response_by_split = stability.loc[
        stability.endpoint == "response_distance"
    ].pivot(index="dataset", columns="prior_weight", values="median_spearman")
    functional_stability_gate = bool(
        (functional_summary["median"] >= 0.70).all()
        and (functional_summary["min"] >= 0.50).all()
        and (response_by_split[0.01] >= response_by_split[0.0] - 0.05).all()
    )

    empirical_by_split = (
        empirical.groupby(["dataset", "prior_weight"], as_index=False)
        .agg(
            capacity_spearman=("capacity_spearman", "median"),
            early_fade_spearman=("early_fade_spearman", "median"),
        )
    )
    selected_empirical = empirical_by_split.loc[
        empirical_by_split.prior_weight == 0.01
    ]
    baseline_empirical = empirical_by_split.loc[
        empirical_by_split.prior_weight == 0.0
    ]
    capacity_alignment = float(selected_empirical.capacity_spearman.median())
    fade_alignment = float(selected_empirical.early_fade_spearman.median())
    tolerance = 1e-12
    alignment_gate = bool(
        capacity_alignment + tolerance >= 0.70
        and fade_alignment + tolerance >= 0.50
        and capacity_alignment + tolerance
        >= float(baseline_empirical.capacity_spearman.median()) - 0.05
        and fade_alignment + tolerance
        >= float(baseline_empirical.early_fade_spearman.median()) - 0.05
    )

    gate = {
        "integrity": integrity,
        "prediction_retention": prediction_gate,
        "functional_stability": functional_stability_gate,
        "scientific_alignment": alignment_gate,
        "overall_advance_to_stage_c2": bool(
            integrity
            and prediction_gate
            and functional_stability_gate
            and alignment_gate
        ),
        "selected_weight": 0.01,
        "selected_to_baseline_median_nrmse_ratio": prediction_ratio,
        "cells_with_ratio_at_most_1_10": retained_cells,
        "capacity_stability_median_of_splits": float(
            functional_summary.loc["capacity_cycle1", "median"]
        ),
        "capacity_stability_min_split": float(
            functional_summary.loc["capacity_cycle1", "min"]
        ),
        "early_fade_stability_median_of_splits": float(
            functional_summary.loc["early_fade_rate", "median"]
        ),
        "early_fade_stability_min_split": float(
            functional_summary.loc["early_fade_rate", "min"]
        ),
        "capacity_empirical_alignment": capacity_alignment,
        "early_fade_empirical_alignment": fade_alignment,
    }
    paired_test = wilcoxon(
        cells.selected_validation_reference_nrmse,
        cells.baseline_weight0_validation_reference_nrmse,
        alternative="two-sided",
    )

    dataset_summary = (
        cells.groupby("dataset", as_index=False)
        .agg(
            baseline_nrmse=("baseline_weight0_validation_reference_nrmse", "median"),
            selected_nrmse=("selected_validation_reference_nrmse", "median"),
            paired_ratio=("selected_to_baseline_nrmse_ratio", "median"),
            selected_wins=(
                "selected_to_baseline_nrmse_ratio",
                lambda values: int((values < 1.0).sum()),
            ),
        )
        .sort_values("dataset")
    )

    cells.to_csv(root / "all_cells.csv", index=False)
    q_values.to_csv(root / "all_validation_q.csv", index=False)
    scores.to_csv(root / "all_weight_scores.csv", index=False)
    stability.to_csv(root / "cross_seed_stability.csv", index=False)
    empirical.to_csv(root / "empirical_correlations.csv", index=False)
    empirical_by_split.to_csv(root / "empirical_correlation_summary.csv", index=False)
    dataset_summary.to_csv(root / "dataset_summary.csv", index=False)
    (root / "gate_decision.json").write_text(json.dumps(gate, indent=2))

    lines = [
        "# NASA protocol-matched functional prior Phase-B report",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-28",
        "- Verification Status: ANALYZED",
        "- Version Label: nasa_protocol_matched_functional_prior_phaseb_v1",
        "",
        f"**Stage-C2 decision:** {'AUTHORIZE' if gate['overall_advance_to_stage_c2'] else 'STOP'}",
        "",
        *_table(
            ["gate", "result"],
            [
                [name, "PASS" if gate[name] else "FAIL"]
                for name in (
                    "integrity",
                    "prediction_retention",
                    "functional_stability",
                    "scientific_alignment",
                )
            ],
        ),
        "",
        "## Per-dataset prediction",
        "",
        *_table(
            ["dataset", "weight 0", "weight 0.01", "paired ratio", "wins/5"],
            [
                [
                    row.dataset,
                    f"{row.baseline_nrmse:.4g}",
                    f"{row.selected_nrmse:.4g}",
                    f"{row.paired_ratio:.3f}",
                    f"{int(row.selected_wins)}/5",
                ]
                for row in dataset_summary.itertuples(index=False)
            ],
        ),
        "",
        f"Overall median-NRMSE ratio: {prediction_ratio:.4f}; cells within 10%: {retained_cells}/15; paired Wilcoxon p={float(paired_test.pvalue):.4g}.",
        "",
        "## Functional endpoints",
        "",
        f"Capacity stability median/min split: {gate['capacity_stability_median_of_splits']:.3f}/{gate['capacity_stability_min_split']:.3f}.",
        f"Early-fade stability median/min split: {gate['early_fade_stability_median_of_splits']:.3f}/{gate['early_fade_stability_min_split']:.3f}.",
        f"Empirical alignment capacity/early fade: {capacity_alignment:.3f}/{fade_alignment:.3f}.",
        "",
        "This protocol did not reselect the weight on structure-validation outcomes. These batteries are protocol-held-out but not globally untouched by the broader project history.",
    ]
    (root / "PHASEB_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
