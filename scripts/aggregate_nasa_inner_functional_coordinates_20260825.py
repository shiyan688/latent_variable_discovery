#!/usr/bin/env python3
"""Aggregate cross-seed stability for the reviewer-clean NASA inner splits."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


FUNCTIONAL_COLUMNS = [
    "capacity_cycle1",
    "capacity_cycle28",
    "early_fade_rate",
    "mid_fade_rate",
    "fade_acceleration",
]


def finite_spearman(left: np.ndarray, right: np.ndarray, context: str) -> float:
    value = float(spearmanr(left, right).statistic)
    if not np.isfinite(value):
        raise ValueError(f"non-finite Spearman correlation for {context}")
    return value


def summarize(
    rows: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    return (
        rows.groupby(group_columns, as_index=False)
        .agg(
            median_spearman=(value_column, "median"),
            min_spearman=(value_column, "min"),
            max_spearman=(value_column, "max"),
            positive_fraction=(value_column, lambda values: float((values > 0).mean())),
            comparisons=(value_column, "size"),
        )
        .sort_values(group_columns)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    analysis_root = run_root / "functional_coordinate_analysis"
    records: list[dict[str, object]] = []
    for result_path in sorted(run_root.glob("*/*/*/result.json")):
        payload = json.loads(result_path.read_text())
        job = payload["job"]
        relative = result_path.relative_to(run_root)
        coordinate_dir = analysis_root / relative.parent
        metadata = json.loads((coordinate_dir / "metadata.json").read_text())
        if metadata["outer_targets_used"]:
            raise ValueError(f"outer targets used by {coordinate_dir}")
        records.append(
            {
                "dataset": job["dataset"],
                "method": job["method"],
                "q_dim": int(job["q_dim"]),
                "seed": int(job["seed"]),
                "q_path": result_path.with_name("train_label_q.csv"),
                "coordinate_path": coordinate_dir / "functional_coordinates.csv",
                "correlation_path": coordinate_dir / "train_descriptor_correlations.csv",
            }
        )

    record_frame = pd.DataFrame(records)
    if len(record_frame) != 30:
        raise ValueError(f"expected 30 result cells, found {len(record_frame)}")
    group_columns = ["dataset", "method", "q_dim"]
    seed_counts = record_frame.groupby(group_columns)["seed"].nunique()
    if not (seed_counts == 5).all():
        raise ValueError(f"expected five seeds per group:\n{seed_counts}")

    q_pair_rows: list[dict[str, object]] = []
    functional_pair_rows: list[dict[str, object]] = []
    descriptor_rows: list[pd.DataFrame] = []

    for keys, group in record_frame.groupby(group_columns):
        dataset, method, q_dim = keys
        by_seed = {int(row.seed): row for row in group.itertuples(index=False)}
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            row_a = by_seed[seed_a]
            row_b = by_seed[seed_b]
            q_a = pd.read_csv(row_a.q_path)
            q_b = pd.read_csv(row_b.q_path)
            q_columns_a = [column for column in q_a if column.startswith("q") and column[1:].isdigit()]
            q_columns_b = [column for column in q_b if column.startswith("q") and column[1:].isdigit()]
            if q_columns_a != q_columns_b or len(q_columns_a) != q_dim:
                raise ValueError(f"q-column mismatch for {dataset}, {method}, seeds {seed_a}/{seed_b}")
            merged_q = q_a.merge(q_b, on="label", suffixes=("_a", "_b"), validate="one_to_one").sort_values("label")
            distances_a = pdist(merged_q[[f"{column}_a" for column in q_columns_a]].to_numpy())
            distances_b = pdist(merged_q[[f"{column}_b" for column in q_columns_b]].to_numpy())
            q_pair_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "q_dim": q_dim,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "distance_spearman": finite_spearman(
                        distances_a,
                        distances_b,
                        f"q distances: {dataset}, {method}, seeds {seed_a}/{seed_b}",
                    ),
                    "entities": len(merged_q),
                }
            )

            coordinates_a = pd.read_csv(row_a.coordinate_path).query("split == 'train'")
            coordinates_b = pd.read_csv(row_b.coordinate_path).query("split == 'train'")
            merged_coordinates = coordinates_a.merge(
                coordinates_b,
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            for coordinate in FUNCTIONAL_COLUMNS:
                functional_pair_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "q_dim": q_dim,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "coordinate": coordinate,
                        "spearman": finite_spearman(
                            merged_coordinates[f"{coordinate}_a"].to_numpy(),
                            merged_coordinates[f"{coordinate}_b"].to_numpy(),
                            f"{coordinate}: {dataset}, {method}, seeds {seed_a}/{seed_b}",
                        ),
                        "entities": len(merged_coordinates),
                    }
                )

        for row in group.itertuples(index=False):
            correlations = pd.read_csv(row.correlation_path)
            correlations.insert(0, "seed", int(row.seed))
            correlations.insert(0, "q_dim", int(row.q_dim))
            correlations.insert(0, "method", row.method)
            correlations.insert(0, "dataset", row.dataset)
            descriptor_rows.append(correlations)

    q_pairs = pd.DataFrame(q_pair_rows).sort_values(group_columns + ["seed_a", "seed_b"])
    q_summary = summarize(q_pairs, group_columns, "distance_spearman")
    functional_pairs = pd.DataFrame(functional_pair_rows).sort_values(
        group_columns + ["coordinate", "seed_a", "seed_b"]
    )
    functional_summary = summarize(functional_pairs, group_columns + ["coordinate"], "spearman")

    descriptor_all = pd.concat(descriptor_rows, ignore_index=True)
    descriptor_groups = group_columns + ["functional_coordinate", "empirical_descriptor"]
    descriptor_summary = (
        descriptor_all.groupby(descriptor_groups, as_index=False)
        .agg(
            median_spearman=("spearman", "median"),
            min_spearman=("spearman", "min"),
            max_spearman=("spearman", "max"),
            positive_fraction=("spearman", lambda values: float((values > 0).mean())),
            abs_ge_0_5_fraction=("spearman", lambda values: float((values.abs() >= 0.5).mean())),
            seeds=("seed", "nunique"),
        )
        .sort_values(descriptor_groups)
    )

    q_across_splits = (
        q_summary.groupby(["method", "q_dim"], as_index=False)
        .agg(
            median_of_split_medians=("median_spearman", "median"),
            min_split_median=("median_spearman", "min"),
            max_split_median=("median_spearman", "max"),
            splits=("dataset", "nunique"),
        )
        .sort_values(["method", "q_dim"])
    )
    functional_across_splits = (
        functional_summary.groupby(["method", "q_dim", "coordinate"], as_index=False)
        .agg(
            median_of_split_medians=("median_spearman", "median"),
            min_split_median=("median_spearman", "min"),
            max_split_median=("median_spearman", "max"),
            all_split_medians_positive=("median_spearman", lambda values: bool((values > 0).all())),
            splits=("dataset", "nunique"),
        )
        .sort_values(["method", "q_dim", "coordinate"])
    )
    descriptor_across_splits = (
        descriptor_summary.groupby(
            ["method", "q_dim", "functional_coordinate", "empirical_descriptor"],
            as_index=False,
        )
        .agg(
            median_of_split_medians=("median_spearman", "median"),
            min_split_median=("median_spearman", "min"),
            max_split_median=("median_spearman", "max"),
            mean_positive_fraction=("positive_fraction", "mean"),
            mean_abs_ge_0_5_fraction=("abs_ge_0_5_fraction", "mean"),
            splits=("dataset", "nunique"),
        )
        .sort_values(["method", "q_dim", "functional_coordinate", "empirical_descriptor"])
    )

    outputs = {
        "cross_seed_q_distance_stability_pairs.csv": q_pairs,
        "cross_seed_q_distance_stability_summary.csv": q_summary,
        "cross_split_q_distance_stability_summary.csv": q_across_splits,
        "cross_seed_functional_stability_pairs.csv": functional_pairs,
        "cross_seed_functional_stability_summary.csv": functional_summary,
        "cross_split_functional_stability_summary.csv": functional_across_splits,
        "train_descriptor_correlations_all.csv": descriptor_all,
        "train_descriptor_correlation_summary.csv": descriptor_summary,
        "cross_split_train_descriptor_correlation_summary.csv": descriptor_across_splits,
    }
    for filename, frame in outputs.items():
        frame.to_csv(analysis_root / filename, index=False)
    print(json.dumps({filename: len(frame) for filename, frame in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
