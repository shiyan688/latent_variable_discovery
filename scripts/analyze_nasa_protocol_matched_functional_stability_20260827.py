#!/usr/bin/env python3
"""Re-evaluate saved NASA q candidates at an observed protocol per battery."""

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


Q_COLUMNS = ("q1", "q2", "q3", "q4")
CYCLES = (1.0, 10.0, 20.0, 28.0)
PROTOCOL_COLUMNS = ("ambient_temperature", "load_current_amp", "cutoff_voltage")
RESPONSE_COLUMNS = tuple(f"matched_response_cycle{int(cycle)}" for cycle in CYCLES)


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
    parser.add_argument(
        "--prepared-records",
        type=Path,
        default=PROJECT_ROOT
        / "data/real_datasets2/prepared/nasa_battery_reviewer_clean_20260825/inner_prepared_datasets.json",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    records = {
        record["name"]: record
        for record in json.loads(args.prepared_records.resolve().read_text())
    }
    device = torch.device(args.device)
    candidate_frames = []
    coverage_rows = []
    descriptor_rows = []

    paths = sorted(root.glob("**/cell_summary.json"))
    if len(paths) != 15:
        raise ValueError(f"expected 15 terminal cells, found {len(paths)}")
    for path in paths:
        summary = json.loads(path.read_text())
        dataset = summary["dataset"]
        seed = int(summary["seed"])
        source_result = PROJECT_ROOT / summary["source_result"]
        _, checkpoint, source, config = matched._load_source(source_result, device)
        train = pd.read_csv(matched._resolve(records[dataset]["train_csv"])).sort_values(
            ["label", "discharge_index"], kind="stable"
        )
        protocols = train.groupby("label", sort=False).first()[list(PROTOCOL_COLUMNS)]
        if seed == 0:
            fixed = (
                (train.ambient_temperature == 24.0)
                & (train.load_current_amp == 2.0)
                & (train.cutoff_voltage == 2.5)
            )
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "rows": int(len(train)),
                    "fixed_probe_exact_rows": int(fixed.sum()),
                    "fixed_probe_exact_fraction": float(fixed.mean()),
                    "labels": int(train.label.nunique()),
                    "labels_with_fixed_probe": int(train.loc[fixed, "label"].nunique()),
                }
            )
            for label, frame in train.groupby("label", sort=False):
                early = frame.loc[frame.discharge_index <= 10]
                slope, intercept = np.polyfit(
                    early.discharge_index, early.target, 1
                )
                descriptor_rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "empirical_capacity_cycle1": float(slope + intercept),
                        "empirical_early_fade_rate": float(-slope),
                    }
                )

        candidates = pd.read_csv(path.parent / "meta_q_candidates.csv")
        rows = []
        with torch.no_grad():
            for row in candidates.itertuples(index=False):
                protocol = protocols.loc[row.label].to_numpy(np.float32)
                features = np.asarray(
                    [[cycle, *protocol] for cycle in CYCLES], dtype=np.float32
                )
                normalized = (
                    features - source.normalizer.feature_mean
                ) / source.normalizer.feature_std
                q_value = torch.tensor(
                    [getattr(row, column) for column in Q_COLUMNS],
                    dtype=torch.float32,
                    device=device,
                )
                model_input = torch.cat(
                    [
                        torch.tensor(normalized, dtype=torch.float32, device=device),
                        q_value.unsqueeze(0).repeat(len(CYCLES), 1),
                    ],
                    dim=1,
                )
                prediction = source.model(model_input).squeeze(1).cpu().numpy()
                prediction = (
                    source.normalizer.target_mean
                    + source.normalizer.target_std * prediction
                )
                values = {
                    column: float(value)
                    for column, value in zip(RESPONSE_COLUMNS, prediction, strict=True)
                }
                rows.append(
                    {
                        **row._asdict(),
                        **dict(zip(PROTOCOL_COLUMNS, protocol, strict=True)),
                        **values,
                        "matched_capacity_cycle1": float(prediction[0]),
                        "matched_early_fade_rate": float(
                            (prediction[0] - prediction[1]) / 9.0
                        ),
                    }
                )
        candidate_frames.append(pd.DataFrame(rows).assign(dataset=dataset, seed=seed))

        del source
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidates = pd.concat(candidate_frames, ignore_index=True)
    functional_rows = []
    response_rows = []
    for (dataset, prior_weight), frame in candidates.groupby(
        ["dataset", "prior_weight"]
    ):
        by_seed = {int(seed): group for seed, group in frame.groupby("seed")}
        functional_values = {
            "matched_capacity_cycle1": [],
            "matched_early_fade_rate": [],
        }
        response_values = []
        for seed_a, seed_b in combinations(sorted(by_seed), 2):
            merged = by_seed[seed_a].merge(
                by_seed[seed_b],
                on="label",
                suffixes=("_a", "_b"),
                validate="one_to_one",
            ).sort_values("label")
            for coordinate in functional_values:
                functional_values[coordinate].append(
                    _spearman(
                        merged[f"{coordinate}_a"].to_numpy(float),
                        merged[f"{coordinate}_b"].to_numpy(float),
                    )
                )
            response_values.append(
                _spearman(
                    pdist(merged[[f"{column}_a" for column in RESPONSE_COLUMNS]]),
                    pdist(merged[[f"{column}_b" for column in RESPONSE_COLUMNS]]),
                )
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
        response_rows.append(
            {
                "dataset": dataset,
                "prior_weight": prior_weight,
                "median_spearman": float(np.median(response_values)),
                "min_pair_spearman": float(np.min(response_values)),
            }
        )

    functional = pd.DataFrame(functional_rows)
    response = pd.DataFrame(response_rows)
    coverage = pd.DataFrame(coverage_rows).sort_values("dataset")
    descriptors = pd.DataFrame(descriptor_rows)
    empirical_rows = []
    for (dataset, seed, prior_weight), frame in candidates.groupby(
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
                    merged.matched_capacity_cycle1.to_numpy(float),
                    merged.empirical_capacity_cycle1.to_numpy(float),
                ),
                "early_fade_spearman": _spearman(
                    merged.matched_early_fade_rate.to_numpy(float),
                    merged.empirical_early_fade_rate.to_numpy(float),
                ),
            }
        )
    empirical = pd.DataFrame(empirical_rows)
    summary = (
        functional.groupby(["prior_weight", "coordinate"], as_index=False)
        .agg(
            median_of_splits=("median_spearman", "median"),
            min_split=("min_pair_spearman", "min"),
        )
        .sort_values(["prior_weight", "coordinate"])
    )

    candidates.to_csv(root / "protocol_matched_candidates.csv", index=False)
    functional.to_csv(root / "protocol_matched_functional_stability.csv", index=False)
    response.to_csv(root / "protocol_matched_response_stability.csv", index=False)
    empirical.to_csv(
        root / "protocol_matched_empirical_correlations.csv", index=False
    )
    coverage.to_csv(root / "fixed_probe_protocol_coverage.csv", index=False)

    lines = [
        "# Protocol-matched functional-coordinate diagnostic",
        "",
        "This is a post-terminal diagnostic of saved q candidates; it does not change the frozen selection or authorize validation.",
        "For each battery, the probe uses the protocol of its first observed discharge and evaluates cycles 1, 10, 20, and 28.",
        "No target value is used to choose the protocol.",
        "",
        "## Fixed-probe support",
        "",
        *_table(
            ["inner split", "exact rows", "fraction", "labels covered"],
            [
                [
                    row.dataset,
                    f"{int(row.fixed_probe_exact_rows)}/{int(row.rows)}",
                    f"{row.fixed_probe_exact_fraction:.3f}",
                    f"{int(row.labels_with_fixed_probe)}/{int(row.labels)}",
                ]
                for row in coverage.itertuples(index=False)
            ],
        ),
        "",
        "## Cross-seed rank stability at observed protocols",
        "",
        *_table(
            ["weight", "coordinate", "median of splits", "worst seed-pair/split"],
            [
                [
                    f"{row.prior_weight:g}",
                    row.coordinate,
                    f"{row.median_of_splits:.3f}",
                    f"{row.min_split:.3f}",
                ]
                for row in summary.itertuples(index=False)
            ],
        ),
        "",
        "## Alignment with empirical early-curve descriptors",
        "",
        "These correlations are post-hoc development diagnostics and use meta-fit target values; they are not selection gates.",
        "",
        *_table(
            ["weight", "capacity Spearman", "early-fade Spearman"],
            [
                [
                    f"{prior_weight:g}",
                    f"{frame.capacity_spearman.median():.3f}",
                    f"{frame.early_fade_spearman.median():.3f}",
                ]
                for prior_weight, frame in empirical.groupby("prior_weight")
            ],
        ),
        "",
        "Interpretation must distinguish a failure of a fixed off-protocol coordinate from a failure of the learned response geometry.",
    ]
    (root / "PROTOCOL_MATCHED_FUNCTIONAL_DIAGNOSTIC.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
