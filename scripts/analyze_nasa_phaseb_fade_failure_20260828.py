#!/usr/bin/env python3
"""Post-terminal diagnosis of the NASA Phase-B early-fade failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared-records", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    q_values = pd.read_csv(root / "all_validation_q.csv").query(
        "prior_weight == 0.01"
    )
    rows = []
    for dataset, frame in q_values.groupby("dataset"):
        fade_seed_sd = float(frame.groupby("label").early_fade_rate.std().median())
        fade_entity_spread = float(
            np.median(
                [group.early_fade_rate.std(ddof=0) for _, group in frame.groupby("seed")]
            )
        )
        capacity_seed_sd = float(
            frame.groupby("label").capacity_cycle1.std().median()
        )
        capacity_entity_spread = float(
            np.median(
                [group.capacity_cycle1.std(ddof=0) for _, group in frame.groupby("seed")]
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "fade_seed_sd": fade_seed_sd,
                "fade_entity_spread": fade_entity_spread,
                "fade_noise_to_signal": fade_seed_sd / fade_entity_spread,
                "capacity_seed_sd": capacity_seed_sd,
                "capacity_entity_spread": capacity_entity_spread,
                "capacity_noise_to_signal": capacity_seed_sd / capacity_entity_spread,
            }
        )
    noise = pd.DataFrame(rows).sort_values("dataset")

    records = json.loads(args.prepared_records.resolve().read_text())
    segment_rows = []
    for record in records:
        validation = pd.read_csv(record["test_csv"]).sort_values(
            ["label", "discharge_index"], kind="stable"
        )
        for label, frame in validation.groupby("label", sort=False):
            early = frame.loc[frame.discharge_index <= 10]
            first_target = float(early.iloc[0].target)
            following_median = float(early.iloc[1:].target.median())
            protocols_to_28 = int(
                frame.loc[frame.discharge_index <= 28,
                          ["ambient_temperature", "load_current_amp", "cutoff_voltage"]]
                .drop_duplicates()
                .shape[0]
            )
            segment_rows.append(
                {
                    "dataset": record["name"],
                    "label": label,
                    "first_discharge_index": float(early.iloc[0].discharge_index),
                    "first_target": first_target,
                    "following_early_median": following_median,
                    "first_to_following_ratio": first_target / following_median,
                    "protocols_through_cycle28": protocols_to_28,
                }
            )
    segments = pd.DataFrame(segment_rows).sort_values(["dataset", "label"])
    flagged = segments.loc[
        (segments.first_to_following_ratio < 0.8)
        | (segments.protocols_through_cycle28 > 1)
    ]

    ranked = q_values.copy()
    ranked["fade_rank"] = ranked.groupby(["dataset", "seed"])[
        "early_fade_rate"
    ].rank(method="average")
    rank_instability = (
        ranked.groupby(["dataset", "label"], as_index=False)
        .agg(
            fade_median=("early_fade_rate", "median"),
            fade_min=("early_fade_rate", "min"),
            fade_max=("early_fade_rate", "max"),
            fade_rank_std=("fade_rank", "std"),
        )
        .sort_values(["dataset", "fade_rank_std"], ascending=[True, False])
    )

    noise.to_csv(root / "fade_noise_summary.csv", index=False)
    segments.to_csv(root / "early_segment_audit.csv", index=False)
    rank_instability.to_csv(root / "fade_rank_instability.csv", index=False)
    lines = [
        "# Phase-B early-fade failure diagnostic",
        "",
        "This is a post-terminal diagnostic. It does not change the frozen Phase-B STOP decision or authorize Stage C2.",
        "",
        "The selected prior retained prediction, capacity, empirical alignment, and the full four-response geometry. The only failed frozen gate is early-fade cross-seed stability on inner2 (split median 0.20); weight 0 has the same 0.20, so the failure is not introduced by weight 0.01.",
        "",
        "## Difference-coordinate noise",
        "",
        *_table(
            ["split", "fade seed SD / entity spread", "capacity seed SD / entity spread"],
            [
                [
                    row.dataset,
                    f"{row.fade_noise_to_signal:.3f}",
                    f"{row.capacity_noise_to_signal:.3f}",
                ]
                for row in noise.itertuples(index=False)
            ],
        ),
        "",
        "Early fade is a small difference between two large decoder outputs. Across splits, seed variation is 36--58% of between-battery fade spread, versus only 3--4% for capacity. Thus stable four-point response distances can coexist with an unstable derivative-like rank.",
        "",
        "## Input-series conditions that violate a simple early-degradation interpretation",
        "",
        *_table(
            ["split", "battery", "first/following capacity", "protocols through cycle 28"],
            [
                [
                    row.dataset,
                    row.label,
                    f"{row.first_to_following_ratio:.3f}",
                    str(int(row.protocols_through_cycle28)),
                ]
                for row in flagged.itertuples(index=False)
            ],
        ),
        "",
        "B0036, B0039, and B0033 begin with large recovery/activation transients rather than monotone fade; B0039/B0040 also change operating protocol within 28 cycles. A cycle-1-to-10 slope is therefore not uniformly the same physical estimand across these batteries.",
        "",
        "## Mechanistic consequence",
        "",
        "Rank 2 intentionally leaves both dominant capacity and response-shape directions unpenalized. It preserves legitimate response geometry but cannot canonicalize a noisy fade direction inside that retained subspace. The next justified screen is rank 1 at protocol-matched probes: preserve the dominant capacity direction and softly regularize residual response shape. This must be treated as sequential development and later confirmed on new batteries.",
    ]
    (root / "FADE_FAILURE_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
