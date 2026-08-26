#!/usr/bin/env python3
"""Prepare a leakage-free NASA battery cohort for the real q closed loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "real_datasets2"
    / "raw"
    / "nasa_battery_data_set"
    / "extracted_batches"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "real_datasets2"
    / "prepared"
    / "nasa_battery_reviewer_clean_20260825"
)

FAMILIES = {
    **{battery: "classic_constant_2a" for battery in ("B0005", "B0006", "B0007", "B0018")},
    **{battery: "room_square_wave" for battery in ("B0025", "B0026", "B0027", "B0028")},
    **{battery: "hot_constant_4a" for battery in ("B0029", "B0030", "B0031", "B0032")},
    **{battery: "room_constant_current" for battery in ("B0033", "B0034", "B0036")},
    **{battery: "multi_temperature_current" for battery in ("B0038", "B0039", "B0040")},
}
CUTOFF_VOLTAGE = {
    "B0005": 2.7,
    "B0006": 2.5,
    "B0007": 2.2,
    "B0018": 2.5,
    "B0025": 2.0,
    "B0026": 2.2,
    "B0027": 2.5,
    "B0028": 2.7,
    "B0029": 2.0,
    "B0030": 2.2,
    "B0031": 2.5,
    "B0032": 2.7,
    "B0033": 2.0,
    "B0034": 2.2,
    "B0036": 2.7,
    "B0038": 2.2,
    "B0039": 2.5,
    "B0040": 2.7,
}
FEATURE_COLUMNS = (
    "discharge_index",
    "ambient_temperature",
    "load_current_amp",
    "cutoff_voltage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def read_battery(path: Path) -> pd.DataFrame:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    battery = payload[[key for key in payload if not key.startswith("__")][0]]
    rows = []
    discharge_index = 0
    for cycle in np.ravel(battery.cycle):
        if str(cycle.type).lower() != "discharge":
            continue
        discharge_index += 1
        current = np.abs(np.asarray(cycle.data.Current_measured, dtype=float).reshape(-1))
        current = current[np.isfinite(current)]
        measured_amplitude = float(np.quantile(current, 0.9))
        nominal_amplitude = float(
            np.asarray((1.0, 2.0, 4.0))[np.argmin(np.abs(np.asarray((1.0, 2.0, 4.0)) - measured_amplitude))]
        )
        rows.append(
            {
                "label": path.stem,
                "protocol_family": FAMILIES[path.stem],
                "discharge_index": float(discharge_index),
                "ambient_temperature": float(np.asarray(cycle.ambient_temperature).reshape(-1)[0]),
                "load_current_amp": nominal_amplitude,
                "cutoff_voltage": CUTOFF_VOLTAGE[path.stem],
                "target": float(np.asarray(cycle.data.Capacity).reshape(-1)[0]),
                "measured_current_q90": measured_amplitude,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    by_battery: dict[str, pd.DataFrame] = {}
    duplicate_paths: dict[str, list[str]] = {}
    for path in sorted(args.raw_root.glob("*/*.mat")):
        if path.stem not in FAMILIES:
            continue
        frame = read_battery(path)
        if path.stem in by_battery:
            if not frame.equals(by_battery[path.stem]):
                raise ValueError(f"Conflicting duplicate files for {path.stem}")
            duplicate_paths[path.stem].append(str(path))
            continue
        by_battery[path.stem] = frame
        duplicate_paths[path.stem] = [str(path)]

    frame = pd.concat(by_battery.values(), ignore_index=True)
    invalid = (~np.isfinite(frame[[*FEATURE_COLUMNS, "target", "measured_current_q90"]])).any(axis=1)
    invalid |= frame["target"] <= 0
    invalid |= frame["measured_current_q90"] < 0.5
    excluded_rows = frame.loc[invalid].copy()
    frame = frame.loc[~invalid].copy()

    rng = np.random.default_rng(args.seed)
    test_labels = []
    for family in sorted(frame["protocol_family"].unique()):
        labels = np.asarray(sorted(frame.loc[frame.protocol_family == family, "label"].unique()))
        test_labels.append(str(rng.choice(labels)))
    test_label_set = set(test_labels)
    train = frame.loc[~frame.label.isin(test_label_set)].sort_values(
        ["label", "discharge_index"], kind="stable"
    )
    test = frame.loc[frame.label.isin(test_label_set)].sort_values(
        ["label", "discharge_index"], kind="stable"
    )

    if set(train.label) & set(test.label):
        raise RuntimeError("Battery identity leaked across the outer split")
    if train.label.nunique() != 13 or test.label.nunique() != 5:
        raise RuntimeError("Expected 13 train and 5 test batteries")
    for _, group in frame.groupby("label"):
        if not group.discharge_index.is_monotonic_increasing:
            raise RuntimeError("Discharge index is not monotonic within a battery")

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_columns = ["label", "protocol_family", *FEATURE_COLUMNS, "target"]
    train_path = args.output_root / "train.csv"
    test_path = args.output_root / "test.csv"
    metadata_path = args.output_root / "metadata.json"
    summary_path = args.output_root / "prepared_datasets.json"
    inner_summary_path = args.output_root / "inner_prepared_datasets.json"
    audit_path = args.output_root / "qualification_audit.json"
    train.loc[:, output_columns].to_csv(train_path, index=False)
    test.loc[:, output_columns].to_csv(test_path, index=False)

    metadata = {
        "name": "nasa_battery_capacity_reviewer_clean",
        "source": "NASA Li-ion Battery Aging Dataset; local source README files",
        "cohort_rule": (
            "Unique battery IDs B0005--B0040 whose source README does not warn that very low "
            "capacity values are unexplained"
        ),
        "excluded_later_batches": "B0041--B0056; source README explicitly flags unexplained very-low-capacity runs",
        "row_filter": "finite positive capacity and measured discharge-current 90th percentile >= 0.5 A",
        "feature_availability": {
            "discharge_index": "known before query",
            "ambient_temperature": "experimental condition known before query",
            "load_current_amp": "nominal experimental setpoint {1,2,4} A inferred from measured-current q90",
            "cutoff_voltage": "battery protocol condition documented in source README",
        },
        "forbidden_same_cycle_features": ["voltage_min", "temperature_mean", "current_abs_mean"],
        "split": "one battery per documented protocol family held out; battery ID is the entity key",
        "split_seed": args.seed,
        "train_labels": sorted(train.label.unique().tolist()),
        "test_labels": sorted(test.label.unique().tolist()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_label_count": int(train.label.nunique()),
        "test_label_count": int(test.label.nunique()),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "target",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = [
        {
            "name": metadata["name"],
            "train_csv": str(train_path.resolve()),
            "test_csv": str(test_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
            "row_count": int(len(frame)),
            "label_count": int(frame.label.nunique()),
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": "target",
        }
    ]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    inner_records = []
    inner_splits = []
    outer_train = train.copy()
    for split_index in range(3):
        validation_labels = []
        for family_index, family in enumerate(sorted(outer_train.protocol_family.unique())):
            labels = sorted(
                outer_train.loc[outer_train.protocol_family == family, "label"].unique()
            )
            validation_labels.append(labels[(split_index + family_index) % len(labels)])
        validation_set = set(validation_labels)
        inner_train = outer_train.loc[~outer_train.label.isin(validation_set)]
        inner_test = outer_train.loc[outer_train.label.isin(validation_set)]
        inner_dir = args.output_root / f"inner_split{split_index}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        inner_train_path = inner_dir / "train.csv"
        inner_test_path = inner_dir / "test.csv"
        inner_train.loc[:, output_columns].to_csv(inner_train_path, index=False)
        inner_test.loc[:, output_columns].to_csv(inner_test_path, index=False)
        inner_records.append(
            {
                "name": f"nasa_battery_capacity_reviewer_clean_inner{split_index}",
                "train_csv": str(inner_train_path.resolve()),
                "test_csv": str(inner_test_path.resolve()),
                "metadata_json": str(metadata_path.resolve()),
                "row_count": int(len(outer_train)),
                "label_count": int(outer_train.label.nunique()),
                "feature_columns": list(FEATURE_COLUMNS),
                "target_column": "target",
            }
        )
        inner_splits.append(
            {
                "split": split_index,
                "meta_train_labels": sorted(inner_train.label.unique().tolist()),
                "structure_validation_labels": sorted(validation_labels),
            }
        )
    inner_summary_path.write_text(json.dumps(inner_records, indent=2), encoding="utf-8")
    audit = {
        "duplicate_battery_ids": {
            label: paths for label, paths in duplicate_paths.items() if len(paths) > 1
        },
        "duplicates_verified_exact": True,
        "excluded_rows": excluded_rows.loc[
            :, ["label", "discharge_index", "target", "measured_current_q90"]
        ].to_dict(orient="records"),
        "outer_identity_overlap": sorted(set(train.label) & set(test.label)),
        "target_nonpositive_after_filter": int((frame.target <= 0).sum()),
        "rows": int(len(frame)),
        "labels": int(frame.label.nunique()),
        "inner_splits": inner_splits,
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
