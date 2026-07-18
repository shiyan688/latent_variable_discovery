#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "real_datasets2" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "real_datasets2" / "prepared"


@dataclass(frozen=True)
class PreparedDataset:
    name: str
    train_csv: Path
    test_csv: Path
    metadata_json: Path
    row_count: int
    label_count: int
    feature_columns: tuple[str, ...]
    target_column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real_datasets2 into the application latent-q CSV format."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uci_gas_drift", "cmapss_fd001", "cmapss_fd003", "nasa_battery"],
        choices=["uci_gas_drift", "cmapss_fd001", "cmapss_fd003", "nasa_battery"],
    )
    parser.add_argument("--test-label-ratio", type=float, default=0.25)
    parser.add_argument("--min-points-per-label", type=int, default=20)
    parser.add_argument("--max-labels", type=int, default=120)
    parser.add_argument(
        "--max-rows-per-label",
        type=int,
        default=30000,
        help="Subsample very dense labels before train/test writing. Set <=0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedDataset] = []
    for dataset in args.datasets:
        if dataset == "uci_gas_drift":
            prepared.append(prepare_uci_gas_drift(args))
        elif dataset.startswith("cmapss_"):
            subset = dataset.removeprefix("cmapss_").upper()
            prepared.append(prepare_cmapss_subset(args, subset))
        elif dataset == "nasa_battery":
            prepared.append(prepare_nasa_battery(args))

    summary = [
        {
            "name": item.name,
            "train_csv": str(item.train_csv),
            "test_csv": str(item.test_csv),
            "metadata_json": str(item.metadata_json),
            "row_count": item.row_count,
            "label_count": item.label_count,
            "feature_columns": list(item.feature_columns),
            "target_column": item.target_column,
        }
        for item in prepared
    ]
    summary_path = args.output_root / "prepared_datasets.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary: {summary_path}")


def prepare_uci_gas_drift(args: argparse.Namespace) -> PreparedDataset:
    raw_dir = args.raw_root / "uci_gas_sensor_array_drift_different_concentrations"
    if not raw_dir.exists():
        raise FileNotFoundError(f"UCI gas drift raw directory not found: {raw_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("batch*.dat"), key=_natural_sort_key):
        batch = path.stem
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                header = parts[0]
                if ";" not in header:
                    continue
                class_text, concentration_text = header.split(";", 1)
                try:
                    gas_class = float(class_text)
                    concentration = float(concentration_text)
                except ValueError:
                    continue
                for item in parts[1:]:
                    if ":" not in item:
                        continue
                    feature_text, value_text = item.split(":", 1)
                    try:
                        feature_id = int(feature_text)
                        response = float(value_text)
                    except ValueError:
                        continue
                    if not np.isfinite(response):
                        continue
                    rows.append(
                        {
                            "label": batch,
                            "gas_class": gas_class,
                            "log_concentration": float(np.log1p(max(concentration, 0.0))),
                            "sensor_id": float((feature_id - 1) // 8 + 1),
                            "sensor_feature_id": float((feature_id - 1) % 8 + 1),
                            "target": float(np.log1p(max(response, 0.0))),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame = _limit_labels_and_rows(
        frame,
        max_labels=args.max_labels,
        min_points_per_label=args.min_points_per_label,
        max_rows_per_label=args.max_rows_per_label,
        seed=args.seed,
    )
    return write_train_test(
        frame,
        args.output_root / "uci_gas_drift" / "sensor_response",
        name="uci_gas_drift_sensor_response",
        feature_columns=("gas_class", "log_concentration", "sensor_id", "sensor_feature_id"),
        target_column="target",
        source={
            "source": "UCI Gas Sensor Array Drift Dataset at Different Concentrations",
            "raw_dir": str(raw_dir),
            "target_transform": "log1p(sensor_feature_value)",
            "latent_label": "batch",
        },
        test_label_ratio=args.test_label_ratio,
        seed=args.seed,
    )


def prepare_cmapss_subset(args: argparse.Namespace, subset: str) -> PreparedDataset:
    raw_dir = args.raw_root / "nasa_cmapss_turbofan_degradation" / "CMAPSSData"
    path = raw_dir / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"C-MAPSS subset not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] < 26:
        raise ValueError(f"Expected at least 26 columns in {path}, got {frame.shape[1]}")
    columns = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{idx}" for idx in range(1, frame.shape[1] - 4)
    ]
    frame.columns = columns
    rows: list[dict[str, Any]] = []
    sensor_columns = [column for column in frame.columns if column.startswith("sensor_")]
    for _, row in frame.iterrows():
        label = f"{subset}_unit{int(row['unit']):03d}"
        for sensor_index, sensor_column in enumerate(sensor_columns, start=1):
            value = float(row[sensor_column])
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "label": label,
                    "cycle": float(row["cycle"]),
                    "op_setting_1": float(row["op_setting_1"]),
                    "op_setting_2": float(row["op_setting_2"]),
                    "op_setting_3": float(row["op_setting_3"]),
                    "sensor_index": float(sensor_index),
                    "target": value,
                }
            )
    long_frame = pd.DataFrame(rows)
    long_frame = _limit_labels_and_rows(
        long_frame,
        max_labels=args.max_labels,
        min_points_per_label=args.min_points_per_label,
        max_rows_per_label=args.max_rows_per_label,
        seed=args.seed,
    )
    return write_train_test(
        long_frame,
        args.output_root / "nasa_cmapss" / subset.lower() / "sensor_response",
        name=f"nasa_cmapss_{subset.lower()}_sensor_response",
        feature_columns=("cycle", "op_setting_1", "op_setting_2", "op_setting_3", "sensor_index"),
        target_column="target",
        source={
            "source": "NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set",
            "raw_file": str(path),
            "target": "sensor measurement value in long format",
            "latent_label": "engine unit",
        },
        test_label_ratio=args.test_label_ratio,
        seed=args.seed,
    )


def prepare_nasa_battery(args: argparse.Namespace) -> PreparedDataset:
    raw_dir = args.raw_root / "nasa_battery_data_set" / "extracted_batches"
    if not raw_dir.exists():
        raise FileNotFoundError(f"NASA battery extracted_batches directory not found: {raw_dir}")
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*/*.mat"), key=lambda item: str(item)):
        rows.extend(_read_nasa_battery_capacity_rows(path))
    frame = pd.DataFrame(rows)
    frame = _limit_labels_and_rows(
        frame,
        max_labels=args.max_labels,
        min_points_per_label=args.min_points_per_label,
        max_rows_per_label=args.max_rows_per_label,
        seed=args.seed,
    )
    return write_train_test(
        frame,
        args.output_root / "nasa_battery" / "capacity",
        name="nasa_battery_capacity",
        feature_columns=(
            "discharge_index",
            "ambient_temperature",
            "current_abs_mean",
            "voltage_min",
            "temperature_mean",
        ),
        target_column="target",
        source={
            "source": "NASA Li-ion Battery Aging Dataset",
            "raw_dir": str(raw_dir),
            "target": "discharge capacity",
            "latent_label": "battery id plus batch/protocol source",
        },
        test_label_ratio=args.test_label_ratio,
        seed=args.seed,
    )


def _read_nasa_battery_capacity_rows(path: Path) -> list[dict[str, Any]]:
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat if not key.startswith("__")]
    if not keys:
        return []
    battery_key = keys[0]
    battery = mat[battery_key]
    cycles = np.ravel(getattr(battery, "cycle", []))
    rows: list[dict[str, Any]] = []
    discharge_index = 0
    label = f"{path.parent.name}_{path.stem}"
    for absolute_index, cycle in enumerate(cycles, start=1):
        if str(getattr(cycle, "type", "")).lower() != "discharge":
            continue
        data = getattr(cycle, "data", None)
        if data is None or not hasattr(data, "Capacity"):
            continue
        capacity = _as_scalar(getattr(data, "Capacity", np.nan))
        if not np.isfinite(capacity):
            continue
        discharge_index += 1
        voltage = _as_float_array(getattr(data, "Voltage_measured", []))
        current = _as_float_array(getattr(data, "Current_measured", []))
        temperature = _as_float_array(getattr(data, "Temperature_measured", []))
        rows.append(
            {
                "label": label,
                "discharge_index": float(discharge_index),
                "ambient_temperature": _as_scalar(getattr(cycle, "ambient_temperature", np.nan)),
                "current_abs_mean": _safe_abs_mean(current),
                "voltage_min": _safe_min(voltage),
                "temperature_mean": _safe_mean(temperature),
                "target": float(capacity),
                "absolute_cycle_index": float(absolute_index),
            }
        )
    return rows


def write_train_test(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    name: str,
    feature_columns: tuple[str, ...],
    target_column: str,
    source: dict[str, Any],
    test_label_ratio: float,
    seed: int,
) -> PreparedDataset:
    if frame.empty:
        raise ValueError(f"No rows available for {name}.")
    required_columns = ("label", *feature_columns, target_column)
    frame = frame.loc[:, list(required_columns)].copy()
    for column in feature_columns + (target_column,):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(feature_columns) + [target_column])
    if frame.empty:
        raise ValueError(f"No finite rows available for {name}.")

    labels = np.array(sorted(frame["label"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    if labels.size <= 1:
        raise ValueError(f"{name} needs at least two labels for group split.")
    test_count = max(1, int(round(labels.size * test_label_ratio)))
    test_count = min(test_count, labels.size - 1)
    test_labels = set(labels[:test_count].tolist())

    train_frame = frame[~frame["label"].astype(str).isin(test_labels)].sample(frac=1.0, random_state=seed)
    test_frame = frame[frame["label"].astype(str).isin(test_labels)].sample(frac=1.0, random_state=seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    test_csv = output_dir / "test.csv"
    metadata_json = output_dir / "metadata.json"
    train_frame.to_csv(train_csv, index=False)
    test_frame.to_csv(test_csv, index=False)
    metadata = {
        "name": name,
        "source": source,
        "row_count": int(frame.shape[0]),
        "train_rows": int(train_frame.shape[0]),
        "test_rows": int(test_frame.shape[0]),
        "label_count": int(labels.size),
        "train_label_count": int(labels.size - test_count),
        "test_label_count": int(test_count),
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "split": "group holdout by label; test q is calibrated within held-out labels",
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PreparedDataset(
        name=name,
        train_csv=train_csv,
        test_csv=test_csv,
        metadata_json=metadata_json,
        row_count=int(frame.shape[0]),
        label_count=int(labels.size),
        feature_columns=feature_columns,
        target_column=target_column,
    )


def _limit_labels_and_rows(
    frame: pd.DataFrame,
    *,
    max_labels: int,
    min_points_per_label: int,
    max_rows_per_label: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    counts = frame["label"].astype(str).value_counts()
    labels = counts[counts >= min_points_per_label].index.to_numpy()
    rng = np.random.default_rng(seed)
    if max_labels > 0 and labels.size > max_labels:
        labels = labels[np.argsort([-counts[label] for label in labels])[:max_labels]]
    kept = frame[frame["label"].astype(str).isin(set(labels.tolist()))].copy()
    if max_rows_per_label <= 0:
        return kept
    sampled: list[pd.DataFrame] = []
    for _, group in kept.groupby("label", sort=False):
        if len(group) > max_rows_per_label:
            sampled.append(group.sample(n=max_rows_per_label, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            sampled.append(group)
    if not sampled:
        return kept.iloc[0:0].copy()
    return pd.concat(sampled, ignore_index=True)


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _as_float_array(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def _as_scalar(value: Any) -> float:
    array = _as_float_array(value)
    if array.size == 0:
        return float("nan")
    return float(array[0])


def _finite_values(array: np.ndarray) -> np.ndarray:
    return array[np.isfinite(array)]


def _safe_mean(array: np.ndarray) -> float:
    values = _finite_values(array)
    return float(values.mean()) if values.size else 0.0


def _safe_abs_mean(array: np.ndarray) -> float:
    values = _finite_values(array)
    return float(np.abs(values).mean()) if values.size else 0.0


def _safe_min(array: np.ndarray) -> float:
    values = _finite_values(array)
    return float(values.min()) if values.size else 0.0


if __name__ == "__main__":
    main()
