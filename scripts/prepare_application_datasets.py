#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import pickle
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "application"

STARRY_DATE = "2025-06-01"
BATTERY_MATR_LINKS = [
    (
        "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",
        "MATR_batch_20170512.mat",
    ),
    (
        "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",
        "MATR_batch_20170630.mat",
    ),
    (
        "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",
        "MATR_batch_20180412.mat",
    ),
    (
        "https://data.matr.io/1/api/v1/file/5dcef152110002c7215b2c90/download",
        "MATR_batch_20190124.mat",
    ),
]
OC20_TUTORIAL_URL = "http://dl.fbaipublicfiles.com/opencatalystproject/data/tutorial_data.tar.gz"
ELEMENT_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
)
ELEMENT_COLUMNS = tuple(f"elem_{symbol}" for symbol in ELEMENT_SYMBOLS)
ELEMENT_ATOMIC_NUMBERS = {symbol: index + 1 for index, symbol in enumerate(ELEMENT_SYMBOLS)}
ELEMENT_PERIODS = {
    **{symbol: 1 for symbol in ("H", "He")},
    **{symbol: 2 for symbol in ("Li", "Be", "B", "C", "N", "O", "F", "Ne")},
    **{symbol: 3 for symbol in ("Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar")},
    **{symbol: 4 for symbol in ("K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr")},
    **{symbol: 5 for symbol in ("Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe")},
    **{symbol: 6 for symbol in ("Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn")},
}
ELEMENT_GROUPS = {
    "H": 1,
    "He": 18,
    "Li": 1,
    "Be": 2,
    "B": 13,
    "C": 14,
    "N": 15,
    "O": 16,
    "F": 17,
    "Ne": 18,
    "Na": 1,
    "Mg": 2,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 1,
    "Ca": 2,
    "Sc": 3,
    "Ti": 4,
    "V": 5,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
    "Cu": 11,
    "Zn": 12,
    "Ga": 13,
    "Ge": 14,
    "As": 15,
    "Se": 16,
    "Br": 17,
    "Kr": 18,
    "Rb": 1,
    "Sr": 2,
    "Y": 3,
    "Zr": 4,
    "Nb": 5,
    "Mo": 6,
    "Tc": 7,
    "Ru": 8,
    "Rh": 9,
    "Pd": 10,
    "Ag": 11,
    "Cd": 12,
    "In": 13,
    "Sn": 14,
    "Sb": 15,
    "Te": 16,
    "I": 17,
    "Xe": 18,
    "Cs": 1,
    "Ba": 2,
    "La": 3,
    "Ce": 3,
    "Pr": 3,
    "Nd": 3,
    "Pm": 3,
    "Sm": 3,
    "Eu": 3,
    "Gd": 3,
    "Tb": 3,
    "Dy": 3,
    "Ho": 3,
    "Er": 3,
    "Tm": 3,
    "Yb": 3,
    "Lu": 3,
    "Hf": 4,
    "Ta": 5,
    "W": 6,
    "Re": 7,
    "Os": 8,
    "Ir": 9,
    "Pt": 10,
    "Au": 11,
    "Hg": 12,
    "Tl": 13,
    "Pb": 14,
    "Bi": 15,
    "Po": 16,
    "At": 17,
    "Rn": 18,
}
COMPOSITION_DESCRIPTOR_COLUMNS = (
    "comp_n_elements",
    "comp_entropy",
    "comp_max_fraction",
    "comp_mean_z",
    "comp_std_z",
    "comp_min_z",
    "comp_max_z",
    "comp_mean_period",
    "comp_mean_group",
)
BATTERY_FEATURE_COLUMNS = (
    "cycle",
    "q_charge",
    "ir",
    "t_avg",
    "t_max",
    "t_min",
    "charge_time",
    "charge_c_rate",
    "charge_percent",
    "discharge_c_rate",
)
BATTERY_FEATURE_SETS = {
    "full": BATTERY_FEATURE_COLUMNS,
    "no_q_charge": tuple(column for column in BATTERY_FEATURE_COLUMNS if column != "q_charge"),
    "protocol": ("cycle", "charge_c_rate", "charge_percent", "discharge_c_rate"),
    "protocol_sensor": (
        "cycle",
        "ir",
        "t_avg",
        "t_max",
        "t_min",
        "charge_time",
        "charge_c_rate",
        "charge_percent",
        "discharge_c_rate",
    ),
}


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
    parser = argparse.ArgumentParser(description="Download and prepare application datasets for latent-q evaluation.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["starry_te", "battery_matr", "oc20_tutorial"],
        choices=["starry_te", "battery_matr", "oc20_tutorial"],
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--starry-date", default=STARRY_DATE)
    parser.add_argument("--battery-max-batches", type=int, default=1)
    parser.add_argument(
        "--battery-feature-set",
        default="full",
        choices=sorted(BATTERY_FEATURE_SETS),
        help="Battery feature policy. Use protocol for main no-leakage prospective runs.",
    )
    parser.add_argument(
        "--starry-feature-set",
        default="element_fractions",
        choices=["element_fractions", "composition_descriptors", "both"],
        help="Starry composition representation. composition_descriptors gives a compact reviewer-friendly baseline.",
    )
    parser.add_argument("--min-points-per-label", type=int, default=8)
    parser.add_argument("--max-labels", type=int, default=80)
    parser.add_argument("--test-label-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedDataset] = []
    for dataset_name in args.datasets:
        if dataset_name == "starry_te":
            prepared.extend(prepare_starry_te(args))
        elif dataset_name == "battery_matr":
            prepared.extend(prepare_battery_matr(args))
        elif dataset_name == "oc20_tutorial":
            prepared.extend(prepare_oc20_tutorial(args))

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


def prepare_starry_te(args: argparse.Namespace) -> list[PreparedDataset]:
    import starrydata

    root = args.output_root / "starry_te"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"starrydata2_{args.starry_date}.zip"

    if args.download and (args.force_download or not _valid_zip(zip_path)):
        ds = starrydata.load_dataset(date=args.starry_date)
        zip_path.write_bytes(ds.zip_data.getvalue())

    if not _valid_zip(zip_path):
        raise FileNotFoundError(
            f"StarryData zip not found or invalid: {zip_path}. Re-run with --download."
        )

    with zipfile.ZipFile(zip_path) as zf:
        curves = pd.read_csv(zf.open("starrydata_curves.csv"))
        samples = pd.read_csv(zf.open("starrydata_samples.csv"))

    prepared: list[PreparedDataset] = []
    feature_columns = _starry_feature_columns(args.starry_feature_set)
    for target_hint, slug in [
        ("Seebeck", "seebeck"),
        ("electrical conductivity", "electrical_conductivity"),
        ("thermal conductivity", "thermal_conductivity"),
        ("ZT", "zt"),
    ]:
        frame = _build_starry_curve_frame(curves, samples, target_hint)
        if frame.empty:
            continue
        frame = _limit_labels(frame, args.max_labels, args.min_points_per_label, args.seed)
        if frame.empty:
            continue
        prepared.append(
            write_train_test(
                frame,
                root / slug,
                name=f"starry_te_{slug}",
                feature_columns=feature_columns,
                target_column="target",
                source={
                    "source": "StarryData2",
                    "snapshot_date": args.starry_date,
                    "target_hint": target_hint,
                    "raw_zip": str(zip_path),
                },
                test_label_ratio=args.test_label_ratio,
                seed=args.seed,
            )
        )
    return prepared


def _build_starry_curve_frame(curves: pd.DataFrame, samples: pd.DataFrame, target_hint: str) -> pd.DataFrame:
    sample_id_col = _first_existing(curves.columns, ["sample_id", "Sample ID", "sampleid", "sample"])
    x_col = _first_existing(curves.columns, ["x", "x_value", "temperature"])
    y_col = _first_existing(curves.columns, ["y", "y_value", "value"])
    property_cols = [
        col
        for col in curves.columns
        if "property" in str(col).lower() or str(col).lower().startswith("prop_")
    ]
    if sample_id_col is None or x_col is None or y_col is None or not property_cols:
        return pd.DataFrame()

    mask = np.zeros(len(curves), dtype=bool)
    for col in property_cols:
        mask |= curves[col].astype(str).str.contains(target_hint, case=False, na=False).to_numpy()
    mask &= _temperature_mask(curves)
    selected = curves.loc[mask, [sample_id_col, x_col, y_col]].copy()
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        xs = _parse_numeric_sequence(row[x_col])
        ys = _parse_numeric_sequence(row[y_col])
        if xs.size == 0 or xs.size != ys.size:
            continue
        for temperature, target in zip(xs, ys):
            if np.isfinite(temperature) and np.isfinite(target):
                rows.append(
                    {
                        "sample_id": row[sample_id_col],
                        "temperature": float(temperature),
                        "target": float(target),
                    }
                )
    selected = pd.DataFrame(rows)
    if selected.empty:
        return pd.DataFrame()

    material_map = _starry_material_map(samples)
    if material_map:
        selected["composition_label"] = selected["sample_id"].map(material_map).fillna(selected["sample_id"].astype(str))
    else:
        selected["composition_label"] = selected["sample_id"].astype(str)
    selected["label"] = selected["sample_id"].astype(str)
    selected["label"] = selected["label"].astype(str)
    composition_features = selected["composition_label"].map(_composition_to_element_fraction)
    for column in ELEMENT_COLUMNS:
        selected[column] = [features.get(column, 0.0) for features in composition_features]
    descriptor_features = composition_features.map(_composition_descriptors_from_fractions)
    for column in COMPOSITION_DESCRIPTOR_COLUMNS:
        selected[column] = [features.get(column, 0.0) for features in descriptor_features]
    return selected.loc[:, ["label", "temperature", *ELEMENT_COLUMNS, *COMPOSITION_DESCRIPTOR_COLUMNS, "target"]]


def _starry_feature_columns(feature_set: str) -> tuple[str, ...]:
    if feature_set == "element_fractions":
        return ("temperature", *ELEMENT_COLUMNS)
    if feature_set == "composition_descriptors":
        return ("temperature", *COMPOSITION_DESCRIPTOR_COLUMNS)
    if feature_set == "both":
        return ("temperature", *ELEMENT_COLUMNS, *COMPOSITION_DESCRIPTOR_COLUMNS)
    raise ValueError(f"Unknown Starry feature set: {feature_set}")


def _temperature_mask(curves: pd.DataFrame) -> np.ndarray:
    candidates = [
        col
        for col in curves.columns
        if any(token in str(col).lower() for token in ["prop_x", "x_name", "x_label", "x_unit", "unit_x"])
    ]
    if not candidates:
        return np.ones(len(curves), dtype=bool)
    mask = np.zeros(len(curves), dtype=bool)
    for col in candidates:
        values = curves[col].astype(str)
        mask |= values.str.contains("temp|kelvin|\\bK\\b|temperature", case=False, regex=True, na=False).to_numpy()
    return mask


def _parse_numeric_sequence(value: Any) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _starry_material_map(samples: pd.DataFrame) -> dict[Any, str]:
    sample_id_col = _first_existing(samples.columns, ["sample_id", "Sample ID", "sampleid", "id"])
    label_col = _first_matching(samples.columns, ["composition", "chemical_formula", "formula", "material"])
    if sample_id_col is None or label_col is None:
        return {}
    labels = samples[[sample_id_col, label_col]].dropna()
    return dict(zip(labels[sample_id_col], labels[label_col].astype(str)))


def prepare_battery_matr(args: argparse.Namespace) -> list[PreparedDataset]:
    root = args.output_root / "battery_matr"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        for url, filename in BATTERY_MATR_LINKS[: args.battery_max_batches]:
            download_file(url, raw_dir / filename, force=args.force_download)

    mat_paths = sorted(raw_dir.glob("MATR_batch_*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No MATR .mat files found in {raw_dir}. Re-run with --download.")

    frames = [item for item in (_read_matr_capacity_curves(path) for path in mat_paths) if not item.empty]
    if not frames:
        return []
    frame = pd.concat(frames, ignore_index=True)
    frame = _limit_labels(frame, args.max_labels, args.min_points_per_label, args.seed)
    if frame.empty:
        return []
    return [
        write_train_test(
            frame,
            root / "capacity",
            name=f"battery_matr_capacity_{args.battery_feature_set}",
            feature_columns=BATTERY_FEATURE_SETS[args.battery_feature_set],
            target_column="target",
            source={
                "source": "MIT/Stanford/Toyota MATR battery cycling",
                "raw_files": [str(path) for path in mat_paths],
                "feature_set": args.battery_feature_set,
            },
            test_label_ratio=args.test_label_ratio,
            seed=args.seed,
        )
    ]


def _read_matr_capacity_curves(path: Path) -> pd.DataFrame:
    from scipy.io import loadmat

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return _read_matr_capacity_curves_hdf5(path)

    batch = mat.get("batch")
    if batch is None:
        return pd.DataFrame()
    cells = np.ravel(batch)
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        summary = getattr(cell, "summary", None)
        if summary is None:
            continue
        q_discharge = _as_float_array(_get_field(summary, ["QDischarge", "QD", "qDischarge"]))
        if q_discharge.size == 0:
            continue
        cycles = _as_float_array(_get_field(summary, ["cycle", "cycles"]))
        if cycles.size != q_discharge.size:
            cycles = np.arange(1, q_discharge.size + 1, dtype=float)
        barcode = _get_field(cell, ["barcode", "policy_readable", "policy"])
        label = f"{path.stem}_cell{cell_index:03d}"
        if barcode is not None:
            label = f"{label}_{str(np.ravel(barcode)[0])[:40]}"
        q_charge = _aligned_summary_array(_get_field(summary, ["QCharge", "QC", "qCharge"]), q_discharge.size)
        ir = _aligned_summary_array(_get_field(summary, ["IR", "internalResistance"]), q_discharge.size)
        t_avg = _aligned_summary_array(_get_field(summary, ["Tavg", "Tmean"]), q_discharge.size)
        t_max = _aligned_summary_array(_get_field(summary, ["Tmax"]), q_discharge.size)
        t_min = _aligned_summary_array(_get_field(summary, ["Tmin"]), q_discharge.size)
        charge_time = _aligned_summary_array(_get_field(summary, ["chargetime", "chargeTime"]), q_discharge.size)
        policy_features = _parse_policy_features(str(np.ravel(barcode)[0]) if barcode is not None else "")
        for idx, (cycle, capacity) in enumerate(zip(cycles, q_discharge)):
            if np.isfinite(cycle) and np.isfinite(capacity):
                if cycle <= 1 and np.isclose(capacity, 0.0):
                    continue
                rows.append(
                    {
                        "label": label,
                        "cycle": float(cycle),
                        "q_charge": float(q_charge[idx]),
                        "ir": float(ir[idx]),
                        "t_avg": float(t_avg[idx]),
                        "t_max": float(t_max[idx]),
                        "t_min": float(t_min[idx]),
                        "charge_time": float(charge_time[idx]),
                        **policy_features,
                        "target": float(capacity),
                    }
                )
    return pd.DataFrame(rows)


def _read_matr_capacity_curves_hdf5(path: Path) -> pd.DataFrame:
    import h5py

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        batch = h5.get("batch")
        if batch is None or "summary" not in batch:
            return pd.DataFrame()
        summaries = batch["summary"]
        policy_readable = batch.get("policy_readable")
        for cell_index in range(summaries.shape[0]):
            summary_group = h5[summaries[cell_index, 0]]
            q_discharge = _as_float_array(summary_group.get("QDischarge"))
            cycles = _as_float_array(summary_group.get("cycle"))
            if q_discharge.size == 0:
                continue
            if cycles.size != q_discharge.size:
                cycles = np.arange(1, q_discharge.size + 1, dtype=float)
            policy_text = ""
            if policy_readable is not None:
                try:
                    policy_text = _decode_hdf5_string(h5[policy_readable[cell_index, 0]])
                except Exception:
                    policy_text = ""
            label = f"{path.stem}_cell{cell_index:03d}"
            if policy_text:
                label = f"{label}_{policy_text}"
            q_charge = _aligned_summary_array(summary_group.get("QCharge"), q_discharge.size)
            ir = _aligned_summary_array(summary_group.get("IR"), q_discharge.size)
            t_avg = _aligned_summary_array(summary_group.get("Tavg"), q_discharge.size)
            t_max = _aligned_summary_array(summary_group.get("Tmax"), q_discharge.size)
            t_min = _aligned_summary_array(summary_group.get("Tmin"), q_discharge.size)
            charge_time = _aligned_summary_array(summary_group.get("chargetime"), q_discharge.size)
            policy_features = _parse_policy_features(policy_text)
            for idx, (cycle, capacity) in enumerate(zip(cycles, q_discharge)):
                if np.isfinite(cycle) and np.isfinite(capacity):
                    if cycle <= 1 and np.isclose(capacity, 0.0):
                        continue
                    rows.append(
                        {
                            "label": label,
                            "cycle": float(cycle),
                            "q_charge": float(q_charge[idx]),
                            "ir": float(ir[idx]),
                            "t_avg": float(t_avg[idx]),
                            "t_max": float(t_max[idx]),
                            "t_min": float(t_min[idx]),
                            "charge_time": float(charge_time[idx]),
                            **policy_features,
                            "target": float(capacity),
                        }
                    )
    return pd.DataFrame(rows)


def _decode_hdf5_string(dataset: Any) -> str:
    values = np.asarray(dataset).reshape(-1)
    if values.dtype.kind in {"u", "i"}:
        chars = [chr(int(value)) for value in values if int(value) != 0]
        return "".join(chars)
    if values.dtype.kind == "S":
        return b"".join(values.tolist()).decode("utf-8", errors="ignore")
    return str(values[0]) if values.size else ""


def _aligned_summary_array(value: Any, expected_size: int) -> np.ndarray:
    array = _as_float_array(value)
    if array.size == expected_size:
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.zeros(expected_size, dtype=float)


def _parse_policy_features(policy_text: str) -> dict[str, float]:
    import re

    numbers = [float(match) for match in re.findall(r"[-+]?\d+(?:\.\d+)?", policy_text)]
    return {
        "charge_c_rate": numbers[0] if len(numbers) >= 1 else 0.0,
        "charge_percent": numbers[1] if len(numbers) >= 2 else 0.0,
        "discharge_c_rate": numbers[2] if len(numbers) >= 3 else 0.0,
    }


def _composition_to_element_fraction(composition: Any) -> dict[str, float]:
    import re

    amounts: dict[str, float] = {}
    for symbol, amount_text in re.findall(r"([A-Z][a-z]?)([-+]?\d*(?:\.\d+)?)", str(composition)):
        if symbol not in ELEMENT_SYMBOLS:
            continue
        amount = float(amount_text) if amount_text and amount_text not in {"+", "-"} else 1.0
        amounts[symbol] = amounts.get(symbol, 0.0) + amount
    total = sum(amounts.values())
    if total <= 0:
        return {}
    return {f"elem_{symbol}": amount / total for symbol, amount in amounts.items()}


def _composition_descriptors_from_fractions(fractions: dict[str, float]) -> dict[str, float]:
    entries: list[tuple[str, float]] = []
    for column, fraction in fractions.items():
        symbol = column.removeprefix("elem_")
        if symbol in ELEMENT_ATOMIC_NUMBERS and fraction > 0:
            entries.append((symbol, float(fraction)))
    if not entries:
        return {column: 0.0 for column in COMPOSITION_DESCRIPTOR_COLUMNS}

    fraction_values = np.array([fraction for _, fraction in entries], dtype=float)
    z_values = np.array([ELEMENT_ATOMIC_NUMBERS[symbol] for symbol, _ in entries], dtype=float)
    period_values = np.array([ELEMENT_PERIODS[symbol] for symbol, _ in entries], dtype=float)
    group_values = np.array([ELEMENT_GROUPS[symbol] for symbol, _ in entries], dtype=float)
    mean_z = float(np.sum(fraction_values * z_values))
    var_z = float(np.sum(fraction_values * (z_values - mean_z) ** 2))
    entropy = float(-np.sum(fraction_values * np.log(np.clip(fraction_values, 1e-12, 1.0))))
    return {
        "comp_n_elements": float(len(entries)),
        "comp_entropy": entropy,
        "comp_max_fraction": float(np.max(fraction_values)),
        "comp_mean_z": mean_z,
        "comp_std_z": float(np.sqrt(max(var_z, 0.0))),
        "comp_min_z": float(np.min(z_values)),
        "comp_max_z": float(np.max(z_values)),
        "comp_mean_period": float(np.sum(fraction_values * period_values)),
        "comp_mean_group": float(np.sum(fraction_values * group_values)),
    }


def prepare_oc20_tutorial(args: argparse.Namespace) -> list[PreparedDataset]:
    root = args.output_root / "oc20_tutorial"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "tutorial_data.tar.gz"
    if args.download:
        download_file(OC20_TUTORIAL_URL, archive_path, force=args.force_download)
        extract_dir = raw_dir / "tutorial_data"
        if args.force_download or not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path) as tar:
                tar.extractall(extract_dir)

    lmdb_path = raw_dir / "tutorial_data" / "s2ef" / "train_100" / "data.lmdb"
    if not lmdb_path.exists():
        return []
    frame = _read_oc20_lmdb_energy_curves(lmdb_path)
    frame = _limit_labels(frame, args.max_labels, args.min_points_per_label, args.seed)
    if frame.empty:
        return []
    return [
        write_train_test(
            frame,
            root / "s2ef_energy",
            name="oc20_tutorial_s2ef_energy",
            feature_columns=("frame", "natoms"),
            target_column="target",
            source={
                "source": "OC20 tutorial S2EF LMDB",
                "raw_lmdb": str(lmdb_path),
            },
            test_label_ratio=args.test_label_ratio,
            seed=args.seed,
        )
    ]


def _read_oc20_lmdb_energy_curves(lmdb_path: Path) -> pd.DataFrame:
    import lmdb
    import torch

    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    rows: list[dict[str, Any]] = []
    with env.begin() as txn:
        raw_length = txn.get(b"length")
        if raw_length is None:
            return pd.DataFrame()
        length = int(pickle.loads(raw_length))
        for index in range(length):
            raw = txn.get(str(index).encode("ascii"))
            if raw is None:
                continue
            try:
                item = pickle.loads(raw)
            except ModuleNotFoundError:
                continue
            store = getattr(item, "__dict__", {})
            sid = _scalar(store.get("sid", getattr(item, "sid", index)))
            fid = _scalar(store.get("fid", getattr(item, "fid", index)))
            y = store.get("y", None)
            if y is None:
                y = getattr(item, "y", None)
            if y is None:
                continue
            energy = float(torch.as_tensor(y).reshape(-1)[0].detach().cpu().item())
            atomic_numbers = store.get("atomic_numbers", getattr(item, "atomic_numbers", []))
            natoms_value = store.get("natoms", None)
            if natoms_value is None:
                natoms = int(torch.as_tensor(atomic_numbers).numel())
            else:
                natoms = int(torch.as_tensor(natoms_value).reshape(-1)[0].detach().cpu().item())
            rows.append({"label": str(sid), "frame": float(fid), "natoms": float(natoms), "target": energy})
    env.close()
    return pd.DataFrame(rows)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = split_by_label(frame, test_label_ratio=test_label_ratio, seed=seed)
    ordered_columns = ["label", *feature_columns, target_column]
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.loc[:, ordered_columns].to_csv(train_path, index=False)
    test.loc[:, ordered_columns].to_csv(test_path, index=False)
    metadata = {
        "name": name,
        "source": source,
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "feature_columns": list(feature_columns),
        "label_column": "label",
        "target_column": target_column,
        "row_count": int(frame.shape[0]),
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "label_count": int(frame["label"].nunique()),
        "min_points_per_label": int(frame.groupby("label").size().min()),
        "max_points_per_label": int(frame.groupby("label").size().max()),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PreparedDataset(
        name=name,
        train_csv=train_path,
        test_csv=test_path,
        metadata_json=metadata_path,
        row_count=int(frame.shape[0]),
        label_count=int(frame["label"].nunique()),
        feature_columns=feature_columns,
        target_column=target_column,
    )


def split_by_label(frame: pd.DataFrame, *, test_label_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = np.array(sorted(frame["label"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    test_count = max(1, int(round(len(labels) * test_label_ratio)))
    test_labels = set(labels[:test_count])
    is_test = frame["label"].isin(test_labels)
    return frame.loc[~is_test].sort_values(["label"]).copy(), frame.loc[is_test].sort_values(["label"]).copy()


def _limit_labels(frame: pd.DataFrame, max_labels: int, min_points: int, seed: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    counts = frame.groupby("label").size()
    labels = counts[counts >= min_points].sort_values(ascending=False).index.to_numpy()
    if labels.size == 0:
        return pd.DataFrame(columns=frame.columns)
    labels = labels[: max_labels * 4]
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    keep = set(labels[:max_labels])
    return frame.loc[frame["label"].isin(keep)].copy()


def download_file(url: str, path: Path, *, force: bool) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "latent-variable-search/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def _valid_zip(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def _first_existing(columns: Iterable[Any], candidates: Iterable[str]) -> str | None:
    column_lookup = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]
    return None


def _first_matching(columns: Iterable[Any], tokens: Iterable[str]) -> str | None:
    for column in columns:
        lowered = str(column).lower()
        if any(token.lower() in lowered for token in tokens):
            return str(column)
    return None


def _get_field(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    return None


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value).reshape(-1)
    return array[0].item() if array.size else value


if __name__ == "__main__":
    main()
