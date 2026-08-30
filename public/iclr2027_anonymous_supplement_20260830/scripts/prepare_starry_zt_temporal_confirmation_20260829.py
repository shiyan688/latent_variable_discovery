#!/usr/bin/env python3
"""Target-blind seal for post-development Starry ZT entities."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "runs/starry_zt_temporal_confirmation_20260829/selection"
PLAN = PROJECT_ROOT / "STARRY_ZT_TEMPORAL_CONFIRMATION_PLAN_20260829.md"
LATEST_ROOT = PROJECT_ROOT / "data/external/starrydata_latest_20260829"
LATEST_CURVES = LATEST_ROOT / "ThermoelectricMaterials_curves.csv.gz"
OLD_ZIP = PROJECT_ROOT / "data/application_reviewer_clean/starry_te/raw/starrydata2_2025-06-01.zip"
DEV_ROOT = PROJECT_ROOT / "data/application_reviewer_clean/starry_te/zt"
EXPECTED_LATEST_SHA = "b82fd98e8595b4c4712e3e21fe992320131826913bfd333c82011c921d9cb16a"
CUTOFF = pd.Timestamp("2025-06-01 03:00:01")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def sequence_length(value: object) -> int:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return int(np.asarray(parsed).reshape(-1).size)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"selection root already exists: {ROOT}")
    if sha256(LATEST_CURVES) != EXPECTED_LATEST_SHA or not PLAN.is_file():
        raise ValueError("frozen latest source and confirmation plan are required")

    columns = [
        "SID",
        "DOI",
        "composition",
        "sample_id",
        "figure_id",
        "prop_x",
        "prop_y",
        "unit_x",
        "unit_y",
        "x",
        "created_at",
        "updated_at",
    ]
    latest = pd.read_csv(LATEST_CURVES, usecols=columns)
    latest["source_row_index"] = np.arange(len(latest))
    strict = latest.loc[
        latest["prop_x"].astype(str).str.fullmatch("Temperature", case=False, na=False)
        & latest["prop_y"].astype(str).str.fullmatch("ZT", case=False, na=False)
        & latest["unit_x"].astype(str).str.fullmatch("K", case=False, na=False)
    ].copy()
    strict["created_timestamp"] = pd.to_datetime(
        strict["created_at"].astype(str).str.slice(0, 24),
        format="%a %b %d %Y %H:%M:%S",
        errors="raise",
    )
    strict["x_count"] = strict["x"].map(sequence_length)

    import zipfile

    with zipfile.ZipFile(OLD_ZIP) as archive:
        old = pd.read_csv(
            archive.open("starrydata_curves.csv"),
            usecols=["sample_id", "DOI", "composition"],
        )
    old_sample_ids = set(old["sample_id"].astype(str))
    development_labels = set(
        pd.concat(
            [pd.read_csv(DEV_ROOT / "train.csv"), pd.read_csv(DEV_ROOT / "test.csv")],
            ignore_index=True,
        )["label"].astype(str)
    )
    development_metadata = old.loc[old["sample_id"].astype(str).isin(development_labels)]
    development_dois = set(
        development_metadata["DOI"].dropna().astype(str).str.strip().str.lower()
    )
    development_compositions = set(
        development_metadata["composition"].dropna().map(normalized_text)
    )

    strict_counts = strict.groupby("sample_id")["sample_id"].transform("size")
    candidates = strict.loc[
        strict["created_timestamp"].gt(CUTOFF)
        & ~strict["sample_id"].astype(str).isin(old_sample_ids)
        & strict_counts.eq(1)
        & strict["x_count"].ge(20)
        & strict["DOI"].notna()
        & strict["composition"].notna()
    ].copy()
    candidates["doi_normalized"] = candidates["DOI"].astype(str).str.strip().str.lower()
    candidates["composition_normalized"] = candidates["composition"].map(normalized_text)
    candidates = candidates.loc[
        candidates["doi_normalized"].ne("")
        & candidates["composition_normalized"].ne("")
        & ~candidates["doi_normalized"].isin(development_dois)
        & ~candidates["composition_normalized"].isin(development_compositions)
    ].sort_values(["x_count", "sample_id"], ascending=[False, True], kind="stable")

    selected_rows = []
    selected_dois = set()
    selected_compositions = set()
    for row in candidates.itertuples(index=False):
        if row.doi_normalized in selected_dois or row.composition_normalized in selected_compositions:
            continue
        selected_rows.append(row._asdict())
        selected_dois.add(row.doi_normalized)
        selected_compositions.add(row.composition_normalized)
    selected = pd.DataFrame(selected_rows)
    if len(selected) < 20:
        raise ValueError("fewer than 20 independent temporal confirmation entities")
    if selected["DOI"].nunique() != len(selected) or selected["composition_normalized"].nunique() != len(selected):
        raise ValueError("confirmation DOI/composition uniqueness failed")

    ROOT.mkdir(parents=True, exist_ok=False)
    output_columns = [
        "source_row_index",
        "sample_id",
        "SID",
        "DOI",
        "composition",
        "figure_id",
        "unit_y",
        "created_at",
        "updated_at",
        "x_count",
        "x",
    ]
    selected[output_columns].to_csv(ROOT / "selected_entities_target_blind.csv", index=False)
    write_json(
        ROOT / "selection_decision.json",
        {
            "scope": "post-2025-06-01 Starry ZT target-blind temporal cohort",
            "selected_entities": len(selected),
            "selected_unique_dois": int(selected["DOI"].nunique()),
            "selected_unique_compositions": int(selected["composition_normalized"].nunique()),
            "minimum_x_count": int(selected["x_count"].min()),
            "median_x_count": float(selected["x_count"].median()),
            "maximum_x_count": int(selected["x_count"].max()),
            "latest_created_at": str(selected["created_timestamp"].max()),
            "target_column_opened": False,
            "target_statistics_computed": False,
            "selection_complete": True,
            "authorize_fixed_evaluation": True,
        },
    )
    write_json(
        ROOT / "manifest.json",
        {
            "plan_sha256": sha256(PLAN),
            "preparer_sha256": sha256(Path(__file__).resolve()),
            "latest_curves_sha256": sha256(LATEST_CURVES),
            "old_snapshot_sha256": sha256(OLD_ZIP),
            "development_train_sha256": sha256(DEV_ROOT / "train.csv"),
            "development_test_sha256": sha256(DEV_ROOT / "test.csv"),
            "target_column_opened": False,
            "files": {
                path.name: sha256(path)
                for path in ROOT.iterdir()
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    print((ROOT / "selection_decision.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
