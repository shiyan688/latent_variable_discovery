#!/usr/bin/env python3
"""Materialize only the sealed ThermoML development vapor-pressure targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN = PROJECT_ROOT / "THERMOML_VAPOR_PRESSURE_PLAN_20260829.md"
DATA_ROOT = PROJECT_ROOT / "data/external/thermoml_2020_archive/extracted"
COHORT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_cohorts_20260829"
SELECTION_MANIFEST = COHORT_ROOT / "selection_manifest.json"
DEVELOPMENT_SELECTION = COHORT_ROOT / "development_selection.csv"
CONFIRMATION_SELECTION = COHORT_ROOT / "confirmation_selection.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_development_data_20260829"
EXPECTED_PLAN_SHA256 = "8793f712b6a32aa514906ffb13ae7169d0de8556f9bda342b1202d94b0bb2deb"
EXPECTED_SELECTION_MANIFEST_SHA256 = "ed866c015e4017532a31251f62ff57ee26311f78232acbc729ec7ae91d8525c4"
EXPECTED_DEVELOPMENT_SELECTION_SHA256 = "0aa8a9cc0c0708a86988158a76c776a7a6015d7db9b542e976ac01af0c81bdd9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("frozen ThermoML plan hash changed")
    if sha256(SELECTION_MANIFEST) != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise RuntimeError("sealed cohort manifest hash changed")
    if sha256(DEVELOPMENT_SELECTION) != EXPECTED_DEVELOPMENT_SELECTION_SHA256:
        raise RuntimeError("development selection hash changed")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

    selection = pd.read_csv(DEVELOPMENT_SELECTION)
    confirmation_files = set(pd.read_csv(CONFIRMATION_SELECTION)["source_file"])
    if set(selection["source_file"]) & confirmation_files:
        raise ValueError("development and confirmation source files overlap")

    rows = []
    source_hashes = {}
    for selected in selection.itertuples(index=False):
        source_path = DATA_ROOT / selected.source_file
        source_hashes[selected.source_file] = sha256(source_path)
        document = json.loads(source_path.read_text(encoding="utf-8"))
        if document["Citation"]["sDOI"] != selected.doi:
            raise ValueError("DOI mismatch")
        table_matches = [
            table
            for table in document["PureOrMixtureData"]
            if int(table["nPureOrMixtureDataNumber"]) == int(selected.table_number)
        ]
        if len(table_matches) != 1:
            raise ValueError("selected table missing or duplicated")
        table = table_matches[0]
        variable_number = int(table["Variable"][0]["nVarNumber"])
        property_number = int(table["Property"][0]["nPropNumber"])
        entity_rows = []
        for point in table["NumValues"]:
            temperature = [
                value
                for value in point["VariableValue"]
                if int(value["nVarNumber"]) == variable_number
            ]
            pressure = [
                value
                for value in point["PropertyValue"]
                if int(value["nPropNumber"]) == property_number
            ]
            if len(temperature) != 1 or len(pressure) != 1:
                raise ValueError("selected variable/property value missing or duplicated")
            entity_rows.append(
                {
                    "entity_id": selected.inchi_key,
                    "doi": selected.doi,
                    "publication_year": int(selected.publication_year),
                    "fold": int(selected.fold),
                    "common_name": selected.common_name,
                    "formula": selected.formula,
                    "source_file": selected.source_file,
                    "table_number": int(selected.table_number),
                    "temperature_k": float(temperature[0]["nVarValue"]),
                    "pressure_kpa": float(pressure[0]["nPropValue"]),
                }
            )
        entity = pd.DataFrame(entity_rows).sort_values("temperature_k", kind="stable")
        if len(entity) != int(selected.rows):
            raise ValueError("selected row count changed")
        if entity["temperature_k"].nunique() != len(entity):
            raise ValueError("duplicate development temperature")
        if not entity["pressure_kpa"].gt(0).all():
            raise ValueError("nonpositive development vapor pressure")
        entity["role"] = ["support" if index % 4 == 0 else "query" for index in range(len(entity))]
        rows.extend(entity.to_dict("records"))

    data = pd.DataFrame(rows)
    data.insert(0, "source_row_id", range(len(data)))
    if data["entity_id"].nunique() != 282 or data["doi"].nunique() != 142:
        raise ValueError("materialized development identity counts changed")
    if len(data) != 9_794:
        raise ValueError("materialized development row count changed")
    data.to_csv(OUTPUT_ROOT / "development_curves.csv", index=False)

    manifest = {
        "scope": "ThermoML vapor-pressure development target materialization",
        "plan_sha256": EXPECTED_PLAN_SHA256,
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "development_selection_sha256": EXPECTED_DEVELOPMENT_SELECTION_SHA256,
        "materializer_sha256": sha256(Path(__file__)),
        "confirmation_source_files_opened": False,
        "confirmation_targets_opened": False,
        "entities": int(data["entity_id"].nunique()),
        "dois": int(data["doi"].nunique()),
        "rows": len(data),
        "support_rows": int(data["role"].eq("support").sum()),
        "query_rows": int(data["role"].eq("query").sum()),
        "pressure_positive": bool(data["pressure_kpa"].gt(0).all()),
        "pressure_min_kpa": float(data["pressure_kpa"].min()),
        "pressure_max_kpa": float(data["pressure_kpa"].max()),
        "development_curves_sha256": sha256(OUTPUT_ROOT / "development_curves.csv"),
        "source_json_sha256": source_hashes,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_json_sha256"}))


if __name__ == "__main__":
    main()
