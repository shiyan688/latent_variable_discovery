#!/usr/bin/env python3
"""Materialize the frozen vapor-pressure structure-selection audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "runs/thermoml_vapor_pressure_development_data_20260829/development_curves.csv"
RESULT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_structure_development_20260829"
OUTPUT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_structure_selection_audit_20260830"
EXPECTED = {
    DATA: "9ebc8ea5a8b870cb98cc829c1700d4ebdad806c043014a0a5051ada8629411b6",
    RESULT_ROOT / "per_entity_metrics.csv": "ead068c547459187efdea2f0935f459bc44962d75ca5e85bec7b3f539326a7e2",
    RESULT_ROOT / "oof_expression_coefficients.csv": "58b773a3e9023f3faf822abf902e421e2b1f64c29a6ad33449a37f42c4d07cfe",
    RESULT_ROOT / "decision.json": "928ee5bc4f21c156a00a49737dca0729b8e9a52f7fb23a165a870b7727adaa26",
}
ORDER = ("v_log", "v_T", "v_inv2")
FAMILIES = ("coarse_cc",) + ORDER


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen input hash changed: {path.relative_to(PROJECT_ROOT)}")

    data = pd.read_csv(DATA)
    metrics = pd.read_csv(RESULT_ROOT / "per_entity_metrics.csv")
    coefficients = pd.read_csv(RESULT_ROOT / "oof_expression_coefficients.csv")
    decision = json.loads((RESULT_ROOT / "decision.json").read_text(encoding="utf-8"))

    entity_folds = data[["entity_id", "fold"]].drop_duplicates()
    if entity_folds["entity_id"].duplicated().any():
        raise ValueError("an entity crosses development folds")
    metrics = metrics.merge(entity_folds, on="entity_id", validate="many_to_one")

    reference_rows = []
    for fold in range(5):
        expected_reference = float(data.loc[~data["fold"].eq(fold), "temperature_k"].median())
        observed = coefficients.loc[
            coefficients["fold"].eq(fold), "temperature_reference_k"
        ].unique()
        if len(observed) != 1 or float(observed[0]) != expected_reference:
            raise ValueError(f"fold {fold} reference temperature mismatch")
        reference_rows.append(
            {
                "fold": fold,
                "outer_training_temperature_median_k": expected_reference,
                "heldout_entities": int(entity_folds["fold"].eq(fold).sum()),
                "heldout_dois": int(data.loc[data["fold"].eq(fold), "doi"].nunique()),
            }
        )

    scored = metrics.loc[metrics["family"].isin(FAMILIES)].copy()
    fold_rows = (
        scored.groupby(["fold", "family"], sort=True)["physical_nrmse"]
        .median()
        .rename("median_entity_physical_nrmse")
        .reset_index()
    )
    aggregate_rows = (
        scored.groupby("family", sort=True)["physical_nrmse"]
        .median()
        .rename("all_entity_median_physical_nrmse")
        .reset_index()
    )
    reported = {
        row["family"]: row["median_entity_physical_nrmse"]
        for row in decision["family_summary"]
    }
    serialization_differences = []
    for row in aggregate_rows.itertuples(index=False):
        difference = abs(
            float(row.all_entity_median_physical_nrmse) - float(reported[row.family])
        )
        serialization_differences.append(difference)
        if difference > 1e-15:
            raise ValueError(f"aggregate score mismatch for {row.family}")

    candidates = aggregate_rows.loc[aggregate_rows["family"].isin(ORDER)]
    best = float(candidates["all_entity_median_physical_nrmse"].min())
    tied = set(
        candidates.loc[
            candidates["all_entity_median_physical_nrmse"] <= best * 1.01,
            "family",
        ]
    )
    selected = next(family for family in ORDER if family in tied)
    if selected != decision["selected_family"]:
        raise ValueError("frozen tie break does not reproduce the decision")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    fold_rows.to_csv(OUTPUT_ROOT / "per_fold_candidate_nrmse.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(OUTPUT_ROOT / "fold_reference_temperatures.csv", index=False)
    audit = {
        "scope": "development-only vapor-pressure structure-selection audit",
        "candidate_order": list(ORDER),
        "reported_unmodified_baseline": "coarse_cc",
        "candidate_terms": {
            "v_log": "log(T/T_ref)",
            "v_T": "(T-T_ref)/T_ref",
            "v_inv2": "(1/T-1/T_ref)^2",
        },
        "fold_grouping": "five DOI-disjoint folds",
        "fold_reference_rule": "median temperature of all entities outside the heldout fold",
        "cross_fold_aggregation": "one median over all 282 heldout-entity physical NRMSE values, not a median of fold medians",
        "tie_rule": "eligible when score <= 1.01 * best; choose first in candidate_order",
        "aggregate_family_scores": aggregate_rows.to_dict(orient="records"),
        "csv_json_serialization_max_abs_difference": max(serialization_differences),
        "eligible_within_one_percent": [family for family in ORDER if family in tied],
        "selected_family": selected,
        "development_query_role": "scores and selects the frozen structure",
        "external_query_role": "scores the sealed structure only",
        "external_fit_role": "only entity support targets estimate coefficients",
        "confirmation_targets_opened": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): expected for path, expected in EXPECTED.items()
        },
    }
    (OUTPUT_ROOT / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "auditor_sha256": sha256(Path(__file__)),
        "files": {
            path.name: sha256(path)
            for path in sorted(OUTPUT_ROOT.iterdir())
            if path.is_file()
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
