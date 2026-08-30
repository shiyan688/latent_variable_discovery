#!/usr/bin/env python3
"""Single fixed evaluation of the sealed post-snapshot Starry ZT cohort."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.prepare_application_datasets as application_data


ROOT = PROJECT_ROOT / "runs/starry_zt_temporal_confirmation_20260829/evaluation"
SELECTION_ROOT = PROJECT_ROOT / "runs/starry_zt_temporal_confirmation_20260829/selection"
PLAN = PROJECT_ROOT / "STARRY_ZT_TEMPORAL_CONFIRMATION_PLAN_20260829.md"
LATEST_CURVES = PROJECT_ROOT / "data/external/starrydata_latest_20260829/ThermoelectricMaterials_curves.csv.gz"
DEV_ROOT = PROJECT_ROOT / "data/application_reviewer_clean/starry_te/zt"
EXPECTED_PLAN_SHA = "9b3943b7d5662f01d9cafd023fecb73b5346de272f349df77841ee1f57648817"
EXPECTED_SELECTION_MANIFEST_SHA = "91f920e27bf41d9d5caa1aff6010261fa3e3a6ed5c14b93f90ea6eab63fdde9c"
FEATURES = ["temperature", *application_data.COMPOSITION_DESCRIPTOR_COLUMNS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_sequence(value: object) -> np.ndarray:
    parsed = ast.literal_eval(value) if isinstance(value, str) else value
    return np.asarray(parsed, dtype=float).reshape(-1)


def split_support_query(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("temperature", kind="stable").reset_index(drop=True)
    support = np.arange(len(ordered)) % 4 == 0
    return ordered.loc[support].copy(), ordered.loc[~support].copy()


def fit_polynomial(frame: pd.DataFrame, mean: float, scale: float, degree: int) -> np.ndarray:
    tau = (frame["temperature"].to_numpy(float) - mean) / scale
    basis = np.column_stack([tau**power for power in range(degree + 1)])
    return np.linalg.lstsq(basis, frame["target"].to_numpy(float), rcond=None)[0]


def predict_polynomial(frame: pd.DataFrame, mean: float, scale: float, q: np.ndarray) -> np.ndarray:
    tau = (frame["temperature"].to_numpy(float) - mean) / scale
    return np.column_stack([tau**power for power in range(len(q))]) @ q


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"confirmation evaluation already consumed: {ROOT}")
    selection_manifest_path = SELECTION_ROOT / "manifest.json"
    selection_decision_path = SELECTION_ROOT / "selection_decision.json"
    selection_table_path = SELECTION_ROOT / "selected_entities_target_blind.csv"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection_decision = json.loads(selection_decision_path.read_text(encoding="utf-8"))
    if (
        sha256(PLAN) != EXPECTED_PLAN_SHA
        or sha256(selection_manifest_path) != EXPECTED_SELECTION_MANIFEST_SHA
        or selection_manifest["target_column_opened"] is not False
        or selection_decision["target_column_opened"] is not False
        or selection_decision["authorize_fixed_evaluation"] is not True
    ):
        raise ValueError("unaltered target-blind selection seal is required")
    selected = pd.read_csv(selection_table_path)
    if len(selected) != 30 or selected["sample_id"].nunique() != 30:
        raise ValueError("sealed 30-entity cohort is required")

    ROOT.mkdir(parents=True, exist_ok=False)
    write_json(
        ROOT / "consumption_receipt.json",
        {
            "selection_manifest_sha256": sha256(selection_manifest_path),
            "selection_table_sha256": sha256(selection_table_path),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "target_access_consumed": True,
            "rerun_authorized": False,
        },
    )

    latest = pd.read_csv(
        LATEST_CURVES,
        usecols=[
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
            "y",
            "created_at",
            "updated_at",
        ],
    )
    source = latest.iloc[selected["source_row_index"].to_numpy(int)].copy()
    if (
        source["sample_id"].astype(int).tolist() != selected["sample_id"].astype(int).tolist()
        or source["DOI"].astype(str).tolist() != selected["DOI"].astype(str).tolist()
        or source["x"].astype(str).tolist() != selected["x"].astype(str).tolist()
    ):
        raise ValueError("sealed source rows changed")

    entity_frames = []
    for row in source.itertuples(index=False):
        temperature = parse_sequence(row.x)
        target = parse_sequence(row.y)
        if len(temperature) != len(target) or len(target) < 20:
            raise ValueError("sealed curve has invalid paired length")
        if not np.isfinite(np.column_stack([temperature, target])).all():
            raise ValueError("sealed curve contains non-finite value")
        fractions = application_data._composition_to_element_fraction(str(row.composition))
        descriptors = application_data._composition_descriptors_from_fractions(fractions)
        frame = pd.DataFrame(
            {
                "label": str(row.sample_id),
                "temperature": temperature,
                "target": target,
                "DOI": str(row.DOI),
                "composition": str(row.composition),
            }
        )
        for column in application_data.COMPOSITION_DESCRIPTOR_COLUMNS:
            frame[column] = descriptors[column]
        entity_frames.append(frame)
    confirmation = pd.concat(entity_frames, ignore_index=True)
    confirmation["_row_id"] = np.arange(len(confirmation))

    development = pd.concat(
        [pd.read_csv(DEV_ROOT / "train.csv"), pd.read_csv(DEV_ROOT / "test.csv")],
        ignore_index=True,
    )
    mean = float(development["temperature"].mean())
    scale = float(development["temperature"].std())
    no_q_quadratic = fit_polynomial(development, mean, scale, degree=2)
    no_q_mlp = TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-4,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=50,
                random_state=20260829,
            ),
        ),
        transformer=StandardScaler(),
    ).fit(development[FEATURES], development["target"])

    prediction_frames = []
    q_rows = []
    leakage_max_difference = 0.0
    for label, entity in confirmation.groupby("label", sort=True):
        support, query = split_support_query(entity)
        q_linear = fit_polynomial(support, mean, scale, degree=1)
        q_quadratic = fit_polynomial(support, mean, scale, degree=2)
        q_rows.append(
            {
                "label": label,
                "support_rows": len(support),
                "query_rows": len(query),
                "q0": float(q_quadratic[0]),
                "q1": float(q_quadratic[1]),
                "q2": float(q_quadratic[2]),
            }
        )
        predictions = {
            "support_knn": KNeighborsRegressor(
                n_neighbors=min(5, len(support)), weights="distance"
            ).fit(
                (support[["temperature"]] - mean) / scale,
                support["target"],
            ).predict((query[["temperature"]] - mean) / scale),
            "linear_req": predict_polynomial(query, mean, scale, q_linear),
            "quadratic_req": predict_polynomial(query, mean, scale, q_quadratic),
            "no_q_global_quadratic": predict_polynomial(query, mean, scale, no_q_quadratic),
            "no_q_mlp": no_q_mlp.predict(query[FEATURES]),
        }
        for family, values in predictions.items():
            scored = query[["label", "DOI", "composition", "temperature", "target"]].copy()
            scored["prediction"] = values
            scored["family"] = family
            prediction_frames.append(scored)

        perturbed = entity.copy()
        perturbed.loc[perturbed["_row_id"].isin(query["_row_id"]), "target"] += 1_000_000.0
        perturbed_support, perturbed_query = split_support_query(perturbed)
        perturbed_q = fit_polynomial(perturbed_support, mean, scale, degree=2)
        perturbed_prediction = predict_polynomial(perturbed_query, mean, scale, perturbed_q)
        leakage_max_difference = max(
            leakage_max_difference,
            float(np.max(np.abs(perturbed_q - q_quadratic))),
            float(np.max(np.abs(perturbed_prediction - predictions["quadratic_req"]))),
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy(float)).all():
        raise ValueError("non-finite temporal confirmation prediction")
    summary_rows = []
    entity_metric_rows = []
    for family, group in predictions.groupby("family", sort=True):
        summary_rows.append(
            {
                "family": family,
                "pooled_r2": float(r2_score(group["target"], group["prediction"])),
                "physical_rmse": float(np.sqrt(np.mean((group["target"] - group["prediction"]) ** 2))),
            }
        )
        for label, entity in group.groupby("label", sort=True):
            target_scale = float(np.std(entity["target"].to_numpy(float)))
            if target_scale == 0.0:
                raise ValueError("constant query target prevents entity R2")
            entity_metric_rows.append(
                {
                    "family": family,
                    "label": label,
                    "r2": float(r2_score(entity["target"], entity["prediction"])),
                    "reference_nrmse": float(
                        np.sqrt(np.mean((entity["target"] - entity["prediction"]) ** 2)) / target_scale
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values("pooled_r2", ascending=False)
    entity_metrics = pd.DataFrame(entity_metric_rows)
    selected_summary = summary.loc[summary["family"].eq("quadratic_req")].iloc[0]
    selected_entities = entity_metrics.loc[entity_metrics["family"].eq("quadratic_req")]
    linear_entities = entity_metrics.loc[entity_metrics["family"].eq("linear_req")][
        ["label", "reference_nrmse"]
    ].rename(columns={"reference_nrmse": "linear_reference_nrmse"})
    ratios = selected_entities.merge(linear_entities, on="label", validate="one_to_one")
    maximum_ratio = float(
        np.max(ratios["reference_nrmse"] / ratios["linear_reference_nrmse"])
    )

    primary_predictions = predictions.loc[predictions["family"].eq("quadratic_req")]
    by_entity = {label: group for label, group in primary_predictions.groupby("label", sort=True)}
    labels = np.array(sorted(by_entity))
    rng = np.random.default_rng(20260829)
    bootstrap = []
    for _ in range(10000):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        target = np.concatenate([by_entity[label]["target"].to_numpy(float) for label in sampled])
        prediction = np.concatenate([by_entity[label]["prediction"].to_numpy(float) for label in sampled])
        bootstrap.append(r2_score(target, prediction))
    interval = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    median_entity_r2 = float(selected_entities["r2"].median())
    fraction_entity_pass = float(np.mean(selected_entities["r2"] >= 0.85))
    passed = bool(
        selected_summary["pooled_r2"] >= 0.85
        and interval[0] > 0.85
        and median_entity_r2 >= 0.85
        and fraction_entity_pass >= 2.0 / 3.0
        and leakage_max_difference == 0.0
        and maximum_ratio <= 10.0
    )
    decision = {
        "scope": "post-snapshot, DOI/composition-disjoint Starry ZT confirmation",
        "formula": "ZT(T)=q0+q1*tau+q2*tau^2",
        "entities": len(labels),
        "query_rows": len(primary_predictions),
        "quadratic_pooled_r2": float(selected_summary["pooled_r2"]),
        "quadratic_physical_rmse": float(selected_summary["physical_rmse"]),
        "entity_bootstrap_r2_95_interval": interval,
        "median_entity_r2": median_entity_r2,
        "fraction_entities_r2_at_least_0_85": fraction_entity_pass,
        "maximum_entity_nrmse_ratio_to_linear_req": maximum_ratio,
        "query_target_input_max_difference": leakage_max_difference,
        "family_summary": summary.to_dict(orient="records"),
        "gates": {
            "pooled_r2_at_least_0_85": bool(selected_summary["pooled_r2"] >= 0.85),
            "bootstrap_lower_bound_above_0_85": interval[0] > 0.85,
            "median_entity_r2_at_least_0_85": median_entity_r2 >= 0.85,
            "two_thirds_entities_r2_at_least_0_85": fraction_entity_pass >= 2.0 / 3.0,
            "finite_and_exact_query_target_invariance": leakage_max_difference == 0.0,
            "no_entity_above_ten_times_linear_req_nrmse": maximum_ratio <= 10.0,
        },
        "temporal_confirmation_passed": passed,
        "predictive_superiority_inferred": False,
        "evaluation_consumed": True,
    }
    confirmation.to_csv(ROOT / "confirmation_data_used.csv", index=False)
    pd.DataFrame(q_rows).to_csv(ROOT / "interpretable_q.csv", index=False)
    predictions.to_csv(ROOT / "query_predictions.csv", index=False)
    summary.to_csv(ROOT / "family_summary.csv", index=False)
    entity_metrics.to_csv(ROOT / "per_entity_metrics.csv", index=False)
    write_json(ROOT / "decision.json", decision)
    write_json(
        ROOT / "manifest.json",
        {
            "plan_sha256": sha256(PLAN),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "selection_manifest_sha256": sha256(selection_manifest_path),
            "selection_decision_sha256": sha256(selection_decision_path),
            "selection_table_sha256": sha256(selection_table_path),
            "latest_curves_sha256": sha256(LATEST_CURVES),
            "development_train_sha256": sha256(DEV_ROOT / "train.csv"),
            "development_test_sha256": sha256(DEV_ROOT / "test.csv"),
            "files": {
                path.name: sha256(path)
                for path in ROOT.iterdir()
                if path.is_file() and path.name != "manifest.json"
            },
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
