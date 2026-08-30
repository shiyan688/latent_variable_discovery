#!/usr/bin/env python3
"""Strict entity-OOF interpretable coefficient re-q on reviewer-clean Starry ZT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "runs/starry_zt_interpretable_req_20260829"
PLAN = PROJECT_ROOT / "STARRY_ZT_INTERPRETABLE_REQ_PLAN_20260829.md"
DATA_ROOT = PROJECT_ROOT / "data/application_reviewer_clean/starry_te/zt"
AUDIT = PROJECT_ROOT / "NEGATIVE_RESULT_REASSESSMENT_20260825.md"
FOLDS = tuple(range(5))
FEATURES = [
    "temperature",
    "comp_n_elements",
    "comp_entropy",
    "comp_max_fraction",
    "comp_mean_z",
    "comp_std_z",
    "comp_min_z",
    "comp_max_z",
    "comp_mean_period",
    "comp_mean_group",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def split_support_query(frame: pd.DataFrame, offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("temperature", kind="stable").reset_index(drop=True)
    support_mask = np.arange(len(ordered)) % 4 == offset
    return ordered.loc[support_mask].copy(), ordered.loc[~support_mask].copy()


def polynomial_q(support: pd.DataFrame, mean: float, scale: float, degree: int) -> np.ndarray:
    tau = (support["temperature"].to_numpy(float) - mean) / scale
    return np.linalg.lstsq(
        np.column_stack([tau**power for power in range(degree + 1)]),
        support["target"].to_numpy(float),
        rcond=None,
    )[0]


def polynomial_prediction(frame: pd.DataFrame, mean: float, scale: float, q: np.ndarray) -> np.ndarray:
    tau = (frame["temperature"].to_numpy(float) - mean) / scale
    return np.column_stack([tau**power for power in range(len(q))]) @ q


def physical_coefficients(q: np.ndarray, mean: float, scale: float) -> tuple[float, float, float]:
    q0, q1, q2 = q
    return (
        float(q0 - mean * q1 / scale + mean**2 * q2 / scale**2),
        float(q1 / scale - 2.0 * mean * q2 / scale**2),
        float(q2 / scale**2),
    )


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"result root already exists: {ROOT}")
    audit_text = AUDIT.read_text(encoding="utf-8")
    if "| ZT | 0/80 |" not in audit_text or not PLAN.is_file():
        raise ValueError("reviewer-clean ZT audit and frozen plan are required")
    train_path = DATA_ROOT / "train.csv"
    test_path = DATA_ROOT / "test.csv"
    data = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    data["_row_id"] = np.arange(len(data))
    labels = sorted(data["label"].unique().tolist())
    if len(labels) != 80 or data.shape[0] != 5216:
        raise ValueError("reviewer-clean ZT cohort changed")
    fold_by_label = {label: index % len(FOLDS) for index, label in enumerate(labels)}
    data["fold"] = data["label"].map(fold_by_label)

    prediction_frames = []
    q_rows = []
    stability_rows = []
    query_target_input_max_difference = 0.0
    for fold in FOLDS:
        train = data.loc[~data["fold"].eq(fold)].copy()
        test = data.loc[data["fold"].eq(fold)].copy()
        mean = float(train["temperature"].mean())
        scale = float(train["temperature"].std())
        global_q = polynomial_q(train, mean, scale, degree=2)
        mlp = TransformedTargetRegressor(
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
                    random_state=20260829 + fold,
                ),
            ),
            transformer=StandardScaler(),
        ).fit(train[FEATURES], train["target"])

        for label, entity in test.groupby("label", sort=True):
            support, query = split_support_query(entity, offset=0)
            q_linear = polynomial_q(support, mean, scale, degree=1)
            q_quadratic = polynomial_q(support, mean, scale, degree=2)
            physical_q = physical_coefficients(q_quadratic, mean, scale)
            q_rows.append(
                {
                    "fold": fold,
                    "label": label,
                    "support_rows": len(support),
                    "query_rows": len(query),
                    "temperature_mean_train": mean,
                    "temperature_scale_train": scale,
                    "q0": float(q_quadratic[0]),
                    "q1": float(q_quadratic[1]),
                    "q2": float(q_quadratic[2]),
                    "physical_intercept": physical_q[0],
                    "physical_linear_temperature": physical_q[1],
                    "physical_quadratic_temperature": physical_q[2],
                }
            )
            predictions = {
                "quadratic_req": polynomial_prediction(query, mean, scale, q_quadratic),
                "linear_req": polynomial_prediction(query, mean, scale, q_linear),
                "support_knn": KNeighborsRegressor(
                    n_neighbors=min(5, len(support)), weights="distance"
                ).fit(
                    ((support[["temperature"]] - mean) / scale), support["target"]
                ).predict((query[["temperature"]] - mean) / scale),
                "no_q_global_quadratic": polynomial_prediction(query, mean, scale, global_q),
                "no_q_mlp": mlp.predict(query[FEATURES]),
            }
            for family, values in predictions.items():
                scored = query[["label", "fold", "temperature", "target"]].copy()
                scored["prediction"] = values
                scored["family"] = family
                prediction_frames.append(scored)

            perturbed = entity.copy()
            perturbed.loc[perturbed["_row_id"].isin(query["_row_id"]), "target"] += 1_000_000.0
            perturbed_support, perturbed_query = split_support_query(perturbed, offset=0)
            perturbed_q = polynomial_q(perturbed_support, mean, scale, degree=2)
            perturbed_prediction = polynomial_prediction(perturbed_query, mean, scale, perturbed_q)
            query_target_input_max_difference = max(
                query_target_input_max_difference,
                float(np.max(np.abs(perturbed_q - q_quadratic))),
                float(np.max(np.abs(perturbed_prediction - predictions["quadratic_req"]))),
            )

            for offset in range(4):
                alternate_support, _ = split_support_query(entity, offset=offset)
                alternate_q = polynomial_q(alternate_support, mean, scale, degree=2)
                alternate_physical = physical_coefficients(alternate_q, mean, scale)
                stability_rows.append(
                    {
                        "fold": fold,
                        "label": label,
                        "offset": offset,
                        "physical_intercept": alternate_physical[0],
                        "physical_linear_temperature": alternate_physical[1],
                        "physical_quadratic_temperature": alternate_physical[2],
                    }
                )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy(float)).all():
        raise ValueError("non-finite ZT prediction")
    summaries = []
    battery_rows = []
    for family, group in predictions.groupby("family", sort=True):
        summaries.append(
            {
                "family": family,
                "pooled_oof_r2": float(r2_score(group["target"], group["prediction"])),
                "physical_rmse": float(np.sqrt(np.mean((group["target"] - group["prediction"]) ** 2))),
            }
        )
        for label, entity in group.groupby("label", sort=True):
            scale = float(np.std(entity["target"].to_numpy(float)))
            battery_rows.append(
                {
                    "family": family,
                    "label": label,
                    "r2": float(r2_score(entity["target"], entity["prediction"])),
                    "reference_nrmse": float(
                        np.sqrt(np.mean((entity["target"] - entity["prediction"]) ** 2)) / scale
                    ),
                }
            )
    summary = pd.DataFrame(summaries).sort_values("pooled_oof_r2", ascending=False)
    per_entity = pd.DataFrame(battery_rows)
    selected = summary.loc[summary["family"].eq("quadratic_req")].iloc[0]
    selected_entity = per_entity.loc[per_entity["family"].eq("quadratic_req")]
    linear_entity = per_entity.loc[per_entity["family"].eq("linear_req")][
        ["label", "reference_nrmse"]
    ].rename(columns={"reference_nrmse": "linear_reference_nrmse"})
    ratios = selected_entity.merge(linear_entity, on="label", validate="one_to_one")
    max_ratio = float(np.max(ratios["reference_nrmse"] / ratios["linear_reference_nrmse"]))

    rng = np.random.default_rng(20260829)
    selected_predictions = predictions.loc[predictions["family"].eq("quadratic_req")]
    grouped = {label: group for label, group in selected_predictions.groupby("label", sort=True)}
    ordered_labels = np.array(sorted(grouped))
    bootstrap = []
    for _ in range(10000):
        sampled = rng.choice(ordered_labels, size=len(ordered_labels), replace=True)
        target = np.concatenate([grouped[label]["target"].to_numpy(float) for label in sampled])
        prediction = np.concatenate([grouped[label]["prediction"].to_numpy(float) for label in sampled])
        bootstrap.append(r2_score(target, prediction))

    stability = pd.DataFrame(stability_rows)
    stability_correlations = []
    q_columns = ["physical_intercept", "physical_linear_temperature", "physical_quadratic_temperature"]
    base_q = stability.loc[stability["offset"].eq(0)].sort_values("label")
    base_scaled = StandardScaler().fit_transform(base_q[q_columns])
    for offset in range(1, 4):
        alternate = stability.loc[stability["offset"].eq(offset)].sort_values("label")
        alternate_scaled = StandardScaler().fit_transform(alternate[q_columns])
        stability_correlations.append(
            {
                "offset": offset,
                "q_distance_spearman_vs_offset0": float(
                    spearmanr(pdist(base_scaled), pdist(alternate_scaled)).statistic
                ),
            }
        )

    empirical_rows = []
    for label, entity in data.groupby("label", sort=True):
        ordered = entity.groupby("temperature", as_index=False)["target"].mean().sort_values("temperature")
        normalized_temperature = (
            ordered["temperature"].to_numpy(float) - ordered["temperature"].min()
        ) / (ordered["temperature"].max() - ordered["temperature"].min())
        empirical_rows.append(
            np.interp(np.linspace(0.0, 1.0, 21), normalized_temperature, ordered["target"].to_numpy(float))
        )
    response_distance = pdist(StandardScaler().fit_transform(np.asarray(empirical_rows)))
    q_distance = pdist(base_scaled)
    continuity = float(spearmanr(q_distance, response_distance).statistic)

    passed = bool(
        selected["pooled_oof_r2"] >= 0.85
        and query_target_input_max_difference == 0.0
        and max_ratio <= 10.0
    )
    decision = {
        "analysis_scope": "reviewer-clean Starry ZT strict entity-OOF/support-query development",
        "selected_formula": "ZT(T)=q0+q1*tau+q2*tau^2; tau=(T-mu_train)/sigma_train",
        "scientific_interpretation": {
            "q0": "reference ZT",
            "q1": "first-order temperature sensitivity",
            "q2": "temperature curvature",
        },
        "selected_pooled_oof_r2": float(selected["pooled_oof_r2"]),
        "selected_physical_rmse": float(selected["physical_rmse"]),
        "entity_bootstrap_r2_95_interval": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "median_entity_r2": float(selected_entity["r2"].median()),
        "fraction_entities_r2_at_least_0_85": float(np.mean(selected_entity["r2"] >= 0.85)),
        "maximum_entity_nrmse_ratio_to_linear_req": max_ratio,
        "q_response_distance_spearman": continuity,
        "q_stability": stability_correlations,
        "family_summary": summary.to_dict(orient="records"),
        "gates": {
            "pooled_oof_r2_at_least_0_85": bool(selected["pooled_oof_r2"] >= 0.85),
            "finite_predictions": True,
            "exact_query_target_input_invariance": query_target_input_max_difference == 0.0,
            "no_entity_above_ten_times_linear_req_nrmse": max_ratio <= 10.0,
        },
        "expression_endpoint_passed_on_development": passed,
        "predictive_superiority_inferred": False,
        "independent_external_confirmation_required_for_confirmatory_claim": True,
    }
    ROOT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"label": labels, "fold": [fold_by_label[label] for label in labels]}).to_csv(
        ROOT / "fold_assignments.csv", index=False
    )
    pd.DataFrame(q_rows).to_csv(ROOT / "oof_interpretable_q.csv", index=False)
    stability.to_csv(ROOT / "q_support_offset_stability.csv", index=False)
    pd.DataFrame(stability_correlations).to_csv(ROOT / "q_stability_summary.csv", index=False)
    predictions.to_csv(ROOT / "oof_query_predictions.csv", index=False)
    summary.to_csv(ROOT / "family_summary.csv", index=False)
    per_entity.to_csv(ROOT / "per_entity_metrics.csv", index=False)
    write_json(ROOT / "decision.json", decision)
    write_json(
        ROOT / "manifest.json",
        {
            "plan_sha256": sha256(PLAN),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "audit_sha256": sha256(AUDIT),
            "train_csv_sha256": sha256(train_path),
            "test_csv_sha256": sha256(test_path),
            "query_target_input_max_difference": query_target_input_max_difference,
            "files": {path.name: sha256(path) for path in ROOT.iterdir() if path.is_file() and path.name != "manifest.json"},
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
