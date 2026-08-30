#!/usr/bin/env python3
"""Consume and evaluate the sealed ThermoML temporal cohort exactly once."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / "runs/_runtime_cache/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(PROJECT_ROOT / "runs/_runtime_cache/xdg")

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import minimize_scalar
from torch import nn

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.backends.torch_mlp import build_torch_model_factory
from lvs.core.pipeline import (
    NormalizationStats,
    _ensure_prediction_column,
    _optimize_calibration_q,
    denormalize_targets,
    normalize_features,
    normalize_targets,
)


PLAN = PROJECT_ROOT / "THERMOML_SINGLE_USE_CONFIRMATION_PLAN_20260829.md"
ANALYZER = PROJECT_ROOT / "scripts/analyze_thermoml_single_use_confirmation_20260829.py"
PACKAGE_ROOT = PROJECT_ROOT / "runs/thermoml_all_development_package_20260829"
PACKAGE_SEAL = PACKAGE_ROOT / "analysis/package_seal.json"
PACKAGE_MANIFEST = PACKAGE_ROOT / "analysis/manifest.json"
COHORT_ROOT = PROJECT_ROOT / "runs/thermoml_vapor_pressure_cohorts_20260829"
COHORT_MANIFEST = COHORT_ROOT / "selection_manifest.json"
CONFIRMATION_SELECTION = COHORT_ROOT / "confirmation_selection.csv"
SOURCE_ROOT = PROJECT_ROOT / "data/external/thermoml_2020_archive/extracted"
OUTPUT_BASE = PROJECT_ROOT / "runs/thermoml_single_use_confirmation_20260829"
EXPECTED_PLAN_SHA256 = "d8b51051e5e6124a5df837f501909c9f1d3b6eb20f49958abac588dec2e80fba"
EXPECTED_CONFIRMATION_SELECTION_SHA256 = "252f1a954ffb933fbde4bf17118c67684619ec5c53ad0604bb9076db838f84cc"
EXPECTED_COHORT_MANIFEST_SHA256 = "ed866c015e4017532a31251f62ff57ee26311f78232acbc729ec7ae91d8525c4"
EXPECTED_ARCHIVE_SHA256 = "231161b5e443dc1ae0e5da8429d86a88474cb722016e5b790817bb31c58d7ec2"
Q_DIM = 4
SEEDS = (0, 1, 2)
SEEDED_FAMILIES = (
    "raw_decoder",
    "raw_q_ridge_expression",
    "functional_v_log",
    "no_q_mlp",
)
FIXED_FAMILIES = (
    "structure_v_log",
    "no_q_global_selected_formula",
    "support_nearest_log",
    "support_linear_log",
    "support_pchip_log",
    "support_antoine",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def canonical_event(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def append_receipt(path: Path, payload: dict) -> str:
    encoded = canonical_event(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(descriptor, encoded + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(encoded).hexdigest()


def expression_design(temperature: np.ndarray, reference: float) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    return np.column_stack(
        [
            np.ones(len(values)),
            1.0 / values - 1.0 / reference,
            np.log(values / reference),
        ]
    )


def antoine_prediction(support: pd.DataFrame, query_temperature: np.ndarray) -> np.ndarray:
    support_c = support["temperature_k"].to_numpy(float) - 273.15
    query_c = np.asarray(query_temperature, dtype=float) - 273.15
    target = support["log_pressure"].to_numpy(float) / np.log(10.0)
    lower = float(-support_c.min() + 1.0e-6)
    grid = np.linspace(lower, 2000.0, 257)

    def fit_at(c_value: float) -> tuple[float, np.ndarray]:
        matrix = np.column_stack([np.ones(len(support_c)), 1.0 / (support_c + c_value)])
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = target - matrix @ coefficients
        return float(np.square(residual).sum()), coefficients

    losses = np.asarray([fit_at(value)[0] for value in grid])
    best = int(np.argmin(losses))
    left = float(grid[max(0, best - 1)])
    right = float(grid[min(len(grid) - 1, best + 1)])
    optimized = minimize_scalar(
        lambda value: fit_at(float(value))[0], bounds=(left, right), method="bounded"
    )
    c_value = float(optimized.x)
    _, coefficients = fit_at(c_value)
    prediction_log10 = np.column_stack(
        [np.ones(len(query_c)), 1.0 / (query_c + c_value)]
    ) @ coefficients
    return np.power(10.0, prediction_log10)


def materialize_confirmation(selection: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    source_hashes = {}
    for selected in selection.itertuples(index=False):
        source_path = SOURCE_ROOT / selected.source_file
        source_hashes[selected.source_file] = sha256(source_path)
        document = read_json(source_path)
        if document["Citation"]["sDOI"] != selected.doi:
            raise ValueError("confirmation DOI mismatch")
        tables = [
            table
            for table in document["PureOrMixtureData"]
            if int(table["nPureOrMixtureDataNumber"]) == int(selected.table_number)
        ]
        if len(tables) != 1:
            raise ValueError("selected confirmation table is missing or duplicated")
        table = tables[0]
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
                raise ValueError("confirmation value is missing or duplicated")
            entity_rows.append(
                {
                    "entity_id": selected.inchi_key,
                    "doi": selected.doi,
                    "publication_year": int(selected.publication_year),
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
            raise ValueError("confirmation row count changed")
        if entity["temperature_k"].nunique() != len(entity):
            raise ValueError("duplicate confirmation temperature")
        if not np.isfinite(entity[["temperature_k", "pressure_kpa"]]).all().all():
            raise ValueError("non-finite confirmation value")
        if not entity["pressure_kpa"].gt(0.0).all():
            raise ValueError("nonpositive confirmation pressure")
        entity["role"] = [
            "support" if index % 4 == 0 else "query" for index in range(len(entity))
        ]
        rows.extend(entity.to_dict("records"))
    data = pd.DataFrame(rows)
    data.insert(0, "source_row_id", range(len(data)))
    data["log_pressure"] = np.log(data["pressure_kpa"].to_numpy(float))
    if (
        len(data) != 2_372
        or data["entity_id"].nunique() != 84
        or data["doi"].nunique() != 45
    ):
        raise ValueError("confirmation identity contract changed")
    return data, source_hashes


def normalizer_from_state(state: dict) -> NormalizationStats:
    return NormalizationStats(
        feature_mean=np.asarray(state["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(state["feature_std"], dtype=np.float32),
        target_mean=float(state["target_mean"]),
        target_std=float(state["target_std"]),
    )


def predict_neural_log(
    model: nn.Module,
    normalizer: NormalizationStats,
    temperature: np.ndarray,
    q: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    features = np.asarray(temperature, dtype=np.float32).reshape(-1, 1)
    normalized = torch.tensor(
        normalize_features(features, normalizer), dtype=torch.float32, device=device
    )
    repeated_q = q.unsqueeze(0).repeat(len(normalized), 1)
    with torch.no_grad():
        prediction = _ensure_prediction_column(
            model(torch.cat([normalized, repeated_q], dim=1))
        ).squeeze(1)
    return denormalize_targets(prediction.cpu().numpy(), normalizer)


def calibrate_q(
    *,
    model: nn.Module,
    normalizer: NormalizationStats,
    config,
    q_prior_mean: torch.Tensor,
    q_prior_std: torch.Tensor,
    support: pd.DataFrame,
    entity_id: str,
    device: torch.device,
) -> tuple[torch.Tensor, int, float, float]:
    features = support[["temperature_k"]].to_numpy(np.float32)
    targets = support["log_pressure"].to_numpy(np.float32)
    feature_tensor = torch.tensor(
        normalize_features(features, normalizer), dtype=torch.float32, device=device
    )
    target_tensor = torch.tensor(
        normalize_targets(targets, normalizer), dtype=torch.float32, device=device
    ).reshape(-1, 1)
    indices = np.arange(len(support), dtype=int)
    label_token = str(entity_id).encode("utf-8")
    label_hash = int.from_bytes(label_token[:8].ljust(8, b"\0"), "little")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, label_hash, 314159]))
    initial = [q_prior_mean.detach().cpu()]
    for _ in range(3):
        draw = torch.tensor(rng.normal(size=Q_DIM).astype(np.float32))
        initial.append(q_prior_mean.detach().cpu() + q_prior_std.detach().cpu() * draw)
    loss_fn = nn.MSELoss()
    candidates = []
    losses = []
    for initial_q in initial:
        candidate = _optimize_calibration_q(
            initial_q,
            steps=1200,
            indices=indices,
            feature_tensor=feature_tensor,
            target_tensor=target_tensor,
            model=model,
            mse_loss=loss_fn,
            q_prior_mean=q_prior_mean,
            q_prior_std=q_prior_std,
            functional_prior_features=None,
            functional_prior_mean=None,
            functional_prior_std=None,
            functional_prior_components=None,
            config=config,
        )
        candidates.append(candidate)
        with torch.no_grad():
            repeated = candidate.unsqueeze(0).repeat(len(indices), 1)
            support_prediction = _ensure_prediction_column(
                model(torch.cat([feature_tensor, repeated], dim=1))
            )
            losses.append(float(loss_fn(support_prediction, target_tensor).item()))
    selected = int(np.argmin(losses))
    calibrated = candidates[selected]
    repeated = _optimize_calibration_q(
        initial[selected],
        steps=1200,
        indices=indices,
        feature_tensor=feature_tensor,
        target_tensor=target_tensor,
        model=model,
        mse_loss=loss_fn,
        q_prior_mean=q_prior_mean,
        q_prior_std=q_prior_std,
        functional_prior_features=None,
        functional_prior_mean=None,
        functional_prior_std=None,
        functional_prior_components=None,
        config=config,
    )
    return calibrated, selected, losses[selected], float(torch.max(torch.abs(repeated - calibrated)))


def prediction_rows(
    query: pd.DataFrame, family: str, seed: int, prediction_kpa: np.ndarray
) -> pd.DataFrame:
    values = np.asarray(prediction_kpa, dtype=float)
    if not np.isfinite(values).all() or not np.all(values > 0.0):
        raise ValueError(f"invalid prediction for {family} seed {seed}")
    frame = query[
        ["source_row_id", "entity_id", "doi", "temperature_k", "pressure_kpa", "log_pressure"]
    ].copy()
    frame["family"] = family
    frame["seed"] = seed
    frame["prediction_kpa"] = values
    frame["prediction_log_pressure"] = np.log(values)
    return frame


def validate_package() -> tuple[dict, dict[str, str]]:
    for path, expected in (
        (PLAN, EXPECTED_PLAN_SHA256),
        (CONFIRMATION_SELECTION, EXPECTED_CONFIRMATION_SELECTION_SHA256),
        (COHORT_MANIFEST, EXPECTED_COHORT_MANIFEST_SHA256),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"frozen confirmation input changed: {path}")
    seal = read_json(PACKAGE_SEAL)
    manifest = read_json(PACKAGE_MANIFEST)
    if sha256(PACKAGE_SEAL) != manifest["package_seal_sha256"]:
        raise ValueError("package seal hash changed")
    if (
        seal["authorize_single_use_confirmation_receipt"] is not True
        or seal["confirmation_targets_opened"] is not False
        or seal["development_expression_passed"] is not True
    ):
        raise ValueError("package does not authorize one-shot confirmation")
    expected_code = seal["confirmation_code_hashes"]
    current_code = {
        "plan": sha256(PLAN),
        "evaluator": sha256(Path(__file__).resolve()),
        "analyzer": sha256(ANALYZER),
    }
    if current_code != expected_code:
        raise ValueError("confirmation code differs from sealed package")
    for relative, expected in seal["artifact_inventory"].items():
        if sha256(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"sealed all-development artifact changed: {relative}")
    return seal, current_code


def main() -> None:
    seal, code_hashes = validate_package()
    seal_hash = sha256(PACKAGE_SEAL)
    output_root = OUTPUT_BASE / seal_hash
    nonce = secrets.token_hex(32)
    fixed_hashes = {
        **code_hashes,
        "package_seal": seal_hash,
        "package_manifest": sha256(PACKAGE_MANIFEST),
        "cohort_manifest": EXPECTED_COHORT_MANIFEST_SHA256,
        "confirmation_selection": EXPECTED_CONFIRMATION_SELECTION_SHA256,
        "official_archive_sha256": EXPECTED_ARCHIVE_SHA256,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    lock_path = output_root / "single_use.lock.json"
    lock = {
        "state": "consumed_before_confirmation_target_access",
        "nonce": nonce,
        "consumed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "fixed_hashes": fixed_hashes,
        "failure_consumes_once": True,
        "rerun_authorized": False,
    }
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, json.dumps(lock, sort_keys=True).encode("utf-8") + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    receipt_path = output_root / "single_use_receipt.jsonl"
    started_hash = append_receipt(
        receipt_path,
        {
            "event": "started_and_consumed",
            "nonce": nonce,
            "previous_event_sha256": None,
            "fixed_hashes": fixed_hashes,
        },
    )

    try:
        selection = pd.read_csv(CONFIRMATION_SELECTION)
        if len(selection) != 84 or selection["inchi_key"].nunique() != 84:
            raise ValueError("sealed confirmation selection count changed")
        data, source_hashes = materialize_confirmation(selection)
        data.to_csv(output_root / "confirmation_data_used.csv", index=False)
        inference = data.copy()
        inference.loc[inference["role"].eq("query"), ["pressure_kpa", "log_pressure"]] = np.nan
        perturbed_truth = data.copy()
        query_mask = perturbed_truth["role"].eq("query")
        perturbed_truth.loc[query_mask, "pressure_kpa"] += 1_000_000.0
        perturbed_truth.loc[query_mask, "log_pressure"] = np.log(
            perturbed_truth.loc[query_mask, "pressure_kpa"].to_numpy(float)
        )
        perturbed_inference = perturbed_truth.copy()
        perturbed_inference.loc[
            perturbed_inference["role"].eq("query"), ["pressure_kpa", "log_pressure"]
        ] = np.nan
        inference_columns = [
            "source_row_id",
            "entity_id",
            "doi",
            "temperature_k",
            "pressure_kpa",
            "log_pressure",
            "role",
        ]
        if not inference[inference_columns].equals(perturbed_inference[inference_columns]):
            raise ValueError("query-target perturbation changed the inference table")

        device_name = seal["device"]
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("sealed CUDA package cannot run on this host")
        device = torch.device(device_name)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        reference = float(seal["temperature_reference_k"])
        prediction_frames = []
        coordinate_rows = []
        fidelity_rows = []
        leakage_max = 0.0

        neural_states = {}
        no_q_states = {}
        ridge_maps = {}
        for seed in SEEDS:
            seed_root = PACKAGE_ROOT / f"seed{seed}"
            neural = torch.load(
                seed_root / "neural_checkpoint.pt", map_location=device, weights_only=False
            )
            neural_model = build_torch_model_factory((256, 128))(1 + Q_DIM).to(device)
            neural_model.load_state_dict(neural["model_state_dict"])
            neural_model.eval()
            neural_states[seed] = (
                neural_model,
                normalizer_from_state(neural["normalizer"]),
                neural,
            )
            no_q = torch.load(
                seed_root / "no_q_checkpoint.pt", map_location=device, weights_only=False
            )
            no_q_model = build_torch_model_factory(tuple(no_q["hidden_widths"]))(1).to(device)
            no_q_model.load_state_dict(no_q["model_state_dict"])
            no_q_model.eval()
            no_q_states[seed] = (
                no_q_model,
                normalizer_from_state(no_q["normalizer"]),
            )
            ridge_maps[seed] = joblib.load(seed_root / "raw_q_to_expression.joblib")

        global_expression = read_json(PACKAGE_ROOT / "seed0/global_expression.json")
        global_coefficients = np.asarray(global_expression["coefficients"], dtype=float)

        for entity_id, truth_entity in data.groupby("entity_id", sort=True):
            inference_entity = inference.loc[inference["entity_id"].eq(entity_id)]
            support = inference_entity.loc[inference_entity["role"].eq("support")].copy()
            query_inference = inference_entity.loc[inference_entity["role"].eq("query")].copy()
            query = truth_entity.loc[truth_entity["role"].eq("query")].copy()
            if query_inference["pressure_kpa"].notna().any():
                raise ValueError("query target survived inference redaction")
            support_temperature = support["temperature_k"].to_numpy(float)
            support_log = support["log_pressure"].to_numpy(float)
            query_temperature = query_inference["temperature_k"].to_numpy(float)
            structure_coefficients = np.linalg.lstsq(
                expression_design(support_temperature, reference), support_log, rcond=None
            )[0]
            structure_log = expression_design(query_temperature, reference) @ structure_coefficients
            prediction_frames.append(
                prediction_rows(query, "structure_v_log", -1, np.exp(structure_log))
            )
            nearest_indices = np.abs(
                query_temperature[:, None] - support_temperature[None, :]
            ).argmin(axis=1)
            fixed_predictions = {
                "no_q_global_selected_formula": np.exp(
                    expression_design(query_temperature, reference) @ global_coefficients
                ),
                "support_nearest_log": np.exp(support_log[nearest_indices]),
                "support_linear_log": np.exp(
                    interp1d(
                        support_temperature,
                        support_log,
                        kind="linear",
                        fill_value="extrapolate",
                    )(query_temperature)
                ),
                "support_pchip_log": np.exp(
                    PchipInterpolator(
                        support_temperature, support_log, extrapolate=True
                    )(query_temperature)
                ),
                "support_antoine": antoine_prediction(support, query_temperature),
            }
            for family, values in fixed_predictions.items():
                prediction_frames.append(prediction_rows(query, family, -1, values))

            for seed in SEEDS:
                model, normalizer, neural = neural_states[seed]
                calibrated_q, selected_start, support_loss, leakage = calibrate_q(
                    model=model,
                    normalizer=normalizer,
                    config=neural["config"],
                    q_prior_mean=neural["q_prior_mean"].to(device),
                    q_prior_std=neural["q_prior_std"].to(device),
                    support=support,
                    entity_id=entity_id,
                    device=device,
                )
                leakage_max = max(leakage_max, leakage)
                raw_decoder_log = predict_neural_log(
                    model, normalizer, query_temperature, calibrated_q, device
                )
                raw_q_coefficients = ridge_maps[seed].predict(
                    pd.DataFrame(
                        calibrated_q.detach().cpu().numpy().reshape(1, -1),
                        columns=[f"raw_q{index}" for index in range(Q_DIM)],
                    )
                )[0]
                raw_q_log = expression_design(query_temperature, reference) @ raw_q_coefficients
                grid_temperature = np.linspace(
                    float(inference_entity["temperature_k"].min()),
                    float(inference_entity["temperature_k"].max()),
                    41,
                )
                decoder_grid_log = predict_neural_log(
                    model, normalizer, grid_temperature, calibrated_q, device
                )
                functional_coefficients = np.linalg.lstsq(
                    expression_design(grid_temperature, reference),
                    decoder_grid_log,
                    rcond=None,
                )[0]
                functional_grid_log = (
                    expression_design(grid_temperature, reference) @ functional_coefficients
                )
                functional_log = (
                    expression_design(query_temperature, reference) @ functional_coefficients
                )
                decoder_grid_physical = np.exp(decoder_grid_log)
                functional_grid_physical = np.exp(functional_grid_log)
                for family, log_values in {
                    "raw_decoder": raw_decoder_log,
                    "raw_q_ridge_expression": raw_q_log,
                    "functional_v_log": functional_log,
                }.items():
                    prediction_frames.append(
                        prediction_rows(query, family, seed, np.exp(log_values))
                    )
                no_q_model, no_q_normalizer = no_q_states[seed]
                normalized_query = torch.tensor(
                    normalize_features(
                        query_temperature.astype(np.float32).reshape(-1, 1), no_q_normalizer
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    no_q_normalized = no_q_model(normalized_query).reshape(-1).cpu().numpy()
                no_q_log = denormalize_targets(no_q_normalized, no_q_normalizer)
                prediction_frames.append(
                    prediction_rows(query, "no_q_mlp", seed, np.exp(no_q_log))
                )
                raw_q = calibrated_q.detach().cpu().numpy()
                coordinate_rows.append(
                    {
                        "seed": seed,
                        "entity_id": entity_id,
                        "doi": truth_entity["doi"].iloc[0],
                        "support_rows": len(support),
                        "query_rows": len(query),
                        "selected_start": selected_start,
                        "support_loss": support_loss,
                        **{f"raw_q{index}": float(value) for index, value in enumerate(raw_q)},
                        **{
                            f"functional_q{index}": float(value)
                            for index, value in enumerate(functional_coefficients)
                        },
                        **{
                            f"structure_q{index}": float(value)
                            for index, value in enumerate(structure_coefficients)
                        },
                        **{
                            f"raw_q_mapped_q{index}": float(value)
                            for index, value in enumerate(raw_q_coefficients)
                        },
                    }
                )
                fidelity_rows.append(
                    {
                        "seed": seed,
                        "entity_id": entity_id,
                        "doi": truth_entity["doi"].iloc[0],
                        "log_projection_r2": 1.0
                        - float(np.square(decoder_grid_log - functional_grid_log).sum())
                        / float(np.square(decoder_grid_log - decoder_grid_log.mean()).sum()),
                        "physical_projection_r2": 1.0
                        - float(
                            np.square(decoder_grid_physical - functional_grid_physical).sum()
                        )
                        / float(
                            np.square(
                                decoder_grid_physical - decoder_grid_physical.mean()
                            ).sum()
                        ),
                    }
                )

        per_seed_predictions = pd.concat(prediction_frames, ignore_index=True)
        expected_query_rows = int(data["role"].eq("query").sum())
        aggregate_frames = []
        for family in SEEDED_FAMILIES:
            family_frame = per_seed_predictions.loc[
                per_seed_predictions["family"].eq(family)
            ]
            if len(family_frame) != expected_query_rows * len(SEEDS):
                raise ValueError(f"seeded family coverage changed: {family}")
            truth_columns = [
                "source_row_id",
                "entity_id",
                "doi",
                "temperature_k",
                "pressure_kpa",
                "log_pressure",
            ]
            truth = family_frame.loc[family_frame["seed"].eq(0), truth_columns]
            medians = (
                family_frame.groupby("source_row_id", as_index=False)[
                    ["prediction_kpa", "prediction_log_pressure"]
                ]
                .median()
                .rename(
                    columns={
                        "prediction_kpa": "seed_median_prediction_kpa",
                        "prediction_log_pressure": "median_seed_log_prediction_diagnostic",
                    }
                )
            )
            aggregate = truth.merge(medians, on="source_row_id", validate="one_to_one")
            aggregate["family"] = family
            aggregate["prediction_kpa"] = aggregate.pop("seed_median_prediction_kpa")
            aggregate["prediction_log_pressure"] = np.log(
                aggregate["prediction_kpa"].to_numpy(float)
            )
            aggregate_frames.append(aggregate)
        for family in FIXED_FAMILIES:
            fixed = per_seed_predictions.loc[
                per_seed_predictions["family"].eq(family)
            ].copy()
            if len(fixed) != expected_query_rows or set(fixed["seed"]) != {-1}:
                raise ValueError(f"fixed family coverage changed: {family}")
            aggregate_frames.append(fixed.drop(columns="seed"))
        aggregate_predictions = pd.concat(aggregate_frames, ignore_index=True)
        if not np.isfinite(
            aggregate_predictions[["prediction_kpa", "prediction_log_pressure"]].to_numpy(float)
        ).all() or not aggregate_predictions["prediction_kpa"].gt(0.0).all():
            raise ValueError("aggregate prediction is invalid")

        per_seed_predictions.to_csv(output_root / "per_seed_query_predictions.csv", index=False)
        aggregate_predictions.to_csv(output_root / "aggregate_query_predictions.csv", index=False)
        pd.DataFrame(coordinate_rows).to_csv(output_root / "coordinates.csv", index=False)
        pd.DataFrame(fidelity_rows).to_csv(output_root / "projection_fidelity.csv", index=False)
        write_json(output_root / "source_hashes.json", source_hashes)
        summary = {
            "status": "completed_once_pending_independent_analysis",
            "single_use_consumed": True,
            "entities": 84,
            "dois": 45,
            "rows": len(data),
            "support_rows": int(data["role"].eq("support").sum()),
            "query_rows": expected_query_rows,
            "seeded_families": list(SEEDED_FAMILIES),
            "fixed_families": list(FIXED_FAMILIES),
            "query_target_input_max_difference": leakage_max,
            "query_target_perturbation_value": 1_000_000.0,
            "perturbed_inference_exactly_equal": True,
            "confirmation_targets_opened": True,
            "refit_or_selection_performed": False,
        }
        write_json(output_root / "execution_summary.json", summary)
        scientific_files = (
            "confirmation_data_used.csv",
            "per_seed_query_predictions.csv",
            "aggregate_query_predictions.csv",
            "coordinates.csv",
            "projection_fidelity.csv",
            "source_hashes.json",
            "execution_summary.json",
        )
        output_manifest = {
            "scope": "ThermoML single-use temporal confirmation raw execution",
            "nonce": nonce,
            "fixed_hashes": fixed_hashes,
            "package_neural_confirmation_claim_eligible": seal[
                "neural_confirmation_claim_eligible"
            ],
            "started_event_sha256": started_hash,
            "files": {name: sha256(output_root / name) for name in scientific_files},
        }
        write_json(output_root / "output_manifest.json", output_manifest)
        terminal = {
            "event": "completed_once_pending_independent_analysis",
            "nonce": nonce,
            "previous_event_sha256": started_hash,
            "output_manifest_sha256": sha256(output_root / "output_manifest.json"),
            "execution_summary_sha256": sha256(output_root / "execution_summary.json"),
        }
        append_receipt(receipt_path, terminal)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        append_receipt(
            receipt_path,
            {
                "event": "failed_after_single_use_consumption",
                "nonce": nonce,
                "previous_event_sha256": started_hash,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
