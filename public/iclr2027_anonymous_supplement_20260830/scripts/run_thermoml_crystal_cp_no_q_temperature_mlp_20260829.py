#!/usr/bin/env python3
"""Run one frozen crystal-Cp temperature-only no-q MLP development cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.backends.torch_mlp import build_torch_model_factory
from scripts.evaluate_thermoml_crystal_cp_baselines_20260829 import load_development_curves


PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
CONTRACT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
DATA_MANIFEST_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/manifest.json"
FORMAL_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_no_q_temperature_mlp_development_20260829"

EXPECTED_PLAN_SHA256 = "2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618"
EXPECTED_CONTRACT_SHA256 = "ec37eff5ab2c5847735e4b3d8db4098fd4db2bcbf67792e4b54a4fb8ba43ea15"
EXPECTED_DATA_SHA256 = "f73d3c676932304c8e5c21e79e7bc9c678e20c84db8d60b59a8e60feee400e4e"
EXPECTED_DATA_MANIFEST_SHA256 = "d88172109a7a244195f13dfa01516cfe6cb16cd038ca797dfd61d2d50694208a"
EXPECTED_ENTITIES = 247
EXPECTED_DOIS = 159
EXPECTED_ROWS = 23_742
EXPECTED_QUERY_ROWS = {"spread": 17_704, "prefix": 17_897, "four_support": 22_754}
REGIME_ROLES = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
HIDDEN_WIDTHS = (256, 128)
EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _check_inputs(data_path: Path) -> None:
    require(data_path == DATA_PATH.resolve(), "runner is bound to sealed crystal-Cp development data")
    for path, expected in (
        (PLAN_PATH, EXPECTED_PLAN_SHA256),
        (CONTRACT_PATH, EXPECTED_CONTRACT_SHA256),
        (DATA_PATH, EXPECTED_DATA_SHA256),
        (DATA_MANIFEST_PATH, EXPECTED_DATA_MANIFEST_SHA256),
    ):
        require(path.is_file() and sha256(path) == expected, f"sealed no-q input mismatch: {path}")


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.square(target - target.mean()).sum())
    return 1.0 - float(np.square(target - prediction).sum()) / denominator if denominator else float("nan")


def _predict(model: torch.nn.Module, temperature: np.ndarray, mean: float, std: float, target_mean: float, target_std: float, device: torch.device) -> np.ndarray:
    features = torch.tensor(((temperature.astype(np.float32) - mean) / std)[:, None], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        normalized = model(features).squeeze(1).cpu().numpy()
    return normalized.astype(float) * target_std + target_mean


def run_cell(
    fold: int,
    seed: int,
    device_name: str,
    output_root: str | Path = FORMAL_ROOT,
    data_path: str | Path = DATA_PATH,
    threads: int = 4,
    smoke: bool = False,
) -> dict[str, Any]:
    require(fold in FOLDS and seed in SEEDS, "fold and seed must be in 0..4")
    data_path, output_root = Path(data_path).resolve(), Path(output_root).resolve()
    _check_inputs(data_path)
    if not smoke:
        require(output_root == FORMAL_ROOT.resolve(), "formal cell must use frozen output root")
    require(device_name in {"cpu", "cuda"}, "device must be cpu or cuda")
    require(device_name != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable")
    cell_root = output_root / f"fold{fold}_seed{seed}"
    cell_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    resolved_device = torch.device(device_name)
    deterministic_seed = 20260829 + 100 * fold + seed
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(deterministic_seed)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    torch.use_deterministic_algorithms(True)

    data = load_development_curves(data_path)
    require(len(data) == EXPECTED_ROWS and data["entity_id"].nunique() == EXPECTED_ENTITIES and data["doi"].nunique() == EXPECTED_DOIS, "development coverage changed")
    train = data.loc[data["fold"].ne(fold)].reset_index(drop=True)
    heldout = data.loc[data["fold"].eq(fold)].reset_index(drop=True)
    require(set(train["doi"].astype(str)).isdisjoint(set(heldout["doi"].astype(str))), "outer DOI split is not disjoint")

    feature_mean = float(train["temperature_k"].mean())
    feature_std = float(train["temperature_k"].std(ddof=0))
    target_mean = float(train["cp_j_per_mol_k"].mean())
    target_std = float(train["cp_j_per_mol_k"].std(ddof=0))
    require(feature_std > 0.0 and target_std > 0.0, "outer-training standardizer is degenerate")
    train_features = torch.tensor(((train["temperature_k"].to_numpy(np.float32) - feature_mean) / feature_std)[:, None], dtype=torch.float32, device=resolved_device)
    train_targets = torch.tensor((train["cp_j_per_mol_k"].to_numpy(np.float32) - target_mean) / target_std, dtype=torch.float32, device=resolved_device)
    entity_codes = pd.factorize(train["entity_id"], sort=True)[0]
    entity_row_counts = np.bincount(entity_codes)
    entity_balance_weights = torch.tensor(1.0 / entity_row_counts[entity_codes], dtype=torch.float32, device=resolved_device)
    model = build_torch_model_factory(HIDDEN_WIDTHS)(1).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(deterministic_seed)
    epochs = 2 if smoke else EPOCHS
    updates = 0
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(train), generator=generator)
        batch_losses = []
        visited = 0
        for start in range(0, len(train), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE].to(resolved_device)
            prediction = model(train_features[indices]).squeeze(1)
            point_loss = torch.square(prediction - train_targets[indices])
            batch_weights = entity_balance_weights[indices]
            loss = torch.sum(point_loss * batch_weights) / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            visited += len(indices)
            batch_losses.append(float(loss.detach().cpu()))
        require(visited == len(train), "epoch did not visit every outer-training row exactly once")
        history.append({"epoch": epoch + 1, "mean_batch_inverse-entity-row-weighted_standardized_physical_mse": float(np.mean(batch_losses)), "updates_completed": updates, "rows_visited": visited})
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    training_seconds = time.perf_counter() - started

    heldout_sorted = heldout.sort_values("source_row_id", kind="stable").reset_index(drop=True)
    heldout_prediction = _predict(model, heldout_sorted["temperature_k"].to_numpy(float), feature_mean, feature_std, target_mean, target_std, resolved_device)
    require(np.isfinite(heldout_prediction).all(), "no-q heldout prediction is nonfinite")
    prediction_by_row = pd.Series(heldout_prediction, index=heldout_sorted["source_row_id"].to_numpy(int))
    prediction_frames = []
    query_rows_by_regime = {}
    r2_by_regime = {}
    negative_by_regime = {}
    perturbation_by_regime = {}
    for regime, role_column in REGIME_ROLES.items():
        query = heldout.loc[heldout[role_column].eq("query")].sort_values("source_row_id", kind="stable").reset_index(drop=True)
        prediction = prediction_by_row.loc[query["source_row_id"].to_numpy(int)].to_numpy(float)
        copied = data.copy()
        perturb_mask = copied["fold"].eq(fold) & copied[role_column].eq("query")
        copied.loc[perturb_mask, "cp_j_per_mol_k"] += 1_000_000.0
        copied_path = cell_root / f"query_target_perturbed_{regime}.csv"
        copied.to_csv(copied_path, index=False)
        reloaded = load_development_curves(copied_path)
        perturbed_heldout = reloaded.loc[reloaded["fold"].eq(fold)].sort_values("source_row_id", kind="stable")
        perturbed_all_prediction = _predict(model, perturbed_heldout["temperature_k"].to_numpy(float), feature_mean, feature_std, target_mean, target_std, resolved_device)
        perturbed_by_row = pd.Series(perturbed_all_prediction, index=perturbed_heldout["source_row_id"].to_numpy(int))
        perturbed_prediction = perturbed_by_row.loc[query["source_row_id"].to_numpy(int)].to_numpy(float)
        difference = float(np.max(np.abs(prediction - perturbed_prediction)))
        require(difference == 0.0, f"{regime} query-target perturbation changed prediction")
        frame = query[["source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k"]].copy()
        frame.insert(0, "regime", regime)
        frame["prediction_cp_j_per_mol_k"] = prediction
        prediction_frames.append(frame)
        query_rows_by_regime[regime] = len(query)
        r2_by_regime[regime] = _r2(query["cp_j_per_mol_k"].to_numpy(float), prediction)
        negative_by_regime[regime] = int(np.count_nonzero(prediction < 0.0))
        perturbation_by_regime[regime] = difference
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    prediction_frame.to_csv(cell_root / "query_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(cell_root / "training_history.csv", index=False)
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "temperature_mean": feature_mean, "temperature_std": feature_std,
        "target_mean": target_mean, "target_std": target_std,
        "hidden_widths": HIDDEN_WIDTHS, "input_features": ("temperature_k",),
        "target": "cp_j_per_mol_k", "fold": fold, "seed": seed,
    }
    torch.save(checkpoint, cell_root / "checkpoint.pt")
    peak_gpu_allocated = int(torch.cuda.max_memory_allocated(resolved_device)) if resolved_device.type == "cuda" else 0
    peak_gpu_reserved = int(torch.cuda.max_memory_reserved(resolved_device)) if resolved_device.type == "cuda" else 0
    peak_process_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    expected_updates = epochs * int(np.ceil(len(train) / BATCH_SIZE))
    require(updates == expected_updates, "optimizer update count mismatch")
    summary = {
        "status": "success", "scientific_selection_eligible": not smoke,
        "fold": fold, "seed": seed, "device": device_name,
        "epochs_completed": epochs, "training_rows": len(train),
        "query_rows_by_regime": query_rows_by_regime, "query_rows_total_long_table": len(prediction_frame),
        "optimizer_updates": updates, "backward_calls": updates,
        "training_seconds": training_seconds,
        "peak_gpu_memory_allocated_bytes": peak_gpu_allocated,
        "peak_gpu_memory_reserved_bytes": peak_gpu_reserved,
        "peak_process_memory_bytes": peak_process_bytes,
        "pooled_physical_r2_by_regime": r2_by_regime,
        "negative_prediction_count_by_regime": negative_by_regime,
        "query_target_perturbation_max_prediction_difference_by_regime": perturbation_by_regime,
        "confirmation_targets_opened": False,
    }
    write_json(cell_root / "cell_summary.json", summary)
    ledger = {"status": "terminal_success", "fold": fold, "seed": seed, "epochs_completed": epochs, "optimizer_updates": updates, "query_rows_by_regime": query_rows_by_regime, "query_target_perturbation_invariant_by_regime": {regime: True for regime in REGIME_ROLES}, "confirmation_targets_opened": False}
    write_json(cell_root / "terminal_ledger.json", ledger)
    artifact_files = {path.name: sha256(path) for path in sorted(cell_root.iterdir()) if path.is_file() and path.name != "manifest.json"}
    manifest = {
        "scope": "crystal_cp_no_q_temperature_mlp_development_cell",
        "scientific_selection_eligible": not smoke, "fold": fold, "seed": seed,
        "device": device_name, "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hidden_widths": HIDDEN_WIDTHS, "epochs": epochs, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "loss": "inverse-entity-row-count-weighted_standardized_physical-target_MSE",
        "deterministic_epoch_full_training_row_pass": True,
        "plan_sha256": EXPECTED_PLAN_SHA256, "execution_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "data_sha256": EXPECTED_DATA_SHA256, "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "outer_train_entities": int(train["entity_id"].nunique()), "outer_test_entities": int(heldout["entity_id"].nunique()),
        "outer_train_dois": int(train["doi"].nunique()), "outer_test_dois": int(heldout["doi"].nunique()),
        "entity_id_input": False, "support_input": False, "q_input": False, "query_target_input": False,
        "normalizer_fit_scope": "outer_training_rows_only", "confirmation_targets_opened": False,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "cuda_available": torch.cuda.is_available()},
        "files": artifact_files,
    }
    write_json(cell_root / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=FOLDS)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_cell(args.fold, args.seed, args.device, args.output_root, args.data, args.threads, args.smoke), indent=2))


if __name__ == "__main__":
    main()
