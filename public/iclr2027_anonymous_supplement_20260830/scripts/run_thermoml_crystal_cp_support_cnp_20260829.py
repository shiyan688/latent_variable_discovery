#!/usr/bin/env python3
"""Run one target-blind ThermoML crystal-Cp support-CNP cell.

This file is deliberately a one-cell runner.  It trains a permutation-invariant
DeepSets conditional predictor on outer-training entities and evaluates the
three frozen support regimes on the held-out DOI fold.  It never reads the
reserved temporal confirmation cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLAN_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_TARGET_BLIND_PLAN_20260829.md"
CONTRACT_PATH = PROJECT_ROOT / "THERMOML_CRYSTAL_CP_EXECUTION_CONTRACT_20260829.md"
DATA_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/development_curves.csv"
DATA_MANIFEST_PATH = PROJECT_ROOT / "runs/thermoml_crystal_cp_development_data_20260829/manifest.json"
FORMAL_ROOT = PROJECT_ROOT / "runs/thermoml_crystal_cp_support_cnp_development_20260829"

EXPECTED_PLAN_SHA256 = "2ae03f71e6ffe9cfee3df0a61c8c7e49e9777268d0d9ccb6f1da8538e2203618"
EXPECTED_CONTRACT_SHA256 = "ec37eff5ab2c5847735e4b3d8db4098fd4db2bcbf67792e4b54a4fb8ba43ea15"
EXPECTED_DATA_SHA256 = "f73d3c676932304c8e5c21e79e7bc9c678e20c84db8d60b59a8e60feee400e4e"
EXPECTED_DATA_MANIFEST_SHA256 = "d88172109a7a244195f13dfa01516cfe6cb16cd038ca797dfd61d2d50694208a"
EXPECTED_ROWS, EXPECTED_ENTITIES, EXPECTED_DOIS = 23_742, 247, 159
FOLDS = tuple(range(5))
SEEDS = tuple(range(5))
REGIMES = ("spread", "prefix", "four_support")
SPREAD_OFFSETS = tuple(range(4))
ENCODER_WIDTHS = (128, 128)
QUERY_HEAD_WIDTHS = (256, 128)
EPOCHS, ENTITY_BATCH_SIZE, LEARNING_RATE = 1000, 16, 1e-3


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
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class SupportCNP(nn.Module):
    """DeepSets support encoder followed by a conditional query head."""

    def __init__(self) -> None:
        super().__init__()
        self.set_encoder = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.query_head = nn.Sequential(
            nn.Linear(1 + 128, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        require(support_x.ndim == 2 and support_x.shape[1] == 1, "support_x shape mismatch")
        require(support_y.ndim == 1 and len(support_x) == len(support_y), "support pair shape mismatch")
        require(query_x.ndim == 2 and query_x.shape[1] == 1 and len(query_x) > 0, "query_x shape mismatch")
        representation = self.set_encoder(torch.cat((support_x, support_y[:, None]), dim=1)).mean(dim=0)
        repeated = representation.expand(len(query_x), -1)
        return self.query_head(torch.cat((query_x, repeated), dim=1)).squeeze(1)


def _load_data(path: Path, smoke: bool) -> pd.DataFrame:
    data = pd.read_csv(path).sort_values(["entity_id", "position"], kind="stable").reset_index(drop=True)
    required = {"source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k", "spread_role", "prefix_role", "four_role"}
    require(required <= set(data), "development data schema mismatch")
    if not smoke:
        require((len(data), data.entity_id.nunique(), data.doi.nunique()) == (EXPECTED_ROWS, EXPECTED_ENTITIES, EXPECTED_DOIS), "sealed development coverage changed")
    require(np.isfinite(data[["temperature_k", "cp_j_per_mol_k"]].to_numpy()).all(), "non-finite development response")
    require(data.temperature_k.gt(0).all(), "temperatures must be positive")
    require(set(data.fold.unique()) == set(FOLDS), "development folds changed")
    require(data.groupby("doi").fold.nunique().max() == 1, "DOI crosses outer folds")
    return data


def _support_positions(curve: pd.DataFrame, regime: str, spread_offset: int = 0) -> np.ndarray:
    positions = curve.position.to_numpy(dtype=int)
    if regime == "spread":
        require(spread_offset in SPREAD_OFFSETS, "spread offset must be 0..3")
        result = positions[positions % 4 == spread_offset]
    elif regime == "prefix":
        require(spread_offset == 0, "prefix has no offset")
        result = positions[: max(5, len(positions) // 4)]
    elif regime == "four_support":
        require(spread_offset == 0, "four-support has no offset")
        result = np.asarray([0, round((len(positions) - 1) / 3), round(2 * (len(positions) - 1) / 3), len(positions) - 1], dtype=int)
    else:
        raise ValueError(regime)
    require(len(result) >= 4 and len(np.unique(result)) == len(result), f"invalid {regime} support")
    return result


def _seed(seed: int, fold: int) -> int:
    return 20260829 + 100 * fold + seed


def _standardizer(train: pd.DataFrame) -> dict[str, float]:
    values = {"temperature_mean": float(train.temperature_k.mean()), "temperature_std": float(train.temperature_k.std(ddof=0)), "target_mean": float(train.cp_j_per_mol_k.mean()), "target_std": float(train.cp_j_per_mol_k.std(ddof=0))}
    require(values["temperature_std"] > 0 and values["target_std"] > 0, "degenerate outer-training standardizer")
    return values


def _episode(curve: pd.DataFrame, epoch: int, stable_index: int) -> tuple[np.ndarray, np.ndarray, str, int]:
    cycle = epoch % 3
    if cycle == 0:
        offset = (epoch // 3 + stable_index) % 4
        regime = "spread"
    elif cycle == 1:
        offset, regime = 0, "prefix"
    else:
        offset, regime = 0, "four_support"
    support = _support_positions(curve, regime, offset)
    query = np.setdiff1d(curve.position.to_numpy(dtype=int), support, assume_unique=True)
    return support, query, regime, offset


def _normalized(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def _predict(model: SupportCNP, support_t: np.ndarray, support_y: np.ndarray, query_t: np.ndarray, stats: dict[str, float], device: torch.device) -> np.ndarray:
    sx = torch.tensor(_normalized(support_t, stats["temperature_mean"], stats["temperature_std"]), device=device)[:, None]
    sy = torch.tensor(_normalized(support_y, stats["target_mean"], stats["target_std"]), device=device)
    qx = torch.tensor(_normalized(query_t, stats["temperature_mean"], stats["temperature_std"]), device=device)[:, None]
    model.eval()
    with torch.no_grad():
        prediction = model(sx, sy, qx).cpu().numpy()
    return prediction.astype(float) * stats["target_std"] + stats["target_mean"]


def _check_inputs(data_path: Path, smoke: bool) -> None:
    for path, expected in ((PLAN_PATH, EXPECTED_PLAN_SHA256), (CONTRACT_PATH, EXPECTED_CONTRACT_SHA256)):
        require(path.is_file() and sha256(path) == expected, f"frozen protocol mismatch: {path}")
    if smoke:
        return
    require(data_path == DATA_PATH.resolve(), "formal CNP must use sealed development CSV")
    require(sha256(DATA_PATH) == EXPECTED_DATA_SHA256 and sha256(DATA_MANIFEST_PATH) == EXPECTED_DATA_MANIFEST_SHA256, "sealed development input changed")
    manifest = json.loads(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest["confirmation_source_files_opened"] is False and manifest["confirmation_targets_opened"] is False, "confirmation access reported")


def run_cell(fold: int, seed: int, device_name: str = "cuda", output_root: str | Path = FORMAL_ROOT, data_path: str | Path = DATA_PATH, threads: int = 4, smoke: bool = False) -> dict[str, Any]:
    require(fold in FOLDS and seed in SEEDS, "fold and seed must be in 0..4")
    output_root, data_path = Path(output_root).resolve(), Path(data_path).resolve()
    _check_inputs(data_path, smoke)
    if not smoke:
        require(output_root == FORMAL_ROOT.resolve(), "formal CNP output root changed")
    require(device_name in {"cpu", "cuda"} and (device_name != "cuda" or torch.cuda.is_available()), "requested device unavailable")
    cell_root = output_root / f"fold{fold}_seed{seed}"
    cell_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    deterministic_seed = _seed(seed, fold)
    np.random.seed(deterministic_seed)
    torch.manual_seed(deterministic_seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(deterministic_seed)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    device = torch.device(device_name)
    data = _load_data(data_path, smoke)
    train = data.loc[data.fold.ne(fold)].reset_index(drop=True)
    heldout = data.loc[data.fold.eq(fold)].reset_index(drop=True)
    require(set(train.doi).isdisjoint(set(heldout.doi)), "outer DOI split is not disjoint")
    stats = _standardizer(train)
    model = SupportCNP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    groups = {str(entity): group.copy() for entity, group in train.groupby("entity_id", sort=True)}
    stable_indices = {entity: index for index, entity in enumerate(groups)}
    epochs = 2 if smoke else EPOCHS
    history: list[dict[str, Any]] = []
    optimizer_updates = 0
    backward_calls = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        rng = np.random.default_rng(np.random.SeedSequence([deterministic_seed, epoch, 271828]))
        entities = list(rng.permutation(np.asarray(list(groups), dtype=object)))
        visited, epoch_losses = 0, []
        for batch_start in range(0, len(entities), ENTITY_BATCH_SIZE):
            batch = entities[batch_start : batch_start + ENTITY_BATCH_SIZE]
            losses = []
            for entity in batch:
                curve = groups[str(entity)]
                support, query, _, _ = _episode(curve, epoch, stable_indices[str(entity)])
                support_frame = curve.loc[curve.position.isin(support)].sort_values("position")
                query_frame = curve.loc[curve.position.isin(query)].sort_values("position")
                sx = torch.tensor(_normalized(support_frame.temperature_k.to_numpy(), stats["temperature_mean"], stats["temperature_std"]), device=device)[:, None]
                sy = torch.tensor(_normalized(support_frame.cp_j_per_mol_k.to_numpy(), stats["target_mean"], stats["target_std"]), device=device)
                qx = torch.tensor(_normalized(query_frame.temperature_k.to_numpy(), stats["temperature_mean"], stats["temperature_std"]), device=device)[:, None]
                qy = torch.tensor(_normalized(query_frame.cp_j_per_mol_k.to_numpy(), stats["target_mean"], stats["target_std"]), device=device)
                losses.append(torch.mean((model(sx, sy, qx) - qy) ** 2))
                visited += len(query_frame)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            backward_calls += 1
            epoch_losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "mean_per_entity_query_standardized_mse": float(np.mean(epoch_losses)), "entities_visited": len(entities), "query_examples_visited": visited, "optimizer_updates_completed": optimizer_updates, "backward_calls_completed": backward_calls})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started

    prediction_rows: list[pd.DataFrame] = []
    offset_prediction_rows: list[pd.DataFrame] = []
    query_rows_by_regime: dict[str, int] = {}
    query_perturbation: dict[str, float] = {}
    r2_by_regime: dict[str, float] = {}
    all_heldout = {str(entity): group.copy() for entity, group in heldout.groupby("entity_id", sort=True)}
    for regime in REGIMES:
        offsets = SPREAD_OFFSETS if regime == "spread" else (0,)
        for offset in offsets:
            parts = []
            for entity, curve in all_heldout.items():
                support = _support_positions(curve, regime, offset)
                query = np.setdiff1d(curve.position.to_numpy(dtype=int), support, assume_unique=True)
                support_frame = curve.loc[curve.position.isin(support)].sort_values("position")
                query_frame = curve.loc[curve.position.isin(query)].sort_values("position")
                prediction = _predict(model, support_frame.temperature_k.to_numpy(), support_frame.cp_j_per_mol_k.to_numpy(), query_frame.temperature_k.to_numpy(), stats, device)
                if regime == "spread":
                    all_prediction = _predict(model, support_frame.temperature_k.to_numpy(), support_frame.cp_j_per_mol_k.to_numpy(), curve.temperature_k.to_numpy(), stats, device)
                    offset_prediction_rows.append(pd.DataFrame({"source_row_id": curve.source_row_id.to_numpy(int), "entity_id": curve.entity_id.to_numpy(), "fold": curve.fold.to_numpy(int), "spread_offset": offset, "temperature_k": curve.temperature_k.to_numpy(float), "prediction_cp_j_per_mol_k": all_prediction}))
                frame = query_frame[["source_row_id", "entity_id", "doi", "fold", "position", "temperature_k", "cp_j_per_mol_k"]].copy()
                frame.insert(0, "regime", regime)
                frame.insert(1, "spread_offset", offset)
                frame["prediction_cp_j_per_mol_k"] = prediction
                parts.append(frame)
            scored = pd.concat(parts, ignore_index=True).sort_values("source_row_id", kind="stable")
            prediction_rows.append(scored)
            if offset == 0:
                query_rows_by_regime[regime] = len(scored)
                target, pred = scored.cp_j_per_mol_k.to_numpy(float), scored.prediction_cp_j_per_mol_k.to_numpy(float)
                denominator = float(np.square(target - target.mean()).sum())
                r2_by_regime[regime] = 1.0 - float(np.square(target - pred).sum()) / denominator if denominator else float("nan")
                perturbed = data.copy()
                role_column = {"spread": "spread_role", "prefix": "prefix_role", "four_support": "four_role"}[regime]
                mask = perturbed.fold.eq(fold) & perturbed[role_column].eq("query")
                perturbed.loc[mask, "cp_j_per_mol_k"] += 1_000_000.0
                perturbed.to_csv(cell_root / f"query_target_perturbed_{regime}.csv", index=False)
                # The model only receives support targets.  Re-evaluate on the perturbed copy
                # and compare by source row, making the leakage audit explicit at generation.
                perturbed_rows = []
                for entity, curve in perturbed.loc[perturbed.fold.eq(fold)].groupby("entity_id", sort=True):
                    support = _support_positions(curve, regime, offset)
                    query = np.setdiff1d(curve.position.to_numpy(dtype=int), support, assume_unique=True)
                    sf = curve.loc[curve.position.isin(support)].sort_values("position")
                    qf = curve.loc[curve.position.isin(query)].sort_values("position")
                    pp = _predict(model, sf.temperature_k.to_numpy(), sf.cp_j_per_mol_k.to_numpy(), qf.temperature_k.to_numpy(), stats, device)
                    perturbed_rows.append(pd.DataFrame({"source_row_id": qf.source_row_id.to_numpy(), "prediction": pp}))
                perturbed_table = pd.concat(perturbed_rows).set_index("source_row_id").sort_index()
                original_table = scored.set_index("source_row_id").sort_index()
                perturbed_prediction = perturbed_table.prediction.to_numpy(float)
                query_perturbation[regime] = float(np.max(np.abs(original_table.prediction_cp_j_per_mol_k.to_numpy(float) - perturbed_prediction)))
                require(query_perturbation[regime] == 0.0, f"{regime} query-target perturbation changed prediction")
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_csv(cell_root / "query_predictions.csv", index=False)
    pd.concat(offset_prediction_rows, ignore_index=True).sort_values(["spread_offset", "source_row_id"], kind="stable").to_csv(cell_root / "spread_offset_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(cell_root / "training_history.csv", index=False)
    checkpoint = {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "normalizer": stats, "encoder_widths": ENCODER_WIDTHS, "query_head_widths": QUERY_HEAD_WIDTHS, "fold": fold, "seed": seed, "target": "cp_j_per_mol_k", "input_features": ("temperature_k", "cp_j_per_mol_k"), "confirmation_targets_opened": False}
    torch.save(checkpoint, cell_root / "checkpoint.pt")
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    expected_updates = epochs * int(math.ceil(len(groups) / ENTITY_BATCH_SIZE))
    require(optimizer_updates == expected_updates and backward_calls == expected_updates, "CNP optimizer/backward count mismatch")
    summary = {"status": "success", "scientific_selection_eligible": not smoke, "fold": fold, "seed": seed, "device": device_name, "epochs_completed": epochs, "outer_training_entities": len(groups), "training_rows": len(train), "query_rows_by_regime": query_rows_by_regime, "query_rows_total_long_table": len(predictions), "optimizer_updates": optimizer_updates, "backward_calls": backward_calls, "entities_per_batch": ENTITY_BATCH_SIZE, "training_seconds": training_seconds, "peak_gpu_memory_allocated_bytes": peak_allocated, "peak_gpu_memory_reserved_bytes": peak_reserved, "peak_process_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "pooled_physical_r2_by_regime": r2_by_regime, "query_target_perturbation_max_prediction_difference_by_regime": query_perturbation, "confirmation_targets_opened": False}
    write_json(cell_root / "cell_summary.json", summary)
    write_json(cell_root / "terminal_ledger.json", {"status": "terminal_success", "fold": fold, "seed": seed, "optimizer_updates": optimizer_updates, "backward_calls": backward_calls, "query_rows_by_regime": query_rows_by_regime, "query_target_perturbation_invariant_by_regime": {key: value == 0.0 for key, value in query_perturbation.items()}, "confirmation_targets_opened": False})
    files = {p.name: sha256(p) for p in cell_root.iterdir() if p.is_file() and p.name != "manifest.json"}
    write_json(cell_root / "manifest.json", {"scope": "thermoml_crystal_cp_support_cnp_development_cell", "scientific_selection_eligible": not smoke, "fold": fold, "seed": seed, "device": device_name, "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "encoder_widths": ENCODER_WIDTHS, "query_head_widths": QUERY_HEAD_WIDTHS, "epochs": epochs, "entity_batch_size": ENTITY_BATCH_SIZE, "learning_rate": LEARNING_RATE, "episode_schedule": {"epoch_mod_3_0": "spread_offset_epoch_div_3_plus_stable_entity_index_mod_4", "epoch_mod_3_1": "prefix", "epoch_mod_3_2": "four_support"}, "loss": "mean_per_entity_query_standardized_MSE", "plan_sha256": EXPECTED_PLAN_SHA256, "execution_contract_sha256": EXPECTED_CONTRACT_SHA256, "data_sha256": EXPECTED_DATA_SHA256 if not smoke else sha256(data_path), "data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256 if not smoke else sha256(DATA_MANIFEST_PATH), "runner_sha256": sha256(Path(__file__).resolve()), "normalizer_fit_scope": "outer_training_rows_only", "entity_id_input": False, "support_input": True, "query_target_input": False, "confirmation_targets_opened": False, "files": files})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_cell(args.fold, args.seed, args.device, args.output_root, args.data, args.threads, args.smoke), indent=2))


if __name__ == "__main__":
    main()
