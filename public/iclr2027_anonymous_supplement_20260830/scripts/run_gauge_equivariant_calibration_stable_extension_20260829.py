#!/usr/bin/env python3
"""Run the numerically stable affine-equivariant calibration extension.

This is a new, provenance-bound wrapper around the original cell implementation.
It keeps the source benchmark and all old outputs immutable, replacing only the
Gauss--Newton solve with float64 ``lstsq`` and the frozen 15-step acceptance
rule from ``GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = PROJECT_ROOT / "scripts/run_gauge_equivariant_calibration_extension_20260829.py"
NUMERICAL_AMENDMENT = PROJECT_ROOT / "GAUGE_EQUIVARIANT_CALIBRATION_NUMERICAL_AMENDMENT_20260829.md"
FAILED_DECISION = PROJECT_ROOT / "runs/gauge_equivariant_calibration_extension_20260829/analysis/decision.json"
FAILED_MANIFEST = PROJECT_ROOT / "runs/gauge_equivariant_calibration_extension_20260829/analysis/manifest.json"
FORMAL_ROOT = PROJECT_ROOT / "runs/gauge_equivariant_calibration_stable_extension_20260829"
ADAM_STEPS = 300
GN_STEPS = 15
GAUGE_COUNT = 5
ENTITY_COUNT = 48
PERTURBATION = 1_000_000.0
EXPECTED_NUMERICAL_AMENDMENT_SHA256 = "d85db0c6d9a5b332aa3499eb9d3f105a2e89a4674b30fe90db0657bd26006613"
EXPECTED_BASE_RUNNER_SHA256 = "13f9d21d1525582a2bb874add150bf5679f41642486b62dbbf8c63a2e3286024"
EXPECTED_FAILED_DECISION_SHA256 = "2214a5ff161573d2d9ba767e1d8dd60134ab536500979dd068c59dd4038d49f0"
EXPECTED_FAILED_MANIFEST_SHA256 = "17ce41703ea7d305d598dc14d877d28c29580f057c1f2bd23b620ab5152e4fc4"
PERTURBATION_DETAILS_FILE = "query_target_perturbation_diagnostics.csv"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("stable_base_extension", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import base extension: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
FAMILIES = BASE.FAMILIES
SEEDS = BASE.SEEDS
Q_DIM = BASE.Q_DIM


def verify_provenance() -> None:
    if sha256(NUMERICAL_AMENDMENT) != EXPECTED_NUMERICAL_AMENDMENT_SHA256:
        raise RuntimeError("numerical amendment hash mismatch")
    if sha256(BASE_RUNNER_PATH) != EXPECTED_BASE_RUNNER_SHA256:
        raise RuntimeError("base extension runner hash mismatch")
    if sha256(FAILED_DECISION) != EXPECTED_FAILED_DECISION_SHA256:
        raise RuntimeError("failed extension decision hash mismatch")
    if sha256(FAILED_MANIFEST) != EXPECTED_FAILED_MANIFEST_SHA256:
        raise RuntimeError("failed extension analysis manifest hash mismatch")


def calibrate_gauss_newton(
    model: Any,
    initial_q: np.ndarray,
    support_x: np.ndarray,
    support_y_norm: np.ndarray,
    x_mean: float,
    x_std: float,
    steps: int = GN_STEPS,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    """Float64 direct least-squares Gauss--Newton with the frozen line search."""
    q = np.asarray(initial_q, dtype=np.float64).copy()
    path: list[dict[str, float]] = []
    line_search = [1.0 / (2**index) for index in range(8)]
    for iteration in range(steps):
        residual, jacobian = BASE._gn_residual_jacobian(
            model, q, support_x, support_y_norm, x_mean, x_std
        )
        residual = np.asarray(residual, dtype=np.float64)
        jacobian = np.asarray(jacobian, dtype=np.float64)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian, tol=1e-12))
        if rank != Q_DIM:
            raise np.linalg.LinAlgError(
                f"Gauss-Newton support Jacobian is rank deficient at iteration {iteration}"
            )
        delta = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        before = float(np.mean(residual**2))
        tolerance = 1e-12 * max(1.0, before)
        accepted_scale = 0.0
        after = before
        for scale in line_search:
            candidate = q + scale * delta
            candidate_loss = BASE._support_loss(
                model, support_x, support_y_norm, candidate, x_mean, x_std
            )
            if candidate_loss < before - tolerance:
                q = candidate
                accepted_scale = scale
                after = candidate_loss
                break
        path.append({
            "iteration": float(iteration),
            "loss": before,
            "loss_after": after,
            "step_scale": accepted_scale,
            "jacobian_rank": float(rank),
            "jacobian_sigma_min": float(singular_values[-1]),
            "jacobian_sigma_max": float(singular_values[0]),
            "jacobian_condition": float(singular_values[0] / singular_values[-1]),
            "delta_norm": float(np.linalg.norm(delta)),
            "acceptance_tolerance": tolerance,
        })
    return q, BASE._support_loss(model, support_x, support_y_norm, q, x_mean, x_std), path


def _write_perturbation_details(
    cell: Path,
    family: str,
    seed: int,
    call_records: list[dict[str, Any]],
    *,
    adam_steps: int,
    gn_steps: int,
    gauge_count: int,
    entity_limit: int,
) -> None:
    """Persist the raw original/perturbed calibration paths and predictions.

    The base runner already computes this twin experiment for its summary.  The
    wrappers capture those exact returned objects, so this file is a serialized
    audit of the same calls rather than a fabricated zero-difference result.
    Query targets are intentionally absent from this file.
    """
    source_artifact, _, _ = BASE.verify_source_terminal(family, seed)
    artifact = BASE.torch.load(source_artifact, map_location="cpu", weights_only=False)
    data = BASE.SOURCE.generate_family_data(family, seed)
    x = data["x"]
    x_mean, x_std = float(artifact["x_mean"]), float(artifact["x_std"])
    y_mean, y_std = float(artifact["y_mean"]), float(artifact["y_std"])
    support_positions, query_positions = BASE.SOURCE.support_query_indices()
    test_targets = data["targets"][BASE.SOURCE.TRAIN_ENTITIES:]
    entity_count = min(entity_limit, BASE.SOURCE.TEST_ENTITIES)
    probe_x = np.linspace(float(x.min()), float(x.max()), BASE.SOURCE.PROBE_SIZE)
    gauges = artifact["gauges"][:gauge_count]
    charts = [("original", -1)] + [(f"gauge_{int(gauge['gauge_id'])}", int(gauge["gauge_id"])) for gauge in gauges]
    expected_main = len(charts) * entity_count * 2
    expected_tail = len(charts) * entity_count * 4
    if len(call_records) != expected_main + expected_tail:
        raise RuntimeError(f"captured calibration-call coverage mismatch: {len(call_records)}")
    rows: list[dict[str, Any]] = []
    tail = call_records[expected_main:]
    tail_index = 0
    for chart_index, (chart, gauge_id) in enumerate(charts):
        for entity_id in range(entity_count):
            support_x = x[support_positions]
            query_x = x[query_positions]
            # The perturbation loop in the base runner calls original and
            # perturbed targets in Adam-then-GN order for each entity.
            paired: dict[tuple[str, str], dict[str, Any]] = {}
            for method in ("mapped_start_adam", "response_metric_gauss_newton"):
                original = tail[tail_index]
                perturbed = tail[tail_index + 1]
                tail_index += 2
                if original["method"] != method or perturbed["method"] != method:
                    raise RuntimeError("captured perturbation method order mismatch")
                paired[(method, "original")] = original
                paired[(method, "perturbed")] = perturbed
            for method, steps in (("mapped_start_adam", adam_steps), ("response_metric_gauss_newton", gn_steps)):
                original = paired[(method, "original")]
                perturbed = paired[(method, "perturbed")]
                if len(original["path"]) != steps or len(perturbed["path"]) != steps:
                    raise RuntimeError("captured perturbation path length mismatch")
                for before, after in zip(original["path"], perturbed["path"]):
                    row: dict[str, Any] = {
                        "record_type": "path", "method": method, "chart": chart,
                        "gauge_id": gauge_id, "entity_id": entity_id,
                        "iteration": int(before["iteration"]), "query_position": np.nan,
                        "x": np.nan,
                    }
                    for field in ("loss", "loss_after", "step_scale", "jacobian_rank", "jacobian_sigma_min", "jacobian_sigma_max", "jacobian_condition", "delta_norm", "acceptance_tolerance"):
                        row[f"{field}_original"] = before.get(field, np.nan)
                        row[f"{field}_perturbed"] = after.get(field, np.nan)
                        row[f"{field}_abs_difference"] = abs(float(row[f"{field}_original"]) - float(row[f"{field}_perturbed"])) if np.isfinite(float(row[f"{field}_original"])) and np.isfinite(float(row[f"{field}_perturbed"])) else np.nan
                    rows.append(row)
                original_q = np.asarray(original["q"], dtype=float)
                perturbed_q = np.asarray(perturbed["q"], dtype=float)
                original_c, _ = BASE._chart_functional(original["model"], original_q, probe_x, family, x_mean, x_std, y_mean, y_std)
                perturbed_c, _ = BASE._chart_functional(perturbed["model"], perturbed_q, probe_x, family, x_mean, x_std, y_mean, y_std)
                row = {
                    "record_type": "calibration", "method": method, "chart": chart,
                    "gauge_id": gauge_id, "entity_id": entity_id,
                    "iteration": np.nan, "query_position": np.nan, "x": np.nan,
                }
                for index in range(BASE.Q_DIM):
                    row[f"q{index}_original"] = original_q[index]
                    row[f"q{index}_perturbed"] = perturbed_q[index]
                    row[f"q{index}_abs_difference"] = abs(original_q[index] - perturbed_q[index])
                    row[f"functional_c{index}_original"] = original_c[index]
                    row[f"functional_c{index}_perturbed"] = perturbed_c[index]
                    row[f"functional_c{index}_abs_difference"] = abs(original_c[index] - perturbed_c[index])
                rows.append(row)
                original_prediction = BASE._normalized_prediction(original["model"], query_x, original_q, x_mean, x_std) * y_std + y_mean
                perturbed_prediction = BASE._normalized_prediction(perturbed["model"], query_x, perturbed_q, x_mean, x_std) * y_std + y_mean
                original_functional = BASE._chart_functional(original["model"], original_q, query_x, family, x_mean, x_std, y_mean, y_std)[0]
                perturbed_functional = BASE._chart_functional(perturbed["model"], perturbed_q, query_x, family, x_mean, x_std, y_mean, y_std)[0]
                basis_query = BASE.SOURCE.family_basis(family, query_x)
                for position, x_value, raw_a, raw_b, func_a, func_b in zip(query_positions, query_x, original_prediction, perturbed_prediction, basis_query @ original_functional, basis_query @ perturbed_functional):
                    rows.append({
                        "record_type": "prediction", "method": method, "chart": chart,
                        "gauge_id": gauge_id, "entity_id": entity_id,
                        "iteration": np.nan, "query_position": int(position), "x": float(x_value),
                        "prediction_original": float(raw_a), "prediction_perturbed": float(raw_b),
                        "prediction_abs_difference": float(abs(raw_a - raw_b)),
                        "functional_prediction_original": float(func_a), "functional_prediction_perturbed": float(func_b),
                        "functional_prediction_abs_difference": float(abs(func_a - func_b)),
                    })
    if tail_index != len(tail):
        raise RuntimeError("captured perturbation calls were not fully consumed")
    pd.DataFrame(rows).to_csv(cell / PERTURBATION_DETAILS_FILE, index=False)


def _rewrite_cell_provenance(cell: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Replace wrapper metadata after the base writes the shared CSV schema."""
    result_path = cell / "result.json"
    manifest_path = cell / "manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update({
        "scope": "gauge_equivariant_calibration_stable_extension_cell",
        "numerical_amendment_sha256": sha256(NUMERICAL_AMENDMENT),
        "base_runner_sha256": sha256(BASE_RUNNER_PATH),
        "stable_solver": "float64_lstsq",
        "gn_acceptance_tolerance": "1e-12*max(1,current_loss)",
        "failed_extension_decision_sha256": sha256(FAILED_DECISION),
        "failed_extension_analysis_manifest_sha256": sha256(FAILED_MANIFEST),
    })
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "scope": "gauge_equivariant_calibration_stable_extension_cell",
        "numerical_amendment_sha256": sha256(NUMERICAL_AMENDMENT),
        "base_runner_sha256": sha256(BASE_RUNNER_PATH),
        "failed_extension_decision_sha256": sha256(FAILED_DECISION),
        "failed_extension_analysis_manifest_sha256": sha256(FAILED_MANIFEST),
        "stable_solver": "float64_lstsq",
        "gn_acceptance_tolerance": "1e-12*max(1,current_loss)",
        "runner_sha256": sha256(Path(__file__)),
        "files": {},
    })
    manifest["files"] = {
        path.name: sha256(path)
        for path in sorted(cell.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_cell(
    family: str,
    seed: int,
    output_root: Path = FORMAL_ROOT,
    *,
    adam_steps: int = ADAM_STEPS,
    gn_steps: int = GN_STEPS,
    gauge_count: int = GAUGE_COUNT,
    entity_limit: int = ENTITY_COUNT,
    threads: int = 1,
    smoke: bool = False,
) -> dict[str, object]:
    verify_provenance()
    BASE.AMENDMENT = NUMERICAL_AMENDMENT
    BASE.EXPECTED_AMENDMENT_SHA256 = EXPECTED_NUMERICAL_AMENDMENT_SHA256
    # Eligibility is tied to this exact frozen formal root; a caller-supplied
    # output directory can never become formal merely by matching the budgets.
    BASE.FORMAL_ROOT = FORMAL_ROOT.resolve()
    BASE.GN_STEPS = GN_STEPS
    original_adam = BASE.calibrate_adam
    original_gn = BASE.calibrate_gauss_newton
    call_records: list[dict[str, Any]] = []

    def capture(method: str, calibrator: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = calibrator(*args, **kwargs)
            call_records.append({
                "method": method, "model": args[0],
                "q": np.asarray(result[0], dtype=float).copy(),
                "support_loss": float(result[1]), "path": result[2],
            })
            return result
        return wrapped

    BASE.calibrate_adam = capture("mapped_start_adam", original_adam)
    BASE.calibrate_gauss_newton = capture("response_metric_gauss_newton", calibrate_gauss_newton)
    try:
        summary = BASE.run_cell(
            family,
            seed,
            Path(output_root),
            adam_steps=adam_steps,
            gn_steps=gn_steps,
            gauge_count=gauge_count,
            entity_limit=entity_limit,
            threads=threads,
            smoke=smoke,
        )
    finally:
        BASE.calibrate_adam = original_adam
        BASE.calibrate_gauss_newton = original_gn
    cell = Path(output_root).resolve() / f"{family}_seed{seed}"
    _write_perturbation_details(
        cell, family, seed, call_records, adam_steps=adam_steps, gn_steps=gn_steps,
        gauge_count=gauge_count, entity_limit=entity_limit,
    )
    rewritten = _rewrite_cell_provenance(cell, summary)
    return rewritten


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--adam-steps", type=int, default=ADAM_STEPS)
    parser.add_argument("--gn-steps", type=int, default=GN_STEPS)
    parser.add_argument("--gauge-count", type=int, default=GAUGE_COUNT)
    parser.add_argument("--entity-limit", type=int, default=ENTITY_COUNT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.adam_steps = min(args.adam_steps, 2)
        args.gn_steps = min(args.gn_steps, 2)
        args.gauge_count = min(args.gauge_count, 1)
        args.entity_limit = min(args.entity_limit, 2)
    run_cell(
        args.family,
        args.seed,
        args.output_root,
        adam_steps=args.adam_steps,
        gn_steps=args.gn_steps,
        gauge_count=args.gauge_count,
        entity_limit=args.entity_limit,
        threads=args.threads,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
