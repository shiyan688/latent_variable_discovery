#!/usr/bin/env python3
"""Run the frozen reviewer-clean NASA inner symbolic-interface matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = PROJECT_ROOT / "NASA_INNER_SYMBOLIC_STRUCTURE_PLAN_20260825.md"
BASELINE_REGIMES = ("condition_only", "condition_support_stats")
Q_REGIMES = ("condition_raw_q", "condition_functional_q")
SUPPORT_COLUMNS = (
    "support_y_mean",
    "support_y_std",
    "support_y_min",
    "support_y_max",
    "support_y_count",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _csv_arg(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _limit_threads(threads: int) -> None:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "JULIA_NUM_THREADS",
    ):
        os.environ[key] = str(threads)


def _source_path(q_root: Path, dataset: str, method: str, seed: int) -> Path:
    matches = sorted((q_root / dataset / method).glob(f"seed{seed}_q4_*/result.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one source for {dataset}, {method}, seed {seed}; found {matches}"
        )
    return matches[0]


def _prepared_records(q_root: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((q_root / "experiment_manifest.json").read_text())
    if len(manifest["prepared_summaries"]) != 1:
        raise ValueError("expected one prepared summary")
    records = json.loads(_resolve(manifest["prepared_summaries"][0]).read_text())
    return {record["name"]: record for record in records}


def _prefix_split(
    frame: pd.DataFrame,
    support_ratio: float,
    order_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(["label", order_column], kind="stable").reset_index(drop=True)
    labels = ordered.label.to_numpy()
    support_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for label in pd.unique(labels):
        indices = np.flatnonzero(labels == label)
        split_point = max(1, int(np.floor(support_ratio * len(indices))))
        split_point = min(split_point, len(indices) - 1)
        support_parts.append(indices[:split_point])
        query_parts.append(indices[split_point:])
    return ordered, np.concatenate(support_parts), np.concatenate(query_parts)


def _support_statistics(
    frame: pd.DataFrame,
    support_indices: np.ndarray,
) -> pd.DataFrame:
    support = frame.iloc[support_indices]
    return (
        support.groupby("label", sort=False).target
        .agg(
            support_y_mean="mean",
            support_y_std=lambda values: float(values.std(ddof=0)),
            support_y_min="min",
            support_y_max="max",
            support_y_count="count",
        )
        .reset_index()
    )


def _q_features(
    source_path: Path,
    q_root: Path,
    functional_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    payload = json.loads(source_path.read_text())
    train_q = pd.read_csv(_resolve(payload["artifacts"]["train_label_q"]))
    test_q = pd.read_csv(_resolve(payload["artifacts"]["test_label_q"]))
    q_columns = [column for column in train_q if column.startswith("q") and column[1:].isdigit()]
    if q_columns != ["q1", "q2", "q3", "q4"]:
        raise ValueError(f"unexpected q columns in {source_path}: {q_columns}")
    test_q = test_q.loc[:, ["label", *q_columns]]

    relative = source_path.relative_to(q_root)
    functional_path = q_root / "functional_coordinate_analysis" / relative.parent / "functional_coordinates.csv"
    functional = pd.read_csv(functional_path)
    train_functional = functional.loc[
        functional.split == "train", ["label", *functional_columns]
    ]
    test_functional = functional.loc[
        functional.split == "outer_test", ["label", *functional_columns]
    ]
    train_features = train_q.merge(train_functional, on="label", validate="one_to_one")
    test_features = test_q.merge(test_functional, on="label", validate="one_to_one")
    return train_features, test_features, payload


def _build_bundle(
    *,
    q_root: Path,
    records: dict[str, dict[str, Any]],
    dataset: str,
    seed: int,
    method: str | None,
    condition_columns: list[str],
    functional_columns: list[str],
    support_ratio: float,
    support_order_column: str,
    perturb_validation_query_targets: bool = False,
) -> dict[str, Any]:
    record = records[dataset]
    train = pd.read_csv(_resolve(record["train_csv"]))
    validation = pd.read_csv(_resolve(record["test_csv"]))
    required = {"label", "target", *condition_columns}
    if not required <= set(train) or not required <= set(validation):
        raise ValueError(f"missing required columns for {dataset}")
    train, train_support, train_query = _prefix_split(
        train, support_ratio, support_order_column
    )
    validation, validation_support, validation_query = _prefix_split(
        validation, support_ratio, support_order_column
    )
    for frame, support, query in (
        (train, train_support, train_query),
        (validation, validation_support, validation_query),
    ):
        for label in pd.unique(frame.label):
            label_support = support[frame.label.to_numpy()[support] == label]
            label_query = query[frame.label.to_numpy()[query] == label]
            if frame.iloc[label_support][support_order_column].max() >= frame.iloc[label_query][support_order_column].min():
                raise ValueError(f"non-prefix support/query order for {dataset}, {label}")

    train_stats = _support_statistics(train, train_support)
    validation_stats = _support_statistics(validation, validation_support)
    if perturb_validation_query_targets:
        validation = validation.copy()
        validation.loc[validation_query, "target"] += 123.456
    train_rows = train.iloc[train_query].merge(train_stats, on="label", validate="many_to_one")
    validation_rows = validation.iloc[validation_query].merge(
        validation_stats, on="label", validate="many_to_one"
    )

    source_path: Path | None = None
    source_payload: dict[str, Any] | None = None
    if method is not None:
        source_path = _source_path(q_root, dataset, method, seed)
        train_features, validation_features, source_payload = _q_features(
            source_path, q_root, functional_columns
        )
        train_rows = train_rows.merge(train_features, on="label", validate="many_to_one")
        validation_rows = validation_rows.merge(
            validation_features, on="label", validate="many_to_one"
        )
        query_artifact = pd.read_csv(_resolve(source_payload["artifacts"]["query_predictions"]))
        raw_query = validation.iloc[validation_query].reset_index(drop=True)
        np.testing.assert_array_equal(query_artifact.label.to_numpy(), raw_query.label.to_numpy())
        artifact_columns = condition_columns if perturb_validation_query_targets else [*condition_columns, "target"]
        for column in artifact_columns:
            np.testing.assert_allclose(
                query_artifact[column].to_numpy(float),
                raw_query[column].to_numpy(float),
            )

    train_labels = set(train.label.unique())
    validation_labels = set(validation.label.unique())
    if train_labels & validation_labels or len(train_labels) != 8 or len(validation_labels) != 5:
        raise ValueError(f"invalid 8/5 entity split for {dataset}")
    return {
        "train": train_rows,
        "validation": validation_rows,
        "train_labels": sorted(train_labels),
        "validation_labels": sorted(validation_labels),
        "reference_scale": float(train.target.std(ddof=0)),
        "source_path": source_path,
        "source_payload": source_payload,
    }


def _input_columns(
    regime: str,
    condition_columns: list[str],
    functional_columns: list[str],
) -> list[str]:
    if regime == "condition_only":
        return condition_columns
    if regime == "condition_support_stats":
        return [*condition_columns, *SUPPORT_COLUMNS]
    if regime == "condition_raw_q":
        return [*condition_columns, "q1", "q2", "q3", "q4"]
    if regime == "condition_functional_q":
        return [*condition_columns, *functional_columns]
    raise ValueError(regime)


def _cell_output(output_root: Path, cell: dict[str, Any]) -> Path:
    method = cell["method"] or "baseline"
    return output_root / cell["dataset"] / f"seed{cell['seed']}" / method / cell["regime"]


def run_cell(cell: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    _limit_threads(config["threads_per_job"])
    from pysr import PySRRegressor
    from sklearn.metrics import r2_score

    q_root = Path(config["q_root"])
    output_root = Path(config["output_root"])
    records = _prepared_records(q_root)
    bundle = _build_bundle(
        q_root=q_root,
        records=records,
        dataset=cell["dataset"],
        seed=cell["seed"],
        method=cell["method"],
        condition_columns=config["condition_columns"],
        functional_columns=config["functional_columns"],
        support_ratio=config["support_ratio"],
        support_order_column=config["support_order_column"],
    )
    train = bundle["train"]
    validation = bundle["validation"]
    columns = _input_columns(
        cell["regime"], config["condition_columns"], config["functional_columns"]
    )
    train_matrix = train.loc[:, columns].to_numpy(float)
    validation_matrix = validation.loc[:, columns].to_numpy(float)
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    retained = std > 1e-12
    retained_columns = [column for column, keep in zip(columns, retained) if keep]
    if not retained_columns:
        raise ValueError(f"all symbolic columns are constant for {cell}")
    train_matrix = (train_matrix[:, retained] - mean[retained]) / std[retained]
    validation_matrix = (validation_matrix[:, retained] - mean[retained]) / std[retained]
    train_target = train.target.to_numpy(float)
    validation_target = validation.target.to_numpy(float)
    target_mean = float(train_target.mean())
    target_std = float(train_target.std())
    standardized_target = (train_target - target_mean) / target_std
    rng = np.random.default_rng(cell["seed"] + 700003)
    fit_indices = np.arange(len(train))
    if len(fit_indices) > config["sample_size"]:
        fit_indices = np.sort(rng.choice(fit_indices, config["sample_size"], replace=False))

    started = time.perf_counter()
    model = PySRRegressor(
        niterations=config["iterations"],
        maxsize=config["maxsize"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sqrt", "square"],
        populations=config["pysr_procs"] * 3,
        procs=config["pysr_procs"],
        parallelism="multiprocessing",
        timeout_in_seconds=config["job_timeout_seconds"],
        random_state=cell["seed"],
        deterministic=False,
        progress=False,
        temp_equation_file=True,
        verbosity=0,
    )
    model.fit(
        train_matrix[fit_indices],
        standardized_target[fit_indices],
        variable_names=retained_columns,
    )
    train_prediction = target_mean + target_std * model.predict(train_matrix)
    validation_prediction = target_mean + target_std * model.predict(validation_matrix)
    if not np.isfinite(train_prediction).all() or not np.isfinite(validation_prediction).all():
        raise ValueError(f"non-finite symbolic prediction for {cell}")

    output = _cell_output(output_root, cell)
    output.mkdir(parents=True)
    expression = str(model.sympy())
    pd.DataFrame(
        {
            "column": retained_columns,
            "fit_mean": mean[retained],
            "fit_std": std[retained],
        }
    ).to_csv(output / "input_scaler.csv", index=False)
    model.equations_.loc[:, ["complexity", "loss", "equation"]].to_csv(
        output / "pareto_front.csv", index=False
    )
    pd.concat(
        [
            train.assign(prediction=train_prediction, symbolic_split="meta_fit"),
            validation.assign(
                prediction=validation_prediction,
                symbolic_split="structure_validation",
            ),
        ],
        ignore_index=True,
    ).loc[:, ["label", "symbolic_split", *config["condition_columns"], "target", "prediction"]].to_csv(
        output / "predictions.csv", index=False
    )
    source_payload = bundle["source_payload"]
    result = {
        **cell,
        "status": "success",
        "best_expression_standardized": expression,
        "complexity": int(model.get_best()["complexity"]),
        "r2_meta_fit": float(r2_score(train_target, train_prediction)),
        "r2_structure_validation": float(
            r2_score(validation_target, validation_prediction)
        ),
        "reference_nrmse_meta_fit": float(
            np.sqrt(np.mean((train_target - train_prediction) ** 2))
            / bundle["reference_scale"]
        ),
        "reference_nrmse_structure_validation": float(
            np.sqrt(np.mean((validation_target - validation_prediction) ** 2))
            / bundle["reference_scale"]
        ),
        "input_columns_before_constant_drop": columns,
        "input_columns": retained_columns,
        "symbols_used": [
            column
            for column in retained_columns
            if re.search(rf"\b{re.escape(column)}\b", expression)
        ],
        "meta_fit_labels": bundle["train_labels"],
        "structure_validation_labels": bundle["validation_labels"],
        "rows_meta_fit": len(train),
        "rows_structure_validation": len(validation),
        "source_result": (
            str(bundle["source_path"].relative_to(PROJECT_ROOT))
            if bundle["source_path"] is not None
            else None
        ),
        "neural_q_reference_nrmse": (
            float(source_payload["prediction"]["reference_nrmse"])
            if source_payload is not None
            else None
        ),
        "target_fit_mean": target_mean,
        "target_fit_std": target_std,
        "reference_scale": bundle["reference_scale"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _cells(
    datasets: list[str], methods: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for dataset in datasets:
        for seed in seeds:
            cells.extend(
                {"dataset": dataset, "seed": seed, "method": None, "regime": regime}
                for regime in BASELINE_REGIMES
            )
            cells.extend(
                {"dataset": dataset, "seed": seed, "method": method, "regime": regime}
                for method in methods
                for regime in Q_REGIMES
            )
    return cells


def _audit(
    *,
    q_root: Path,
    records: dict[str, dict[str, Any]],
    datasets: list[str],
    methods: list[str],
    seeds: list[int],
    condition_columns: list[str],
    functional_columns: list[str],
    support_ratio: float,
    support_order_column: str,
) -> dict[str, Any]:
    max_leakage_difference = 0.0
    audited_sources = 0
    for dataset in datasets:
        for seed in seeds:
            for method in methods:
                original = _build_bundle(
                    q_root=q_root,
                    records=records,
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    condition_columns=condition_columns,
                    functional_columns=functional_columns,
                    support_ratio=support_ratio,
                    support_order_column=support_order_column,
                )
                perturbed = _build_bundle(
                    q_root=q_root,
                    records=records,
                    dataset=dataset,
                    seed=seed,
                    method=method,
                    condition_columns=condition_columns,
                    functional_columns=functional_columns,
                    support_ratio=support_ratio,
                    support_order_column=support_order_column,
                    perturb_validation_query_targets=True,
                )
                feature_columns = [
                    *condition_columns,
                    *SUPPORT_COLUMNS,
                    "q1",
                    "q2",
                    "q3",
                    "q4",
                    *functional_columns,
                ]
                difference = np.abs(
                    original["validation"].loc[:, feature_columns].to_numpy(float)
                    - perturbed["validation"].loc[:, feature_columns].to_numpy(float)
                ).max()
                max_leakage_difference = max(max_leakage_difference, float(difference))
                audited_sources += 1
    if max_leakage_difference != 0.0:
        raise ValueError(f"query-target leakage probe failed: {max_leakage_difference}")
    return {
        "audited_q_sources": audited_sources,
        "max_query_target_feature_difference": max_leakage_difference,
        "entity_split": "8 meta-fit / 5 structure-validation",
        "prefix_order_verified": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-q-root", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--functional-columns", required=True)
    parser.add_argument("--condition-columns", required=True)
    parser.add_argument("--support-ratio", type=float, required=True)
    parser.add_argument("--support-order-column", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--maxsize", type=int, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--pysr-procs", type=int, required=True)
    parser.add_argument("--max-parallel", type=int, required=True)
    parser.add_argument("--threads-per-job", type=int, required=True)
    parser.add_argument("--job-timeout-seconds", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_root = args.inner_q_root.resolve()
    output_root = args.output_root.resolve()
    methods = _csv_arg(args.methods)
    seeds = [int(seed) for seed in _csv_arg(args.seeds)]
    functional_columns = _csv_arg(args.functional_columns)
    condition_columns = _csv_arg(args.condition_columns)
    source_manifest = json.loads((q_root / "experiment_manifest.json").read_text())
    if methods != source_manifest["methods"] or seeds != source_manifest["seeds"]:
        raise ValueError("requested methods/seeds do not match the frozen q manifest")
    if args.support_ratio != source_manifest["support_ratio"]:
        raise ValueError("support ratio does not match the frozen q manifest")
    if args.support_order_column != source_manifest["support_order_column"]:
        raise ValueError("support order column does not match the frozen q manifest")
    datasets = source_manifest["datasets"]
    records = _prepared_records(q_root)
    cells = _cells(datasets, methods, seeds)
    if len(cells) != 90 or len({json.dumps(cell, sort_keys=True) for cell in cells}) != 90:
        raise ValueError("the frozen symbolic matrix must contain 90 unique cells")
    audit = _audit(
        q_root=q_root,
        records=records,
        datasets=datasets,
        methods=methods,
        seeds=seeds,
        condition_columns=condition_columns,
        functional_columns=functional_columns,
        support_ratio=args.support_ratio,
        support_order_column=args.support_order_column,
    )
    config = {
        "path_base": "repository_root",
        "q_root": str(q_root.relative_to(PROJECT_ROOT)),
        "output_root": str(output_root.relative_to(PROJECT_ROOT)),
        "condition_columns": condition_columns,
        "functional_columns": functional_columns,
        "support_ratio": args.support_ratio,
        "support_order_column": args.support_order_column,
        "iterations": args.iterations,
        "maxsize": args.maxsize,
        "sample_size": args.sample_size,
        "pysr_procs": args.pysr_procs,
        "max_parallel": args.max_parallel,
        "threads_per_job": args.threads_per_job,
        "job_timeout_seconds": args.job_timeout_seconds,
    }
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan": str(PLAN_PATH.relative_to(PROJECT_ROOT)),
        "plan_sha256": _sha256(PLAN_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "source_manifest": str((q_root / "experiment_manifest.json").relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": _sha256(q_root / "experiment_manifest.json"),
        "datasets": datasets,
        "methods": methods,
        "seeds": seeds,
        "regimes": [*BASELINE_REGIMES, *Q_REGIMES],
        "cells": cells,
        "planned": len(cells),
        "config": config,
        "audit": audit,
        "train_q_information": "complete meta-fit curves",
        "structure_validation_q_information": "prefix support targets only",
        "structure_validation_query_targets_used_for_structure_selection": False,
    }
    if args.dry_run:
        print(json.dumps({key: value for key, value in manifest.items() if key != "cells"}, indent=2))
        return
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse existing output root: {output_root}")
    output_root.mkdir(parents=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    status_path = output_root / "status.jsonl"
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(run_cell, cell, config): cell for cell in cells}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with status_path.open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(
                f"[{len(results)}/90] {result['dataset']} seed{result['seed']} "
                f"{result['method'] or 'baseline'} {result['regime']} "
                f"held_nrmse={result['reference_nrmse_structure_validation']:.6g}",
                flush=True,
            )
    table = pd.DataFrame(results).sort_values(["dataset", "seed", "method", "regime"])
    table.to_csv(output_root / "results.csv", index=False)
    (output_root / "status.json").write_text(
        json.dumps({"state": "completed_all", "planned": 90, "success": 90, "failed": 0}, indent=2)
    )


if __name__ == "__main__":
    main()
