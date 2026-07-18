from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from lvs.core.expression_library import describe_expression_support, load_expression_library
from lvs.workflows.single import (
    WorkflowConfig,
    add_workflow_arguments,
    list_expressions,
    run_workflow,
    workflow_config_to_json,
)


@dataclass(frozen=True)
class BatchWorkflowConfig:
    workflow_template: WorkflowConfig
    expression_ids: Optional[tuple[int, ...]]
    max_expressions: Optional[int]
    max_workers: int
    fail_fast: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the benchmark workflow for all supported expressions in the expression library, "
            "with optional filtering, parallel execution, and batch-level summaries."
        )
    )
    add_workflow_arguments(
        parser,
        include_expression_selection=False,
        include_list_expressions=True,
    )
    parser.add_argument(
        "--expression-ids",
        type=str,
        default=None,
        help="Optional comma-separated expression ids to run. Defaults to all supported expressions.",
    )
    parser.add_argument(
        "--max-expressions",
        type=int,
        default=None,
        help="Optional cap on the number of supported expressions to run after filtering.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker processes used for parallel execution. Defaults to 1 (sequential).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch early if any expression run fails.",
    )
    return parser


def parse_expression_ids(raw_value: Optional[str]) -> Optional[tuple[int, ...]]:
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts:
        return None
    expression_ids = tuple(int(part) for part in parts)
    if len(set(expression_ids)) != len(expression_ids):
        raise ValueError("expression_ids must not contain duplicates.")
    return expression_ids


def namespace_to_batch_config(args: argparse.Namespace) -> BatchWorkflowConfig:
    workflow_template = WorkflowConfig(
        library_csv=args.library_csv,
        expression_id=None,
        expression_name=None,
        label_count=args.label_count,
        train_samples_per_label=args.train_samples_per_label,
        test_samples_per_label=args.test_samples_per_label,
        noise_std=args.noise_std,
        seed=args.seed,
        backend=args.backend,
        q_dim=args.q_dim,
        output_root=args.output_root,
        max_attempts_per_row=args.max_attempts_per_row,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        auto_target_scale=args.auto_target_scale,
        target_scale_min_magnitude=args.target_scale_min_magnitude,
        target_scale_desired_magnitude=args.target_scale_desired_magnitude,
        cal_steps=args.cal_steps,
        cal_lr=args.cal_lr,
        cal_ratio=args.cal_ratio,
        early_stop_enabled=args.early_stop_enabled,
        early_stop_r2_threshold=args.early_stop_r2_threshold,
        early_stop_patience=args.early_stop_patience,
        latent_feature_orthogonality_weight=args.latent_feature_orthogonality_weight,
        latent_feature_orthogonality_type=args.latent_feature_orthogonality_type,
        latent_feature_stats_mode=args.latent_feature_stats_mode,
        latent_curve_continuity_weight=args.latent_curve_continuity_weight,
        latent_curve_continuity_grid_size=args.latent_curve_continuity_grid_size,
        calibration_q_prior_weight=args.calibration_q_prior_weight,
        latent_q_l2_weight=args.latent_q_l2_weight,
        prediction_loss_type=args.prediction_loss_type,
        latent_q_whitening_weight=args.latent_q_whitening_weight,
        latent_jacobian_disentanglement_weight=args.latent_jacobian_disentanglement_weight,
        latent_q_canonicalization_mode=args.latent_q_canonicalization_mode,
        latent_q_smoothness_weight=args.latent_q_smoothness_weight,
        latent_q_smoothness_epsilon=args.latent_q_smoothness_epsilon,
        device=args.device,
        quiet=args.quiet,
        hidden_sizes=args.hidden_sizes,
        kan_grid=args.kan_grid,
        kan_order=args.kan_order,
    )
    max_workers = int(args.max_workers)
    if max_workers <= 0:
        raise ValueError("max_workers must be a positive integer.")
    if args.max_expressions is not None and args.max_expressions <= 0:
        raise ValueError("max_expressions must be a positive integer when provided.")
    return BatchWorkflowConfig(
        workflow_template=workflow_template,
        expression_ids=parse_expression_ids(args.expression_ids),
        max_expressions=args.max_expressions,
        max_workers=max_workers,
        fail_fast=bool(args.fail_fast),
    )


def create_batch_dir(template: WorkflowConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_all_{template.backend}_qdim{template.q_dim}"
    batch_dir = template.output_root / base_name
    suffix = 1
    while batch_dir.exists():
        batch_dir = template.output_root / f"{base_name}_{suffix}"
        suffix += 1
    batch_dir.mkdir(parents=True, exist_ok=False)
    return batch_dir


def ensure_batch_structure(batch_dir: Path) -> dict[str, Path]:
    paths = {
        "configs": batch_dir / "configs",
        "summaries": batch_dir / "summaries",
        "runs": batch_dir / "runs",
        "logs": batch_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def select_supported_expression_entries(
    library_csv: Path,
    *,
    requested_expression_ids: Optional[tuple[int, ...]],
    max_expressions: Optional[int],
) -> list[dict[str, Any]]:
    records = load_expression_library(library_csv)
    descriptions = describe_expression_support(records)

    by_id = {item["expression_id"]: item for item in descriptions}
    if requested_expression_ids is not None:
        missing_ids = [expression_id for expression_id in requested_expression_ids if expression_id not in by_id]
        if missing_ids:
            raise ValueError(
                f"Some expression ids do not exist in the library: {', '.join(str(item) for item in missing_ids)}"
            )

    selected_entries: list[dict[str, Any]] = []
    for item in descriptions:
        expression_id = int(item["expression_id"])
        if requested_expression_ids is not None and expression_id not in requested_expression_ids:
            continue
        if item["status"] != "supported":
            continue
        selected_entries.append(
            {
                "expression_id": expression_id,
                "formula_name": item["formula_name"],
                "observed_feature_variables": list(item["observed_feature_variables"]),
                "latent_variables": list(item["latent_variables"]),
            }
        )

    if requested_expression_ids is not None:
        order_map = {expression_id: index for index, expression_id in enumerate(requested_expression_ids)}
        selected_entries.sort(key=lambda item: order_map[item["expression_id"]])

    if max_expressions is not None:
        selected_entries = selected_entries[:max_expressions]

    if not selected_entries:
        raise ValueError("No supported expressions were selected for execution.")
    return selected_entries


def workflow_config_from_json(data: dict[str, Any]) -> WorkflowConfig:
    return WorkflowConfig(
        library_csv=Path(data["library_csv"]),
        expression_id=data["expression_id"],
        expression_name=data["expression_name"],
        label_count=data["label_count"],
        train_samples_per_label=data["train_samples_per_label"],
        test_samples_per_label=data["test_samples_per_label"],
        noise_std=data["noise_std"],
        seed=data["seed"],
        backend=data["backend"],
        q_dim=data["q_dim"],
        output_root=Path(data["output_root"]),
        max_attempts_per_row=data["max_attempts_per_row"],
        epochs=data["epochs"],
        batch_size=data["batch_size"],
        lr=data["lr"],
        auto_target_scale=data["auto_target_scale"],
        target_scale_min_magnitude=data["target_scale_min_magnitude"],
        target_scale_desired_magnitude=data["target_scale_desired_magnitude"],
        cal_steps=data["cal_steps"],
        cal_lr=data["cal_lr"],
        cal_ratio=data["cal_ratio"],
        early_stop_enabled=data["early_stop_enabled"],
        early_stop_r2_threshold=data["early_stop_r2_threshold"],
        early_stop_patience=data["early_stop_patience"],
        latent_feature_orthogonality_weight=data["latent_feature_orthogonality_weight"],
        latent_feature_orthogonality_type=data.get("latent_feature_orthogonality_type", "pearson"),
        latent_feature_stats_mode=data.get("latent_feature_stats_mode", "mean_std"),
        latent_curve_continuity_weight=data.get("latent_curve_continuity_weight", 0.0),
        latent_curve_continuity_grid_size=data.get("latent_curve_continuity_grid_size", 64),
        calibration_q_prior_weight=data.get("calibration_q_prior_weight", 0.0),
        latent_q_l2_weight=data.get("latent_q_l2_weight", 0.0),
        prediction_loss_type=data.get("prediction_loss_type", "mse"),
        latent_q_whitening_weight=data.get("latent_q_whitening_weight", 0.0),
        latent_jacobian_disentanglement_weight=data.get("latent_jacobian_disentanglement_weight", 0.0),
        latent_q_canonicalization_mode=data.get("latent_q_canonicalization_mode", "none"),
        latent_q_smoothness_weight=data.get("latent_q_smoothness_weight", 0.0),
        latent_q_smoothness_epsilon=data.get("latent_q_smoothness_epsilon", 0.05),
        device=data["device"],
        quiet=data["quiet"],
        hidden_sizes=data["hidden_sizes"],
        kan_grid=data["kan_grid"],
        kan_order=data["kan_order"],
    )


def discover_created_run_dir(output_root: Path, existing_dirs: set[str]) -> Optional[Path]:
    current_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}
    new_dirs = sorted(current_dirs - existing_dirs)
    if not new_dirs:
        return None
    return output_root / new_dirs[-1]


def execute_single_expression(payload: dict[str, Any]) -> dict[str, Any]:
    entry = dict(payload["entry"])
    config_data = dict(payload["workflow_config"])
    config = workflow_config_from_json(config_data)
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    existing_dirs = {path.name for path in output_root.iterdir() if path.is_dir()}

    expression_id = int(entry["expression_id"])
    formula_name = str(entry["formula_name"])
    run_config = WorkflowConfig(**{**asdict(config), "expression_id": expression_id, "expression_name": None})

    try:
        result = run_workflow(run_config)
        return {
            "expression_id": expression_id,
            "formula_name": formula_name,
            "status": "success",
            "run_dir": str(result.run_dir),
            "q_dim_model": result.q_dim_model,
            "ground_truth_latent_dim": result.ground_truth_latent_dim,
            "train_r2_last_epoch": result.metrics.get("train_r2_last_epoch"),
            "train_mse_last_epoch": result.metrics.get("train_mse_last_epoch"),
            "test_r2": result.metrics.get("test_r2"),
            "test_mse": result.metrics.get("test_mse"),
            "error": None,
        }
    except Exception as exc:
        run_dir = discover_created_run_dir(output_root, existing_dirs)
        return {
            "expression_id": expression_id,
            "formula_name": formula_name,
            "status": "failed",
            "run_dir": None if run_dir is None else str(run_dir),
            "q_dim_model": config.q_dim,
            "ground_truth_latent_dim": None,
            "train_r2_last_epoch": None,
            "train_mse_last_epoch": None,
            "test_r2": None,
            "test_mse": None,
            "error": str(exc),
        }


def save_batch_results(results: list[dict[str, Any]], summaries_dir: Path) -> dict[str, Path]:
    csv_path = summaries_dir / "batch_results.csv"
    json_path = summaries_dir / "batch_results.json"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "batch_results_csv": csv_path,
        "batch_results_json": json_path,
    }


def build_batch_summary(
    *,
    batch_dir: Path,
    batch_config: BatchWorkflowConfig,
    resolved_workflow_template: WorkflowConfig,
    selected_entries: list[dict[str, Any]],
    results: list[dict[str, Any]],
    saved_paths: dict[str, Path],
) -> dict[str, Any]:
    success_count = sum(1 for item in results if item["status"] == "success")
    failed_count = len(results) - success_count
    return {
        "batch_dir": str(batch_dir),
        "workflow_template": workflow_config_to_json(resolved_workflow_template),
        "batch_config": {
            "expression_ids": None if batch_config.expression_ids is None else list(batch_config.expression_ids),
            "max_expressions": batch_config.max_expressions,
            "max_workers": batch_config.max_workers,
            "fail_fast": batch_config.fail_fast,
        },
        "selected_expressions": selected_entries,
        "result_stats": {
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
        },
        "saved_paths": {key: str(value) for key, value in saved_paths.items()},
    }


def run_batch(batch_config: BatchWorkflowConfig) -> tuple[Path, list[dict[str, Any]], dict[str, Path]]:
    batch_dir = create_batch_dir(batch_config.workflow_template)
    paths = ensure_batch_structure(batch_dir)

    selected_entries = select_supported_expression_entries(
        batch_config.workflow_template.library_csv,
        requested_expression_ids=batch_config.expression_ids,
        max_expressions=batch_config.max_expressions,
    )
    run_output_root = paths["runs"]
    workflow_template = WorkflowConfig(
        **{**asdict(batch_config.workflow_template), "output_root": run_output_root}
    )

    write_json(paths["configs"] / "batch_config.json", {
        "workflow_template": workflow_config_to_json(workflow_template),
        "expression_ids": None if batch_config.expression_ids is None else list(batch_config.expression_ids),
        "max_expressions": batch_config.max_expressions,
        "max_workers": batch_config.max_workers,
        "fail_fast": batch_config.fail_fast,
    })
    write_json(paths["configs"] / "selected_expressions.json", {"expressions": selected_entries})

    payloads = [
        {
            "entry": entry,
            "workflow_config": workflow_config_to_json(
                WorkflowConfig(**{**asdict(workflow_template), "expression_id": entry["expression_id"], "expression_name": None})
            ),
        }
        for entry in selected_entries
    ]

    results: list[dict[str, Any]] = []
    if batch_config.max_workers == 1:
        for index, payload in enumerate(payloads, start=1):
            result = execute_single_expression(payload)
            results.append(result)
            print(
                f"[{index}/{len(payloads)}] expression_id={result['expression_id']} "
                f"name={result['formula_name']} status={result['status']}"
            )
            if batch_config.fail_fast and result["status"] != "success":
                break
    else:
        executor_class = ProcessPoolExecutor
        executor_label = "process"
        executor = None
        try:
            executor = executor_class(max_workers=batch_config.max_workers)
        except Exception as exc:
            print(
                "Falling back to thread-based parallelism because process-based parallelism "
                f"could not be started: {exc}"
            )
            executor_class = ThreadPoolExecutor
            executor_label = "thread"
            executor = executor_class(max_workers=batch_config.max_workers)

        with executor:
            print(f"Using {executor_label}-based parallelism with max_workers={batch_config.max_workers}")
            future_map = {executor.submit(execute_single_expression, payload): payload for payload in payloads}
            for index, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{index}/{len(payloads)}] expression_id={result['expression_id']} "
                    f"name={result['formula_name']} status={result['status']}"
                )
                if batch_config.fail_fast and result["status"] != "success":
                    for pending_future in future_map:
                        pending_future.cancel()
                    break
        results.sort(key=lambda item: item["expression_id"])

    saved_paths = save_batch_results(results, paths["summaries"])
    batch_summary = build_batch_summary(
        batch_dir=batch_dir,
        batch_config=batch_config,
        resolved_workflow_template=workflow_template,
        selected_entries=selected_entries,
        results=results,
        saved_paths=saved_paths,
    )
    saved_paths["batch_summary_json"] = write_json(paths["summaries"] / "batch_summary.json", batch_summary)
    return batch_dir, results, saved_paths


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "list_expressions", False):
        list_expressions(args.library_csv)
        return

    batch_config = namespace_to_batch_config(args)
    batch_dir, results, saved_paths = run_batch(batch_config)
    success_count = sum(1 for item in results if item["status"] == "success")
    failed_count = len(results) - success_count
    print(f"Batch directory: {batch_dir}")
    print(f"Total expressions: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    for name, path in saved_paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
