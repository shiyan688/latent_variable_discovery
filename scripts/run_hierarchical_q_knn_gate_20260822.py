#!/usr/bin/env python3
"""Run the frozen train-entity-validation q/kNN regime gate."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/lvs-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lvs-xdg-cache")

import run_iclr_real_discovery as real
import run_q_knn_reliability_selector_20260822 as base
from lvs.backends.torch_mlp import build_torch_model_factory, parse_hidden_sizes
from lvs.core.pipeline import (
    OutputConfig,
    build_dataset_from_arrays,
    calibrate_latent_q_for_test_labels,
    evaluate_latent_q_pipeline,
    train_latent_q_model,
)

DEFAULT_ROOT = PROJECT_ROOT / "runs" / "hierarchical_q_knn_gate_confirm_20260822"
PLAN_PATH = PROJECT_ROOT / "HIERARCHICAL_Q_KNN_GATE_PLAN_20260822.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-job")
    run.add_argument("--prepared-summary", type=Path, required=True)
    run.add_argument("--dataset", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    base._experiment_args(run)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--gpus", default="0,1,6,7")
    launch.add_argument("--seeds", default=",".join(str(seed) for seed in range(30, 40)))
    launch.add_argument("--poll-seconds", type=float, default=15.0)
    launch.add_argument("--single-job-timeout-minutes", type=float, default=240.0)
    launch.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    launch.add_argument("--dry-run", action="store_true")
    base._experiment_args(launch)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def _job_config(args: argparse.Namespace) -> dict[str, Any]:
    job = base._job_config(args)
    job.update(
        {
            "method": "train_entity_validation_q_knn_gate",
            "development_label_fit_ratio": 0.75,
            "development_split_seed": 20260822,
            "development_episodes": 3,
            "development_policy_score": "median_reference_nrmse",
            "development_tie_rule": "support_knn",
        }
    )
    return job


def _label_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(pd.unique(labels))
    rng = np.random.default_rng(20260822)
    shuffled = unique[rng.permutation(len(unique))]
    count = min(len(unique) - 1, max(1, int(np.floor(0.75 * len(unique)))))
    fit_labels = shuffled[:count]
    validation_labels = shuffled[count:]
    return np.flatnonzero(np.isin(labels, fit_labels)), np.flatnonzero(
        np.isin(labels, validation_labels)
    )


def run_job(args: argparse.Namespace) -> Path:
    job = _job_config(args)
    run_dir = args.output_root / "results" / args.dataset / f"seed{args.seed}_{base._stable_hash(job)}"
    result_path = run_dir / "result.json"
    if result_path.exists() and args.resume:
        existing = json.loads(result_path.read_text())
        if existing.get("status") == "success" and existing.get("job") == job:
            return result_path
    run_dir.mkdir(parents=True, exist_ok=True)

    record = real._load_record(args.prepared_summary, args.dataset)
    features = list(record["feature_columns"])
    train = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["train_csv"])),
        args.max_train_per_label,
        args.subsample_seed,
    )
    test = real._cap_rows_per_label(
        pd.read_csv(real._resolve_path(record["test_csv"])),
        args.max_test_per_label,
        args.subsample_seed + 10000,
    )
    train_x = train[features].to_numpy(np.float32)
    train_y = train.target.to_numpy(np.float32)
    train_labels = train.label.to_numpy()
    test_x = test[features].to_numpy(np.float32)
    test_y = test.target.to_numpy(np.float32)
    test_labels = test.label.to_numpy()
    meta_fit, meta_validation = _label_split(train_labels)
    assert not set(train_labels[meta_fit]) & set(train_labels[meta_validation])

    started = time.perf_counter()
    config = base._latent_config(args)
    fit_dataset = build_dataset_from_arrays(
        train_x[meta_fit], train_labels[meta_fit], train_y[meta_fit], feature_names=features
    )
    validation_dataset = build_dataset_from_arrays(
        train_x[meta_validation], train_labels[meta_validation], train_y[meta_validation],
        feature_names=features,
    )
    development_training = train_latent_q_model(
        fit_dataset, build_torch_model_factory(parse_hidden_sizes(args.hidden_sizes)), config
    )
    development_rows = []
    for episode in range(3):
        episode_seed = args.seed + 300000 + episode * 100003
        episode_config = replace(config, seed=episode_seed)
        q_episode = evaluate_latent_q_pipeline(
            fit_dataset, validation_dataset, development_training, episode_config,
            output_config=OutputConfig(save_csv=False, save_plot=False),
        )
        support, query = real._support_query_indices(
            train_labels[meta_validation], args.support_ratio, episode_seed
        )
        np.testing.assert_array_equal(q_episode.eval_indices, query)
        knn_episode = real._run_support_knn(
            train_x[meta_validation], train_y[meta_validation], train_labels[meta_validation],
            support, query,
        )
        q_score = base._metrics(
            q_episode.eval_targets, q_episode.eval_predictions, q_episode.eval_labels,
            train_y[meta_fit],
        )["reference_nrmse"]
        knn_score = base._metrics(
            q_episode.eval_targets, knn_episode, q_episode.eval_labels, train_y[meta_fit]
        )["reference_nrmse"]
        development_rows.append(
            {"episode": episode, "seed": episode_seed, "q_reference_nrmse": q_score,
             "knn_reference_nrmse": knn_score}
        )
    development = pd.DataFrame(development_rows)
    q_development = float(development.q_reference_nrmse.median())
    knn_development = float(development.knn_reference_nrmse.median())
    selected_component = "latent_q" if q_development < knn_development else "support_knn"

    train_dataset = build_dataset_from_arrays(
        train_x, train_labels, train_y, feature_names=features
    )
    test_dataset = build_dataset_from_arrays(test_x, test_labels, test_y, feature_names=features)
    training = train_latent_q_model(
        train_dataset, build_torch_model_factory(parse_hidden_sizes(args.hidden_sizes)), config
    )
    q_result = evaluate_latent_q_pipeline(
        train_dataset, test_dataset, training, config,
        output_config=OutputConfig(save_csv=False, save_plot=False),
    )
    support, query = real._support_query_indices(test_labels, args.support_ratio, args.seed)
    np.testing.assert_array_equal(q_result.eval_indices, query)
    q_prediction = q_result.eval_predictions
    truth = q_result.eval_targets
    query_labels = q_result.eval_labels
    knn_prediction = real._run_support_knn(test_x, test_y, test_labels, support, query)
    hierarchical_prediction = q_prediction if selected_component == "latent_q" else knn_prediction

    support_dataset = build_dataset_from_arrays(
        test_x[support], test_labels[support], test_y[support], feature_names=features
    )
    selector_seed = args.seed + 104729
    selector_config = replace(config, calibration_ratio=args.selector_fit_ratio, seed=selector_seed)
    q_selector = calibrate_latent_q_for_test_labels(support_dataset, training, selector_config)
    selector_fit, selector_validation = real._support_query_indices(
        test_labels[support], args.selector_fit_ratio, selector_seed
    )
    np.testing.assert_array_equal(q_selector.eval_indices, selector_validation)
    knn_selector = real._run_support_knn(
        test_x[support], test_y[support], test_labels[support], selector_fit, selector_validation
    )
    choose_q = {}
    for label in pd.unique(q_selector.eval_labels):
        selected = q_selector.eval_labels == label
        choose_q[label] = np.mean(np.abs(q_selector.eval_predictions[selected] - q_selector.eval_targets[selected])) < np.mean(np.abs(knn_selector[selected] - q_selector.eval_targets[selected]))
    entity_prediction = np.asarray([
        q_prediction[index] if choose_q[label] else knn_prediction[index]
        for index, label in enumerate(query_labels)
    ])
    oracle_q = {}
    for label in pd.unique(query_labels):
        selected = query_labels == label
        oracle_q[label] = np.sqrt(np.mean((q_prediction[selected] - truth[selected]) ** 2)) < np.sqrt(np.mean((knn_prediction[selected] - truth[selected]) ** 2))
    oracle_prediction = np.asarray([
        q_prediction[index] if oracle_q[label] else knn_prediction[index]
        for index, label in enumerate(query_labels)
    ])

    altered_y = test_y.copy()
    altered_y[query] += np.float32(123.0 * max(float(np.std(train_y)), 1.0))
    altered_dataset = build_dataset_from_arrays(test_x, test_labels, altered_y, feature_names=features)
    altered_q = calibrate_latent_q_for_test_labels(altered_dataset, training, config)
    altered_knn = real._run_support_knn(test_x, altered_y, test_labels, support, query)
    altered_entity = np.asarray([
        altered_q.eval_predictions[index] if choose_q[label] else altered_knn[index]
        for index, label in enumerate(query_labels)
    ])
    altered_hierarchical = altered_q.eval_predictions if selected_component == "latent_q" else altered_knn
    leakage = float(max(
        np.max(np.abs(q_prediction - altered_q.eval_predictions)),
        np.max(np.abs(knn_prediction - altered_knn)),
        np.max(np.abs(entity_prediction - altered_entity)),
        np.max(np.abs(hierarchical_prediction - altered_hierarchical)),
    ))
    assert leakage <= 1e-7

    query_frame = test.iloc[query].copy()
    query_frame["q_prediction"] = q_prediction
    query_frame["knn_prediction"] = knn_prediction
    query_frame["hierarchical_prediction"] = hierarchical_prediction
    query_frame["entity_selector_prediction"] = entity_prediction
    query_path = run_dir / "query_predictions.csv"
    query_frame.to_csv(query_path, index=False)
    development_path = run_dir / "development_episodes.csv"
    development.to_csv(development_path, index=False)
    q_columns = [column for column in q_result.test_output if column.startswith("q")]
    q_by_label = q_result.test_output.groupby("label", sort=False)[q_columns].mean()
    q_mapping = {label: q_by_label.loc[label].to_numpy(float) for label in q_by_label.index}
    geometry, geometry_labels, geometry_q = base._q_geometry(query_frame, features, q_mapping, args.seed)
    q_frame = pd.DataFrame({"label": geometry_labels})
    for index in range(geometry_q.shape[1]):
        q_frame[f"q{index + 1}"] = geometry_q[:, index]
    q_path = run_dir / "test_label_q.csv"
    q_frame.to_csv(q_path, index=False)
    checkpoint_path = run_dir / "training_checkpoints.pt"
    torch.save(
        {"job": job, "development_model_state_dict": development_training.model.state_dict(),
         "full_model_state_dict": training.model.state_dict(),
         "full_embedding_state_dict": training.embedding.state_dict()}, checkpoint_path
    )
    metrics = {
        "hierarchical_gate": base._metrics(truth, hierarchical_prediction, query_labels, train_y),
        "latent_q": base._metrics(truth, q_prediction, query_labels, train_y),
        "support_knn": base._metrics(truth, knn_prediction, query_labels, train_y),
        "entity_selector": base._metrics(truth, entity_prediction, query_labels, train_y),
        "query_oracle_non_deployable": base._metrics(truth, oracle_prediction, query_labels, train_y),
    }
    payload = {
        "status": "success", "created_at": datetime.now(timezone.utc).isoformat(),
        "job": job,
        "dataset": {"train_rows": len(train), "test_rows": len(test),
                    "meta_fit_labels": int(pd.Series(train_labels[meta_fit]).nunique()),
                    "meta_validation_labels": int(pd.Series(train_labels[meta_validation]).nunique()),
                    "support_rows": len(support), "query_rows": len(query)},
        "development": {"selected_component": selected_component,
                        "q_median_reference_nrmse": q_development,
                        "knn_median_reference_nrmse": knn_development,
                        "fit_labels": train_labels[meta_fit][~pd.Series(train_labels[meta_fit]).duplicated()].tolist(),
                        "validation_labels": train_labels[meta_validation][~pd.Series(train_labels[meta_validation]).duplicated()].tolist()},
        "metrics": metrics, "q_geometry": geometry,
        "optimization_counters": {"development": asdict(development_training.optimization_counters),
                                  "full": asdict(training.optimization_counters)},
        "query_leakage_probe_max_abs_difference": leakage,
        "wall_time_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(torch.device(args.device))},
        "artifacts": {"query_predictions": str(query_path),
                      "development_episodes": str(development_path),
                      "test_label_q": str(q_path), "training_checkpoints": str(checkpoint_path)},
    }
    base._write_json_atomic(result_path, payload)
    return result_path


def summarize(root: Path) -> None:
    payloads = [json.loads(path.read_text()) for path in root.glob("results/*/seed*/result.json")]
    if len(payloads) != 40:
        raise RuntimeError(f"expected 40 successful results, found {len(payloads)}")
    rows = []
    for payload in payloads:
        for method, metrics in payload["metrics"].items():
            rows.append({"dataset": payload["job"]["dataset"], "seed": payload["job"]["seed"],
                         "method": method, **metrics,
                         "selected_component": payload["development"]["selected_component"],
                         "leakage": payload["query_leakage_probe_max_abs_difference"]})
    all_runs = pd.DataFrame(rows).sort_values(["dataset", "seed", "method"])
    all_runs.to_csv(root / "all_runs.csv", index=False)
    summary = all_runs.groupby(["dataset", "method"]).agg(
        median_nrmse=("reference_nrmse", "median"), p90_nrmse=("reference_nrmse", lambda x: x.quantile(.9)),
        max_nrmse=("reference_nrmse", "max"), catastrophic_runs=("reference_nrmse", lambda x: int((x > 1).sum()))
    ).reset_index()
    summary.to_csv(root / "method_summary.csv", index=False)
    selected = all_runs[all_runs.method == "hierarchical_gate"]
    effects = []
    for dataset in sorted(selected.dataset.unique()):
        for anchor in ("latent_q", "support_knn", "entity_selector"):
            pair = selected[selected.dataset == dataset].merge(
                all_runs[(all_runs.dataset == dataset) & (all_runs.method == anchor)], on="seed", suffixes=("_gate", "_anchor")
            )
            delta = pair.reference_nrmse_gate - pair.reference_nrmse_anchor
            effects.append({"dataset": dataset, "anchor": anchor, "wins": int((delta < 0).sum()),
                            "median_delta": float(delta.median()),
                            "wilcoxon_p": float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0})
    effects_frame = pd.DataFrame(effects)
    effects_frame["wilcoxon_bh_q"] = base._bh_adjust(effects_frame.wilcoxon_p.tolist())
    effects_frame.to_csv(root / "paired_effects.csv", index=False)
    lookup = summary.set_index(["dataset", "method"])
    q_counts = selected.assign(q=selected.selected_component.eq("latent_q")).groupby("dataset").q.sum()
    nasa = "nasa_battery_capacity"
    starry = [dataset for dataset, _ in base.DATASETS if dataset != nasa]
    gates = {
        "integrity": bool(len(all_runs) == 200 and all_runs.leakage.max() <= 1e-7),
        "nasa": bool(q_counts[nasa] >= 8 and lookup.loc[(nasa, "hierarchical_gate"), "median_nrmse"] <= 1.05 * lookup.loc[(nasa, "latent_q"), "median_nrmse"]),
        "starry": bool(all(q_counts[dataset] <= 2 and lookup.loc[(dataset, "hierarchical_gate"), "catastrophic_runs"] == 0 and lookup.loc[(dataset, "hierarchical_gate"), "median_nrmse"] <= 1.05 * lookup.loc[(dataset, "support_knn"), "median_nrmse"] for dataset in starry)),
        "pooled": bool(selected.reference_nrmse.median() <= all_runs[all_runs.method == "entity_selector"].reference_nrmse.median()),
    }
    audit = {"results": len(payloads), "method_rows": len(all_runs), "max_leakage": float(all_runs.leakage.max()),
             "q_selections": {key: int(value) for key, value in q_counts.items()}, "gates": gates,
             "advancement": "PASS" if all(gates.values()) else "FAIL"}
    base._write_json_atomic(root / "terminal_audit.json", audit)
    columns = summary.columns.tolist()
    table_rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.itertuples(index=False, name=None):
        table_rows.append(
            "| "
            + " | ".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in row)
            + " |"
        )
    table = "\n".join(table_rows)
    (root / "HIERARCHICAL_Q_KNN_GATE_RESULTS.md").write_text(
        "# Hierarchical q–kNN gate results\n\n## Material Passport\n\n- Origin Skill: experiment-agent\n- Verification Status: VERIFIED TERMINAL\n- Version Label: hierarchical_q_knn_gate_v1\n\n"
        + table + "\n\n## Predeclared gates\n\n```json\n" + json.dumps(audit, indent=2) + "\n```\n"
    )


def main() -> None:
    args = parse_args()
    if args.command == "run-job":
        print(run_job(args))
    elif args.command == "summarize":
        summarize(args.output_root)
    else:
        base.DEFAULT_ROOT = DEFAULT_ROOT
        base.PLAN_PATH = PLAN_PATH
        base.__file__ = __file__
        base.launch(args)


if __name__ == "__main__":
    main()
