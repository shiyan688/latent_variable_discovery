#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from latent_expression_library import evaluate_scalar_expression


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether learned latent q distances are continuous with respect to "
            "whole-response function distances for a generated-expression workflow run."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--n-probes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--pairwise-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_run(
        args.run_dir,
        split=args.split,
        n_probes=args.n_probes,
        seed=args.seed,
        output_json=args.output_json,
        pairwise_csv=args.pairwise_csv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def analyze_run(
    run_dir: Path,
    *,
    split: str = "train",
    n_probes: int = 512,
    seed: int = 42,
    output_json: Path | None = None,
    pairwise_csv: Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    if n_probes <= 0:
        raise ValueError("n_probes must be positive.")

    metadata_path = run_path / "artifacts" / "data" / "expression_metadata.json"
    latent_truth_path = run_path / "artifacts" / "data" / "latent_truth.csv"
    q_path = run_path / "artifacts" / "latent_q" / f"{split}_with_q.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    latent_truth = pd.read_csv(latent_truth_path)
    q_frame = pd.read_csv(q_path)

    label_col = "label"
    q_columns = [column for column in q_frame.columns if column.startswith("q")]
    latent_columns = [column for column in latent_truth.columns if column != label_col]
    if not q_columns:
        raise ValueError(f"No q columns found in {q_path}.")
    if not latent_columns:
        raise ValueError(f"No latent truth columns found in {latent_truth_path}.")

    learned_q_columns = [f"learned_{column}" for column in q_columns]
    learned_q_by_label = q_frame.groupby(label_col)[q_columns].mean().reset_index()
    learned_q_by_label = learned_q_by_label.rename(columns=dict(zip(q_columns, learned_q_columns)))
    label_table = learned_q_by_label.merge(latent_truth, on=label_col, how="inner")
    if len(label_table) < 2:
        raise ValueError("At least two labels are required for pairwise continuity analysis.")

    probes = sample_feature_probes(metadata, n_probes=n_probes, seed=seed)
    responses = evaluate_label_responses(metadata, label_table, latent_columns, probes)

    learned_q = label_table[learned_q_columns].to_numpy(dtype=np.float64)
    true_q = label_table[latent_columns].to_numpy(dtype=np.float64)
    pairwise_rows = build_pairwise_rows(
        labels=label_table[label_col].to_numpy(),
        learned_q=learned_q,
        true_q=true_q,
        responses=responses,
    )
    pairwise_frame = pd.DataFrame(pairwise_rows)

    metrics = {
        "run_dir": str(run_path),
        "split": split,
        "label_count": int(len(label_table)),
        "q_columns": learned_q_columns,
        "latent_truth_columns": latent_columns,
        "n_probes": int(n_probes),
        "learned_q_distance_vs_function_distance": correlation_metrics(
            pairwise_frame["learned_q_distance"].to_numpy(),
            pairwise_frame["function_rmse_distance"].to_numpy(),
        ),
        "true_q_distance_vs_function_distance": correlation_metrics(
            pairwise_frame["true_q_distance"].to_numpy(),
            pairwise_frame["function_rmse_distance"].to_numpy(),
        ),
        "learned_q_distance_vs_true_q_distance": correlation_metrics(
            pairwise_frame["learned_q_distance"].to_numpy(),
            pairwise_frame["true_q_distance"].to_numpy(),
        ),
    }

    json_path = output_json or run_path / "artifacts" / "latent_q" / f"{split}_latent_continuity.json"
    csv_path = pairwise_csv or run_path / "artifacts" / "latent_q" / f"{split}_latent_continuity_pairwise.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pairwise_frame.to_csv(csv_path, index=False)
    metrics["saved_paths"] = {
        "metrics_json": str(json_path),
        "pairwise_csv": str(csv_path),
    }
    return metrics


def sample_feature_probes(metadata: dict[str, Any], *, n_probes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    feature_names = list(metadata["observed_feature_variables"])
    ranges = metadata["variable_ranges"]
    probes = {}
    for feature in feature_names:
        lower, upper = ranges[feature]
        probes[feature] = rng.uniform(float(lower), float(upper), size=n_probes)
    return pd.DataFrame(probes)


def evaluate_label_responses(
    metadata: dict[str, Any],
    label_table: pd.DataFrame,
    latent_columns: list[str],
    probes: pd.DataFrame,
) -> np.ndarray:
    expression = str(metadata["rhs_expression"])
    feature_names = list(metadata["observed_feature_variables"])
    responses: list[np.ndarray] = []
    for _, label_row in label_table.iterrows():
        latent_assignment = {column: float(label_row[column]) for column in latent_columns}
        values = []
        for _, probe_row in probes.iterrows():
            feature_assignment = {feature: float(probe_row[feature]) for feature in feature_names}
            values.append(evaluate_scalar_expression(expression, {**latent_assignment, **feature_assignment}))
        responses.append(np.asarray(values, dtype=np.float64))
    return np.vstack(responses)


def build_pairwise_rows(
    *,
    labels: np.ndarray,
    learned_q: np.ndarray,
    true_q: np.ndarray,
    responses: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        rows.append(
            {
                "label_i": labels[i],
                "label_j": labels[j],
                "learned_q_distance": float(np.linalg.norm(learned_q[i] - learned_q[j])),
                "true_q_distance": float(np.linalg.norm(true_q[i] - true_q[j])),
                "function_rmse_distance": float(np.sqrt(np.mean((responses[i] - responses[j]) ** 2))),
            }
        )
    return rows


def correlation_metrics(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


if __name__ == "__main__":
    main()
