#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize orthogonality-loss type scan outputs.")
    parser.add_argument("--expression-root", type=Path, default=None)
    parser.add_argument("--application-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expression_root:
        summarize_expression_root(resolve(args.expression_root))
    if args.application_root:
        summarize_application_root(resolve(args.application_root))


def summarize_expression_root(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for method_dir in sorted((root / "expressions").glob("*")):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        for summary_path in (method_dir / "runs").glob("*/run_summary.json"):
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            expression = data.get("expression") or {}
            config = data.get("workflow_config") or {}
            metrics = (data.get("latent_q") or {}).get("metrics") or {}
            rows.append(
                {
                    "orth_type": method,
                    "status": data.get("status"),
                    "expr": expression.get("expression_id") or config.get("expression_id"),
                    "name": expression.get("formula_name"),
                    "true_qdim": expression.get("ground_truth_latent_dim"),
                    "qdim": config.get("q_dim") or (data.get("latent_q") or {}).get("q_dim_model"),
                    "test_r2": metrics.get("test_r2"),
                    "test_mse": metrics.get("test_mse"),
                    "train_r2": metrics.get("train_r2_last_epoch"),
                    "latent_feature_corr_mean_abs": metrics.get("latent_feature_corr_mean_abs"),
                    "latent_feature_corr_max_abs": metrics.get("latent_feature_corr_max_abs"),
                    "epochs_completed": metrics.get("epochs_completed"),
                    "error": data.get("error"),
                    "run_summary": str(summary_path),
                }
            )
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "expression_all_runs.csv", index=False)
    if frame.empty:
        print(f"No expression summaries found under {root}")
        return
    successes = frame[frame["status"] == "success"].copy()
    successes["test_r2"] = pd.to_numeric(successes["test_r2"], errors="coerce")
    successes["latent_feature_corr_mean_abs"] = pd.to_numeric(
        successes["latent_feature_corr_mean_abs"], errors="coerce"
    )
    run_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for method, group in successes.groupby("orth_type"):
        run_rows.append(
            {
                "orth_type": method,
                "success_runs": len(group),
                "exprs": group["expr"].nunique(),
                "mean_r2": group["test_r2"].mean(),
                "median_r2": group["test_r2"].median(),
                "ge99": int((group["test_r2"] >= 0.99).sum()),
                "ge95": int((group["test_r2"] >= 0.95).sum()),
                "ge80": int((group["test_r2"] >= 0.80).sum()),
                "mean_corr": group["latent_feature_corr_mean_abs"].mean(),
            }
        )
        idx = group.groupby("expr")["test_r2"].idxmax()
        best = group.loc[idx].copy()
        counts = {int(key): int(value) for key, value in best["qdim"].value_counts().sort_index().items()}
        best_rows.append(
            {
                "orth_type": method,
                "best_exprs": len(best),
                "best_mean_r2": best["test_r2"].mean(),
                "best_median_r2": best["test_r2"].median(),
                "best_ge99": int((best["test_r2"] >= 0.99).sum()),
                "best_ge95": int((best["test_r2"] >= 0.95).sum()),
                "best_ge80": int((best["test_r2"] >= 0.80).sum()),
                "best_qdim_counts": counts,
            }
        )
    pd.DataFrame(run_rows).sort_values("orth_type").to_csv(root / "expression_method_run_level.csv", index=False)
    pd.DataFrame(best_rows).sort_values("orth_type").to_csv(root / "expression_method_best_per_expr.csv", index=False)
    print(f"Saved expression summaries under {root}")


def summarize_application_root(root: Path) -> None:
    frames: list[pd.DataFrame] = []
    for path in sorted((root / "applications").glob("*/*/*/summary.csv")):
        frame = pd.read_csv(path)
        frame["summary_path"] = str(path)
        frames.append(frame)
    root.mkdir(parents=True, exist_ok=True)
    if not frames:
        print(f"No application summaries found under {root}")
        return
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs.to_csv(root / "application_all_runs.csv", index=False)
    successes = all_runs[all_runs["status"] == "success"].copy()
    successes["test_r2"] = pd.to_numeric(successes["test_r2"], errors="coerce")
    successes["latent_feature_corr_mean_abs"] = pd.to_numeric(
        successes["latent_feature_corr_mean_abs"], errors="coerce"
    )
    method_rows: list[dict[str, Any]] = []
    best_rows: list[pd.DataFrame] = []
    for method, group in successes.groupby("orth_type"):
        method_rows.append(
            {
                "orth_type": method,
                "success_runs": len(group),
                "datasets": group["dataset"].nunique(),
                "mean_r2": group["test_r2"].mean(),
                "median_r2": group["test_r2"].median(),
                "ge95": int((group["test_r2"] >= 0.95).sum()),
                "ge90": int((group["test_r2"] >= 0.90).sum()),
                "mean_corr": group["latent_feature_corr_mean_abs"].mean(),
            }
        )
        idx = group.groupby("dataset")["test_r2"].idxmax()
        best_rows.append(group.loc[idx].copy())
    best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    if not best.empty:
        best.sort_values(["dataset", "test_r2"], ascending=[True, False]).to_csv(
            root / "application_best_per_dataset_method.csv",
            index=False,
        )
    pd.DataFrame(method_rows).sort_values("orth_type").to_csv(root / "application_method_summary.csv", index=False)
    print(f"Saved application summaries under {root}")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


if __name__ == "__main__":
    main()
