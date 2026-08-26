#!/usr/bin/env python3
"""Apply the frozen gates to the NASA information-matched prefix-q pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHODS = ("prefix_q_continuity_step1", "prefix_q_mse_step1")
CONTINUITY = "prefix_q_continuity_step1"
BASELINE_METHOD = {
    "prefix_q_continuity_step1": "joint_continuity_step1",
    "prefix_q_mse_step1": "joint_mse_step1",
}
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _max_abs_z(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
) -> float:
    mean = train[columns].mean().to_numpy(float)
    std = np.maximum(train[columns].std(ddof=0).to_numpy(float), 1e-8)
    return float(
        np.abs((validation[columns].to_numpy(float) - mean) / std).max()
    )


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    baseline_root = args.baseline_root.resolve()
    rows = []
    for result_path in sorted(root.glob("*/*/seed*_q*/result.json")):
        payload = json.loads(result_path.read_text())
        job = payload["job"]
        if payload["status"] != "success" or job["method"] not in METHODS:
            raise ValueError(f"unexpected result: {result_path}")
        config = payload["latent_config"]
        counters = payload["optimization_counters"]
        train_q = pd.read_csv(_resolve(payload["artifacts"]["train_label_q"]))
        test_q = pd.read_csv(_resolve(payload["artifacts"]["test_label_q"]))
        q_columns = [column for column in train_q if column.startswith("q")]
        relative_parent = result_path.relative_to(root).parent
        coordinates = pd.read_csv(
            root
            / "functional_coordinate_analysis"
            / relative_parent
            / "functional_coordinates.csv"
        )
        train_functional = coordinates.loc[coordinates.split == "train"]
        validation_functional = coordinates.loc[coordinates.split == "outer_test"]
        rows.append(
            {
                "status": payload["status"],
                "dataset": job["dataset"],
                "method": job["method"],
                "seed": int(job["seed"]),
                "reference_nrmse": float(payload["prediction"]["reference_nrmse"]),
                "train_labels": int(len(train_q)),
                "validation_labels": int(len(test_q)),
                "optimization_schedule": config["optimization_schedule"],
                "q_training_split_mode": config["q_training_split_mode"],
                "q_training_ratio": float(config["q_training_ratio"]),
                "q_training_order_feature_index": int(
                    config["q_training_order_feature_index"]
                ),
                "theta_steps": int(counters["theta_steps"]),
                "q_steps": int(counters["q_steps"]),
                "backward_passes": int(counters["backward_passes"]),
                "raw_q_validation_max_abs_z": _max_abs_z(
                    train_q, test_q, q_columns
                ),
                "functional_validation_max_abs_z": _max_abs_z(
                    train_functional,
                    validation_functional,
                    list(FUNCTIONAL_COLUMNS),
                ),
                "result_path": str(result_path.relative_to(PROJECT_ROOT)),
            }
        )
    cells = pd.DataFrame(rows).sort_values(["method", "dataset", "seed"])
    if len(cells) != 30 or set(cells.method) != set(METHODS):
        raise ValueError(f"expected 30 cells across {METHODS}")

    baseline = pd.read_csv(baseline_root / "all_cells.csv")[
        ["dataset", "method", "seed", "recalibrated_validation_reference_nrmse"]
    ].rename(
        columns={"recalibrated_validation_reference_nrmse": "baseline_reference_nrmse"}
    )
    cells["baseline_method"] = cells.method.map(BASELINE_METHOD)
    cells = cells.merge(
        baseline,
        left_on=["dataset", "baseline_method", "seed"],
        right_on=["dataset", "method", "seed"],
        suffixes=("", "_baseline"),
        validate="one_to_one",
    ).drop(columns=["method_baseline"])
    cells["prediction_nrmse_ratio"] = (
        cells.reference_nrmse / cells.baseline_reference_nrmse
    )

    analysis_root = root / "functional_coordinate_analysis"
    q_stability = pd.read_csv(
        analysis_root / "cross_split_q_distance_stability_summary.csv"
    )
    functional_stability = pd.read_csv(
        analysis_root / "cross_split_functional_stability_summary.csv"
    )
    continuity = cells.loc[cells.method == CONTINUITY]
    continuity_q_stability = q_stability.loc[q_stability.method == CONTINUITY].iloc[0]
    continuity_functional = functional_stability.loc[
        (functional_stability.method == CONTINUITY)
        & functional_stability.coordinate.isin(FUNCTIONAL_COLUMNS)
    ].set_index("coordinate")

    numeric = [
        "reference_nrmse",
        "baseline_reference_nrmse",
        "prediction_nrmse_ratio",
        "raw_q_validation_max_abs_z",
        "functional_validation_max_abs_z",
    ]
    gates = {
        "gate_1_integrity": bool(
            np.isfinite(cells[numeric].to_numpy(float)).all()
            and (cells.train_labels == 8).all()
            and (cells.validation_labels == 5).all()
            and (cells.optimization_schedule == "alternating").all()
            and (cells.q_training_split_mode == "prefix").all()
            and np.allclose(cells.q_training_ratio, 0.3)
            and (cells.q_training_order_feature_index == 0).all()
            and (cells.theta_steps == cells.q_steps).all()
            and (cells.backward_passes == cells.theta_steps + cells.q_steps).all()
        ),
        "gate_2_prediction_retention": bool(
            continuity.reference_nrmse.median()
            <= 1.05 * continuity.baseline_reference_nrmse.median()
            and int((continuity.prediction_nrmse_ratio <= 1.10).sum()) >= 10
        ),
        "gate_3_interface_safety": bool(
            continuity.raw_q_validation_max_abs_z.median() <= 3.0
            and continuity.functional_validation_max_abs_z.median() <= 3.0
            and int((continuity.functional_validation_max_abs_z <= 6.0).sum()) >= 12
        ),
        "gate_4_representation_stability": bool(
            continuity_q_stability.min_split_median >= 0.80
            and (
                continuity_functional.median_of_split_medians >= 0.70
            ).all()
            and (continuity_functional.min_split_median >= 0.50).all()
        ),
    }
    gates["advance_to_bounded_symbolic_stage_c2"] = bool(all(gates.values()))

    summary = (
        cells.groupby("method", as_index=False)
        .agg(
            cells=("seed", "size"),
            nrmse_median=("reference_nrmse", "median"),
            baseline_nrmse_median=("baseline_reference_nrmse", "median"),
            ratio_median=("prediction_nrmse_ratio", "median"),
            retained_cells=("prediction_nrmse_ratio", lambda values: int((values <= 1.10).sum())),
            raw_z_median=("raw_q_validation_max_abs_z", "median"),
            raw_z_max=("raw_q_validation_max_abs_z", "max"),
            functional_z_median=("functional_validation_max_abs_z", "median"),
            functional_z_max=("functional_validation_max_abs_z", "max"),
            wall_backward_passes=("backward_passes", "median"),
        )
        .sort_values("method")
    )

    cells.to_csv(root / "pilot_cells.csv", index=False)
    summary.to_csv(root / "pilot_method_summary.csv", index=False)
    (root / "gate_decision.json").write_text(json.dumps(gates, indent=2))

    table_rows = []
    for row in summary.itertuples(index=False):
        table_rows.append(
            [
                row.method,
                f"{row.nrmse_median:.4g}",
                f"{row.baseline_nrmse_median:.4g}",
                f"{row.ratio_median:.4g}",
                f"{row.retained_cells}/15",
                f"{row.raw_z_median:.4g} / {row.raw_z_max:.4g}",
                f"{row.functional_z_median:.4g} / {row.functional_z_max:.4g}",
                f"{row.wall_backward_passes:.0f}",
            ]
        )
    stability_rows = [
        [
            "q distance",
            f"{continuity_q_stability.median_of_split_medians:.4g}",
            f"{continuity_q_stability.min_split_median:.4g}",
        ]
    ]
    for coordinate in FUNCTIONAL_COLUMNS:
        row = continuity_functional.loc[coordinate]
        stability_rows.append(
            [
                coordinate,
                f"{row.median_of_split_medians:.4g}",
                f"{row.min_split_median:.4g}",
            ]
        )
    report = [
        "# NASA information-matched prefix-q training pilot",
        "",
        f"**冻结判定：** {'ADVANCE' if gates['advance_to_bounded_symbolic_stage_c2'] else 'DO NOT ADVANCE'}",
        "",
        "## 方法",
        "",
        "训练时 q 每个 batch 只由各实体最早 30% 行更新，随后 decoder 用完整 batch 更新；测试 q 仍只由前 30% support 校准。continuity 的训练实体响应距离同样只由前缀计算。",
        "",
        "## 预测与接口",
        "",
        *_table(
            [
                "方法",
                "NRMSE",
                "旧基线",
                "ratio",
                "保持 cells",
                "raw max|z| 中位/最大",
                "functional max|z| 中位/最大",
                "反传/ cell",
            ],
            table_rows,
        ),
        "",
        "## Continuity 表征稳定性",
        "",
        *_table(["对象", "split 中位的中位", "最差 split 中位"], stability_rows),
        "",
        "## Gate",
        "",
        *_table(
            ["Gate", "结果"],
            [[key, "PASS" if value else "FAIL"] for key, value in gates.items()],
        ),
        "",
        "本轮使用已参与方法开发的 inner splits，只能决定是否值得进入下一开发阶段，不能作为独立确认性证据。",
    ]
    (root / "PREFIX_Q_TRAINING_REPORT.md").write_text("\n".join(report) + "\n")
    (root / "status.json").write_text(
        json.dumps(
            {
                "state": "completed_all",
                "planned": 30,
                "success": 30,
                "failed": 0,
                "advance": gates["advance_to_bounded_symbolic_stage_c2"],
            },
            indent=2,
        )
    )
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
