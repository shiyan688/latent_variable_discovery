#!/usr/bin/env python3
"""Aggregate the frozen NASA convex-support-q diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("joint_continuity_step1", "joint_mse_step1")
CONTINUITY = "joint_continuity_step1"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cell_frames = []
    q_frames = []
    weight_frames = []
    prediction_frames = []
    for method in METHODS:
        status = json.loads((root / method / "status.json").read_text())
        expected = {"state": "completed_all", "planned": 15, "success": 15, "failed": 0}
        if status != expected:
            raise ValueError(f"nonterminal method status for {method}: {status}")
        cells = pd.read_csv(root / method / "cell_summary.csv")
        if len(cells) != 15 or (cells.status != "success").any():
            raise ValueError(f"expected 15 successful cells for {method}")
        cell_frames.append(cells)
        for path in sorted((root / method).glob("**/convex_q.csv")):
            metadata = json.loads((path.parent / "cell_summary.json").read_text())
            keys = {
                "dataset": metadata["dataset"],
                "method": metadata["method"],
                "seed": metadata["seed"],
            }
            q_frames.append(pd.read_csv(path).assign(**keys))
            weight_frames.append(
                pd.read_csv(path.parent / "convex_weights.csv").assign(**keys)
            )
            prediction_frames.append(
                pd.read_csv(path.parent / "query_predictions.csv").assign(**keys)
            )

    cells = pd.concat(cell_frames, ignore_index=True).sort_values(
        ["method", "dataset", "seed"]
    )
    q_all = pd.concat(q_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(cells) != 30 or len(q_all) != 150 or len(weights) != 1200:
        raise ValueError("expected 30 cells, 5 q rows and 40 weights per cell")

    numeric_columns = [
        "query_target_leakage_max_q_difference",
        "convex_weight_sum_max_abs_error",
        "convex_min_weight",
        "raw_q_validation_max_abs_z",
        "functional_validation_max_abs_z",
        "convex_reference_nrmse",
        "unconstrained_reference_nrmse",
        "prediction_nrmse_ratio",
        "effective_anchors_median",
        "max_anchor_weight_median",
    ]
    finite = bool(np.isfinite(cells[numeric_columns].to_numpy(float)).all())
    continuity = cells.loc[cells.method == CONTINUITY]
    continuity_median_convex = float(continuity.convex_reference_nrmse.median())
    continuity_median_unconstrained = float(
        continuity.unconstrained_reference_nrmse.median()
    )
    gates = {
        "gate_1_integrity": bool(
            finite
            and (cells.anchor_labels == 8).all()
            and (cells.structure_validation_labels == 5).all()
            and cells.query_target_leakage_max_q_difference.max() == 0.0
            and cells.convex_weight_sum_max_abs_error.max() <= 1e-6
            and cells.convex_min_weight.min() >= 0.0
        ),
        "gate_2_convex_containment": bool(
            (continuity.raw_q_validation_max_abs_z <= 3.0).all()
        ),
        "gate_3_functional_shift": bool(
            continuity.functional_validation_max_abs_z.median() <= 3.0
            and int((continuity.functional_validation_max_abs_z <= 6.0).sum()) >= 12
        ),
        "gate_4_prediction_retention": bool(
            continuity_median_convex <= 1.05 * continuity_median_unconstrained
            and int((continuity.prediction_nrmse_ratio <= 1.10).sum()) >= 10
        ),
    }
    gates["advance_to_bounded_symbolic_stage_c2"] = bool(all(gates.values()))

    summary = (
        cells.groupby("method", as_index=False)
        .agg(
            cells=("seed", "size"),
            raw_z_median=("raw_q_validation_max_abs_z", "median"),
            raw_z_max=("raw_q_validation_max_abs_z", "max"),
            functional_z_median=("functional_validation_max_abs_z", "median"),
            functional_z_max=("functional_validation_max_abs_z", "max"),
            convex_nrmse_median=("convex_reference_nrmse", "median"),
            unconstrained_nrmse_median=("unconstrained_reference_nrmse", "median"),
            nrmse_ratio_median=("prediction_nrmse_ratio", "median"),
            retained_cells=("prediction_nrmse_ratio", lambda values: int((values <= 1.10).sum())),
            safe_functional_cells=(
                "functional_validation_max_abs_z",
                lambda values: int((values <= 6.0).sum()),
            ),
            effective_anchors_median=("effective_anchors_median", "median"),
            max_anchor_weight_median=("max_anchor_weight_median", "median"),
        )
        .sort_values("method")
    )

    cells.to_csv(root / "all_cells.csv", index=False)
    q_all.to_csv(root / "all_convex_q.csv", index=False)
    weights.to_csv(root / "all_convex_weights.csv", index=False)
    predictions.to_csv(root / "all_query_predictions.csv", index=False)
    summary.to_csv(root / "method_summary.csv", index=False)
    (root / "gate_decision.json").write_text(json.dumps(gates, indent=2))

    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            [
                row.method,
                f"{row.raw_z_median:.4g} / {row.raw_z_max:.4g}",
                f"{row.functional_z_median:.4g} / {row.functional_z_max:.4g}",
                f"{row.convex_nrmse_median:.4g}",
                f"{row.unconstrained_nrmse_median:.4g}",
                f"{row.nrmse_ratio_median:.4g}",
                f"{row.retained_cells}/15",
                f"{row.safe_functional_cells}/15",
                f"{row.effective_anchors_median:.3g}",
            ]
        )
    report = [
        "# NASA convex-support q 诊断",
        "",
        f"**冻结判定：** {'ADVANCE' if gates['advance_to_bounded_symbolic_stage_c2'] else 'DO NOT ADVANCE'}",
        "",
        "## 设计",
        "",
        "30 个 decoder/checkpoint 均保持冻结。每个 structure-validation 电池只用前 30% support 目标，把 q 校准为同一 cell 八个 support-matched meta-fit q 锚点的 softmax 凸组合；query target 扰动只用于泄漏审计。未约束 support-matched q 是预先冻结的预测保持比较对象。",
        "",
        "## 汇总",
        "",
        *_table(
            [
                "方法",
                "raw max|z| 中位/最大",
                "functional max|z| 中位/最大",
                "凸约束 NRMSE",
                "未约束 NRMSE",
                "cell ratio 中位",
                "预测保持",
                "functional 安全",
                "有效锚点",
            ],
            rows,
        ),
        "",
        "## Gate",
        "",
        *_table(
            ["Gate", "结果"],
            [[key, "PASS" if value else "FAIL"] for key, value in gates.items()],
        ),
        "",
        "只有四个预声明 gate 全部通过，才可另行冻结有界 symbolic Stage C2。凸组合消除 raw-q 外插不等于已证明下游符号价值；预测保持 gate 用于排除仅靠牺牲任务拟合换取坐标安全的情形。",
    ]
    (root / "DIAGNOSTIC_REPORT.md").write_text("\n".join(report) + "\n")
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
