#!/usr/bin/env python3
"""Aggregate the frozen NASA support-box-q diagnostic."""

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
        for path in sorted((root / method).glob("**/box_q.csv")):
            metadata = json.loads((path.parent / "cell_summary.json").read_text())
            keys = {
                "dataset": metadata["dataset"],
                "method": metadata["method"],
                "seed": metadata["seed"],
            }
            q_frames.append(pd.read_csv(path).assign(**keys))
            prediction_frames.append(
                pd.read_csv(path.parent / "query_predictions.csv").assign(**keys)
            )

    cells = pd.concat(cell_frames, ignore_index=True).sort_values(
        ["method", "dataset", "seed"]
    )
    q_all = pd.concat(q_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(cells) != 30 or len(q_all) != 150:
        raise ValueError("expected 30 cells and 5 q rows per cell")

    q_columns = ["q1", "q2", "q3", "q4"]
    numeric_columns = [
        "upstream_query_target_leakage_max_q_difference",
        "coordinate_box_max_violation",
        "coordinate_clip_fraction",
        "clip_l2_median",
        "raw_q_validation_max_abs_z",
        "functional_validation_max_abs_z",
        "box_reference_nrmse",
        "unconstrained_reference_nrmse",
        "prediction_nrmse_ratio",
    ]
    finite = bool(
        np.isfinite(cells[numeric_columns].to_numpy(float)).all()
        and np.isfinite(q_all[q_columns].to_numpy(float)).all()
        and np.isfinite(predictions[["target", "prediction"]].to_numpy(float)).all()
    )
    continuity = cells.loc[cells.method == CONTINUITY]
    median_box = float(continuity.box_reference_nrmse.median())
    median_unconstrained = float(continuity.unconstrained_reference_nrmse.median())
    gates = {
        "gate_1_integrity": bool(
            finite
            and (cells.anchor_labels == 8).all()
            and (cells.structure_validation_labels == 5).all()
            and cells.upstream_query_target_leakage_max_q_difference.max() == 0.0
            and cells.coordinate_box_max_violation.max() <= 1e-7
        ),
        "gate_2_box_containment": bool(
            (continuity.raw_q_validation_max_abs_z <= 3.0).all()
        ),
        "gate_3_functional_shift": bool(
            continuity.functional_validation_max_abs_z.median() <= 3.0
            and int((continuity.functional_validation_max_abs_z <= 6.0).sum()) >= 12
        ),
        "gate_4_prediction_retention": bool(
            median_box <= 1.05 * median_unconstrained
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
            box_nrmse_median=("box_reference_nrmse", "median"),
            unconstrained_nrmse_median=("unconstrained_reference_nrmse", "median"),
            nrmse_ratio_median=("prediction_nrmse_ratio", "median"),
            retained_cells=("prediction_nrmse_ratio", lambda values: int((values <= 1.10).sum())),
            safe_functional_cells=(
                "functional_validation_max_abs_z",
                lambda values: int((values <= 6.0).sum()),
            ),
            coordinate_clip_fraction_median=("coordinate_clip_fraction", "median"),
            clip_l2_median=("clip_l2_median", "median"),
        )
        .sort_values("method")
    )

    cells.to_csv(root / "all_cells.csv", index=False)
    q_all.to_csv(root / "all_box_q.csv", index=False)
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
                f"{row.box_nrmse_median:.4g}",
                f"{row.unconstrained_nrmse_median:.4g}",
                f"{row.nrmse_ratio_median:.4g}",
                f"{row.retained_cells}/15",
                f"{row.safe_functional_cells}/15",
                f"{row.coordinate_clip_fraction_median:.3g}",
            ]
        )
    report = [
        "# NASA support-box q 诊断",
        "",
        f"**冻结判定：** {'ADVANCE' if gates['advance_to_bounded_symbolic_stage_c2'] else 'DO NOT ADVANCE'}",
        "",
        "## 设计",
        "",
        "每个 structure-validation q 先按既有 support-only 协议校准，再逐坐标裁剪到同一 cell 八个 support-matched meta-fit q 的最小/最大范围。该变换无训练、无超参数，也不选择某一个训练实体；query target 只参与最终预测计分。",
        "",
        "## 汇总",
        "",
        *_table(
            [
                "方法",
                "raw max|z| 中位/最大",
                "functional max|z| 中位/最大",
                "box NRMSE",
                "未约束 NRMSE",
                "cell ratio 中位",
                "预测保持",
                "functional 安全",
                "坐标裁剪比例",
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
        "本轮是连续使用同一 inner cells 的开发诊断。只有四个 gate 全部通过，才可另行冻结有界 symbolic Stage C2；即使通过，也不能把本轮称为独立确认。",
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
