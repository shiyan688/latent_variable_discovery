#!/usr/bin/env python3
"""Aggregate the frozen NASA support-matched-q diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHODS = ("joint_continuity_step1", "joint_mse_step1")
LEGACY_CONTINUITY_RAW_Z = 22.1915
LEGACY_CONTINUITY_FUNCTIONAL_Z = 7.3658


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
    frames = []
    q_frames = []
    jacobian_frames = []
    for method in METHODS:
        status = json.loads((root / method / "status.json").read_text())
        if status != {"state": "completed_all", "planned": 15, "success": 15, "failed": 0}:
            raise ValueError(f"nonterminal method status for {method}: {status}")
        frame = pd.read_csv(root / method / "cell_summary.csv")
        if len(frame) != 15 or (frame.status != "success").any():
            raise ValueError(f"expected 15 successful cells for {method}")
        frames.append(frame)
        for path in sorted((root / method).glob("**/support_matched_q.csv")):
            cell = json.loads((path.parent / "cell_summary.json").read_text())
            q_frames.append(
                pd.read_csv(path).assign(
                    dataset=cell["dataset"], method=cell["method"], seed=cell["seed"]
                )
            )
        for path in sorted((root / method).glob("**/support_jacobians.csv")):
            cell = json.loads((path.parent / "cell_summary.json").read_text())
            jacobian_frames.append(
                pd.read_csv(path).assign(
                    dataset=cell["dataset"], method=cell["method"], seed=cell["seed"]
                )
            )

    cells = pd.concat(frames, ignore_index=True).sort_values(
        ["method", "dataset", "seed"]
    )
    q_all = pd.concat(q_frames, ignore_index=True)
    jacobians = pd.concat(jacobian_frames, ignore_index=True)
    if len(cells) != 30 or len(q_all) != 390 or len(jacobians) != 390:
        raise ValueError("expected 30 cells and 13 labels per cell")

    continuity = cells.loc[cells.method == "joint_continuity_step1"]
    gates = {
        "gate_1_integrity": bool(
            len(cells) == 30
            and cells.query_target_leakage_max_q_difference.max() == 0.0
            and (cells.meta_fit_labels == 8).all()
            and (cells.structure_validation_labels == 5).all()
        ),
        "gate_2_reproduction": bool(
            cells.test_q_reproduction_max_abs.max() <= 1e-4
            and cells.validation_reference_nrmse_abs_difference.max() <= 1e-5
        ),
        "gate_3_continuity_shift": bool(
            continuity.matched_raw_q_validation_max_abs_z.median()
            <= LEGACY_CONTINUITY_RAW_Z / 2.0
            and continuity.matched_functional_validation_max_abs_z.median() <= 3.0
        ),
        "gate_4_continuity_tail": bool(
            np.isfinite(continuity.matched_functional_validation_max_abs_z).all()
            and int((continuity.matched_functional_validation_max_abs_z <= 6.0).sum()) >= 12
        ),
    }
    gates["advance_to_bounded_symbolic_stage_c2"] = bool(all(gates.values()))

    summary = (
        cells.groupby("method", as_index=False)
        .agg(
            cells=("seed", "size"),
            raw_z_median=("matched_raw_q_validation_max_abs_z", "median"),
            raw_z_max=("matched_raw_q_validation_max_abs_z", "max"),
            functional_z_median=("matched_functional_validation_max_abs_z", "median"),
            functional_z_max=("matched_functional_validation_max_abs_z", "max"),
            nearest_q_median=("matched_q_nearest_distance_median", "median"),
            train_q_displacement_median=(
                "meta_fit_q_displacement_from_full_curve_median",
                "median",
            ),
            jacobian_smin_median=("support_jacobian_smin_median", "median"),
            jacobian_condition_median=("support_jacobian_condition_median", "median"),
            jacobian_rank_median=("support_jacobian_effective_rank_median", "median"),
            meta_fit_prefix_nrmse_median=("meta_fit_prefix_reference_nrmse", "median"),
            validation_nrmse_median=("recalibrated_validation_reference_nrmse", "median"),
        )
        .sort_values("method")
    )

    root.mkdir(parents=True, exist_ok=True)
    cells.to_csv(root / "all_cells.csv", index=False)
    q_all.to_csv(root / "all_support_matched_q.csv", index=False)
    jacobians.to_csv(root / "all_support_jacobians.csv", index=False)
    summary.to_csv(root / "method_summary.csv", index=False)
    (root / "gate_decision.json").write_text(json.dumps(gates, indent=2))

    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            [
                row.method,
                f"{row.raw_z_median:.4g}",
                f"{row.functional_z_median:.4g}",
                f"{row.nearest_q_median:.4g}",
                f"{row.jacobian_smin_median:.4g}",
                f"{row.jacobian_condition_median:.4g}",
                f"{row.jacobian_rank_median:.2f}",
                f"{row.validation_nrmse_median:.4g}",
            ]
        )
    report = [
        "# NASA support-matched q 接口诊断",
        "",
        f"**冻结判定：** {'ADVANCE' if gates['advance_to_bounded_symbolic_stage_c2'] else 'DO NOT ADVANCE'}",
        "",
        "## 设计",
        "",
        "30 个既有 checkpoint 均不重新训练 decoder。每个 meta-fit 电池只用前 30% support 重新校准 q，并从 q 先验中排除该实体自己的 full-curve embedding；structure-validation 电池仍使用全部 8 个 meta-fit embedding 构成先验。query target 扰动只用于泄漏审计。",
        "",
        "## 汇总",
        "",
        *_table(
            ["方法", "raw-q max|z| 中位", "functional max|z| 中位", "最近 q 距离", "Jacobian smin", "条件数", "有效秩", "validation NRMSE"],
            rows,
        ),
        "",
        f"旧 Stage C continuity raw/functional max-|z| 中位数分别为 {LEGACY_CONTINUITY_RAW_Z:.4f}/{LEGACY_CONTINUITY_FUNCTIONAL_Z:.4f}。当前冻结 gate 要求 raw 至少减半，functional 中位数不超过 3，且至少 12/15 个 functional cells 不超过 6。",
        "",
        "## Gate",
        "",
        *_table(
            ["Gate", "结果"],
            [[key, "PASS" if value else "FAIL"] for key, value in gates.items()],
        ),
        "",
        "support Jacobian 奇异值、条件数和有效秩是诊断量，不参与本轮 advancement gate；它们用于判断 q 校准困难是输入分布错配还是局部不可辨识。只有全部四个冻结 gate 通过，才独立设计有界 symbolic Stage C2；本分析不会自动启动符号回归或结构化 decoder。",
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
