#!/usr/bin/env python3
"""Audit and summarize the frozen NASA Stage-C symbolic experiment."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

import run_nasa_inner_symbolic_structure_20260825 as runner


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUNCTIONAL_COLUMNS = ("capacity_cycle1", "early_fade_rate")
PAIR_KEYS = ["dataset", "seed"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _method_label(value: object) -> str:
    return "baseline" if pd.isna(value) else str(value)


def _select(frame: pd.DataFrame, method: str, regime: str) -> pd.DataFrame:
    labels = frame["method"].map(_method_label)
    selected = frame.loc[(labels == method) & (frame["regime"] == regime)].copy()
    return selected.set_index(PAIR_KEYS).sort_index()


def _paired_comparison(
    frame: pd.DataFrame,
    candidate_method: str,
    candidate_regime: str,
    anchor_method: str,
    anchor_regime: str,
) -> dict[str, object]:
    candidate = _select(frame, candidate_method, candidate_regime)
    anchor = _select(frame, anchor_method, anchor_regime)
    if not candidate.index.equals(anchor.index) or len(candidate) != 15:
        raise ValueError("paired symbolic comparison must contain the same 15 cells")
    delta = (
        candidate["reference_nrmse_structure_validation"]
        - anchor["reference_nrmse_structure_validation"]
    )
    return {
        "candidate_method": candidate_method,
        "candidate_regime": candidate_regime,
        "anchor_method": anchor_method,
        "anchor_regime": anchor_regime,
        "n": len(delta),
        "wins": int((delta < 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta > 0).sum()),
        "median_nrmse_delta": float(delta.median()),
        "median_relative_delta": float(
            (
                candidate["reference_nrmse_structure_validation"]
                / anchor["reference_nrmse_structure_validation"]
                - 1.0
            ).median()
        ),
    }


def _expression_motifs(expression: str) -> dict[str, object]:
    names = [
        "discharge_index",
        "ambient_temperature",
        "load_current_amp",
        "cutoff_voltage",
        *FUNCTIONAL_COLUMNS,
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    symbols = {name: sp.Symbol(name) for name in names}
    parsed = sp.sympify(expression, locals=symbols)
    used = {str(symbol) for symbol in parsed.free_symbols}
    discharge = symbols["discharge_index"]
    derivative = sp.diff(parsed, discharge)
    slope_coordinates = [
        name
        for name in FUNCTIONAL_COLUMNS
        if symbols[name] in parsed.free_symbols
        and sp.simplify(sp.diff(derivative, symbols[name])) != 0
    ]
    return {
        "uses_discharge_and_functional": (
            "discharge_index" in used and bool(used & set(FUNCTIONAL_COLUMNS))
        ),
        "functional_slope_modulation": bool(slope_coordinates),
        "slope_modulating_coordinates": ";".join(slope_coordinates),
        "uses_exp": bool(parsed.has(sp.exp)),
        "uses_division": "/" in expression,
        "uses_square_or_power": "**" in expression,
    }


def _audit_and_diagnostics(root: Path, frame: pd.DataFrame, manifest: dict) -> tuple[dict, pd.DataFrame]:
    result_paths = sorted(root.glob("**/result.json"))
    prediction_paths = sorted(root.glob("**/predictions.csv"))
    pareto_paths = sorted(root.glob("**/pareto_front.csv"))
    scaler_paths = sorted(root.glob("**/input_scaler.csv"))
    status_lines = [line for line in (root / "status.jsonl").read_text().splitlines() if line]
    status = _read_json(root / "status.json")
    q_root = Path(manifest["config"]["q_root"])
    records = runner._prepared_records(q_root)

    expected_keys = {
        (cell["dataset"], int(cell["seed"]), cell["method"], cell["regime"])
        for cell in manifest["cells"]
    }
    observed_keys: set[tuple[object, ...]] = set()
    diagnostic_rows: list[dict[str, object]] = []
    finite_metrics = True
    finite_predictions = True
    label_isolation = True
    required_artifacts = True

    metric_names = (
        "complexity",
        "r2_meta_fit",
        "r2_structure_validation",
        "reference_nrmse_meta_fit",
        "reference_nrmse_structure_validation",
    )
    for path in result_paths:
        result = _read_json(path)
        key = (
            result["dataset"],
            int(result["seed"]),
            result["method"],
            result["regime"],
        )
        observed_keys.add(key)
        finite_metrics &= all(np.isfinite(float(result[name])) for name in metric_names)
        meta_labels = set(result["meta_fit_labels"])
        validation_labels = set(result["structure_validation_labels"])
        label_isolation &= (
            len(meta_labels) == 8
            and len(validation_labels) == 5
            and not (meta_labels & validation_labels)
        )
        required_artifacts &= all(
            (path.parent / name).exists()
            for name in ("predictions.csv", "pareto_front.csv", "input_scaler.csv")
        )
        predictions = pd.read_csv(path.parent / "predictions.csv")
        finite_predictions &= np.isfinite(
            predictions.loc[:, ["target", "prediction"]].to_numpy(float)
        ).all()

        bundle = runner._build_bundle(
            q_root=q_root,
            records=records,
            dataset=result["dataset"],
            seed=int(result["seed"]),
            method=result["method"],
            condition_columns=manifest["config"]["condition_columns"],
            functional_columns=manifest["config"]["functional_columns"],
            support_ratio=manifest["config"]["support_ratio"],
            support_order_column=manifest["config"]["support_order_column"],
        )
        columns = result["input_columns"]
        train_inputs = bundle["train"].loc[:, columns].to_numpy(float)
        validation_inputs = bundle["validation"].loc[:, columns].to_numpy(float)
        mean = train_inputs.mean(axis=0)
        std = train_inputs.std(axis=0)
        validation_z = (validation_inputs - mean) / std
        validation_predictions = predictions.loc[
            predictions["symbolic_split"] == "structure_validation", "prediction"
        ].to_numpy(float)
        motifs = _expression_motifs(result["best_expression_standardized"])
        row: dict[str, object] = {
            "dataset": result["dataset"],
            "seed": int(result["seed"]),
            "method": result["method"] or "baseline",
            "regime": result["regime"],
            "reference_nrmse_meta_fit": float(result["reference_nrmse_meta_fit"]),
            "reference_nrmse_structure_validation": float(
                result["reference_nrmse_structure_validation"]
            ),
            "complexity": int(result["complexity"]),
            "max_abs_validation_z": float(np.abs(validation_z).max()),
            "max_abs_validation_prediction": float(np.abs(validation_predictions).max()),
            "catastrophic_nrmse_gt_10": bool(
                result["reference_nrmse_structure_validation"] > 10
            ),
            "catastrophic_nrmse_gt_100": bool(
                result["reference_nrmse_structure_validation"] > 100
            ),
            "expression": result["best_expression_standardized"],
            **motifs,
        }
        for index, column in enumerate(columns):
            row[f"max_abs_z__{column}"] = float(np.abs(validation_z[:, index]).max())
        diagnostic_rows.append(row)

    integrity = {
        "status_completed_all": status == {
            "state": "completed_all",
            "planned": 90,
            "success": 90,
            "failed": 0,
        },
        "result_json_count": len(result_paths),
        "prediction_csv_count": len(prediction_paths),
        "pareto_front_count": len(pareto_paths),
        "input_scaler_count": len(scaler_paths),
        "status_line_count": len(status_lines),
        "results_csv_rows": len(frame),
        "unique_expected_cells": len(expected_keys),
        "unique_observed_cells": len(observed_keys),
        "cell_sets_equal": expected_keys == observed_keys,
        "all_metrics_finite": bool(finite_metrics),
        "all_predictions_finite": bool(finite_predictions),
        "exact_8_5_entity_isolation": bool(label_isolation),
        "all_required_cell_artifacts_present": bool(required_artifacts),
        "audited_q_sources": manifest["audit"]["audited_q_sources"],
        "max_query_target_feature_difference": manifest["audit"][
            "max_query_target_feature_difference"
        ],
        "prefix_order_verified": manifest["audit"]["prefix_order_verified"],
    }
    integrity["passed"] = bool(
        integrity["status_completed_all"]
        and all(integrity[name] == 90 for name in (
            "result_json_count",
            "prediction_csv_count",
            "pareto_front_count",
            "input_scaler_count",
            "status_line_count",
            "results_csv_rows",
            "unique_expected_cells",
            "unique_observed_cells",
        ))
        and integrity["cell_sets_equal"]
        and integrity["all_metrics_finite"]
        and integrity["all_predictions_finite"]
        and integrity["exact_8_5_entity_isolation"]
        and integrity["all_required_cell_artifacts_present"]
        and integrity["audited_q_sources"] == 30
        and integrity["max_query_target_feature_difference"] == 0.0
        and integrity["prefix_order_verified"]
    )
    return integrity, pd.DataFrame(diagnostic_rows).sort_values(
        ["dataset", "seed", "method", "regime"]
    )


def _format_number(value: float) -> str:
    if abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "nasa_battery_reviewer_clean_inner_symbolic_20260825",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "manifest.json")
    frame = pd.read_csv(root / "results.csv")
    if len(frame) != 90 or (frame["status"] != "success").any():
        raise ValueError("expected 90 successful rows")

    integrity, diagnostics = _audit_and_diagnostics(root, frame, manifest)
    (root / "integrity_audit.json").write_text(json.dumps(integrity, indent=2))
    diagnostics.to_csv(root / "cell_diagnostics.csv", index=False)

    summary = (
        diagnostics.groupby(["method", "regime"], as_index=False)
        .agg(
            n=("reference_nrmse_structure_validation", "size"),
            nrmse_median=("reference_nrmse_structure_validation", "median"),
            nrmse_q1=("reference_nrmse_structure_validation", lambda x: x.quantile(0.25)),
            nrmse_q3=("reference_nrmse_structure_validation", lambda x: x.quantile(0.75)),
            nrmse_max=("reference_nrmse_structure_validation", "max"),
            meta_fit_nrmse_median=("reference_nrmse_meta_fit", "median"),
            complexity_median=("complexity", "median"),
            validation_z_median=("max_abs_validation_z", "median"),
            validation_z_max=("max_abs_validation_z", "max"),
            nrmse_gt_10=("catastrophic_nrmse_gt_10", "sum"),
            nrmse_gt_100=("catastrophic_nrmse_gt_100", "sum"),
        )
        .sort_values(["method", "regime"])
    )
    summary.to_csv(root / "method_regime_summary.csv", index=False)

    comparisons = pd.DataFrame(
        [
            _paired_comparison(frame, "joint_continuity_step1", "condition_functional_q", "baseline", "condition_only"),
            _paired_comparison(frame, "joint_continuity_step1", "condition_functional_q", "baseline", "condition_support_stats"),
            _paired_comparison(frame, "joint_continuity_step1", "condition_functional_q", "joint_continuity_step1", "condition_raw_q"),
            _paired_comparison(frame, "joint_mse_step1", "condition_functional_q", "joint_mse_step1", "condition_raw_q"),
            _paired_comparison(frame, "joint_continuity_step1", "condition_functional_q", "joint_mse_step1", "condition_functional_q"),
        ]
    )
    comparisons.to_csv(root / "paired_comparisons.csv", index=False)

    functional = diagnostics.loc[diagnostics["regime"] == "condition_functional_q"].copy()
    motifs = (
        functional.groupby("method", as_index=False)
        .agg(
            n=("seed", "size"),
            discharge_plus_functional=("uses_discharge_and_functional", "sum"),
            functional_slope_modulation=("functional_slope_modulation", "sum"),
            uses_exp=("uses_exp", "sum"),
            uses_division=("uses_division", "sum"),
            uses_square_or_power=("uses_square_or_power", "sum"),
        )
        .sort_values("method")
    )
    split_motifs = (
        functional.groupby(["method", "dataset"], as_index=False)
        .agg(
            n=("seed", "size"),
            discharge_plus_functional=("uses_discharge_and_functional", "sum"),
            functional_slope_modulation=("functional_slope_modulation", "sum"),
        )
        .sort_values(["method", "dataset"])
    )
    motifs.to_csv(root / "motif_recurrence.csv", index=False)
    split_motifs.to_csv(root / "split_motif_recurrence.csv", index=False)

    continuity_summary = summary.set_index(["method", "regime"])
    continuity_functional = continuity_summary.loc[("joint_continuity_step1", "condition_functional_q")]
    continuity_raw = continuity_summary.loc[("joint_continuity_step1", "condition_raw_q")]
    condition_only = continuity_summary.loc[("baseline", "condition_only")]
    support_stats = continuity_summary.loc[("baseline", "condition_support_stats")]
    paired = comparisons.set_index(["anchor_method", "anchor_regime"])
    continuity_motifs = int(
        motifs.set_index("method").loc["joint_continuity_step1", "discharge_plus_functional"]
    )
    mse_motifs = int(
        motifs.set_index("method").loc["joint_mse_step1", "discharge_plus_functional"]
    )
    per_split_continuity = split_motifs.loc[
        split_motifs["method"] == "joint_continuity_step1", "discharge_plus_functional"
    ]
    gate_decision = {
        "gate_1_integrity": integrity["passed"],
        "gate_2_downstream_value": bool(
            continuity_functional.nrmse_median < condition_only.nrmse_median
            and continuity_functional.nrmse_median < support_stats.nrmse_median
            and paired.loc[("baseline", "condition_only"), "wins"] >= 9
            and paired.loc[("baseline", "condition_support_stats"), "wins"] >= 9
        ),
        "gate_3_motif_recurrence": bool(
            continuity_motifs >= 8
            and int((per_split_continuity >= 3).sum()) >= 2
        ),
        "gate_4_readability": bool(
            continuity_functional.complexity_median <= continuity_raw.complexity_median
        ),
        "gate_5_representation_diagnostic": bool(continuity_motifs > mse_motifs),
        "continuity_functional_motif_count": continuity_motifs,
        "mse_functional_motif_count": mse_motifs,
    }
    gate_decision["all_frozen_gates_pass"] = all(
        gate_decision[f"gate_{index}_{name}"]
        for index, name in (
            (1, "integrity"),
            (2, "downstream_value"),
            (3, "motif_recurrence"),
            (4, "readability"),
            (5, "representation_diagnostic"),
        )
    )
    (root / "gate_decision.json").write_text(json.dumps(gate_decision, indent=2))

    method_rows = []
    for row in summary.itertuples(index=False):
        method_rows.append(
            [
                row.method,
                row.regime,
                _format_number(row.nrmse_median),
                f"[{_format_number(row.nrmse_q1)}, {_format_number(row.nrmse_q3)}]",
                _format_number(row.nrmse_max),
                f"{int(row.nrmse_gt_10)}/{row.n}",
                _format_number(row.complexity_median),
                _format_number(row.validation_z_median),
            ]
        )
    paired_rows = [
        [
            f"{row.candidate_method}/{row.candidate_regime}",
            f"{row.anchor_method}/{row.anchor_regime}",
            f"{row.wins}/{row.n}",
            _format_number(row.median_nrmse_delta),
            f"{100 * row.median_relative_delta:.1f}%",
        ]
        for row in comparisons.itertuples(index=False)
    ]
    split_rows = [
        [
            row.method,
            row.dataset.rsplit("inner", 1)[-1],
            f"{int(row.discharge_plus_functional)}/{row.n}",
            f"{int(row.functional_slope_modulation)}/{row.n}",
        ]
        for row in split_motifs.itertuples(index=False)
    ]
    catastrophic = diagnostics.loc[diagnostics["catastrophic_nrmse_gt_10"]].sort_values(
        "reference_nrmse_structure_validation", ascending=False
    )
    catastrophic_rows = [
        [
            row.dataset.rsplit("inner", 1)[-1],
            row.seed,
            row.method,
            row.regime,
            _format_number(row.reference_nrmse_structure_validation),
            _format_number(row.max_abs_validation_z),
            "yes" if row.uses_exp else "no",
            "yes" if row.uses_division else "no",
        ]
        for row in catastrophic.itertuples(index=False)
    ]

    report = [
        "# NASA reviewer-clean Stage C：冻结后分析",
        "",
        "**日期：** 2026-08-25  ",
        "**判定：** 运行和信息边界完整；5 个冻结 gate 中通过 3 个（完整性、宽 motif 复现、representation diagnostic），未通过下游预测与可读性 gate，因此不能直接进入 Stage D。",
        "",
        "## 1. 这 90 个实验回答什么",
        "",
        "每个公式只在 8 块 meta-fit 电池的后 70% 周期上拟合，然后在未参与公式拟合的 5 块 structure-validation 电池后 70% 周期上评分。验证电池的 q 和 support statistics 只使用最早 30% target；query target 扰动不改变任何公式输入。比较对象是 physical conditions only、conditions + support summaries、conditions + raw q、conditions + decoder-functional q。",
        "",
        "## 2. 完整性结论",
        "",
        f"90/90 cells 成功，90/90 预测文件、Pareto fronts 和输入缩放记录存在；所有保存指标和预测有限；每个 cell 都是严格 8/5 实体隔离；30 个 q source 的 query-target leakage probe 最大差为 {integrity['max_query_target_feature_difference']:.1f}。完整性 gate：**{'PASS' if integrity['passed'] else 'FAIL'}**。",
        "",
        "## 3. 主要数值结果",
        "",
        *_markdown_table(
            ["方法", "接口", "validation NRMSE 中位数", "IQR", "最大值", "NRMSE>10", "复杂度中位数", "validation max-abs z 中位数"],
            method_rows,
        ),
        "",
        "`NRMSE>10` 和 `>100` 不是冻结 gate，只是透明呈现外推爆炸的描述阈值。均值被极端有限值支配，因此主表使用中位数、IQR、最大值和尾部计数。",
        "",
        "### 配对比较",
        "",
        *_markdown_table(
            ["候选", "锚点", "胜场", "NRMSE 中位差", "配对相对中位差"],
            paired_rows,
        ),
        "",
        "continuity functional-q 的 validation NRMSE 中位数为 1.7552，condition-only 为 0.9365，support-statistics 为 1.0332；它对两者都只有 4/15 胜场。冻结 downstream-value gate 明确 **FAIL**。但 functionalization 将 continuity raw-q 的中位 NRMSE 从 3.0812 降到 1.7552，并取得 10/15 配对胜场，说明先把自由 q 坐标映射为 decoder 功能坐标是正确方向，只是当前两个坐标和外推接口还不够。",
        "",
        "## 4. 公式里重复出现了什么",
        "",
        *_markdown_table(
            ["q 训练目标", "inner split", "cycle + functional", "functional 调制 cycle slope"],
            split_rows,
        ),
        "",
        f"宽 motif（公式同时使用 `discharge_index` 与至少一个功能坐标）在 continuity 中出现 {continuity_motifs}/15，在 MSE 中出现 {mse_motifs}/15；continuity 三个 split 分别为 "
        + "/5、".join(str(int(value)) for value in per_split_continuity)
        + "/5。因此冻结 motif gate **PASS**，且 continuity 仅以 12 对 11 略强于 MSE。更具体的“功能坐标调制退化斜率”只在 continuity 7/15、MSE 6/15 出现，而且跨 split 不均匀；它只能作为 Stage D 候选线索，不能称为已确认结构。",
        "",
        "continuity functional-q 的复杂度中位数为 13，raw-q 为 11，故 readability gate **FAIL**。这也说明功能坐标虽然更可比较，却没有在当前无约束算子库中自动产生更简单公式。",
        "",
        "## 5. 为什么会发生极端误差",
        "",
        *_markdown_table(
            ["split", "seed", "方法", "接口", "NRMSE", "max-abs z", "exp", "除法"],
            catastrophic_rows,
        ),
        "",
        "物理 condition 在 validation 中最多约 2.24 个训练标准差，而 raw q 的 group-level validation 最大 |z| 中位数为 22.19（continuity）和 12.38（MSE），最大达到 35.06。PySR 随后把这些域外 q 放进 `exp`、嵌套 `exp` 或接近零的分母，产生了最大 6.36e44 的有限预测误差。functional q 缩小了 shift（最大 |z| 中位数 7.37 和 4.42），但没有消除；3 个 functional cells 仍超过 NRMSE 10。",
        "",
        "这不是数据泄漏或运行失败。最直接的协议诊断是：meta-fit q 来自完整训练曲线的联合 auto-decoder 优化，而 validation q 来自前缀 support 的逆向校准；两者的信息量和优化路径不同。符号回归把二者当作同一坐标分布使用，所以测试的同时包含了“q 是否有信息”和“训练 q 与校准 q 是否坐标兼容”两个问题。当前结果首先否定的是这个未约束接口，而不是潜变量是否有用。",
        "",
        "## 6. 冻结 gate 总表",
        "",
        *_markdown_table(
            ["Gate", "结果", "证据"],
            [
                ["1 完整性", "PASS" if gate_decision["gate_1_integrity"] else "FAIL", "90/90；finite；8/5；leakage=0"],
                ["2 下游价值", "PASS" if gate_decision["gate_2_downstream_value"] else "FAIL", "中位数未优于两基线；均仅 4/15 胜"],
                ["3 motif 复现", "PASS" if gate_decision["gate_3_motif_recurrence"] else "FAIL", f"continuity {continuity_motifs}/15；每 split 至少 3/5"],
                ["4 可读性", "PASS" if gate_decision["gate_4_readability"] else "FAIL", "functional complexity 13 > raw 11"],
                ["5 表示诊断", "PASS" if gate_decision["gate_5_representation_diagnostic"] else "FAIL", f"continuity motif {continuity_motifs} > MSE {mse_motifs}"],
            ],
        ),
        "",
        "## 7. 下一步边界",
        "",
        "按冻结计划，不能在这 90 个结果上更换阈值、词表或算子后宣称同一 gate 通过，也不能直接把某条漂亮公式塞进 decoder。Stage D 前应先冻结一个独立的接口修复实验：让 meta-fit 实体也通过同样的 prefix-support q calibration 得到公式输入；同时对 functional coordinates 使用训练包络投影或有界系数结构，禁止 `exp(q)`、嵌套 `exp` 和无保护 q 分母。然后在新的开发划分上检验 broad motif 是否仍复现、误差是否优于 condition-only。只有通过后，才把候选结构收缩为“功能坐标控制初始容量/退化斜率 + 小残差”。",
        "",
        "全部原始失败公式和极端值保留在 `cell_diagnostics.csv`；本报告不删除、不 winsorize、不重跑任何 cell。",
    ]
    (root / "STAGE_C_ANALYSIS.md").write_text("\n".join(report) + "\n")
    print(json.dumps(gate_decision, indent=2))


if __name__ == "__main__":
    main()
