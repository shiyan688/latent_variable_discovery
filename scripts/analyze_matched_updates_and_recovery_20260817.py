#!/usr/bin/env python3
"""Freeze paired 300/1000-epoch and symbolic-recovery summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "matched_update_analysis_20260817"
RUN_300 = PROJECT_ROOT / "runs" / "full46_synthetic_core_exploratory_20260811" / "all_runs.csv"
RUN_1000 = (
    PROJECT_ROOT
    / "runs"
    / "full46_synthetic_neural_1000ep_matchedupdates_20260811"
    / "all_runs.csv"
)
Q_ABLATION = PROJECT_ROOT / "runs" / "q_recovery_ablation_20260812" / "all_runs.csv"
SYMBOLIC = PROJECT_ROOT / "runs" / "symbolic_all46_20260814" / "controls_results.csv"
RECOVERY = PROJECT_ROOT / "runs" / "recovery_final_20260814" / "recovery_final.csv"
METRIC = "reference_nrmse"
KEYS = ["expression_id", "seed"]

COMPARISONS = (
    ("1000 no-q vs 300 no-q", "1000", "no_q_mlp", "300", "no_q_mlp"),
    ("1000 oracle-q vs 300 oracle-q", "1000", "oracle_q_mlp", "300", "oracle_q_mlp"),
    ("1000 joint-MSE step1 vs 300 joint-MSE step2", "1000", "joint_mse_step1", "300", "joint_mse"),
    (
        "1000 joint-continuity step1 vs 300 joint-continuity step2",
        "1000",
        "joint_continuity_step1",
        "300",
        "joint_continuity",
    ),
    ("1000 joint-MSE step1 vs support-kNN", "1000", "joint_mse_step1", "300", "support_knn"),
    (
        "1000 joint-continuity step1 vs support-kNN",
        "1000",
        "joint_continuity_step1",
        "300",
        "support_knn",
    ),
    ("1000 no-q vs support-kNN", "1000", "no_q_mlp", "300", "support_knn"),
    ("1000 joint-MSE step1 vs Random Forest", "1000", "joint_mse_step1", "300", "random_forest"),
    (
        "1000 joint-continuity step1 vs Random Forest",
        "1000",
        "joint_continuity_step1",
        "300",
        "random_forest",
    ),
)


def _bootstrap_median_ci(values: np.ndarray, *, seed: int = 20260817) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _method_summary(frame: pd.DataFrame, protocol: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, group in frame.groupby("method", sort=True):
        values = group[METRIC].to_numpy(float)
        rows.append(
            {
                "protocol": protocol,
                "method": method,
                "runs": len(group),
                "median_reference_nrmse": float(np.median(values)),
                "mean_reference_nrmse": float(np.mean(values)),
                "p90_reference_nrmse": float(np.quantile(values, 0.90)),
                "max_reference_nrmse": float(np.max(values)),
                "median_wall_time_seconds": float(np.median(group["wall_time_seconds"])),
            }
        )
    return rows


def _paired(
    frames: dict[str, pd.DataFrame],
    *,
    comparison: str,
    candidate_protocol: str,
    candidate_method: str,
    anchor_protocol: str,
    anchor_method: str,
) -> dict[str, object]:
    candidate = frames[candidate_protocol]
    anchor = frames[anchor_protocol]
    left = candidate[candidate.method == candidate_method][KEYS + [METRIC]].rename(
        columns={METRIC: "candidate"}
    )
    right = anchor[anchor.method == anchor_method][KEYS + [METRIC]].rename(
        columns={METRIC: "anchor"}
    )
    paired = left.merge(right, on=KEYS, validate="one_to_one")
    if len(paired) != 138:
        raise ValueError(f"{comparison}: expected 138 paired cells, found {len(paired)}")
    expression = paired.groupby("expression_id", sort=True)[["candidate", "anchor"]].median()
    expression_delta = (expression.candidate - expression.anchor).to_numpy(float)
    ci_low, ci_high = _bootstrap_median_ci(expression_delta)
    row_delta = paired.candidate.to_numpy(float) - paired.anchor.to_numpy(float)
    return {
        "comparison": comparison,
        "candidate_protocol": candidate_protocol,
        "candidate_method": candidate_method,
        "anchor_protocol": anchor_protocol,
        "anchor_method": anchor_method,
        "paired_rows": len(paired),
        "paired_expressions": len(expression),
        "candidate_median": float(np.median(paired.candidate)),
        "anchor_median": float(np.median(paired.anchor)),
        "row_win_rate": float(np.mean(paired.candidate < paired.anchor)),
        "expression_win_rate": float(np.mean(expression.candidate < expression.anchor)),
        "median_row_delta": float(np.median(row_delta)),
        "median_expression_delta": float(np.median(expression_delta)),
        "expression_delta_bootstrap_ci_low": ci_low,
        "expression_delta_bootstrap_ci_high": ci_high,
        "median_ratio": float(
            np.median(paired.candidate.to_numpy(float) / np.maximum(paired.anchor, 1e-12))
        ),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(formats.get(column, "{}").format(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    output = OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    run_300 = pd.read_csv(RUN_300)
    run_1000 = pd.read_csv(RUN_1000)
    frames = {"300": run_300, "1000": run_1000}
    if len(run_300) != 828 or len(run_1000) != 552:
        raise ValueError("The two source campaigns are not terminal at their frozen sizes.")
    for frame in frames.values():
        if not np.isfinite(frame[METRIC]).all():
            raise ValueError("Non-finite reference NRMSE in a source campaign.")

    methods = pd.DataFrame(
        [*_method_summary(run_300, "300ep_original"), *_method_summary(run_1000, "1000ep_matched")]
    )
    methods.to_csv(output / "method_summary.csv", index=False)
    paired = pd.DataFrame(
        [
            _paired(
                frames,
                comparison=name,
                candidate_protocol=candidate_protocol,
                candidate_method=candidate_method,
                anchor_protocol=anchor_protocol,
                anchor_method=anchor_method,
            )
            for name, candidate_protocol, candidate_method, anchor_protocol, anchor_method in COMPARISONS
        ]
    )
    paired.to_csv(output / "paired_effects.csv", index=False)

    expression_rows = []
    for protocol, frame in frames.items():
        grouped = frame.groupby(["expression_id", "method"], as_index=False)[METRIC].median()
        grouped.insert(0, "protocol", protocol)
        expression_rows.append(grouped)
    pd.concat(expression_rows, ignore_index=True).to_csv(
        output / "expression_method_medians.csv", index=False
    )

    q_ablation = pd.read_csv(Q_ABLATION)
    q_summary = (
        q_ablation.groupby("method", as_index=False)
        .agg(
            runs=(METRIC, "size"),
            median_reference_nrmse=(METRIC, "median"),
            mean_reference_nrmse=(METRIC, "mean"),
            p90_reference_nrmse=(METRIC, lambda values: values.quantile(0.90)),
            median_cca=("cca_mean", "median"),
            median_continuity_auc=("continuity_auc", "median"),
            median_wall_time_seconds=("wall_time_seconds", "median"),
        )
        .sort_values("median_reference_nrmse")
    )
    q_summary.to_csv(output / "q_recovery_ablation_summary.csv", index=False)

    symbolic = pd.read_csv(SYMBOLIC)
    if len(symbolic) != 138 or set(symbolic.status) != {"success"}:
        raise ValueError("The symbolic controls are not 138/138 successful.")
    symbolic_summary_rows = []
    for regime, group in symbolic.groupby("regime", sort=True):
        values = group.r2_heldout_labels.to_numpy(float)
        symbolic_summary_rows.append(
            {
                "regime": regime,
                "expressions": len(group),
                "median_heldout_r2": float(np.median(values)),
                "p25_heldout_r2": float(np.quantile(values, 0.25)),
                "p75_heldout_r2": float(np.quantile(values, 0.75)),
                "fraction_r2_ge_0p9": float(np.mean(values >= 0.9)),
                "fraction_r2_ge_0p5": float(np.mean(values >= 0.5)),
                "minimum_heldout_r2": float(np.min(values)),
            }
        )
    symbolic_summary = pd.DataFrame(symbolic_summary_rows)
    symbolic_summary.to_csv(output / "symbolic_control_summary.csv", index=False)

    recovery = pd.read_csv(RECOVERY)
    recovery_counts = (
        recovery.groupby(["category", "outcome_final"], as_index=False)
        .size()
        .rename(columns={"size": "expressions"})
    )
    recovery_counts.to_csv(output / "recovery_outcomes_by_category.csv", index=False)
    final_counts = recovery.outcome_final.value_counts().to_dict()
    summary = {
        "source_rows": {"300ep": len(run_300), "1000ep": len(run_1000)},
        "recovery_denominator": len(recovery),
        "recovery_counts": final_counts,
        "recovered_rate": float(final_counts.get("recovered", 0) / len(recovery)),
        "symbolic_successes": len(symbolic),
        "q_recovery_ablation_rows": len(q_ablation),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected_methods = methods[
        methods.method.isin(
            [
                "joint_mse_step1",
                "joint_continuity_step1",
                "no_q_mlp",
                "oracle_q_mlp",
                "support_knn",
                "random_forest",
            ]
        )
    ].copy()
    selected_pairs = paired[
        paired.comparison.isin(
            [
                "1000 no-q vs 300 no-q",
                "1000 joint-MSE step1 vs 300 joint-MSE step2",
                "1000 joint-continuity step1 vs 300 joint-continuity step2",
                "1000 joint-MSE step1 vs support-kNN",
                "1000 joint-continuity step1 vs support-kNN",
                "1000 no-q vs support-kNN",
            ]
        )
    ].copy()
    report = [
        "# 1000-epoch 公平性复跑与隐变量恢复结果",
        "",
        "## 一句话结论",
        "",
        "在 46 个合成表达式上，把所有神经方法加到 1000 epoch 并匹配每个 batch 的更新次数后，",
        "无隐变量 MLP 并没有明显改善；显式 q 的 joint-MSE 仍显著优于它，也在 78.3% 的表达式上优于固定的 support-kNN。",
        "因此合成任务上的优势不是由 300/1000 epoch 不公平造成的。真实数据与 PDEBench 是否同样成立，",
        "由 2026-08-17 启动的独立 230-cell 任务回答，不能用本报告的合成结果提前代替。",
        "",
        "## 全局结果（NRMSE 越低越好）",
        "",
        _markdown_table(
            selected_methods,
            [
                "protocol",
                "method",
                "runs",
                "median_reference_nrmse",
                "p90_reference_nrmse",
                "max_reference_nrmse",
                "median_wall_time_seconds",
            ],
            {
                "median_reference_nrmse": "{:.6g}",
                "p90_reference_nrmse": "{:.6g}",
                "max_reference_nrmse": "{:.6g}",
                "median_wall_time_seconds": "{:.2f}",
            },
        ),
        "",
        "300-epoch latent joint 方法每 batch 更新两次；1000-epoch `step1` 方法每 batch 更新一次。",
        "所以两组不仅 epoch 不同，总更新数也从约 4800 增至 8000；这里回答的是最终长训练公平复跑，",
        "不是纯粹只改变 epoch 的单因素消融。kNN 与 RF 没有 epoch，复用相同 split 的固定锚点。",
        "",
        "## 配对比较",
        "",
        _markdown_table(
            selected_pairs,
            [
                "comparison",
                "candidate_median",
                "anchor_median",
                "expression_win_rate",
                "median_expression_delta",
                "expression_delta_bootstrap_ci_low",
                "expression_delta_bootstrap_ci_high",
            ],
            {
                "candidate_median": "{:.6g}",
                "anchor_median": "{:.6g}",
                "expression_win_rate": "{:.1%}",
                "median_expression_delta": "{:.6g}",
                "expression_delta_bootstrap_ci_low": "{:.6g}",
                "expression_delta_bootstrap_ci_high": "{:.6g}",
            },
        ),
        "",
        "差值定义为候选方法减锚点，负数表示候选更好。置信区间以 46 个表达式为重采样单位，",
        "避免把同一表达式的三个种子误当成 138 个完全独立问题。",
        "",
        "## q 恢复与符号对照",
        "",
        f"- 最终固定分母为 46 个表达式：恢复 {final_counts.get('recovered', 0)} 个（{summary['recovered_rate']:.1%}），",
        f"未恢复 {final_counts.get('not_recovered', 0)} 个，优化发散 {final_counts.get('optimization_diverged', 0)} 个，",
        f"对照间隔不足 {final_counts.get('weak_control_margin', 0)} 个。",
        "- 138 个符号回归对照全部成功。使用 learned q 时 held-out R² 中位数为 "
        f"{float(symbolic[symbolic.regime == 'with_q'].r2_heldout_labels.median()):.4f}；"
        f"no-q 为 {float(symbolic[symbolic.regime == 'no_q'].r2_heldout_labels.median()):.4f}，"
        f"entity one-hot 为 {float(symbolic[symbolic.regime == 'onehot'].r2_heldout_labels.median()):.4f}。",
        "- 恢复消融没有产生可推广的新赢家：原始 `joint_continuity` 的预测 NRMSE 中位数最低；",
        "固定范数和 affine quotient 明显恶化预测，交替优化的 q 学习率变体也未稳定胜出。",
        "",
        "## 结论边界",
        "",
        "这些是完整合成表达式库上的探索性证据。它支持“学到的连续 q 可作为符号发现接口”，",
        "但 28/46 的严格恢复率也说明优化发散、分母结构和多 q 交互仍是实质失败模式。",
        "任何真实数据/PDEBench 结论必须等待正在运行的独立任务完整结束后再冻结。",
        "",
    ]
    (output / "MATCHED_UPDATE_AND_RECOVERY_RESULTS.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
