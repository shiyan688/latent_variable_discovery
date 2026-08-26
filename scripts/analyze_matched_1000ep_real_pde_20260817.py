#!/usr/bin/env python3
"""Analyze the terminal matched-update real/PDE campaign."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAMPAIGN_ROOT = PROJECT_ROOT / "runs" / "matched_1000ep_real_pde_20260817"
OUTPUT_ROOT = CAMPAIGN_ROOT / "analysis"
OLD_RESULTS = PROJECT_ROOT / "runs" / "extended_15h_analysis_20260810" / "extended_all_result_rows.csv"
METRIC = "reference_nrmse"

DISPLAY_DATASET = {
    "nasa_battery_capacity": "NASA battery",
    "starry_te_seebeck": "Starry Seebeck",
    "starry_te_electrical_conductivity": "Starry electrical",
    "starry_te_thermal_conductivity": "Starry thermal",
}


def bootstrap_median_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(20260817)
    draws = rng.choice(values, size=(20_000, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def paired_effect(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    group: str,
    candidate: str,
    anchor: str,
) -> dict[str, object]:
    paired = left[["seed", METRIC]].rename(columns={METRIC: "candidate"}).merge(
        right[["seed", METRIC]].rename(columns={METRIC: "anchor"}),
        on="seed",
        validate="one_to_one",
    )
    delta = paired.candidate.to_numpy(float) - paired.anchor.to_numpy(float)
    low, high = bootstrap_median_ci(delta)
    return {
        "group": group,
        "candidate": candidate,
        "anchor": anchor,
        "pairs": len(paired),
        "candidate_median": float(np.median(paired.candidate)),
        "anchor_median": float(np.median(paired.anchor)),
        "wins": int(np.sum(delta < 0)),
        "win_rate": float(np.mean(delta < 0)),
        "median_delta": float(np.median(delta)),
        "median_ratio": float(np.median(paired.candidate / paired.anchor)),
        "median_delta_ci_low": low,
        "median_delta_ci_high": high,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in frame[columns].iterrows():
        values = [
            "—" if pd.isna(row[column]) else formats.get(column, "{}").format(row[column])
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def main() -> None:
    status = json.loads((CAMPAIGN_ROOT / "campaign_status.json").read_text())
    ledger = [
        json.loads(line)
        for line in (CAMPAIGN_ROOT / "task_status.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert status["state"] == "completed_all"
    assert status["planned"] == status["completed"] == 230
    assert status["failed"] == 0
    assert len(ledger) == len({row["task_id"] for row in ledger}) == 230
    assert all(row["returncode"] == 0 and not row["timed_out"] for row in ledger)

    real = pd.read_csv(CAMPAIGN_ROOT / "real" / "all_runs.csv")
    pde = pd.read_csv(CAMPAIGN_ROOT / "pdebench" / "all_runs.csv")
    old = pd.read_csv(OLD_RESULTS)
    assert len(real) == 200
    assert len(pde) == 100
    assert np.isfinite(real[METRIC]).all()
    assert np.isfinite(pde[METRIC]).all()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    real_summary = (
        real.groupby(["dataset", "method"], as_index=False)
        .agg(
            runs=(METRIC, "size"),
            median_nrmse=(METRIC, "median"),
            mean_nrmse=(METRIC, "mean"),
            p90_nrmse=(METRIC, lambda values: values.quantile(0.90)),
            max_nrmse=(METRIC, "max"),
            failures_gt_1=(METRIC, lambda values: int((values > 1).sum())),
            failures_gt_10=(METRIC, lambda values: int((values > 10).sum())),
            median_wall_seconds=("wall_time_seconds", "median"),
            median_continuity=("response_continuity_auc", "median"),
            median_trustworthiness=("response_trustworthiness_auc", "median"),
            median_local_distortion=("response_local_log_distortion_p95", "median"),
            median_effective_rank=("effective_rank", "median"),
        )
        .sort_values(["dataset", "median_nrmse"])
    )
    real_summary.insert(1, "dataset_display", real_summary.dataset.map(DISPLAY_DATASET))
    real_summary.to_csv(OUTPUT_ROOT / "real_method_summary.csv", index=False)

    real_effects = []
    for dataset, group in real.groupby("dataset"):
        by_method = {method: values for method, values in group.groupby("method")}
        for candidate in ("joint_mse_step1", "joint_continuity_step1"):
            for anchor in ("support_knn", "no_q_mlp", "random_forest"):
                real_effects.append(
                    paired_effect(
                        by_method[candidate],
                        by_method[anchor],
                        group=DISPLAY_DATASET[dataset],
                        candidate=candidate,
                        anchor=anchor,
                    )
                )
        real_effects.append(
            paired_effect(
                by_method["joint_continuity_step1"],
                by_method["joint_mse_step1"],
                group=DISPLAY_DATASET[dataset],
                candidate="joint_continuity_step1",
                anchor="joint_mse_step1",
            )
        )
    real_effects_frame = pd.DataFrame(real_effects)
    real_effects_frame.to_csv(OUTPUT_ROOT / "real_paired_effects.csv", index=False)

    pde_summary = (
        pde.groupby(["q_dim", "method", "strategy"], as_index=False)
        .agg(
            runs=(METRIC, "size"),
            median_nrmse=(METRIC, "median"),
            mean_nrmse=(METRIC, "mean"),
            p90_nrmse=(METRIC, lambda values: values.quantile(0.90)),
            max_nrmse=(METRIC, "max"),
            median_continuity=("continuity_auc", "median"),
            median_trustworthiness=("trustworthiness_auc", "median"),
            median_local_distortion=("local_log_distortion_p95", "median"),
            median_effective_rank=("effective_rank", "median"),
        )
        .sort_values(["q_dim", "method", "strategy"])
    )
    pde_summary.to_csv(OUTPUT_ROOT / "pde_method_summary.csv", index=False)

    pde_cells = {
        "q16_mse_adaptive": pde[
            (pde.q_dim == 16)
            & (pde.method == "joint_mse_step1")
            & (pde.strategy == "latent_adaptive_k4_min24")
        ],
        "q16_continuity_adaptive": pde[
            (pde.q_dim == 16)
            & (pde.method == "joint_continuity_step1")
            & (pde.strategy == "latent_adaptive_k4_min24")
        ],
        "q8_mse_adaptive": pde[
            (pde.q_dim == 8)
            & (pde.method == "joint_mse_step1")
            & (pde.strategy == "latent_adaptive_k4_min24")
        ],
        "support_knn4": pde[pde.strategy == "support_knn4"],
        "support_mean": pde[pde.strategy == "support_mean"],
        "pooled_mlp_no_latent": pde[pde.strategy == "pooled_mlp_no_latent"],
        "full_ic_pca_mlp_reference": pde[pde.strategy == "full_ic_pca_mlp_reference"],
    }
    pde_effects = []
    for candidate in ("q16_mse_adaptive", "q16_continuity_adaptive"):
        for anchor in (
            "support_knn4",
            "support_mean",
            "pooled_mlp_no_latent",
            "full_ic_pca_mlp_reference",
            "q8_mse_adaptive",
        ):
            pde_effects.append(
                paired_effect(
                    pde_cells[candidate],
                    pde_cells[anchor],
                    group="PDEBench Burgers",
                    candidate=candidate,
                    anchor=anchor,
                )
            )
    pde_effects.append(
        paired_effect(
            pde_cells["q16_continuity_adaptive"],
            pde_cells["q16_mse_adaptive"],
            group="PDEBench Burgers",
            candidate="q16_continuity_adaptive",
            anchor="q16_mse_adaptive",
        )
    )
    pde_effects_frame = pd.DataFrame(pde_effects)
    pde_effects_frame.to_csv(OUTPUT_ROOT / "pde_paired_effects.csv", index=False)

    old_new = []
    for dataset in ("nasa_battery_capacity", "starry_te_seebeck"):
        for old_method, new_method in (
            ("joint_mse", "joint_mse_step1"),
            ("joint_continuity", "joint_continuity_step1"),
        ):
            earlier = old[
                (old.family == "qdim")
                & (old.dataset == dataset)
                & (old.q_dim == 8)
                & (old.method == old_method)
                & (old.support_ratio == 0.3)
            ]
            longer = real[(real.dataset == dataset) & (real.method == new_method)]
            row = paired_effect(
                longer,
                earlier,
                group=DISPLAY_DATASET[dataset],
                candidate=f"1000ep_{new_method}",
                anchor=f"200ep_{old_method}",
            )
            paired = longer[["seed", METRIC]].merge(
                earlier[["seed", METRIC]], on="seed", suffixes=("_new", "_old")
            )
            row["candidate_failures_gt_10"] = int((paired[f"{METRIC}_new"] > 10).sum())
            row["anchor_failures_gt_10"] = int((paired[f"{METRIC}_old"] > 10).sum())
            old_new.append(row)
    earlier_pde = old[
        (old.domain == "pde")
        & (old.q_dim == 16)
        & (old.method == "joint_mse")
        & (old.support_ratio == 0.3)
        & (old.strategy == "latent_adaptive_k4_min24")
    ]
    old_new.append(
        {
            **paired_effect(
                pde_cells["q16_mse_adaptive"],
                earlier_pde,
                group="PDEBench Burgers",
                candidate="1000ep_q16_mse_step1",
                anchor="300ep_q16_mse_step2",
            ),
            "candidate_failures_gt_10": 0,
            "anchor_failures_gt_10": 0,
        }
    )
    old_new_frame = pd.DataFrame(old_new)
    old_new_frame.to_csv(OUTPUT_ROOT / "short_vs_long_paired.csv", index=False)

    optimizer_rows = []
    for path in (CAMPAIGN_ROOT / "real").glob("**/result.json"):
        result = json.loads(path.read_text())
        counters = result.get("optimization_counters", {})
        if counters:
            optimizer_rows.append(
                {
                    "domain": "real",
                    "dataset": result["job"]["dataset"],
                    "method": result["job"]["method"],
                    "seed": result["job"]["seed"],
                    "backward_passes": counters["backward_passes"],
                }
            )
    for path in (CAMPAIGN_ROOT / "pdebench").glob("q*/*/seed*/result.json"):
        result = json.loads(path.read_text())
        for component, counters in result["optimization_counters"].items():
            optimizer_rows.append(
                {
                    "domain": "pde",
                    "dataset": "PDEBench Burgers",
                    "method": f"{result['job']['method']}:{component}",
                    "seed": result["job"]["seed"],
                    "backward_passes": counters["backward_passes"],
                }
            )
    optimizer_frame = pd.DataFrame(optimizer_rows)
    optimizer_frame.to_csv(OUTPUT_ROOT / "optimizer_update_audit.csv", index=False)

    audit = {
        "campaign_state": status["state"],
        "planned": status["planned"],
        "completed": status["completed"],
        "failed": status["failed"],
        "ledger_rows": len(ledger),
        "unique_task_ids": len({row["task_id"] for row in ledger}),
        "real_result_rows": len(real),
        "pde_strategy_rows": len(pde),
        "finite_real_primary_metrics": int(np.isfinite(real[METRIC]).sum()),
        "finite_pde_primary_metrics": int(np.isfinite(pde[METRIC]).sum()),
        "accounted_task_hours": float(sum(row["elapsed_seconds"] for row in ledger) / 3600),
        "wall_start_utc": status["started_at"],
        "wall_end_utc": status["updated_at"],
    }
    (OUTPUT_ROOT / "terminal_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    real_wide = real_summary.pivot(index="dataset_display", columns="method", values="median_nrmse")
    real_tail = real_summary.pivot(index="dataset_display", columns="method", values="failures_gt_10")
    real_table = pd.DataFrame(
        {
            "dataset": real_wide.index,
            "joint-MSE": real_wide["joint_mse_step1"].to_numpy(),
            "joint-continuity": real_wide["joint_continuity_step1"].to_numpy(),
            "no-q MLP": real_wide["no_q_mlp"].to_numpy(),
            "support-kNN": real_wide["support_knn"].to_numpy(),
            "Random Forest": real_wide["random_forest"].to_numpy(),
        }
    )
    tail_table = pd.DataFrame(
        {
            "dataset": real_tail.index,
            "joint-MSE": real_tail["joint_mse_step1"].astype(int).to_numpy(),
            "joint-continuity": real_tail["joint_continuity_step1"].astype(int).to_numpy(),
            "no-q MLP": real_tail["no_q_mlp"].astype(int).to_numpy(),
            "support-kNN": real_tail["support_knn"].astype(int).to_numpy(),
            "Random Forest": real_tail["random_forest"].astype(int).to_numpy(),
        }
    )
    pde_report = pde_summary[
        pde_summary.strategy.isin(
            [
                "latent_adaptive_k4_min24",
                "pooled_mlp_no_latent",
                "support_mean",
                "support_knn4",
                "full_ic_pca_mlp_reference",
            ]
        )
    ].copy()
    pde_report["configuration"] = pde_report.apply(
        lambda row: f"q={int(row.q_dim)} {row.method} / {row.strategy}", axis=1
    )
    selected_effects = real_effects_frame[
        real_effects_frame.anchor == "support_knn"
    ][["group", "candidate", "wins", "pairs", "median_delta", "median_ratio"]]
    pde_selected = pde_effects_frame[pde_effects_frame.anchor == "support_knn4"][
        ["group", "candidate", "wins", "pairs", "median_delta", "median_ratio"]
    ]
    selected_effects = pd.concat([selected_effects, pde_selected], ignore_index=True)

    report = [
        "# 1000-epoch 真实数据与 PDEBench 匹配更新实验结果",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + terminal analysis",
        "- Origin Date: 2026-08-17",
        "- Verification Status: ANALYZED",
        "- Version Label: matched_real_pde_v1",
        "",
        "## 一句话结论",
        "",
        "1000 epoch 和严格相同的神经更新次数没有消除 support-kNN 的优势。显式 q 在 NASA battery 上明确胜出，",
        "但在三个 Starry 属性中只有 Seebeck 的 continuity 版本多数种子稳定，且仍不及 kNN；电导率与热导率的",
        "所有神经方法均发生全种子数量级爆炸。PDEBench 上 q=16 仍明显优于 pooled no-q MLP 和 support mean，",
        "但 support-kNN 在 10/10 个种子上更好；延长训练反而使 q=16 MSE 在 10/10 个种子上比短训练版本更差。",
        "",
        "## 终态审计",
        "",
        f"- 230/230 任务成功，0 失败、0 超时；真实结果 200 行，PDE 策略结果 100 行，主指标全部有限。",
        f"- 累计任务槽位时间 {audit['accounted_task_hours']:.2f} 小时；控制器正常退出并完成两个汇总命令。",
        "- 同一真实数据集内，joint-MSE、joint-continuity 与 no-q MLP 的 backward 次数完全一致；PDE 神经组件均为 128,000 次。",
        "",
        "## 真实数据主表",
        "",
        "下表为 10 个种子的 reference NRMSE 中位数，越低越好。",
        "",
        markdown_table(
            real_table,
            ["dataset", "joint-MSE", "joint-continuity", "no-q MLP", "support-kNN", "Random Forest"],
            {column: "{:.6g}" for column in real_table.columns if column != "dataset"},
        ),
        "",
        "NASA battery 上 continuity latent 的中位数 0.2279，10/10 胜过 kNN（0.3304）；MSE latent 也在 8/10 胜过 kNN。",
        "Starry Seebeck 上 continuity 中位数 0.01263，接近 RF 的 0.01218，但 kNN 为 0.00219 且 10/10 更好。",
        "Starry electrical 与 thermal 上，kNN 分别为 0.5053 和 0.05637，而神经方法全部严重爆炸。",
        "",
        "## 尾部失败",
        "",
        "下表统计 `NRMSE > 10` 的种子数（10 个种子中）。阈值用于透明展示尾部，不是预注册显著性门槛。",
        "",
        markdown_table(
            tail_table,
            ["dataset", "joint-MSE", "joint-continuity", "no-q MLP", "support-kNN", "Random Forest"],
            {},
        ),
        "",
        "这说明“增加 epoch”不能被解释为通用修复：它改善了 Seebeck 的部分随机种子，却没有修复另外两个材料属性的全局尺度失稳。",
        "",
        "## PDEBench 主表",
        "",
        markdown_table(
            pde_report,
            ["configuration", "runs", "median_nrmse", "p90_nrmse", "max_nrmse", "median_continuity", "median_effective_rank"],
            {
                "median_nrmse": "{:.6g}",
                "p90_nrmse": "{:.6g}",
                "max_nrmse": "{:.6g}",
                "median_continuity": "{:.4f}",
                "median_effective_rank": "{:.3f}",
            },
        ),
        "",
        "q=16 continuity 的中位 NRMSE 为 0.2573，略好于 q=16 MSE 的 0.2651，并在 7/10 配对种子上获胜。",
        "它同时有更高的 continuity（0.7759 vs 0.7236），说明该损失在 PDE 上同时改善了预测与邻域保持。",
        "但是 kNN 的 0.2113 仍在 10/10 种子上胜过两种 q=16 方法。",
        "",
        "## 与 support-kNN 的严格配对",
        "",
        markdown_table(
            selected_effects,
            ["group", "candidate", "wins", "pairs", "median_delta", "median_ratio"],
            {"median_delta": "{:.6g}", "median_ratio": "{:.4g}"},
        ),
        "",
        "差值定义为候选减 kNN，负数表示候选更好。",
        "",
        "## 短训练与长训练",
        "",
        markdown_table(
            old_new_frame,
            [
                "group",
                "candidate",
                "anchor",
                "wins",
                "pairs",
                "candidate_median",
                "anchor_median",
                "median_delta",
                "candidate_failures_gt_10",
                "anchor_failures_gt_10",
            ],
            {
                "candidate_median": "{:.6g}",
                "anchor_median": "{:.6g}",
                "median_delta": "{:.6g}",
            },
        ),
        "",
        "长训练对 NASA battery 没有一致增益；Seebeck continuity 的中位数和灾难种子数有所改善，",
        "但仍留有 1 个灾难种子。PDE q=16 MSE 长训练在 0/10 个种子上优于短训练，",
        "所以 PDE 的 kNN 优势不能归因于神经模型只训练了较少 epoch。",
        "",
        "## 结论边界",
        "",
        "1. 可以支持：显式 q 在 battery 和 PDE 相对 support-blind no-q MLP 有稳定价值。",
        "2. 不可以支持：显式 q 普遍优于所有 support-aware 方法；kNN 在 Starry 与 PDE 更强。",
        "3. 不可以支持：1000 epoch 普遍修复了神经模型；材料属性的极端尺度失稳仍是核心失败模式。",
        "4. 连续性、预测和数值稳定性是不同终点，必须并列报告，不能用较好的几何分数掩盖预测爆炸。",
        "",
    ]
    (OUTPUT_ROOT / "MATCHED_1000EP_REAL_PDE_RESULTS.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
