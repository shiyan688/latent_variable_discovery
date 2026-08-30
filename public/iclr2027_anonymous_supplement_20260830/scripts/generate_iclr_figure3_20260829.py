#!/usr/bin/env python3
"""Generate the real-transfer/tails figure from sealed confirmation outputs."""

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "_runtime_cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "runs" / "_runtime_cache" / "xdg_cache"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper.figures.paper_plot_style import COLORS

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "svg.hashsalt": "latent_variable_search_figure3_20260829",
        "pdf.fonttype": 42,
    }
)


ZT_ROOT = ROOT / "runs" / "starry_zt_temporal_confirmation_20260829" / "evaluation"
VP_ROOT = ROOT / "runs" / "thermoml_single_use_confirmation_20260829" / "c947fbd6cc82bf8d880a1449f16f859ede8e05b58f6c3e11504cdf24d05c38c4"
ZT_STABILITY = ROOT / "runs" / "starry_zt_interpretable_req_20260829" / "q_support_offset_stability.csv"
VP_STABILITY = ROOT / "runs" / "thermoml_q_stability_development_20260829" / "offset_q_stability.csv"
STYLE = ROOT / "paper" / "figures" / "paper_plot_style.py"
OUT = ROOT / "paper" / "figures"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representative(metrics, family):
    rows = metrics[metrics.family == family].copy()
    middle = float(rows.reference_nrmse.median()) if "reference_nrmse" in rows else float(rows.physical_nrmse.median())
    column = "reference_nrmse" if "reference_nrmse" in rows else "physical_nrmse"
    rows["distance"] = (rows[column] - middle).abs()
    id_column = "label" if "label" in rows else "entity_id"
    return str(rows.sort_values(["distance", id_column]).iloc[0][id_column]), middle


def pairwise_spearman(frame, columns):
    values = {}
    for column in columns:
        correlations = []
        pivot = frame.pivot(index=["fold", "label"], columns="offset", values=column)
        for left in range(4):
            for right in range(left + 1, 4):
                left_rank = pivot[left].rank(method="average")
                right_rank = pivot[right].rank(method="average")
                correlations.append(float(left_rank.corr(right_rank)))
        values[column] = {"median": float(np.median(correlations)), "minimum": float(np.min(correlations))}
    return values


def main():
    zt_predictions_path = ZT_ROOT / "query_predictions.csv"
    zt_metrics_path = ZT_ROOT / "per_entity_metrics.csv"
    zt_data_path = ZT_ROOT / "confirmation_data_used.csv"
    vp_predictions_path = VP_ROOT / "aggregate_query_predictions.csv"
    vp_metrics_path = VP_ROOT / "analysis" / "per_entity_metrics.csv"
    vp_data_path = VP_ROOT / "confirmation_data_used.csv"

    zt_predictions = pd.read_csv(zt_predictions_path)
    zt_metrics = pd.read_csv(zt_metrics_path)
    zt_data = pd.read_csv(zt_data_path)
    vp_predictions = pd.read_csv(vp_predictions_path)
    vp_metrics = pd.read_csv(vp_metrics_path)
    vp_data = pd.read_csv(vp_data_path)

    zt_entity, zt_median = representative(zt_metrics, "quadratic_req")
    vp_entity, vp_median = representative(vp_metrics, "structure_v_log")

    zt_expression = zt_predictions[(zt_predictions.family == "quadratic_req") & (zt_predictions.label.astype(str) == zt_entity)].sort_values("temperature")
    zt_baseline = zt_predictions[(zt_predictions.family == "support_knn") & (zt_predictions.label.astype(str) == zt_entity)].sort_values("temperature")
    zt_all = zt_data[zt_data.label.astype(str) == zt_entity].sort_values("temperature", kind="stable").reset_index(drop=True)
    support_mask = np.arange(len(zt_all)) % 4 == 0
    zt_support = zt_all.loc[support_mask]
    zt_query = zt_all.loc[~support_mask]
    if not np.array_equal(zt_query.temperature.to_numpy(), zt_expression.temperature.to_numpy()) or not np.array_equal(zt_query.target.to_numpy(), zt_expression.target.to_numpy()):
        raise ValueError("Frozen Starry every-fourth-row support split does not reproduce the sealed query table")

    vp_expression = vp_predictions[(vp_predictions.family == "structure_v_log") & (vp_predictions.entity_id == vp_entity)].sort_values("temperature_k")
    vp_baseline = vp_predictions[(vp_predictions.family == "support_pchip_log") & (vp_predictions.entity_id == vp_entity)].sort_values("temperature_k")
    vp_all = vp_data[vp_data.entity_id == vp_entity].sort_values("temperature_k")
    vp_support = vp_all[vp_all.role == "support"]
    vp_name = str(vp_all.common_name.iloc[0])

    if len(zt_metrics[zt_metrics.family == "quadratic_req"]) != 30 or len(vp_metrics[vp_metrics.family == "structure_v_log"]) != 84:
        raise ValueError("Unexpected confirmation entity count")

    zt_pair = zt_metrics[zt_metrics.family.isin(["quadratic_req", "support_knn"])].pivot(index="label", columns="family", values="reference_nrmse")
    vp_pair = vp_metrics[vp_metrics.family.isin(["structure_v_log", "support_pchip_log"])].pivot(index="entity_id", columns="family", values="physical_nrmse")
    if not vp_data["pressure_kpa"].gt(0.0).all() or not (zt_pair.gt(0.0).all().all() and vp_pair.gt(0.0).all().all()):
        raise ValueError("log-scale Figure 3 inputs must be strictly positive")

    zt_stability = pairwise_spearman(
        pd.read_csv(ZT_STABILITY),
        ["physical_intercept", "physical_linear_temperature", "physical_quadratic_temperature"],
    )
    vp_stability_frame = pd.read_csv(VP_STABILITY)
    vp_columns = [
        "q0_log_pressure_at_reference_spearman",
        "effective_delta_hvap_kj_mol_spearman",
        "effective_delta_cp_j_mol_k_spearman",
    ]
    vp_stability = {
        column: {
            "median": float(vp_stability_frame[column].median()),
            "minimum": float(vp_stability_frame[column].min()),
        }
        for column in vp_columns
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15), gridspec_kw={"hspace": 0.72, "wspace": 0.34})

    ax = axes[0, 0]
    ax.scatter(zt_expression.temperature, zt_expression.target, s=11, color="#999999", label="Query", zorder=2)
    ax.scatter(zt_support.temperature, zt_support.target, s=24, facecolor="white", edgecolor="black", linewidth=0.8, label="Support", zorder=4)
    ax.plot(zt_expression.temperature, zt_expression.prediction, color=COLORS["blue"], label="Quadratic", zorder=3)
    ax.plot(zt_baseline.temperature, zt_baseline.prediction, color=COLORS["orange"], linestyle="--", label="kNN", zorder=3)
    composition = str(zt_all.composition.iloc[0])
    ax.set_title(f"Starry ZT: {composition}", loc="left", fontsize=8, pad=28)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("ZT")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=6.1, columnspacing=0.55, handletextpad=0.3)
    ax.text(0.01, 0.98, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    ax = axes[0, 1]
    ax.scatter(vp_expression.temperature_k, vp_expression.pressure_kpa, s=11, color="#999999", label="Query", zorder=2)
    ax.scatter(vp_support.temperature_k, vp_support.pressure_kpa, s=24, facecolor="white", edgecolor="black", linewidth=0.8, label="Support", zorder=4)
    ax.plot(vp_expression.temperature_k, vp_expression.prediction_kpa, color=COLORS["blue"], label="Expression", zorder=3)
    ax.plot(vp_baseline.temperature_k, vp_baseline.prediction_kpa, color=COLORS["orange"], linestyle="--", label="Log-P PCHIP", zorder=3)
    ax.set_yscale("log")
    ax.set_title(f"Vapor pressure: {vp_name}", loc="left", fontsize=8, pad=28)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (kPa, log scale)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=6.1, columnspacing=0.55, handletextpad=0.3)
    ax.text(0.01, 0.98, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    ax = axes[1, 0]
    ax.scatter(zt_pair.support_knn, zt_pair.quadratic_req, s=18, alpha=0.75, color=COLORS["blue"], label="ZT: expression vs kNN")
    ax.scatter(vp_pair.support_pchip_log, vp_pair.structure_v_log, s=18, alpha=0.65, color=COLORS["green"], marker="^", label="Vapor: expression vs log-PCHIP")
    lower = min(zt_pair.min().min(), vp_pair.min().min()) * 0.75
    upper = max(zt_pair.max().max(), vp_pair.max().max()) * 1.35
    ax.plot([lower, upper], [lower, upper], color=COLORS["gray"], linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Strongest support baseline entity NRMSE")
    ax.set_ylabel("Expression entity NRMSE")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=6.5, handletextpad=0.35)
    ax.text(0.98, 0.04, "better expression\n16/30 ZT; 37/84 vapor", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
    ax.text(0.01, 0.98, "c", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    ax = axes[1, 1]
    stability_values = list(zt_stability.values()) + list(vp_stability.values())
    labels = ["ZT\nlevel", "ZT\nslope", "ZT\ncurve", "VP\nlevel", "VP\n$\\Delta H$", "VP\n$\\Delta C_p$"]
    x = np.arange(len(labels))
    medians = np.array([row["median"] for row in stability_values])
    minima = np.array([row["minimum"] for row in stability_values])
    colors = [COLORS["blue"]] * 3 + [COLORS["green"]] * 3
    ax.vlines(x, minima, medians, color=colors, linewidth=2.0, alpha=0.75)
    ax.scatter(x, medians, color=colors, s=28, label="Pairwise median", zorder=3)
    ax.scatter(x, minima, facecolor="white", edgecolor=colors, s=24, marker="v", linewidth=0.8, label="Pairwise minimum", zorder=3)
    ax.axhline(0.85, color=COLORS["gray"], linestyle="--", linewidth=0.8, label="0.85 reference")
    ax.axvline(2.5, color="#BBBBBB", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.65, 1.015)
    ax.set_ylabel("Spearman across support offsets")
    ax.legend(frameon=False, loc="lower left", ncol=1, handletextpad=0.35)
    ax.text(0.01, 0.98, "d", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "figure3_real_transfer.pdf"
    svg_path = OUT / "figure3_real_transfer.svg"
    png_path = OUT / "figure3_real_transfer.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(svg_path, metadata={"Date": None})
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    sources = [zt_predictions_path, zt_metrics_path, zt_data_path, vp_predictions_path, vp_metrics_path, vp_data_path, ZT_STABILITY, VP_STABILITY]
    values = {
        "scope": "Figure 3: sealed temporal transfer, paired tails, and development support-offset stability",
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "code": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
            str(STYLE.relative_to(ROOT)): sha256(STYLE),
        },
        "representative_selection": "entity nearest the expression-family median entity NRMSE; lexicographic ID tie-break",
        "representatives": {
            "starry_zt": {"entity_id": zt_entity, "composition": composition, "family_median_nrmse": zt_median, "support_rows": int(len(zt_support)), "query_rows": int(len(zt_expression))},
            "vapor_pressure": {"entity_id": vp_entity, "common_name": vp_name, "family_median_nrmse": vp_median, "support_rows": int(len(vp_support)), "query_rows": int(len(vp_expression))},
        },
        "paired_entity_wins": {"starry_zt_expression_vs_knn": int((zt_pair.quadratic_req < zt_pair.support_knn).sum()), "vapor_expression_vs_pchip": int((vp_pair.structure_v_log < vp_pair.support_pchip_log).sum())},
        "support_offset_stability": {"starry_zt": zt_stability, "vapor_pressure": vp_stability},
        "outputs": {
            "figure3_real_transfer.pdf": sha256(pdf_path),
            "figure3_real_transfer.svg": sha256(svg_path),
            "figure3_real_transfer.png": sha256(png_path),
        },
    }
    (OUT / "figure3_values.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
