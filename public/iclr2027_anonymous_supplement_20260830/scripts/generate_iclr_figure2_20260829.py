#!/usr/bin/env python3
"""Generate Figure 2 directly from sealed gauge and controlled-GIRD CSVs."""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / "_runtime_cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "runs" / "_runtime_cache" / "xdg_cache"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from paper.figures.paper_plot_style import COLORS


GAUGE_CSV = ROOT / "runs" / "gauge_equivariant_calibration_stable_extension_20260829" / "analysis" / "gauge_invariance_summary.csv"
GIRD_CSV = ROOT / "runs" / "gird_controlled_discovery_20260829" / "analysis" / "regime_method_summary.csv"
ENTITY_CSV = ROOT / "runs" / "gird_controlled_discovery_20260829" / "analysis" / "regime_entity_metrics.csv"
STYLE_PY = ROOT / "paper" / "figures" / "paper_plot_style.py"
GENERATOR_PY = Path(__file__).resolve()
OUT = ROOT / "paper" / "figures"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gauge_rows = read_rows(GAUGE_CSV)
    gird_rows = read_rows(GIRD_CSV)
    entity_rows = read_rows(ENTITY_CSV)

    method_style = {
        "mapped_start_adam": ("Mapped-start Adam", COLORS["orange"], "o"),
        "response_metric_gauss_newton": ("Stable response-GN", COLORS["blue"], "^"),
    }
    gauge = {}
    for method in method_style:
        rows = [row for row in gauge_rows if row["method"] == method]
        gauge[method] = {
            "raw_change": np.array([float(row["raw_q_max_abs_change"]) for row in rows]),
            "response_difference": np.array([float(row["query_response_max_abs_difference"]) for row in rows]),
        }
        if len(rows) != 75:
            raise ValueError(f"Expected 75 gauge interventions for {method}, found {len(rows)}")

    families = ["polynomial", "relaxation", "thermodynamic_chart"]
    family_label = {"polynomial": "Polynomial", "relaxation": "Relaxation", "thermodynamic_chart": "Thermodynamic"}
    methods = ["gird_gn_lambda_0.0", "gird_gn_selected", "fpca", "direct_target_omp_lambda_0.0"]
    bootstrap_draws = 10_000
    rng = np.random.default_rng(20260829)
    summary = {}
    for family in families:
        selected = {
            row["method"]: float(row["median_entity_nrmse"])
            for row in gird_rows
            if row["family"] == family
            and row["support_regime"] == "four_support"
            and row["method"] in methods
        }
        if set(selected) != set(methods):
            raise ValueError(f"Missing four-support summary rows for {family}: {selected}")
        entity_count = len(
            {
                row["entity_id"]
                for row in entity_rows
                if row["family"] == family
                and row["support_regime"] == "four_support"
                and row["method"] == "gird_gn_selected"
            }
        )
        if entity_count != 48:
            raise ValueError(f"Expected 48 entities for {family}, found {entity_count}")
        base = selected["gird_gn_lambda_0.0"]
        entity_values = {}
        entity_ids = None
        for method in ("gird_gn_lambda_0.0", "gird_gn_selected", "fpca"):
            rows = sorted(
                (
                    row
                    for row in entity_rows
                    if row["family"] == family
                    and row["support_regime"] == "four_support"
                    and row["method"] == method
                ),
                key=lambda row: int(row["entity_id"]),
            )
            ids = [int(row["entity_id"]) for row in rows]
            if entity_ids is None:
                entity_ids = ids
            if ids != entity_ids:
                raise ValueError(f"Entity alignment mismatch for {family}/{method}")
            entity_values[method] = np.array([float(row["nrmse"]) for row in rows])
        bootstrap_indices = rng.integers(0, entity_count, size=(bootstrap_draws, entity_count))
        bootstrap_base = np.median(entity_values["gird_gn_lambda_0.0"][bootstrap_indices], axis=1)
        bootstrap_ci = {}
        for method in ("gird_gn_selected", "fpca"):
            ratios = np.median(entity_values[method][bootstrap_indices], axis=1) / bootstrap_base
            bootstrap_ci[method] = [float(value) for value in np.quantile(ratios, [0.025, 0.975])]
        summary[family] = {
            "entity_count": entity_count,
            "lambda0_median_entity_nrmse": base,
            "gird_selected_median_entity_nrmse": selected["gird_gn_selected"],
            "fpca_median_entity_nrmse": selected["fpca"],
            "direct_target_omp_median_entity_nrmse": selected["direct_target_omp_lambda_0.0"],
            "gird_ratio": selected["gird_gn_selected"] / base,
            "fpca_ratio": selected["fpca"] / base,
            "gird_ratio_paired_entity_bootstrap_95ci": bootstrap_ci["gird_gn_selected"],
            "fpca_ratio_paired_entity_bootstrap_95ci": bootstrap_ci["fpca"],
        }

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    for method, (label, color, marker) in method_style.items():
        values = gauge[method]
        ax.scatter(
            values["raw_change"],
            values["response_difference"],
            s=15,
            alpha=0.68,
            color=color,
            marker=marker,
            linewidths=0.25,
            edgecolors="white",
            label=f"{label} (n=75)",
        )
    ax.axhline(1e-6, color=COLORS["gray"], linestyle="--", linewidth=0.8, label=r"$10^{-6}$ gate")
    ax.set_yscale("log")
    ax.set_xlabel(r"Raw-coordinate change, $\max |\Delta q|$")
    ax.set_ylabel("Max query-response difference")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
    ax.text(-0.16, 1.04, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1]
    x = np.arange(len(families))
    width = 0.33
    gird_ratio = np.array([summary[family]["gird_ratio"] for family in families])
    fpca_ratio = np.array([summary[family]["fpca_ratio"] for family in families])
    gird_ci = np.array([summary[family]["gird_ratio_paired_entity_bootstrap_95ci"] for family in families])
    fpca_ci = np.array([summary[family]["fpca_ratio_paired_entity_bootstrap_95ci"] for family in families])
    bars_gird = ax.bar(
        x - width / 2,
        gird_ratio,
        width,
        color=COLORS["blue"],
        label="Selected GIRD-GN",
        yerr=np.vstack((gird_ratio - gird_ci[:, 0], gird_ci[:, 1] - gird_ratio)),
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax.bar(
        x + width / 2,
        fpca_ratio,
        width,
        color=COLORS["green"],
        label="FPCA",
        yerr=np.vstack((fpca_ratio - fpca_ci[:, 0], fpca_ci[:, 1] - fpca_ratio)),
        error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7},
    )
    ax.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=0.8, label=r"GIRD-GN, $\lambda=0$ (support-only)")
    for bar, ratio in zip(bars_gird, gird_ratio):
        improvement = 100.0 * (1.0 - ratio)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{improvement:.1f}%", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, [family_label[family] for family in families])
    ax.set_ylim(0.0, 1.20)
    ax.set_ylabel("Ratio of median entity NRMSE")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.58, 1.01),
        ncol=3,
        fontsize=6.1,
        handletextpad=0.35,
        columnspacing=0.55,
    )
    ax.text(0.01, 0.98, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT / "figure2_gauge_gird.pdf"
    png_path = OUT / "figure2_gauge_gird.png"
    fig.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    manifest = {
        "scope": "Figure 2: gauge intervention and controlled four-support GIRD",
        "sources": {
            str(GAUGE_CSV.relative_to(ROOT)): sha256(GAUGE_CSV),
            str(GIRD_CSV.relative_to(ROOT)): sha256(GIRD_CSV),
            str(ENTITY_CSV.relative_to(ROOT)): sha256(ENTITY_CSV),
        },
        "code": {
            str(GENERATOR_PY.relative_to(ROOT)): sha256(GENERATOR_PY),
            str(STYLE_PY.relative_to(ROOT)): sha256(STYLE_PY),
        },
        "gauge_interventions_per_method": 75,
        "gauge_case_design": {"families": 3, "seeds_per_family": 5, "gauges_per_family_seed": 5},
        "paired_entity_bootstrap": {"draws": bootstrap_draws, "seed": 20260829},
        "median_raw_coordinate_change": {
            method: median(values["raw_change"].tolist()) for method, values in gauge.items()
        },
        "maximum_response_difference": {
            method: float(values["response_difference"].max()) for method, values in gauge.items()
        },
        "four_support": summary,
        "outputs": {
            "figure2_gauge_gird.pdf": sha256(pdf_path),
            "figure2_gauge_gird.png": sha256(png_path),
        },
    }
    (OUT / "figure2_values.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
