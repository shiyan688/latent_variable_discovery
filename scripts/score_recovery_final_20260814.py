"""Final recovery accounting: no sensitivity gate, denominator fixed at 46.

An admission gate based on q-sensitivity was investigated and rejected. Using the
minimum per-latent sensitivity wrongly excludes expr037 (substitution R^2 = 0.9999)
because its saturation constant is unidentifiable while its amplitude factor is
perfectly recoverable. Using the maximum cannot separate expr032 (max 0.00286,
recovered at R^2 0.9954) from expr052 (max 0.00202, diverged at R^2 -1.6e5) --- they
differ by only 1.4x. No threshold on this statistic can be defended, because
median|dy/dq| * sd(q) / sd(y) is a crude proxy for support-set identifiability: it
flattens effects concentrated in narrow regions and ignores interactions between q
components.

So the denominator stays 46 and nothing is filtered. Two sub-categories are broken
out, and both rest on objective facts rather than tuned thresholds:

  optimization diverged   the substitution fit itself failed (non-finite or
                          astronomically negative R^2). This is a numerical
                          statement about the fit, not a claim about recovery.
  weak control margin     learned - shuffled < 0.05, i.e. the fitted f can absorb
                          the q slot with a constant, so the cell cannot
                          discriminate a real latent from a permuted one.

Sensitivity survives only as an appendix covariate, reported per latent alongside
its empirical relationship to recovery, never as a filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = PROJECT_ROOT / "runs" / "recovery_failure_taxonomy_20260813" / "failure_taxonomy.csv"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "recovery_final_20260814"

RECOVERY_R2 = 0.9
CONTROL_MARGIN = 0.05
# A fit this far below zero is a diverged optimization, not a weak recovery: the
# worst genuine failure sits near -3.6 while diverged cells reach -1e5 and beyond.
DIVERGENCE_FLOOR = -100.0

def classify(row: pd.Series, args) -> str:
    learned = row["r2_learned"]
    shuffled = row["r2_shuffled"]
    if not np.isfinite(learned) or learned < args.divergence_floor:
        return "optimization_diverged"
    if learned >= args.recovery_r2:
        if np.isfinite(shuffled) and (learned - shuffled) < args.control_margin:
            return "weak_control_margin"
        return "recovered"
    return "not_recovered"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-r2", type=float, default=RECOVERY_R2)
    parser.add_argument("--control-margin", type=float, default=CONTROL_MARGIN)
    parser.add_argument("--divergence-floor", type=float, default=DIVERGENCE_FLOOR)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TAXONOMY)
    table["outcome_final"] = table.apply(lambda r: classify(r, args), axis=1)
    table["margin"] = table["r2_learned"] - table["r2_shuffled"]

    total = len(table)
    counts = table["outcome_final"].value_counts()
    recovered = int(counts.get("recovered", 0))

    print("=== headline: R_latent, denominator unfiltered ===")
    print(f"recovered {recovered}/{total} = {100 * recovered / total:.1f}%")
    for outcome in ("recovered", "not_recovered", "weak_control_margin", "optimization_diverged"):
        n = int(counts.get(outcome, 0))
        ids = sorted(int(v) for v in table.loc[table["outcome_final"] == outcome, "expression_id"])
        print(f"  {outcome:22s} {n:2d}  {ids}")

    print("\n=== structural category (holds under any denominator) ===")
    for category, group in table.groupby("category"):
        n = int((group["outcome_final"] == "recovered").sum())
        print(
            f"  {category:20s} {n:2d}/{len(group):2d}  "
            f"median CCA={group['cca_mean'].median():.3f}"
        )

    print("\n=== method-attributable failures (high sensitivity, still not recovered) ===")
    candidates = table[
        (table["outcome_final"] == "not_recovered") & (table["min_q_sensitivity"] >= 0.1)
    ].sort_values("min_q_sensitivity", ascending=False)
    for _, row in candidates.iterrows():
        note = " (margin NEGATIVE: worse than permuted q)" if row["margin"] < 0 else ""
        print(
            f"  expr{int(row['expression_id']):03d} [{row['category']:19s}] "
            f"R2={row['r2_learned']:7.4f} margin={row['margin']:7.4f} "
            f"cca={row['cca_mean']:6.3f} sens={row['min_q_sensitivity']:.3g}{note}"
        )

    print("\n=== appendix covariate: sensitivity vs outcome (not a filter) ===")
    for outcome, group in table.groupby("outcome_final"):
        print(
            f"  {outcome:22s} n={len(group):2d} "
            f"median min-sens={group['min_q_sensitivity'].median():.3g} "
            f"median max-sens={group['max_q_sensitivity'].median():.3g}"
        )

    # Rank-correlation between sensitivity and recovery, reported instead of used to gate.
    finite = table[np.isfinite(table["r2_learned"]) & (table["r2_learned"] > args.divergence_floor)]
    if len(finite) > 3:
        from scipy.stats import spearmanr

        rho, pvalue = spearmanr(finite["min_q_sensitivity"], finite["r2_learned"])
        print(
            f"  Spearman(min q-sensitivity, substitution R2) = {rho:.3f} "
            f"(p={pvalue:.3g}, n={len(finite)}, diverged cells excluded)"
        )

    table.to_csv(OUTPUT_ROOT / "recovery_final.csv", index=False)
    summary = {
        "thresholds": {
            "recovery_heldout_r2": args.recovery_r2,
            "control_margin": args.control_margin,
            "divergence_floor": args.divergence_floor,
        },
        "denominator": total,
        "no_sensitivity_gate": True,
        "counts": {k: int(v) for k, v in counts.items()},
        "recovered_rate": recovered / total,
        "by_category": {
            category: {
                "recovered": int((group["outcome_final"] == "recovered").sum()),
                "total": int(len(group)),
            }
            for category, group in table.groupby("category")
        },
        "outcome_by_expression": {
            int(row["expression_id"]): row["outcome_final"] for _, row in table.iterrows()
        },
    }
    (OUTPUT_ROOT / "recovery_final_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT_ROOT / 'recovery_final.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
