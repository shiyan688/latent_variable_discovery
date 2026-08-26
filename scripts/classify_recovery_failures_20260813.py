"""Classify substitution-recovery outcomes by how q enters the true expression.

The aggregate recovery rate hides the scientifically interesting question: which
kinds of governing factors can a support-conditioned latent actually recover?
This script assigns each of the 46 expressions a structural category based on how
its latent variables appear symbolically, and computes an empirical q-sensitivity
so that genuine non-identifiability is separable from optimization failure.

Categories (a cell may match several; the reported label is the highest-priority
match, since the hardest mechanism dominates recoverability):
  multi_q_interaction  two or more latents multiplied/divided together
  exponent_power       a latent sits in an exponent
  nested_nonlinear     a latent sits inside exp/log/sqrt/trig
  denominator          a latent appears in a denominator
  multiplicative       latent enters only through +,-,* at the top level
q-sensitivity is the median over sampled rows of |dy/dq| * sd(q) / sd(y), i.e. the
fraction of target variation attributable to a one-standard-deviation change in
each true latent. Low sensitivity means the data barely constrain q, so a
recovery failure there is a property of the task, not of the method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.core.expression_library import (  # noqa: E402
    load_expression_library,
    sample_expression_dataset,
    select_expression_task,
)

FULL46_ROOT = PROJECT_ROOT / "runs" / "full46_synthetic_core_exploratory_20260811"
SCORES_PATH = PROJECT_ROOT / "runs" / "q_substitution_recovery_all46_20260812" / "substitution_scores.json"
OUTPUT_ROOT = PROJECT_ROOT / "runs" / "recovery_failure_taxonomy_20260813"
DATA_SEED = 20260811
RECOVERY_THRESHOLD = 0.9
MARGIN_THRESHOLD = 0.05
OVERFLOW_FLOOR = -10.0

# sqrt is Pow(x, 1/2) rather than a Function, so it is detected separately below.
NONLINEAR_FUNCS = (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.tanh, sp.Abs)


def _latent_symbols(expression: sp.Expr, latent_names: list[str]) -> set[sp.Symbol]:
    return {s for s in expression.free_symbols if s.name in latent_names}


def classify_structure(rhs: str, latent_names: list[str]) -> dict:
    """Describe how each latent enters the expression."""
    expression = sp.sympify(rhs)
    latents = set(latent_names)
    flags = {
        "multi_q_interaction": False,
        "exponent_power": False,
        "nested_nonlinear": False,
        "denominator": False,
    }

    # A latent in an exponent, including q1**q2 and x**q.
    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Pow):
            base, exponent = node.as_base_exp()
            if _latent_symbols(exponent, latent_names):
                flags["exponent_power"] = True
            if _latent_symbols(base, latent_names):
                # Negative powers act as denominators; fractional ones are roots.
                if exponent.is_number and exponent < 0:
                    flags["denominator"] = True
                if exponent.is_Rational and not exponent.is_Integer:
                    flags["nested_nonlinear"] = True
        if isinstance(node, sp.Function) and node.func in NONLINEAR_FUNCS:
            if any(_latent_symbols(arg, latent_names) for arg in node.args):
                flags["nested_nonlinear"] = True

    # Denominators via explicit division.
    numerator, denominator = sp.fraction(sp.together(expression))
    if _latent_symbols(denominator, latent_names):
        flags["denominator"] = True

    # Two or more latents inside a single multiplicative/division term.
    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Mul):
            present = {s.name for s in _latent_symbols(node, latent_names)}
            if len(present) >= 2:
                flags["multi_q_interaction"] = True

    for priority in ("multi_q_interaction", "exponent_power", "nested_nonlinear", "denominator"):
        if flags[priority]:
            label = priority
            break
    else:
        label = "multiplicative"

    return {"category": label, "structure_flags": flags, "latent_count": len(latents)}


def q_sensitivity(expression_id: int) -> dict:
    """Fraction of target sd explained by a one-sd change in each true latent.

    Computed on the same deterministic rows used everywhere else. Low values mark
    tasks where the data barely identify q, so a recovery failure is intrinsic.
    """
    task = select_expression_task(
        load_expression_library(PROJECT_ROOT / "data" / "latent_variable_expressions.csv"),
        expression_id=expression_id,
    )
    generated = sample_expression_dataset(
        task,
        label_count=32,
        validation_label_count=16,
        test_label_count=32,
        train_samples_per_label=60,
        validation_samples_per_label=60,
        test_samples_per_label=60,
        label_split_mode="disjoint",
        seed=DATA_SEED,
    )
    frame = generated.test_frame
    truth = generated.latent_truth_frame
    latent_names = list(task.latent_variables)
    feature_names = list(task.feature_columns)

    merged = frame.merge(truth.loc[:, ["label", *latent_names]], on="label", how="inner")
    features = merged.loc[:, feature_names].to_numpy(float)
    q_true = merged.loc[:, latent_names].to_numpy(float)
    y = merged["target"].to_numpy(float)
    y_sd = float(np.std(y)) or 1.0

    expression = sp.sympify(task.rhs_expression)
    feature_symbols = [sp.Symbol(n) for n in feature_names]
    latent_symbols = [sp.Symbol(n) for n in latent_names]

    sensitivities = {}
    for index, latent in enumerate(latent_names):
        derivative = sp.lambdify(
            (feature_symbols, latent_symbols), sp.diff(expression, latent_symbols[index]), modules=["numpy"]
        )
        with np.errstate(all="ignore"):
            raw = np.asarray(
                derivative(
                    [features[:, i] for i in range(features.shape[1])],
                    [q_true[:, j] for j in range(q_true.shape[1])],
                ),
                dtype=float,
            )
        raw = np.broadcast_to(raw, (len(y),))
        finite = np.isfinite(raw)
        q_sd = float(np.std(q_true[:, index])) or 1.0
        sensitivities[latent] = (
            float(np.median(np.abs(raw[finite])) * q_sd / y_sd) if finite.any() else float("nan")
        )

    values = [v for v in sensitivities.values() if np.isfinite(v)]
    return {
        "per_latent_sensitivity": sensitivities,
        "min_sensitivity": float(min(values)) if values else float("nan"),
        "max_sensitivity": float(max(values)) if values else float("nan"),
        "target_dynamic_range_log10": float(
            np.log10(max(np.abs(y).max(), 1e-300)) - np.log10(max(np.abs(y[y != 0]).min(), 1e-300))
        )
        if (y != 0).any()
        else float("nan"),
    }


def outcome_of(record: dict) -> str:
    learned = record.get("best_r2_heldout_learned")
    shuffled = record.get("best_r2_heldout_shuffled")
    if learned is None or not np.isfinite(learned) or learned < OVERFLOW_FLOOR:
        return "numerical_overflow"
    if learned >= RECOVERY_THRESHOLD:
        if shuffled is not None and np.isfinite(shuffled) and (learned - shuffled) < MARGIN_THRESHOLD:
            return "weak_q_slot"
        return "recovered"
    return "failed"


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    scores = json.loads(SCORES_PATH.read_text())

    cca = {}
    for path in FULL46_ROOT.glob("expr*/joint_continuity/seed13_*/result.json"):
        payload = json.loads(path.read_text())
        cca[payload["job"]["expression_id"]] = payload.get("spatial", {}).get("cca_mean")

    rows = []
    for record in scores:
        expression_id = record["expression_id"]
        rhs = record.get("true_rhs")
        latent_names = record.get("latent_variables") or []
        if not rhs:
            continue
        structure = classify_structure(rhs, latent_names)
        sensitivity = q_sensitivity(expression_id)
        learned = record.get("best_r2_heldout_learned")
        shuffled = record.get("best_r2_heldout_shuffled")
        linear = record.get("families", {}).get("linear", {}).get("r2_heldout_labels")
        rows.append(
            {
                "expression_id": expression_id,
                "q_dim": record["q_dim"],
                "true_rhs": rhs,
                "n_latents": len(latent_names),
                "category": structure["category"],
                "outcome": outcome_of(record),
                "r2_learned": learned,
                "r2_shuffled": shuffled,
                "r2_linear_f": linear,
                "margin": (learned - shuffled)
                if all(v is not None and np.isfinite(v) for v in (learned, shuffled))
                else None,
                "cca_mean": cca.get(expression_id),
                "min_q_sensitivity": sensitivity["min_sensitivity"],
                "max_q_sensitivity": sensitivity["max_sensitivity"],
                "target_dynamic_range_log10": sensitivity["target_dynamic_range_log10"],
                **{f"flag_{k}": v for k, v in structure["structure_flags"].items()},
            }
        )

    table = pd.DataFrame(rows).sort_values(["outcome", "category", "expression_id"])
    table.to_csv(OUTPUT_ROOT / "failure_taxonomy.csv", index=False)

    print("=== outcome x category ===")
    print(pd.crosstab(table["category"], table["outcome"]).to_string())

    print("\n=== recovery rate by category ===")
    for category, group in table.groupby("category"):
        scorable = group[group["outcome"].isin(["recovered", "failed", "weak_q_slot"])]
        recovered = (scorable["outcome"] == "recovered").sum()
        print(
            f"{category:20s} {recovered}/{len(scorable)}  "
            f"median CCA={scorable['cca_mean'].median():.3f}  "
            f"median min-q-sens={scorable['min_q_sensitivity'].median():.4g}"
        )

    print("\n=== the 12 genuine failures ===")
    failures = table[table["outcome"] == "failed"]
    for _, row in failures.iterrows():
        print(
            f"expr{row['expression_id']:03d} [{row['category']:20s}] "
            f"R2={row['r2_learned']:.4f} cca={row['cca_mean']:.3f} "
            f"min_qsens={row['min_q_sensitivity']:.4g}  {row['true_rhs'][:44]}"
        )

    print("\n=== q-sensitivity: recovered vs failed ===")
    for outcome in ("recovered", "failed", "weak_q_slot", "numerical_overflow"):
        group = table[table["outcome"] == outcome]
        if group.empty:
            continue
        print(
            f"{outcome:20s} n={len(group):2d}  "
            f"median min-q-sensitivity={group['min_q_sensitivity'].median():.4g}  "
            f"median CCA={group['cca_mean'].median():.3f}"
        )

    summary = {
        "counts_by_outcome": table["outcome"].value_counts().to_dict(),
        "counts_by_category": table["category"].value_counts().to_dict(),
        "recovery_by_category": {
            category: {
                "recovered": int((group["outcome"] == "recovered").sum()),
                "scorable": int(group["outcome"].isin(["recovered", "failed", "weak_q_slot"]).sum()),
            }
            for category, group in table.groupby("category")
        },
        "thresholds": {
            "recovery_r2": RECOVERY_THRESHOLD,
            "control_margin": MARGIN_THRESHOLD,
            "overflow_floor": OVERFLOW_FLOOR,
        },
    }
    (OUTPUT_ROOT / "taxonomy_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTPUT_ROOT / 'failure_taxonomy.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
