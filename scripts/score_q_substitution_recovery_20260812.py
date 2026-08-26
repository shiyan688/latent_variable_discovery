"""Substitution-based recovery scoring for learned latent q.

Judging rule (user-defined): keep the TRUE expression structure fixed, replace
each true latent slot ``q_k`` by a fitted function ``f_k(learned_q)``, then fit
only the coefficients of ``f`` and report R^2. High R^2 means the learned q is a
sufficient reparameterization of the true latent factor under the true physics.

This deliberately separates two failure modes that a raw PySR run conflates:
  1. can the symbolic search find the right x-structure, and
  2. is learned_q an adequate stand-in for the true q.
This script measures only (2).

``f`` families, in increasing capacity:
  linear      f_k(qhat) = a0 + sum_j a_j qhat_j
  quadratic   linear plus squares and pairwise products
  Additional per-expression scaling is absorbed by the fit, so a learned q that
  differs from the true q by any function in the family counts as recovered.

Fitting is nonlinear least squares on the substituted structure (the substitution
enters inside exp/sqrt/denominators, so this is not a linear problem in general).
Coefficients are fit on training labels and R^2 is reported on held-out labels of
the same test split, so a high score cannot come from per-label memorization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvs.core.expression_library import (  # noqa: E402
    load_expression_library,
    sample_expression_dataset,
    select_expression_task,
)

FULL46_ROOT = PROJECT_ROOT / "runs" / "full46_synthetic_core_exploratory_20260811"
DATA_SEED = 20260811
TRAIN_LABELS = 32
VALIDATION_LABELS = 16
TEST_LABELS = 32
SAMPLES_PER_LABEL = 60

PROBE_CELLS = (
    {"expression_id": 3, "q_dim": 1},
    {"expression_id": 16, "q_dim": 1},
    {"expression_id": 22, "q_dim": 1},
    {"expression_id": 10, "q_dim": 2},
    {"expression_id": 54, "q_dim": 2},
    {"expression_id": 36, "q_dim": 2},
    {"expression_id": 51, "q_dim": 3},
    {"expression_id": 49, "q_dim": 3},
)


def discover_cells(seed: int) -> list[dict]:
    """Every expression with a joint_continuity q artifact for this seed."""
    cells = []
    for q_path in sorted(FULL46_ROOT.glob(f"expr*/joint_continuity/seed{seed}_*/test_label_q.csv")):
        result_path = q_path.parent / "result.json"
        if not result_path.exists():
            continue
        job = json.loads(result_path.read_text())["job"]
        cells.append({"expression_id": int(job["expression_id"]), "q_dim": int(job["q_dim"])})
    cells.sort(key=lambda c: c["expression_id"])
    return cells


def apply_q_mode(qhat: np.ndarray, labels: np.ndarray, q_mode: str, seed: int) -> np.ndarray:
    """Controls that keep the fitting problem identical but destroy real q content.

    ``shuffled`` is the sharper control: it permutes the per-label q vectors across
    labels, preserving the exact marginal distribution while breaking the
    entity-to-q correspondence. ``random`` replaces q with standard normals
    matched to the observed per-column mean/scale.
    """
    if q_mode == "learned":
        return qhat
    rng = np.random.default_rng(seed + 9161)
    unique = np.unique(labels)
    if q_mode == "shuffled":
        source = np.stack([qhat[labels == lab][0] for lab in unique])
        permuted = source[rng.permutation(len(unique))]
        mapping = {lab: permuted[i] for i, lab in enumerate(unique)}
    elif q_mode == "random":
        center = qhat.mean(axis=0)
        scale = qhat.std(axis=0) + 1e-12
        mapping = {lab: center + scale * rng.normal(size=qhat.shape[1]) for lab in unique}
    else:
        raise ValueError(f"Unknown q_mode: {q_mode}")
    return np.stack([mapping[lab] for lab in labels])


def basis_terms(q_symbols: list[sp.Symbol], family: str) -> list[sp.Expr]:
    """Symbolic basis for the substitution function f_k."""
    terms: list[sp.Expr] = [sp.Integer(1)]
    terms.extend(q_symbols)
    if family == "quadratic":
        for a, b in combinations_with_replacement(q_symbols, 2):
            terms.append(a * b)
    elif family != "linear":
        raise ValueError(f"Unknown f family: {family}")
    return terms


def build_substituted_callable(
    task,
    q_dim: int,
    family: str,
) -> tuple[callable, int, list[str]]:
    """Return (evaluator, n_coefficients, coefficient names).

    The evaluator takes (coefficient vector, feature matrix, learned-q matrix)
    and returns predictions under the true structure with every true latent slot
    replaced by f_k(learned_q).
    """
    qhat_symbols = [sp.Symbol(f"qhat{j}", real=True) for j in range(1, q_dim + 1)]
    basis = basis_terms(qhat_symbols, family)

    true_rhs = sp.sympify(task.rhs_expression)
    latent_names = list(task.latent_variables)
    feature_names = list(task.feature_columns)

    coeff_symbols: list[sp.Symbol] = []
    coeff_names: list[str] = []
    substitutions = {}
    for latent in latent_names:
        row: list[sp.Expr] = []
        for term_index, term in enumerate(basis):
            symbol = sp.Symbol(f"c_{latent}_{term_index}", real=True)
            coeff_symbols.append(symbol)
            coeff_names.append(f"{latent}:{sp.srepr(term) if term == 1 else str(term)}")
            row.append(symbol * term)
        substitutions[sp.Symbol(latent)] = sp.Add(*row)

    substituted = true_rhs.subs(substitutions, simultaneous=True)

    feature_symbols = [sp.Symbol(name) for name in feature_names]
    evaluator = sp.lambdify(
        (coeff_symbols, feature_symbols, qhat_symbols),
        substituted,
        modules=["numpy"],
    )

    def predict(coefficients: np.ndarray, features: np.ndarray, qhat: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            raw = evaluator(
                list(coefficients),
                [features[:, i] for i in range(features.shape[1])],
                [qhat[:, j] for j in range(qhat.shape[1])],
            )
        return np.broadcast_to(np.asarray(raw, dtype=float), (features.shape[0],)).copy()

    return predict, len(coeff_symbols), coeff_names


def load_cell_frame(expression_id: int, q_dim: int, seed: int):
    """Regenerate deterministic test rows and attach learned q by label."""
    task = select_expression_task(
        load_expression_library(PROJECT_ROOT / "data" / "latent_variable_expressions.csv"),
        expression_id=expression_id,
    )
    generated = sample_expression_dataset(
        task,
        label_count=TRAIN_LABELS,
        validation_label_count=VALIDATION_LABELS,
        test_label_count=TEST_LABELS,
        train_samples_per_label=SAMPLES_PER_LABEL,
        validation_samples_per_label=SAMPLES_PER_LABEL,
        test_samples_per_label=SAMPLES_PER_LABEL,
        label_split_mode="disjoint",
        seed=DATA_SEED,
    )
    q_paths = sorted(FULL46_ROOT.glob(f"expr{expression_id:03d}/joint_continuity/seed{seed}_*/test_label_q.csv"))
    if len(q_paths) != 1:
        raise RuntimeError(f"expr{expression_id:03d} seed{seed}: expected one q source, found {q_paths}")

    label_q = pd.read_csv(q_paths[0])
    learned_cols = [f"learned_q{j}" for j in range(1, q_dim + 1)]
    truth_cols = [name for name in task.latent_variables if name in label_q.columns]
    frame = generated.test_frame.merge(
        label_q.loc[:, ["label", *learned_cols, *truth_cols]], on="label", how="inner"
    )
    if frame.empty:
        raise RuntimeError(f"expr{expression_id:03d}: empty join")
    return task, frame, learned_cols, truth_cols, str(q_paths[0].relative_to(PROJECT_ROOT))


def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total <= 0:
        return float("nan")
    return 1.0 - residual / total


def fit_substitution(
    predict,
    n_coeff: int,
    features_fit: np.ndarray,
    qhat_fit: np.ndarray,
    y_fit: np.ndarray,
    n_restarts: int,
    seed: int,
    max_nfev: int,
):
    """Multi-start nonlinear least squares on the substitution coefficients."""
    rng = np.random.default_rng(seed)
    target_scale = float(np.std(y_fit)) or 1.0
    best = None

    for restart in range(n_restarts):
        if restart == 0:
            start = np.zeros(n_coeff)
            start[::2] = 1.0
        else:
            start = rng.normal(0.0, 1.0, n_coeff)

        def residuals(coefficients: np.ndarray) -> np.ndarray:
            prediction = predict(coefficients, features_fit, qhat_fit)
            bad = ~np.isfinite(prediction)
            if bad.any():
                prediction = np.where(bad, 0.0, prediction)
            error = (prediction - y_fit) / target_scale
            if bad.any():
                error[bad] = 1e3
            return error

        try:
            solution = least_squares(residuals, start, method="trf", max_nfev=max_nfev)
        except Exception:  # noqa: BLE001 - a diverging start should not kill the cell
            continue
        cost = float(solution.cost)
        if np.isfinite(cost) and (best is None or cost < best.cost):
            best = solution

    return best


def score_cell(cell: dict, args) -> dict:
    expression_id = cell["expression_id"]
    q_dim = cell["q_dim"]
    started = time.perf_counter()
    record = {"expression_id": expression_id, "q_dim": q_dim, "seed": args.seed, "status": "failed"}

    try:
        task, frame, learned_cols, truth_cols, q_source = load_cell_frame(expression_id, q_dim, args.seed)
        feature_cols = list(task.feature_columns)

        labels = np.sort(frame["label"].unique())
        rng = np.random.default_rng(args.seed)
        shuffled = rng.permutation(labels)
        n_fit = max(2, int(round(len(shuffled) * args.label_fit_ratio)))
        fit_labels = set(shuffled[:n_fit].tolist())

        is_fit = frame["label"].isin(fit_labels).to_numpy()
        features = frame.loc[:, feature_cols].to_numpy(float)
        y = frame["target"].to_numpy(float)
        qhat = frame.loc[:, learned_cols].to_numpy(float)

        record.update(
            {
                "true_rhs": task.rhs_expression,
                "latent_variables": list(task.latent_variables),
                "feature_columns": feature_cols,
                "q_source": q_source,
                "labels_total": int(len(labels)),
                "labels_fit": int(n_fit),
                "labels_heldout": int(len(labels) - n_fit),
                "rows_fit": int(is_fit.sum()),
                "rows_heldout": int((~is_fit).sum()),
                "families": {},
            }
        )

        labels_all = frame["label"].to_numpy()
        record["q_modes"] = {}
        for q_mode in args.q_modes:
            qhat_mode = apply_q_mode(qhat, labels_all, q_mode, args.seed)
            mode_entry: dict = {}
            for family in args.families:
                predict, n_coeff, coeff_names = build_substituted_callable(task, q_dim, family)
                solution = fit_substitution(
                    predict,
                    n_coeff,
                    features[is_fit],
                    qhat_mode[is_fit],
                    y[is_fit],
                    args.restarts,
                    args.seed,
                    args.max_nfev,
                )
                if solution is None:
                    mode_entry[family] = {"status": "no_convergent_start", "n_coefficients": n_coeff}
                    continue

                pred_fit = predict(solution.x, features[is_fit], qhat_mode[is_fit])
                pred_held = predict(solution.x, features[~is_fit], qhat_mode[~is_fit])
                ok_fit = np.isfinite(pred_fit)
                ok_held = np.isfinite(pred_held)

                mode_entry[family] = {
                    "status": "success",
                    "n_coefficients": n_coeff,
                    "coefficient_names": coeff_names,
                    "coefficients": [float(v) for v in solution.x],
                    "r2_fit_labels": r2(y[is_fit][ok_fit], pred_fit[ok_fit]),
                    "r2_heldout_labels": r2(y[~is_fit][ok_held], pred_held[ok_held]),
                    "nonfinite_fit_rows": int((~ok_fit).sum()),
                    "nonfinite_heldout_rows": int((~ok_held).sum()),
                }
            record["q_modes"][q_mode] = mode_entry
            best = max(
                (v.get("r2_heldout_labels", -np.inf) for v in mode_entry.values() if v.get("status") == "success"),
                default=float("nan"),
            )
            record[f"best_r2_heldout_{q_mode}"] = float(best)

        record["families"] = record["q_modes"].get("learned", {})

        # Reference ceiling: substitute the TRUE q into the TRUE structure. Any gap
        # between this and the learned-q scores is attributable to q, not structure.
        if len(truth_cols) == len(task.latent_variables):
            true_symbols = [sp.Symbol(name) for name in task.latent_variables]
            feature_symbols = [sp.Symbol(name) for name in feature_cols]
            oracle = sp.lambdify(
                (feature_symbols, true_symbols), sp.sympify(task.rhs_expression), modules=["numpy"]
            )
            with np.errstate(all="ignore"):
                q_true = frame.loc[:, list(task.latent_variables)].to_numpy(float)
                pred_oracle = np.asarray(
                    oracle(
                        [features[:, i] for i in range(features.shape[1])],
                        [q_true[:, j] for j in range(q_true.shape[1])],
                    ),
                    dtype=float,
                )
            ok = np.isfinite(pred_oracle)
            record["oracle_true_q_r2"] = r2(y[ok], pred_oracle[ok])

        best_family = max(
            (f for f, v in record["families"].items() if v.get("status") == "success"),
            key=lambda f: record["families"][f].get("r2_heldout_labels", -np.inf),
            default=None,
        )
        if best_family is not None:
            record["best_family"] = best_family
            record["best_r2_heldout"] = record["families"][best_family]["r2_heldout_labels"]
            record["recovered"] = bool(record["best_r2_heldout"] >= args.recovery_threshold)
            # Preregistered primary reading: linear f only, no per-cell family selection.
            linear = record["families"].get("linear", {})
            if linear.get("status") == "success":
                record["recovered_linear"] = bool(
                    linear["r2_heldout_labels"] >= args.recovery_threshold
                )
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001 - record and continue
        record["error"] = f"{type(exc).__name__}: {exc}"

    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/q_substitution_recovery_20260812")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--families", nargs="+", default=["linear", "quadratic"])
    parser.add_argument("--label-fit-ratio", type=float, default=0.5)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--max-nfev", type=int, default=4000)
    parser.add_argument("--recovery-threshold", type=float, default=0.9)
    parser.add_argument("--expressions", default="")
    parser.add_argument("--q-modes", nargs="+", default=["learned", "shuffled", "random"])
    parser.add_argument("--all-expressions", action="store_true", help="score every expression with a q artifact")
    args = parser.parse_args()

    cells = discover_cells(args.seed) if args.all_expressions else list(PROBE_CELLS)
    if args.expressions:
        wanted = {int(v) for v in args.expressions.split(",") if v.strip()}
        cells = [c for c in cells if c["expression_id"] in wanted]
    if not cells:
        raise SystemExit("No cells selected.")

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for cell in cells:
        record = score_cell(cell, args)
        records.append(record)
        families = record.get("families", {})
        summary = "  ".join(
            f"{name}={value.get('r2_heldout_labels'):.4f}"
            if isinstance(value.get("r2_heldout_labels"), float)
            else f"{name}={value.get('status')}"
            for name, value in families.items()
        )
        controls = "  ".join(
            f"{mode}={record.get(f'best_r2_heldout_{mode}'):.4f}"
            for mode in args.q_modes
            if isinstance(record.get(f"best_r2_heldout_{mode}"), float)
        )
        print(
            f"expr{record['expression_id']:03d} q{record['q_dim']} {record['status']}  "
            f"oracle={record.get('oracle_true_q_r2')}  {summary}  [{controls}]  "
            f"recovered={record.get('recovered')} linear={record.get('recovered_linear')}",
            flush=True,
        )

    (output_root / "substitution_scores.json").write_text(json.dumps(records, indent=2))

    rows = []
    for record in records:
        for q_mode, families in record.get("q_modes", {}).items():
            for family, value in families.items():
                rows.append(
                    {
                        "expression_id": record["expression_id"],
                        "q_dim": record["q_dim"],
                        "true_rhs": record.get("true_rhs"),
                        "q_mode": q_mode,
                        "family": family,
                        "n_coefficients": value.get("n_coefficients"),
                        "r2_fit_labels": value.get("r2_fit_labels"),
                        "r2_heldout_labels": value.get("r2_heldout_labels"),
                        "oracle_true_q_r2": record.get("oracle_true_q_r2"),
                        "recovered_best_family": record.get("recovered"),
                        "recovered_linear_only": record.get("recovered_linear"),
                    }
                )
    pd.DataFrame(rows).to_csv(output_root / "substitution_scores.csv", index=False)

    scored = [r for r in records if r.get("recovered") is not None]
    recovered = [r for r in scored if r["recovered"]]
    linear_scored = [r for r in records if r.get("recovered_linear") is not None]
    linear_recovered = [r for r in linear_scored if r["recovered_linear"]]
    print(
        f"\nsubstitution recovery rate (best family): {len(recovered)}/{len(scored)}"
        f"\nsubstitution recovery rate (linear f only, primary): "
        f"{len(linear_recovered)}/{len(linear_scored)}"
        f"\nthreshold: held-out R^2 >= {args.recovery_threshold}"
    )
    for mode in args.q_modes:
        values = [
            r[f"best_r2_heldout_{mode}"]
            for r in records
            if isinstance(r.get(f"best_r2_heldout_{mode}"), float)
            and np.isfinite(r[f"best_r2_heldout_{mode}"])
        ]
        if values:
            passing = sum(1 for v in values if v >= args.recovery_threshold)
            print(
                f"  q_mode={mode:9s} median held-out R^2={np.median(values):.4f}  "
                f"cells >= threshold: {passing}/{len(values)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
