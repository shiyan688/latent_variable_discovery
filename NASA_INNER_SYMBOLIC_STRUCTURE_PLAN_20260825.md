# Reviewer-clean NASA inner symbolic-structure development plan

**Frozen:** 2026-08-25, before any cell was run

**Scope:** Stage C development only; no outer-test formula fitting or decoder modification

## Matrix

The frozen matrix has 90 PySR cells:

- three predeclared 8 meta-fit / 5 structure-validation battery splits;
- seeds 0--4;
- two q-independent interfaces per split/seed: physical conditions only and physical conditions plus prefix-support target summaries;
- two interfaces for each q source (`joint_continuity_step1`, `joint_mse_step1`): conditions plus raw q and conditions plus decoder-functional q.

The functional-q vocabulary is frozen from meta-fit-only stability evidence before symbolic scoring: decoder-derived `capacity_cycle1` and `early_fade_rate`. The physical-condition vocabulary is `discharge_index`, `ambient_temperature`, `load_current_amp`, and `cutoff_voltage`; entity-constant conditions are retained because they vary across batteries and identify experimental interventions.

All cells receive the same PySR budget: 60 iterations, maximum complexity 24, operators `{+, -, *, /, exp, log, sqrt, square}`, at most 1,200 meta-fit rows, two PySR processes, and a 1,800-second per-cell timeout.

## Information boundary

For every battery, rows are stably ordered by `discharge_index`; the earliest 30% are support and the remainder are symbolic query rows. Formulas are fitted only to query rows from the eight meta-fit batteries and scored only on query rows from the five structure-validation batteries. Support summaries use only prefix-support targets. Structure-validation raw q and functional q come from the already completed support-only calibration artifacts; perturbing structure-validation query targets must leave every symbolic input unchanged.

The learned train-entity q embeddings and their decoder functionals were optimized from the complete meta-fit curves, as in the underlying auto-decoder training protocol. This is allowed training information, but it is not described as information-matched to support-only validation q. The held-out-entity score is therefore the decisive transfer test.

## Frozen gates

1. Integrity: 90 unique terminal successes, finite predictions and metrics, exact 8/5 entity isolation, strict prefix ordering, and maximum query-target leakage-probe difference zero.
2. Downstream value: continuity functional-q has lower pooled median structure-validation NRMSE than both condition-only and support-statistics baselines, and wins at least 9/15 paired split/seed cells against each.
3. Motif recurrence: at least one selected expression uses `discharge_index` and one of the two frozen functional coordinates in at least 8/15 continuity functional-q cells, with recurrence in at least two splits and at least three seeds in each such split.
4. Readability: continuity functional-q median formula complexity is no greater than continuity raw-q median complexity.
5. Representation diagnostic: report the same raw/functional comparison for MSE q; continuity is favored as a symbolic interface only if its motif recurrence is stronger, regardless of its worse neural prediction NRMSE.

Passing all gates advances one recurring motif to the minimal structured-decoder design. Failure does not reject latent q: first diagnose feature scaling, train-q versus calibrated-q shift, symbolic budget, and split-specific protocol coverage. No threshold, operator set, or functional vocabulary may be tuned on these same 90 validation outcomes.

## Exact command

```bash
/public/home/wangyg/workspace/llm_pysr_project/.venv/bin/python \
  scripts/run_nasa_inner_symbolic_structure_20260825.py \
  --inner-q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --methods joint_continuity_step1,joint_mse_step1 \
  --seeds 0,1,2,3,4 \
  --functional-columns capacity_cycle1,early_fade_rate \
  --condition-columns discharge_index,ambient_temperature,load_current_amp,cutoff_voltage \
  --support-ratio 0.3 \
  --support-order-column discharge_index \
  --iterations 60 --maxsize 24 --sample-size 1200 \
  --pysr-procs 2 --max-parallel 2 --threads-per-job 1 \
  --job-timeout-seconds 1800 \
  --output-root runs/nasa_battery_reviewer_clean_inner_symbolic_20260825
```
