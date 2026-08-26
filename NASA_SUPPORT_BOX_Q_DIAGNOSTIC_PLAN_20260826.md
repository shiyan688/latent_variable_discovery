# NASA support-box q diagnostic

**Frozen:** 2026-08-26 21:08 CST, before any support-box diagnostic cell was run

**Status:** Sequential development evidence. The same 30 inner cells have informed the support-matched and convex-support hypotheses, so this stage can select a repair for a later fresh evaluation but cannot itself be called independent confirmation.

## Question

The convex-support diagnostic made q geometrically safe but usually collapsed to one training-entity anchor and failed prediction retention. This diagnostic separates hard geometric bounding from discrete anchor selection: start with the already audited unconstrained support-calibrated q and apply the minimum coordinate-wise change required to place every q coordinate inside the range of the eight support-matched meta-fit q anchors.

## Frozen method and information boundary

- Source: the same 30 completed inner-q checkpoints and terminal support-matched q cells: three 8/5 inner splits × five seeds × `joint_continuity_step1` and `joint_mse_step1`, q=4.
- For each cell and q coordinate j, compute `[min_i Q_anchor[i,j], max_i Q_anchor[i,j]]` from the eight support-matched meta-fit entities.
- For each structure-validation entity, set `q_box[j] = clip(q_unconstrained[j], min_j, max_j)`. There is no fitted hyperparameter, softmax, anchor choice, or additional access to support/query targets.
- The source unconstrained q was calibrated only from the earliest 30% support and already passed exact query-target perturbation auditing. The box transformation is deterministic and target-free; source leakage must be zero in every consumed cell.
- Decoder, normalization, splits, conditions, and functional coordinates remain frozen. Query targets are used only to score predictions from `q_box`.
- The unconstrained support-matched predictor remains the frozen performance comparator. MSE q is diagnostic; advancement is governed by continuity q.

## Frozen advancement gates

1. **Integrity:** 30/30 successful cells; exactly 8 anchors and 5 structure-validation entities per cell; all q, predictions, and recorded metrics finite; every upstream query-leakage audit exactly zero; and maximum coordinate-box violation at most `1e-7`.
2. **Box containment:** all 15 continuity cells have raw-q validation max-|z| at most `3.0`.
3. **Functional shift:** continuity median functional max-|z| at most `3.0`, with at least 12/15 cells at most `6.0`. These thresholds are unchanged from the previous two diagnostics.
4. **Prediction retention:** continuity median box-projected NRMSE at most 1.05 times the median unconstrained support-matched NRMSE, with at least 10/15 per-cell NRMSE ratios at most `1.10`.

Only all four passes may authorize a separately frozen bounded symbolic Stage C2. A pass would establish a prediction-retaining, support-only bounded q interface on these development splits; it would not itself establish symbolic downstream value or an independent paper claim. Failure ends this exact coordinate-box repair without tuning wider margins on these cells.

## Exact commands

Run from the repository root after confirming physical GPUs 0 and 7 are empty. Each process sees one physical card and therefore uses logical `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_nasa_support_box_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --matched-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --method joint_continuity_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_support_box_q_diagnostic_20260826 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=7 python scripts/run_nasa_support_box_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --matched-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --method joint_mse_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_support_box_q_diagnostic_20260826 \
  --device cuda:0

python scripts/analyze_nasa_support_box_q_diagnostic_20260826.py \
  --root runs/nasa_support_box_q_diagnostic_20260826
```
