# NASA convex-support q diagnostic

**Frozen:** 2026-08-26 16:41 CST, before any convex-support diagnostic cell was run

**Scope:** Test one minimal repair of the failed support-matched q interface. This diagnostic does not retrain the decoder, fit or select symbolic formulas, alter a decoder structure, or inspect the five reviewer-clean outer-test batteries.

## Question

The support-matched diagnostic reduced the continuity functional-coordinate shift by about 42%, but failed its frozen median and tail gates. Its remaining shift tracked distance from the support-matched meta-fit q manifold much more strongly than support-Jacobian condition number. This diagnostic asks whether forcing each held-out q to remain inside the convex hull of the eight support-matched meta-fit q anchors removes that interface failure while retaining prediction quality.

## Frozen method and information boundary

- Source: the same 30 completed inner-q checkpoints and the 30 terminal support-matched cells: three frozen 8/5 inner splits × five seeds × `joint_continuity_step1` and `joint_mse_step1`, q=4.
- Decoder, normalization statistics, splits, support ratio, support-internal selection rule, optimizer learning rate, fit steps, and refinement steps remain frozen.
- For each structure-validation entity, define `q = softmax(alpha) @ Q_anchor`, where `Q_anchor` contains the eight support-matched meta-fit q values from the same split, method, and seed.
- Optimize `alpha` from the earliest 30% support targets only. Use four starts: uniform weights, a near-one-hot start at the best individual anchor, and two deterministic random starts. Select and refine with the same support-only protocol as the source calibration.
- The five structure-validation query targets are scoring-only. Adding 123.456 to every query target must change no q value.
- The prior unconstrained support-matched prediction is the frozen performance comparator. No optimizer setting or gate may be tuned after inspecting convex-support outcomes.
- MSE q remains a matched diagnostic comparator. Advancement is governed by continuity q.

## Frozen advancement gates

1. **Integrity:** 30/30 successful cells; exactly 8 anchors and 5 structure-validation entities in every cell; all recorded metrics finite; maximum query-target leakage difference exactly zero; simplex weight-sum error at most `1e-6`; and no negative mixture weight.
2. **Convex containment:** all 15 continuity cells have raw-q validation max-|z| at most `3.0`. With eight anchors this is also a geometry/integrity check, not a fitted effect threshold.
3. **Functional shift:** continuity median functional max-|z| is at most `3.0`, and at least 12/15 continuity cells have functional max-|z| at most `6.0`. These are unchanged from the failed support-matched safety gate.
4. **Prediction retention:** continuity median convex-support NRMSE is at most 1.05 times the median unconstrained support-matched NRMSE, and at least 10/15 continuity cells have a per-cell NRMSE ratio at most `1.10`.

Only if all four gates pass may a separate plan freeze bounded symbolic Stage C2. Passing would show that a support-only q interface can be kept on the training manifold without materially losing the source predictor; it would not by itself show symbolic downstream value. Failure ends this exact convex-hull repair without lowering gates or choosing a favorable subset.

## Exact commands

Run from the repository root after confirming physical GPUs 0 and 7 are empty. Each process sees one physical card and therefore uses logical `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_nasa_convex_support_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --matched-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --method joint_continuity_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_convex_support_q_diagnostic_20260826 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=7 python scripts/run_nasa_convex_support_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --matched-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --method joint_mse_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_convex_support_q_diagnostic_20260826 \
  --device cuda:0

python scripts/analyze_nasa_convex_support_q_diagnostic_20260826.py \
  --root runs/nasa_convex_support_q_diagnostic_20260826
```
