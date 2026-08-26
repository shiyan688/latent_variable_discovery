# NASA support-matched q interface diagnostic

**Frozen:** 2026-08-26, before any diagnostic cell was run

**Scope:** Diagnose and repair the Stage C q interface. This stage does not retrain the decoder, fit symbolic formulas, modify a model structure, or inspect the five reviewer-clean outer-test batteries.

## Question

Stage C compared full-curve jointly optimized meta-fit q with prefix-support-calibrated structure-validation q. The resulting coordinate shift was then amplified by unbounded symbolic operators. This diagnostic asks whether applying the same prefix-support inverse problem to both sides makes raw and decoder-functional q coordinates distributionally compatible.

## Frozen matrix and information boundary

- Source: all 30 completed inner-q checkpoints: three frozen 8/5 inner splits × five seeds × `joint_continuity_step1` and `joint_mse_step1`, q=4.
- Decoder and normalization statistics remain frozen.
- For each meta-fit entity, q is recalibrated only from its earliest 30% target rows. Its own full-curve train embedding is excluded from the initialization prior; the other seven embeddings define the leave-one-entity-out prior.
- For each structure-validation entity, q is recalibrated from its earliest 30% target rows using the eight meta-fit embeddings as the prior, exactly matching the original held-out protocol.
- Calibration keeps the original four starts, 200 fit steps, 25% support-internal selection when available, and 50 refinement steps.
- Query targets are scoring-only. Adding 123.456 to every structure-validation query target must change no calibrated q value.
- Decoder-functional coordinates remain the predeclared capacity at cycle 1 and early fade from cycle 1 to 10 at the common condition grid. No new coordinate is selected.

## Diagnostics

For every cell and entity, save support-calibrated q, the two functional coordinates, calibration candidate dispersion, q displacement from the full-curve embedding, standardized distance to the matched meta-fit q manifold, and the singular values of the support-output Jacobian with respect to q. Jacobian effective rank uses `s_i > 1e-4 s_max`; the reported condition number floors the denominator at `max(1e-8 s_max, 1e-12)`.

## Frozen advancement gates

1. **Integrity:** 30/30 successful cells, exact 8/5 entities per cell, and maximum query-target leakage difference exactly zero.
2. **Reproduction:** recalibrated structure-validation q differs from the saved q by at most `1e-4`, and reference NRMSE differs by at most `1e-5`, in every cell.
3. **Continuity shift:** pooled continuity median raw-q max-|z| is no more than half the Stage C value `22.1915`, and pooled continuity median functional max-|z| is at most `3.0` (Stage C: `7.3658`).
4. **Continuity tail:** at least 12/15 continuity cells have functional max-|z| at most `6.0`, with no non-finite shift metric.

Support-Jacobian conditioning is reported but is not an advancement gate. It distinguishes remaining coordinate shift from local inverse-problem non-identifiability. MSE q is a matched diagnostic comparator; advancement is governed by continuity q because continuity supplied the stable representation motif in the frozen inner audit.

Only if all four gates pass may a separate plan freeze bounded symbolic Stage C2. This diagnostic never launches Stage C2 automatically. Failure triggers diagnosis of the prefix inverse problem and q parameterization rather than post-hoc threshold changes.

## Exact commands

Run from the repository root after confirming GPUs 0 and 7 are empty. Each process sees one physical card and therefore uses logical `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_nasa_support_matched_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --method joint_continuity_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --device cuda:0

CUDA_VISIBLE_DEVICES=7 python scripts/run_nasa_support_matched_q_diagnostic_20260826.py \
  --q-root runs/nasa_battery_reviewer_clean_inner_q_20260825 \
  --method joint_mse_step1 --seeds 0,1,2,3,4 \
  --output-root runs/nasa_support_matched_q_diagnostic_20260826 \
  --device cuda:0

python scripts/analyze_nasa_support_matched_q_diagnostic_20260826.py \
  --root runs/nasa_support_matched_q_diagnostic_20260826
```
