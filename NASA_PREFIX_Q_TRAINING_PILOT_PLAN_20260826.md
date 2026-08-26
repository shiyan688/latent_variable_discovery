# NASA information-matched prefix-q training pilot

**Frozen:** 2026-08-26 21:19 CST, before any prefix-q training cell was run

**Status:** Sequential inner-split development. The method and gates may select a candidate for later fresh evaluation; this pilot is not independent confirmation.

## Hypothesis

Post-training convex and coordinate-box projection both made q geometrically safe but failed prediction retention. This indicates that a decoder trained with full-curve entity embeddings may rely on a q interface that cannot be repaired after training. The proposed change aligns the per-entity training and test pathways: training q updates use only the earliest 30% of each meta-fit entity, while the shared decoder updates use all training rows. At test time q continues to use the earliest 30% support and the decoder predicts later rows.

## Frozen method

- Matrix: three reviewer-clean NASA inner splits × five seeds × q=4 × prefix-q MSE/continuity = 30 cells.
- Start from the existing step1 architecture, hidden sizes, normalization, 1,000 epochs, batch size 256, learning rates, four-start test calibration, and 30% prefix support protocol.
- Use alternating blocks in each training batch. First update q once using only rows that fall in the earliest 30% for their training entity; then update decoder theta once using the full batch with q frozen.
- For continuity, compute the fixed train-entity response-distance target from prefix rows only. The continuity weight remains 0.05.
- Per batch, theta and q each receive one optimizer step, matching the per-parameter update count of the old joint step1 method. Total backward passes double because the two parameter blocks now have separate information sets; runtime and counters must be reported.
- Test calibration code, support-internal selection, refinement, decoder architecture, and q dimension do not change.
- Method names: `prefix_q_mse_step1` and `prefix_q_continuity_step1`.

## Frozen gates

1. **Integrity:** 30/30 finite successful cells; exact 8/5 train/validation entities; every latent config records alternating optimization, prefix q training, ratio 0.3, and discharge-index feature 0; theta steps equal q steps; backward passes equal their sum; all required q/checkpoint/prediction artifacts exist.
2. **Prediction retention:** pooled prefix-q continuity median NRMSE is at most 1.05 times the same-cell old unconstrained support-matched continuity median, and at least 10/15 per-cell ratios are at most 1.10.
3. **Interface safety:** pooled prefix-q continuity median train-to-validation raw-q and decoder-functional max-|z| are each at most 3.0, and at least 12/15 functional cells are at most 6.0. Functional coordinates are the frozen cycle-1 capacity and early fade.
4. **Representation stability:** for prefix-q continuity, every inner split has median cross-seed q-distance Spearman at least 0.80. Cycle-1 capacity and early-fade cross-seed Spearman must each have a median-of-split-medians at least 0.70 and a minimum split median at least 0.50.

Only all four gates may authorize a separately frozen bounded symbolic Stage C2. Failure triggers training-dynamics diagnosis; it does not authorize wider post-hoc q bounds, a favorable split subset, or a structured decoder.

## Exact training command

```bash
python scripts/run_iclr_real_discovery.py launch \
  --prepared-summary data/real_datasets2/prepared/nasa_battery_reviewer_clean_20260825/inner_prepared_datasets.json \
  --methods prefix_q_continuity_step1,prefix_q_mse_step1 \
  --seeds 0,1,2,3,4 --q-dims 4 \
  --gpus 0,6,7 --gpu-memory-threshold-mib 128 \
  --epochs 1000 --cal-steps 200 \
  --cal-init-mode prior_random --cal-num-starts 4 \
  --cal-selection-ratio 0.25 --cal-selection-min-rows 24 \
  --cal-refine-steps 50 --cal-refine-only-after-selection \
  --support-ratio 0.3 --support-split-mode prefix \
  --support-order-column discharge_index \
  --batch-size 256 --hidden-sizes 256,128 \
  --max-train-per-label 0 --max-test-per-label 0 \
  --subsample-seed 20260825 --save-artifacts --no-resume \
  --output-root runs/nasa_prefix_q_training_pilot_20260826
```

The launcher dispatches only cards below 128 MiB, never terminates another job, has no campaign deadline, and does not automatically retry a failed cell.

## Frozen post-run analysis

For each successful `result.json`, run `scripts/analyze_nasa_q_functional_coordinates_20260825.py` into the mirrored `functional_coordinate_analysis/` directory. Then run:

```bash
python scripts/aggregate_nasa_inner_functional_coordinates_20260825.py \
  --run-root runs/nasa_prefix_q_training_pilot_20260826

python scripts/analyze_nasa_prefix_q_training_pilot_20260826.py \
  --root runs/nasa_prefix_q_training_pilot_20260826 \
  --baseline-root runs/nasa_support_matched_q_diagnostic_20260826
```
