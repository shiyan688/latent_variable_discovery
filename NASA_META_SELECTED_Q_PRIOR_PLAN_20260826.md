# NASA meta-selected soft q-prior diagnostic

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-26
- Verification Status: UNVERIFIED before execution
- Version Label: nasa_meta_q_prior_v1

**Frozen:** 2026-08-26 21:43 CST, before any meta-selected-prior diagnostic cell was run

**Status:** Sequential inner-split development, not independent confirmation.

**Execution-only resource amendment (2026-08-27, before any formal cell):** the originally named physical GPUs 0 and 7 were occupied. A fresh host snapshot found physical GPUs 4 and 5 at 4 MiB with no compute processes, so the two method processes use 4 and 5 instead. This changes only physical device assignment; the source checkpoints, matrix, prior grid, selection rule, seeds, gates, output root, and no-retry/no-deadline rules are unchanged.

## Objective and hypothesis

Information-matched prefix-q training improved 10/15 paired continuity cells and preserved q-distance stability, but held-out calibration still left the training coordinate range. Train q co-evolves with the decoder under Adam lr 0.001, whereas held-out q uses a frozen decoder, lr 0.05, and 200+50 steps. The hypothesis is that a soft standardized train-q prior, selected strictly inside meta-fit support, can control calibration drift without the prediction loss caused by hard convex/box projection.

## Frozen matrix and method

- Source: all 30 completed prefix-q checkpoints: three inner splits × five seeds × `prefix_q_continuity_step1`/`prefix_q_mse_step1`, q=4.
- Candidate prior weights: `{0, 0.001, 0.01, 0.1, 1.0}`. No candidate may be added after outcomes are inspected.
- For each meta-fit entity and candidate weight, exclude that entity's embedding from the q prior, calibrate q from its earliest 30% support with the original four starts/200 steps/support-internal selection/50-step refinement, and record the support-internal selection loss. The later 70% target is diagnostic only and does not select the prior.
- Select one weight per split/method/seed by the median support-internal selection loss over all eight meta-fit entities; ties choose the smaller weight.
- Reuse the eight selected, support-calibrated meta-fit q values as the prior population for calibrating the five structure-validation entities. Decoder, normalization, q dimension, starts, learning rate, and step counts stay frozen.
- Query targets never select the weight or q. Adding 123.456 to every structure-validation query target must leave selected q unchanged.
- Save all five meta-fit score rows, selected support-matched q/functional coordinates, validation predictions, selected weight, and calibration diagnostics.

## Variables and confounds

- Independent variable: standardized q-prior weight selected from the frozen grid.
- Primary outcomes: held-out reference NRMSE, raw/functional train-to-validation max-|z|, and cross-seed stability of selected support-calibrated meta-fit q.
- Fixed controls: checkpoint, entity split, support/query split, optimizer steps, decoder, q dimension, seed, and functional coordinate definitions.
- Remaining limitation: the decoder saw all meta-fit targets during training. Support-internal prior selection avoids direct use of later targets, but this remains development on previously exposed inner splits.

## Frozen advancement gates

1. **Integrity:** 30/30 finite successful cells; exactly 8 meta-fit and 5 structure-validation q values per cell; all five frozen weights scored; selected weight belongs to the grid; and maximum query-target perturbation difference exactly zero.
2. **Prediction retention:** continuity selected-prior median NRMSE is at most 1.05 times the old unconstrained support-matched continuity median, and at least 10/15 per-cell ratios to that old comparator are at most 1.10.
3. **Interface safety:** continuity median raw-q and functional max-|z| are each at most 3.0, with at least 12/15 functional cells at most 6.0.
4. **Representation stability:** selected continuity meta-fit q has median cross-seed distance Spearman at least 0.80 in every split. Cycle-1 capacity and early fade each have median-of-split-medians at least 0.70 and minimum split median at least 0.50.

All four gates are mandatory. Failure ends this grid and selection rule without choosing weights from structure-validation results. Passing authorizes only a separately frozen bounded symbolic Stage C2.

## Exact commands

The originally frozen example named physical GPUs 0 and 7. Under the execution-only amendment above, substitute the freshly verified empty GPUs 4 and 5:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_nasa_meta_selected_q_prior_20260826.py \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --method prefix_q_continuity_step1 \
  --output-root runs/nasa_meta_selected_q_prior_20260826 --device cuda:0

CUDA_VISIBLE_DEVICES=5 python scripts/run_nasa_meta_selected_q_prior_20260826.py \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --method prefix_q_mse_step1 \
  --output-root runs/nasa_meta_selected_q_prior_20260826 --device cuda:0

python scripts/analyze_nasa_meta_selected_q_prior_20260826.py \
  --root runs/nasa_meta_selected_q_prior_20260826 \
  --old-baseline-root runs/nasa_support_matched_q_diagnostic_20260826
```

Monitoring uses process-alive, cell-count, status-ledger, and GPU-ownership checks. There is no automatic retry and no experiment deadline; any nonzero cell stops its method process and remains visible.

## Expected outputs and success criteria

| Output | Path | Success criterion |
|---|---|---|
| Atomic q/prior cells | `runs/nasa_meta_selected_q_prior_20260826/<method>/<dataset>/seed*/` | 30 complete cells |
| Method status | `<method>/status.json` | 15/15 success, zero failed |
| Consolidated cells/q | `all_cells.csv`, `all_selected_q.csv` | finite, exact counts |
| Frozen decision | `gate_decision.json` | generated only after terminal matrix |
| Readable report | `META_SELECTED_Q_PRIOR_REPORT.md` | reports all gates and both methods |
