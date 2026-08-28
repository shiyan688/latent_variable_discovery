# NASA protocol-matched functional prior Phase-B validation

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED before Phase-B execution
- Version Label: nasa_protocol_matched_functional_prior_phaseb_v1

**Frozen before reading any Phase-B output.** The preceding meta-fit screen selected functional-subspace weight `0.01` from the fixed grid without reading structure-validation data. This phase compares only that weight with weight 0; it may not switch weights after seeing validation outcomes.

## Scope and scientific question

For each of three reviewer-clean NASA inner splits and seeds 0--4, the eight meta-fit batteries remain the trained reference population and the five structure-validation batteries are held out for this protocol. These batteries were used by earlier project experiments, so this is an honest protocol-specific holdout, not a globally untouched paper test set.

The question is whether the meta-selected, gauge-invariant rank-2 functional prior transfers to unseen batteries while preserving prediction and producing stable, physically aligned q-derived coordinates.

## Frozen matrix and data boundary

- Source checkpoints: 15 completed `prefix_q_continuity_step1`, q=4 checkpoints.
- Methods: functional prior weights `{0, 0.01}` only. Both use subspace rank 2; raw-q prior weight is zero.
- Each structure-validation battery is calibrated independently from its earliest 30% rows. Later 70% targets are evaluation only.
- Probe protocol: ambient temperature, load current, and cutoff voltage from that battery's first row after stable sorting by `discharge_index`; target values do not select the protocol.
- Probe cycles: `{1,10,20,28}` at that frozen protocol.
- The eight original meta-fit train embeddings define the standardized response population and rank-2 SVD subspace for each held-out battery's probe features. Standard-deviation floor remains `0.05`.
- Calibration uses the existing deterministic four starts, 200 fit steps, support-internal candidate selection, and 50-step refinement.
- Adding `123.456` to later query targets must leave every calibrated q unchanged.
- Empirical capacity and early-fade descriptors are fit from actual structure-validation targets at cycles at most 10. They are evaluation endpoints only and never enter q calibration or weight selection.

## Frozen endpoints and advancement rule

1. **Integrity:** 15/15 finite cells; exactly five validation labels, two weights and ten q rows per cell; maximum query-target perturbation q difference zero; exact selected weight `0.01` in every cell.
2. **Prediction retention:** pooled selected/baseline NRMSE ratio at most `1.05`, and at least 10/15 paired cells have ratio at most `1.10`.
3. **Functional stability:** for the selected method, cycle-1 capacity and early fade each have median-of-split-medians at least `0.70` and minimum split median at least `0.50`. In every split, selected median response-distance stability may be no more than `0.05` below weight 0.
4. **Scientific alignment:** selected capacity has median-of-split-medians Spearman at least `0.70` with empirical initial capacity; selected early fade has at least `0.50` with empirical early fade. For both coordinates, selected alignment may be no more than `0.05` below weight 0.

Report paired wins, per-split NRMSE, raw-q stability, worst seed pairs, empirical correlations, and a paired Wilcoxon test as diagnostics. The test does not replace effect sizes or gates.

All four gates must pass to authorize a separately frozen bounded symbolic Stage C2 using the selected functional interface. Failure stops this exact prior from Stage C2 but does not negate the protocol-matched weight-0 functional coordinates. No formula, structured decoder, or new weight is selected in Phase B.

## Exact command

```bash
CUDA_VISIBLE_DEVICES=<empty> MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/run_nasa_protocol_matched_functional_prior_phaseb_20260828.py \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_protocol_matched_functional_prior_phaseb_20260828 \
  --device cuda:0

MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/analyze_nasa_protocol_matched_functional_prior_phaseb_20260828.py \
  --root runs/nasa_protocol_matched_functional_prior_phaseb_20260828
```

Use one writer per output root, no deadline, no automatic retry, and preserve any failure. Before execution verify the formal root is absent and the physical GPU has no foreign process or material resident memory.

## Expected outputs

| Output | Path | Success criterion |
|---|---|---|
| Manifest | `<root>/manifest.json` | frozen plan/runner hashes and selected weight 0.01 |
| Per-cell evidence | `<root>/<dataset>/seed*/` | q, query predictions, summaries for both weights |
| Terminal status | `<root>/status.json` | 15/15 success, zero failed |
| Gate audit | `<root>/gate_decision.json` | four explicit booleans and overall decision |
| Readable report | `<root>/PHASEB_REPORT.md` | per-dataset and aggregate results |
