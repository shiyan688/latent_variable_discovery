# NASA protocol-matched functional-subspace prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-27
- Verification Status: UNVERIFIED before formal execution
- Version Label: nasa_protocol_matched_functional_subspace_v1

**Frozen before any formal protocol-matched-prior cell.** This is sequential development on already exposed NASA inner meta-fit splits, not independent confirmation.

## Why this correction is necessary

The completed fixed-probe rank-2 screen retained prediction and four-point response geometry but failed the named early-fade coordinate. A post-terminal, no-retraining diagnostic then found that the frozen `(24°C, 2A, 2.5V)` probe had zero exact rows among 716 inner1 training rows and covered only 168/711 rows in inner0 and inner2. At each battery's first observed protocol, the saved weight-1 q candidates recovered capacity rank stability to `1.000/0.976` and early-fade stability to `0.893/0.833` when measured by the already frozen median-of-split-medians/minimum-split-median estimand. Thus the next experiment corrects an off-protocol measurement rather than relaxing a failed method threshold.

## Frozen Phase-A matrix

- Source: the same 15 completed `prefix_q_continuity_step1` checkpoints: three inner splits × seeds 0--4, q=4.
- Data boundary: only the eight meta-fit batteries in each cell. Structure-validation batteries and targets are not read.
- Per-battery protocol: sort that battery's meta-fit rows by `discharge_index` and take the ambient temperature, load current, and cutoff voltage from the first row. These input features are known at calibration time; target values do not choose the protocol.
- Functional probes: cycles `{1,10,20,28}` at that battery's frozen first-observed protocol.
- For each held-out battery, evaluate the other seven train embeddings at the held-out battery's probe features, standardize each response coordinate using their leave-one-out mean and a standard-deviation floor of `0.05`, retain the rank-2 SVD subspace, and penalize only the orthogonal residual.
- Rank two remains tied to the independently frozen two-coordinate interface: cycle-1 capacity and early fade. The prior remains invariant to affine reparameterization of raw q.
- Candidate weights: `{0, 0.001, 0.01, 0.1, 1}`; raw-q prior weight zero.
- Infer every q from the earliest 30% support using the existing deterministic four starts, 200 fit steps, support-internal candidate selection, and 50-step refinement.
- Later 70% meta-fit targets are development scoring only. Adding `123.456` to them must leave every candidate q unchanged.

The preceding protocol-matched analysis of saved q values is diagnostic only. It did not rerun calibration with the corrected prior and cannot select a formal weight.

## Frozen selection and stop rule

Aggregate all 15 cells before choosing a method:

1. **Integrity:** 15/15 finite cells, five weights and eight q rows per weight per cell, exact frozen grid, and maximum query-target perturbation difference zero.
2. **Prediction retention:** candidate median meta-query NRMSE is at most `1.05 ×` the weight-0 median.
3. **Named functional stability:** cycle-1 capacity and early fade each have median-of-split-medians at least `0.70` and minimum split median at least `0.50`.
4. **Response-geometry retention:** within every split, the candidate's median cross-seed Spearman correlation of protocol-matched four-probe pairwise response distances is no more than `0.05` below the same split's weight-0 value.

These are the same functional-response estimands and thresholds frozen for the preceding rank-2 screen. Raw-q distance stability and the worst individual seed-pair correlation are diagnostics only. Among eligible candidates, select the lowest median meta-query NRMSE; ties select the smaller weight. If none is eligible, stop without structure-validation access. Eligibility authorizes only a separately frozen Phase-B outer validation protocol.

## Exact command and monitoring

Run only after a fresh host check identifies a genuinely empty physical GPU:

```bash
CUDA_VISIBLE_DEVICES=<empty> .venv-lvs-gpu/bin/python \
  scripts/run_nasa_functional_response_prior_meta_20260827.py \
  --plan-path NASA_PROTOCOL_MATCHED_FUNCTIONAL_SUBSPACE_PLAN_20260827.md \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_protocol_matched_functional_subspace_20260827 \
  --functional-prior-subspace-rank 2 \
  --functional-prior-protocol first-observed --device cuda:0

.venv-lvs-gpu/bin/python \
  scripts/analyze_nasa_functional_response_prior_meta_20260827.py \
  --root runs/nasa_protocol_matched_functional_subspace_20260827 \
  --representation-gate functional-response
```

Monitor process ownership, append-only cell count, output growth, and GPU ownership every 30--60 seconds. There is no experiment deadline and no automatic retry. Any nonzero cell remains visible and stops advancement.

## Expected outputs

| Output | Path | Success criterion |
|---|---|---|
| Per-cell grid | `<root>/<dataset>/seed*/` | 15 cells, five weights and eight q rows/weight |
| Terminal status | `<root>/status.json` | 15/15 success, zero failed |
| Eligibility audit | `<root>/weight_eligibility.csv` | prediction, response geometry, capacity and fade gates |
| Frozen decision | `<root>/selected_weight.json` | selected weight or explicit STOP; no structure-validation use |
| Readable report | `<root>/FUNCTIONAL_RESPONSE_PRIOR_META_REPORT.md` | all fixed weights and gates |
