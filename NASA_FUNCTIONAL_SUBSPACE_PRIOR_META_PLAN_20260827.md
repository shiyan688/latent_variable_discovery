# NASA functional-subspace prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-27
- Verification Status: UNVERIFIED before formal execution
- Version Label: nasa_functional_subspace_prior_meta_v1

**Frozen before any formal subspace-prior cell.** This is sequential development on already exposed NASA inner meta-fit splits, not independent confirmation.

## Objective and hypothesis

The completed functional-response mean-prior screen passed integrity but stopped before structure validation. Increasing its weight compressed between-entity four-probe response distances while enlarging raw-q distances and worsening later-cycle meta-query prediction. The next minimal hypothesis is that the mean prior penalized legitimate capacity/fade variation together with unsafe functional directions.

This screen preserves the leading two standardized response-signature directions and penalizes only the orthogonal residual. Rank two is fixed because cycle-1 capacity and early fade were the two independently frozen functional coordinates before either response-prior screen. The penalty remains invariant to an equivalent affine reparameterization of raw q.

## Frozen Phase-A matrix

- Source: the same 15 completed `prefix_q_continuity_step1` checkpoints: three inner splits × seeds 0--4, q=4.
- Data boundary: only the eight meta-fit batteries in each cell. Structure-validation batteries and targets are not read.
- Functional probes: `(cycle, ambient, load, cutoff)` = `(1,24,2,2.5)`, `(10,24,2,2.5)`, `(20,24,2,2.5)`, `(28,24,2,2.5)`.
- For each held-out meta-fit battery, the other seven train embeddings define normalized four-probe signatures. Standardize each probe using the leave-one-out mean and a standard-deviation floor of `0.05`.
- Compute the rank-2 SVD subspace of those seven standardized signatures. The prior loss is the mean squared residual after projecting the candidate signature onto that subspace. No penalty is applied along the two retained directions.
- Candidate weights remain `{0, 0.001, 0.01, 0.1, 1}`; raw-q prior weight remains zero. No smaller weight is added after observing the failed mean-prior screen.
- Infer every q from the earliest 30% support with the existing deterministic four starts, 200 fit steps, support-internal candidate selection, and 50-step refinement.
- Later 70% meta-fit targets are development scoring only. Adding `123.456` to them must leave every candidate q unchanged.

The single inner0/seed0/weight-0.001 GPU smoke is structural only: it produced eight finite q rows, zero leakage, support loss `0.01438`, and meta-query NRMSE `0.05593`, versus `0.05619` for the already completed weight-0 cell. It is not a formal outcome.

## Frozen selection and stop rule

Aggregate all 15 cells before any method choice:

1. **Integrity:** 15/15 finite cells, five weights and eight q rows per weight per cell, exact frozen grid, and maximum query-target perturbation difference zero.
2. **Prediction retention:** candidate median meta-query NRMSE is at most `1.05 ×` the weight-0 median.
3. **Named functional stability:** cycle-1 capacity and early fade each have median-of-split-medians at least `0.70` and minimum split median at least `0.50`, unchanged from the prior screens.
4. **Response-geometry retention:** within every split, the candidate's median cross-seed Spearman correlation of four-probe pairwise response distances is no more than `0.05` below the same split's weight-0 value.

Raw-q distance stability is saved as a diagnostic, not an eligibility gate. This is a change of estimand rather than a lowered threshold: the proposed paper interface exposes decoder-functional coordinates, and exact q/first-layer affine gauge means Euclidean raw-q distances are not invariant. The response-geometry rule replaces that non-invariant endpoint before any subspace-prior formal outcome is observed.

Among eligible candidates, select the lowest median meta-query NRMSE; ties select the smaller weight. If none is eligible, stop this exact rank/grid without structure-validation access. An eligible candidate authorizes only a separately frozen Phase-B validation protocol; it does not authorize Stage C2 or a paper claim.

## Exact command and monitoring

Run only after a fresh host check identifies a genuinely empty physical GPU:

```bash
CUDA_VISIBLE_DEVICES=<empty> python scripts/run_nasa_functional_response_prior_meta_20260827.py \
  --plan-path NASA_FUNCTIONAL_SUBSPACE_PRIOR_META_PLAN_20260827.md \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_functional_subspace_prior_meta_20260827 \
  --functional-prior-subspace-rank 2 --device cuda:0

python scripts/analyze_nasa_functional_response_prior_meta_20260827.py \
  --root runs/nasa_functional_subspace_prior_meta_20260827 \
  --representation-gate functional-response
```

Monitor the single writer's process, append-only cell count, output growth, and GPU ownership every 30--60 seconds. There is no experiment deadline and no automatic retry. Any nonzero cell remains visible and stops advancement.

## Expected outputs

| Output | Path | Success criterion |
|---|---|---|
| Per-cell grid | `<root>/<dataset>/seed*/` | 15 cells, five weights and eight q rows/weight |
| Terminal status | `<root>/status.json` | 15/15 success, zero failed |
| Eligibility audit | `<root>/weight_eligibility.csv` | prediction, response geometry, capacity and fade gates |
| Frozen decision | `<root>/selected_weight.json` | selected weight or explicit STOP; no structure-validation use |
| Readable report | `<root>/FUNCTIONAL_RESPONSE_PRIOR_META_REPORT.md` | all fixed weights and gates |
