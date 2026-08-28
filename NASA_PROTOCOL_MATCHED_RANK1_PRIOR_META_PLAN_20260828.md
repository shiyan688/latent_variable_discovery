# NASA protocol-matched rank-1 functional prior meta screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED before formal rank-1 execution
- Version Label: nasa_protocol_matched_rank1_prior_meta_v1

**Frozen before any formal rank-1 cell.** This is sequential method development after Phase B exposed a rank-2 early-fade failure. It is not independent confirmation.

## Hypothesis

Rank 0 over-shrank legitimate capacity and fade variation. Rank 2 preserved prediction and full response geometry but intentionally left the unstable fade direction unregularized. In Phase B, capacity seed noise was only 3--4% of between-battery spread, while early-fade noise was 36--58%; several batteries also showed activation/recovery transients or protocol changes that make cycle-1-to-10 fade heterogeneous.

The minimal intermediate hypothesis is rank 1: preserve the dominant standardized response direction, expected to represent overall capacity, while softly penalizing orthogonal response-shape directions. This changes only the retained subspace rank.

## Frozen matrix

- Same 15 `prefix_q_continuity_step1`, q=4 checkpoints: three meta-fit splits × seeds 0--4.
- Meta-fit batteries only; structure-validation inputs and targets are not read.
- Each leave-one-out battery uses cycle `{1,10,20,28}` probes at its first observed ambient/load/cutoff protocol, selected from input features without targets.
- The other seven original train embeddings define standardized probe responses with standard-deviation floor `0.05`; retain SVD rank 1 and penalize only the residual.
- Fixed weights `{0,0.001,0.01,0.1,1}`; raw-q prior zero.
- Existing earliest-30% support calibration, four starts, 200 fit steps, support-internal selection, and 50-step refinement.
- Later 70% meta-fit targets are development scoring only. Adding `123.456` to them must leave q unchanged.

## Frozen gates

Use the same analyzer and thresholds as the preceding protocol-matched rank-2 meta screen:

1. 15/15 finite integrity, exact grid/entity counts, zero query-target leakage.
2. Candidate meta-query NRMSE at most `1.05 ×` weight 0.
3. Capacity and early fade median-of-split-medians at least `0.70`, minimum split median at least `0.50`.
4. In every split, candidate median four-response distance stability no more than `0.05` below weight 0.

Select the eligible weight with lowest pooled meta-query NRMSE, ties choosing the smaller weight. No eligible weight means STOP. Because the earlier Phase B already exposed structure-validation outcomes, any later rank-1 evaluation on those batteries is development replication only; a paper claim requires new batteries or another real dataset.

## Exact command

```bash
CUDA_VISIBLE_DEVICES=<empty> MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/run_nasa_functional_response_prior_meta_20260827.py \
  --plan-path NASA_PROTOCOL_MATCHED_RANK1_PRIOR_META_PLAN_20260828.md \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_protocol_matched_rank1_prior_meta_20260828 \
  --functional-prior-subspace-rank 1 \
  --functional-prior-protocol first-observed --device cuda:0

MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/analyze_nasa_functional_response_prior_meta_20260827.py \
  --root runs/nasa_protocol_matched_rank1_prior_meta_20260828 \
  --representation-gate functional-response
```

Use a single writer, no deadline, no retry, and preserve failures. Verify the exact root is absent and GPU ownership immediately before launch.
