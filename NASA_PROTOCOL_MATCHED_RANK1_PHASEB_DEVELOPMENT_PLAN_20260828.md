# NASA protocol-matched rank-1 Phase-B development replication

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-28
- Verification Status: UNVERIFIED before execution
- Version Label: nasa_protocol_matched_rank1_phaseb_development_v1

**Frozen before rank-1 Phase-B output.** The same structure-validation batteries were already exposed by rank-2 Phase B. This is a development replication to test the rank mechanism, not an independent holdout and not final paper confirmation.

## Frozen comparison

- Same 15 continuity checkpoints and five structure-validation batteries per split.
- Compare weights `{0,0.01}` only; `0.01` was selected by the completed rank-1 meta screen without structure-validation access.
- Functional subspace rank is 1. Raw-q prior is zero.
- Each held-out battery uses its first observed input protocol and cycle `{1,10,20,28}` probes.
- q calibration sees earliest 30% targets only; later 70% are evaluation and receive the unchanged +123.456 leakage audit.
- No target selects the protocol or weight.

## Frozen endpoints

Reuse the rank-2 Phase-B gates without modification: 15/15 integrity; median selected/baseline NRMSE ratio at most 1.05 and at least 10/15 cells within 10%; selected capacity/fade stability at least 0.70 median-of-splits and 0.50 minimum split, with response geometry no more than 0.05 below baseline per split; empirical capacity/fade alignment at least 0.70/0.50 and no more than 0.05 below baseline.

All gates passing would show that rank 1 repairs the exposed development cohort and authorize planning a genuinely new-battery confirmation. It does not itself authorize a confirmatory paper claim. Failure stops rank 1 on this cohort.

## Exact command

```bash
CUDA_VISIBLE_DEVICES=<empty> MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/run_nasa_protocol_matched_functional_prior_phaseb_20260828.py \
  --plan-path NASA_PROTOCOL_MATCHED_RANK1_PHASEB_DEVELOPMENT_PLAN_20260828.md \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_protocol_matched_rank1_phaseb_development_20260828 \
  --functional-prior-subspace-rank 1 --device cuda:0

MPLCONFIGDIR=/tmp/lvs-mpl .venv-lvs-gpu/bin/python \
  scripts/analyze_nasa_protocol_matched_functional_prior_phaseb_20260828.py \
  --root runs/nasa_protocol_matched_rank1_phaseb_development_20260828
```

Use one writer, no deadline or retry, and verify the formal root and GPU ownership immediately before execution.
