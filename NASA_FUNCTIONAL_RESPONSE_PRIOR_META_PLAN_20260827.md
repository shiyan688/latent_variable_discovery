# NASA functional-response prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run → validate
- Origin Date: 2026-08-27
- Verification Status: UNVERIFIED before execution
- Version Label: nasa_functional_response_prior_meta_v1

**Frozen before any formal meta-screen cell.** This is sequential development on the already exposed NASA inner splits, not independent confirmation.

## Objective and hypothesis

The completed raw-q Gaussian prior grid cannot satisfy the existing representation gate at any fixed weight. Its reference distribution is expressed in each seed's arbitrary q coordinates, so it preserves the exact q/first-layer affine gauge. The next minimal hypothesis is to regularize the decoder response at four already frozen reference conditions instead. Decoder outputs are invariant to an equivalent affine reparameterization of q and are the source of the capacity/fade coordinates used by the symbolic interface.

## Phase A: meta-fit only

- Source: the 15 completed `prefix_q_continuity_step1` checkpoints: three inner splits × seeds 0--4, q=4.
- Entities: only the eight meta-fit batteries in each cell. Structure-validation batteries and targets are not read in Phase A.
- Functional probes: the previously frozen conditions `(cycle, ambient, load, cutoff)` = `(1,24,2,2.5)`, `(10,24,2,2.5)`, `(20,24,2,2.5)`, `(28,24,2,2.5)`.
- For each meta-fit battery, exclude its learned embedding. The remaining seven embeddings define the mean and scale of the four decoder-probe responses.
- The probe responses are measured in the decoder's normalized target coordinates. Their per-probe population standard deviation is floored at `0.05`, matching the existing raw-q prior floor. The added loss is the candidate weight times the mean squared standardized deviation of the candidate's four probe responses from that leave-one-out population mean.
- Candidate functional-prior weights: `{0, 0.001, 0.01, 0.1, 1}`. Raw-q prior weight is fixed at zero.
- For every weight, infer q from the battery's earliest 30% cycles with the existing deterministic four starts, 200 fit steps, support-internal candidate selection, and 50-step refinement.
- The later 70% meta-fit cycles are used only for development scoring. They do not enter q optimization. Perturbing all later targets by +123.456 must leave every candidate q unchanged.
- Save all candidate q, four probe responses, capacity-cycle1, early-fade, support selection losses, later-cycle meta-query NRMSE, and leakage audit.

## Frozen Phase-A selection and stop rule

For each fixed weight, aggregate all 15 cells before any structure-validation run:

1. Integrity requires 15/15 finite cells, five weights per cell, eight q rows per weight, and maximum query-target perturbation difference exactly zero.
2. Prediction eligibility requires the candidate's 15-cell median meta-query NRMSE to be at most 1.05 times the weight-0 median.
3. Representation eligibility reuses the existing gate: continuity q-distance median Spearman is at least 0.80 in every split; capacity-cycle1 and early-fade each have median-of-split-medians at least 0.70 and minimum split median at least 0.50.
4. Among candidates satisfying both eligibility conditions, choose the lowest median meta-query NRMSE; ties choose the smaller weight.

If no weight is eligible, stop this exact functional-prior grid without touching structure-validation data. If one is selected, that choice alone authorizes writing a separately frozen Phase-B validation command. Phase B must retain the old prediction gate, functional-interface gate, stability gate, and query-leakage audit. Raw-q shift remains diagnostic because the proposed symbolic interface exposes decoder-functional coordinates, not raw q.

## Command and monitoring

Run only after a fresh host check finds a genuinely empty physical GPU:

```bash
CUDA_VISIBLE_DEVICES=<empty> python scripts/run_nasa_functional_response_prior_meta_20260827.py \
  --q-root runs/nasa_prefix_q_training_pilot_20260826 \
  --output-root runs/nasa_functional_response_prior_meta_20260827 \
  --device cuda:0

python scripts/analyze_nasa_functional_response_prior_meta_20260827.py \
  --root runs/nasa_functional_response_prior_meta_20260827
```

Monitor process-alive, cell count, append-only status, output growth, and GPU ownership every 30--60 seconds. There is no experiment deadline and no automatic retry. A nonzero cell stops the process and remains visible.

## Expected outputs

| Output | Path | Success criterion |
|---|---|---|
| Per-cell grid | `<root>/<dataset>/seed*/` | 15 cells, five weights and eight q rows/weight |
| Terminal status | `<root>/status.json` | 15/15 success, zero failed |
| Selection audit | `<root>/selected_weight.json` | selected weight or explicit no-eligible-candidate stop |
| Readable report | `<root>/FUNCTIONAL_RESPONSE_PRIOR_META_REPORT.md` | integrity, prediction and representation eligibility for every weight |
