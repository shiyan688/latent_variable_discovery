# ICLR Experiments draft audit

## Mini-outline

1. Define four research questions, the common unseen-entity protocol, metrics,
   leakage test, information paths, and strong baseline families.
2. Verify the gauge claim and the conditional learned-prior mechanism on
   controlled response families.
3. Lead with the two completed one-shot temporal symbolic confirmations.
4. Report coefficient stability, falsifiable stage meaning, uncertainty, tails,
   and adjacent strongest baselines for ZT and vapor pressure.
5. Separate raw q, decoder-functional projection, and support structure re-q.
6. Present Crystal-Cp as a difficult development stress test with all baselines
   and reserve a frozen placeholder for its one-shot result.

## Reverse outline

| Block | One message | Evidence | Status |
|---|---|---|---|
| Protocol | Expression fidelity, neural-prior value, and prediction SOTA are distinct tests | frozen contracts and analyzers | Supported |
| Gauge intervention | Stable response-metric calibration preserves paired affine response paths | controlled stable-extension artifacts | Supported |
| Controlled GIRD | A finite prior can help when support is weak, but not universally | relaxation four-support gain and negative families/FPCA | Supported conditionally |
| Temporal Table 1 | Two compact support-only expressions transfer to future entities | ZT and vapor-pressure single-use decisions | Supported |
| ZT paragraph | Reference/slope/curvature are stable enough to motivate a turnover test | temporal decision and offset-stability CSV | Supported; no SOTA claim |
| Vapor paragraph | Reference pressure/enthalpy/correction coordinates transfer with a visible collinearity limit | temporal decision and stability analysis | Supported; q2 interpretation limited |
| Neural ablation | Decoder response is much more readable than direct raw q, while the stronger bridge can still fail | neural bridge decisions | Supported as diagnostic |
| Crystal Table 2 | The stage expression beats global learned baselines but loses to interpolation | v4 and baseline raw results | Development-supported |
| Crystal placeholder | Future wording is selected once by the frozen branch rule | unopened confirmation | Needs evidence; no number written |

## Claim--evidence map

- Claim: stable calibration is numerically affine-equivariant in the tested
  scope. | Evidence: maximum response/coordinate differences
  `3.66e-10/6.23e-9`, identical line searches. | Status: **supported**.
- Claim: finite GIRD universally improves prediction. | Evidence: contrary
  controlled families and FPCA result. | Status: **not claimed**.
- Claim: ZT expression transfers. | Evidence: 30 future entities, 919 queries,
  physical `R²=0.988810`, bootstrap lower bound `0.973306`. | Status:
  **supported**.
- Claim: vapor-pressure expression transfers. | Evidence: 84 future compounds,
  45 DOI, physical `R²=0.999581`, entity/DOI lower bounds
  `0.998880/0.998562`. | Status: **supported**.
- Claim: the expressions are prediction SOTA. | Evidence: PCHIP and paired kNN
  comparisons contradict this. | Status: **explicitly rejected**.
- Claim: decoder-functional coordinates repair raw-q readability. | Evidence:
  vapor external `0.990390` versus raw-q ridge `0.304492`; stronger geometry
  gate fails. | Status: **supported only as diagnostic readability**.
- Claim: Crystal-Cp expression and GIRD transfer temporally. | Evidence:
  confirmation unopened. | Status: **needs evidence**.

## Self-review

- **Clarity:** every result states which information path produced it.
- **Flow:** protocol → controlled mechanism → confirmed external expressions →
  neural ablation → pending third-domain stress test.
- **Fairness:** strongest interpolation, functional, learned support, and no-q
  baselines are visible; prediction superiority is not inferred.
- **Completeness:** pooled metrics appear with uncertainty, entity pass fraction,
  tails, negatives, stability, and leakage requirements where available.
- **Negative evidence:** failed geometry, nonsignificant paired ZT comparison,
  FPCA superiority in controlled GIRD, Crystal negatives, and bootstrap lower
  bounds are retained.
- **Missing evidence:** exact compute/update accounting belongs in the final
  table or appendix; Crystal neural/GIRD and temporal cells remain pending.
- **Page/format risk:** the section uses exactly two tables, no vertical rules,
  consistent metric direction, and adjacent baseline caveats. Figure references
  will be added only after sealed plots exist.

