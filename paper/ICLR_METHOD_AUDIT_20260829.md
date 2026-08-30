# ICLR Method draft audit

## Mini-outline

1. Define unseen-entity support/query inference and separate the core and
   optional learned-prior branches.
2. Define probe-response equivalence and fixed named canonical coordinates.
3. State quotient invariance, approximate projection stability, and exact
   limitations.
4. Define support structure re-q and the support-rank/query-amplification
   diagnostics.
5. State stable response-metric Gauss--Newton and its affine-only calibration
   guarantee.
6. Define GIRD development cross-fitting, function-metric fusion, risk
   decomposition, and the rank-aware deployment rule.

## Module design table

| Module | Input → process → output | Why needed | Verifiable advantage |
|---|---|---|---|
| Canonical response map | decoder/state → fixed physical probes → weighted named-basis projection → `c` | raw q has gauge freedom | exact equality for response-equivalent representations; approximate bound via basis singular value |
| Support structure re-q | unseen support → named design → stable least squares → `c0` | provides an auditable raw-q-free endpoint | explicit rank, conditioning and query-amplification diagnostics |
| Stable decoder calibration | unseen support → QR/SVD GN + response-loss line search → probe response | ordinary optimization can break affine chart pairing numerically | paired affine response paths under stated rank/margin conditions |
| GIRD dictionary development | cross-fitted held-out responses → deterministic OMP/nested selection → frozen atoms/K/lambda/rule | prevents using a hand-selected equation or favorable prior weight | complete OMP path, margin and fold provenance |
| Rank-aware GIRD inference | `c_f`, support design → probe-Gram whitening → conditional ridge fusion → symbolic prediction | a decoder prior is only plausible when support misses directions | exact `lambda=0` fallback and visible endpoint comparisons |

## Pipeline figure sketch

```text
development entities
  ├─ entity/DOI cross-fit decoder ─ support-only stable GN ─ probe responses ┐
  └─ target curves ──────────────────────────────────────────────────────────┤
                                                                              ▼
                      deterministic symbolic dictionary + frozen rank rule
                                                                              │
unseen entity support ────────────────────────────────────────────────────────┤
  ├─ named support design ─ rank/conditioning ───────────────┐                │
  └─ frozen decoder ─ stable GN ─ canonical functional prior ├─ GIRD ─ equation
                                                             │
                         well-identified support ─ exact lambda=0 fallback
```

## Reverse outline

| Paragraph | Message | Evidence/proof obligation | Status |
|---|---|---|---|
| Overview | Both branches output the same named coefficients; raw q is not the scientific object | Paper contract and method registry | Supported |
| Canonical design | Probes/basis/weights/units define the coordinate | Frozen basis adapters and formulas | Supported |
| Quotient theorem | Equal probe responses imply equal coordinates | Fixed linear projection proof | Supported |
| Support estimator | Coefficient stability depends on support singular values and query amplification | Least-squares bound; ThermoML condition diagnostics | Supported |
| Stable calibration | Affine decoder charts can be calibrated along paired response paths | Stable gauge experiment and theorem assumptions | Supported within stated scope |
| GIRD motivation | Prior fusion is only justified for weak support directions | Singular-direction risk expression | Supported as conditional theory |
| Development algorithm | K/lambda/rank choices are nested and frozen | Formal cell/preparer/package contracts | Supported by code/contract; terminal result pending |
| GIRD objective | Fusion occurs in probe-function units | Gram-whitened objective and endpoints | Supported |
| Deployment rule | Good support falls back exactly; both-bad cases remain visible | Frozen analyzer and result-to-narrative contract | Supported by implementation tests; real gain pending |

## Claim--evidence map

- Claim: canonical coordinates are invariant to any representation with the
  same declared probe response. | Evidence: Proposition 1A and fixed-projection
  proof. | Status: **supported**, probe/basis relative.
- Claim: response-space projection is stable under approximate response error.
  | Evidence: pseudoinverse singular-value bound. | Status: **supported**.
- Claim: stable GN is affine-equivariant under full-rank and fixed-line-search
  assumptions. | Evidence: paired-iterate proof and independent maximum response
  difference `3.6567e-10`. | Status: **supported**, not nonlinear equivariance.
- Claim: exact response-equivalent affine counterfactuals preserve the canonical
  coordinate. | Evidence: 375 exact interventions, maximum prediction and
  coefficient differences `4.50e-15/2.04e-14`, while raw coordinates change by
  as much as `4.996`. | Status: **supported**; distinct from recalibration.
- Claim: support re-q identifies every named coefficient. | Evidence: none under
  rank deficiency or near collinearity. | Status: **not claimed**.
- Claim: finite GIRD can lower risk in weak support directions. | Evidence:
  directional bias--variance formula and controlled relaxation result. | Status:
  **consistent with a conditional mechanism**, not a universal empirical claim;
  zero-mean independent errors are a modeling assumption.
- Claim: GIRD improves real Crystal-Cp prediction beyond direct support fitting.
  | Evidence: formal matrix and temporal confirmation not terminal. | Status:
  **needs evidence**; prohibited from headline wording.

## Self-review

- **Clarity:** every subsection defines its input, operation, output, motivation,
  and measurable diagnostics.
- **Flow:** invariant object → direct support estimator → numerical calibration
  → optional prior is a single dependency chain.
- **Terminology:** `canonical response coordinate`, `structure re-q`, `functional
  prior`, and `GIRD` have one meaning each.
- **Unsupported claims:** no true-q recovery, universal identifiability,
  nonlinear optimizer equivariance, automatic dictionary novelty, or guaranteed
  prior gain appears.
- **Missing evidence:** the 25-cell aggregate and temporal learned-prior result
  still decide whether GIRD is a headline method or a diagnostic.
- **Page risk:** the full draft is intentionally explicit; proof details and the
  directional risk derivation can move to the appendix if the combined Method
  exceeds the planned 2.70 pages.
