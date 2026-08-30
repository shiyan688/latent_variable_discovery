# ICLR introduction draft audit

## Mini-outline

1. Define support-conditioned scientific curve interpretation and why prediction
   alone is insufficient.
2. Expose latent gauge ambiguity as the technical obstacle.
3. Show why existing equation discovery still needs a safe representation
   interface.
4. Introduce fixed physical probe responses, named canonical coordinates,
   support-only calibration, and conditional GIRD.
5. State the pragmatic expression endpoint and the two confirmed real results,
   with the strongest baseline caveat adjacent.
6. List three contributions whose wording survives Branch A or Branch B.

## Reverse outline

| Paragraph | Topic sentence/message | Evidence or explanation | Maps to thesis? |
|---|---|---|---|
| P1 opening | Scientific curves need both sparse-support prediction and a nameable entity description | ZT, vapor pressure and Cp examples; support/query definition | Yes: defines the paper's task |
| P2 challenge | Learned task vectors are chart coordinates, not automatically scientific variables | Exact affine intervention motivation; identifiability and canonicalization prior work | Yes: establishes the core technical problem |
| P3 gap | Equation discovery assumes or learns coordinates but does not settle which unseen-entity coordinate is safe to interpret | SINDy-AE, neural-symbolic extraction, parametric equations, UPINN | Yes: locates the missing interface |
| P4 method | Fixed physical probe response is the invariant object | Quotient projection, stable support calibration, structure re-q, conditional GIRD | Yes: states the complete mechanism |
| P5 evidence | Interpretation is a falsifiable expression-fidelity endpoint, not SOTA prediction | ZT `0.9888`, vapor pressure `0.9996`, PCHIP/kNN caveats, stage hypotheses | Yes: proves practical value without overclaiming |
| P6 contributions | Theory, auditable inference, and sealed real evaluation are the contributions | Existing theory/experiments and frozen Branch A/B rule | Yes: summarizes the paper contract |

## Claim--evidence map

| Claim in draft | Evidence | Status |
|---|---|---|
| Raw q can change under prediction-equivalent affine charts | Exact chart intervention and stable gauge benchmark | Supported |
| Canonical coefficients depend only on declared probe response | Quotient-invariance proposition | Supported, basis/probe relative |
| Stable support calibration is affine-equivariant under stated conditions | Independent stable-GN response difference about `3.66e-10` | Supported, not arbitrary nonlinear equivariance |
| ZT temporal expression reaches `0.9888` | `runs/starry_zt_temporal_confirmation_20260829/evaluation/decision.json` | Supported |
| Vapor-pressure temporal expression reaches `0.9996` | single-use ThermoML analysis decision | Supported |
| Named coordinates are support-offset stable | ZT and vapor-pressure stability CSVs; stage-hypothesis audit | Supported descriptively |
| PCHIP can be better and ZT paired advantage is nonsignificant | confirmation baseline/paired analyses | Supported |
| Learned decoder prior adds value on real Crystal-Cp | active 25-cell and future temporal decisions | Needs evidence; not claimed as a result |
| Crystal-Cp temporal expression transfers | unopened single-use evaluation | Needs evidence; explicitly pending |

## Self-review

- **Clarity:** each paragraph has one first-sentence message; support, query,
  response coordinate, and raw latent chart are defined before reuse.
- **Flow:** task → gauge challenge → equation-discovery gap → method → evidence
  → contributions is monotone; GIRD is not introduced before the core branch.
- **Terminology:** use `raw q`, `probe response`, `canonical response coordinate`,
  `support structure re-q`, and `GIRD` consistently.
- **Unsupported claims:** no priority, causal, unique-law, universal bridge, or
  prediction-SOTA claim appears. Crystal-Cp and real GIRD gains remain pending.
- **Missing evidence:** final Branch A/B wording, citations/BibTeX audit, actual
  Crystal result, and final page fit remain open.

## Citation gate

The citation keys in the TeX draft are placeholders until each BibTeX entry is
verified against the primary proceeding or publisher record. The prose may be
reviewed now, but the section is not citation-complete and must not be treated as
submission-ready.

