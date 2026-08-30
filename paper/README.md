# Paper workspace

## Current target

The active submission target is ICLR. The current scientific story is
gauge-aware response-space canonicalization for interpretable few-shot
scientific curves.

Authoritative planning and claim files are at the repository root:

- `PAPER_PLAN.md`
- `ICLR_CANONICAL_RESPONSE_CLAIM_GATE_20260829.md`
- `ICLR_RESULT_TO_NARRATIVE_DECISION_RULE_20260829.md`
- `ICLR_TOP50_COMPLETION_AUDIT_20260829.md`
- `SYMBOLIC_STAGEWISE_HYPOTHESES_20260829.md`

Current ICLR prose:

- `sections/abstract.tex`: branch-neutral 183-word abstract integrated in the
  current wrapper;
- `iclr_draft.tex`: current generic two-column integration wrapper; this is the
  active ICLR story, while `main.tex` remains legacy;
- `iclr2027_draft.tex`: official ICLR 2027 submission-style wrapper.  The
  unmodified official style assets are under `iclr2027/`; keep
  `\iclrfinalcopy` disabled for double-blind submission;
- `ANONYMOUS_SUPPLEMENT_README.md` and `anonymous_supplement_files.txt`: the
  scope and explicit allowlist for the isolated reproducibility package;
- `sections/introduction.tex`: branch-neutral Introduction draft;
- `sections/method.tex`: canonical-response and rank-aware GIRD Method draft;
- `sections/experiments.tex`: confirmed evidence, strong baselines, negative
  results, and frozen Crystal-Cp placeholder;
- `sections/related_work.tex`: bounded positioning against function
  representations, identifiability/canonicalization, and equation discovery;
- `sections/limitations_conclusion.tex`: branch-safe Limitations and Conclusion;
- `sections/ai_reproducibility_statements.tex`: mandatory ICLR 2027 AI-use
  disclosure and the recommended reproducibility statement;
- `ICLR_INTRODUCTION_AUDIT_20260829.md`: mini-outline, reverse outline,
  claim--evidence map, self-review, and citation gate.
- `ICLR_METHOD_AUDIT_20260829.md`: module table, pipeline sketch, reverse
  outline, claim--evidence map, and method self-review.
- `ICLR_EXPERIMENTS_AUDIT_20260829.md`: experimental reverse outline,
  claim--evidence audit, fairness checks, and missing-evidence gate.
- `ICLR_RELATED_WORK_AUDIT_20260829.md`: closest-work reverse outline,
  novelty-boundary checks, and citation-completion gate.
- `ICLR_CONCLUSION_AUDIT_20260829.md`: conclusion mini-outline, reverse outline,
  claim--evidence map, and branch-reconciliation gate.
- `DATA_PROVENANCE.md`: frozen Starrydata and ThermoML source paths, dated
  snapshot identities, file sizes, SHA-256 values, and numerical-method source
  notes; mutable upstream `latest` links are explicitly not reproduction inputs.
- `figures/figure1_variant_a.svg`: preferred full-width horizontal hero
  schematic; pure editable vector, branch-neutral.
- `figures/figure1_variant_a.pdf`: deterministic paper-ready vector rendering
  of the preferred hero schematic;
- `figures/figure1_variant_b.svg`: alternative 2-by-2 warm-layout hero
  schematic for presentations or a taller paper layout.
- `figures/figure1_values.json`: exact source/output hashes and the claim
  boundary for the two sealed temporal values shown in Figure 1;
- `figures/figure2_gauge_gird.pdf`: deterministic vector plot of affine-gauge
  response preservation and family-dependent four-support GIRD behavior, with
  paired entity-bootstrap intervals and an independently corrected caption;
- `figures/figure2_values.json`: source hashes and exact values for Figure 2;
- `figures/figure3_real_transfer.pdf`: deterministic four-panel real-transfer
  figure with representative support/query curves, paired entity comparisons,
  and development coefficient stability;
- `figures/figure3_values.json`: Figure 3 source hashes, deterministic
  representative-selection rule, exact plotted values, and output hashes;
- `CITATION_AUDIT.md` / `CITATION_AUDIT.json`: terminal same-family provisional
  PASS for all 21 current cited entries, including the six dataset/numerical
  sources added on 2026-08-29, with per-entry official-source traces under
  `.aris/`;
- `CITATION_AUDIT_FRESHNESS_20260830.{md,json}`: current official-wrapper hashes
  and a fresh check that the same 21 cite keys retain unchanged semantic scope;
- `CITATION_AUDIT.html`: reader view of the audit; its independent render-only
  review sidecar is PASS with zero warnings;
- `iclr_refs.bib`: current ICLR-only bibliography; it is not the legacy
  `refs.bib`.

The current draft's citation audit is terminal same-family provisional PASS. `iclr_draft.tex` now
integrates the branch-neutral abstract and Figures 1--3 and compiles with BibTeX
to ten generic two-column pages: the main paper, including the conclusion, ends
on page 8 and references begin later on page 8 and continue through page 10. There are no undefined citations
or references, LaTeX errors, overfull boxes, or BibTeX warnings. Figure 3 is
placed before the Conclusion and references rather than deferred after the
bibliography. The wrapper refresh introduced no cite key or citation-context
change. The currently used dataset, PCHIP, FPCA, and Shomate sources are now
added and audited; citations for any later-added Gauss--Newton, QR/SVD, OMP,
DeepONet, or appendix material must still be verified before submission.

## Legacy draft

`main.tex` and the AAAI-27 template files are a historical draft from an older
research story. They are retained only to preserve worktree history and are not
the current manuscript. Numbers, claims, datasets, and bibliography entries in
that draft must not be copied into the ICLR paper without a raw-evidence audit.
The user-excluded dataset material has been removed from the legacy manuscript
and bibliography.

The legacy file still compiles with:

```bash
latexmk -pdf main.tex
```

Compilation was rechecked on 2026-08-29. The generic `iclr_draft.tex` wrapper is
retained as a footprint check.  The current submission target is
`iclr2027_draft.tex`, built with the unmodified official ICLR 2027 style: main
text occupies pages 1--9, while the required AI-use statement, recommended
reproducibility statement, and references begin on page 10 and do not count
toward the submission limit.  The settled build has no undefined citations or
references, overfull boxes, BibTeX warnings, or label drift.

From any working directory, reproduce the official build with:

```bash
bash scripts/build_iclr2027_paper.sh
```

The script resolves the repository root from its own location and places all
TeX variable/font caches below `runs/_runtime_cache/`; it contains no
machine-local absolute path.

The anonymous package is built with:

```bash
bash scripts/build_iclr2027_anonymous_supplement.sh
```

The verified 2026-08-30 package is under
`public/iclr2027_anonymous_supplement_20260830/`.  It contains 101 indexed files,
passes `sha256sum -c SHA256SUMS`, and rebuilds a byte-identical official-template
PDF from its own root.  Its scope is manuscript compilation plus compact
claim-level evidence; third-party source archives and training checkpoints are
identified by provenance and hashes rather than redistributed.

## Frozen writing boundary

Do not promote GIRD to the title, abstract, or headline contribution until both
the formal development and single-use temporal learned-prior gates pass. If
they fail, use Branch B and present GIRD only as a controlled diagnostic. Never
describe the confirmed support structure re-q expressions as recovery of the
original or true raw q.
