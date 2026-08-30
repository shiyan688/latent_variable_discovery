# Citation Audit Report

| Field | Value |
|---|---|
| Date | 2026-08-29 |
| Audited manuscript | `iclr_draft.tex` |
| Bib file | `iclr_refs.bib` |
| Audited cited entries | 21 |
| Assurance | same-family provisional (`gpt-5.6-sol`, `xhigh`, fresh reviewer per entry) |

## Summary

| Verdict | Count |
|---|---:|
| KEEP | 21 |
| FIX | 0 |
| REPLACE | 0 |
| REMOVE | 0 |

**Overall verdict: PASS.** Every cited work was found in an official publisher,
proceedings, or PMLR record; the current bibliography metadata agrees with those
records; and every current citation context is supported at the scope stated.
The current 21 cite keys and the 21 entries in `iclr_refs.bib` are identical.

## Scope corrections completed during the audit

- The identifiability claim was narrowed from arbitrary latent regularization to
  low dimensionality and common regularized disentanglement objectives still
  requiring assumptions or inductive bias.
- Functa prior learning, instance adaptation, and partial-observation MAP
  completion are now stated separately.
- Nonlinear ICA is described as identifying sources only up to specified
  equivalence classes under explicit assumptions.
- UPINN is described as symbolically recovering neural representations of
  unknown differential-equation terms, not generic operators.
- Parametric symbolic models are scoped to parameterized instances, and LaSR's
  LLM-induced concept library is stated concretely.

These were sentence-scope corrections. The final re-audit verdict for every
entry is KEEP; no citation replacement or deletion remains.

## Dataset and numerical-source audit

- The Starrydata project paper, ThermoML data archive, and IUPAC ThermoML
  standard were verified against official publisher, NIST, and IUPAC records.
- Fritsch--Butland was verified both at SIAM and as SciPy's algorithm reference
  for the implemented `PchipInterpolator`; Ramsay--Silverman was verified at
  Springer as an FPCA source.
- The first NIST WebBook review found metadata drift only: 2025 is the data
  update year, not a canonical publication year. The entry now uses NIST's
  editor citation model, `n.d.`, the update/access note, and the stable DOI. A
  fresh post-fix reviewer returned KEEP. The official solid-phase page gives
  exactly the cited five-term Shomate heat-capacity form.

## All-clean entries

- `champion2019data`
- `cranmer2020discovering`
- `dupont2022functa`
- `dym2024equivariant`
- `garnelo2018conditional`
- `gondal2021function`
- `grayeli2024lasr`
- `frenkel2006thermoml`
- `fritsch1984pchip`
- `katsura2025starrydata`
- `khemakhem2020variational`
- `locatello2019challenging`
- `ma2024canonicalization`
- `nist2025webbook`
- `podina2023universal`
- `ramsay2005functional`
- `riccardi2021thermoml`
- `sitzmann2020metasdf`
- `syrota2025metric`
- `xu2020metafun`
- `zhang2022parametric`

## Compile verification

`iclr_draft.tex` was compiled with BibTeX and settling LaTeX passes. The current
result is a 10-page generic two-column draft including the audited abstract and
Figures 1--3; main text through the conclusion ends on page 8 and references
begin later on page 8 and continue through page 10. It has no undefined citation, undefined reference, fatal
LaTeX error, BibTeX warning, or overfull box. The theorem-scope and canonical
coefficient-notation refresh changed no cite key or cited claim; only the NIST
context line anchor moved. The original-context ledger remains SHA-256
`cb87e3c57862114ae295404182d7d6c2f71c40628e84b1c5dc66af9bac6ea454`.
This is a compilation check, not yet the final ICLR style build.

Original per-entry traces are stored at
`.aris/traces/citation-audit/2026-08-29_run01/`; the six dataset/numerical
reviews and the NIST post-fix review are under `2026-08-29_run05/`, with the
aggregate index under `2026-08-29_run06/`.
