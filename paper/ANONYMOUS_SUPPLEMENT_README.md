# Anonymous reproducibility supplement

This package supports the paper *Canonical Response Coordinates for
Interpretable Few-Shot Scientific Curves*.  It contains the official-template
manuscript sources, figure inputs, the code paths used for the displayed
experiments, and compact claim-level evidence.  It deliberately excludes Git
metadata, machine-local paths, caches, model checkpoints, and the unopened
Crystal confirmation responses.

## Fast verification

From the package root:

```bash
bash scripts/build_iclr2027_paper.sh
sha256sum -c SHA256SUMS
```

The verified build has 13 pages.  Main text ends on page 9; page 10 begins with
the AI-use statement.  The build requires `pdflatex`, `bibtex`, `pdftotext`, and
standard ICLR style dependencies.

## Evidence index

The evidence is organized by paper claim rather than by experiment date:

- Gauge invariance and affine recalibration:
  `runs/gauge_*_20260829/analysis/`.
- Controlled GIRD boundary study:
  `runs/gird_controlled_discovery_20260829/analysis/`.
- Starry ZT development and temporal transfer:
  `runs/starry_zt_interpretable_req_20260829/` and
  `runs/starry_zt_temporal_confirmation_20260829/evaluation/`.
- Vapor structure selection, selection audit, temporal transfer, and coordinate
  stability: `runs/thermoml_vapor_pressure_structure_*_202608*/`,
  `runs/thermoml_single_use_confirmation_20260829/`, and
  `runs/thermoml_q_stability_development_20260829/`.
- Crystal-Cp development stress test and baselines:
  `runs/thermoml_crystal_cp_*_development_20260829/analysis/` plus the compact
  router result, entity, and fold files.  No Crystal confirmation result exists
  in this package.
- Exact figure source hashes and plotted values:
  `paper/figures/figure{1,2,3}_values.json`.

`paper/DATA_PROVENANCE.md` identifies non-redistributed third-party snapshots by
source and SHA-256.  Re-running data extraction requires obtaining those source
archives; the compact evidence bundled here is sufficient to audit the paper's
reported aggregates and rebuild the manuscript, but is not a redistribution of
the full 32 GB local data cache or training checkpoints.

## Structure-selection boundary

The ZT quadratic endpoint was fixed a priori.  Vapor alone selected among three
predeclared one-term corrections in DOI-disjoint development folds.  Development
queries score that finite choice; after sealing, external query targets only
score the structure and external support targets alone estimate coefficients.
The exact per-fold reference temperatures and candidate scores are in
`runs/thermoml_vapor_pressure_structure_selection_audit_20260830/`.

