# Frozen real-data provenance

This file distinguishes the immutable experimental inputs from mutable upstream
download endpoints. Raw files and per-article subsets remain governed by their
upstream licenses and citation requirements.

## Starrydata thermoelectric data

- Project citation: Katsura et al., *Starrydata: From Published Plots to Shared
  Materials Data* (2025), DOI `10.1080/27660400.2025.2506976`.
- Upstream release convention: the official
  `starrydata/starrydata_datasets` repository publishes daily releases named
  `data-YYYYMMDD` and a separately mutable `latest` release.
- Frozen release used for the temporal confirmation: official daily release
  `data-20260829`, downloaded 2026-08-29.
- Local source files:

| File | Bytes | SHA-256 |
|---|---:|---|
| `data/external/starrydata_latest_20260829/ThermoelectricMaterials_curves.csv.gz` | 25,265,374 | `b82fd98e8595b4c4712e3e21fe992320131826913bfd333c82011c921d9cb16a` |
| `data/external/starrydata_latest_20260829/ThermoelectricMaterials_samples.csv.gz` | 2,065,534 | `d4e7ee51027790399a484ef591d9cb354e15250735bbc29d53ebf20128da8673` |
| `data/external/starrydata_latest_20260829/ThermoelectricMaterials_papers.csv.gz` | 1,462,583 | `0d6fe6963f839b0bc10a6099c017c921752906a5072bb0bbfde0294ca3103e44` |
| `data/application_reviewer_clean/starry_te/raw/starrydata2_2025-06-01.zip` | 48,508,520 | `f4b8bf412bf62d0b6e9263ddfc4d423f4b1e0a2f85e2d0ea6f7798980b62ffde` |

The exact confirmation selection, pre-target seal, consumption receipt, and
derived-file hashes are under
`runs/starry_zt_temporal_confirmation_20260829/`. Each selected curve retains
its original paper DOI. The upstream `latest` URL is not a reproduction input.

## NIST ThermoML data

- Dataset citation: Riccardi et al., *ThermoML/Data Archive*, DOI
  `10.18434/mds2-2422`.
- Schema citation: Frenkel et al., *Pure and Applied Chemistry* 78, 541--612
  (2006), DOI `10.1351/pac200678030541`.
- Frozen archive: `ThermoML.v2020-09-30.tgz`, 189,433,115 bytes, SHA-256
  `231161b5e443dc1ae0e5da8429d86a88474cb722016e5b790817bb31c58d7ec2`.
- Local path: `data/external/thermoml_2020_archive/ThermoML.v2020-09-30.tgz`.

The exact vapor-pressure source JSON hashes are in
`runs/thermoml_single_use_confirmation_20260829/c947fbd6cc82bf8d880a1449f16f859ede8e05b58f6c3e11504cdf24d05c38c4/source_hashes.json`.
Crystal-Cp cohort and archive hashes are in
`runs/thermoml_crystal_cp_cohorts_20260829/sha256_manifest.json`.

## Numerical baselines and named form

- PCHIP uses `scipy.interpolate.PchipInterpolator`, whose derivative rule cites
  Fritsch and Butland (1984), DOI `10.1137/0905021`.
- FPCA follows the functional principal-components baseline described by Ramsay
  and Silverman (2005), DOI `10.1007/b98888`, with all interpolation, component,
  ridge, and train-only selection rules frozen in the experiment manifests.
- “Shomate5” denotes the NIST-documented heat-capacity basis
  `1, t, t^2, t^3, t^-2`; it is used as a named response family, not asserted as
  the true microscopic mechanism of every crystal curve.
