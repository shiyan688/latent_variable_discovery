# NASA protocol-matched functional prior Phase-B report

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-28
- Verification Status: ANALYZED
- Version Label: nasa_protocol_matched_functional_prior_phaseb_v1

**Stage-C2 decision:** STOP

| gate | result |
|---|---|
| integrity | PASS |
| prediction_retention | FAIL |
| functional_stability | FAIL |
| scientific_alignment | PASS |

## Per-dataset prediction

| dataset | weight 0 | weight 0.01 | paired ratio | wins/5 |
|---|---|---|---|---|
| nasa_battery_capacity_reviewer_clean_inner0 | 1.387 | 1.459 | 1.037 | 1/5 |
| nasa_battery_capacity_reviewer_clean_inner1 | 1.475 | 1.474 | 0.999 | 3/5 |
| nasa_battery_capacity_reviewer_clean_inner2 | 1.154 | 1.153 | 0.999 | 3/5 |

Overall median-NRMSE ratio: 1.0516; cells within 10%: 14/15; paired Wilcoxon p=0.4543.

## Functional endpoints

Capacity stability median/min split: 1.000/0.900.
Early-fade stability median/min split: 0.700/0.200.
Empirical alignment capacity/early fade: 1.000/0.500.

This protocol did not reselect the weight on structure-validation outcomes. These batteries are protocol-held-out but not globally untouched by the broader project history.
