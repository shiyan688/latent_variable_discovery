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
| prediction_retention | PASS |
| functional_stability | FAIL |
| scientific_alignment | PASS |

## Per-dataset prediction

| dataset | weight 0 | weight 0.01 | paired ratio | wins/5 |
|---|---|---|---|---|
| nasa_battery_capacity_reviewer_clean_inner0 | 1.387 | 1.388 | 0.998 | 3/5 |
| nasa_battery_capacity_reviewer_clean_inner1 | 1.475 | 1.475 | 1.000 | 2/5 |
| nasa_battery_capacity_reviewer_clean_inner2 | 1.154 | 1.153 | 1.000 | 2/5 |

Overall median-NRMSE ratio: 1.0009; cells within 10%: 15/15; paired Wilcoxon p=0.7615.

## Functional endpoints

Capacity stability median/min split: 1.000/1.000.
Early-fade stability median/min split: 0.850/0.200.
Empirical alignment capacity/early fade: 1.000/0.500.

This protocol did not reselect the weight on structure-validation outcomes. These batteries are protocol-held-out but not globally untouched by the broader project history.
