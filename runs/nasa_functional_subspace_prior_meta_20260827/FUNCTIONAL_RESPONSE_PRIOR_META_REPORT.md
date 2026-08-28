# NASA functional-response prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-27
- Verification Status: ANALYZED
- Version Label: nasa_functional_response_prior_meta_v1_functional-response

**Phase-B decision:** STOP

本屏只使用八个 meta-fit batteries；structure-validation 数据未被读取。候选必须同时保留 later-cycle meta-query prediction 并通过预先声明的 representation gate。

| weight | meta-query | pred | q 中位/最差 | response 中位/最差 | capacity 中位/最差 | fade 中位/最差 | repr | eligible |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.07235 | PASS | 0.685/0.401 | 0.710/0.680 | 0.631/0.560 | 0.690/-0.024 | FAIL | NO |
| 0.001 | 0.07254 | PASS | 0.664/0.394 | 0.776/0.680 | 0.631/0.560 | 0.690/-0.024 | FAIL | NO |
| 0.01 | 0.07193 | PASS | 0.693/0.343 | 0.781/0.662 | 0.548/0.536 | 0.655/0.190 | FAIL | NO |
| 0.1 | 0.07941 | FAIL | 0.589/0.331 | 0.734/0.679 | 0.655/0.643 | 0.655/0.274 | FAIL | NO |
| 1 | 0.07052 | PASS | 0.771/0.573 | 0.789/0.631 | 0.631/0.619 | 0.655/0.143 | FAIL | NO |

selected weight: none
