# NASA functional-response prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-27
- Verification Status: ANALYZED
- Version Label: nasa_functional_response_prior_meta_v1

**Phase-B decision:** STOP

本屏只使用八个 meta-fit batteries；structure-validation 数据未被读取。候选必须同时保留 later-cycle meta-query prediction 并通过既有 representation gate。

| weight | meta-query | pred | q 中位/最差 | capacity 中位/最差 | fade 中位/最差 | repr | eligible |
|---|---|---|---|---|---|---|---|
| 0 | 0.07235 | PASS | 0.685/0.401 | 0.631/0.560 | 0.690/-0.024 | FAIL | NO |
| 0.001 | 0.08917 | FAIL | 0.628/0.380 | 0.595/0.440 | 0.738/-0.071 | FAIL | NO |
| 0.01 | 0.1142 | FAIL | 0.529/0.207 | 0.690/0.548 | 0.571/0.107 | FAIL | NO |
| 0.1 | 0.1774 | FAIL | 0.432/0.077 | 0.595/0.393 | 0.286/-0.071 | FAIL | NO |
| 1 | 0.2341 | FAIL | 0.343/-0.019 | 0.393/-0.024 | 0.048/0.036 | FAIL | NO |

selected weight: none
