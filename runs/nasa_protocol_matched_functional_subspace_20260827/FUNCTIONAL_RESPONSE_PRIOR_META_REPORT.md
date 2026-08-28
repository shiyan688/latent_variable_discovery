# NASA functional-response prior meta-only screen

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-27
- Verification Status: ANALYZED
- Version Label: nasa_functional_response_prior_meta_v1_functional-response

**Phase-B decision:** AUTHORIZE

本屏只使用八个 meta-fit batteries；structure-validation 数据未被读取。候选必须同时保留 later-cycle meta-query prediction 并通过预先声明的 representation gate。

| weight | meta-query | pred | q 中位/最差 | response 中位/最差 | capacity 中位/最差 | fade 中位/最差 | repr | eligible |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.07235 | PASS | 0.685/0.401 | 0.994/0.935 | 1.000/1.000 | 0.869/0.774 | PASS | YES |
| 0.001 | 0.07235 | PASS | 0.673/0.375 | 0.994/0.933 | 1.000/1.000 | 0.869/0.774 | PASS | YES |
| 0.01 | 0.07136 | PASS | 0.669/0.394 | 0.994/0.936 | 1.000/1.000 | 0.821/0.786 | PASS | YES |
| 0.1 | 0.07706 | FAIL | 0.562/0.455 | 0.992/0.941 | 1.000/1.000 | 0.869/0.810 | PASS | NO |
| 1 | 0.07295 | PASS | 0.536/0.371 | 0.992/0.961 | 1.000/0.976 | 0.857/0.821 | PASS | YES |

selected weight: 0.01
