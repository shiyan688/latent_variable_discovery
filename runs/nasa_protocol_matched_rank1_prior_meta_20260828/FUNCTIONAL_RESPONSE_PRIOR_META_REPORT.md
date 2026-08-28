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
| 0.001 | 0.07083 | PASS | 0.662/0.388 | 0.994/0.940 | 1.000/1.000 | 0.821/0.786 | PASS | YES |
| 0.01 | 0.07041 | PASS | 0.685/0.472 | 0.991/0.940 | 1.000/1.000 | 0.857/0.821 | PASS | YES |
| 0.1 | 0.08679 | FAIL | 0.577/0.560 | 0.996/0.963 | 1.000/1.000 | 0.893/0.881 | PASS | NO |
| 1 | 0.08084 | FAIL | 0.555/0.371 | 0.996/0.963 | 1.000/0.976 | 0.905/0.810 | PASS | NO |

selected weight: 0.01
