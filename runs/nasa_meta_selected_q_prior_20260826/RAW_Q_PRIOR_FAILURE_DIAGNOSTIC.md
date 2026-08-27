# NASA raw-q prior failure diagnostic

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-27
- Verification Status: ANALYZED
- Version Label: nasa_raw_q_prior_failure_v1

这是 formal gate 之后的事后机制诊断，不改变原 gate，也不使用 structure-validation query 来选择新权重。meta-query oracle 只使用 meta-fit 实体后 70% 目标，属于不可部署的诊断参照。

## Support 选择代理

| 方法 | 匹配 meta-query oracle | rank 相关中位 | selected meta-query | oracle meta-query |
|---|---|---|---|---|
| prefix_q_continuity_step1 | 6/15 | 0.900 | 0.07235 | 0.06266 |
| prefix_q_mse_step1 | 6/15 | 0.700 | 0.0825 | 0.07328 |

## Continuity 固定权重的 representation gate

| weight | q 中位/最差 split | capacity 中位/最差 | early fade 中位/最差 | 结果 |
|---|---|---|---|---|
| 0 | 0.685 / 0.401 | 0.631 / 0.560 | 0.690 / -0.024 | FAIL |
| 0.001 | 0.777 / 0.614 | 0.702 / 0.512 | 0.714 / 0.274 | FAIL |
| 0.01 | 0.808 / 0.792 | 0.667 / 0.643 | 0.560 / -0.095 | FAIL |
| 0.1 | 0.685 / 0.680 | 0.571 / 0.548 | 0.179 / 0.143 | FAIL |
| 1 | 0.578 / 0.562 | 0.726 / 0.536 | -0.071 / -0.321 | FAIL |

## 解释

support 内部 prediction loss 与 meta-query NRMSE 的排序总体同向，但它偏好最弱或零 raw-q 正则；正式 continuity 因而在 12/15 cells 选择 weight 0。更重要的是，五个固定权重没有一个能通过既有 representation gate，因此无需再用 structure-validation query 补跑固定权重。raw-q Gaussian prior 仍以每个 seed 自己的 embedding 坐标为参照，不能消除 q/第一层的 affine gauge；下一项方法应把约束定义在 decoder response/functional space，而不是继续调 raw-q 权重。
