# NASA meta-selected soft q-prior diagnostic

**冻结判定：** DO NOT ADVANCE

prior weight 只由八个 meta-fit 实体各自前 30% support 内部的 selection loss 选择；后续目标和 structure-validation query 不参与选择。

## 预测与接口

| 方法 | selected | prefix λ=0 | 旧方法 | 对旧 ratio | 保持 | raw z 中位/最大 | functional z 中位/最大 |
|---|---|---|---|---|---|---|---|
| prefix_q_continuity_step1 | 1.356 | 1.387 | 1.193 | 0.9786 | 10/15 | 8.188 / 20.66 | 3.273 / 9.134 |
| prefix_q_mse_step1 | 1.08 | 1.109 | 0.9175 | 1.035 | 8/15 | 5.051 / 12.12 | 3.378 / 5.35 |

## Continuity selected meta-fit 稳定性

| 对象 | split 中位的中位 | 最差 split 中位 |
|---|---|---|
| q distance | 0.7581 | 0.4614 |
| capacity_cycle1 | 0.6548 | 0.6548 |
| early_fade_rate | 0.6905 | 0.119 |

## 选择的 prior weight

| 方法 | weight | cells |
|---|---|---|
| prefix_q_continuity_step1 | 0 | 12 |
| prefix_q_continuity_step1 | 0.001 | 3 |
| prefix_q_mse_step1 | 0 | 5 |
| prefix_q_mse_step1 | 0.001 | 9 |
| prefix_q_mse_step1 | 0.01 | 1 |

## Gate

| Gate | 结果 |
|---|---|
| gate_1_integrity | PASS |
| gate_2_prediction_retention | FAIL |
| gate_3_interface_safety | FAIL |
| gate_4_representation_stability | FAIL |
| advance_to_bounded_symbolic_stage_c2 | FAIL |

本轮仍是已暴露 inner splits 上的开发诊断；即使通过也只允许独立冻结 bounded symbolic Stage C2。
