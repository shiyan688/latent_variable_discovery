# NASA information-matched prefix-q training pilot

**冻结判定：** DO NOT ADVANCE

## 方法

训练时 q 每个 batch 只由各实体最早 30% 行更新，随后 decoder 用完整 batch 更新；测试 q 仍只由前 30% support 校准。continuity 的训练实体响应距离同样只由前缀计算。

## 预测与接口

| 方法 | NRMSE | 旧基线 | ratio | 保持 cells | raw max|z| 中位/最大 | functional max|z| 中位/最大 | 反传/ cell |
|---|---|---|---|---|---|---|---|
| prefix_q_continuity_step1 | 1.387 | 1.193 | 0.9684 | 10/15 | 8.815 / 19.44 | 3.389 / 12.84 | 6000 |
| prefix_q_mse_step1 | 1.109 | 0.9175 | 1.104 | 7/15 | 6.378 / 15.02 | 3.841 / 6.854 | 6000 |

## Continuity 表征稳定性

| 对象 | split 中位的中位 | 最差 split 中位 |
|---|---|---|
| q distance | 0.98 | 0.945 |
| capacity_cycle1 | 0.7262 | 0.6905 |
| early_fade_rate | 0.75 | -0.119 |

## Gate

| Gate | 结果 |
|---|---|
| gate_1_integrity | PASS |
| gate_2_prediction_retention | FAIL |
| gate_3_interface_safety | FAIL |
| gate_4_representation_stability | FAIL |
| advance_to_bounded_symbolic_stage_c2 | FAIL |

本轮使用已参与方法开发的 inner splits，只能决定是否值得进入下一开发阶段，不能作为独立确认性证据。
