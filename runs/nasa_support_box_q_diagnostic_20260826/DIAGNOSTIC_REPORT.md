# NASA support-box q 诊断

**冻结判定：** DO NOT ADVANCE

## 设计

每个 structure-validation q 先按既有 support-only 协议校准，再逐坐标裁剪到同一 cell 八个 support-matched meta-fit q 的最小/最大范围。该变换无训练、无超参数，也不选择某一个训练实体；query target 只参与最终预测计分。

## 汇总

| 方法 | raw max|z| 中位/最大 | functional max|z| 中位/最大 | box NRMSE | 未约束 NRMSE | cell ratio 中位 | 预测保持 | functional 安全 | 坐标裁剪比例 |
|---|---|---|---|---|---|---|---|---|
| joint_continuity_step1 | 2.276 / 2.547 | 2.436 / 5.549 | 1.702 | 1.193 | 1.208 | 6/15 | 15/15 | 0.6 |
| joint_mse_step1 | 2.14 / 2.564 | 2.348 / 3.137 | 1.145 | 0.9175 | 1.243 | 6/15 | 15/15 | 0.6 |

## Gate

| Gate | 结果 |
|---|---|
| gate_1_integrity | PASS |
| gate_2_box_containment | PASS |
| gate_3_functional_shift | PASS |
| gate_4_prediction_retention | FAIL |
| advance_to_bounded_symbolic_stage_c2 | FAIL |

本轮是连续使用同一 inner cells 的开发诊断。只有四个 gate 全部通过，才可另行冻结有界 symbolic Stage C2；即使通过，也不能把本轮称为独立确认。
