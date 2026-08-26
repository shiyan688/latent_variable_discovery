# NASA convex-support q 诊断

**冻结判定：** DO NOT ADVANCE

## 设计

30 个 decoder/checkpoint 均保持冻结。每个 structure-validation 电池只用前 30% support 目标，把 q 校准为同一 cell 八个 support-matched meta-fit q 锚点的 softmax 凸组合；query target 扰动只用于泄漏审计。未约束 support-matched q 是预先冻结的预测保持比较对象。

## 汇总

| 方法 | raw max|z| 中位/最大 | functional max|z| 中位/最大 | 凸约束 NRMSE | 未约束 NRMSE | cell ratio 中位 | 预测保持 | functional 安全 | 有效锚点 |
|---|---|---|---|---|---|---|---|---|
| joint_continuity_step1 | 2.321 / 2.547 | 2.351 / 2.542 | 1.696 | 1.193 | 1.213 | 6/15 | 15/15 | 1 |
| joint_mse_step1 | 1.824 / 2.426 | 1.664 / 2.125 | 1.197 | 0.9175 | 1.231 | 5/15 | 15/15 | 1.54 |

## Gate

| Gate | 结果 |
|---|---|
| gate_1_integrity | PASS |
| gate_2_convex_containment | PASS |
| gate_3_functional_shift | PASS |
| gate_4_prediction_retention | FAIL |
| advance_to_bounded_symbolic_stage_c2 | FAIL |

只有四个预声明 gate 全部通过，才可另行冻结有界 symbolic Stage C2。凸组合消除 raw-q 外插不等于已证明下游符号价值；预测保持 gate 用于排除仅靠牺牲任务拟合换取坐标安全的情形。
