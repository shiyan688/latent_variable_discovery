# NASA support-matched q 接口诊断

**冻结判定：** DO NOT ADVANCE

## 设计

30 个既有 checkpoint 均不重新训练 decoder。每个 meta-fit 电池只用前 30% support 重新校准 q，并从 q 先验中排除该实体自己的 full-curve embedding；structure-validation 电池仍使用全部 8 个 meta-fit embedding 构成先验。query target 扰动只用于泄漏审计。

## 汇总

| 方法 | raw-q max|z| 中位 | functional max|z| 中位 | 最近 q 距离 | Jacobian smin | 条件数 | 有效秩 | validation NRMSE |
|---|---|---|---|---|---|---|---|
| joint_continuity_step1 | 9.709 | 4.282 | 6.087 | 0.09684 | 137.1 | 4.00 | 1.193 |
| joint_mse_step1 | 9.425 | 4.891 | 4.262 | 0.121 | 78.28 | 4.00 | 0.9175 |

旧 Stage C continuity raw/functional max-|z| 中位数分别为 22.1915/7.3658。当前冻结 gate 要求 raw 至少减半，functional 中位数不超过 3，且至少 12/15 个 functional cells 不超过 6。

## Gate

| Gate | 结果 |
|---|---|
| gate_1_integrity | PASS |
| gate_2_reproduction | PASS |
| gate_3_continuity_shift | FAIL |
| gate_4_continuity_tail | FAIL |
| advance_to_bounded_symbolic_stage_c2 | FAIL |

support Jacobian 奇异值、条件数和有效秩是诊断量，不参与本轮 advancement gate；它们用于判断 q 校准困难是输入分布错配还是局部不可辨识。只有全部四个冻结 gate 通过，才独立设计有界 symbolic Stage C2；本分析不会自动启动符号回归或结构化 decoder。
