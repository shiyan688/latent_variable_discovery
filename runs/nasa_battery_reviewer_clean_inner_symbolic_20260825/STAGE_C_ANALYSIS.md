# NASA reviewer-clean Stage C：冻结后分析

**日期：** 2026-08-25

**判定：** 运行和信息边界完整；5 个冻结 gate 中通过 3 个（完整性、宽 motif 复现、representation diagnostic），未通过下游预测与可读性 gate，因此不能直接进入 Stage D。

## 1. 这 90 个实验回答什么

每个公式只在 8 块 meta-fit 电池的后 70% 周期上拟合，然后在未参与公式拟合的 5 块 structure-validation 电池后 70% 周期上评分。验证电池的 q 和 support statistics 只使用最早 30% target；query target 扰动不改变任何公式输入。比较对象是 physical conditions only、conditions + support summaries、conditions + raw q、conditions + decoder-functional q。

## 2. 完整性结论

90/90 cells 成功，90/90 预测文件、Pareto fronts 和输入缩放记录存在；所有保存指标和预测有限；每个 cell 都是严格 8/5 实体隔离；30 个 q source 的 query-target leakage probe 最大差为 0.0。完整性 gate：**PASS**。

## 3. 主要数值结果

| 方法 | 接口 | validation NRMSE 中位数 | IQR | 最大值 | NRMSE>10 | 复杂度中位数 | validation max-abs z 中位数 |
|---|---|---|---|---|---|---|---|
| baseline | condition_only | 0.9365 | [0.8551, 0.9850] | 0.9850 | 0/15 | 11.0000 | 2.2330 |
| baseline | condition_support_stats | 1.0332 | [0.8578, 1.1897] | 1.5845 | 0/15 | 12.0000 | 4.5767 |
| joint_continuity_step1 | condition_functional_q | 1.7552 | [0.7779, 2.5653] | 151.2229 | 1/15 | 13.0000 | 7.3658 |
| joint_continuity_step1 | condition_raw_q | 3.0812 | [1.4953, 40.5157] | 2.619e+13 | 6/15 | 11.0000 | 22.1915 |
| joint_mse_step1 | condition_functional_q | 0.9333 | [0.8282, 3.4164] | 25.2977 | 2/15 | 14.0000 | 4.4156 |
| joint_mse_step1 | condition_raw_q | 1.2982 | [0.8089, 4.1554] | 6.357e+44 | 2/15 | 11.0000 | 12.3794 |

`NRMSE>10` 和 `>100` 不是冻结 gate，只是透明呈现外推爆炸的描述阈值。均值被极端有限值支配，因此主表使用中位数、IQR、最大值和尾部计数。

### 配对比较

| 候选 | 锚点 | 胜场 | NRMSE 中位差 | 配对相对中位差 |
|---|---|---|---|---|
| joint_continuity_step1/condition_functional_q | baseline/condition_only | 4/15 | 0.7702 | 78.2% |
| joint_continuity_step1/condition_functional_q | baseline/condition_support_stats | 4/15 | 0.5020 | 48.6% |
| joint_continuity_step1/condition_functional_q | joint_continuity_step1/condition_raw_q | 10/15 | -0.5909 | -25.2% |
| joint_mse_step1/condition_functional_q | joint_mse_step1/condition_raw_q | 7/15 | 0.0913 | 10.5% |
| joint_continuity_step1/condition_functional_q | joint_mse_step1/condition_functional_q | 8/15 | -0.1218 | -13.9% |

continuity functional-q 的 validation NRMSE 中位数为 1.7552，condition-only 为 0.9365，support-statistics 为 1.0332；它对两者都只有 4/15 胜场。冻结 downstream-value gate 明确 **FAIL**。但 functionalization 将 continuity raw-q 的中位 NRMSE 从 3.0812 降到 1.7552，并取得 10/15 配对胜场，说明先把自由 q 坐标映射为 decoder 功能坐标是正确方向，只是当前两个坐标和外推接口还不够。

## 4. 公式里重复出现了什么

| q 训练目标 | inner split | cycle + functional | functional 调制 cycle slope |
|---|---|---|---|
| joint_continuity_step1 | 0 | 3/5 | 1/5 |
| joint_continuity_step1 | 1 | 5/5 | 5/5 |
| joint_continuity_step1 | 2 | 4/5 | 1/5 |
| joint_mse_step1 | 0 | 3/5 | 0/5 |
| joint_mse_step1 | 1 | 5/5 | 4/5 |
| joint_mse_step1 | 2 | 3/5 | 2/5 |

宽 motif（公式同时使用 `discharge_index` 与至少一个功能坐标）在 continuity 中出现 12/15，在 MSE 中出现 11/15；continuity 三个 split 分别为 3/5、5/5、4/5。因此冻结 motif gate **PASS**，且 continuity 仅以 12 对 11 略强于 MSE。更具体的“功能坐标调制退化斜率”只在 continuity 7/15、MSE 6/15 出现，而且跨 split 不均匀；它只能作为 Stage D 候选线索，不能称为已确认结构。

continuity functional-q 的复杂度中位数为 13，raw-q 为 11，故 readability gate **FAIL**。这也说明功能坐标虽然更可比较，却没有在当前无约束算子库中自动产生更简单公式。

## 5. 为什么会发生极端误差

| split | seed | 方法 | 接口 | NRMSE | max-abs z | exp | 除法 |
|---|---|---|---|---|---|---|---|
| 2 | 4 | joint_mse_step1 | condition_raw_q | 6.357e+44 | 19.1342 | yes | no |
| 2 | 2 | joint_continuity_step1 | condition_raw_q | 2.619e+13 | 35.0620 | yes | no |
| 2 | 0 | joint_continuity_step1 | condition_raw_q | 3128.2737 | 20.5654 | yes | no |
| 0 | 2 | joint_continuity_step1 | condition_raw_q | 2589.1608 | 24.4801 | yes | no |
| 0 | 2 | joint_mse_step1 | condition_raw_q | 2265.3681 | 9.3564 | yes | no |
| 1 | 3 | joint_continuity_step1 | condition_functional_q | 151.2229 | 14.2457 | no | no |
| 2 | 3 | joint_continuity_step1 | condition_raw_q | 68.8946 | 22.1915 | yes | yes |
| 2 | 3 | joint_mse_step1 | condition_functional_q | 25.2977 | 6.8605 | yes | no |
| 2 | 2 | joint_mse_step1 | condition_functional_q | 15.5155 | 6.0764 | no | no |
| 0 | 4 | joint_continuity_step1 | condition_raw_q | 12.1369 | 13.8489 | no | yes |
| 2 | 1 | joint_continuity_step1 | condition_raw_q | 10.3030 | 31.1447 | no | yes |

物理 condition 在 validation 中最多约 2.24 个训练标准差，而 raw q 的 group-level validation 最大 |z| 中位数为 22.19（continuity）和 12.38（MSE），最大达到 35.06。PySR 随后把这些域外 q 放进 `exp`、嵌套 `exp` 或接近零的分母，产生了最大 6.36e44 的有限预测误差。functional q 缩小了 shift（最大 |z| 中位数 7.37 和 4.42），但没有消除；3 个 functional cells 仍超过 NRMSE 10。

这不是数据泄漏或运行失败。最直接的协议诊断是：meta-fit q 来自完整训练曲线的联合 auto-decoder 优化，而 validation q 来自前缀 support 的逆向校准；两者的信息量和优化路径不同。符号回归把二者当作同一坐标分布使用，所以测试的同时包含了“q 是否有信息”和“训练 q 与校准 q 是否坐标兼容”两个问题。当前结果首先否定的是这个未约束接口，而不是潜变量是否有用。

## 6. 冻结 gate 总表

| Gate | 结果 | 证据 |
|---|---|---|
| 1 完整性 | PASS | 90/90；finite；8/5；leakage=0 |
| 2 下游价值 | FAIL | 中位数未优于两基线；均仅 4/15 胜 |
| 3 motif 复现 | PASS | continuity 12/15；每 split 至少 3/5 |
| 4 可读性 | FAIL | functional complexity 13 > raw 11 |
| 5 表示诊断 | PASS | continuity motif 12 > MSE 11 |

## 7. 下一步边界

按冻结计划，不能在这 90 个结果上更换阈值、词表或算子后宣称同一 gate 通过，也不能直接把某条漂亮公式塞进 decoder。Stage D 前应先冻结一个独立的接口修复实验：让 meta-fit 实体也通过同样的 prefix-support q calibration 得到公式输入；同时对 functional coordinates 使用训练包络投影或有界系数结构，禁止 `exp(q)`、嵌套 `exp` 和无保护 q 分母。然后在新的开发划分上检验 broad motif 是否仍复现、误差是否优于 condition-only。只有通过后，才把候选结构收缩为“功能坐标控制初始容量/退化斜率 + 小残差”。

全部原始失败公式和极端值保留在 `cell_diagnostics.csv`；本报告不删除、不 winsorize、不重跑任何 cell。
