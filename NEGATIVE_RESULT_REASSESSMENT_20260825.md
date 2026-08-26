# 历史负结果与数据集资格复核

**日期：** 2026-08-25

**性质：** 内部证据审计；没有启动新训练，也没有改变任何原始结果

**目的：** 区分“方法确实失败”与“数据、任务、协议或可辨识性不足”，防止看到负结果后过早否定显式潜变量方法

## 1. 先冻结判定规则

数据集或实验只有同时满足以下规则，才有资格对核心方法作正面或负面判断：

1. 一个 `label` 对应一个物理实体或一个定义清楚、近似静态的系统状态，不能把多个样品、论文或实验条件合并成同一个实体。
2. target 是单一物理量，单位和语义一致；不能混合原值、对数值、总量、分量或按其他量归一后的值。
3. q 表示 support 中可辨识但 observed x 没有完整给出的实体因素；如果 x 已经完整编码 label，必须明确 q 只表示未观测的样品残差状态。
4. support/query 切分要匹配论文问题。随机点切分主要检验曲线内插，不能直接支持未来区间外推或系统参数辨识的主张。
5. 训练、验证和测试实体数足以支撑所用模型和统计结论。符号回归尤其不能用少数几个实体拟合多个 q 坐标后作一般性结论。
6. 合成真 q 的恢复只在真因素对观测有足够敏感度、任务数值稳定、且恢复等价类定义正确时才成立。

排除规则必须在看方法成绩之前应用，并同时作用于好结果和坏结果。被判无效的原始文件保留作审计记录，但不进入论文聚合。

## 2. StarryData2：发现了确定的数据构造问题

### 2.1 后期 `application_full_features` 版本无资格评价方法

后期大规模实验引用：

```text
data/application_full_features/prepared_datasets.json
```

实际 CSV 与当前文档不一致：`label` 是化学组成字符串，不是 `sample_id`。同时 87 个输入特征中，除 temperature 外的 86 个元素分数在每个 label 内恒定，并且几乎唯一识别该 composition label。因此这个版本把材料组成完整提供给 x，又用固定 q 去吸收同组成下不同样品、论文和工艺的冲突残差。

对实际选中的 80 个 composition labels 回查原始 StarryData2：

| 属性 | 每个 label 包含的 sample IDs（中位） | 多 sample label | 每个 label 包含的 DOI（中位） | 多 DOI label |
|---|---:|---:|---:|---:|
| Seebeck | 13.5 | 80/80 | 6 | 64/80 |
| electrical conductivity | 8 | 78/80 | 3 | 57/80 |
| thermal conductivity | 6 | 74/80 | 3 | 57/80 |

这不是“一条实体曲线对应一个固定 q”的任务。它还存在属性解析混杂：

| 属性 | 当前展开行中的混杂 |
|---|---|
| Seebeck | 1.5% 使用 inverse temperature / `K^-1`；少量 `unit_y=V` |
| electrical conductivity | 11.5% `log(electrical conductivity)`、1.3% `ln(...)`；22.6% inverse temperature，另有幂次温标 |
| thermal conductivity | 17.6% lattice、约 1.8% electronic/electron，另混入 carrier/total 等定义 |

当前随机 support/query 还使约 96%--97% query 温度落在 support 温区内部，主要评价局部内插。三项数据的原始 target 也出现明显异常或语义混合：Seebeck 最小值 -166.6768，而 1% 分位仅 -5.49e-4；electrical 出现负值且最大 1.227956e8；thermal 出现负值。

**裁决：** 所有以 `application_full_features` 三项 Starry 为依据的优劣结论作废。不能再用它们证明 q 爆炸、kNN 更好、selector 应选择 kNN、或 q 不适合真实符号回归；相同产物上的任何正结果也一并作废。StarryData2 这个来源本身尚未被排除，可以按严格规则重建后重新判断。

### 2.2 较早 `application_reviewer_clean` 版本要逐属性处理

该版本已经使用 `sample_id` 作 label，因此不具有上述 composition 合并错误，但旧的模糊字符串属性筛选仍存在：

| 属性 | 非精确记录影响的实体 | 裁决 |
|---|---:|---|
| ZT | 0/80 | 保留；当前检查未发现该类混杂 |
| Seebeck | 2/80，其中 test 1 个 | provisional；严格移除倒温度记录后才能作为确认结果 |
| electrical conductivity | 21/80，其中 test 5 个 | 作废；混入 log 值和非温度坐标 |
| thermal conductivity | 41/80，其中 test 12 个 | 作废；混入 lattice/electronic 等不同目标 |

磁盘审计中，至少 1,035 个 `result.json` 明确链接到 `application_full_features` Starry，另有 234 个链接到 reviewer-clean electrical/thermal；这些原子结果不再具备论文证据资格。148 个 reviewer-clean Seebeck 结果降级为 provisional；117 个 reviewer-clean ZT 结果不受本次问题影响。这里统计的是当前磁盘原子文件，不等于独立假设数量，也不把重复引用包装成独立证据。

## 3. 其他真实和数值数据集

| 数据集/实验 | 场景资格 | 对历史负结论的处理 |
|---|---|---|
| NASA battery | 科学场景强匹配，但原 prepared 数据不合格：重复 battery ID 跨 train/test，且含同周期响应统计特征 | 18-entity reviewer-clean blocked-cycle anchor 已完成；q 优于 support-blind MLP/RF，但不及 prefix-support kNN；符号闭环仍待 inner splits 完成 |
| MATR battery | 强匹配：一块电池一条协议条件下的容量曲线，46 个实体 | 保留；但随机 support 的 kNN 胜出只说明内插性能，需用前段 support / 后段 query 再评价外推价值 |
| UCI gas drift | 语义上可把 batch drift 视为 q，但只有 10 个实体、约 2 个 test 实体 | 降级为小样本压力测试；117 个现有原子结果不足以形成方法负结论 |
| 已移出论文范围的发动机任务 | 固定 engine q 与时间变化健康状态/故障过程的关系不够符合当前论文主线 | 不进入论文，也不再用于否定核心方法；磁盘上约 1,042 个相关原子结果只保留历史记录 |
| PDEBench Burgers | 公共且有效，但每条轨迹由高维初值决定；随机 30% 网格 support 让局部 kNN 天然适合内插 | 保留为外部数值压力测试。q 胜过 no-q、FPCA、DeepONet 的结论有效；kNN 胜 q 只限定在该随机稀疏内插协议，不是否定低维物理状态方法的核心证据 |

PDEBench 的关键实验至少包含 164 个预设任务（extended 114、matched 30、functional baselines 20）。这些任务本身有效，问题是过去把“局部内插基线领先”过度外推成了对核心方法的一般限制。

## 4. 真实 q 的符号接口被过早判为方法失败

NASA battery 原实验不只存在 9 个 test 实体再分成 4/5 的小样本问题，还把 B0025--B0028 的重复文件当成不同实体，其中三块相同电池跨越 train/test；同周期最低电压、平均温度和平均电流也进入了 query 特征。后续 canonical-q 的 PLS 又只用 4 个实体拟合 8 维 q 到 64 维响应签名，统计上明显不足。

三个 Starry 属性的 45 个 symbolic cells 又继承了无效的 `application_full_features` q。由此，当前 48-cell raw-q gate 和 12-cell canonical-q gate 的运行完整性仍然成立，但它们不能支持“真实 q 没有符号价值”这一科学结论。

**裁决：** 两个 gate 的 FAIL 仍作为预注册流程结果保留；科学解释改为“当前数据资格与 symbolic-fit 实体规模均不足，不能判定”。NASA 已冻结 18 个唯一 battery ID 的 clean cohort、13/5 outer split 和三个 8/5 inner split；Starry 只能在严格重建后进入该实验。

## 5. 合成恢复的 18 个负结局不能统一叫方法失败

全 46 表达式审计记录为 28 recovered、12 not recovered、3 optimization diverged、3 weak control margin。重新解释如下：

- 3 个 numerical overflow 是数值范围/优化失败，不是潜变量原理失败。
- 3 个 weak-control-margin 表示真 q 对观测贡献太弱，或 no-q/shuffled 控制已近乎同样好；它们不具备强恢复可辨识性。
- 12 个 not recovered 是“下游符号恢复未过阈值”，不能等同于“q 预测失败”。例如 expression 2、12、31、38、47 的 matched 1,000-epoch q 模型中位 NRMSE 分别约为 0.0122、0.0127、0.0148、0.00424、0.0160，却仍未通过符号恢复门槛。
- denominator、幂指数、嵌套指数和 multi-q interaction 还存在奇点、31 个数量级动态范围、非线性重参数化和不可唯一辨识等问题。

**裁决：** 28/46 可以保留为一个严格的“当前 symbolic recovery protocol 通过率”，不能再表述为“方法只有 60.9% 有效”。至少 6 个负结局明确属于数值或可辨识性不足；剩余 12 个是待分解的 downstream-recovery 难例，其中多项已经证明 q 在预测上有效。

## 6. 历史结论需要怎样改写

| 旧结论 | 新状态 | 原因 |
|---|---|---|
| Starry 上 q 普遍失败、kNN 普遍更好 | 撤回 | 后期实体定义错误；早期 electrical/thermal 目标混杂 |
| bounded CNP / selector 解决了 Starry 灾难 | 降级为无效数据上的工程诊断 | 它控制了输出范围，但没有在合格 Starry 任务上证明科学优势 |
| learned support encoder / CNP 没有一般价值 | 撤回一般化表述 | 跨数据集 gate 包含无效 Starry、无效 NASA prepared 数据和已移出范围的任务 |
| 真实 q 缺少下游符号价值 | 撤回 | Starry q 来源无效；NASA 只有 4 个 symbolic-fit 实体 |
| kNN 在多个任务领先，所以显式 q 不必要 | 降级 | 多个任务无效或弱匹配，且随机 support 主要测局部内插；需要在合格数据和 blocked/extrapolative split 上重算 |
| continuity/其他正则常损害真实任务 | 重新聚合 | clean NASA 中 continuity q=4 预测只在 1/5 seeds 胜 MSE q=4，却把 q-distance 跨 seed Spearman 从 0.287 提高到 0.996；预测和表示几何必须分开判断 |
| q 只在 28/46 表达式有效 | 撤回这种表述 | 28/46 是符号恢复门槛通过率，不是预测有效率；6 个负结局明确不是方法失败 |

不是所有负结果都应翻案。以下判断目前仍站得住：

- NASA 上两个 selector 的冻结门槛确实按原协议失败，但原 prepared 数据已失去论文资格；这些记录既不能证明 selector，也不能证明或否定纯 q。
- 精确 Burgers 中 prediction-optimal q 与 affine physical alignment 不一致，且部分几何正则在独立 seeds 上未复现；这一结论来自有效解析任务和确认种子。
- 支持 kNN 是随机稀疏内插协议下必须保留的强基线；不能因为它不是学习方法而删除。
- 早期 battery `q_charge` 泄漏、adaptive24 过小内部切分和 7 个 CUDA 初始化失败都已经被正确归因为实验问题，没有被误写成方法失败。

## 7. 本轮复核后的论文证据层级

1. **核心确认：** 可辨识且数值稳定的合成表达式、精确 Burgers。
2. **优先补强：** reviewer-clean NASA 的 blocked/forward 闭环；MATR battery 的 blocked/forward split；严格重建后的 sample-level Starry；足量实体的真实 symbolic downstream。
3. **外部压力测试：** PDEBench 随机稀疏网格。
4. **只作历史/附录：** 已移出范围的发动机任务、UCI gas 小实体任务、所有旧 Starry 无效构造。

NASA 的合格 blocked-cycle anchor 已完成，但不能恢复旧的“q 在 NASA 纯预测领先”主张：q=4 在 5/5 seeds 中优于 no-q MLP 与 RF，也在 5/5 中不及 prefix-support kNN。可保留的新线索是 continuity q 的跨 seed 几何和功能坐标显著更稳定。Starry 尚未完成合格重建，真实 q 的符号闭环也尚未完成。因此正确的当前结论是：**核心方法在合成和解析 PDE 上已有强证据；clean NASA 证明 q 含有额外实体信息与稳定表示价值，但真实材料泛化和真实符号价值仍未被合格实验充分检验。**

真实数据上的 q 下游价值不是可因当前负 gate 而删除的目标。后续必须完成 [REAL_Q_SYMBOLIC_STRUCTURE_LOOP_MILESTONE_20260825.md](REAL_Q_SYMBOLIC_STRUCTURE_LOOP_MILESTONE_20260825.md) 定义的闭环：q → 阶段性符号表达式 → 结构改造 → 重新学习 q → 更稳定、可解释的符号表达式。
