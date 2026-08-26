# 真实数据 q→符号→结构→q 闭环里程碑

**冻结日期：** 2026-08-25；同日数据资格审计后修订

**状态：** 阶段 A、阶段 B 及 30-cell inner q/功能坐标审计已完成；阶段 C 的同预算符号结构发现待执行

**首选真实系统：** NASA battery capacity

**第二真实系统：** reviewer-clean MATR battery capacity protocol

## 1. 必须实现的最终证据

论文必须在至少一个通过数据资格审计的真实系统上展示完整闭环：

```text
support observations
  -> infer latent q
  -> obtain a compact stage-wise symbolic explanation
  -> extract a stable structural motif from that expression
  -> modify the shared decoder around that motif
  -> relearn train-entity q and recalibrate unseen-entity q
  -> obtain a more stable/interpretable symbolic expression
  -> validate prediction and interpretation on held-out entities
```

这里的表达式不宣称是真实自然定律。它必须是数据支持、跨实体可验证、对模型结构有实际指导作用的阶段性解释，而不能只是对单条曲线的漂亮拟合。

单次失败只淘汰当前公式、坐标化或结构设计，不淘汰这一研究目标。每次失败必须先按数据资格、信息边界、样本量、q 可辨识性、符号预算、优化和模型结构的顺序诊断。

## 2. 为什么仍首选 NASA battery

- 电池级容量轨迹仍与“少量早期观测推断实体状态，再预测后期响应”的目标高度一致。
- 本地 README 明确给出环境温度、负载电流和截止电压等实验条件，可以把输入限制为预测时已知的干预条件。
- 去重和来源资格筛选后仍有 18 个唯一电池，覆盖 5 个有文档依据的协议族，可作 13/5 的外层实体隔离和 8/5 的内层结构验证。
- 该任务允许把 q 的 decoder 响应命名为初始容量、早/晚衰减率和曲率等阶段性功能坐标，因而比混合属性的材料表更适合完成解释闭环。

原 prepared NASA 数据不再是可引用证据：B0025--B0028 在两个批次目录中完全重复，原切分使 B0025、B0026、B0028 的相同曲线横跨 train/test；`voltage_min`、`temperature_mean` 和 `current_abs_mean` 又来自同一放电周期。此前 q 对 kNN 的 10-seed 优势必须在 clean protocol 上重跑，不能继续作为首选 NASA 的实证理由。

NASA 的限制也必须公开：clean cohort 只有 18 个实体，所有实体都已被前期探索间接看过，所以第一轮闭环是 development evidence，而不是完全未曝光的最终确认。最终确认需要冻结结构后使用外部电池数据或预先隔离的新实体。

## 3. 信息边界和实体划分

第一轮使用 2026-08-25 冻结的 reviewer-clean cohort：

- 实体键改为真实 battery ID；四组重复文件先验证逐行完全相同，再只保留一个副本；
- 排除 B0041--B0056，因为其来源 README 明确警告若干极低容量的原因尚未分析；不按模型表现选择电池；
- 仅保留有限正容量且实际放电电流 90% 分位不低于 0.5 A 的周期；
- 输入为 `discharge_index`、实验前已知的 `ambient_temperature`、离散名义 `load_current_amp` 和 README 记录的 `cutoff_voltage`；禁用同周期 `voltage_min`、`temperature_mean`、`current_abs_mean`；
- 5 个协议族各冻结 1 块 outer battery，共 13 train / 5 outer-test；
- 13 个 outer-train batteries 内预先写出 3 个 inner splits，每个 split 为 8 meta-fit / 5 structure-validation，并保证每个协议族各留 1 块验证电池；
- outer evaluation 只在结构冻结后评分；
- test q 只能由该电池 support targets 校准；任何 query-target 扰动都不得改变 q、结构选择或预测。

主任务采用前段 support、后段 query 的 blocked cycle split，以检验退化结构和外推；原随机 support split 只作与历史工作的次要对照。这样不会再把局部 kNN 内插优势误解为对实体级 q 的否定。

## 4. 六个执行阶段

### 2026-08-25 执行快照

Reviewer-clean outer anchor 已按冻结前缀 support 协议完成 35/35 cells，0 失败。q=4 MSE/continuity 都在 5/5 seeds 中优于 support-blind no-q MLP 和 Random Forest，但在 5/5 中不及 prefix-support kNN；因此 clean NASA 不是显式 q 的纯预测胜例。另一方面，continuity q=4 的跨 seed q-distance Spearman 中位数为 0.996，远高于 MSE q=4 的 0.287；cycle-1/cycle-28 capacity 与早/中期 fade 等功能坐标也具有中高跨 seed 稳定性。这使 continuity q=4 成为结构发现的合理候选，但不是因为它预测最好。

三组冻结 inner splits 的 q 阶段也已完成：3 splits × 2 losses × 5 seeds = 30/30 cells，且 30/30 checkpoint 的功能坐标分析完整。continuity 在 15 个结构验证配对单元里只以预测误差胜过 MSE 4 次，配对 NRMSE 中位差为 +0.342803（+32.86%），因此仍不是预测优选；但其三个 split 的 q-distance 跨 seed 中位 Spearman 分别为 0.996716、0.999453、0.998905，MSE 仅为 0.485769、0.308429、0.390805。continuity 的 cycle-1 capacity 和 early-fade 功能坐标在 3/3 splits 都比 MSE 更稳定；它们跨 split 稳定性中位数分别为 0.904762 和 0.821429，并与独立训练曲线描述符保持方向一致（匹配相关的跨 split 中位数分别为 0.880952 和 0.523810）。加速项的经验对齐只有 0.119048，不进入首轮结构改造。

Outer 结果只作为 anchor 和边界诊断，不能用于拟合或选择公式。三组 inner splits 上的 Stage C 已按冻结预算完成 90/90 symbolic cells，运行完整性、8/5 实体隔离、前缀顺序和 query-target leakage=0 全部通过。continuity functional-q 的 structure-validation NRMSE 中位数为 1.755191，差于 condition-only 的 0.936510 和 support summaries 的 1.033182，并且对二者都只有 4/15 配对胜场；其复杂度中位数 13 也高于 continuity raw-q 的 11。因此下游价值和可读性 gate 未通过，不能直接进入 Stage D。

Stage C 同时留下了不可忽略的正线索：`discharge_index` 与至少一个冻结功能坐标共同进入公式，在 continuity 中复现 12/15 次，三个 inner splits 分别为 3/5、5/5、4/5；MSE 为 11/15。functionalization 还在 continuity raw/function 比较中取得 10/15 胜场，把中位 NRMSE 从 3.081231 降到 1.755191。失败诊断显示 raw q 在 structure-validation 上相对 meta-fit 分布的最大绝对 z-score 中位数达 22.19、最大 35.06，配合 `exp` 和小分母产生最大 `6.356848e44` 的有限 NRMSE；functional coordinates 缩小但未消除该 shift。

所以当前冻结判定是：宽 `cycle + functional-q` motif 具有跨 split 复现性，但“完整曲线联合训练 embedding → 前缀 support 校准 test q → 无界符号公式”的接口不具备预测有效性和外推安全性。Stage D 前必须另行冻结信息匹配的接口修复：meta-fit 电池也通过同样的 30% prefix support 反演 q，并审计 support Jacobian 条件性与 q 到训练流形的距离；确认性词表使用有界功能坐标，不允许 `exp(q)`、嵌套指数或无保护 q 分母。完整结果见 `runs/nasa_battery_reviewer_clean_inner_symbolic_20260825/STAGE_C_ANALYSIS.md`；训练动力学解释见 `NEURAL_TRAINING_DYNAMICS_FOR_LATENT_Q_20260825.md`。尚未选择最终公式或 backbone，强制闭环目标仍未完成。

### A. 合格数据与基准冻结

审计每块电池的 cycle 单调性、容量单位、零值/非物理值和协议特征可用时点。任何过滤规则必须是物理规则并在看方法结果前冻结。重跑 no-q MLP、support kNN 和当前 q anchor，确认 blocked split 的信息流和有限指标。

### B. 从 q 构造可比较的功能坐标

不直接把任意旋转的 `q1...q8` 当作物理词汇。对每个训练实体，用冻结 decoder 在统一 cycle/工况网格上的响应提取少数功能坐标，例如初始容量、早期衰减、晚期衰减、曲率或 knee-like 位置。具体坐标由训练实体上的响应变化定义，并且必须是 q 和冻结 decoder 的确定函数。

同时保留 raw q 作对照。功能坐标的目标是消除随机旋转/置换，使不同 seed 的符号词汇可以比较，不是利用 held-out targets 监督对齐。

### C. 找到粗的阶段性表达式

在 meta-fit 实体上比较以下同预算 symbolic interfaces：

- condition only；
- condition + support summaries；
- condition + raw q；
- condition + functional q coordinates。

公式必须在未参与拟合的 structure-validation 实体上评分。除误差和复杂度外，记录跨 seed/inner split 重复出现的变量、算子和结构 motif。只有重复出现且在 validation 上有效的 motif 才能进入结构改造；不以单个最低误差公式作选择。

### D. 用公式 motif 改造 decoder

把稳定 motif 变成最小结构改动，例如：

```text
y_hat = interpretable_backbone(x, coefficients(q)) + controlled_residual(x, q)
```

backbone 的基函数或系数关系来自阶段 C；q 到少数结构系数的映射应尽量直接。residual 用于吸收阶段公式不可能覆盖的真实噪声，但必须报告其相对预测能量，避免完全绕开可解释 backbone。一次只加入阶段 C 明确支持的结构，不做无边界架构搜索。

### E. 重新学习和校准 q

在相同训练实体和相同更新预算下，重新训练结构化 decoder 与 train q；对 unseen battery 仍只用 support 校准 q。比较：

- 原始 q decoder；
- 公式 backbone only；
- 公式 backbone + controlled residual；
- no-q、support kNN 基线。

同时比较 q/功能坐标的跨 seed 对齐、邻域保持和与退化描述符的关系。

### F. 第二次符号回归与闭环验证

对重新学习的 q/结构系数再次执行阶段 C。闭环要回答：公式是否更简单、跨 seed 是否更稳定、held-out entity 是否更准，以及 q 是否更接近可命名的退化坐标。

## 5. 闭环成功标准

以下条件同时满足，才称为“完成一个具有阶段性解释意义的真实闭环”：

1. 数据资格、实体隔离、support-only q calibration 和 query leakage probe 全部通过。
2. 最终表达式在 held-out entities 上优于 condition-only symbolic baseline，并实际使用至少一个 q-derived functional coordinate 和一个物理 condition。
3. 至少一个核心结构 motif 在多数 inner splits/seeds 中重复出现；不能只展示单 seed 最漂亮公式。
4. 公式引导的结构化模型在 outer evaluation 上保持原 q anchor 的主要预测能力；默认容忍线为中位 NRMSE 不恶化超过 5%。
5. 第二次符号回归相对第一次至少在“held-out error、复杂度、跨 seed motif 稳定性”三项中的两项改善。
6. 至少一个最终功能坐标与一个预先定义的电池曲线描述符具有稳定关系，并在 held-out entities 上保持方向；它被称为阶段性退化坐标，不冒充唯一真实物理参数。
7. 原始值、所有 seeds、失败公式和结构选择记录完整保留。

如果第 4 条暂时不通过，可以保留 backbone 作为解释模型继续迭代，但不能宣称闭环完成。如果公式误差改善而解释稳定性不改善，优先处理 q 坐标化；如果解释稳定但预测损失过大，优先调整 backbone/residual 分工；不能据此结束目标。

## 6. 与现有失败实验的关系

现有 real symbolic raw-q 和 canonical-q 开发的程序性 FAIL 不取消本里程碑：

- NASA 当时只有 4 个 symbolic-fit 实体；
- raw q 具有旋转/置换不可辨识性；
- 4 个样本上用 PLS 将 8D q 对齐到 64D 响应签名明显欠定；
- 三个 Starry 分支又继承了无效 prepared data。

它们提供的是下一次设计线索：使用足量训练实体、先定义功能坐标、用跨 split motif 而非单条公式指导结构，然后重新学习 q。它们不是放弃真实闭环的依据。

## 7. 论文中的最终呈现

主文至少需要一张闭环流程图和一组配对表：原 q、第一次公式、结构化 q、第二次公式、no-q 和 kNN。公式旁边必须解释每个功能坐标如何由 q/decoder 得到、它对应哪种曲线行为、在哪些 held-out batteries 上成立、在哪些情形失效。

论文措辞使用“data-supported stage-wise mechanistic surrogate”或“阶段性机理代理”，不使用“发现了真实退化定律”，除非未来有独立物理实验验证。
