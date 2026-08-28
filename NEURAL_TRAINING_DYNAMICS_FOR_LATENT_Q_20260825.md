# 从神经网络训练动力学理解本项目的隐变量 q

**日期：** 2026-08-25

**MATR 更新：** 2026-08-28

**读者定位：** 不要求预先掌握 NTK、表示可辨识性或逆问题理论

**范围：** 解释当前 latent-q auto-decoder、continuity loss、unseen-entity q calibration 与符号回归接口；不把理想化理论冒充对有限宽 ReLU 网络的完整证明

## 1. 一页结论

本项目训练的不是普通的 $y=f(x)$，而是

\[
\hat y_{ir}=f_\theta(x_{ir},q_i),
\]

其中 $i$ 是电池或其他科学实体，$r$ 是该实体上的观测点，$x$ 是可见条件，$q_i\in\mathbb R^d$ 是每个训练实体一份、与 decoder 参数 $\theta$ 一起学习的隐向量。遇到新实体时冻结 $\theta$，只用少量 support 点反向优化一个新 $q$，再预测 query。

从训练动力学看，当前最重要的不是“网络能不能拟合”，而是下面四件事：

1. **MSE 只能约束 $q$ 与 decoder 的组合，不能唯一决定 q 的坐标系。** 对第一层权重作相反变换，可以旋转、缩放、平移 q 而保持所有预测完全不变。raw `q1...q4` 因此不是天然物理量。
2. **continuity loss 是一种几何选规（gauge fixing）。** 它使实体响应曲线相似时 q 距离也相似，打破大部分任意仿射变换，所以跨 seed 的距离几何非常稳定；但它仍允许旋转、反射、平移和整体缩放，不会自动把 `q1` 命名为“初始容量”。
3. **训练 q 与新实体校准 q 来自两条不同动力学路径。** 训练 q 与 decoder 联合优化并看完整训练曲线；测试 q 在冻结 decoder 下只看前 30% support。若 support 对某些 q 方向不敏感，许多相距很远的 q 都能给出相似 support loss，校准就会离开训练 q 流形。
4. **这解释了当前 Stage C 的表面矛盾。** continuity q 的跨 seed 距离 Spearman 约 0.997–0.999，`cycle + functional-q` motif 也在 12/15 公式中复现；但 raw q 在新实体上偏离 meta-fit 分布 10–35 个标准差，再经过 `exp` 或小分母便产生极端符号外推。表示“有结构”和接口“可安全迁移”是两个不同问题。

因此，最有理论根据的下一步不是盲目加 epoch 或换更大 MLP，而是：让训练实体也经过与测试实体相同的 prefix-support q 反演；测量 support Jacobian 的秩与条件数；约束 q 校准在训练流形/功能坐标包络内；随后才把稳定 motif 写进有界的结构化 decoder。

## 2. 当前代码实际上优化什么

当前 NASA inner-q 配置是：$d=4$，两层 ReLU MLP（256、128 hidden units），Adam、学习率 $10^{-3}$、1000 epochs、joint update；每个 epoch 对 $\theta$ 和 q 各产生相同的 joint optimizer steps。MSE 版本的目标是

\[
L_{\mathrm{MSE}}(\theta,Q)
=\frac{1}{N}\sum_{i,r}
\left[f_\theta(x_{ir},q_i)-y_{ir}\right]^2.
\]

continuity 版本增加权重 $\lambda=0.05$ 的实体距离匹配：

\[
L(\theta,Q)=L_{\mathrm{MSE}}(\theta,Q)
+\lambda L_{\mathrm{cont}}(Q),
\]

\[
L_{\mathrm{cont}}
=\frac{1}{M(M-1)}\sum_{i\ne j}
\left[
\operatorname{Norm}(\lVert q_i-q_j\rVert_2)
-\operatorname{Norm}(D^{\mathrm{curve}}_{ij})
\right]^2.
\]

$D^{\mathrm{curve}}_{ij}$ 是两条训练响应曲线在统一 64 点 cycle 网格上的距离。代码中的 `Norm` 对所有非对角距离做均值/标准差标准化，因此 continuity 关心的是相对距离结构，不关心 q 的绝对平移和整体尺度。

一个很重要的实现细节是：

- decoder 参数 $\theta$ 只收到 prediction loss 的直接梯度；
- q embedding 同时收到 prediction 与 continuity 的梯度；
- continuity 通过 q 的变化间接改变 decoder 后续看到的输入分布。

所以这里的“多目标梯度冲突”首先发生在 q 上，不应笼统地对所有网络参数使用梯度手术。

在连续时间近似下，joint gradient flow 是

\[
\dot\theta=-\nabla_\theta L_{\mathrm{MSE}},
\qquad
\dot q_i=-\nabla_{q_i}L_{\mathrm{MSE}}
-\lambda\nabla_{q_i}L_{\mathrm{cont}}.
\]

prediction 部分对某个实体 q 的梯度为

\[
\nabla_{q_i}L_{\mathrm{MSE}}
=\frac{2}{N}\sum_r
J_q(x_{ir},q_i)^\top
\left[f_\theta(x_{ir},q_i)-y_{ir}\right],
\]

其中 $J_q=\partial f_\theta/\partial q$。这条式子给出一个直接解释：只有 decoder 对 q 敏感的方向才有可观梯度；若 $J_q$ 某些方向接近零，MSE 不可能稳定辨识这些 q 分量。

## 3. 第一条核心理论：q 有精确的坐标对称性

设 MLP 第一层写成

\[
h=\sigma(W_xx+W_qq+b).
\]

任取可逆矩阵 $A$ 和向量 $c$，定义

\[
q'_i=Aq_i+c,\qquad
W'_q=W_qA^{-1},\qquad
b'=b-W_qA^{-1}c.
\]

那么

\[
W'_qq'_i+b'=W_qq_i+b,
\]

因此所有训练和测试预测完全不变。这不是近似，而是当前架构第一层的精确重参数化对称性。

它带来三个后果：

- 单独解释 `q1`、`q2` 没有理论依据；不同 seed 可以选择不同旋转、缩放或剪切后的坐标。
- 训练误差相同的解形成一整条等价轨道；初始化、学习率、Adam 的历史状态和 minibatch 顺序都可能决定最终落在哪个代表元上。
- “q 能预测”不推出“q 的每一维可跨 seed 做符号回归”。必须解释等价类不变量，例如 q 的距离几何或 decoder 在固定物理 probe 上的响应。

这与无监督 disentanglement 的一般不可辨识性结论一致：没有额外的模型或数据归纳偏置，不能期望从观测中唯一找回有名字的非线性潜在因素。[Locatello et al., ICML 2019](https://proceedings.mlr.press/v97/locatello19a.html) 给出了这一类不可辨识性的理论与大规模实证。矩阵分解研究也说明，梯度下降会在大量等价解中产生隐式偏好，而且该偏好依赖参数化、初始化和优化路径，并不等价于显式物理约束。[Gunasekar et al., NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html)

### continuity 到底固定了什么

一般的剪切或各向异性缩放 $A$ 会改变欧氏距离，所以 continuity 会惩罚这类重参数化。可是平移、正交旋转/反射以及整体缩放在距离标准化后仍不变。也就是说，continuity 把 MSE 的大致 $GL(d)$ 自由度收缩到相似变换自由度，但没有唯一固定坐标轴。

这正好预测了当前结果：continuity 的 q-distance 跨 seed Spearman 极高，但 raw q 坐标不应被期待逐维一致。decoder-functional coordinate

\[
z_k(q)=f_\theta(x_k^{\mathrm{probe}},q)

\]

在 q 与 decoder 同时作上述反向重参数化时保持不变，因此比 raw q 更适合作为跨 seed 的符号词汇。Stage C 中 functionalization 把 continuity raw-q 的验证 NRMSE 中位数从 3.081 降到 1.755，并有 10/15 配对胜场，符合这一预测。

## 4. 第二条核心理论：continuity 在重塑 q，但不保证最佳预测

对 q 而言，两个梯度可能有三种关系：

\[
\cos\phi_i=
\frac{
\langle \nabla_{q_i}L_{\mathrm{MSE}},
\nabla_{q_i}L_{\mathrm{cont}}\rangle
}{
\lVert\nabla_{q_i}L_{\mathrm{MSE}}\rVert
\lVert\nabla_{q_i}L_{\mathrm{cont}}\rVert
}.
\]

- $\cos\phi_i>0$：两个目标大体协同；
- $\cos\phi_i\approx0$：continuity 主要沿 MSE 的平坦方向选择一个更规则的 q；这是最理想的情形；
- $\cos\phi_i<0$：为了匹配曲线几何而牺牲当前预测下降方向，可能出现表示更稳定但预测稍差。

多任务优化文献已经系统研究了负 cosine、曲率和梯度尺度差异如何造成有害干扰；例如 [Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)。但对本项目，正确的诊断是先记录 q 上两个分量的 cosine 与范数，而不是直接套用 PCGrad。

当前 30 个 NASA inner-q runs 中，continuity 和 MSE 的训练 $R^2$ 中位数分别为 0.9547 与 0.9529，几乎相同；continuity 的跨 seed 几何却大幅更稳定，说明正则很可能主要在训练误差的平坦等价方向上选解。continuity 的 terminal raw loss 中位数约 $9.9\times10^{-5}$，乘 0.05 后约 $5\times10^{-6}$，不能据此说它“太小所以没作用”：它可能在早期先移动了 q，等距离已经匹配后自然衰减。必须看整条梯度轨迹，不能只看最后一个标量。

## 5. 第三条核心理论：宽网络有“惰性拟合”与“特征学习”两种动力学

在无限宽、合适缩放和小参数位移的理想条件下，网络训练可由初始化处的一阶线性化描述：

\[
f_{\theta_t}(u)\approx f_{\theta_0}(u)
+\nabla_\theta f_{\theta_0}(u)^\top(\theta_t-\theta_0).
\]

此时函数演化由 Neural Tangent Kernel 控制，误差沿核的较大特征值方向收敛更快。[Jacot et al., NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html) 建立了这一无限宽结果；[Chizat, Oyallon & Bach, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html) 说明这种 lazy regime 与参数缩放密切相关，且真正需要学习新特征的任务可能不适合过度惰性化。

对本项目，这给出两个相反但可检验的风险：

1. 若 decoder 太宽、学习率/初始化使其近似 lazy，网络倾向用初始随机特征解释各实体，q 只在固定切空间里补残差，未必形成紧凑物理坐标。
2. 若 q 和 decoder 都大幅移动，系统进入强 feature-learning regime；这有机会学到共享机理，但 q-decoder 的坐标系也可能随训练共同漂移，使 calibration landscape 与随机 seed 更敏感。

因此“再加宽网络”不必然改善 q。应测量训练过程中的 decoder 参数相对位移、经验 NTK/feature Gram 漂移和 q 几何形成时间。如果 q 的稳定几何在 decoder 几乎不动时已形成，continuity 主要是在固定特征上做度量嵌入；如果几何形成伴随显著 kernel 漂移，则结构化 decoder 更可能改变结果。

## 6. 第四条核心理论：谱偏置解释为什么先学到粗退化趋势

ReLU 网络通常先学低频、全局平滑的函数成分，较高频或局部变化学习更慢；这被称为 spectral bias。[Rahaman et al., ICML 2019](https://proceedings.mlr.press/v97/rahaman19a.html) 从 Fourier 视角给出了理论与实证分析。

在 battery curve 中，这意味着：

- 全局容量水平和近似线性早期衰减比 knee、局部恢复、异常波动更早进入 decoder；
- continuity 使用统一网格上的整条曲线距离，会首先被这些低频、大幅度差异主导；
- 当前最稳定的功能坐标恰好是 cycle-1 capacity 和 early fade，而 acceleration 对齐很弱，这与谱偏置方向一致。

但这仍是项目级推论，不是现成定理对 NASA 数据的直接证明。验证方式是对训练 checkpoint 的 decoder curves 做低/高频分解，比较每个频带的误差、q 可解释度和形成时间。如果 early fade 的跨 seed 稳定性早于 curvature/knee 出现，才支持这一解释。

## 7. 第五条核心理论：unseen q calibration 是一个病态逆问题

新电池只用 support 集 $S_i$ 求

\[
q_i^*=\arg\min_q
\sum_{r\in S_i}
\left[f_{\hat\theta}(x_{ir},q)-y_{ir}\right]^2.
\]

这种 test-time latent optimization 与 auto-decoder 方法一脉相承；例如 [DeepSDF, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html) 在训练时联合学习实例 latent codes，并在测试时冻结 decoder、从观测优化新 code。

在某个候选 $q_0$ 附近线性化：

\[
f(x,q_0+\delta q)\approx f(x,q_0)+J_S\delta q.
\]

局部 Gauss–Newton Hessian 近似为

\[
H_q\approx 2J_S^\top J_S.
\]

若 $J_S$ 的最小奇异值很小，support 几乎看不见某个 q 方向；若秩小于 $d$，局部存在整片等价 q。此时增加 calibration steps 只能更充分地沿一个病态谷底优化，不能创造缺失的信息，甚至可能让 q 走得更远。

这能解释三个现象：

- 当前四起点 calibration 的候选 q dispersion 中位数，continuity 为 0.246、MSE 为 0.410，说明 continuity 确实让逆问题更集中，但没有消除不确定性。
- Stage C raw q 在 structure-validation 上相对 meta-fit q 达到 10–35 个标准差；物理 conditions 只有约 2.24 个标准差。
- functional q 降低但未消除分布偏移，因为 decoder 在训练 q 流形之外仍可输出数值，却不保证这些数值具有训练时相同的语义。

因此 calibration 的关键指标不只是 support MSE，而应包括 $J_S$ 奇异值、有效秩、条件数、不同初始化 q dispersion、到训练 q 流形的距离，以及这些量与 query error 的相关性。

## 8. 用训练动力学重新解释 Stage C

| 观察 | 训练动力学解释 | 当前证据强度 |
|---|---|---|
| continuity q-distance 跨 seed 极稳定 | distance stress 消除了大部分仿射 gauge，自由度缩小到相似变换 | 强；3 splits 中位 Spearman 0.997–0.999 |
| functional coordinates 比 raw q 稳定 | decoder probe response 对 q-decoder 联合重参数化不变 | 强；既有跨 seed/cross-split analysis + Stage C raw/function 配对 |
| continuity 预测不如 MSE | continuity 在 q 上选择几何，不保证与 support-query 外推最优方向一致 | 中；需要分量梯度 cosine 轨迹确认 |
| cycle + functional motif 12/15 复现 | 低频退化趋势先被学到，功能坐标提供实体级水平/斜率差异 | 中；宽 motif 强，但具体 slope modulation 仅 7/15 且 split 不均匀 |
| raw-q 符号公式爆炸 | train embedding 与 support-calibrated q 不同分布；无界 `exp`/除法放大域外输入 | 强；最大 \|z\| 35.06，最大有限 NRMSE 6.36e44 |
| terminal continuity loss 很小 | 可能早期已完成选规，后期梯度自然衰减 | 合理但未验证；当前没有 component-wise gradient trace |

这里最重要的认识是：**q 的信息价值、q 的几何稳定性、q 的坐标可命名性、q 的 support 可辨识性、以及 q 进入符号模型后的外推安全性，是五个不同命题。** 一个命题失败不能自动推翻其他四个。

## 9. 最小而有判别力的理论验证计划

以下实验按信息增益排序，不是无边界调参。

### A. 训练阶段动力学轨迹

在一个冻结 inner split、MSE/continuity 各 5 seeds 上，每 20 epochs 保存：

- prediction 与 continuity loss；
- q 上两个分量梯度的范数及 cosine；
- decoder 梯度范数、q/decoder 参数相对位移；
- train $R^2$、q-distance 跨 seed 对齐、functional-coordinate 对齐；
- 固定 probe 上的 empirical feature/NTK Gram 漂移。

成功判据不是谁的 loss 更低，而是能回答：continuity 几何在何时形成、主要沿 MSE 平坦方向还是冲突方向形成、decoder 是否处于近似 lazy regime。

### B. calibration 可辨识性审计

不重新训练模型。对现有 30 个 checkpoint 和每块 validation 电池：

- 在最终校准 q 处计算 support Jacobian $J_S$ 的奇异值谱；
- 记录 rank、condition number、最弱方向及四起点 dispersion；
- 关联 `max |z|`、functional shift、neural query NRMSE 与 symbolic query NRMSE。

若小奇异值/大条件数稳定预测 q shift 和 query error，就确认主要瓶颈是逆问题而不是 q 不含信息。

### C. 信息匹配的 symbolic interface

对 meta-fit 电池也只用各自最早 30% support，在冻结 decoder 下重新校准 q；公式拟合与 validation 均使用同一种 support-inferred q。训练 embedding 只作为 prior/初始化，不直接作为公式输入。

这是 Stage C 后信息增益最高的协议修复。它隔离“q 是否有下游价值”与“完整曲线 train q / prefix-support test q 的 domain shift”。

### D. 有界校准与有界公式

比较同预算的三种边界：无约束、到训练 q 分布的 Mahalanobis/trust-region 约束、decoder-functional coordinate 的训练包络投影。符号词表只允许对 q-derived coordinates 作线性、乘法和有界变换；`exp(q)`、嵌套 `exp` 与无保护 q 分母不进入确认性搜索。

如果尾部爆炸消失但中位性能不变，说明这是数值安全修复；如果中位数和胜场也改善，说明 calibration manifold constraint 同时提高了科学泛化。

### E. 最小结构化 decoder

只有 C/D 在新冻结验证上通过后，才测试

\[
\hat y(t,x,q)
=C_0(q,x)-k(q,x)t+r_\theta(t,x,q),
\]

其中 $C_0$ 和 $k$ 是有界、低维、可读的系数头，residual 的能量受控并单独报告。当前 Stage C 支持的是“cycle 与 functional coordinate 共同出现”的宽 motif；更具体的 slope modulation 还未跨 split 充分复现，所以这只是候选结构，不是已确认公式。

## 10. 哪些说法现在可以写，哪些不可以

可以写：

- continuity regularization substantially stabilizes the latent distance geometry across random seeds；
- decoder-functional coordinates reduce the coordinate arbitrariness and raw-q symbolic extrapolation risk；
- current support-only calibration is not distribution-compatible with full-curve train embeddings, revealing an identifiability/interface bottleneck；
- a recurring stage-wise motif links cycle progression with q-derived capacity/fade coordinates, motivating a bounded structured decoder after independent confirmation。

现在不能写：

- `q1` 或 `q2` 就是某个真实物理参数；
- continuity 理论上必然提高预测；
- 1000 epochs 足以解决 support 中不存在的信息；
- Stage C 已经发现了真实电池退化定律；
- 删除极端公式后 functional q 就胜过基线。

最稳妥的论文主张是：**本方法学习了一个对实体响应几何有稳定编码的潜在等价类；功能探针把这个等价类变成可比较的阶段性坐标。当前主要瓶颈是新实体逆向校准与训练坐标流形不匹配，而不是缺少表示结构。** 这条主张已经有证据，但仍需要 A–D 中至少 calibration conditioning 与 information-matched interface 的确认。

## 参考的一手文献

1. Jacot, Gabriel & Hongler. [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://papers.nips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html). NeurIPS 2018.
2. Chizat, Oyallon & Bach. [On Lazy Training in Differentiable Programming](https://proceedings.neurips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html). NeurIPS 2019.
3. Rahaman et al. [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html). ICML 2019.
4. Locatello et al. [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://proceedings.mlr.press/v97/locatello19a.html). ICML 2019.
5. Gunasekar et al. [Implicit Regularization in Matrix Factorization](https://proceedings.neurips.cc/paper_files/paper/2017/hash/58191d2a914c6dae66371c9dcdc91b41-Abstract.html). NeurIPS 2017.
6. Yu et al. [Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). NeurIPS 2020.
7. Park et al. [DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html). CVPR 2019.

这些文献支持的是相应的一般理论背景；本文关于当前代码的 gauge 变换、梯度分流、局部 $J_S^\top J_S$ 条件性及 Stage C 解释，是基于项目实现与结果作出的明确推导和可检验推论。

## 11. MATR 新发现：同一个正则权重会随 mini-batch 数量改变实际剂量

MATR Batch1 有 37,693 个训练观测，batch size 为 256，所以每个 epoch 有 148 个 mini-batch。`latent_curve_continuity` 却不是一个只依赖当前 mini-batch 的局部量；它每次都使用完整 41 个训练电池的 q embedding 和完整曲线距离矩阵：

\[
R(Q)=L_{\mathrm{cont}}(q_1,\ldots,q_{41}).
\]

旧实现对每个 batch 都优化

\[
L_m=L_{\mathrm{pred},m}+\lambda R(Q).
\]

因此一整个 epoch 对全局项的一阶累计暴露近似是

\[
\sum_{m=1}^{M}\lambda R(Q)=M\lambda R(Q).
\]

当 NASA 小切分每个 epoch 只有约 3 个 batch、MATR 有 148 个 batch 时，同样写着 `λ=0.05`，并不是同一个 epoch-level objective：MATR 对 continuity 的重复施加次数约高 49 倍。五个未归一化 MATR seed 都在 1,000-epoch 训练中得到 non-finite loss，而且没有产生任何 Batch2 prediction result。这首先是损失离散化随数据规模变化的问题，不能被解释成“q 没有预测价值”。

修正后的定义令第 $m$ 个 batch 含 $n_m$ 行、总训练行数为 $N$，只对完整 embedding 上计算的全局正则乘

\[
s_m=\frac{n_m}{N},\qquad \sum_{m=1}^{M}s_m=1.
\]

于是

\[
L_m=L_{\mathrm{pred},m}+s_m\lambda R(Q),
\]

每个 epoch 的全局正则总质量不再依赖数据被切成多少个 batch。最后一个不满 256 行的 batch 使用自己的实际 $n_m$，所以这不是粗略的 `1/148` 常数近似。只缩放完整 q population 上的 continuity、feature orthogonality、q-L2 和 q-whitening；prediction、Jacobian 与当前 batch 上的 smoothness 不缩放。

这项修复也不是 gradient clipping。clipping 会在梯度过大以后截断方向和幅值；epoch normalization 是先把离散目标定义成跨数据规模可比较的量。Adam 的动量与非线性更新意味着“每 epoch 完全等价”不是严格定理，但它消除了最直接的 $M$ 倍重复暴露。

Batch1-only 的五 epoch 配对诊断没有读取 Batch2 target。两个版本短程都保持有限，但未归一化与归一化版本第一次记录到的 q-gradient norm 约为 0.165 与 0.0296；两者每 epoch 都是 148 个 outer batches，短程观测到的 q-phase support 行数 min/median/max 为 4/28/45，而全局正则 scale sum 分别为 148 和 1。这个短程结果只证明修正实际改变了全局正则的训练动力学剂量；它没有覆盖长期随机排列中可能出现的零-support batch。

五个 epoch-normalized 正式训练随后在 epoch 240--445 全部终止，而且都发生在最后一个 raw mini-batch：continuity 仍是有限值，prediction 却是 `NaN`。进一步检查发现这是第二个、彼此独立的离散化问题。prefix q phase 先对当前 raw batch 做

\[
B_m^q=B_m\cap S_{\mathrm{prefix}}.
\]

长期随机排列下，特别是最后一个不满 batch，完全可能有 $|B_m^q|=0$。旧代码仍对空张量计算平均 prediction loss；空集均值按定义不可计算，因此直接产生 `NaN`。这不是梯度爆炸，也不需要调学习率或 continuity 权重。最小修复是当且仅当 $|B_m^q|=0$ 时跳过该次 q update，同时照常执行该 raw batch 的 theta update；所有非空 q update 完全不变。实现记录 `q_phase_empty_batches_skipped`，并用 batch-size-one 的确定性测试验证 40 次 theta update、4 次有效 q update、36 次空 q 跳过和有限训练损失。

这两个问题必须分开写：epoch normalization 使全局 q 正则的每-epoch 剂量跨数据规模可比；empty-prefix skip 使随机 raw batching 对 support 子集具有合法定义。它们都是训练目标离散化修复，不是根据 Batch2 预测效果选择的新超参数。

公平性仍需分开陈述。prefix alternating 每个 raw batch 有一次 q backward 和一次 theta backward，而 no-q MLP 只有一次 theta backward。epoch normalization 只修复正则语义，不会自动让总计算量相等。论文必须同时报告 theta/q steps、backward passes、examples processed、wall time、参数量和 q-gradient trace，表述为相同 epoch 与相同 theta update budget，而不是相同总算力。

## 12. 从 raw q 到“方程坐标”：为什么 decoder-functional canonicalization 是真正的选规

Starry ZT 的新结果把前面的 gauge 讨论推进了一步。现在需要区分三种对象：

1. 神经网络内部任意坐标 `raw q`；
2. 由 decoder 响应定义的、与坐标重命名无关的函数对象；
3. 把该函数对象投影到可读基以后得到的“方程坐标”。

这三者不是同一个 q。第三种才是论文中可以命名为参考值、敏感度和曲率的 q。

### 12.1 raw q 为什么原则上不能直接命名

设 decoder 为

\[
f_\theta(x,q).
\]

对任意可逆变换 $h$，令

\[
q'=h(q),\qquad
f_{\theta'}(x,q')=f_\theta(x,h^{-1}(q')).
\]

则所有预测完全不变。只要训练目标只看预测、q 距离或其他不完全消除该对称性的量，优化器就可以在这一整族等价参数化中选择任意代表。因而，即使两个 seed 学到相同的实体响应函数，它们的 `q1,q2,q3,q4` 仍可发生旋转、反射、缩放和非线性扭曲。直接对 raw q 做符号回归，等于要求符号方法猜中神经网络偶然选出的内部坐标系。

旧 ZT 资产给出了一个很直观的诊断。在固定 60 个训练实体上用 leave-one-entity-out 选择 ridge，再把四维 raw q 映射为三个二次系数，20 个未见实体 query 的 R² 为 `-0.3335`；同一批未见实体用完全相同的 support 直接重估三系数则为 `0.9677`。这说明可解释结构存在，但没有被 raw q 的逐维坐标稳定暴露出来。

### 12.2 decoder 响应是 gauge-invariant 的对象

选择一组固定 probe $x_1,\ldots,x_m$，定义 decoder response vector

\[
F_\theta(q)=
\begin{bmatrix}
f_\theta(x_1,q)\\
\vdots\\
f_\theta(x_m,q)
\end{bmatrix}.
\]

在上面的联合重参数化下，

\[
F_{\theta'}(h(q))=F_\theta(q).
\]

所以 raw q 虽然改变，decoder 所代表的函数不改变。这就是“functionalization”比直接看 q 更可靠的严格原因：它不是希望优化器恰好学到同一坐标，而是主动把等价坐标都映射到同一个函数对象。

对有实体条件 $c$ 的问题，例如材料组成，probe 写成 $(x_j,c)$。比较不同材料时必须固定 probe 规则，而不是为每个 seed 临时挑一组最有利的点。当前 ZT 桥接实验使用每个未见材料已知的温度区间、41 个等距温度和固定组成；它只使用 query covariates，不读取 query ZT。

### 12.3 方程系数是函数空间里的 canonical q

给定可读基函数

\[
\Phi(x)=\left[\phi_0(x),\ldots,\phi_k(x)\right],
\]

在 probe 上形成满列秩矩阵 $\mathbf\Phi$。将 decoder response 投影到这组基：

\[
a(q)
=\arg\min_a\|F_\theta(q)-\mathbf\Phi a\|_W^2
=(\mathbf\Phi^\top W\mathbf\Phi)^{-1}
\mathbf\Phi^\top W F_\theta(q).
\]

由于 $a(q)$ 只依赖 gauge-invariant 的 $F_\theta(q)$，所以在 decoder 与 q 同时作任意可逆重参数化时，$a$ 保持不变。只要基函数顺序、归一化和 probe 规则固定，$a_0,\ldots,a_k$ 就是跨 seed 可比较的 canonical coordinates。

ZT 的冻结基为

\[
\Phi(T)=[1,\tau,\tau^2],\qquad
\tau=(T-\mu_{\rm train})/\sigma_{\rm train}.
\]

因此：

- $a_0$ 是训练温度中心附近的参考 ZT；
- $a_1$ 是一阶温度敏感度；
- $a_2$ 是温度曲率。

它们是“响应方程坐标”，不是被宣称为唯一微观材料参数。这正好符合用户给出的成功标准：表达式只需有阶段性解释和启发意义，不必恢复最初 raw q 或唯一真定律。

### 12.4 为什么需要同时报告 neural error 和 projection error

令真实 query 响应为 $y$，神经 decoder 为 $f$，其符号投影为 $g=P_\Phi f$。在相同离散范数下，

\[
\|y-g\|
\leq
\underbrace{\|y-f\|}_{\text{neural prediction error}}
+
\underbrace{\|f-P_\Phi f\|}_{\text{symbolic projection error}}.
\]

这个分解给出四种完全不同的诊断：

| neural error | projection error | 含义 |
|---|---|---|
| 小 | 小 | 神经模型学对了，且响应可压缩成该符号结构；桥接成功 |
| 小 | 大 | raw q 有预测信息，但选定公式族太弱 |
| 大 | 小 | decoder 很平滑、很好压缩，但压缩的是错误函数 |
| 大 | 大 | 神经训练与符号结构都不合适 |

因此只报“decoder response 被二次式拟合得很好”是不够的。两 epoch smoke 中 projection R² 已接近 1，但 physical query R² 只有约 0.70；这恰好是第三种情况，且 smoke 分数不能作为科学结果。正式实验必须同时报告 raw decoder 的物理 R²、二次投影的物理 R²和 decoder-response reconstruction R²。

### 12.5 canonical q 的稳定性由 probe 设计条件数控制

若 decoder response 扰动为 $\delta F$，则系数扰动满足

\[
\|\delta a\|
\leq
\left\|(\mathbf\Phi^\top W\mathbf\Phi)^{-1}
\mathbf\Phi^\top W\right\|\,\|\delta F\|.
\]

所以 functionalization 并不会自动保证稳定；稳定性取决于两点：decoder 函数是否跨 seed 稳定，以及 probe 上的基矩阵是否良态。温度点若全部挤在很窄范围，$1,\tau,\tau^2$ 近共线，曲率会极不稳定。当前实验使用覆盖整个已知温区的 41 点网格，并以 outer-train 温度均值和标准差定义 $\tau$，就是为了控制这一放大因子。

论文中应同时给出：

- raw q 未对齐坐标稳定性；
- raw q 的距离几何稳定性；
- functional coefficient 的逐维稳定性；
- functional coefficient 的距离几何稳定性；
- 每个 fold 的 decoder projection fidelity。

若 functional coefficient 只在逐维上稳定、距离几何不稳定，说明命名变好了但实体关系仍不可靠；若两者都稳定，才支持“canonical q 是可比较科学坐标”。

### 12.6 support re-q 是结构确认，不是绕过神经模型

decoder-functional 系数回答“神经模型内部学到了怎样的响应形状”；support re-q 则在结构被冻结后，用未见实体的真实 support 重新估计同名系数：

\[
\hat a_S=\arg\min_a\|y_S-\Phi_S a\|^2.
\]

二者承担不同角色：前者发现并验证结构接口，后者消除 raw-q gauge 和 calibration manifold mismatch，把最终 q 放回物理可读坐标。若只做 support 二次拟合，方法会显得像普通插值；若只做 decoder 投影，又可能忠实解释一个预测不准的神经函数。完整闭环必须展示：

\[
\text{raw neural q}
\rightarrow
\text{decoder response}
\rightarrow
\text{shared symbolic basis}
\rightarrow
\text{support re-q}
\rightarrow
\text{held-out physical prediction}.
\]

当前 ZT 外部确认已经证明最后两步跨时间、跨论文和跨组成成立。5-fold × 3-seed 神经桥接也给出了有边界的前三步证据：raw q 直接映射二次系数的 R² 为 `-1.907461`，decoder-functional 二次式为 `0.944683`，decoder 响应投影 fidelity 最低 `0.985033`。所以函数商空间选规在 pooled 意义下成立；但原始绝对 MSE 版本有 19/80 个实体超过 structure re-q 十倍 NRMSE，不能写成完整、无尾部的桥接成功。

准确的论文主线因此是：**任意 latent task coordinate 可以先在 decoder 所表示的函数商空间中选规，再变成可命名、可重估、可外部确认的 equation coordinates；这种选规解决可读性问题，但最终实体稳健性仍受神经训练目标影响。**

## 13. 为什么 target scale 会改变 latent-q 的训练动力

### 13.1 label-balanced 不等于误差尺度平衡

原模型使用 label-balanced MSE。它保证不同材料以接近的频率进入梯度，但每个材料贡献的预测梯度仍与物理残差大小成正比。设标准化前残差为

\[
\delta y=\hat y-y.
\]

对 ZT 约为 `10^-3` 的曲线，预测成 `10^-2` 在全局物理范围里仍是小绝对误差，却可能比该材料自身变化尺度大几十倍。因此“每个 label 被同样频繁采样”和“每个 label 的相对曲线形状被同样重视”是两件不同的事。

这解释了第一版桥接看似矛盾的结果：pooled R² 为 `0.944683`，但单实体 R² 中位数只有 `0.846748`，19/80 个材料触发十倍尾部。大尺度材料主导 pooled 平方和，小尺度材料在优化与 pooled 指标里都容易被淹没。

### 13.2 asinh 变换对应什么局部误差权重

尺度修复使用 outer-train-only 的

\[
z=\operatorname{asinh}(y/s),
\]

其中 $s$ 是训练实体曲线标准差的中位数。它可逆，允许负数和零值，并且

\[
\frac{\mathrm dz}{\mathrm dy}
=\frac{1}{\sqrt{s^2+y^2}}.
\]

当预测误差较小时，一阶展开给出

\[
(\delta z)^2
\approx
\frac{(\delta y)^2}{s^2+y^2}.
\]

因此它在训练动力上近似一种连续的相对误差加权：

- 当 $|y|\ll s$，权重约为 $1/s^2$，近零曲线不会失去梯度；
- 当 $|y|\gg s$，权重约为 $1/y^2$，大幅值曲线的绝对误差不再完全支配优化；
- 在 $|y|\approx s$ 附近平滑过渡，没有对零值做奇异除法。

这不是仅改变打印单位。模型在 $z$ 空间训练、测试 support 也在 $z$ 空间校准，所以 prediction gradient 与 q-prior 的相对作用都发生了变化；输出只在 decoder 预测完成后逆变换回物理 ZT。因而必须把它称为一个训练目标消融，而不是无害的数据标准化。

### 13.3 实验是否符合这个动力学预测

符合，而且效应方向很集中：

| 指标 | 绝对 MSE | asinh scale-aware |
|---|---:|---:|
| functional degree-2 pooled R² | 0.944683 | 0.942488 |
| 单实体 R² 中位数 | 0.846748 | 0.940261 |
| 单体 R²≥0.85 | 40/80 | 52/80 |
| 十倍尾部实体 | 19/80 | 9/80 |
| 最坏倍率 | 744.872 | 139.163 |
| target std 与 functional NRMSE Spearman | -0.614885 | -0.193718 |

尺度偏差相关性大幅减弱，59/80 个实体 NRMSE 改善，说明原诊断不只是事后故事。functional 距离几何的跨 seed 中位 Spearman 也从 `0.738385` 提到 `0.835419`，超过新模型 raw q 的 `0.748142`。一种合理的动力学解释是：低尺度实体终于对共享 $\theta$ 和实体 q 产生足够梯度，使不同 seed 不再主要由大幅值曲线决定同一个函数流形。

### 13.4 为什么仍不能继续无限压缩

严格十倍尾部仍有 9/80 未过，其中最极端的近零材料虽显著改善，仍因 structure re-q 几乎无误差而保留巨大相对倍率；同时两个正常幅值材料 `6363` 和 `2100` 由原先不过十倍变成超过十倍。这正是上式的另一面：更强的对数式压缩可能继续照顾小 $y$，却进一步降低大 $y$ 区域的梯度。

所以当前证据不授权盲目把 $s$ 调得更小。若未来专门追求 all-entity tail，合理方向是用训练实体内部交叉验证冻结一个异方差损失或 support-only 专家选择器，并在新的实体 cohort 上确认；不能根据这 80 个 query 的尾部继续挑阈值。对当前论文的必要表达式端点也没有这个需要：可解释二次式在严格实体外推上的 pooled R² 已远高于 `0.85`，原始 q 恢复与十倍最坏尾部都是更强、独立的要求。
