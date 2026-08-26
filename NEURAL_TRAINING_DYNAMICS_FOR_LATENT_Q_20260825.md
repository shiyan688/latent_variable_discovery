# 从神经网络训练动力学理解本项目的隐变量 q

**日期：** 2026-08-25

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
