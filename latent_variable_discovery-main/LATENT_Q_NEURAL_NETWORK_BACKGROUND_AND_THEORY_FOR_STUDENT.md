# 隐变量神经网络：背景、问题定义与理论基础

更新时间：2026-07-15

用途：给参与项目的学生做第一轮背景介绍。本材料只讨论前半段“隐变量神经网络”如何从成组响应数据中学习低维潜变量，暂不展开后续符号回归、工程运行命令和实验结果。

---

## 1. 一句话说明这个工作

对于同一类实验对象，我们往往观测到一族响应曲线。每条曲线都有显式输入 `x` 和响应 `y`，但曲线之间的差异还受到某些没有记录下来的材料性质、样品状态或实验条件影响。

本项目为每个材料或实验对象学习一个低维连续向量 `q`，再用共享神经网络

```text
y_hat = f_theta(x, q)
```

描述整族曲线。希望 `q` 不仅提高预测精度，还尽量满足三个性质：

1. 对响应有用，即没有 `q` 时无法解释的曲线差异可以由 `q` 解释。
2. 不只是样品 ID 或显变量采样方式的替身。
3. 在几何上连续，即响应函数相似的对象具有相近的 `q`。

更准确的名称是“潜在描述符学习”，而不是无条件地声称“恢复了唯一真实物理量”。

---

## 2. 科学背景：为什么实际数据需要隐藏变量

### 2.1 普通监督学习的隐含假设

普通回归通常写成：

```text
y = f(x) + noise
```

它假设给定显变量 `x` 后，目标 `y` 的系统性变化已经基本确定。但科学实验中经常不是这样。

例如，在相同温度下，不同热电材料样品具有不同的 Seebeck 系数；在相同循环数下，不同电池具有不同容量；在相同浓度下，不同传感器批次具有不同响应。原因可能包括：

- 未完整记录的成分、微结构和缺陷；
- 合成与加工历史；
- 样品质量和老化状态；
- 仪器漂移、批次差异和环境条件；
- 已知存在但难以直接测量的有效物理参数。

因此更合理的生成关系是：

```text
y = g(x, u) + noise
```

其中 `u` 是真实但未观测到的对象级内在状态。项目中的 `q` 是通过数据学习得到的 `u` 的低维代理。

### 2.2 为什么需要“每个对象一条曲线”

如果每个对象只有一个 `(x, y)` 点，很难判断差异来自对象本身还是随机噪声。若同一对象在多个 `x` 上有重复观测，则整条响应曲线为隐藏状态提供了更丰富的约束。

直观上，一条曲线是对象的“功能指纹”。我们用少量维度的 `q` 压缩这种功能差异，再让一个共享网络解释所有对象。

### 2.3 典型应用形式

数据通常具有如下结构：

```text
label, x1, x2, ..., xd, y
```

其中：

- `label`：材料、样品、电池、发动机、实验批次或一条曲线的标识；
- `x`：可以测量或控制的显变量；
- `y`：响应值；
- 同一 `label` 下有多组 `(x, y)`。

必须强调：`label` 不作为数值变量输入网络。它只在训练时用于查找该对象对应的 `q_label`。

---

## 3. 问题的数学定义

设共有 `L` 个训练对象。第 `l` 个对象的数据为：

```text
D_l = {(x_li, y_li)}_{i=1}^{n_l}
```

其中 `x_li in R^d`，目标为标量 `y_li`。为每个对象引入：

```text
q_l in R^k
```

共享预测模型为：

```text
y_hat_li = f_theta([x_li, q_l])
```

其中：

- `theta` 是所有对象共享的网络参数；
- `q_l` 是对象特有的低维潜在描述符；
- `k = q_dim` 是潜变量维度。

训练阶段联合优化：

```text
theta*, Q* = argmin_{theta,Q} L_train
```

其中：

```text
Q = [q_1, q_2, ..., q_L]^T
```

当前 Torch 实现使用一个 `nn.Embedding(L, k)` 保存训练对象的 `Q`，并把网络参数与 embedding 参数一起用 Adam 更新。

---

## 4. 生成视角：内禀因素与采样机制要分开

为了说明为什么需要正交或独立性约束，可以使用如下生成模型：

```text
u_l  = 对象的隐藏内禀状态
a_l  = 实验采样或采集策略
x_li ~ P(x | a_l)
y_li = g(x_li, u_l) + epsilon_li
q_l  ~= h(u_l)
```

这里存在两个不同的 label-level 因素：

- `u_l` 决定对象如何响应，是希望 `q` 表示的内容；
- `a_l` 决定实验在哪些 `x` 上采样，是不希望 `q` 偷偷编码的 nuisance factor。

例如，某些材料只在高温区测量。如果没有约束，模型可能用 `q` 记住“这个 label 的温度都偏高”，而不是学习材料性质。

因此项目的独立性目标不是简单要求样本级 `q_l` 与 `x_li` 独立，而是要求：

```text
q_l 尽量不依赖该对象的显变量采样分布 P_l^X
```

先把每个对象的采样分布表示成向量：

```text
A_l = Phi(P_l^X)
```

再惩罚 `Q` 与 `A` 的统计依赖。这是 Sampling-Invariant Latent Descriptor Learning（SILoD）叙事的核心。

---

## 5. 网络结构与训练机制

### 5.1 共享预测器

当前主模型是前馈 MLP：

```text
[x, q] -> Linear -> ReLU -> Linear -> ReLU -> Linear -> y_hat
```

实际主实验常用隐藏层宽度为 `256, 128`。输入维度为 `d + k`，输出一个标量。

`x` 和 `y` 都使用训练集统计量进行标准化；测试集沿用训练集的均值和标准差，避免测试信息泄露。

### 5.2 训练对象的 q

训练开始时，每个 `q_l` 从小尺度高斯分布初始化：

```text
q_l ~ N(0, 0.1^2 I)
```

每个 mini-batch 根据 `label` 查出对应的 `q_l`，拼接到 `x` 后送入网络。预测误差对网络和 `q_l` 同时反向传播。

因此 `q_l` 不是网络直接输出的，也没有真实 q 标签；它是一个通过整条曲线的预测任务反推出来的对象级参数。这类做法与 auto-decoder、矩阵分解和 task embedding 有相似之处。

### 5.3 新对象的 few-shot 校准

测试对象在训练时没有对应的 embedding。对每个新 label：

1. 将该对象的数据分成 calibration/support 集与 evaluation/query 集；
2. 冻结共享网络参数 `theta`；
3. 初始化一个新的 `q_test`；
4. 只使用 calibration 样本优化 `q_test`；
5. 在未参与校准的 query 样本上报告 R2/MSE。

形式化地：

```text
q_test* = argmin_q L_calibration(q; C_test, theta*)
```

随后：

```text
y_hat = f_theta*(x, q_test*)  for (x, y) in Q_test
```

这模拟了实际应用：面对新材料或新设备，先测少量点识别其隐藏状态，再预测其余条件下的完整响应。

---

## 6. 总损失函数

当前方法可概括为：

```text
L_train = L_pred
        + lambda_indep * L_indep(Q, A)
        + lambda_cont  * L_cont(Q, D_curve)
        + beta         * L_q_l2
        + optional symbolic-friendly regularizers
```

测试校准阶段为：

```text
L_cal = L_cal_pred
      + lambda_prior * L_q_prior
```

下面分别解释每一项。

---

## 7. 预测损失：保证 q 对任务有用

最基础的损失为：

```text
L_mse = (1/N) * sum_i (y_hat_i - y_i)^2
```

它迫使 `q` 与网络共同解释响应。如果某个 q 维度对预测没有贡献，它就没有“预测充分性”。

当不同 label 的样本数差异很大时，普通样本级 MSE 会让点数多的曲线主导训练。当前还实现了 label-balanced MSE：

```text
L_balanced = (1/L_batch) * sum_l [ (1/n_l) * sum_{i in l} (y_hat_li-y_li)^2 ]
```

它先在每个 label 内平均，再对 label 平均，使每条曲线的权重更接近一致。

注意：当前实现是在每个 mini-batch 内对出现的 label 做平衡，并不严格等价于整个 epoch 上的全局 label-balanced objective。

---

## 8. 采样分布表示 A_l

正交约束作用于 `q_l` 和 label-level 采样分布表示 `A_l`，而不是直接作用于每个样本的 `x_li`。

### 8.1 mean_std

最简单的表示是：

```text
A_l = [mean(X_l), std(X_l)]
```

它容易解释，但只能描述一阶和二阶边际统计。

### 8.2 rich statistics

当前 `rich` 模式包含：

```text
mean, std, min, max, range,
5%/25%/50%/75%/95% quantiles,
covariance upper triangle,
log(1 + sample_count)
```

它可以感知采样窗口、偏态、多维相关结构和不同对象的样本数量。

### 8.3 RFF kernel mean embedding

Kernel mean embedding 用核特征均值表示一个分布：

```text
mu_P = E_{x~P}[phi(x)]
```

RBF 核对应的特征空间通常是无限维，因此使用 Random Fourier Features 近似：

```text
z(x) = sqrt(2/m) * cos(W^T x + b)
A_l^RFF = (1/n_l) * sum_i z(x_li)
```

当前实现使用三个尺度 `0.5, 1.0, 2.0`，每个尺度 32 个随机特征，并固定随机种子。多尺度的目的，是同时感知粗粒度和细粒度的分布差异。

### 8.4 rich_rff_kme

主候选表示为：

```text
A_l = concat(A_l^rich, A_l^RFF)
```

它把可解释统计量与非参数分布特征结合起来。`rich_rff_kme` 不是一个独立性 loss，而是独立性 loss 所使用的采样分布表示。

---

## 9. 独立性与正交策略

### 9.1 Pearson 去相关

将 `Q` 与 `A` 的各列标准化，计算交叉相关矩阵：

```text
C = Q_std^T A_std / L
L_pearson = mean(C^2)
```

优点是稳定、直观；缺点是只能检测线性依赖。零相关不等于独立。

### 9.2 HSIC / normalized kernel alignment

对 `Q` 和 `A` 分别构造 RBF 核矩阵：

```text
K_ij = k(q_i, q_j)
G_ij = k(a_i, a_j)
```

经过中心化后，比较两个核矩阵的对齐程度：

```text
L_HSIC = <HKH, HGH>_F /
         sqrt(||HKH||_F^2 * ||HGH||_F^2)
```

当前代码中的 `hsic` 和 `nhsic` 是同一个归一化实现，更准确地说是 normalized HSIC / centered kernel alignment。RBF 带宽使用 label 间距离的 median heuristic。

它能检测非线性依赖，但估计质量依赖 label 数量和核带宽。

### 9.3 Distance correlation

分别计算 `Q` 和 `A` 的 pairwise distance matrix，双中心化后比较：

```text
L_dCor = <D_Q_centered, D_A_centered> /
         sqrt(||D_Q_centered||^2 * ||D_A_centered||^2)
```

它不要求显式选择特征方向，同样可以检测广泛的非线性依赖。有限样本下可能存在偏差，并且对异常点和 label 数量敏感。

### 9.4 Adversarial predictor

训练一个小网络 `h_phi(q)` 尝试从 `q` 预测采样分布表示 `A`：

```text
phi 试图最小化 ||h_phi(q)-A||^2
q   试图让该预测变困难
```

当前实现交替训练 adversary，并使用 `exp(-MSE_adversary)` 作为 embedding 一侧的惩罚。直觉上，如果 `A` 能从 `q` 被准确预测，说明 `q` 泄露了采样机制。

这种方法表达能力强，但优化是一个 min-max 问题，稳定性通常弱于 Pearson、HSIC 和 distance correlation。

### 9.5 Propensity-style weighting

当前 `propensity` 策略先在 `A` 空间估计每个 label 的局部密度，再给稀疏区域较大权重，最后计算加权相关损失。

它应被称为 propensity-style 或 inverse-density correction，而不是严格的因果 propensity score；当前实现没有显式处理处理组或干预概率。

---

## 10. 函数几何连续性

仅靠预测损失，两个相似对象的 q 仍可能相距很远。连续性约束希望：

```text
响应函数相似  =>  q 相近
响应函数差异大 => q 更远
```

当前实现先为每个 label 建立一条响应 profile：

1. 只取第一个显变量作为主坐标；
2. 在全局范围建立公共网格；
3. 对每个 label 的曲线排序、合并重复坐标并线性插值；
4. 每条 profile 去均值并按 RMS 归一化；
5. 计算 label 间 profile 的欧氏距离 `D_curve`；
6. 计算 q 的欧氏距离 `D_q`；
7. 对两个距离矩阵的非对角元素标准化并做均方匹配。

公式为：

```text
L_cont = mean_{l != m} [normalize(||q_l-q_m||_2)
                         - normalize(D_curve(l,m))]^2
```

从理论上看，它把函数空间中的相似性关系蒸馏到低维潜变量空间中，类似一种结构保持的度量学习或多维尺度分析。

当前实现有两个重要边界：

- 只沿第一个显变量构造 profile，多显变量响应面尚未被完整表示；
- profile 去均值并归一化后，连续性主要比较曲线形状，不直接保留绝对幅值差异。

因此多维真实数据上不能把当前 `L_cont` 描述成完整响应面几何约束。

---

## 11. q 的尺度、维度与可分解性

### 11.1 L2 正则

```text
L_q_l2 = mean(||q_l||_2^2)
```

它限制 q 的尺度，减少 embedding 为了补偿网络权重而无界增大，也对测试校准更友好。

### 11.2 Whitening

Whitening loss 约束：

```text
mean(Q) ~= 0
std(Q_j) ~= 1
Cov(Q) ~= I
```

它改善不同 q 维度的尺度和冗余问题，但不保证每一维对应唯一物理因素。

### 11.3 Jacobian disentanglement

对每个样本计算：

```text
J_q = partial f(x,q) / partial q
```

并惩罚不同 q 维度对应梯度方向之间的相关性。目标是让不同 q 维度以不同方式影响输出。

### 11.4 q 方向平滑性

使用有限差分近似输出对 q 的二阶导数：

```text
d2f/dq_j2 ~= [f(x,q+eps*e_j)-2f(x,q)+f(x,q-eps*e_j)] / eps^2
```

惩罚过大曲率，可让网络关于 q 的关系更平滑，也更利于后续符号表达式拟合。

### 11.5 Canonicalization

当前 canonicalization 通过中心化和 whitening 固定 q 的位置与尺度。它能改善跨运行比较，但仍不能完全消除旋转、符号翻转和维度置换的不确定性。

这些项属于“让多维 q 更规整、更适合解释和符号回归”的增强项，不应与采样不变性的核心独立性约束混为一谈。

---

## 12. 测试时经验先验

新对象只有少量 calibration 点，若只优化预测误差，`q_test` 可能跑到训练 q 分布之外并过拟合少量观测。

当前实现从训练 embedding 估计逐维均值和标准差：

```text
mu_q = mean_l(q_l)
sigma_q = std_l(q_l), lower bounded by 0.05
```

校准损失加入：

```text
L_q_prior = mean_j [((q_test_j-mu_q_j)/sigma_q_j)^2]
```

这等价于使用一个对角高斯经验先验的负对数项，可以理解为轻量的 empirical Bayes / MAP calibration：

- calibration 数据决定新对象应落在哪里；
- 训练 q 分布防止它因数据太少而极端漂移。

`lambda_prior` 太小会过拟合 calibration 点；太大则会把所有新对象拉向训练均值，降低个体差异表达能力。

---

## 13. 为什么不能直接把 label ID 当成一个输入变量

把样品编号设成 `1,2,3,...` 后直接送入网络，会产生三个问题：

1. 数字距离没有物理意义，ID 2 并不比 ID 10 更接近 ID 1；
2. 网络可以记忆训练对象，但无法为未见对象给出合理编号；
3. 后续若公式出现 `sin(sample_id)` 或 `1/sample_id`，预测上可能有效，物理上不可解释。

本项目中的 label embedding 与直接输入 ID 的区别是：

- label 只负责索引待学习的 q，不参与连续代数；
- q 是通过整个对象的响应数据反推的；
- 新对象的 q 可用少量观测重新校准；
- q 受到独立性、连续性、尺度和先验约束。

但必须承认：若只有预测损失、q 维度过大且没有 unseen-label 测试，q embedding 仍可能退化成软性的 label lookup code。因此约束和评估协议不可缺少。

---

## 14. 与几类相关方法的区别

### 14.1 普通 MLP

普通 MLP 只学 `y=f(x)`，无法解释同一 `x` 下不同对象的系统性差异。它应作为最基本的 no-q baseline。

### 14.2 直接 ID embedding

训练形式可能与本项目很像，但通常只追求训练对象预测，没有 sampling invariance、函数几何和 unseen-label calibration。我们的重点是把 embedding 变成受约束、可迁移的对象级描述符。

### 14.3 VAE

VAE 通常训练 encoder `q_phi(z|data)` 和 decoder，并通过概率先验构造生成模型。当前方法没有 amortized encoder；每个训练 q 直接优化，测试 q 也通过梯度校准。因此当前方法更接近 auto-decoder，而不是标准 VAE。

### 14.4 Neural Process / Set Encoder

Neural Process 类方法把一组 context `(x,y)` 编码成对象表示，再预测 query。它可以一次前向推理得到 q，测试速度更快。当前梯度校准方法更简单、直接，但校准成本较高。未来可加入 set encoder 作为基线或扩展。

### 14.5 Meta-learning

MAML 等方法学习“如何快速更新模型参数”。当前方法只让新对象更新 q，不更新共享网络，因此适配空间更小、更稳定，也更适合把 q 作为下游待解释变量。

### 14.6 矩阵分解

若 `x` 是固定离散网格，问题可看成对象-条件响应矩阵的低秩分解。神经网络版本允许连续、多维且不规则的 x，并用非线性函数组合 x 与 q。

---

## 15. 可识别性：最需要讲清楚的理论边界

假设已经学到：

```text
y = f(x,q)
```

对任意可逆变换 `T`，可以定义：

```text
q' = T(q)
f'(x,q') = f(x,T^{-1}(q'))
```

两者给出完全相同的预测。因此只靠重构误差，q 在旋转、缩放、符号、置换乃至一般可逆变换下都可能不唯一。

独立性、连续性、whitening、平滑性和低维瓶颈会缩小这种等价解空间，但通常不能证明 q 等于唯一真实物理量。无监督 disentanglement 本身也存在类似不可识别性问题。

论文和汇报中可以严格声称：

```text
我们学习了对响应具有预测充分性、减少采样分布泄露、并保留函数相似性几何的低维对象级潜在描述符。
```

不能仅凭高 R2 声称：

```text
模型唯一恢复了真实的、因果的、可直接命名的物理内禀变量。
```

若要更接近“发现真实变量”，还需要：

- 在合成数据上有真实 q 并验证可恢复性；
- 检查 q 与已知但未参与训练的物理量之间的关系；
- 做跨采样策略、跨批次和跨域验证；
- 验证多次随机初始化后 q 的稳定性；
- 控制所有已知物理输入和数据泄露来源。

---

## 16. 评估协议为什么比训练 R2 更重要

一个可信实验至少应满足：

1. 按 label 划分训练和测试，不能把同一条曲线的相邻点分到两边；
2. 测试 label 在训练时不可见；
3. calibration 只使用 support target，query target 不参与 q 优化；
4. label/ID 不作为数值输入；
5. 显变量必须在任务设定的预测时刻可获得；
6. 同周期目标代理、未来信息和由 target 计算出的特征必须排除；
7. 报告不同 calibration ratio、q_dim 和随机种子；
8. 除 R2 外，报告 q 对采样分布的泄露指标和函数几何保持指标；
9. 与 no-q、普通 ID embedding、无正交约束等基线比较。

当前代码的一个具体注意点：每个测试 label 的 calibration 集取该 label 数据顺序中的前 `calibration_ratio` 部分，而不是随机抽样。如果数据按时间、温度或循环数排序，这相当于“前段观测预测后段”，是合理但更困难的外推协议；若论文口径是随机 few-shot，则需要显式修改和对照，不能混用两种说法。

---

## 17. 学生需要先掌握的几个判断

### 判断一：q 是对象级变量，不是样本级变量

同一 label 的所有 `(x,y)` 共享同一个 q。q 表示对象或曲线，不表示单个数据点。

### 判断二：高预测 R2 不等于发现了物理量

高 R2 只说明 `(x,q)` 足以拟合响应。还必须检查 q 是否编码 ID、采样范围、批次或泄漏特征。

### 判断三：正交针对的是采样分布表示 A_l

不是把所有样本的 q 和 x 粗暴做点级相关，而是比较 label-level 的 q 与 label-level acquisition embedding。

### 判断四：连续性针对整条函数

不是要求相邻样本点 q 接近，因为同一对象本来就共享 q；而是要求相似曲线对应相似对象描述符。

### 判断五：多维 q 不天然可解释

qdim 增大通常提高容量，但也更容易记忆 label。每个维度是否独立、稳定、可命名，需要专门实验支持。

---

## 18. 给学生介绍时可直接使用的三分钟口径

> 我们研究的是一族科学响应曲线。显变量 x 是温度、循环数、浓度等已知条件，y 是物性或响应。同一个材料或实验对象有多个 x-y 点，但对象之间还有没有记录下来的差异，所以只学 y=f(x) 不够。
>
> 我们给每个对象一个低维向量 q，学习共享模型 y=f(x,q)。训练时网络参数和每个训练对象的 q 一起优化；遇到新对象时冻结网络，只用少量观测校准新的 q，再预测剩余曲线。
>
> 难点不是让 q 提高拟合，而是防止 q 变成样品编号或采样范围的代码。因此我们把每个对象的 x 采样分布表示为 A，用 Pearson、HSIC 或 distance correlation 约束 q 与 A 的依赖；同时让响应曲线相似的对象在 q 空间中相近。测试校准还加入训练 q 的经验先验，避免少量点把 q 推到异常区域。
>
> 我们把 q 称为潜在描述符，而不是直接声称它是真实物理量。最终要靠合成真值、外部物理变量、跨采样验证和稳定性实验，判断它是否真的具有科学意义。

---

## 19. 建议先读的相关思想

以下文献不是说它们与本项目完全相同，而是帮助理解项目所处的方法背景：

- Park et al., *DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation*, CVPR 2019：理解直接优化对象级 latent code 的 auto-decoder 思想。
- Garnelo et al., *Conditional Neural Processes*, ICML 2018：理解从 context observations 表示一条函数或一个任务。
- Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks*, ICML 2017：理解少量样本下的快速适配。
- Gretton et al., *Measuring Statistical Dependence with Hilbert-Schmidt Norms*, ALT 2005：HSIC 的基础。
- Szekely, Rizzo, and Bakirov, *Measuring and Testing Dependence by Correlation of Distances*, Annals of Statistics 2007：distance correlation 的基础。
- Rahimi and Recht, *Random Features for Large-Scale Kernel Machines*, NeurIPS 2007：Random Fourier Features。
- Muandet et al., *Kernel Mean Embedding of Distributions: A Review and Beyond*, Foundations and Trends in Machine Learning 2017：分布核均值嵌入。
- Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*, ICML 2019：理解无监督潜变量可识别性的限制。

---

## 20. 当前代码对应关系

核心实现：

```text
latent_q_pipeline.py
q_optimize_torch.py
scripts/run_application_latent_q.py
```

代码中的主要参数与本文符号对应：

```text
q_dim                                      -> k
prediction_loss_type                       -> L_pred
latent_feature_orthogonality_weight        -> lambda_indep
latent_feature_orthogonality_type          -> dependence estimator
latent_feature_stats_mode                  -> Phi(P_l^X)
latent_curve_continuity_weight             -> lambda_cont
latent_q_l2_weight                         -> beta
calibration_ratio                          -> support/query split
calibration_q_prior_weight                 -> lambda_prior
latent_q_whitening_weight                  -> whitening regularizer
latent_jacobian_disentanglement_weight     -> Jacobian disentanglement
latent_q_smoothness_weight                 -> q-direction curvature penalty
latent_q_canonicalization_mode             -> q coordinate canonicalization
```

读代码时建议按如下顺序：

```text
数据读取与标准化
-> train_latent_q_model
-> label embedding + shared MLP
-> total training loss
-> calibrate_latent_q_for_test_labels
-> support/query evaluation
-> train_with_q.csv / test_with_q.csv
```

这份材料中的数学解释应以当前代码为准；后续若修改 calibration split、多维连续性或 q inference encoder，需要同步更新本文档。
