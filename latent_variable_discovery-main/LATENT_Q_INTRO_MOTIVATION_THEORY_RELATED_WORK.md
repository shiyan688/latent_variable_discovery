# 隐变量神经网络介绍材料

副标题：研究动机、理论原理、相关工作与当前实现

更新时间：2026-07-15

本文用于向新参与项目的学生介绍前半段“隐变量神经网络”。不包含实验结果、运行命令和后续符号回归细节。

---

## 1. 我们想解决什么问题

很多科学数据不是普通的独立样本，而是一族由不同材料、样品或实验对象产生的响应曲线。

对第 `l` 个对象，我们有多组观测：

```text
D_l = {(x_li, y_li)}_{i=1}^{n_l}
```

其中 `x` 是温度、循环数、浓度、压力或实验协议等显变量，`y` 是材料性质或系统响应。

普通回归假设：

```text
y = f(x) + noise
```

但现实中，即使 `x` 相同，不同对象的 `y` 也可能系统性不同。这些差异可能来自没有记录或难以直接测量的因素，例如微结构、缺陷、合成历史、老化状态、批次漂移或有效物理参数。

更合理的生成模型是：

```text
y = g(x, u) + noise
```

其中 `u` 是对象级隐藏状态。我们的目标是从每个对象的整条响应曲线中学习一个低维连续代理 `q`：

```text
y_hat = f_theta(x, q)
```

这里的 `q` 称为 latent descriptor，即潜在描述符。

---

## 2. 为什么这个问题值得做

### 2.1 预测层面的动机

如果只使用显变量 `x`，模型会把同一 `x` 下不同对象的响应平均掉。引入对象级 `q` 后，共享网络可以学习一族条件响应函数：

```text
f_theta(x, q_1), f_theta(x, q_2), ..., f_theta(x, q_L)
```

`q` 用较少维度解释对象之间的系统性差异。

### 2.2 科学层面的动机

科学上更关心的不只是拟合，而是对象之间是否存在低维、连续的隐藏控制因素。如果相似响应曲线能被组织到相近的 q 区域，q 就可能成为：

- 未观测材料性质的代理；
- 样品状态或退化程度的描述符；
- 曲线族中的低维控制坐标；
- 后续可解释建模和符号回归的候选变量。

### 2.3 新对象少样本适配的动机

面对训练时未见过的新材料或新设备，我们通常只能先测少量点。理想方法应利用少量 calibration/support 数据识别新对象的 q，再预测其余响应，而不是为每个新对象重新训练整个网络。

---

## 3. 核心模型原理

为每个训练 label 学习：

```text
q_l in R^k
```

共享预测器为：

```text
y_hat_li = f_theta([x_li, q_l])
```

训练时联合优化共享网络参数和所有训练对象的 q：

```text
theta*, Q* = argmin_{theta,Q} L_train
```

其中：

```text
Q = [q_1, ..., q_L]^T
```

它可以理解为一种非线性函数分解：

- `theta` 学习所有对象共享的响应规律；
- `q_l` 学习第 `l` 个对象相对于共享规律的低维变化。

与把 label 编号直接作为输入不同，label 在这里只用于索引 q。编号 `1,2,3` 本身不参与乘除、距离或三角函数等连续运算。

---

## 4. 为什么不能只用“预测误差 + label embedding”

如果只最小化 MSE，并且 q 维度和网络容量足够大，q 可能退化成对象身份的记忆代码。模型虽然拟合得好，但 q 未必有科学意义。

我们希望 q 同时具有三种性质：

### 4.1 Predictive sufficiency

q 必须包含对 y 有用、但 x 中没有的信息。预测损失负责这一点。

### 4.2 Acquisition invariance

q 不应主要编码该对象在哪些 x 上被测量，即不应成为采样范围、采样密度或实验窗口的替身。

### 4.3 Functional continuity

响应函数相似的对象应具有相近的 q，使 latent space 保留曲线族的几何结构。

因此这个工作的核心不是“给网络加一个 embedding”，而是学习一个受预测性、采样不变性和函数连续性共同约束的对象级描述符。

---

## 5. 采样不变性的理论动机

把对象级内在状态和采样机制区分开：

```text
u_l  = 对象的隐藏内在状态
a_l  = 数据采集或实验采样策略
x_li ~ P(x | a_l)
y_li = g(x_li, u_l) + epsilon_li
```

我们希望：

```text
q_l 主要表示 u_l，而不是 a_l
```

由于 `a_l` 通常没有直接标签，我们用对象的显变量采样分布表示它：

```text
P_l^X = distribution of x under label l
A_l   = Phi(P_l^X)
```

然后约束：

```text
q_l 尽量独立于 A_l
```

注意，目标不是简单要求样本级 `q_l` 与 `x_li` 独立。q 是对象级常量，x 是曲线上的查询坐标；真正需要抑制的是 q 对 label-level acquisition distribution 的泄露。

---

## 6. 当前总体损失

当前实现可写为：

```text
L_train = L_pred
        + lambda_indep * L_indep(Q, A)
        + lambda_cont  * L_cont(Q, D_curve)
        + beta         * ||Q||^2
        + optional regularizers
```

新对象校准时使用：

```text
L_cal = L_support_prediction
      + lambda_prior * L_q_prior
```

各项含义如下。

### 6.1 Prediction loss

基本项是 MSE。样本数不均衡时使用 label-balanced MSE：先在每个 label 内平均，再对 mini-batch 中出现的 label 平均，降低长曲线对训练的支配。

### 6.2 Acquisition representation A

当前有四种对象级采样分布表示：

- `mean_std`：均值和标准差；
- `rich`：均值、标准差、最值、范围、多个分位数、协方差和样本数；
- `rff_kme`：多尺度 Random Fourier Features 的对象内均值；
- `rich_rff_kme`：rich statistics 与 RFF kernel mean embedding 拼接。

`rich_rff_kme` 的目的，是同时保留透明统计特征和对复杂分布形状的非参数近似。

### 6.3 Dependence penalty

当前实现支持：

- Pearson：惩罚 Q 和 A 的线性交叉相关；
- HSIC/nHSIC：使用 RBF 核和中心化核对齐检测非线性依赖；
- distance correlation：比较双中心化距离矩阵；
- adversarial：训练小网络从 q 预测 A，同时让 q 使该预测变困难；
- propensity-style：按 A 空间的逆局部密度加权后计算相关性。

其中当前代码里的 `hsic` 与 `nhsic` 是同一个归一化核对齐实现；`propensity` 是 inverse-density correction，不是严格因果意义的 propensity score。

### 6.4 Function-geometry continuity

当前实现把每个 label 的响应曲线插值到第一显变量的公共网格上，计算 label 间曲线距离 `D_curve`，再匹配 q 的 pairwise distance：

```text
L_cont = mean_{l != m} [normalize(||q_l-q_m||)
                         - normalize(D_curve(l,m))]^2
```

这相当于把函数空间几何蒸馏到低维 q 空间。

当前版本只沿第一个显变量建立 profile，并对 profile 去均值和 RMS 归一化，因此主要保留曲线形状，不是完整的多维响应面距离。

### 6.5 q L2 regularization

限制训练 q 的尺度，减少 embedding 无界增长，并让新对象校准更稳定。

### 6.6 Optional regularizers

当前还实现了：

- q whitening：约束均值、方差和协方差；
- Jacobian disentanglement：降低不同 q 维度对应的 `partial f/partial q_j` 之间的相关性；
- q-direction smoothness：惩罚模型沿 q 方向的二阶有限差分；
- canonicalization：对 q 做中心化和 whitening。

这些项主要用于减少多维 q 的冗余、改善跨运行稳定性，并让 q 更适合后续解释；它们不等价于严格识别真实物理因子。

---

## 7. 当前新对象推断原理

训练完成后，共享网络被冻结。对每个 unseen label：

1. 将该对象数据分成 support/calibration 和 query/evaluation；
2. 随机初始化一个新的 q；
3. 只在 support 数据上用 Adam 优化 q；
4. 在 query 数据上评估预测。

为了防止少量 support 点把 q 推到训练分布之外，使用训练 q 的逐维经验均值和标准差：

```text
mu_q    = mean(Q_train)
sigma_q = std(Q_train)
```

加入：

```text
L_q_prior = mean[((q_test-mu_q)/sigma_q)^2]
```

这可解释为对角高斯经验先验下的 MAP calibration：数据负责识别新对象，训练 q 分布负责抑制少样本过拟合。

当前实现按每个测试 label 的原始数据顺序取前 `calibration_ratio` 作为 support。因此若数据按时间、温度或循环数排序，它是“前段观测预测后段”的外推协议，不应描述成随机 few-shot。

---

## 8. 当前实现的完整计算流程

```text
输入：label, observed x, target y
        |
        v
按 label 划分训练对象和测试对象
        |
        v
用训练集统计量标准化 x 和 y
        |
        v
训练对象：label -> trainable q embedding
        |
        v
拼接 [x, q] -> shared ReLU MLP -> y_hat
        |
        v
联合优化预测、独立性、曲线连续性和 q 正则
        |
        v
冻结 shared MLP
        |
        v
测试新对象：support 数据梯度优化 q_test + empirical prior
        |
        v
query 数据评估，并输出每个对象对应的 q
```

当前 q 不是 encoder 一次前向计算得到的，而是训练期直接优化、测试期梯度校准得到的。因此方法在结构上更接近 auto-decoder。

---

## 9. 相关工作及与我们的区别

### 9.1 Functional PCA / functional data analysis

FPCA 把曲线表示成均值函数与少量主成分函数的线性组合，每个对象获得一组低维 score。它是“从曲线提取对象级低维坐标”的经典方法。

与我们相比：

- FPCA 通常是线性函数空间分解；
- 我们用非线性条件网络 `f(x,q)` 表示曲线族；
- 我们额外处理显变量采样分布泄露、few-shot q 校准和下游解释需求。

FPCA 应作为重要 baseline，因为它直接回答“神经网络潜变量是否优于经典曲线降维”。

### 9.2 Auto-decoder / DeepSDF

[DeepSDF](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.html) 为每个训练形状直接优化 latent code，并用坐标与 latent code 共同预测连续场；测试新形状时固定 decoder、优化新 code。

这是与我们结构最接近的工作：

```text
DeepSDF: spatial coordinate + shape code -> signed distance
ours:    scientific x + object q       -> scientific response
```

我们的差别不在“直接优化 latent code”本身，而在于面向科学响应曲线增加 label-level acquisition invariance、函数几何连续性和可解释性约束。

### 9.3 Conditional Neural Processes

[Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a) 从一组 context `(x,y)` 编码函数表示，再预测新 query，目标同样是从少量观测推断一条新函数。

区别是：

- CNP 使用 set encoder 做 amortized inference，一次前向得到表示；
- 当前方法不使用 encoder，而是对 q 做梯度校准；
- 当前方法显式约束 q 与采样分布表示的依赖，并把 q 作为后续科学解释对象。

Neural Process 是后续必须比较的强基线或扩展方向。

### 9.4 Meta-learning / MAML

[MAML](https://proceedings.mlr.press/v70/finn17a) 学习一个容易通过少量梯度步骤适配到新任务的参数初始化。

我们的新对象适配也使用梯度，但只更新低维 q，不更新共享网络。这样适配空间更小、结果更稳定，也更容易把适配变量解释为对象级状态。

### 9.5 Neural operator / DeepONet

[DeepONet](https://www.nature.com/articles/s42256-021-00302-5) 学习从输入函数到输出函数的非线性算子，使用 branch net 编码输入函数、trunk net 编码输出坐标。

我们的模型不是完整的 operator learner：当前没有把一条输入函数编码成另一条输出函数，而是学习由对象潜变量 q 参数化的一族响应函数。但两者都把科学学习对象从有限维点预测扩展到函数族，是相关的科学机器学习背景。

### 9.6 HSIC、distance correlation 与 distribution embedding

[HSIC](https://www.gatsby.ucl.ac.uk/~gretton/papers/GreBouSmoSch05.pdf) 提供基于 RKHS 的非线性依赖度量；distance correlation 从样本间距离定义依赖；[Random Fourier Features](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) 提供平移不变核的低维随机近似。

我们把这些工具组合到 label-level：先把每个 label 的 `P_l^X` 表示成 A，再约束 Q 与 A，而不是对原始样本直接做点级独立性惩罚。

### 9.7 Disentangled representation 与可识别性

[Locatello et al.](https://proceedings.mlr.press/v97/locatello19a) 说明，无监督 disentanglement 在缺少归纳偏置时不能被唯一识别。

这对本项目的直接启示是：低维、独立性、连续性、whitening 和 few-shot 预测协议都是人为加入的归纳偏置。它们能让 q 更有用、更规整，但不能仅凭高 R2 证明 q 就是真实且唯一的因果物理变量。

---

## 10. 理论上能声称什么

只依靠预测重构，q 存在可逆变换不唯一性。若 `T` 可逆，则：

```text
q' = T(q)
f'(x,q') = f(x,T^{-1}(q'))
```

与原模型产生相同预测。因此 q 的绝对坐标、符号、尺度和维度顺序通常没有天然物理意义。

当前方法合理的主张是：

> 学习一个对响应具有预测充分性、减少显变量采样分布泄露、保留对象间函数相似性、并可由新对象少量观测校准的低维潜在描述符。

当前方法不能单独保证：

> 唯一恢复真实、因果、可直接命名的内禀物理变量。

若要支持更强科学结论，还需要合成真值恢复、跨随机种子稳定性、与独立外部物理量关联、采样分布干预和跨域验证。

---

## 11. 一段可直接对学生讲的话

> 我们面对的是一族材料或系统响应曲线。显变量 x 只能解释曲线内部随条件的变化，无法完全解释不同对象之间的差异，所以给每个对象学习一个低维 q，用共享网络 y=f(x,q) 建模整族曲线。训练时网络与训练对象的 q 联合优化；测试新对象时冻结网络，只用少量观测校准新 q。
>
> 真正的研究问题不是简单加入 embedding，而是如何让 q 更像隐藏内在状态，而不是样品编号或采样范围的代码。我们把每个对象的 x 采样分布表示成 A，用 Pearson、HSIC、distance correlation 等方法抑制 q 对 A 的依赖，同时让相似响应曲线在 q 空间中靠近。测试校准还加入训练 q 的经验先验，避免少量观测导致 q 极端漂移。
>
> 这套方法在结构上最接近 auto-decoder，在新函数少样本推断上与 Neural Process 和元学习相关，在曲线降维上与 FPCA 相关，在独立性约束上使用核方法和距离相关。我们把 q 严格称为潜在描述符；是否能进一步解释为真实物理量，需要额外实验验证。
