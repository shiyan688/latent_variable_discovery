# 主流程 Loss 创新说明

更新时间：2026-05-21 17:11 CST

本文档记录当前 `latent_variable_search` 主流程里已经实现的 loss 设计、代码对应位置、论文叙述逻辑、现阶段证据和需要继续补强的风险点。目标是后续可以直接抽取为论文的 Method / Ablation / Discussion。

## 1. 问题设定

当前任务不是普通监督学习，而是希望机器从观测数据中自动发现物质或物理系统的隐藏内禀属性。

每条样本包含：

- 显变量：`x`，即实验中可直接控制或测量的变量。
- 目标值：`y`，即物性、响应函数或实验输出。
- 组标签：`label`，表示同一种材料、同一条函数曲线、同一实验对象或同一隐藏条件。训练时同一个 label 下有多组 `(x, y)`。
- 隐变量：`q_label`，模型需要为每个 label 学出一个低维连续向量。这个向量不直接给真值，作为待发现的材料或系统内禀属性。

神经主模型写成：

```text
y_hat = f_theta(x, q_label)
```

训练阶段同时学习：

- 共享函数逼近器 `f_theta`
- 每个训练 label 的 embedding `q_label`

测试阶段遇到新 label 时，冻结 `f_theta`，只用该 label 的少量 calibration 样本优化一个新的 `q_test`，再在该 label 剩余样本上评估泛化。

这个设定的核心是：`q` 不能只是帮助拟合训练数据的任意自由参数，而应该满足内禀属性的几个约束：

1. 能解释目标函数变化。
2. 尽量独立或正交于显变量分布。
3. 相近的 `q` 应对应相近的整体函数曲线。
4. 测试校准时不应漂移到训练分布之外的无意义区域。

因此当前主流程把普通 MSE 拓展为带结构先验的隐变量发现目标。

## 2. 总 Loss

训练阶段的总目标为：

```text
L_train = L_pred
        + lambda_orth * L_orth
        + lambda_cont * L_cont
```

测试 label 校准阶段的目标为：

```text
L_calibration = L_cal_pred + lambda_prior * L_q_prior
```

代码位置：

- 训练 loss 组合：`latent_q_pipeline.py` 中训练循环，`loss = mse_loss(...)` 后加入 orth 和 continuity。
- 测试校准 prior：`latent_q_pipeline.py` 中 `evaluate_latent_q_model` 的 calibration loop。
- 显变量正交 loss：`_latent_feature_correlation_penalty`。
- 曲线连续性 loss：`_compute_label_curve_distance_matrix` 和 `_latent_curve_continuity_penalty`。

当前主要参数名：

```text
latent_feature_orthogonality_weight = lambda_orth
latent_curve_continuity_weight      = lambda_cont
calibration_q_prior_weight          = lambda_prior
latent_curve_continuity_grid_size   = curve profile grid size
```

## 3. 预测 Loss：保证 q 有解释目标值的必要性

基础预测 loss 是标准 MSE：

```text
L_pred = mean((f_theta(x_i, q_label_i) - y_i)^2)
```

作用：

- 保证学出来的 `q` 不是任意编码，而是必须对 `y` 的变化有贡献。
- 让共享网络 `f_theta` 学习统一的显变量-隐变量-目标函数关系。
- 后续符号回归可以在 `(x, q, y)` 数据上寻找显式表达式。

注意：只有 MSE 时，`q` 容易成为一个黑箱 label code。它能拟合，但不一定满足“内禀属性”的物理解释要求。因此需要下面两个结构 loss。

## 4. 显变量正交 Loss：让 q 尽量独立于 x 分布

### 4.1 当前实现

对每个 label，当前代码先计算该 label 下显变量的统计量：

```text
s_label = [mean(x_label), std(x_label)]
```

也就是说，如果有 `d` 个显变量，每个 label 会得到一个 `2d` 维的 feature statistics。

然后对所有 label 的隐变量矩阵 `Q` 和显变量统计矩阵 `S` 分别按列标准化，计算相关矩阵：

```text
C = corr(Q, S)
```

正交 loss 为相关矩阵平方均值：

```text
L_orth = mean(C^2)
```

等价地，它惩罚任意一个隐变量维度和任意一个显变量统计维度之间的线性相关。

### 4.2 直觉

如果 `q` 真的是材料或系统的隐藏内禀属性，那么它不应该只是显变量采样范围、显变量均值、显变量方差的替身。

例如：

- 如果不同材料 label 的温度采样范围略有不同，纯 MSE 可能把这种采样偏差编码进 `q`。
- 如果某些 label 的实验点集中在高压区，模型可能把“高压采样偏差”误当成材料属性。
- 正交 loss 要求 `q` 对这些显变量分布统计尽量不敏感，从而更像材料自身属性。

### 4.3 论文表述建议

可以称为：

```text
Observed-variable orthogonality regularization
```

或：

```text
Latent-feature decorrelation constraint
```

中文可写为：

```text
显变量正交约束 / 显变量去相关约束
```

核心贡献点：

> 我们不直接监督隐变量真值，而是要求发现的隐变量在 label 层面与显变量采样统计解耦，从而降低模型把实验设计偏差误识别为材料内禀属性的风险。

### 4.4 当前限制

当前实现只约束一阶均值和二阶标准差，属于弱独立性约束，不等价于严格统计独立。

后续如果要冲更高等级论文，建议升级为：

- HSIC independence loss。
- mutual information upper bound / adversarial predictor。
- distance correlation。
- label-level propensity correction。

但当前版本已经比纯 MSE 更接近“内禀属性应独立于显变量”的科学假设。

## 5. 曲线连续性 Loss：让 q 的距离对应函数曲线距离

### 5.1 当前实现

对每个 label，当前代码基于第一个显变量 `x1` 构造响应曲线 profile：

1. 取训练数据中的 primary feature，即 `feature_tensor[:, 0]`。
2. 在该特征的范围内建立均匀 grid。
3. 对每个 label，把该 label 的 `(x1, y)` 按 `x1` 排序。
4. 如果同一 `x1` 有多个点，则先平均。
5. 用一维线性插值把每个 label 的响应曲线投到同一个 grid 上。
6. 对每条 profile 去均值并按 RMS 归一化。
7. 计算 label 与 label 之间的曲线距离矩阵 `D_curve`。
8. 对距离矩阵做标准化。

训练时，模型同时计算当前隐变量 embedding 的距离矩阵：

```text
D_q = pairwise_distance(q_label)
```

并标准化。连续性 loss 为非对角元素的均方差：

```text
L_cont = mean((normalize(D_q) - normalize(D_curve))^2)
```

### 5.2 直觉

如果两个材料或系统的整体响应函数很相似，它们的隐藏内禀属性也应该相近；如果响应函数差异很大，它们的 `q` 也应该更远。

这个 loss 不要求知道真实隐变量，而是用“整体函数曲线相似性”给隐变量空间提供几何结构。

它实际在做：

```text
function-space geometry -> latent-space geometry
```

也就是把函数空间里的相似关系蒸馏到低维隐变量空间。

### 5.3 论文表述建议

可以称为：

```text
Function-geometry continuity regularization
```

或：

```text
Response-curve continuity loss
```

中文可写为：

```text
响应曲线连续性约束 / 函数几何连续性约束
```

核心贡献点：

> 我们通过 label-level 响应曲线距离约束隐变量距离，使发现的隐变量空间不仅服务于点预测，还保留材料响应函数之间的连续拓扑结构。

### 5.4 当前限制

当前实现的连续性 profile 只使用第一个显变量 `x1`。这在单主变量曲线任务中合理，但对于多显变量真实数据还不够完整。

如果要提高 reviewer 认可度，建议后续改成：

- 多维 Sobol / Latin hypercube 网格上评估响应面。
- 对每个 label 拟合局部 surrogate 后计算函数距离。
- 使用随机 anchors：在共同 `x` anchor set 上比较不同 label 的预测曲线。
- 使用梯度信息：约束 `q` 邻近时 `f(x,q)` 和 `grad_x f(x,q)` 都相近。

当前版本可以作为第一版 proof-of-concept，但论文里必须诚实说明它是 primary-axis response profile。

## 6. 测试校准 q Prior：防止新 label 的 q 漂移

### 6.1 当前实现

训练结束后，代码从训练 label 的 embedding 中计算：

```text
mu_q  = mean(q_train)
std_q = std(q_train), lower bounded by 0.05
```

测试新 label 校准时，只优化该 label 的 `q_test`，并加入：

```text
L_q_prior = mean(((q_test - mu_q) / std_q)^2)
```

因此：

```text
L_calibration = MSE_on_calibration_samples + lambda_prior * L_q_prior
```

### 6.2 直觉

测试 label 的 calibration 样本通常很少。如果只用少量点优化 `q_test`，它可能为了拟合 calibration 点漂移到训练隐空间之外，导致 evaluation 样本上泛化变差。

`q_prior` 相当于一个轻量的 empirical Bayes prior：

- 允许新材料有自己的 `q`。
- 但鼓励它落在训练材料隐空间的合理范围内。

### 6.3 参数解释

`calibration_q_prior_weight` 越大，测试校准出的 `q` 越保守。

- 太小：容易过拟合 calibration 点。
- 太大：新 label 的 `q` 被拉回训练均值，表达能力不足。

当前真实数据参数扫描里，`0.01` 和 `0.03` 经常表现更好；更大的 `0.1/0.2/0.3` 不一定提升 test R2。

## 7. 当前主流程的创新性判断

当前 loss 设计的创新点不是“发明一个更复杂的神经网络”，而是把“隐藏内禀属性”的科学要求写进训练目标：

1. `L_pred`：隐变量必须解释目标响应。
2. `L_orth`：隐变量不能退化成显变量采样偏差。
3. `L_cont`：隐变量空间应保留响应函数的连续几何。
4. `L_q_prior`：新样本校准时保持在训练隐空间的合理区域。

这套组合与后面的符号回归天然衔接：神经网络先发现连续隐变量，再把 `(x, q, y)` 交给符号回归，寻找可解释闭式表达式。

论文里可以包装成：

```text
regularized latent-variable discovery for symbolic scientific model construction
```

或：

```text
physics-oriented latent calibration followed by symbolic regression
```

## 8. 现有实验对 loss 的支持程度

真实应用数据上，orth + continuity 对部分任务明显有效：

- `starry_te_thermal_conductivity`：orth+cont best R2 `0.9890`，高于 MSE best `0.9766`。
- `starry_te_electrical_conductivity`：orth+cont best R2 `0.9105`，高于 MSE best `0.8205`。
- `starry_te_zt`：orth+cont best R2 `0.9483`，高于 MSE best `0.9152`。

但不是所有任务都提升：

- `battery_matr_capacity_protocol`：MSE best `0.9885`，orth+cont best `0.9834`，差距很小。
- `starry_te_seebeck`：MSE best `0.5598`，orth+cont best `0.2972`，当前 loss 没有解决该任务。

表达式数据集上，截至当前扫描，纯 R2 指标并不总是由 orth+cont 取得最高。当前最好的完整设置之一是：

```text
orth_cont_cal05_prior001:
best per expression mean R2 = 0.8493
best median R2 = 0.9992
best >= 0.95 = 35 / 45
best >= 0.80 = 37 / 45
```

它略好于：

```text
mse_only_cal03_prior001:
best per expression mean R2 = 0.8480
best median R2 = 0.9989
best >= 0.95 = 34 / 45
best >= 0.80 = 36 / 45
```

因此目前最准确的结论是：

> 新 loss 在部分真实任务和部分表达式任务上提升 test R2，并能降低隐变量与显变量统计的相关性；但它还没有在所有任务上稳定支配 MSE-only。它的主要价值应同时从预测性能、隐变量可解释性、显变量去相关、函数连续性和后续符号回归成功率来证明，而不能只看 test R2。

## 9. 后续最重要的改进方向

如果目标是 Nature 子刊、AAAI 或同等级论文，建议优先做以下增强：

1. 把 continuity loss 从单 `x1` 曲线升级为多维响应面 continuity。
2. 用 HSIC / distance correlation 替代或补充当前 Pearson decorrelation。
3. 增加 latent identifiability 实验：真实 q 已知时，比较 recovered q 与 ground truth q 的单调性、相关性和可逆变换一致性。
4. 加入 symbolic regression 后，比较是否能恢复更短、更准、更物理可解释的公式。
5. 对 loss 做完整消融：MSE、MSE+orth、MSE+cont、MSE+orth+cont、不同 `q_dim`、不同 calibration ratio、不同 prior weight。
6. 对奇异表达式统一物理定义域，避免 `x` 接近 0 导致 heavy-tail / non-finite target，这会掩盖 loss 本身效果。

## 10. 推荐论文 Method 段落草稿

可以这样写：

> We parameterize each material or experimental condition by a trainable latent vector and learn a shared response model conditioned on both observed variables and the latent vector. To prevent the latent representation from degenerating into a label identifier or a proxy for the observed-variable sampling distribution, we augment the prediction loss with two structure-preserving regularizers. The first penalizes correlations between the learned latent vectors and label-level statistics of the observed variables, encouraging the latent factors to encode intrinsic properties rather than experimental design artifacts. The second matches pairwise distances in the latent space to distances between label-level response profiles, enforcing continuity of the latent manifold with respect to the global functional response. For unseen labels, the response model is frozen and only the latent vector is calibrated from a small subset of observations, with an empirical Gaussian prior derived from the training latent distribution to prevent out-of-distribution calibration.

对应中文：

> 我们为每个材料或实验条件引入一个可学习隐向量，并训练一个同时依赖显变量和隐向量的共享响应模型。为了避免隐向量退化为普通 label 编码，或错误吸收显变量采样分布偏差，我们在预测误差之外加入两个结构正则项：其一惩罚隐向量与 label 层面显变量统计量的相关性，使隐变量更接近独立于实验设计的内禀属性；其二约束隐空间距离与 label 响应曲线距离一致，使隐空间保留整体函数响应的连续几何。对于测试阶段的新 label，我们冻结共享模型，仅用少量校准样本优化其隐向量，并通过由训练隐向量分布估计得到的经验 prior 抑制校准时的分布外漂移。
