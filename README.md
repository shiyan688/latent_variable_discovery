# Latent Variable Discovery

面向分组科学数据的隐变量发现代码。给定可观测变量 `x`、组/材料标识 `label` 和响应 `y`，模型学习

```text
y_hat = f_theta(x, q_label)
```

其中 `q_label` 是每个训练组的低维连续描述符。对未见组，模型参数保持冻结，仅用该组的一部分校准样本估计 `q`，再在严格不重叠的样本上评估。因此 `label` 只用于索引或校准隐变量，不作为数值显变量输入模型。

当前公开版本止于 Torch/KAN 隐变量学习、校准与评估。符号回归代码、第三方 DAGPartition/PySR 环境及实验结果暂不包含。

## 方法概览

训练目标由预测误差和可选的结构约束组成：

```text
L_train = L_pred
        + lambda_orth   R_independence(q, A)
        + lambda_cont   R_continuity(q, curves)
        + lambda_q      ||q||^2
        + lambda_white  R_whitening(q)
        + lambda_jac    R_jacobian(f)
        + lambda_smooth R_smooth(q)
```

`A` 是按 label 聚合的采样分布描述，可选均值/标准差、丰富统计量、RFF 核均值嵌入或 `rich_rff_kme`。独立性约束支持 Pearson、HSIC、normalized HSIC、distance correlation、adversarial 和 propensity weighting。测试阶段优化：

```text
q* = argmin_q MSE(calibration subset) + lambda_prior ||q||^2
```

随后只在该 label 的 held-out evaluation subset 上报告 R2/MSE。实现细节见 [主流程损失说明](MAIN_WORKFLOW_LOSS_INNOVATION.md) 和 [背景与理论](LATENT_Q_INTRO_MOTIVATION_THEORY_RELATED_WORK.md)。

## 目录

```text
lvs/core/pipeline.py              核心数据、训练、校准、损失与评估
lvs/core/expression_library.py    合成表达式解析与数据生成
lvs/backends/torch_mlp.py         默认 Torch MLP 后端
lvs/backends/kan.py               可选 KAN 后端
lvs/workflows/single.py           单表达式完整隐变量工作流
lvs/workflows/batch.py            多表达式并行工作流
scripts/                          数据准备、消融与真实数据运行脚本
data/latent_variable_*.csv        小型表达式基准定义
tests/                            核心回归测试
```

顶层 `latent_q_pipeline.py`、`q_optimize_torch.py` 和 `run_workflow.py` 是历史兼容入口；正式实现以 `lvs/` 包为准。

## GPU 环境

建议 Python 3.11。先按机器 CUDA/驱动安装 GPU 版 PyTorch，再安装本项目；不要让通用依赖安装命令覆盖已经选好的 CUDA wheel。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 示例：适用于支持 CUDA 12.8 wheel 的驱动；其他机器按 PyTorch 官方选择器调整。
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

可选 KAN 后端：

```bash
pip install -e '.[kan]'
```

## 快速运行

列出表达式并运行一个 Torch 实验：

```bash
python -m lvs workflow --list-expressions

python -m lvs workflow \
  --expression-id 1 \
  --backend torch \
  --q-dim 1 \
  --epochs 500 \
  --cal-steps 1200 \
  --latent-feature-orthogonality-weight 0.05 \
  --latent-feature-orthogonality-type distance_correlation \
  --latent-feature-stats-mode rich_rff_kme \
  --output-root runs/quickstart
```

批量运行：

```bash
python -m lvs batch \
  --expression-ids 1,2,3 \
  --backend torch \
  --q-dim 1 \
  --epochs 500 \
  --max-workers 2
```

直接处理无表头 CSV 时，默认列约定是 `label, x1, target`，即 `label_col=0`、`feature_cols=1`、`target_col=-1`。多显变量必须显式指定，例如：

```bash
lvs-torch \
  --train-csv train.csv \
  --test-csv test.csv \
  --feature-cols 1,2,3 \
  --label-col 0 \
  --target-col -1 \
  --q-dim 2 \
  --device cuda:0
```

从科学意义上无法解释的行号、sensor ID、数据库主键和目标派生列不应放入 `--feature-cols`。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖输入列默认值、校准/评估隔离、label-balanced MSE、采样分布嵌入和一个 CPU 端到端 smoke run。
