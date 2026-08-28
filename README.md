# Latent Variable Discovery

面向分组科学数据的隐变量发现代码。给定可观测变量 `x`、组/材料标识 `label` 和响应 `y`，模型学习

```text
y_hat = f_theta(x, q_label)
```

其中 `q_label` 是每个训练组的低维连续描述符。对未见组，模型参数保持冻结，仅用该组的一部分校准样本估计 `q`，再在严格不重叠的样本上评估。因此 `label` 只用于索引或校准隐变量，不作为数值显变量输入模型。

当前研究分支包含 Torch/KAN 隐变量学习、校准与评估，主要实验控制器，以及 reviewer-clean NASA Stage C 的紧凑汇总证据。大型原始数据、训练 checkpoint、逐 cell 预测和第三方运行环境不进入 Git；它们是显式外部输入，具体边界见“跨机器复现”一节。

当前真实可解释表达式端点已经通过开发和时间外推确认。开发阶段对 80 个 reviewer-clean Starry ZT 材料用约 25% 温度分层 support 估计 `q0,q1,q2`，再以 `ZT=q0+q1*tau+q2*tau^2` 预测其余 query，五折 pooled OOF `R²=0.980668`。结构冻结后，官方 2026-08-29 发布中按目标盲规则选出的 30 个新 sample ID、30 个不同 DOI/组成一次性确认达到 `R²=0.988810`，entity-bootstrap 下界为 `0.973306`，六项预注册门槛全部通过，query-target 扰动影响严格为 0。确认中 pooled 分数虽高于 support kNN，但实体配对差异不显著，所以这证明的是紧凑、连续、可解释 response re-q 的跨时间迁移，不是普遍预测 SOTA。完整协议和结果见 [主报告第 23 节](COMPLETE_RESEARCH_REPORT_20260809.md)、`runs/starry_zt_interpretable_req_20260829/` 和 `runs/starry_zt_temporal_confirmation_20260829/`。

对 learned neural q 的补充桥接给出更细的边界：raw q 直接映射二次系数失败（`R²=-1.907461`），而先把 decoder 响应投影到具名二次系数达到 pooled `R²=0.944683`、decoder fidelity 最低 `0.985033`。但近零 ZT 材料出现严重实体尾部，functional 距离几何也没有比 raw q 更稳定，所以完整 bridge gate 仍失败；它不能覆盖或削弱上面已经独立确认的 structure re-q 表达式。详见 [主报告 23.6](COMPLETE_RESEARCH_REPORT_20260809.md) 和 `runs/starry_zt_neural_canonical_bridge_20260829/NEURAL_CANONICAL_BRIDGE_RESULTS.md`。

只改变 train-fold-only `asinh` 目标尺度的后续实验把 neural-functional 单实体 R² 中位数从 `0.846748` 提到 `0.940261`，R²≥0.85 的实体从 40/80 增到 52/80，十倍尾部从 19 个减到 9 个，并使 functional 距离稳定性超过 raw q；pooled R² 仍为 `0.942488`。最坏倍率仍未过严格门，因此报告为有效诊断与部分修复，不冒充完整 bridge pass。见 [主报告 23.7](COMPLETE_RESEARCH_REPORT_20260809.md) 和 `runs/starry_zt_scale_aware_neural_bridge_20260829/SCALE_AWARE_BRIDGE_RESULTS.md`。

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
pip install -e '.[dev,experiments]'

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

可选 KAN 后端：

```bash
pip install -e '.[kan]'
```

NASA 数据准备、PDEBench 与其他实验脚本需要 `experiments` 依赖；Stage C 符号回归还需要 PySR 1.5.10 及其 Julia 环境：

```bash
pip install -e '.[experiments,symbolic]'
python -c "import pysr; print(pysr.__version__)"
```

所有 Python campaign 控制器都使用启动它们的 `sys.executable`，不会绑定某个固定虚拟环境目录。因此应先激活目标环境，再用该环境的 `python` 启动控制器。

## 跨机器复现

| 内容 | Git 是否包含 | 新机器上的操作 |
|---|---|---|
| 核心包、合成表达式定义、测试 | 是 | 安装依赖后可直接运行 |
| Stage C 汇总表、公式诊断和 gate 判定 | 是 | 可直接阅读和复核汇总数字 |
| NASA/PDEBench/其他真实原始数据 | 否 | 按数据来源准备到本地，并通过脚本参数或仓库相对路径提供 |
| 上游 inner-q 产物、checkpoint、逐 cell 预测/Pareto front | 否 | 从实验归档恢复，或先运行对应上游阶段 |
| PySR/Julia 运行时 | 否 | 安装 `symbolic` 可选依赖并完成 PySR 的 Julia 初始化 |

仓库中的运行时路径以仓库根目录为基准。历史报告可以记录当时的机器和环境名称，但执行代码不依赖这些历史地址。项目运行产物、测试夹具和缓存统一放在仓库的 `runs/` 下；默认缓存位于 `runs/_runtime_cache/`，不使用 `/tmp`。仍可通过 `MPLCONFIGDIR` 和 `XDG_CACHE_HOME` 环境变量覆盖。

PDEBench Burgers 数据可用带尺寸和校验和检查的脚本准备：

```bash
bash scripts/download_pdebench_burgers_nu002.sh
```

NASA reviewer-clean 数据准备器接受显式路径：

```bash
python scripts/prepare_nasa_battery_reviewer_clean_20260825.py \
  --raw-root /path/to/nasa_battery/extracted_batches \
  --output-root data/real_datasets2/prepared/nasa_battery_reviewer_clean_20260825
```

完整 Stage C 还需要未纳入 Git 的上游 `runs/nasa_battery_reviewer_clean_inner_q_20260825`。恢复该目录后，运行 [冻结计划](NASA_INNER_SYMBOLIC_STRUCTURE_PLAN_20260825.md) 中的相对路径命令。若上游产物位于其他位置，重新分析时显式传入：

```bash
python scripts/analyze_nasa_inner_symbolic_structure_20260825.py \
  --root runs/nasa_battery_reviewer_clean_inner_symbolic_20260825 \
  --q-root /path/to/nasa_battery_reviewer_clean_inner_q_20260825
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
python -m pytest -q
```

测试覆盖输入列默认值、校准/评估隔离、损失预设、连续性与几何指标、support-conditioned baseline 和 CPU 端到端 smoke run。
