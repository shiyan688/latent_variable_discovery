import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from lvs.core.pipeline import (
    CSVColumnConfig, LatentQConfig, load_csv_dataset,
    _compute_label_curve_distance_matrix, _normalize_label_value, EPSILON,
    normalize_features, normalize_targets, fit_normalization,
    split_calibration_and_eval_indices,
)


# ============================================================
# MDS 方法 1: exp(d/d0 + d0/d)
# ============================================================
def mds1_exp(distances: torch.Tensor, q_dim: int) -> torch.Tensor:
    n = distances.shape[0]
    if n <= q_dim:
        return torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    high_dim, low_dim = n, q_dim
    nhd = (distances / high_dim).clamp_min(EPSILON)
    q = torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    q.requires_grad_(True)
    opt = optim.Adam([q], lr=0.05)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    for ep in range(2000):
        opt.zero_grad()
        qd = q.unsqueeze(0) - q.unsqueeze(1)
        ld = qd.abs().sum(-1)
        nld = (ld / low_dim).clamp_min(EPSILON)
        r = (nhd / nld) + (nld / nhd)
        r = r.clamp_max(50.0)
        loss = torch.exp(r).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q], max_norm=1.0)
        opt.step(); sch.step()
    q = q.detach()
    return (q - q.mean(0, keepdim=True)) / q.std(0, keepdim=True).clamp_min(EPSILON)


# ============================================================
# MDS 方法 2: (d/d0)^2 + (d0/d)^2
# ============================================================
def mds2_sqratio(distances: torch.Tensor, q_dim: int) -> torch.Tensor:
    n = distances.shape[0]
    if n <= q_dim:
        return torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    high_dim, low_dim = n, q_dim
    nhd = (distances / high_dim).clamp_min(EPSILON)
    q = torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    q.requires_grad_(True)
    opt = optim.Adam([q], lr=0.05)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    for ep in range(2000):
        opt.zero_grad()
        qd = q.unsqueeze(0) - q.unsqueeze(1)
        ld = qd.abs().sum(-1)
        nld = (ld / low_dim).clamp_min(EPSILON)
        r1 = nhd / nld
        r2 = nld / nhd
        loss = (r1.pow(2) + r2.pow(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q], max_norm=1.0)
        opt.step(); sch.step()
    q = q.detach()
    return (q - q.mean(0, keepdim=True)) / q.std(0, keepdim=True).clamp_min(EPSILON)


# ============================================================
# MDS 方法 3: 经典MDS (特征值分解解析解)
# ============================================================
def mds3_classic(distances: torch.Tensor, q_dim: int) -> torch.Tensor:
    n = distances.shape[0]
    if n <= q_dim:
        return torch.randn(n, q_dim, dtype=distances.dtype, device=distances.device) * 0.1
    dev = distances.device; dt = distances.dtype
    D = distances.double().cpu().numpy()
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    pe = np.clip(eigvals[:q_dim], 0, None)
    Q = eigvecs[:, :q_dim] * np.sqrt(pe)[np.newaxis, :]
    q = torch.tensor(Q, dtype=dt, device=dev)
    return (q - q.mean(0, keepdim=True)) / q.std(0, keepdim=True).clamp_min(EPSILON)


# ============================================================
def create_mlp(idim):
    return nn.Sequential(
        nn.Linear(idim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )


def run_one_dataset(train_ds, test_ds, mds_func, mds_name: str, config: LatentQConfig):
    device = 'cpu'
    torch.manual_seed(config.seed); np.random.seed(config.seed)
    fc = train_ds.features.shape[1]; qdim = config.q_dim
    norm = fit_normalization(train_ds.features, train_ds.targets)
    ntf = normalize_features(train_ds.features, norm)
    ntt = normalize_targets(train_ds.targets, norm).reshape(-1, 1)
    ft = torch.tensor(ntf, dtype=torch.float32)
    tt = torch.tensor(ntt, dtype=torch.float32)
    ulbl = [_normalize_label_value(l) for l in pd.unique(train_ds.labels)]
    l2i = {l: i for i, l in enumerate(ulbl)}
    ilbl = np.array([l2i[_normalize_label_value(l)] for l in train_ds.labels], dtype=np.int64)
    lt = torch.tensor(ilbl, dtype=torch.long)
    lc = len(ulbl)

    # 曲线距离 + MDS
    cd = _compute_label_curve_distance_matrix(
        ft, tt.squeeze(1), lt, label_count=lc,
        grid_size=config.latent_curve_continuity_grid_size,
    )
    t0 = time.time()
    iq = mds_func(cd, qdim)
    mt = time.time() - t0

    # MDS质量
    qd = torch.cdist(iq, iq, p=2)
    mask = ~np.eye(lc, dtype=bool)
    cf = cd.cpu().numpy()[mask]; qf = qd.cpu().numpy()[mask]
    sp = float(spearmanr(cf, qf)[0]) if cf.std() > 1e-8 and qf.std() > 1e-8 else np.nan
    pe_corr = float(pearsonr(cf, qf)[0]) if cf.std() > 1e-8 and qf.std() > 1e-8 else np.nan

    # Embedding + 模型
    emb = nn.Embedding(lc, qdim)
    emb.weight.data.copy_(iq)
    for p in emb.parameters(): p.requires_grad = False
    model = create_mlp(fc + qdim)
    opt = optim.Adam(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    N = ft.shape[0]; B = min(config.batch_size, N)

    # 训练
    t0 = time.time()
    for ep in range(config.epochs):
        model.train()
        perm = torch.randperm(N)
        for s in range(0, N, B):
            idx = perm[s:s + B]
            inp = torch.cat([ft[idx], emb(lt[idx])], dim=1)
            pred = model(inp)
            loss = mse(pred, tt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    trt = time.time() - t0
    model.eval()
    with torch.no_grad():
        inp = torch.cat([ft, emb(lt)], dim=1)
        tr_pred = model(inp)
        tr_r2 = float(r2_score(tt.numpy(), tr_pred.numpy()))

    # 测试：校准Q + 评估
    t0 = time.time()
    ntf2 = normalize_features(test_ds.features, norm)
    ntt2 = normalize_targets(test_ds.targets, norm).reshape(-1, 1)
    ft2 = torch.tensor(ntf2, dtype=torch.float32)
    tt2 = torch.tensor(ntt2, dtype=torch.float32)
    trq = emb.weight.detach()
    qm = trq.mean(0); qs = trq.std(0).clamp_min(0.05)
    epr = []; etr = []
    model.eval()
    for rl in pd.unique(test_ds.labels):
        label = _normalize_label_value(rl)
        li = np.flatnonzero(test_ds.labels == rl)
        ci, ei = split_calibration_and_eval_indices(li, config.calibration_ratio)
        if len(ci) == 0 or len(ei) == 0: continue
        iqv = torch.tensor(np.random.randn(qdim) * 0.1, dtype=torch.float32)
        if label in l2i: iqv = emb.weight[l2i[label]].detach().clone()
        qp = nn.Parameter(iqv)
        qo = optim.Adam([qp], lr=config.calibration_lr)
        for _ in range(config.calibration_steps):
            cf2 = ft2[ci]
            inp_c = torch.cat([cf2, qp.unsqueeze(0).repeat(cf2.shape[0], 1)], dim=1)
            p_c = model(inp_c)
            l_c = mse(p_c, tt2[ci])
            if config.calibration_q_prior_weight > 0:
                sq = (qp - qm) / qs
                l_c = l_c + config.calibration_q_prior_weight * torch.mean(sq.pow(2))
            qo.zero_grad(); l_c.backward(); qo.step()
        with torch.no_grad():
            ef = ft2[ei]
            inp_e = torch.cat([ef, qp.detach().unsqueeze(0).repeat(ef.shape[0], 1)], dim=1)
            p_e = model(inp_e)
            epr.append(p_e.numpy()); etr.append(tt2[ei].numpy())
    tet = time.time() - t0
    if len(epr) == 0:
        return None
    ap = np.concatenate(epr).reshape(-1); at = np.concatenate(etr).reshape(-1)
    sc = norm.target_std if norm.target_std > EPSILON else 1.0
    pd_ = ap * sc + norm.target_mean; td_ = at * sc + norm.target_mean
    te_r2 = float(r2_score(td_, pd_))
    te_mse = float(mean_squared_error(td_, pd_))
    nz = td_ != 0
    te_mape = float(mean_absolute_percentage_error(td_[nz], pd_[nz])) if np.any(nz) else np.nan
    return dict(
        mds_name=mds_name,
        mds_spearman=sp, mds_pearson=pe_corr, mds_time=mt,
        train_time=trt, test_time=tet,
        train_r2=tr_r2, test_r2=te_r2, test_mse=te_mse, test_mape=te_mape,
    )


def main():
    datasets = [
        ("seebeck", "Seebeck系数"),
        ("electrical_conductivity", "电导率"),
        ("thermal_conductivity", "热导率"),
        ("zt", "ZT值"),
    ]
    base = os.path.join(_HERE, 'data', 'application', 'starry_te')

    mds_methods = [
        ("M1_exp(d/d0+d0/d)", mds1_exp),
        ("M2_(d/d0)2+(d0/d)2", mds2_sqratio),
        ("M3_经典MDS特征分解", mds3_classic),
    ]

    config = LatentQConfig(
        q_dim=3, epochs=1000, batch_size=128, lr=5e-4, seed=42,
        verbose=False, early_stop_enabled=False,
        calibration_steps=200, calibration_lr=0.05, calibration_ratio=0.3,
    )

    all_results = []
    print("=" * 80, flush=True)
    print("三种MDS初始化方法对比测试 (q_dim=3, Q固定后只训练θ)", flush=True)
    print("=" * 80, flush=True)

    for ds_key, ds_name in datasets:
        print(f"\n{'─' * 80}", flush=True)
        print(f"【数据集】{ds_name} ({ds_key})", flush=True)
        print(f"{'─' * 80}", flush=True)
        train_path = f'{base}/{ds_key}/train.csv'
        test_path = f'{base}/{ds_key}/test.csv'
        train_ds = load_csv_dataset(train_path, CSVColumnConfig(
            feature_cols=(1,2,3,4,5,6,7,8,9,10), label_col=0, target_col=11, has_header=True,
        ))
        test_ds = load_csv_dataset(test_path, CSVColumnConfig(
            feature_cols=(1,2,3,4,5,6,7,8,9,10), label_col=0, target_col=11, has_header=True,
        ))
        ntr_l = len(np.unique(train_ds.labels)); nte_l = len(np.unique(test_ds.labels))
        ntr_s = len(train_ds.targets); nte_s = len(test_ds.targets)
        print(f"  训练集: {ntr_l} labels × {ntr_s} samples  |  测试集: {nte_l} labels × {nte_s} samples", flush=True)

        for mds_name, mds_func in mds_methods:
            print(f"\n  ▶ 方法: {mds_name}", flush=True)
            t0 = time.time()
            try:
                res = run_one_dataset(train_ds, test_ds, mds_func, mds_name, config)
                if res is None:
                    print(f"    ❌ 无评估样本", flush=True); continue
                total = time.time() - t0
                res['dataset'] = ds_name; res['dataset_key'] = ds_key
                all_results.append(res)
                print(f"    MDS:  Spearman={res['mds_spearman']:.4f}  Pearson={res['mds_pearson']:.4f}  耗时={res['mds_time']:.1f}s", flush=True)
                print(f"    训练:  R²={res['train_r2']:.4f}  耗时={res['train_time']:.1f}s", flush=True)
                print(f"    测试:  R²={res['test_r2']:.4f}  MSE={res['test_mse']:.3e}", end="", flush=True)
                if not np.isnan(res['test_mape']):
                    print(f"  MAPE={res['test_mape']*100:.2f}%", flush=True, end="")
                print(f"  耗时={res['test_time']:.1f}s", flush=True)
                print(f"    单方法总耗时: {total:.1f}s", flush=True)
            except Exception as e:
                print(f"    ❌ 异常: {e}", flush=True)
                import traceback; traceback.print_exc()

    # ===== 汇总对比 =====
    print(f"\n\n{'=' * 80}", flush=True)
    print("                         ╔══════════════════════════════════════════╗", flush=True)
    print("                         ║      三 种 MDS 方 法 对 比 结 果        ║", flush=True)
    print("                         ╚══════════════════════════════════════════╝", flush=True)
    print(f"{'=' * 80}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)

        print(f"\n━━━ 测试 R² (核心指标) ━━━", flush=True)
        pivot_r2 = df.pivot(index='dataset', columns='mds_name', values='test_r2')
        # 找每一行最大值
        for ds in pivot_r2.index:
            best = pivot_r2.loc[ds].idxmax()
            vals = pivot_r2.loc[ds]
            print(f"\n  {ds}:", flush=True)
            for col in pivot_r2.columns:
                marker = " ← ★ 最佳" if col == best else ""
                print(f"    {col:30s} →  R² = {vals[col]:.4f}{marker}", flush=True)

        print(f"\n━━━ 训练 R² ━━━", flush=True)
        p_tr = df.pivot(index='dataset', columns='mds_name', values='train_r2')
        for ds in p_tr.index:
            print(f"  {ds}:", flush=True)
            for col in p_tr.columns:
                print(f"    {col:30s} →  R² = {p_tr.loc[ds, col]:.4f}", flush=True)

        print(f"\n━━━ MDS 坐标保持质量 (Spearman 相关) ━━━", flush=True)
        p_sp = df.pivot(index='dataset', columns='mds_name', values='mds_spearman')
        for ds in p_sp.index:
            print(f"  {ds}:", flush=True)
            for col in p_sp.columns:
                print(f"    {col:30s} →  ρ = {p_sp.loc[ds, col]:.4f}", flush=True)

        print(f"\n━━━ 训练 / 测试 R² 对比表 ━━━", flush=True)
        summary_rows = []
        for ds in df['dataset'].unique():
            sub = df[df['dataset'] == ds]
            best_row = sub.loc[sub['test_r2'].idxmax()]
            summary_rows.append({
                '数据集': ds,
                '最佳MDS方法': best_row['mds_name'],
                '最佳测试R²': f"{best_row['test_r2']:.4f}",
                'M1测试R²': f"{sub[sub['mds_name']==mds_methods[0][0]]['test_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[0][0]])>0 else 'N/A',
                'M2测试R²': f"{sub[sub['mds_name']==mds_methods[1][0]]['test_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[1][0]])>0 else 'N/A',
                'M3测试R²': f"{sub[sub['mds_name']==mds_methods[2][0]]['test_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[2][0]])>0 else 'N/A',
                'M1训练R²': f"{sub[sub['mds_name']==mds_methods[0][0]]['train_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[0][0]])>0 else 'N/A',
                'M2训练R²': f"{sub[sub['mds_name']==mds_methods[1][0]]['train_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[1][0]])>0 else 'N/A',
                'M3训练R²': f"{sub[sub['mds_name']==mds_methods[2][0]]['train_r2'].values[0]:.4f}" if len(sub[sub['mds_name']==mds_methods[2][0]])>0 else 'N/A',
            })
        sdf = pd.DataFrame(summary_rows)
        pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
        print(sdf.to_string(index=False), flush=True)

        os.makedirs(os.path.join(_HERE, 'runs'), exist_ok=True)
        out = os.path.join(_HERE, 'runs', 'mds_methods_comparison_final.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        sum_out = os.path.join(_HERE, 'runs', 'mds_methods_comparison_summary.csv')
        sdf.to_csv(sum_out, index=False, encoding='utf-8-sig')
        print(f"\n✅ 完整详细结果: {out}", flush=True)
        print(f"✅ 汇总对比表:   {sum_out}", flush=True)

        # ===== 详细分析 =====
        print(f"\n{'='*80}", flush=True)
        print("                         分 析 与 结 论", flush=True)
        print(f"{'='*80}", flush=True)
        print("\n1. 各方法理论差异：", flush=True)
        print("   M1 (exp)  : 对距离不匹配惩罚极度陡峭，适合强调保持距离比例", flush=True)
        print("   M2 (平方) : 中等强度惩罚，梯度更平滑稳定", flush=True)
        print("   M3 (经典) : 解析解，L2距离最优保持，无局部最优问题，速度快", flush=True)
        print("\n2. 各数据集最佳方法：", flush=True)
        for r in summary_rows:
            print(f"   {r['数据集']:8s} → {r['最佳MDS方法']} (R²={r['最佳测试R²']})", flush=True)
        print("\n3. MDS质量与测试R²的关系：", flush=True)
        sp_mean = p_sp.mean()
        r2_mean = pivot_r2.mean()
        print(f"   方法平均 Spearman: ", flush=True)
        for c in p_sp.columns: print(f"     {c:30s}: {sp_mean[c]:.4f}", flush=True)
        print(f"   方法平均 测试R²: ", flush=True)
        for c in pivot_r2.columns: print(f"     {c:30s}: {r2_mean[c]:.4f}", flush=True)

        # 胜负统计
        print("\n4. 各方法对数据集的胜率：", flush=True)
        wins = {m[0]: 0 for m in mds_methods}
        for ds in pivot_r2.index:
            wins[pivot_r2.loc[ds].idxmax()] += 1
        for k, v in wins.items():
            print(f"   {k:30s} : 赢{v}次", flush=True)

    print("\n全部完成！", flush=True)


if __name__ == '__main__':
    main()
