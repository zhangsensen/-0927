#!/usr/bin/env python3
"""
因子 Alpha 深度检验 | Factor Alpha Deep Analysis
=================================================
对 25 个因子在 43 只 ETF 上的预测能力进行全方位检验。

检验维度:
  1. 单因子滚动 IC (Spearman) — 均值 IC、IC_IR、命中率、t 检验
  2. Top-2 选股收益 — 模拟 POS_SIZE=2 的真实选股效果
  3. 因子间相关矩阵 — 冗余度分析
  4. A 股 vs QDII 拆分 — alpha 来源归因
  5. 因子稳定性 — 排名自相关 (rank autocorrelation)
  6. IC 衰减曲线 — 最优持有周期

用法:
  uv run python scripts/factor_alpha_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor

# ── 常量 ──────────────────────────────────────────────────
FREQ = 3  # 调仓频率
POS_SIZE = 2  # 持仓数量
COMMISSION = 0.0002  # 单边佣金 2bp

QDII_CODES = {"513100", "513500", "159920", "513050", "513130"}

IC_HORIZONS = [1, 3, 5, 10, 20]  # IC 衰减检验的前瞻天数

# ── 分隔线工具 ────────────────────────────────────────────
SEP = "=" * 80
THIN = "-" * 80


def load_data():
    """加载 OHLCV 数据"""
    print(f"\n{SEP}")
    print("Step 1: 加载数据")
    print(SEP)
    data_dir = str(PROJECT_ROOT / "raw" / "ETF" / "daily")
    loader = DataLoader(data_dir=data_dir)
    ohlcv = loader.load_ohlcv(start_date="2020-01-01", end_date="2025-12-31")
    close = ohlcv["close"]
    print(f"  日期范围: {close.index[0].date()} ~ {close.index[-1].date()}")
    print(f"  ETF 数量: {len(close.columns)}")
    print(f"  交易日数: {len(close)}")

    # 标记 A 股 vs QDII
    a_share_cols = [c for c in close.columns if c not in QDII_CODES]
    qdii_cols = [c for c in close.columns if c in QDII_CODES]
    print(f"  A 股: {len(a_share_cols)} 只, QDII: {len(qdii_cols)} 只")
    return ohlcv, a_share_cols, qdii_cols


def compute_factors(ohlcv):
    """计算所有因子并标准化"""
    print(f"\n{SEP}")
    print("Step 2: 计算因子")
    print(SEP)
    lib = PreciseFactorLibrary()
    raw = lib.compute_all_factors(prices=ohlcv)

    factor_names = sorted(lib.list_factors().keys())
    factors_dict = {}
    for name in factor_names:
        factors_dict[name] = raw[name]

    print(f"  原始因子数: {len(factors_dict)}")

    # 标准化
    proc = CrossSectionProcessor(verbose=False)
    std_factors = proc.process_all_factors(factors_dict)
    print(f"  标准化完成: {len(std_factors)} 个因子")

    # 因子元数据
    metadata = lib.list_factors()

    return std_factors, metadata, factor_names


def compute_forward_returns(close: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    计算前瞻 N 日收益率 (用于分析, 非实盘)
    forward_ret[t] = close[t+N] / close[t] - 1
    """
    return close.shift(-periods) / close - 1


def spearman_ic_series(factor_df: pd.DataFrame, return_df: pd.DataFrame) -> pd.Series:
    """
    逐日计算横截面 Spearman IC

    返回: pd.Series(index=date, values=IC)
    """
    common_idx = factor_df.index.intersection(return_df.index)
    common_cols = factor_df.columns.intersection(return_df.columns)
    f = factor_df.loc[common_idx, common_cols]
    r = return_df.loc[common_idx, common_cols]

    ics = []
    dates = []
    for dt in common_idx:
        fv = f.loc[dt].values
        rv = r.loc[dt].values
        mask = np.isfinite(fv) & np.isfinite(rv)
        n_valid = mask.sum()
        if n_valid < 5:  # 至少 5 个有效观测
            continue
        corr, _ = stats.spearmanr(fv[mask], rv[mask])
        if np.isfinite(corr):
            ics.append(corr)
            dates.append(dt)

    return pd.Series(ics, index=dates, name="IC")


def single_factor_ic_report(std_factors, close, factor_names, metadata):
    """单因子 IC 综合报告 (含多重检验校正)"""
    print(f"\n{SEP}")
    print("Step 3: 单因子 IC 分析 (forward {}-day returns, 含 Bonferroni 校正)".format(FREQ))
    print(SEP)

    fwd_ret = compute_forward_returns(close, FREQ)
    results = []

    for name in factor_names:
        fdf = std_factors[name]
        ic_s = spearman_ic_series(fdf, fwd_ret)

        if len(ic_s) < 30:
            print(f"  {name}: 有效 IC 数 < 30, 跳过")
            continue

        mean_ic = ic_s.mean()
        std_ic = ic_s.std()
        ic_ir = mean_ic / std_ic if std_ic > 1e-10 else 0.0
        hit_rate = (ic_s > 0).mean()
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_s))) if std_ic > 1e-10 else 0.0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ic_s) - 1))

        # 因子方向
        direction = metadata[name].direction
        bounded = metadata[name].bounded
        dimension = metadata[name].dimension

        results.append({
            "因子": name,
            "维度": dimension,
            "方向": direction,
            "有界": "Y" if bounded else "N",
            "Mean_IC": mean_ic,
            "Std_IC": std_ic,
            "IC_IR": ic_ir,
            "命中率": hit_rate,
            "t_stat": t_stat,
            "p_value": p_value,
            "有效天数": len(ic_s),
        })

    df = pd.DataFrame(results)

    # ── 多重检验校正 (Bonferroni) ──
    from statsmodels.stats.multitest import multipletests

    p_values = df["p_value"].values
    n_tests = len(p_values)

    # Bonferroni 校正
    reject_bonf, p_adj_bonf, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")

    # FDR (Benjamini-Hochberg) 校正 (可选, 更宽松)
    reject_fdr, p_adj_fdr, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")

    df["p_adj_Bonf"] = p_adj_bonf
    df["显著_Bonf"] = ["***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")) for p in p_adj_bonf]
    df["p_adj_FDR"] = p_adj_fdr
    df["显著_FDR"] = ["***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")) for p in p_adj_fdr]

    df = df.sort_values("IC_IR", ascending=False, key=abs)

    print(f"\n{'因子':<35s} {'MeanIC':>8s} {'IC_IR':>7s} {'命中率':>6s} {'t_stat':>7s} {'Bonf':>4s} {'FDR':>4s}")
    print(THIN)
    for _, row in df.iterrows():
        print(
            f"  {row['因子']:<33s} "
            f"{row['Mean_IC']:>+8.4f} {row['IC_IR']:>7.3f} {row['命中率']:>6.1%} "
            f"{row['t_stat']:>7.2f} {row['显著_Bonf']:>4s} {row['显著_FDR']:>4s}"
        )

    # 简要结论
    bonf_sig = df[df["p_adj_Bonf"] < 0.05]
    fdr_sig = df[df["p_adj_FDR"] < 0.05]
    print(f"\n  显著因子 (Bonferroni p_adj<0.05): {len(bonf_sig)}/{len(df)}")
    print(f"  显著因子 (FDR p_adj<0.05): {len(fdr_sig)}/{len(df)}")
    print(f"  Bonferroni 阈值: p < {0.05 / n_tests:.4f} (严格)")
    if len(bonf_sig) > 0:
        print(f"  最强 (Bonf): {bonf_sig.iloc[0]['因子']} (IC_IR={bonf_sig.iloc[0]['IC_IR']:.3f})")

    return df


def long_short_sharpe_backtest(std_factors, close, factor_names, metadata):
    """
    多空配对 Top-2 回测 (市场中性策略)

    策略:
        Long: 因子得分最高 2 只 ETF
        Short: 因子得分最低 2 只 ETF
        净头寸 = (long_ret - short_ret) / 2

    目的: 消除 Top-2 单向回测的过拟合风险
    """
    print(f"\n{SEP}")
    print("Step 4A: 多空配对回测 (Long-Short Market Neutral, FREQ={}, POS_SIZE={})".format(FREQ, POS_SIZE))
    print(SEP)

    daily_ret = close.pct_change()
    dates = close.index
    n_dates = len(dates)

    results = []

    for name in factor_names:
        fdf = std_factors[name]
        direction = metadata[name].direction

        # Long-Short 策略不关心方向 (多空对冲)
        portfolio_long_rets = []
        portfolio_short_rets = []
        rebalance_days = list(range(0, n_dates - FREQ, FREQ))

        for rb_idx in rebalance_days:
            dt = dates[rb_idx]
            scores = fdf.loc[dt].dropna()
            if len(scores) < POS_SIZE * 2:  # 至少需要 4 只 ETF
                continue

            # Long: 得分最高 2 只, Short: 得分最低 2 只
            long_selected = scores.nlargest(POS_SIZE).index
            short_selected = scores.nsmallest(POS_SIZE).index

            # T+1 开始持有
            hold_start = rb_idx + 1
            hold_end = min(rb_idx + FREQ + 1, n_dates)
            if hold_start >= n_dates:
                continue

            # Long / Short 组合收益
            long_ret = daily_ret.iloc[hold_start:hold_end][list(long_selected)].mean(axis=1)
            short_ret = daily_ret.iloc[hold_start:hold_end][list(short_selected)].mean(axis=1)

            portfolio_long_rets.append(long_ret)
            portfolio_short_rets.append(short_ret)

        if not portfolio_long_rets:
            continue

        # 多空组合收益 (市场中性)
        long_ret_series = pd.concat(portfolio_long_rets)
        short_ret_series = pd.concat(portfolio_short_rets)
        ls_ret_series = (long_ret_series - short_ret_series) / 2  # 多空对冲

        ls_cum = (1 + ls_ret_series).cumprod()

        total_return = ls_cum.iloc[-1] - 1 if len(ls_cum) > 0 else 0
        ann_return = (1 + total_return) ** (252 / len(ls_ret_series)) - 1 if len(ls_ret_series) > 0 else 0
        ann_vol = ls_ret_series.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0
        max_dd = (ls_cum / ls_cum.cummax() - 1).min()
        win_rate = (ls_ret_series > 0).mean()

        results.append({
            "因子": name,
            "方向": direction,
            "LS总收益": total_return,
            "LS年化": ann_return,
            "LS波动": ann_vol,
            "LS_Sharpe": sharpe,
            "LS最大回撤": max_dd,
            "LS胜率": win_rate,
            "交易天数": len(ls_ret_series),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("LS_Sharpe", ascending=False)

    print(f"\n{'因子':<35s} {'方向':<14s} {'LS_Sharpe':>10s} {'LS年化':>8s} {'LS胜率':>7s} {'LS回撤':>8s}")
    print(THIN)
    for _, row in df.iterrows():
        print(
            f"  {row['因子']:<33s} {row['方向']:<14s} "
            f"{row['LS_Sharpe']:>+10.2f} {row['LS年化']:>+8.1%} {row['LS胜率']:>7.1%} "
            f"{row['LS最大回撤']:>8.1%}"
        )

    print(f"\n  筛选标准: LS_Sharpe > 0.5 (市场中性策略)")
    qualified = df[df["LS_Sharpe"] > 0.5]
    print(f"  通过数: {len(qualified)}/{len(df)}")

    return df


def top2_selection_return(std_factors, close, factor_names, metadata):
    """
    模拟 Top-2 选股收益 (匹配 POS_SIZE=2, FREQ=3)
    每 FREQ 天调仓, 选因子得分最高/最低的 2 只 ETF, 等权持有

    ⚠️ 警告: 单向回测容易过拟合, 仅供参考. 请结合 Long-Short Sharpe 综合判断
    """
    print(f"\n{SEP}")
    print("Step 4B: Top-2 单向选股回测 (FREQ={}, POS_SIZE={})".format(FREQ, POS_SIZE))
    print("  ⚠️  警告: 单向回测易过拟合, 需结合多空 Sharpe 验证")
    print(SEP)

    # 日收益率
    daily_ret = close.pct_change()
    dates = close.index
    n_dates = len(dates)

    results = []

    for name in factor_names:
        fdf = std_factors[name]
        direction = metadata[name].direction

        # 确定排序方向: high_is_good → 选最大, low_is_good → 选最小
        ascending = direction == "low_is_good"

        # 模拟调仓
        portfolio_rets = []
        rebalance_days = list(range(0, n_dates - FREQ, FREQ))

        for rb_idx in rebalance_days:
            dt = dates[rb_idx]
            scores = fdf.loc[dt]
            valid_scores = scores.dropna()
            if len(valid_scores) < POS_SIZE:
                continue

            # 选 Top-2
            if ascending:
                selected = valid_scores.nsmallest(POS_SIZE).index
            else:
                selected = valid_scores.nlargest(POS_SIZE).index

            # 持有 FREQ 天的收益
            hold_start = rb_idx + 1  # T+1 开始 (避免 lookahead)
            hold_end = min(rb_idx + FREQ + 1, n_dates)
            if hold_start >= n_dates:
                continue

            period_ret = daily_ret.iloc[hold_start:hold_end][list(selected)].mean(axis=1)
            portfolio_rets.append(period_ret)

        if not portfolio_rets:
            continue

        port_ret = pd.concat(portfolio_rets)
        cum_ret = (1 + port_ret).cumprod()

        total_return = cum_ret.iloc[-1] - 1 if len(cum_ret) > 0 else 0
        ann_return = (1 + total_return) ** (252 / len(port_ret)) - 1 if len(port_ret) > 0 else 0
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()

        results.append({
            "因子": name,
            "方向": direction,
            "总收益": total_return,
            "年化收益": ann_return,
            "年化波动": ann_vol,
            "Sharpe": sharpe,
            "最大回撤": max_dd,
            "交易天数": len(port_ret),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("总收益", ascending=False)

    print(f"\n{'因子':<35s} {'方向':<14s} {'总收益':>8s} {'年化':>8s} {'Sharpe':>7s} {'最大回撤':>8s}")
    print(THIN)
    for _, row in df.iterrows():
        print(
            f"  {row['因子']:<33s} {row['方向']:<14s} "
            f"{row['总收益']:>+8.1%} {row['年化收益']:>+8.1%} {row['Sharpe']:>7.2f} "
            f"{row['最大回撤']:>8.1%}"
        )

    # 基准: 等权全部 43 只 ETF
    eq_ret = daily_ret.mean(axis=1)
    eq_cum = (1 + eq_ret).cumprod()
    eq_total = eq_cum.iloc[-1] - 1
    print(f"\n  基准 (等权43ETF): 总收益 {eq_total:+.1%}")

    return df


def factor_correlation_matrix(std_factors, factor_names):
    """因子间截面相关矩阵"""
    print(f"\n{SEP}")
    print("Step 5: 因子间相关矩阵 (日均 rank 相关)")
    print(SEP)

    # 逐日计算因子对之间的 rank 相关, 取均值
    n = len(factor_names)
    corr_matrix = np.zeros((n, n))

    # 取所有因子的日均值构建简化版
    # 使用某一天的截面数据计算因子间相关度
    # 更准确: 对每天的截面数据做 rank 相关, 然后取平均
    sample_dates = list(std_factors[factor_names[0]].index[::5])  # 每 5 天采样一次

    corr_accum = np.zeros((n, n))
    count = 0

    for dt in sample_dates:
        cross_section = pd.DataFrame(index=std_factors[factor_names[0]].columns)
        all_valid = True
        for i, name in enumerate(factor_names):
            if dt not in std_factors[name].index:
                all_valid = False
                break
            cross_section[name] = std_factors[name].loc[dt]

        if not all_valid:
            continue

        cs_clean = cross_section.dropna()
        if len(cs_clean) < 10:
            continue

        # Spearman rank correlation
        rc = cs_clean.rank().corr()
        corr_accum += rc.values
        count += 1

    if count > 0:
        corr_matrix = corr_accum / count

    corr_df = pd.DataFrame(corr_matrix, index=factor_names, columns=factor_names)

    # 找高度相关的因子对 (|corr| > 0.6)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = corr_df.iloc[i, j]
            if abs(c) > 0.6:
                pairs.append((factor_names[i], factor_names[j], c))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n  采样天数: {count}")
    print(f"\n  高度相关因子对 (|corr| > 0.6):")
    if not pairs:
        print("    无 — 因子正交性良好")
    else:
        for a, b, c in pairs:
            print(f"    {a:<35s} ↔ {b:<35s}  corr={c:+.3f}")

    # 中等相关
    mid_pairs = [(a, b, c) for i in range(n) for j in range(i + 1, n)
                 for a, b, c in [(factor_names[i], factor_names[j], corr_df.iloc[i, j])]
                 if 0.4 < abs(c) <= 0.6]
    mid_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if mid_pairs:
        print(f"\n  中等相关因子对 (0.4 < |corr| ≤ 0.6):")
        for a, b, c in mid_pairs:
            print(f"    {a:<35s} ↔ {b:<35s}  corr={c:+.3f}")

    return corr_df


def ashare_vs_qdii_ic(std_factors, close, factor_names, metadata, a_share_cols, qdii_cols):
    """A 股 vs QDII 因子 IC 对比"""
    print(f"\n{SEP}")
    print("Step 6: A 股 vs QDII 因子 IC 对比")
    print(SEP)

    fwd_ret_all = compute_forward_returns(close, FREQ)

    results = []
    for name in factor_names:
        fdf = std_factors[name]

        # A 股 IC
        a_cols = [c for c in a_share_cols if c in fdf.columns]
        if len(a_cols) >= 5:
            ic_a = spearman_ic_series(fdf[a_cols], fwd_ret_all[a_cols])
            mean_ic_a = ic_a.mean() if len(ic_a) > 0 else np.nan
            hit_a = (ic_a > 0).mean() if len(ic_a) > 0 else np.nan
        else:
            mean_ic_a = np.nan
            hit_a = np.nan

        # QDII IC (仅 5 只, IC 极不稳定, 仅供参考)
        q_cols = [c for c in qdii_cols if c in fdf.columns]
        if len(q_cols) >= 3:
            ic_q = spearman_ic_series(fdf[q_cols], fwd_ret_all[q_cols])
            mean_ic_q = ic_q.mean() if len(ic_q) > 0 else np.nan
            hit_q = (ic_q > 0).mean() if len(ic_q) > 0 else np.nan
        else:
            mean_ic_q = np.nan
            hit_q = np.nan

        # 全池 IC (参照)
        ic_all = spearman_ic_series(fdf, fwd_ret_all)
        mean_ic_all = ic_all.mean() if len(ic_all) > 0 else np.nan

        results.append({
            "因子": name,
            "全池IC": mean_ic_all,
            "A股IC": mean_ic_a,
            "A股命中率": hit_a,
            "QDII_IC": mean_ic_q,
            "QDII命中率": hit_q,
            "差异": (mean_ic_q - mean_ic_a) if np.isfinite(mean_ic_q) and np.isfinite(mean_ic_a) else np.nan,
        })

    df = pd.DataFrame(results)
    df = df.sort_values("全池IC", ascending=False, key=abs)

    print(f"\n{'因子':<35s} {'全池IC':>8s} {'A股IC':>8s} {'A股命中':>8s} {'QDII_IC':>8s} {'QDII命中':>8s}")
    print(THIN)
    for _, row in df.iterrows():
        a_ic = f"{row['A股IC']:>+8.4f}" if np.isfinite(row['A股IC']) else "    N/A "
        q_ic = f"{row['QDII_IC']:>+8.4f}" if np.isfinite(row['QDII_IC']) else "    N/A "
        a_hit = f"{row['A股命中率']:>8.1%}" if np.isfinite(row['A股命中率']) else "    N/A "
        q_hit = f"{row['QDII命中率']:>8.1%}" if np.isfinite(row['QDII命中率']) else "    N/A "
        print(f"  {row['因子']:<33s} {row['全池IC']:>+8.4f} {a_ic} {a_hit} {q_ic} {q_hit}")

    print(f"\n  注: QDII 仅 5 只, 截面 IC 极不稳定, 仅供参考方向")

    return df


def time_series_cv_ic(std_factors, close, factor_names, metadata, n_splits=3):
    """
    时序交叉验证 (Time-Series Cross-Validation)

    3-fold 滚动窗口: 每次用 2/3 训练, 1/3 测试
    要求: 所有 fold 的训练 + 测试 IC 都显著且同号

    目的: 消除全样本回测的过拟合, 验证因子 OOS 稳定性
    """
    print(f"\n{SEP}")
    print("Step 6A: 时序交叉验证 (Time-Series CV, {}-fold)".format(n_splits))
    print(SEP)

    fwd_ret = compute_forward_returns(close, FREQ)
    results = []

    for name in factor_names:
        fdf = std_factors[name]

        n = len(fdf)
        fold_size = n // n_splits

        train_ics = []
        test_ics = []

        for i in range(n_splits):
            # 滚动切分
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else n

            test_idx = list(range(test_start, test_end))
            train_idx = [j for j in range(n) if j not in test_idx]

            # 训练集 IC
            train_ic_series = spearman_ic_series(fdf.iloc[train_idx], fwd_ret.iloc[train_idx])
            train_ic = train_ic_series.mean() if len(train_ic_series) > 0 else np.nan

            # 测试集 IC (OOS)
            test_ic_series = spearman_ic_series(fdf.iloc[test_idx], fwd_ret.iloc[test_idx])
            test_ic = test_ic_series.mean() if len(test_ic_series) > 0 else np.nan

            train_ics.append(train_ic)
            test_ics.append(test_ic)

        # 筛选标准: 所有 fold 的 train/test IC 都同号且 > 0.02
        all_positive = all(ic > 0.02 for ic in train_ics + test_ics if np.isfinite(ic))
        all_negative = all(ic < -0.02 for ic in train_ics + test_ics if np.isfinite(ic))
        same_sign = all_positive or all_negative

        cv_passed = same_sign and (all_positive or all_negative)

        results.append({
            "因子": name,
            "Train_IC_mean": np.mean(train_ics),
            "Test_IC_mean": np.mean(test_ics),
            "Train_IC_std": np.std(train_ics),
            "Test_IC_std": np.std(test_ics),
            "CV通过": "✓" if cv_passed else "✗",
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Test_IC_mean", ascending=False, key=abs)

    print(f"\n{'因子':<35s} {'Train_IC':>9s} {'Test_IC':>9s} {'Test_Std':>8s} {'CV':>3s}")
    print(THIN)
    for _, row in df.iterrows():
        print(
            f"  {row['因子']:<33s} {row['Train_IC_mean']:>+9.4f} {row['Test_IC_mean']:>+9.4f} "
            f"{row['Test_IC_std']:>8.4f} {row['CV通过']:>3s}"
        )

    passed = df[df["CV通过"] == "✓"]
    print(f"\n  CV 通过: {len(passed)}/{len(df)} (要求所有 fold 的 IC 同号且显著)")

    return df


def factor_rank_stability(std_factors, factor_names):
    """因子排名稳定性 (rank autocorrelation)"""
    print(f"\n{SEP}")
    print("Step 7: 因子排名稳定性 (lag-{} rank autocorrelation)".format(FREQ))
    print(SEP)

    results = []
    for name in factor_names:
        fdf = std_factors[name]
        # 排名
        ranked = fdf.rank(axis=1, method="average")

        # 每 FREQ 天采样排名, 计算前后两期的 rank 相关
        sample_idx = list(range(0, len(ranked) - FREQ, FREQ))
        autocorrs = []
        for i in range(len(sample_idx) - 1):
            r1 = ranked.iloc[sample_idx[i]]
            r2 = ranked.iloc[sample_idx[i + 1]]
            mask = np.isfinite(r1) & np.isfinite(r2)
            if mask.sum() < 10:
                continue
            c, _ = stats.spearmanr(r1[mask], r2[mask])
            if np.isfinite(c):
                autocorrs.append(c)

        if autocorrs:
            mean_ac = np.mean(autocorrs)
            std_ac = np.std(autocorrs)
        else:
            mean_ac = np.nan
            std_ac = np.nan

        results.append({
            "因子": name,
            "Rank_AutoCorr": mean_ac,
            "Std": std_ac,
            "稳定性": "高" if mean_ac > 0.8 else ("中" if mean_ac > 0.5 else "低"),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("Rank_AutoCorr", ascending=False)

    print(f"\n{'因子':<35s} {'RankAutoCorr':>12s} {'Std':>8s} {'稳定性':>6s}")
    print(THIN)
    for _, row in df.iterrows():
        ac_str = f"{row['Rank_AutoCorr']:>12.4f}" if np.isfinite(row['Rank_AutoCorr']) else "         N/A"
        print(f"  {row['因子']:<33s} {ac_str} {row['Std']:>8.4f} {row['稳定性']:>6s}")

    print(f"\n  解读: AutoCorr > 0.8 = 排名稳定, 适合低频调仓 (FREQ={FREQ})")
    print(f"       AutoCorr < 0.5 = 排名变化快, 可能引入高换手")

    return df


def ic_decay_analysis(std_factors, close, factor_names, metadata):
    """IC 衰减分析: 不同前瞻天数下的 IC"""
    print(f"\n{SEP}")
    print("Step 8: IC 衰减分析 (horizons: {})".format(IC_HORIZONS))
    print(SEP)

    results = []
    for name in factor_names:
        fdf = std_factors[name]
        row = {"因子": name}
        for h in IC_HORIZONS:
            fwd = compute_forward_returns(close, h)
            ic_s = spearman_ic_series(fdf, fwd)
            row[f"IC_{h}d"] = ic_s.mean() if len(ic_s) > 0 else np.nan
        results.append(row)

    df = pd.DataFrame(results)

    # 找每个因子的最优 horizon
    ic_cols = [f"IC_{h}d" for h in IC_HORIZONS]
    df["最优Horizon"] = df[ic_cols].apply(lambda r: IC_HORIZONS[np.argmax(np.abs(r.values))], axis=1)
    df["最优IC"] = df[ic_cols].apply(lambda r: r.values[np.argmax(np.abs(r.values))], axis=1)

    df = df.sort_values("最优IC", ascending=False, key=abs)

    header = f"{'因子':<33s}"
    for h in IC_HORIZONS:
        header += f"  IC_{h}d"
    header += "  最优"
    print(f"\n{header}")
    print(THIN)
    for _, row in df.iterrows():
        line = f"  {row['因子']:<31s}"
        for h in IC_HORIZONS:
            v = row[f"IC_{h}d"]
            line += f"  {v:>+.4f}" if np.isfinite(v) else "     N/A"
        line += f"  {row['最优Horizon']}d"
        print(line)

    return df


def comprehensive_verdict(ic_df, top2_df, corr_df, stability_df, split_df, decay_df, metadata, factor_names, ls_df, cv_df):
    """综合评判 (结合 LS Sharpe + Time-Series CV)"""
    print(f"\n{SEP}")
    print("Step 9: 综合评判 (FDR + LS Sharpe + CV)")
    print(SEP)

    # Top 4 候选的因子
    top_combos = {
        "#1 (BT 109.4%)": ["CORRELATION_TO_MARKET_20D", "MAX_DD_60D", "MOM_20D", "SLOPE_20D", "VOL_RATIO_20D", "VORTEX_14D"],
        "#2 (BT 120.6%)": ["AMIHUD_ILLIQUIDITY", "CORRELATION_TO_MARKET_20D", "SHARPE_RATIO_20D", "SLOPE_20D", "SPREAD_PROXY", "TURNOVER_ACCEL_5_20", "VOL_RATIO_20D"],
        "#3 (BT 92.5%)": ["CORRELATION_TO_MARKET_20D", "MAX_DD_60D", "RELATIVE_STRENGTH_VS_MARKET_20D", "SLOPE_20D", "VOL_RATIO_20D", "VORTEX_14D"],
        "#4 (BT 57.6%)": ["ADX_14D", "BREAKOUT_20D", "CORRELATION_TO_MARKET_20D", "PRICE_POSITION_120D", "SLOPE_20D", "VORTEX_14D"],
    }

    # ── 9.1 因子价值总评 (新增 LS Sharpe + CV) ──
    print(f"\n  ── 9.1 单因子价值总评 (FDR + LS Sharpe + CV) ──")
    ic_lookup = {row["因子"]: row for _, row in ic_df.iterrows()}
    top2_lookup = {row["因子"]: row for _, row in top2_df.iterrows()}
    stab_lookup = {row["因子"]: row for _, row in stability_df.iterrows()}
    ls_lookup = {row["因子"]: row for _, row in ls_df.iterrows()}
    cv_lookup = {row["因子"]: row for _, row in cv_df.iterrows()}

    verdicts = []
    for name in factor_names:
        ic_row = ic_lookup.get(name, {})
        t2_row = top2_lookup.get(name, {})
        st_row = stab_lookup.get(name, {})
        ls_row = ls_lookup.get(name, {})
        cv_row = cv_lookup.get(name, {})

        mean_ic = ic_row.get("Mean_IC", np.nan)
        ic_ir = ic_row.get("IC_IR", np.nan)
        hit = ic_row.get("命中率", np.nan)
        p_adj_fdr = ic_row.get("p_adj_FDR", 1.0)
        total_ret = t2_row.get("总收益", np.nan)
        rank_ac = st_row.get("Rank_AutoCorr", np.nan)
        ls_sharpe = ls_row.get("LS_Sharpe", np.nan)
        cv_passed = cv_row.get("CV通过", "✗") == "✓"

        # 综合评分 (FDR 标准)
        score = 0
        notes = []

        # IC 显著性 (FDR Benjamini-Hochberg 校正)
        if p_adj_fdr < 0.01:
            score += 3
            notes.append("IC高度显著")
        elif p_adj_fdr < 0.05:
            score += 2
            notes.append("IC显著")
        elif p_adj_fdr < 0.1:
            score += 1
            notes.append("IC边缘")

        # LS Sharpe (市场中性策略收益)
        if np.isfinite(ls_sharpe) and ls_sharpe > 1.0:
            score += 3
            notes.append("LS优秀")
        elif np.isfinite(ls_sharpe) and ls_sharpe > 0.5:
            score += 2
            notes.append("LS良好")
        elif np.isfinite(ls_sharpe) and ls_sharpe > 0:
            score += 1
            notes.append("LS正")

        # Time-Series CV (OOS 稳定性)
        if cv_passed:
            score += 2
            notes.append("CV通过")

        # 稳定性
        if np.isfinite(rank_ac) and rank_ac > 0.8:
            score += 1
            notes.append("排名稳")

        # 方向一致性
        direction = metadata[name].direction
        if direction == "high_is_good" and mean_ic > 0:
            score += 1
        elif direction == "low_is_good" and mean_ic < 0:
            score += 1
        elif direction == "neutral":
            score += 0.5

        # 减分项
        if not metadata[name].production_ready:
            score -= 3
            notes.append("非生产级")

        # Top-2 收益 (参考指标, 不作为主要评分)
        if np.isfinite(total_ret) and total_ret > 0.3:
            notes.append(f"Top2={total_ret:.1%}")

        verdict = "强" if score >= 6 else ("中" if score >= 3 else ("弱" if score >= 1 else "无效"))
        verdicts.append({
            "因子": name,
            "综合评分": score,
            "评级": verdict,
            "LS_Sharpe": ls_sharpe,
            "CV通过": "✓" if cv_passed else "✗",
            "备注": ", ".join(notes) if notes else "-",
        })

    vdf = pd.DataFrame(verdicts).sort_values("综合评分", ascending=False)

    print(f"\n  {'因子':<35s} {'评分':>5s} {'评级':>4s} {'LS_Sharpe':>10s} {'CV':>3s} {'备注'}")
    print(f"  {THIN}")
    for _, row in vdf.iterrows():
        ls_str = f"{row['LS_Sharpe']:>+10.2f}" if np.isfinite(row['LS_Sharpe']) else "       N/A"
        print(f"  {row['因子']:<33s} {row['综合评分']:>5.1f} {row['评级']:>4s} {ls_str} {row['CV通过']:>3s}  {row['备注']}")

    # ── 9.2 Top 候选策略因子质量 ──
    print(f"\n  ── 9.2 Top 候选策略因子质量 ──")
    for combo_name, factors in top_combos.items():
        print(f"\n  {combo_name}:")
        combo_scores = []
        for f in factors:
            v = next((r for r in verdicts if r["因子"] == f), None)
            if v:
                combo_scores.append(v["综合评分"])
                ic_val = ic_lookup.get(f, {}).get("Mean_IC", np.nan)
                print(f"    {f:<35s} 评级={v['评级']:<4s} IC={ic_val:+.4f}" if np.isfinite(ic_val) else f"    {f:<35s} 评级={v['评级']:<4s} IC=  N/A")
        if combo_scores:
            avg_score = np.mean(combo_scores)
            min_score = np.min(combo_scores)
            print(f"    → 均分={avg_score:.1f}, 短板={min_score:.1f}")

    # ── 9.3 关键发现 ──
    print(f"\n  ── 9.3 关键发现 ──")

    # 无效因子
    ineffective = vdf[vdf["评级"] == "无效"]["因子"].tolist()
    if ineffective:
        print(f"  ⚠️  无效因子: {', '.join(ineffective)}")

    # 高度相关警告
    high_corr_pairs = []
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            if abs(corr_df.iloc[i, j]) > 0.7:
                high_corr_pairs.append((factor_names[i], factor_names[j], corr_df.iloc[i, j]))
    if high_corr_pairs:
        print(f"  ⚠️  高度冗余因子对 (|corr|>0.7):")
        for a, b, c in high_corr_pairs:
            print(f"      {a} ↔ {b} (corr={c:+.3f})")

    # 强因子
    strong = vdf[vdf["评级"] == "强"]["因子"].tolist()
    print(f"  ✓ 强因子: {', '.join(strong) if strong else '无'}")

    # 中等因子
    medium = vdf[vdf["评级"] == "中"]["因子"].tolist()
    print(f"  ○ 中等因子: {', '.join(medium) if medium else '无'}")


def orthogonal_set_validation(ic_df, corr_df, factor_names, metadata):
    """Step 10: 正交因子集 (active_factors) 验证"""
    print(f"\n{SEP}")
    print("Step 10: 正交因子集验证 (active_factors from config)")
    print(SEP)

    # 读取 active_factors
    config_path = PROJECT_ROOT / "configs" / "combo_wfo_config.yaml"
    if not config_path.exists():
        print("  ⚠️  配置文件不存在, 跳过正交验证")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    active_factors = config.get("active_factors")
    if not active_factors:
        print("  ⚠️  config 中未定义 active_factors, 跳过")
        return

    active_set = set(active_factors)
    excluded_set = set(factor_names) - active_set

    print(f"\n  正交因子集: {len(active_factors)}/{len(factor_names)} 个因子")
    print(f"  已排除: {sorted(excluded_set)}")

    # ── 10.1 正交集 IC 报告 ──
    print(f"\n  ── 10.1 正交集 IC 报告 ──")
    if ic_df is not None and len(ic_df) > 0:
        active_ic = ic_df[ic_df["因子"].isin(active_set)].copy()
        active_ic = active_ic.sort_values("IC_IR", ascending=False, key=abs)

        print(f"\n  {'因子':<35s} {'MeanIC':>8s} {'IC_IR':>7s} {'命中率':>6s} {'Bonf':>4s}")
        print(f"  {THIN}")
        for _, row in active_ic.iterrows():
            print(
                f"  {row['因子']:<33s} {row['Mean_IC']:>+8.4f} {row['IC_IR']:>7.3f} "
                f"{row['命中率']:>6.1%} {row['显著_Bonf']:>4s}"
            )

        sig_count = len(active_ic[active_ic["p_value"] < 0.05])
        total = len(active_ic)
        print(f"\n  显著因子 (p<0.05): {sig_count}/{total}")

    # ── 10.2 残留高相关因子对 ──
    print(f"\n  ── 10.2 残留高相关因子对 (|corr| > 0.7) ──")
    active_list = sorted(active_set & set(factor_names))
    high_pairs = []
    for i in range(len(active_list)):
        for j in range(i + 1, len(active_list)):
            fi, fj = active_list[i], active_list[j]
            if fi in corr_df.index and fj in corr_df.columns:
                c = corr_df.loc[fi, fj]
                if abs(c) > 0.7:
                    high_pairs.append((fi, fj, c))

    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if high_pairs:
        for a, b, c in high_pairs:
            print(f"    {a:<35s} ↔ {b:<35s}  corr={c:+.3f}")
        print(f"\n  ⚠️  {len(high_pairs)} 对残留高相关 — 组合中共存时信息冗余")
    else:
        print("    无 — 正交性良好")

    # ── 10.3 combo 空间对比 ──
    from math import comb
    n_full = len(factor_names)
    n_active = len(active_list)
    sizes = [2, 3, 4, 5, 6, 7]
    space_full = sum(comb(n_full, s) for s in sizes)
    space_active = sum(comb(n_active, s) for s in sizes)
    reduction = (1 - space_active / space_full) * 100 if space_full > 0 else 0

    print(f"\n  ── 10.3 Combo 空间 ──")
    print(f"    全量 ({n_full} 因子): {space_full:,} combo")
    print(f"    正交集 ({n_active} 因子): {space_active:,} combo")
    print(f"    缩减: {reduction:.1f}%")


def main():
    print("\n" + "█" * 80)
    print("  ETF 因子 Alpha 深度检验")
    print("  25 因子 × 43 ETF × 5 年 (2020-2025)")
    print("█" * 80)

    # Step 1: Load
    ohlcv, a_share_cols, qdii_cols = load_data()
    close = ohlcv["close"]

    # Step 2: Factors
    std_factors, metadata, factor_names = compute_factors(ohlcv)

    # Step 3: Single-factor IC (含多重检验校正)
    ic_df = single_factor_ic_report(std_factors, close, factor_names, metadata)

    # Step 4A: Long-Short Sharpe (市场中性)
    ls_df = long_short_sharpe_backtest(std_factors, close, factor_names, metadata)

    # Step 4B: Top-2 selection returns (单向, 易过拟合)
    top2_df = top2_selection_return(std_factors, close, factor_names, metadata)

    # Step 5: Correlation matrix
    corr_df = factor_correlation_matrix(std_factors, factor_names)

    # Step 6A: Time-Series CV (时序交叉验证)
    cv_df = time_series_cv_ic(std_factors, close, factor_names, metadata, n_splits=3)

    # Step 6B: A-share vs QDII
    split_df = ashare_vs_qdii_ic(std_factors, close, factor_names, metadata, a_share_cols, qdii_cols)

    # Step 7: Rank stability
    stability_df = factor_rank_stability(std_factors, factor_names)

    # Step 8: IC decay
    decay_df = ic_decay_analysis(std_factors, close, factor_names, metadata)

    # Step 9: Comprehensive verdict (结合 LS Sharpe + CV)
    comprehensive_verdict(ic_df, top2_df, corr_df, stability_df, split_df, decay_df, metadata, factor_names, ls_df, cv_df)

    # Step 10: Orthogonal set validation
    orthogonal_set_validation(ic_df, corr_df, factor_names, metadata)

    print(f"\n{SEP}")
    print("分析完成")
    print(SEP)


if __name__ == "__main__":
    main()
