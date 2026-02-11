#!/usr/bin/env python3
"""
验证"38只A股ETF截面alpha本质一维"假设

两项验证:
  1) 截面收益 PCA: PC1 解释度是否长期 >50%
  2) S1 因子 RankIC 相关性: 4个因子是否高度同向 (刻画同一 latent trend)

输出: 终端表格 + results/alpha_dimension_analysis/ 下的图表
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from etf_strategy.core.data_loader import DataLoader
from etf_strategy.core.precise_factor_library_v2 import PreciseFactorLibrary
from etf_strategy.core.cross_section_processor import CrossSectionProcessor


def load_data(config):
    """Load OHLCV and compute factors."""
    loader = DataLoader(
        data_dir=config["data"].get("data_dir"),
        cache_dir=config["data"].get("cache_dir"),
    )
    ohlcv = loader.load_ohlcv(
        etf_codes=config["data"]["symbols"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
    )
    return ohlcv


def pca_analysis(ohlcv, qdii_tickers):
    """Cross-sectional return PCA analysis."""
    print("\n" + "=" * 80)
    print("验证 1: 截面收益 PCA — PC1 解释度")
    print("=" * 80)

    # Extract close prices
    close = ohlcv["close"].copy()

    # Filter to A-share only
    a_share_cols = [c for c in close.columns if c not in qdii_tickers]
    close = close[a_share_cols]

    # Daily returns
    returns = close.pct_change().dropna(how="all")
    # Drop columns with too many NaNs (late-IPO ETFs)
    min_obs = int(len(returns) * 0.5)
    returns = returns.dropna(axis=1, thresh=min_obs)
    returns = returns.fillna(0)

    print(f"  ETF数量: {returns.shape[1]}")
    print(f"  日期范围: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"  交易日数: {len(returns)}")

    # Rolling PCA: 60-day windows
    window = 60
    pc1_ratios = []
    pc2_ratios = []
    dates = []

    from sklearn.decomposition import PCA

    for i in range(window, len(returns)):
        window_ret = returns.iloc[i - window : i].values
        # Skip if too many zeros
        if np.std(window_ret) < 1e-10:
            continue
        pca = PCA(n_components=min(5, window_ret.shape[1]))
        pca.fit(window_ret)
        pc1_ratios.append(pca.explained_variance_ratio_[0])
        pc2_ratios.append(pca.explained_variance_ratio_[1])
        dates.append(returns.index[i])

    pc1_series = pd.Series(pc1_ratios, index=dates)
    pc2_series = pd.Series(pc2_ratios, index=dates)

    print(f"\n  滚动 {window}日 PCA 结果:")
    print(f"  PC1 解释度 — 均值: {pc1_series.mean():.1%}, 中位数: {pc1_series.median():.1%}")
    print(f"  PC2 解释度 — 均值: {pc2_series.mean():.1%}, 中位数: {pc2_series.median():.1%}")
    print(f"  PC1 > 50% 的比例: {(pc1_series > 0.5).mean():.1%}")
    print(f"  PC1 > 40% 的比例: {(pc1_series > 0.4).mean():.1%}")
    print(f"  PC1+PC2 > 60% 的比例: {((pc1_series + pc2_series) > 0.6).mean():.1%}")

    # Yearly breakdown
    yearly = pc1_series.groupby(pc1_series.index.year).agg(["mean", "median", "min", "max"])
    yearly.columns = ["均值", "中位数", "最小", "最大"]
    print(f"\n  年度 PC1 解释度:")
    for year, row in yearly.iterrows():
        print(f"    {year}: 均值={row['均值']:.1%}  中位={row['中位数']:.1%}  "
              f"范围=[{row['最小']:.1%}, {row['最大']:.1%}]")

    return pc1_series, pc2_series


def factor_ic_correlation(ohlcv, config, qdii_tickers):
    """S1 factor RankIC correlation analysis."""
    print("\n" + "=" * 80)
    print("验证 2: S1 因子 RankIC 相关性 — 是否刻画同一 latent trend")
    print("=" * 80)

    S1_FACTORS = ["ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"]

    # Compute factors
    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    processor = CrossSectionProcessor(verbose=False)

    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    std_factors = processor.process_all_factors(raw_factors)

    # Filter to A-share only
    a_share_cols = [c for c in ohlcv["close"].columns if c not in qdii_tickers]

    close = ohlcv["close"][a_share_cols]
    fwd_ret = close.pct_change().shift(-1)  # Forward 1-day return for IC calc

    # Compute daily RankIC for each S1 factor
    print(f"\n  S1 因子: {S1_FACTORS}")

    daily_ics = {}
    for fname in S1_FACTORS:
        if fname not in std_factors:
            print(f"  ⚠️ {fname} not found in computed factors, skipping")
            continue
        factor_vals = std_factors[fname][a_share_cols]
        ics = []
        dates = []
        for date in factor_vals.index:
            if date not in fwd_ret.index:
                continue
            f_row = factor_vals.loc[date].dropna()
            r_row = fwd_ret.loc[date].reindex(f_row.index).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 10:
                continue
            ic = f_row[common].rank().corr(r_row[common].rank())
            if not np.isnan(ic):
                ics.append(ic)
                dates.append(date)
        daily_ics[fname] = pd.Series(ics, index=dates)

    # RankIC summary
    print(f"\n  因子 RankIC 汇总 (日频):")
    print(f"  {'因子':<25} {'均值IC':>8} {'IC>0比例':>8} {'IC_IR':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for fname in S1_FACTORS:
        if fname not in daily_ics:
            continue
        ic = daily_ics[fname]
        mean_ic = ic.mean()
        pos_rate = (ic > 0).mean()
        ic_ir = ic.mean() / ic.std() if ic.std() > 0 else 0
        print(f"  {fname:<25} {mean_ic:>+8.4f} {pos_rate:>8.1%} {ic_ir:>8.3f}")

    # Pairwise IC correlation (are the factors measuring the same thing?)
    print(f"\n  因子间 RankIC 时间序列相关性 (Pearson):")
    ic_df = pd.DataFrame(daily_ics)
    ic_df = ic_df.dropna()
    corr_matrix = ic_df.corr()

    pairs = []
    for i, f1 in enumerate(S1_FACTORS):
        for f2 in S1_FACTORS[i + 1 :]:
            if f1 in corr_matrix.columns and f2 in corr_matrix.columns:
                r = corr_matrix.loc[f1, f2]
                pairs.append((f1, f2, r))
                print(f"  {f1:<25} × {f2:<25} = {r:+.3f}")

    avg_corr = np.mean([p[2] for p in pairs]) if pairs else 0
    print(f"\n  平均两两相关性: {avg_corr:+.3f}")

    if avg_corr > 0.5:
        print(f"  → 高相关 (>0.5): 4个因子高度同向，支持'同一 latent trend'假设")
    elif avg_corr > 0.3:
        print(f"  → 中等相关 (0.3~0.5): 部分重叠但有独立信息")
    else:
        print(f"  → 低相关 (<0.3): 4个因子各有独立信息，'一维'假设不成立")

    # Also check: factor cross-sectional rank correlation (are ranks similar across assets?)
    print(f"\n  因子截面 Rank 相关性 (时间平均):")
    rank_corrs = {}
    for i, f1 in enumerate(S1_FACTORS):
        for f2 in S1_FACTORS[i + 1 :]:
            if f1 not in std_factors or f2 not in std_factors:
                continue
            v1 = std_factors[f1][a_share_cols]
            v2 = std_factors[f2][a_share_cols]
            daily_xcorr = []
            for date in v1.index:
                if date not in v2.index:
                    continue
                r1 = v1.loc[date].dropna().rank()
                r2 = v2.loc[date].reindex(r1.index).dropna().rank()
                common = r1.index.intersection(r2.index)
                if len(common) < 10:
                    continue
                c = r1[common].corr(r2[common])
                if not np.isnan(c):
                    daily_xcorr.append(c)
            avg_xc = np.mean(daily_xcorr) if daily_xcorr else 0
            rank_corrs[(f1, f2)] = avg_xc
            print(f"  {f1:<25} × {f2:<25} = {avg_xc:+.3f}")

    avg_rank_corr = np.mean(list(rank_corrs.values())) if rank_corrs else 0
    print(f"\n  平均截面 Rank 相关性: {avg_rank_corr:+.3f}")
    if avg_rank_corr > 0.4:
        print(f"  → 高截面重叠: 因子选出来的ETF高度重合，信息冗余，支持'一维'假设")
    elif avg_rank_corr > 0.2:
        print(f"  → 中等截面重叠: 部分重合，因子有一定互补")
    else:
        print(f"  → 低截面重叠: 因子选出不同ETF，互补性强，'一维'假设不成立")

    return daily_ics, corr_matrix


def all_factor_ic_analysis(ohlcv, config, qdii_tickers):
    """Extended analysis: all 17 active factors' cross-sectional rank correlations."""
    print("\n" + "=" * 80)
    print("验证 2b: 全部活跃因子截面 Rank 相关性矩阵 (验证信息维度)")
    print("=" * 80)

    active_factors = config.get("active_factors", [])
    print(f"  活跃因子数: {len(active_factors)}")

    factor_lib = PreciseFactorLibrary()
    raw_factors_df = factor_lib.compute_all_factors(ohlcv)
    processor = CrossSectionProcessor(verbose=False)

    factor_names = sorted(raw_factors_df.columns.get_level_values(0).unique().tolist())
    raw_factors = {fname: raw_factors_df[fname] for fname in factor_names}
    std_factors = processor.process_all_factors(raw_factors)

    a_share_cols = [c for c in ohlcv["close"].columns if c not in qdii_tickers]

    # Compute average cross-sectional rank correlation for all factor pairs
    available = [f for f in active_factors if f in std_factors]
    n = len(available)
    corr_mat = np.zeros((n, n))

    for i in range(n):
        corr_mat[i, i] = 1.0
        for j in range(i + 1, n):
            f1, f2 = available[i], available[j]
            v1 = std_factors[f1][a_share_cols]
            v2 = std_factors[f2][a_share_cols]
            daily_xcorr = []
            # Sample every 5th day for speed
            sample_dates = v1.index[::5]
            for date in sample_dates:
                if date not in v2.index:
                    continue
                r1 = v1.loc[date].dropna().rank()
                r2 = v2.loc[date].reindex(r1.index).dropna().rank()
                common = r1.index.intersection(r2.index)
                if len(common) < 10:
                    continue
                c = r1[common].corr(r2[common])
                if not np.isnan(c):
                    daily_xcorr.append(c)
            avg = np.mean(daily_xcorr) if daily_xcorr else 0
            corr_mat[i, j] = avg
            corr_mat[j, i] = avg

    corr_df = pd.DataFrame(corr_mat, index=available, columns=available)

    # PCA on the correlation matrix to find effective dimensionality
    from sklearn.decomposition import PCA
    eigenvalues = np.linalg.eigvalsh(corr_mat)[::-1]
    total = eigenvalues.sum()
    explained = eigenvalues / total

    print(f"\n  因子相关矩阵 PCA (有效维度分析):")
    cumulative = 0
    for i, (ev, ex) in enumerate(zip(eigenvalues[:8], explained[:8])):
        cumulative += ex
        marker = " ◀ 80%" if cumulative >= 0.8 and cumulative - ex < 0.8 else ""
        print(f"    PC{i+1}: 特征值={ev:.2f}  解释={ex:.1%}  累积={cumulative:.1%}{marker}")

    effective_dim = np.sum(eigenvalues > 1.0)  # Kaiser criterion
    print(f"\n  Kaiser准则有效维度 (特征值>1): {effective_dim}")
    print(f"  前3个PC累积解释: {explained[:3].sum():.1%}")

    # Print high-correlation pairs
    print(f"\n  高相关因子对 (|r| > 0.4):")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr_mat[i, j]) > 0.4:
                print(f"    {available[i]:<25} × {available[j]:<25} = {corr_mat[i,j]:+.3f}")

    return corr_df, eigenvalues


def main():
    config_path = ROOT / "configs" / "combo_wfo_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    qdii_tickers = config.get("universe", {}).get("qdii_tickers", [])

    print("=" * 80)
    print("📊 Alpha 维度假设验证: '38只A股ETF截面alpha本质一维'")
    print("=" * 80)

    # Load data
    print("\n加载数据...")
    ohlcv = load_data(config)

    # Verification 1: PCA on cross-sectional returns
    pc1, pc2 = pca_analysis(ohlcv, qdii_tickers)

    # Verification 2: S1 factor IC correlation
    daily_ics, ic_corr = factor_ic_correlation(ohlcv, config, qdii_tickers)

    # Verification 2b: All-factor correlation dimensionality
    factor_corr, eigenvalues = all_factor_ic_analysis(ohlcv, config, qdii_tickers)

    # Save results
    out_dir = ROOT / "results" / "alpha_dimension_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_corr.to_csv(out_dir / "factor_cross_section_corr.csv")
    pd.DataFrame({"pc1": pc1, "pc2": pc2}).to_csv(out_dir / "pca_timeseries.csv")
    pd.DataFrame({"eigenvalue": eigenvalues}).to_csv(out_dir / "factor_pca_eigenvalues.csv")

    # Final verdict
    print("\n" + "=" * 80)
    print("综合判定")
    print("=" * 80)

    pc1_median = pc1.median()
    kaiser_dim = np.sum(eigenvalues > 1.0)

    print(f"  收益截面 PC1 中位解释度: {pc1_median:.1%}", end="")
    if pc1_median > 0.5:
        print(" → ✅ 强一维 (>50%)")
    elif pc1_median > 0.35:
        print(" → ⚠️ 中等 (35-50%), 不完全一维但主成分占优")
    else:
        print(" → ❌ 多维 (<35%), 一维假设不成立")

    print(f"  因子空间有效维度 (Kaiser): {kaiser_dim}", end="")
    if kaiser_dim <= 3:
        print(f" → ✅ 低维 (≤3), 因子高度冗余")
    elif kaiser_dim <= 5:
        print(f" → ⚠️ 中等 (4-5), 有部分独立信息")
    else:
        print(f" → ❌ 高维 (>5), 因子空间丰富")

    print(f"\n  结果已保存至: {out_dir}/")


if __name__ == "__main__":
    main()
