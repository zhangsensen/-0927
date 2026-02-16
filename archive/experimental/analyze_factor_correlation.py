#!/usr/bin/env python3
"""
折溢价因子与现有17因子相关性矩阵计算
截面相关性分析 - 评估因子独立性
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

# 17个现有因子列表
ACTIVE_FACTORS = [
    "ADX_14D",
    "AMIHUD_ILLIQUIDITY",
    "BREAKOUT_20D",
    "CALMAR_RATIO_60D",
    "CORRELATION_TO_MARKET_20D",
    "GK_VOL_RATIO_20D",
    "MAX_DD_60D",
    "MOM_20D",
    "OBV_SLOPE_10D",
    "PRICE_POSITION_20D",
    "PRICE_POSITION_120D",
    "PV_CORR_20D",
    "SHARPE_RATIO_20D",
    "SLOPE_20D",
    "UP_DOWN_VOL_RATIO_20D",
    "VOL_RATIO_20D",
    "VORTEX_14D",
]


def load_premium_factor(factors_dir: Path) -> pd.DataFrame:
    """加载折溢价因子数据，计算偏离度，并整合为宽格式"""
    files = list(factors_dir.glob("premium_rate_*.parquet"))

    all_data = []
    for f in files:
        symbol = f.stem.replace("premium_rate_", "")
        df = pd.read_parquet(f)
        df["symbol"] = symbol
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 计算20日滚动偏离度 (premium_deviation)
        df = df.sort_values("trade_date")
        df["rolling_mean"] = (
            df["premium_rate"].rolling(window=20, min_periods=10).mean()
        )
        df["rolling_std"] = df["premium_rate"].rolling(window=20, min_periods=10).std()
        df["premium_deviation"] = (df["premium_rate"] - df["rolling_mean"]) / (
            df["rolling_std"] + 1e-8
        )

        all_data.append(df[["trade_date", "symbol", "premium_deviation"]])

    combined = pd.concat(all_data, ignore_index=True)
    # 转为宽格式: date x symbol
    premium_wide = combined.pivot(
        index="trade_date", columns="symbol", values="premium_deviation"
    )
    premium_wide.index = pd.to_datetime(premium_wide.index)
    return premium_wide


def load_existing_factors(factors_dir: Path) -> Dict[str, pd.DataFrame]:
    """加载现有17个因子数据"""
    factor_data = {}
    for factor_name in ACTIVE_FACTORS:
        fpath = factors_dir / f"{factor_name}.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # parquet已经是宽格式: index=trade_date, columns=symbols
            df.index = pd.to_datetime(df.index)
            factor_data[factor_name] = df
    return factor_data


def calculate_cross_sectional_correlation(
    premium_df: pd.DataFrame, factor_df: pd.DataFrame
) -> pd.Series:
    """
    计算截面相关性
    对于每一天，计算premium_deviation与factor的相关系数
    返回每日相关系数的时间序列
    """
    # 对齐日期和symbol
    common_dates = premium_df.index.intersection(factor_df.index)
    common_symbols = premium_df.columns.intersection(factor_df.columns)

    if len(common_dates) == 0 or len(common_symbols) == 0:
        return pd.Series(dtype=float)

    premium_aligned = premium_df.loc[common_dates, common_symbols]
    factor_aligned = factor_df.loc[common_dates, common_symbols]

    # 计算每日截面相关性
    daily_corr = []
    for date in common_dates:
        p_vals = premium_aligned.loc[date].values
        f_vals = factor_aligned.loc[date].values

        # 去除NaN
        mask = ~(np.isnan(p_vals) | np.isnan(f_vals))
        if mask.sum() >= 5:  # 至少需要5个样本
            corr = np.corrcoef(p_vals[mask], f_vals[mask])[0, 1]
            daily_corr.append({"date": date, "correlation": corr})

    return pd.DataFrame(daily_corr).set_index("date")["correlation"]


def calculate_factor_characteristics(corr_series: pd.Series) -> Dict:
    """计算相关性统计特征"""
    if len(corr_series) == 0:
        return {}

    return {
        "mean": corr_series.mean(),
        "std": corr_series.std(),
        "median": corr_series.median(),
        "abs_mean": corr_series.abs().mean(),
        "abs_median": corr_series.abs().median(),
        "max_abs": corr_series.abs().max(),
        "positive_ratio": (corr_series > 0).mean(),
        "strong_corr_ratio": (corr_series.abs() > 0.5).mean(),
        "moderate_corr_ratio": (corr_series.abs() > 0.3).mean(),
        "weak_corr_ratio": (corr_series.abs() < 0.1).mean(),
        "count": len(corr_series),
    }


def analyze_correlations(
    premium_dir: Path, existing_dir: Path, output_dir: Path
) -> None:
    """主分析函数"""
    print("=" * 80)
    print("折溢价因子(Premium Deviation) 与 现有17因子 相关性分析")
    print("=" * 80)

    # 加载数据
    print("\n[1/4] 加载折溢价因子数据...")
    premium_wide = load_premium_factor(premium_dir)
    print(
        f"      折溢价因子: {len(premium_wide)} 天 x {len(premium_wide.columns)} 只ETF"
    )

    print("\n[2/4] 加载现有17因子数据...")
    factor_data = load_existing_factors(existing_dir)
    print(f"      成功加载: {len(factor_data)} / {len(ACTIVE_FACTORS)} 个因子")

    # 计算相关性
    print("\n[3/4] 计算截面相关性...")
    results = []

    for factor_name, factor_df in factor_data.items():
        corr_series = calculate_cross_sectional_correlation(premium_wide, factor_df)
        if len(corr_series) > 0:
            stats = calculate_factor_characteristics(corr_series)
            stats["factor"] = factor_name
            results.append(stats)
            print(
                f"      {factor_name:30s} | 平均|相关|: {stats['abs_mean']:6.3f} | 样本: {stats['count']}天"
            )

    # 创建结果表
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("abs_mean", ascending=False)

    # 保存详细结果
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        output_dir / "correlation_matrix.csv", index=False, float_format="%.4f"
    )

    # 打印报告
    print("\n[4/4] 相关性分析报告")
    print("=" * 80)

    # 相关性等级分类
    high_corr = results_df[results_df["abs_mean"] > 0.3]
    moderate_corr = results_df[
        (results_df["abs_mean"] >= 0.1) & (results_df["abs_mean"] <= 0.3)
    ]
    low_corr = results_df[results_df["abs_mean"] < 0.1]

    print(f"\n高相关性因子 (|ρ| > 0.3): {len(high_corr)} 个")
    if len(high_corr) > 0:
        for _, row in high_corr.iterrows():
            direction = "正相关" if row["mean"] > 0 else "负相关"
            print(
                f"  ⚠️  {row['factor']:30s} | ρ={row['mean']:+.3f} | |ρ|={row['abs_mean']:.3f} | {direction}"
            )
    else:
        print("  ✓ 无")

    print(f"\n中等相关性因子 (0.1 ≤ |ρ| ≤ 0.3): {len(moderate_corr)} 个")
    for _, row in moderate_corr.iterrows():
        direction = "正相关" if row["mean"] > 0 else "负相关"
        print(
            f"  ~  {row['factor']:30s} | ρ={row['mean']:+.3f} | |ρ|={row['abs_mean']:.3f} | {direction}"
        )

    print(f"\n低相关性因子 (|ρ| < 0.1): {len(low_corr)} 个")
    for _, row in low_corr.iterrows():
        direction = "正相关" if row["mean"] > 0 else "负相关"
        print(
            f"  ✓  {row['factor']:30s} | ρ={row['mean']:+.3f} | |ρ|={row['abs_mean']:.3f} | {direction}"
        )

    # 总体评估
    print("\n" + "=" * 80)
    print("总体评估")
    print("=" * 80)

    avg_abs_corr = results_df["abs_mean"].mean()
    median_abs_corr = results_df["abs_mean"].median()

    print(f"\n平均绝对相关性: {avg_abs_corr:.3f}")
    print(f"中位数绝对相关性: {median_abs_corr:.3f}")

    if avg_abs_corr < 0.1:
        independence_level = "独立性强 ✓✓✓"
        recommendation = "可以直接加入WFO因子库"
    elif avg_abs_corr < 0.2:
        independence_level = "独立性良好 ✓✓"
        recommendation = "建议加入，可能带来增量信息"
    elif avg_abs_corr < 0.3:
        independence_level = "独立性中等 ~"
        recommendation = "可考虑正交化后加入，或仅作为筛选条件"
    else:
        independence_level = "独立性弱 ✗"
        recommendation = "不建议直接加入，存在较强冗余"

    print(f"\n独立性评级: {independence_level}")
    print(f"建议: {recommendation}")

    # 特别关注点
    print("\n特别关注点:")

    # 检查与动量因子的相关性
    momentum_factors = ["MOM_20D", "SLOPE_20D", "BREAKOUT_20D"]
    momentum_corr = results_df[results_df["factor"].isin(momentum_factors)][
        "abs_mean"
    ].mean()
    print(
        f"  • 与动量因子相关性: {momentum_corr:.3f} ({'高' if momentum_corr > 0.3 else '中等' if momentum_corr > 0.1 else '低'})"
    )

    # 检查与波动率因子的相关性
    vol_factors = ["VOL_RATIO_20D", "GK_VOL_RATIO_20D", "VORTEX_14D", "ADX_14D"]
    vol_corr = results_df[results_df["factor"].isin(vol_factors)]["abs_mean"].mean()
    print(
        f"  • 与波动率因子相关性: {vol_corr:.3f} ({'高' if vol_corr > 0.3 else '中等' if vol_corr > 0.1 else '低'})"
    )

    # 检查与成交量因子的相关性
    volume_factors = ["OBV_SLOPE_10D", "UP_DOWN_VOL_RATIO_20D", "AMIHUD_ILLIQUIDITY"]
    volume_corr = results_df[results_df["factor"].isin(volume_factors)][
        "abs_mean"
    ].mean()
    print(
        f"  • 与成交量因子相关性: {volume_corr:.3f} ({'高' if volume_corr > 0.3 else '中等' if volume_corr > 0.1 else '低'})"
    )

    print(f"\n详细结果已保存: {output_dir / 'correlation_matrix.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    PREMIUM_DIR = Path("raw/ETF/factors")
    EXISTING_DIR = Path("results/run_20260212_005855/factors")
    OUTPUT_DIR = Path("results/premium_factor_correlation")

    analyze_correlations(PREMIUM_DIR, EXISTING_DIR, OUTPUT_DIR)
