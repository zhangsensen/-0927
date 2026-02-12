#!/usr/bin/env python3
"""
ETF折溢价率因子分层收益测试

将46只ETF按premium_deviation分为3组，测试组合收益
验证因子单调性和多空收益
"""

import sys

sys.path.insert(0, "src")

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_prepare_data(n_groups: int = 3) -> pd.DataFrame:
    """
    加载数据并计算分层收益

    Returns:
        DataFrame包含每日分组和未来收益
    """
    # 1. 加载折溢价率数据
    factors_path = Path("raw/ETF/factors")
    files = list(factors_path.glob("premium_rate_*.parquet"))

    all_factors = []
    for f in files:
        code = f.stem.replace("premium_rate_", "")
        df = pd.read_parquet(f)
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # 计算premium_deviation（20日均值偏离）
        df = df.sort_values("trade_date")
        df["premium_ma20"] = (
            df["premium_rate"].rolling(window=20, min_periods=10).mean()
        )
        df["premium_deviation"] = df["premium_rate"] - df["premium_ma20"]

        all_factors.append(
            df[["trade_date", "code", "premium_rate", "premium_deviation"]]
        )

    factor_df = pd.concat(all_factors, ignore_index=True)
    logger.info(f"加载了 {len(files)} 只ETF的因子数据")

    # 2. 加载价格数据计算未来收益
    import glob

    price_path = Path("raw/ETF/daily")

    all_prices = []
    for code in factor_df["code"].unique():
        pattern = f"{code}.*_daily_*.parquet"
        files = glob.glob(str(price_path / pattern))

        if not files:
            continue

        df = pd.read_parquet(files[0])
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df["code"] = code
        df = df.sort_values("trade_date")

        # 使用复权收盘价
        close_col = "adj_close" if "adj_close" in df.columns else "close"

        # 计算未来5日收益
        df["future_return_5d"] = df[close_col].shift(-5) / df[close_col] - 1

        all_prices.append(df[["trade_date", "code", close_col, "future_return_5d"]])

    price_df = pd.concat(all_prices, ignore_index=True)
    logger.info(f"加载了 {price_df['code'].nunique()} 只ETF的价格数据")

    # 3. 合并
    merged = pd.merge(factor_df, price_df, on=["trade_date", "code"], how="inner")

    logger.info(f"合并后数据: {len(merged)} 条记录")
    return merged


def calculate_group_returns(df: pd.DataFrame, n_groups: int = 3) -> pd.DataFrame:
    """
    每日按因子分层，计算各组收益

    Args:
        df: 包含因子和未来收益的DataFrame
        n_groups: 分组数

    Returns:
        每日各组收益的时间序列
    """
    daily_results = []

    for date, group in df.groupby("trade_date"):
        # 剔除缺失值
        valid = group.dropna(subset=["premium_deviation", "future_return_5d"])

        if len(valid) < n_groups * 3:  # 每组至少3只
            continue

        # 按因子排序分组
        valid = valid.sort_values("premium_deviation")
        valid["group"] = pd.qcut(
            valid["premium_deviation"],
            n_groups,
            labels=[f"G{i + 1}" for i in range(n_groups)],
        )

        # 计算每组等权收益
        group_returns = valid.groupby("group")["future_return_5d"].mean()

        result = {"trade_date": date}
        result.update(group_returns.to_dict())

        # 多空收益：低溢价组(G1) - 高溢价组(G3)
        if "G1" in group_returns and f"G{n_groups}" in group_returns:
            result["long_short"] = group_returns["G1"] - group_returns[f"G{n_groups}"]

        daily_results.append(result)

    return pd.DataFrame(daily_results)


def analyze_layered_returns(group_df: pd.DataFrame, n_groups: int = 3):
    """
    分析分层收益特征
    """
    logger.info("\n" + "=" * 80)
    logger.info("分层收益分析结果")
    logger.info("=" * 80)

    # 1. 各组平均收益
    logger.info("\n【各组平均5日收益】:")
    for i in range(n_groups):
        col = f"G{i + 1}"
        if col in group_df.columns:
            mean_ret = group_df[col].mean()
            logger.info(
                f"  {col} (低溢价→高溢价): {mean_ret:.4f} ({mean_ret * 100:.2f}%)"
            )

    # 2. 单调性检验
    means = [
        group_df[f"G{i + 1}"].mean()
        for i in range(n_groups)
        if f"G{i + 1}" in group_df.columns
    ]
    logger.info(f"\n【单调性】: {'✓ 递减' if means[0] > means[-1] else '✗ 非递减'}")
    logger.info(f"  G1 vs G{n_groups}差异: {means[0] - means[-1]:.4f}")

    # 3. 多空组合分析
    if "long_short" in group_df.columns:
        ls = group_df["long_short"]
        logger.info("\n【多空组合 (G1 - G3)】:")
        logger.info(f"  平均收益: {ls.mean():.4f} ({ls.mean() * 100:.2f}%)")
        logger.info(f"  收益标准差: {ls.std():.4f}")
        logger.info(f"  夏普比率(年化): {ls.mean() / ls.std() * np.sqrt(252 / 5):.2f}")
        logger.info(f"  胜率: {(ls > 0).mean():.2%}")

        # 累计收益
        cumulative = (1 + ls).cumprod()
        logger.info(f"  累计收益: {cumulative.iloc[-1]:.4f}")
        logger.info(f"  最大回撤: {(cumulative / cumulative.cummax() - 1).min():.4f}")

    # 4. 分年度表现
    group_df["year"] = pd.to_datetime(group_df["trade_date"]).dt.year
    logger.info("\n【分年度多空收益】:")
    yearly = group_df.groupby("year")["long_short"].mean()
    for year, ret in yearly.items():
        logger.info(f"  {year}: {ret:.4f} ({ret * 100:.2f}%)")

    return group_df


def analyze_factor_correlation():
    """
    分析折溢价因子与现有因子的相关性
    """
    logger.info("\n" + "=" * 80)
    logger.info("因子相关性分析")
    logger.info("=" * 80)

    # 加载折溢价因子
    factors_path = Path("raw/ETF/factors")
    files = list(factors_path.glob("premium_rate_*.parquet"))[:5]  # 抽样5只

    correlations = []

    for f in files:
        code = f.stem.replace("premium_rate_", "")

        # 读取折溢价因子
        prem_df = pd.read_parquet(f)
        prem_df["trade_date"] = pd.to_datetime(prem_df["trade_date"])
        prem_df = prem_df.sort_values("trade_date")
        prem_df["premium_deviation"] = (
            prem_df["premium_rate"] - prem_df["premium_rate"].rolling(20).mean()
        )

        # 读取价格数据计算动量
        import glob

        price_files = glob.glob(f"raw/ETF/daily/{code}.*_daily_*.parquet")
        if price_files:
            price_df = pd.read_parquet(price_files[0])
            price_df["trade_date"] = pd.to_datetime(
                price_df["trade_date"], format="%Y%m%d"
            )
            price_df = price_df.sort_values("trade_date")

            close_col = "adj_close" if "adj_close" in price_df.columns else "close"
            price_df["momentum_20d"] = (
                price_df[close_col] / price_df[close_col].shift(20) - 1
            )

            # 合并计算相关性
            merged = pd.merge(
                prem_df[["trade_date", "premium_deviation"]],
                price_df[["trade_date", "momentum_20d"]],
                on="trade_date",
            )

            corr = merged["premium_deviation"].corr(merged["momentum_20d"])
            correlations.append(corr)

    if correlations:
        avg_corr = np.mean(correlations)
        logger.info(f"\n折溢价因子 vs 20日动量因子:")
        logger.info(f"  平均相关性: {avg_corr:.4f}")
        logger.info(
            f"  解释: {'高度相关' if abs(avg_corr) > 0.7 else '中度相关' if abs(avg_corr) > 0.3 else '低相关'}"
        )


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("ETF折溢价率因子分层收益测试")
    logger.info("=" * 80)

    # 1. 加载数据
    merged_df = load_and_prepare_data(n_groups=3)

    # 2. 计算分层收益
    group_returns = calculate_group_returns(merged_df, n_groups=3)
    logger.info(f"\n计算了 {len(group_returns)} 天的分层收益")

    # 3. 分析结果
    analyzed_df = analyze_layered_returns(group_returns, n_groups=3)

    # 4. 因子相关性
    analyze_factor_correlation()

    # 5. 保存结果
    output_dir = Path("results/premium_factor_ic")
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzed_df.to_csv(output_dir / "layered_returns.csv", index=False)

    logger.info("\n" + "=" * 80)
    logger.info("分层收益测试完成！")
    logger.info(f"详细结果: {output_dir / 'layered_returns.csv'}")
    logger.info("=" * 80)
