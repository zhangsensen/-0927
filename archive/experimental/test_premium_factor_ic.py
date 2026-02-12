#!/usr/bin/env python3
"""
ETF折溢价率因子IC测试

测试4个因子变体 × 3个持有期 = 12组IC结果
使用Rank IC（Spearman相关），天然对异常值稳健

因子变体：
1. premium_raw - 原始折溢价率
2. premium_ma5 - 5日平滑
3. premium_deviation - 20日均值偏离（预期最强）
4. premium_delta_5d - 5日变化量

持有期：1日、5日、20日
"""

import sys

sys.path.insert(0, "src")

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# 异常ETF列表（折溢价率绝对值过大）
ANOMALY_ETFS = ["512800"]  # 银行ETF，折价-20%异常


def load_premium_data(factors_dir: str = "raw/ETF/factors") -> pd.DataFrame:
    """
    加载所有ETF的折溢价率数据

    Returns:
        DataFrame: columns=['trade_date', 'code', 'premium_rate', ...]
    """
    factors_path = Path(factors_dir)
    files = list(factors_path.glob("premium_rate_*.parquet"))

    all_data = []
    for f in files:
        code = f.stem.replace("premium_rate_", "")
        df = pd.read_parquet(f)
        df["code"] = code
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"加载了 {len(files)} 只ETF的折溢价率数据，共 {len(combined)} 条记录")
    return combined


def calculate_factor_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算折溢价率的4个因子变体

    Args:
        df: 单只ETF的折溢价率数据，需按日期排序

    Returns:
        添加了因子变体的DataFrame
    """
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 1. 原始折溢价率
    df["premium_raw"] = df["premium_rate"]

    # 2. 5日平滑
    df["premium_ma5"] = df["premium_rate"].rolling(window=5, min_periods=3).mean()

    # 3. 20日均值偏离（当前值 - 20日均值）
    df["premium_ma20"] = df["premium_rate"].rolling(window=20, min_periods=10).mean()
    df["premium_deviation"] = df["premium_rate"] - df["premium_ma20"]

    # 4. 5日变化量
    df["premium_delta_5d"] = df["premium_rate"].diff(5)

    return df


def load_price_data(
    etf_codes: List[str], price_dir: str = "raw/ETF/daily"
) -> pd.DataFrame:
    """
    加载ETF价格数据并计算未来收益

    Returns:
        DataFrame: columns=['trade_date', 'code', 'close', 'future_return_1d', ...]
    """
    price_path = Path(price_dir)
    all_prices = []

    import glob

    for code in etf_codes:
        # 找到对应的价格文件
        pattern = f"{code}.*_daily_*.parquet"
        files = glob.glob(str(price_path / pattern))

        if not files:
            logger.warning(f"{code}: 未找到价格数据")
            continue

        df = pd.read_parquet(files[0])
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df["code"] = code
        df = df.sort_values("trade_date")

        # 使用复权收盘价计算收益
        if "adj_close" in df.columns:
            close_col = "adj_close"
        elif "close" in df.columns:
            close_col = "close"
        else:
            logger.warning(f"{code}: 无收盘价数据")
            continue

        # 计算未来收益（1日、5日、20日）
        df["future_return_1d"] = df[close_col].shift(-1) / df[close_col] - 1
        df["future_return_5d"] = df[close_col].shift(-5) / df[close_col] - 1
        df["future_return_20d"] = df[close_col].shift(-20) / df[close_col] - 1

        all_prices.append(
            df[
                [
                    "trade_date",
                    "code",
                    close_col,
                    "future_return_1d",
                    "future_return_5d",
                    "future_return_20d",
                ]
            ]
        )

    combined = pd.concat(all_prices, ignore_index=True)
    logger.info(f"加载了 {len(etf_codes)} 只ETF的价格数据")
    return combined


def calculate_ic(
    factor_df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    min_observations: int = 10,
) -> Tuple[float, float, int]:
    """
    计算单期Rank IC（Spearman相关系数）

    Returns:
        (IC值, p值, 有效观测数)
    """
    # 剔除缺失值
    valid = factor_df[[factor_col, return_col]].dropna()

    if len(valid) < min_observations:
        return np.nan, np.nan, 0

    # 计算Rank IC
    ic, pvalue = spearmanr(valid[factor_col], valid[return_col])

    return ic, pvalue, len(valid)


def calculate_ic_series(
    merged_df: pd.DataFrame, factor_col: str, return_col: str
) -> pd.DataFrame:
    """
    计算IC时间序列（每日截面IC）

    Returns:
        DataFrame: columns=['trade_date', 'ic', 'pvalue', 'n_obs']
    """
    ic_results = []

    for date, group in merged_df.groupby("trade_date"):
        ic, pvalue, n_obs = calculate_ic(group, factor_col, return_col)

        if not np.isnan(ic):
            ic_results.append(
                {"trade_date": date, "ic": ic, "pvalue": pvalue, "n_obs": n_obs}
            )

    return pd.DataFrame(ic_results)


def analyze_ic(ic_df: pd.DataFrame, factor_name: str, return_name: str) -> Dict:
    """
    分析IC序列的统计特征

    Returns:
        包含IC均值、标准差、IR、胜率等信息的字典
    """
    if ic_df.empty:
        return {
            "factor": factor_name,
            "return_period": return_name,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "win_rate": np.nan,
            "n_days": 0,
        }

    ic_mean = ic_df["ic"].mean()
    ic_std = ic_df["ic"].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    win_rate = (ic_df["ic"] > 0).mean()

    return {
        "factor": factor_name,
        "return_period": return_name,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "win_rate": win_rate,
        "n_days": len(ic_df),
    }


def run_ic_analysis(exclude_anomalies: bool = False) -> pd.DataFrame:
    """
    运行完整的IC分析

    Args:
        exclude_anomalies: 是否剔除异常ETF

    Returns:
        包含所有因子-收益组合的IC统计结果
    """
    logger.info("=" * 80)
    logger.info(f"开始IC分析 (剔除异常: {exclude_anomalies})")
    logger.info("=" * 80)

    # 1. 加载折溢价率数据
    premium_df = load_premium_data()

    # 2. 计算因子变体
    logger.info("\n计算因子变体...")
    factor_dfs = []
    for code, group in premium_df.groupby("code"):
        if exclude_anomalies and code in ANOMALY_ETFS:
            continue
        factor_df = calculate_factor_variants(group)
        factor_dfs.append(factor_df)

    premium_with_factors = pd.concat(factor_dfs, ignore_index=True)
    logger.info(f"计算完成，共 {premium_with_factors['code'].nunique()} 只ETF")

    # 3. 加载价格数据
    etf_codes = premium_with_factors["code"].unique().tolist()
    price_df = load_price_data(etf_codes)

    # 4. 合并数据
    logger.info("\n合并因子和收益数据...")
    merged = pd.merge(
        premium_with_factors, price_df, on=["trade_date", "code"], how="inner"
    )
    logger.info(f"合并后数据: {len(merged)} 条记录")

    # 5. 计算IC（4个因子 × 3个持有期）
    factor_cols = [
        "premium_raw",
        "premium_ma5",
        "premium_deviation",
        "premium_delta_5d",
    ]
    return_cols = ["future_return_1d", "future_return_5d", "future_return_20d"]

    results = []

    logger.info("\n计算IC时间序列...")
    for factor_col in factor_cols:
        for return_col in return_cols:
            logger.info(f"  {factor_col} vs {return_col}")

            # 计算每日IC
            ic_series = calculate_ic_series(merged, factor_col, return_col)

            # 统计IC特征
            result = analyze_ic(ic_series, factor_col, return_col)
            results.append(result)

            logger.info(
                f"    IC均值: {result['ic_mean']:.4f}, "
                f"IR: {result['ic_ir']:.4f}, "
                f"胜率: {result['win_rate']:.2%}"
            )

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # 运行两组测试：包含异常值 和 剔除异常值
    results_with_anomaly = run_ic_analysis(exclude_anomalies=False)
    results_without_anomaly = run_ic_analysis(exclude_anomalies=True)

    # 保存结果
    output_dir = Path("results/premium_factor_ic")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_with_anomaly.to_csv(output_dir / "ic_results_with_anomaly.csv", index=False)
    results_without_anomaly.to_csv(
        output_dir / "ic_results_without_anomaly.csv", index=False
    )

    # 输出汇总
    logger.info("\n" + "=" * 80)
    logger.info("IC分析完成！")
    logger.info("=" * 80)
    logger.info(f"\n包含异常值的测试结果:")
    logger.info(f"  保存至: {output_dir / 'ic_results_with_anomaly.csv'}")
    logger.info(f"\n剔除异常值后的测试结果:")
    logger.info(f"  保存至: {output_dir / 'ic_results_without_anomaly.csv'}")

    # 找出最佳因子-收益组合
    best = results_with_anomaly.loc[results_with_anomaly["ic_ir"].idxmax()]
    logger.info(f"\n最佳因子组合 (按IC_IR):")
    logger.info(f"  因子: {best['factor']}")
    logger.info(f"  持有期: {best['return_period']}")
    logger.info(f"  IC均值: {best['ic_mean']:.4f}")
    logger.info(f"  IC_IR: {best['ic_ir']:.4f}")
    logger.info(f"  胜率: {best['win_rate']:.2%}")
