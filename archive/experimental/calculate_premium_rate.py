#!/usr/bin/env python3
"""
批量获取ETF的IOPV（基金份额参考净值）历史数据

IOPV是计算折溢价率的关键：
premium_rate = (close_price - iopv) / iopv
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from etf_data.crawlers import EastmoneyETFCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def fetch_historical_iopv(etf_codes: List[str], output_dir: str = "raw/ETF/iopv"):
    """
    获取多只ETF的历史IOPV数据

    注意：东财实时行情API只能获取当前数据，历史IOPV需要其他数据源
    这里提供两种方案：
    1. 从已有净值数据估算（如果有NAV）
    2. 接入Wind/Tushare获取历史IOPV
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # TODO: 实现历史IOPV获取逻辑
    # 方案A：用已有净值数据（如果有）
    # 方案B：用Tushare的fund_daily接口（可能有iopv字段）
    # 方案C：从东财ETF详情页解析历史净值

    logger.info(f"需要获取 {len(etf_codes)} 只ETF的IOPV数据")
    logger.info("当前可用方案：从东财详情页获取净值历史作为IOPV代理")

    return output_path


def calculate_premium_rate(etf_code: str, market: str = "SH"):
    """
    计算ETF的折溢价率

    需要：交易价格（已有） + IOPV/NAV（需获取）

    premium_rate = (close_price - iopv) / iopv

    正值 = 溢价（交易价格高于净值，看多情绪）
    负值 = 折价（交易价格低于净值，看空情绪）
    """
    from etf_data.crawlers import EastmoneyDetailCrawler

    crawler = EastmoneyDetailCrawler()

    # 获取净值历史
    nav_df = crawler.get_networth_history(etf_code)

    # 读取交易价格
    price_file = Path(f"raw/ETF/daily/{etf_code}.{market}_daily_*.parquet")
    import glob

    files = glob.glob(str(price_file))

    if not files or nav_df.empty:
        logger.error(f"数据缺失: {etf_code}")
        return None

    price_df = pd.read_parquet(files[0])

    # 合并计算折溢价
    merged = pd.merge(
        price_df[["trade_date", "adj_close"]],
        nav_df[["trade_date", "nav"]],
        on="trade_date",
        how="inner",
    )

    merged["premium_rate"] = (merged["adj_close"] - merged["nav"]) / merged["nav"] * 100

    logger.info(f"{etf_code}: 计算了 {len(merged)} 天的折溢价率")
    logger.info(f"  平均折溢价: {merged['premium_rate'].mean():.3f}%")
    logger.info(f"  最大溢价: {merged['premium_rate'].max():.3f}%")
    logger.info(f"  最大折价: {merged['premium_rate'].min():.3f}%")

    return merged


if __name__ == "__main__":
    # 测试计算510300的折溢价率
    print("=" * 60)
    print("计算ETF折溢价率示例")
    print("=" * 60)

    result = calculate_premium_rate("510300", "SH")
    if result is not None:
        print("\n最近10天的折溢价率:")
        print(result[["trade_date", "adj_close", "nav", "premium_rate"]].tail(10))
