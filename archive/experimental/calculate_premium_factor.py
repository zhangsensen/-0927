#!/usr/bin/env python3
"""
折溢价率因子计算脚本 - 使用现有数据（带局限性说明）

⚠️  重要说明：
当前价格数据只有adj_close（后复权价），没有原始close或复权因子。
这会导致长期历史计算的折溢价率有系统性偏差。

解决方案：
1. 短期：使用近期数据（如最近1年），偏差较小
2. 中期：重新下载含原始价格的数据
3. 长期：建立标准化数据pipeline

计算方式：
premium_rate = (adj_close - nav) / nav * 100

注意：adj_close包含历史分红复权，nav是当日实际值，两者口径不完全一致。
"""

import sys

sys.path.insert(0, "src")

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from etf_data.crawlers import EastmoneyDetailCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_premium_rate_approximate(
    etf_code: str, market: str = "SH", use_recent_days: int | None = None
) -> pd.DataFrame:
    """
    计算ETF折溢价率（近似值，使用adj_close）

    Args:
        etf_code: ETF代码
        market: 市场
        use_recent_days: 只使用最近N天数据（减少复权累积误差）

    Returns:
        DataFrame包含日期和折溢价率
    """
    logger.info(f"计算 {etf_code} 的折溢价率...")

    # 获取净值
    crawler = EastmoneyDetailCrawler()
    nav_df = crawler.get_networth_history(etf_code)

    if nav_df.empty:
        logger.error(f"{etf_code}: 无法获取净值数据")
        return pd.DataFrame()

    nav_df["trade_date"] = pd.to_datetime(nav_df["trade_date"])

    # 读取价格数据
    import glob

    files = glob.glob(f"raw/ETF/daily/{etf_code}.{market}_*.parquet")

    if not files:
        logger.error(f"{etf_code}: 找不到价格数据文件")
        return pd.DataFrame()

    price_df = pd.read_parquet(files[0])
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"], format="%Y%m%d")

    # 合并数据
    merged = pd.merge(
        price_df[["trade_date", "adj_close"]],
        nav_df[["trade_date", "nav"]],
        on="trade_date",
        how="inner",
    )

    if merged.empty:
        logger.error(f"{etf_code}: 日期对齐失败")
        return pd.DataFrame()

    # 计算折溢价率
    merged["premium_rate"] = (merged["adj_close"] - merged["nav"]) / merged["nav"] * 100

    # 只使用近期数据（减少复权误差）
    if use_recent_days and len(merged) > use_recent_days:
        merged = merged.tail(use_recent_days).reset_index(drop=True)
        logger.info(f"{etf_code}: 使用最近 {use_recent_days} 天数据")

    # 统计
    logger.info(f"{etf_code}: 计算了 {len(merged)} 天的折溢价率")
    logger.info(f"  平均: {merged['premium_rate'].mean():.4f}%")
    logger.info(f"  标准差: {merged['premium_rate'].std():.4f}%")
    logger.info(
        f"  范围: [{merged['premium_rate'].min():.4f}%, {merged['premium_rate'].max():.4f}%]"
    )

    return merged[["trade_date", "premium_rate"]].copy()


def batch_calculate_premium_rates(
    etf_list_file: str | None = None, output_dir: str = "raw/ETF/factors"
):
    """
    批量计算多只ETF的折溢价率

    Args:
        etf_list_file: ETF列表文件，为None时自动发现
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有ETF代码
    if etf_list_file is None:
        import glob

        files = glob.glob("raw/ETF/daily/*.parquet")
        etf_codes = []
        for f in files:
            # 从文件名提取代码 (e.g., "510300.SH" -> "510300")
            code_with_market = Path(f).stem.split("_")[0]
            code = code_with_market.split(".")[0]  # 去掉.SH/.SZ后缀
            etf_codes.append(code)
        etf_codes = list(set(etf_codes))  # 去重
    else:
        etf_codes = pd.read_csv(etf_list_file)["code"].tolist()

    logger.info(f"批量计算 {len(etf_codes)} 只ETF的折溢价率...")

    results = {}
    failed = []

    for i, code in enumerate(etf_codes, 1):
        logger.info(f"\n[{i}/{len(etf_codes)}] 处理 {code}...")

        try:
            # 判断市场
            market = "SH" if code.startswith("5") else "SZ"

            df = calculate_premium_rate_approximate(code, market, use_recent_days=None)

            if not df.empty:
                # 保存
                output_file = output_path / f"premium_rate_{code}.parquet"
                df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")
                df.to_parquet(output_file, index=False)
                logger.info(f"  ✓ 已保存: {output_file}")
                results[code] = len(df)
            else:
                failed.append(code)

        except Exception as e:
            logger.error(f"  ✗ 失败: {e}")
            failed.append(code)

    # 汇总报告
    logger.info("\n" + "=" * 60)
    logger.info("批量计算完成")
    logger.info(f"成功: {len(results)}/{len(etf_codes)}")
    logger.info(f"失败: {len(failed)}")
    if failed:
        logger.info(f"失败列表: {failed}")
    logger.info("=" * 60)

    return results, failed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # 批量计算模式
        print("\n" + "=" * 60)
        print("开始批量计算所有ETF折溢价率")
        print("=" * 60)
        batch_calculate_premium_rates()
    else:
        # 测试单只
        print("=" * 60)
        print("测试计算510300折溢价率")
        print("=" * 60)

        result = calculate_premium_rate_approximate("510300", "SH", use_recent_days=252)

        if not result.empty:
            print("\n最近10天折溢价率:")
            display_df = result.tail(10).copy()
            display_df["trade_date"] = pd.to_datetime(
                display_df["trade_date"]
            ).dt.strftime("%Y-%m-%d")
            print(display_df.to_string(index=False))
