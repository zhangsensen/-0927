#!/usr/bin/env python3
"""ETF因子面板日频生产脚本"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from factor_system.factor_engine import api

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_etf_list(etf_list_file: str) -> list[str]:
    """加载ETF列表"""
    with open(etf_list_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="生产ETF因子面板")
    parser.add_argument(
        "--start-date", required=True, help="开始日期(YYYYMMDD)", type=str
    )
    parser.add_argument(
        "--end-date", required=True, help="结束日期(YYYYMMDD)", type=str
    )
    parser.add_argument(
        "--etf-list",
        default="etf_rotation/configs/etf_universe.txt",
        help="ETF列表文件",
    )
    parser.add_argument(
        "--output", default="factor_output/etf_rotation", help="输出目录"
    )
    parser.add_argument(
        "--factor-set",
        choices=["core", "extended"],
        default="core",
        help="因子集选择: core(6个核心因子) | extended(64个扩展因子)",
    )
    parser.add_argument(
        "--factor-set-file",
        help="自定义因子集配置文件路径",
    )

    args = parser.parse_args()

    # 加载ETF列表
    logger.info(f"加载ETF列表: {args.etf_list}")
    etf_list = load_etf_list(args.etf_list)
    logger.info(f"ETF总数: {len(etf_list)}")

    # 转换日期
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")

    # 确定因子集
    if args.factor_set_file:
        factor_set_config = args.factor_set_file
        logger.info(f"使用自定义因子集: {factor_set_config}")
    elif args.factor_set == "extended":
        factor_set_config = (
            "factor_system/factor_engine/factors/factor_sets/etf_price_extended.yaml"
        )
        logger.info("使用扩展因子集")
    else:
        # 默认核心因子集
        factor_ids = [
            "Momentum252",
            "Momentum126",
            "Momentum63",
            "VOLATILITY_120D",
            "MOM_ACCEL",
            "DRAWDOWN_63D",
            "ATR14",
            "TA_ADX_14",
        ]
        logger.info(f"使用核心因子集: {len(factor_ids)} 个因子")

    # 计算因子
    logger.info(f"开始计算因子: {start_date} - {end_date}")

    if args.factor_set != "core" and not args.factor_set_file:
        # 加载因子集配置
        with open(factor_set_config) as f:
            factor_set = yaml.safe_load(f)
        factor_ids = factor_set.get("factors", [])
        logger.info(f"从配置加载因子集: {len(factor_ids)} 个因子")

    try:
        factors = api.calculate_factors(
            factor_ids=factor_ids,
            symbols=etf_list,
            timeframe="daily",
            start_date=start_date,
            end_date=end_date,
        )

        if factors.empty:
            logger.error("因子计算结果为空")
            return

        logger.info(f"因子计算完成: {factors.shape}")

        # 保存面板
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"panel_{args.start_date}_{args.end_date}.parquet"
        factors.to_parquet(output_file)

        logger.info(f"✅ 因子面板已保存: {output_file}")

        # 统计信息
        logger.info(f"面板形状: {factors.shape}")
        logger.info(f"ETF数量: {factors.index.get_level_values(0).nunique()}")
        logger.info(
            f"日期范围: {factors.index.get_level_values(1).min()} - {factors.index.get_level_values(1).max()}"
        )
        logger.info(f"因子列: {list(factors.columns)}")

        # 覆盖率统计
        logger.info("\n因子覆盖率:")
        for col in factors.columns:
            coverage = (1 - factors[col].isna().sum() / len(factors)) * 100
            logger.info(f"  {col}: {coverage:.1f}%")

    except Exception as e:
        logger.error(f"因子计算失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
