#!/usr/bin/env python3
"""ETF轮动系统自验脚本"""

import logging
from datetime import datetime

import pandas as pd
import yaml

from etf_rotation.portfolio import PortfolioBuilder
from etf_rotation.scorer import ETFScorer
from etf_rotation.universe_manager import ETFUniverseManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_panel_quality(panel_file: str):
    """验证面板质量"""
    logger.info("=" * 60)
    logger.info("1. 面板质量自验")
    logger.info("=" * 60)

    panel = pd.read_parquet(panel_file)
    logger.info(f"面板形状: {panel.shape}")

    # 检查3个交易日的横截面
    dates = panel.index.get_level_values(0).unique()
    sample_dates = [dates[len(dates) // 4], dates[len(dates) // 2], dates[-1]]

    for date in sample_dates:
        cross_section = panel.loc[date, :]
        non_null_ratio = (1 - cross_section.isnull().sum() / len(cross_section)).mean()

        logger.info(f"\n日期: {date}")
        logger.info(f"  横截面ETF数: {len(cross_section)}")
        logger.info(f"  非空比例: {non_null_ratio:.1%}")

        # 因子覆盖率
        for factor in ["Momentum252", "Momentum126", "Momentum63"]:
            if factor in cross_section.columns:
                coverage = 1 - cross_section[factor].isna().sum() / len(cross_section)
                logger.info(f"  {factor}覆盖率: {coverage:.1%}")

        # Top 5动量排序验证
        if "Momentum126" in cross_section.columns:
            top5 = cross_section.nlargest(5, "Momentum126")
            logger.info(f"  Top 5 ETF (按M126):")
            for i, (code, row) in enumerate(top5.iterrows(), 1):
                logger.info(
                    f"    {i}. {code}: M126={row.get('Momentum126', 0):.2%}, "
                    f"M63={row.get('Momentum63', 0):.2%}"
                )


def verify_universe_funnel(trade_date: str, panel_file: str):
    """验证宇宙筛选漏斗"""
    logger.info("=" * 60)
    logger.info("2. 宇宙筛选漏斗")
    logger.info("=" * 60)

    # 加载配置
    with open("etf_rotation/configs/scoring.yaml") as f:
        config = yaml.safe_load(f)

    # 1. 初始宇宙
    universe_mgr = ETFUniverseManager("etf_download_manager/config/etf_config.yaml")
    initial_universe = universe_mgr.get_monthly_universe(
        trade_date, min_amount_20d=config["liquidity"]["min_amount_20d"]
    )
    logger.info(f"初始候选: {len(initial_universe)} 只ETF")

    # 2. 加载面板
    panel = pd.read_parquet(panel_file)
    trade_date_dt = pd.to_datetime(trade_date, format="%Y%m%d")
    available_dates = pd.to_datetime(panel.index.get_level_values(0).unique())
    closest_date = available_dates[available_dates <= trade_date_dt].max()

    # 提取横截面（只取存在的ETF）
    cross_section = panel.loc[closest_date, :]
    available_etfs = [etf for etf in initial_universe if etf in cross_section.index]
    universe_panel = cross_section.loc[available_etfs, :]
    logger.info(f"面板中存在: {len(universe_panel)} 只ETF")

    # 3. 评分与过滤
    scorer = ETFScorer(weights=config["factor_weights"])
    scored = scorer.score(universe_panel)
    logger.info(f"经评分过滤: {len(scored)} 只ETF")

    # 4. 组合构建
    portfolio = PortfolioBuilder(
        n_holdings=config["portfolio"]["n_holdings"],
        max_single=config["portfolio"]["max_single"],
    ).build(scored)
    logger.info(f"最终持仓: {len(portfolio)} 只ETF")

    if len(portfolio) < 6:
        logger.warning(f"⚠️  最终持仓不足6只，可能需要放宽过滤条件")

    return portfolio


def verify_portfolio_constraints(portfolio: dict):
    """验证组合约束"""
    logger.info("=" * 60)
    logger.info("3. 组合约束验证")
    logger.info("=" * 60)

    if not portfolio:
        logger.warning("组合为空，跳过验证")
        return

    total_weight = sum(portfolio.values())
    max_weight = max(portfolio.values())

    logger.info(f"总权重: {total_weight:.4f} (应=1.0)")
    logger.info(f"最大单票权重: {max_weight:.2%} (应≤20%)")

    if abs(total_weight - 1.0) > 0.01:
        logger.error(f"❌ 权重归一化失败: {total_weight}")
    else:
        logger.info("✅ 权重归一化正确")

    if max_weight > 0.21:
        logger.error(f"❌ 单票权重超限: {max_weight:.2%}")
    else:
        logger.info("✅ 单票权重约束满足")


def main():
    # 验证面板
    panel_file = "factor_output/etf_rotation/panel_20240101_20241014.parquet"
    verify_panel_quality(panel_file)

    # 验证宇宙筛选（选2024年10月）
    portfolio = verify_universe_funnel("20241014", panel_file)

    # 验证组合约束
    verify_portfolio_constraints(portfolio)

    logger.info("=" * 60)
    logger.info("✅ 系统自验完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
