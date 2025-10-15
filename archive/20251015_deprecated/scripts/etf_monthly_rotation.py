#!/usr/bin/env python3
"""ETF月度轮动决策脚本"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from etf_rotation.portfolio import PortfolioBuilder
from etf_rotation.scorer import ETFScorer
from etf_rotation.universe_manager import ETFUniverseManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_scoring_config(config_path: str) -> dict:
    """加载评分配置"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ETF月度轮动决策")
    parser.add_argument(
        "--trade-date", required=True, help="交易日期(YYYYMMDD)", type=str
    )
    parser.add_argument(
        "--panel-file",
        required=True,
        help="因子面板文件路径",
    )
    parser.add_argument("--config-dir", default="etf_rotation/configs", help="配置目录")
    parser.add_argument("--output", default="rotation_output", help="输出目录")

    args = parser.parse_args()

    # 加载配置
    config_file = f"{args.config_dir}/scoring.yaml"
    scoring_config = load_scoring_config(config_file)
    logger.info(f"评分配置: {scoring_config}")

    # 检查是否使用扩展因子集
    use_extended = config_file.endswith("extended_scoring.yaml")
    if use_extended:
        logger.info("使用扩展因子集评分")
    else:
        logger.info("使用核心因子集评分")

    # 1. 月度宇宙锁定
    logger.info("=" * 60)
    logger.info("Step 1: 月度宇宙锁定")
    logger.info("=" * 60)
    universe_mgr = ETFUniverseManager("etf_download_manager/config/etf_config.yaml")
    monthly_universe = universe_mgr.get_monthly_universe(
        args.trade_date,
        min_amount_20d=scoring_config["liquidity"]["min_amount_20d"],
    )
    logger.info(f"候选ETF: {len(monthly_universe)} 只")

    # 2. 加载因子面板
    logger.info("=" * 60)
    logger.info("Step 2: 加载因子面板")
    logger.info("=" * 60)
    panel = pd.read_parquet(args.panel_file)
    logger.info(f"面板形状: {panel.shape}")

    # 提取指定日期的横截面
    trade_date_dt = pd.to_datetime(args.trade_date, format="%Y%m%d")
    if isinstance(panel.index, pd.MultiIndex):
        # MultiIndex: (datetime, symbol)
        available_dates = panel.index.get_level_values(0).unique()
        available_dates = pd.to_datetime(available_dates)

        # 重要：在trade_date当天，只能看到前一天的数据
        # 因子计算已经做了T+1处理，所以这里用trade_date - 1天
        signal_date = trade_date_dt - pd.Timedelta(days=1)
        closest_date = available_dates[available_dates <= signal_date].max()
        logger.info(f"信号日期: {signal_date} -> 使用面板日期: {closest_date}")

        # 提取横截面：指定日期+宇宙ETF
        universe_panel = panel.loc[closest_date, :].loc[monthly_universe, :]
    else:
        logger.error("面板索引格式不正确")
        return

    logger.info(f"宇宙面板形状: {universe_panel.shape}")

    # 3. 评分与筛选
    logger.info("=" * 60)
    logger.info("Step 3: 评分与筛选")
    logger.info("=" * 60)

    # 根据配置类型初始化评分器
    if use_extended:
        # 使用扩展评分配置
        scorer = ETFScorer(
            config_file=f"{args.config_dir}/extended_scoring.yaml",
            factor_selection=scoring_config.get("factor_selection", "equal_weight"),
            correlation_threshold=scoring_config.get("correlation_threshold", 0.9),
        )
    else:
        # 使用传统权重方式
        scorer = ETFScorer(weights=scoring_config["factor_weights"])

    scored_etfs = scorer.score(universe_panel)

    if scored_etfs.empty:
        logger.warning("评分后无ETF通过筛选")
        return

    logger.info(f"Top 10 ETF:")
    for i, (code, row) in enumerate(scored_etfs.head(10).iterrows(), 1):
        logger.info(
            f"  {i}. {code}: 评分={row['composite_score']:.3f}, "
            f"M252={row.get('Momentum252', 0):.2%}, "
            f"M126={row.get('Momentum126', 0):.2%}"
        )

    # 4. 组合构建
    logger.info("=" * 60)
    logger.info("Step 4: 组合构建")
    logger.info("=" * 60)
    portfolio = PortfolioBuilder(
        n_holdings=scoring_config["portfolio"]["n_holdings"],
        max_single=scoring_config["portfolio"]["max_single"],
    ).build(scored_etfs)

    logger.info(f"最终持仓: {len(portfolio)} 只ETF")
    for code, weight in portfolio.items():
        logger.info(f"  {code}: {weight:.2%}")

    # 5. 保存结果
    logger.info("=" * 60)
    logger.info("Step 5: 保存结果")
    logger.info("=" * 60)
    output_path = Path(args.output) / args.trade_date[:6]  # YYYYMM
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存权重
    weights_df = pd.DataFrame.from_dict(portfolio, orient="index", columns=["weight"])
    weights_file = output_path / f"weights_{args.trade_date}.csv"
    weights_df.to_csv(weights_file)
    logger.info(f"✅ 权重已保存: {weights_file}")

    # 保存评分明细
    scored_file = output_path / f"scored_{args.trade_date}.csv"
    scored_etfs.to_csv(scored_file)
    logger.info(f"✅ 评分明细已保存: {scored_file}")

    logger.info("=" * 60)
    logger.info("✅ ETF月度轮动决策完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
