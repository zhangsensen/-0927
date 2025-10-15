#!/usr/bin/env python3
"""测试扩展因子评分系统"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from etf_rotation.scorer import ETFScorer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_extended_scoring():
    """测试扩展因子评分系统"""
    logger.info("=" * 60)
    logger.info("测试扩展因子评分系统")
    logger.info("=" * 60)

    # 加载测试数据
    panel_file = "factor_output/etf_rotation/panel_20240101_20251014.parquet"
    if not Path(panel_file).exists():
        logger.error(f"面板文件不存在: {panel_file}")
        return

    panel = pd.read_parquet(panel_file)
    logger.info(f"加载面板: {panel.shape}")

    # 取一个横截面进行测试
    sample_date = panel.index.get_level_values(0).unique()[-10]  # 取倒数第10个日期
    sample_panel = panel.loc[sample_date, :].dropna(how="all")

    if sample_panel.empty:
        logger.error("没有找到可用的横截面数据")
        return

    logger.info(f"测试横截面: {sample_date}, 形状: {sample_panel.shape}")
    logger.info(f"可用因子: {list(sample_panel.columns)[:10]}...")

    # 测试扩展评分器
    logger.info("\n测试扩展评分器...")
    extended_scorer = ETFScorer(
        config_file="etf_rotation/configs/extended_scoring.yaml",
        factor_selection="equal_weight",
    )

    # 评分
    scored = extended_scorer.score(sample_panel)

    logger.info(f"扩展评分结果: {len(scored)} 只ETF")
    if not scored.empty:
        logger.info("Top 10 ETF:")
        for i, (code, row) in enumerate(scored.head(10).iterrows(), 1):
            logger.info(f"  {i:2d}. {code}: 综合评分={row['composite_score']:.3f}")

    # 对比核心评分器
    logger.info("\n对比核心评分器...")
    core_weights = {
        "Momentum20": 0.3,
        "Momentum15": 0.2,
        "Momentum10": 0.2,
        "ATR14": 0.15,
        "MACD": 0.15,
    }
    core_scorer = ETFScorer(weights=core_weights)
    core_scored = core_scorer.score(sample_panel)

    logger.info(f"核心评分结果: {len(core_scored)} 只ETF")
    if not core_scored.empty:
        logger.info("Top 10 ETF:")
        for i, (code, row) in enumerate(core_scored.head(10).iterrows(), 1):
            logger.info(f"  {i:2d}. {code}: 综合评分={row['composite_score']:.3f}")

    # 比较结果
    logger.info("\n评分结果对比:")
    logger.info(f"  扩展因子数: {len(extended_scorer.weights)} -> {len(scored)} 只ETF")
    logger.info(f"  核心因子数: {len(core_scorer.weights)} -> {len(core_scored)} 只ETF")

    # 保存测试结果
    output_dir = Path("rotation_output/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_results = pd.DataFrame(
        {
            "extended_score": (
                scored["composite_score"] if not scored.empty else pd.Series()
            ),
            "core_score": (
                core_scored["composite_score"] if not core_scored.empty else pd.Series()
            ),
        }
    )

    test_results_file = output_dir / "scoring_comparison.parquet"
    test_results.to_parquet(test_results_file)
    logger.info(f"✅ 测试结果已保存: {test_results_file}")

    return scored, core_scored


if __name__ == "__main__":
    test_extended_scoring()
