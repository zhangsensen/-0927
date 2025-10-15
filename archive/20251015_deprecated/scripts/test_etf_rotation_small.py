#!/usr/bin/env python3
"""ETF轮动小样本测试"""

import logging

import pandas as pd

from etf_rotation.portfolio import PortfolioBuilder
from etf_rotation.scorer import ETFScorer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 加载面板
panel = pd.read_parquet("factor_output/etf_rotation/panel_20240101_20241014.parquet")
logger.info(f"面板形状: {panel.shape}")

# 提取最新日期横截面
latest_date = panel.index.get_level_values(0).max()
logger.info(f"最新日期: {latest_date}")

cross_section = panel.loc[latest_date, :]
logger.info(f"横截面形状: {cross_section.shape}")
logger.info(f"ETF列表: {list(cross_section.index)}")

# 评分
weights = {"Momentum126": 0.5, "Momentum63": 0.3, "VOLATILITY_120D": -0.2}
scorer = ETFScorer(weights=weights)
scored = scorer.score(cross_section)

logger.info(f"\n评分结果（Top 10）:")
for i, (code, row) in enumerate(scored.head(10).iterrows(), 1):
    logger.info(
        f"{i}. {code}: 评分={row['composite_score']:.3f}, "
        f"M126={row.get('Momentum126', 0):.2%}, "
        f"M63={row.get('Momentum63', 0):.2%}"
    )

# 组合构建
portfolio = PortfolioBuilder(n_holdings=5, max_single=0.25).build(scored)

logger.info(f"\n最终持仓（Top 5）:")
for code, weight in portfolio.items():
    logger.info(f"  {code}: {weight:.2%}")

logger.info("\n✅ 测试完成")
