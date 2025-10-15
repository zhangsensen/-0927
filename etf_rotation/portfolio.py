"""ETF组合构建：Top N等权"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioBuilder:
    """组合构建器：Top N等权+约束"""

    def __init__(self, n_holdings: int = 8, max_single: float = 0.20):
        """
        初始化组合构建器

        Args:
            n_holdings: 持仓数量，默认8只
            max_single: 单票最大权重，默认20%
        """
        self.n_holdings = n_holdings
        self.max_single = max_single

    def build(self, scored_etfs: pd.DataFrame) -> dict:
        """
        Top N等权，单票≤20%

        Args:
            scored_etfs: 评分后的ETF DataFrame（已排序）

        Returns:
            权重字典 {ETF代码: 权重}
        """
        if scored_etfs.empty:
            logger.warning("评分ETF为空，无法构建组合")
            return {}

        # 取Top N
        top_n = scored_etfs.head(self.n_holdings)

        if len(top_n) == 0:
            logger.warning("Top N为空")
            return {}

        # 等权
        weight = 1.0 / len(top_n)
        weight = min(weight, self.max_single)  # 单票上限

        weights = {code: weight for code in top_n.index}

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.info(f"组合构建完成：{len(weights)} 只ETF，单票权重 {weight:.2%}")
        return weights
