#!/usr/bin/env python3
"""
WFO优化器
在IS期优化策略参数
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from optimization.strategy_optimizer import OptimizationConfig, StrategyOptimizer
except ImportError:
    # 如果导入失败，使用简化版
    from dataclasses import dataclass

    @dataclass
    class OptimizationConfig:
        sharpe_weight: float = 0.6
        return_weight: float = 0.3
        drawdown_weight: float = 0.1

    class StrategyOptimizer:
        def __init__(self, config=None):
            self.config = config or OptimizationConfig()


logger = logging.getLogger(__name__)


class WFOOptimizer:
    """WFO优化器 - 在IS期优化参数"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化WFO优化器

        Args:
            config: 优化配置
        """
        self.base_optimizer = StrategyOptimizer(config)
        self.config = config or OptimizationConfig()

        logger.info("WFO优化器初始化完成")

    def optimize_for_period(
        self,
        candidate_strategies: List[Dict],
        period_start: str,
        period_end: str,
    ) -> List[Dict]:
        """
        针对特定时期优化策略

        Args:
            candidate_strategies: 候选策略列表
            period_start: 时期开始日期
            period_end: 时期结束日期

        Returns:
            优化后的策略列表
        """
        logger.info(f"优化时期: {period_start} ~ {period_end}")
        logger.info(f"候选策略数: {len(candidate_strategies)}")

        # 评估所有候选策略
        evaluated = []
        for strategy in candidate_strategies:
            score = self._evaluate_strategy(strategy)
            evaluated.append(
                {
                    **strategy,
                    "score": score,
                }
            )

        # 按评分排序
        evaluated.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"优化完成，最高得分: {evaluated[0]['score']:.4f}")

        return evaluated

    def _evaluate_strategy(self, strategy: Dict) -> float:
        """
        评估策略得分

        Args:
            strategy: 策略字典

        Returns:
            综合得分
        """
        # 提取指标
        sharpe = strategy.get("sharpe_ratio", 0)
        ret = strategy.get("total_return", 0)
        dd = strategy.get("max_drawdown", 0)

        # 综合评分
        score = (
            self.config.sharpe_weight * sharpe
            + self.config.return_weight * (ret / 100)
            + self.config.drawdown_weight * (dd / -100)
        )

        return score
