#!/usr/bin/env python3
"""
WFO回测运行器
负责在指定时间窗口上运行VBT回测
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WFOBacktestRunner:
    """WFO回测运行器 - 在指定时间窗口运行回测"""

    def __init__(self, vbt_results_path: str):
        """
        初始化回测运行器

        Args:
            vbt_results_path: VBT回测结果路径
        """
        self.vbt_results_path = Path(vbt_results_path)
        self.base_results = None
        self._load_base_results()

        logger.info(f"WFO回测运行器初始化完成")
        logger.info(f"基础结果路径: {self.vbt_results_path}")

    def _load_base_results(self):
        """加载VBT基础回测结果"""
        results_file = self.vbt_results_path / "results.csv"

        if not results_file.exists():
            raise FileNotFoundError(f"未找到results.csv: {results_file}")

        self.base_results = pd.read_csv(results_file)
        logger.info(f"成功加载 {len(self.base_results)} 个基础策略结果")

    def run_batch_backtest(
        self,
        strategies: List[Dict],
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict]:
        """
        在指定时间窗口批量回测策略

        Args:
            strategies: 策略列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果列表
        """
        logger.info(f"批量回测 {len(strategies)} 个策略")
        logger.info(f"时间窗口: {start_date.date()} ~ {end_date.date()}")

        results = []

        for strategy in strategies:
            # 提取策略参数
            weights = strategy.get("weights", {})
            top_n = strategy.get("top_n", 3)
            rebalance_freq = strategy.get("rebalance_freq", 13)

            # 运行单个策略回测
            result = self._run_single_strategy(
                weights, top_n, rebalance_freq, start_date, end_date
            )

            results.append(result)

        logger.info(
            f"批量回测完成，平均Sharpe: {np.mean([r['sharpe_ratio'] for r in results]):.3f}"
        )

        return results

    def _run_single_strategy(
        self,
        weights: Dict,
        top_n: int,
        rebalance_freq: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict:
        """
        运行单个策略回测

        注意：这里使用模拟方法。实际部署时应调用真实VBT引擎

        Args:
            weights: 因子权重
            top_n: Top-N参数
            rebalance_freq: 调仓频率
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            回测结果字典
        """
        # =====================================================================
        # 重要提示：这是简化的模拟实现
        #
        # 实际部署时应：
        # 1. 加载该时间窗口的ETF价格数据
        # 2. 计算该时间窗口的因子值
        # 3. 使用VectorBT运行真实回测
        # 4. 返回真实的净值曲线和指标
        #
        # 当前实现是为了快速验证WFO架构，使用统计模拟生成结果
        # =====================================================================

        # 设置随机种子（基于参数组合）
        seed = hash(
            str(weights) + str(top_n) + str(rebalance_freq) + str(start_date)
        ) % (2**32)
        np.random.seed(seed)

        # 基础性能估算
        base_sharpe = self._estimate_sharpe(weights, top_n, rebalance_freq)

        # 时间窗口长度影响
        window_days = (end_date - start_date).days
        window_factor = 1.0 if window_days >= 300 else 0.9  # 短窗口略降低

        # 添加时间窗口特定的随机性（模拟市场regime变化）
        time_noise = np.random.normal(0, 0.08)  # 更大的时间特定噪声

        # 最终Sharpe
        sharpe_ratio = max(0.1, base_sharpe * window_factor + time_noise)

        # 其他指标
        total_return = sharpe_ratio * 40 + np.random.normal(0, 10)  # 简化关系
        max_drawdown = -abs(np.random.normal(25, 8))
        volatility = abs(total_return / (sharpe_ratio + 0.01))

        return {
            "weights": weights,
            "top_n": top_n,
            "rebalance_freq": rebalance_freq,
            "sharpe_ratio": round(sharpe_ratio, 4),
            "total_return": round(total_return, 2),
            "max_drawdown": round(max_drawdown, 2),
            "volatility": round(volatility, 2),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "window_days": window_days,
        }

    def _estimate_sharpe(self, weights: Dict, top_n: int, rebalance_freq: int) -> float:
        """
        基于参数估算Sharpe比率

        这是简化的估算方法，实际应使用历史数据回测
        """
        # 基础Sharpe
        base = 0.45

        # Top-N影响
        if top_n == 1:
            base += 0.08
        elif top_n == 2:
            base += 0.05
        elif top_n <= 3:
            base += 0.02

        # 调仓频率影响
        if 10 <= rebalance_freq <= 20:
            base += 0.03

        # 权重集中度影响
        if weights:
            max_weight = max(weights.values())
            if max_weight > 0.4:
                base += 0.02

        # 因子数量影响
        effective_factors = sum(1 for w in weights.values() if w > 0.01)
        if 2 <= effective_factors <= 4:
            base += 0.01

        return base

    def prepare_strategies_from_results(self, top_k: int = 200) -> List[Dict]:
        """
        从VBT结果准备策略列表

        Args:
            top_k: 提取Top-K策略

        Returns:
            策略列表
        """
        if self.base_results is None:
            raise ValueError("未加载基础结果")

        # 按Sharpe排序
        sorted_results = self.base_results.sort_values("sharpe_ratio", ascending=False)
        top_results = sorted_results.head(top_k)

        strategies = []
        for _, row in top_results.iterrows():
            # 解析权重字符串
            import ast

            weights = ast.literal_eval(row["weights"])

            strategies.append(
                {
                    "weights": weights,
                    "top_n": int(row["top_n"]),
                    "rebalance_freq": int(row["rebalance_freq"]),
                }
            )

        logger.info(f"从基础结果提取了 {len(strategies)} 个策略")

        return strategies
