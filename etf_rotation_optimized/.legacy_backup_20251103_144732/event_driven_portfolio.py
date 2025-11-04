#!/usr/bin/env python3
"""
事件驱动持仓构建器
基于信号质量评估，动态调整持仓，支持最小持有期和换手预算约束
"""

import logging
from typing import Dict, Optional

import numpy as np

from .signal_quality_evaluator import SignalQualityEvaluator

logger = logging.getLogger(__name__)


class EventDrivenPortfolioConstructor:
    """
    事件驱动持仓构建器

    核心逻辑：
    1. 每日评估信号质量
    2. 根据买卖信号调整持仓
    3. 应用执行约束（最小持有期、换手预算）
    4. 生成等权重矩阵

    状态追踪：
    - holdings: 当前持仓（布尔矩阵）
    - holding_days: 持有天数（整数矩阵）
    """

    def __init__(
        self,
        top_n: int = 6,
        min_holding_days: int = 3,
        max_daily_turnover: float = 0.3,
        fill_policy: str = "cash",
        evaluator: Optional[SignalQualityEvaluator] = None,
    ):
        """
        参数:
            top_n: 目标持仓数
            min_holding_days: 最小持有期（天）
            max_daily_turnover: 每日最大换手比例
            fill_policy: 缺额填充策略（"cash"=留现金, "fill"=用次强填满）
            evaluator: 信号质量评估器
        """
        self.top_n = top_n
        self.min_holding_days = min_holding_days
        self.max_daily_turnover = max_daily_turnover
        self.fill_policy = fill_policy
        self.evaluator = evaluator or SignalQualityEvaluator()

        logger.info(
            f"初始化EventDrivenPortfolioConstructor: "
            f"Top-N={top_n}, "
            f"最小持有期={min_holding_days}天, "
            f"换手预算={max_daily_turnover:.1%}, "
            f"填充策略={fill_policy}"
        )

    def construct(
        self,
        scores: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        构建事件驱动持仓

        参数:
            scores: (T, N) 信号分数矩阵

        返回:
            字典包含:
                - weights: (T, N) 权重矩阵（等权）
                - holdings: (T, N) 持仓矩阵（布尔）
                - trades: (T, N) 交易矩阵（1=买入, -1=卖出, 0=不动）
                - turnover: (T,) 每日换手率
        """
        T, N = scores.shape

        logger.info(f"开始构建事件驱动持仓: T={T}, N={N}")

        # 1. 生成信号
        signals = self.evaluator.generate_signals(scores, self.top_n)
        buy_signal = signals["buy"]
        sell_signal = signals["sell"]
        ranks = signals["ranks"]
        strength = signals["strength"]
        stability = signals["stability"]

        # 暖启动：在稳定性窗口内放宽稳定性要求，避免长时间空仓
        warmup_days = getattr(self.evaluator, "stability_window", 5)
        buy_mask = buy_signal.copy()
        if warmup_days and warmup_days > 1:
            early = slice(0, warmup_days - 1)
            in_topn_early = ranks[early] <= self.top_n
            buy_mask[early] = strength[early] & in_topn_early

        # 2. 初始化状态
        holdings = np.zeros((T, N), dtype=bool)
        holding_days = np.zeros((T, N), dtype=int)
        trades = np.zeros((T, N), dtype=int)
        turnover = np.zeros(T)

        # 3. 逐日构建持仓（状态机）
        for t in range(T):
            if t == 0:
                # 第一天：买入Top-N强信号
                holdings[t] = buy_signal[t]
                holding_days[t] = np.where(holdings[t], 1, 0)
                trades[t] = np.where(holdings[t], 1, 0)
                # 第一天换手率=买入数量/目标持仓数（可能>1）
                turnover[t] = min(1.0, holdings[t].sum() / max(self.top_n, 1))
            else:
                # 继承前一天持仓
                holdings[t] = holdings[t - 1].copy()
                holding_days[t] = holding_days[t - 1].copy()

                # 更新持有天数
                holding_days[t, holdings[t]] += 1

                # 卖出决策（满足最小持有期）
                can_sell = holding_days[t] >= self.min_holding_days
                # 卖出规则收敛：跌出缓冲 或 (不再强且不在Top-N)
                out_of_buffer = ranks[t] > (self.top_n + self.evaluator.rank_buffer)
                not_strong_not_top = (~strength[t]) & (ranks[t] > self.top_n)
                refined_sell = out_of_buffer | not_strong_not_top
                should_sell = refined_sell & holdings[t] & can_sell

                # 执行卖出
                if should_sell.any():
                    holdings[t, should_sell] = False
                    holding_days[t, should_sell] = 0
                    trades[t, should_sell] = -1

                # 买入决策
                current_count = holdings[t].sum()
                available_slots = self.top_n - current_count

                if available_slots > 0:
                    # 候选：买入信号（经暖启动放宽）且未持有
                    can_buy = buy_mask[t] & ~holdings[t]

                    if can_buy.any():
                        # 按分数排序，买入最强的
                        buy_scores = np.where(can_buy, scores[t], -np.inf)
                        buy_indices = np.argsort(buy_scores)[::-1]

                        # 应用换手预算（向上取整，至少1），避免只买入1只导致暴露长期不足
                        from math import ceil

                        budget_cap = max(1, ceil(self.top_n * self.max_daily_turnover))
                        max_buy_count = min(available_slots, budget_cap)

                        # 执行买入
                        buy_indices = buy_indices[:max_buy_count]
                        holdings[t, buy_indices] = True
                        holding_days[t, buy_indices] = 1
                        trades[t, buy_indices] = 1

                # 如仍未满仓且策略允许填充，用满足信号质量过滤（强 + 稳定）的次强标的填满以维持目标暴露
                if self.fill_policy == "fill":
                    current_count = holdings[t].sum()
                    gap = self.top_n - current_count
                    if gap > 0:
                        # 候选：未持有 & 强 & 稳定
                        fill_candidates = (~holdings[t]) & strength[t] & stability[t]
                        # 如无候选，跳过填充（维持现金）
                        if np.any(fill_candidates):
                            fill_scores = np.where(fill_candidates, scores[t], -np.inf)
                            # 选择分数最高的 gap 只
                            ranked_idx = np.argsort(fill_scores)[::-1]
                            ranked_idx = ranked_idx[
                                np.isfinite(fill_scores[ranked_idx])
                            ]
                            fill_indices = ranked_idx[:gap]
                            if fill_indices.size > 0:
                                holdings[t, fill_indices] = True
                                # 记录为买入交易，计入换手
                                trades[t, fill_indices] = 1
                                holding_days[t, fill_indices] = np.where(
                                    holding_days[t, fill_indices] > 0,
                                    holding_days[t, fill_indices],
                                    1,
                                )

                # 计算换手率（变化仓位/目标仓位）
                # 注意：这里换手率可能>max_daily_turnover，因为卖出不受预算限制
                # 只有买入受预算限制
                changed = (trades[t] != 0).sum()
                turnover[t] = changed / max(self.top_n, 1)

        # 4. 生成权重（等权）
        weights = holdings.astype(float)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        # 5. 统计
        avg_holdings = holdings.sum(axis=1).mean()
        avg_turnover = turnover.mean()
        total_trades = (trades != 0).sum()

        logger.info(
            f"持仓构建完成: "
            f"平均持仓={avg_holdings:.1f}/{self.top_n}, "
            f"平均换手={avg_turnover:.2%}, "
            f"总交易次数={total_trades}"
        )

        return {
            "weights": weights,
            "holdings": holdings,
            "trades": trades,
            "turnover": turnover,
        }
