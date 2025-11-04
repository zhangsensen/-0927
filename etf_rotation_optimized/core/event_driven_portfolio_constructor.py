"""
事件驱动组合构建器 (简版，Lite)

基于每日信号，应用严格 T+1：t-1 信号 → t 日持仓。
支持 Top-N 等权、最小持有天数、日换手上限三项核心约束。

该实现用于快速评估与对比（不含成本模型），生产版请参考
本文件后半段 "A股ETF专用" 的构建器或 vectorbt_backtest 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EventConstructorLiteConfig:
    top_n: int = 6
    min_holding_days: int = 2
    max_daily_turnover: float = 0.6  # 0~1，按权重之和界定


class EventDrivenPortfolioConstructorLite:
    def __init__(self, cfg: Optional[EventConstructorLiteConfig] = None):
        self.cfg = cfg or EventConstructorLiteConfig()

    def build(self, signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
            根据信号与约束，返回日度组合收益序列（已是净收益，不含成本模型）。

        简化处理：只施加 T+1、Top-N、min_holding_days 与日换手上限，
            不模拟成本；WFO 阶段以相对比较为主。
        """
        T, N = returns.shape
        top_n = max(1, int(self.cfg.top_n))
        min_days = max(1, int(self.cfg.min_holding_days))
        max_turn = float(self.cfg.max_daily_turnover)

        weights = np.zeros((T, N), dtype=float)
        holding_days = np.zeros(N, dtype=int)

        for t in range(1, T):
            sig_prev = signals[t - 1]
            if np.all(np.isnan(sig_prev)):
                weights[t] = weights[t - 1]
                # 继续累计持有天数
                holding_days[weights[t] > 0] += 1
                continue

            # 基于 t-1 的信号选 Top-N
            valid = ~np.isnan(sig_prev)
            idx = np.where(valid)[0]
            if idx.size == 0:
                weights[t] = weights[t - 1]
                holding_days[weights[t] > 0] += 1
                continue

            ranked = idx[np.argsort(sig_prev[valid])[::-1]]
            target = ranked[:top_n]

            # 先复制昨日权重
            new_w = weights[t - 1].copy()

            # 卖出：对不在 target 且已满足最小持有天数的仓位清零
            to_sell = np.setdiff1d(np.where(new_w > 0)[0], target)
            for j in to_sell:
                if holding_days[j] >= min_days:
                    new_w[j] = 0.0
                    holding_days[j] = 0

            # 买入：补足到 target 等权（受日换手上限约束）
            in_target = target
            if in_target.size > 0:
                eq = 1.0 / in_target.size
                for j in in_target:
                    new_w[j] = eq

            # 计算当日换手并施加上限
            turnover = np.sum(np.abs(new_w - weights[t - 1]))
            if turnover > max_turn:
                # 线性收缩到满足上限
                if turnover > 1e-12:
                    alpha = max_turn / turnover
                    new_w = weights[t - 1] + alpha * (new_w - weights[t - 1])

            weights[t] = new_w
            holding_days[weights[t] > 0] += 1

        # 组合日收益 = sum(weights[t] * returns[t])
        port_ret = (weights * returns).sum(axis=1)
        port_ret[0] = 0.0
        return port_ret


"""
事件驱动持仓构建器 - A股ETF专用

核心特性:
1. T+1交易约束（今天买，明天才能卖）
2. 最小持有期（避免频繁交易）
3. 信号质量过滤（只在信号强时交易）
4. 每日评估，有信号就交易（事件驱动）

作者: Linus Mode
日期: 2025-11-03
"""

from typing import Dict, List, Tuple

import numpy as np

from .trading_cost_model import AShareETFTradingCost


class EventDrivenPortfolioConstructor:
    """
    事件驱动持仓构建器（A股ETF专用）

    特性:
    - 每日评估信号（事件驱动）
    - T+1交易约束
    - 最小持有期
    - 信号质量过滤
    """

    def __init__(
        self,
        top_n: int = 5,
        min_holding_days: int = 3,
        max_daily_turnover: float = 0.5,
        signal_strength_threshold: float = 0.0,
        trading_cost_model: AShareETFTradingCost = None,
    ):
        """
        初始化事件驱动构建器

        Args:
            top_n: 持仓数量
            min_holding_days: 最小持有期（天）
            max_daily_turnover: 每日最大换手率
            signal_strength_threshold: 信号强度阈值（Z-score）
            trading_cost_model: 交易成本模型
        """
        self.top_n = top_n
        self.min_holding_days = min_holding_days
        self.max_daily_turnover = max_daily_turnover
        self.signal_strength_threshold = signal_strength_threshold
        self.cost_model = trading_cost_model or AShareETFTradingCost()

    def construct_portfolio(
        self, factor_signals: np.ndarray, etf_prices: np.ndarray, etf_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        构建事件驱动投资组合

        Args:
            factor_signals: 因子信号 (T, N)
            etf_prices: ETF价格 (T, N)
            etf_names: ETF名称列表

        Returns:
            portfolio_weights: 持仓权重 (T, N)
            transaction_costs: 交易成本 (T,)
            stats: 统计信息

        注意:
            🔧 T+1约束 - 今天买入的ETF，明天才能卖出
            🔧 信号延迟 - 使用T-1信号构建T时刻持仓
        """
        T, N = factor_signals.shape
        portfolio_weights = np.zeros((T, N))
        transaction_costs = np.zeros(T)
        daily_turnover = np.zeros(T)

        # 持仓状态跟踪
        current_weights = np.zeros(N)
        holding_days = np.zeros(N)  # 每个ETF的持有天数
        last_buy_day = np.full(N, -999)  # 每个ETF的最后买入日

        # 统计信息
        trade_count = 0
        signal_triggered_days = 0

        for t in range(T):
            # 🔧 T-1信号延迟
            if t == 0:
                portfolio_weights[t] = current_weights
                continue

            # 使用t-1时刻的信号
            signals_t = factor_signals[t - 1]

            # 更新持有天数
            holding_days[current_weights > 0] += 1

            # 1. 信号质量过滤
            valid_mask = ~np.isnan(signals_t)
            if not np.any(valid_mask):
                portfolio_weights[t] = current_weights
                continue

            # 标准化信号（Z-score）
            valid_signals = signals_t[valid_mask]
            signal_mean = np.mean(valid_signals)
            signal_std = np.std(valid_signals)

            if signal_std < 1e-10:
                # 信号无差异，保持当前持仓
                portfolio_weights[t] = current_weights
                continue

            z_scores = (signals_t - signal_mean) / signal_std

            # 只考虑信号强度超过阈值的ETF
            strong_signal_mask = (
                z_scores > self.signal_strength_threshold
            ) & valid_mask

            if not np.any(strong_signal_mask):
                # 无强信号，保持当前持仓
                portfolio_weights[t] = current_weights
                continue

            # 2. 选择Top-N ETF（基于信号强度）
            strong_indices = np.where(strong_signal_mask)[0]
            strong_z_scores = z_scores[strong_indices]

            # 按Z-score降序排序
            sorted_idx = np.argsort(strong_z_scores)[::-1]
            top_indices = strong_indices[sorted_idx[: self.top_n]]

            # 3. 构建目标持仓
            target_weights = np.zeros(N)
            if len(top_indices) > 0:
                weight_per_etf = 1.0 / len(top_indices)
                target_weights[top_indices] = weight_per_etf

            # 4. 应用T+1约束和最小持有期
            new_weights = self._apply_trading_constraints(
                current_weights=current_weights,
                target_weights=target_weights,
                holding_days=holding_days,
                last_buy_day=last_buy_day,
                current_day=t,
            )

            # 5. 应用每日换手限制
            turnover = np.sum(np.abs(new_weights - current_weights))
            daily_turnover[t] = turnover
            if turnover > self.max_daily_turnover:
                # 按信号强度优先级调整
                new_weights = self._limit_turnover(
                    current_weights=current_weights,
                    target_weights=new_weights,
                    z_scores=z_scores,
                    max_turnover=self.max_daily_turnover,
                )
                turnover = np.sum(np.abs(new_weights - current_weights))
                daily_turnover[t] = turnover

            # 6. 计算交易成本
            if turnover > 1e-10:
                portfolio_value = 1.0
                trade_value = portfolio_value * turnover
                # 卖出与买入都按简化模型计入佣金与滑点；印花税(若设置)>0仅卖出时计
                # 这里缺少逐笔方向拆分，简化按双向成本估算
                cost = self.cost_model.calculate_cost(trade_value, is_sell=False)
                transaction_costs[t] = cost

                # 更新统计
                trade_count += 1
                signal_triggered_days += 1

                # 更新买入日期
                buy_mask = new_weights > current_weights
                last_buy_day[buy_mask] = t

                # 重置卖出ETF的持有天数
                sell_mask = new_weights < current_weights
                holding_days[sell_mask] = 0

            # 更新当前持仓
            current_weights = new_weights.copy()
            portfolio_weights[t] = current_weights

        # 统计信息
        stats = {
            "trade_count": trade_count,
            "signal_triggered_days": signal_triggered_days,
            "avg_turnover": float(np.mean(daily_turnover)) if T > 0 else 0.0,
            "trade_frequency": trade_count / T if T > 0 else 0,
        }

        return portfolio_weights, transaction_costs, stats

    def _apply_trading_constraints(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        holding_days: np.ndarray,
        last_buy_day: np.ndarray,
        current_day: int,
    ) -> np.ndarray:
        """
        应用交易约束（T+1 + 最小持有期）

        Args:
            current_weights: 当前持仓
            target_weights: 目标持仓
            holding_days: 持有天数
            last_buy_day: 最后买入日
            current_day: 当前日期

        Returns:
            调整后的持仓
        """
        new_weights = target_weights.copy()

        # 遍历每个ETF
        for i in range(len(current_weights)):
            # 规则：最小持有期 - 持有不足 min_holding_days 的不能减仓
            if current_weights[i] > 0 and holding_days[i] < self.min_holding_days:
                # 强制保持持仓
                new_weights[i] = max(new_weights[i], current_weights[i])

        return new_weights

    def _limit_turnover(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        z_scores: np.ndarray,
        max_turnover: float,
    ) -> np.ndarray:
        """
        限制每日换手率

        策略: 按信号强度优先级，优先交易信号最强的ETF

        Args:
            current_weights: 当前持仓
            target_weights: 目标持仓
            z_scores: 信号Z-score
            max_turnover: 最大换手率

        Returns:
            调整后的持仓
        """
        # 计算每个ETF的交易量和信号强度
        trade_amounts = np.abs(target_weights - current_weights)

        # 按信号强度排序（买入优先强信号，卖出优先弱信号）
        buy_mask = target_weights > current_weights
        sell_mask = target_weights < current_weights

        # 优先级: 强信号买入 > 弱信号卖出
        priority = np.zeros_like(z_scores)
        priority[buy_mask] = z_scores[buy_mask]  # 买入优先级=信号强度
        priority[sell_mask] = -z_scores[sell_mask]  # 卖出优先级=负信号强度

        # 按优先级排序
        sorted_indices = np.argsort(priority)[::-1]

        # 逐个添加交易，直到达到换手限制
        new_weights = current_weights.copy()
        cumulative_turnover = 0.0

        for idx in sorted_indices:
            if trade_amounts[idx] < 1e-10:
                continue

            # 尝试添加这笔交易
            potential_turnover = cumulative_turnover + trade_amounts[idx]

            if potential_turnover <= max_turnover:
                # 可以完整执行
                new_weights[idx] = target_weights[idx]
                cumulative_turnover = potential_turnover
            else:
                # 部分执行
                remaining_capacity = max_turnover - cumulative_turnover
                if remaining_capacity > 1e-10:
                    if buy_mask[idx]:
                        new_weights[idx] = current_weights[idx] + remaining_capacity
                    else:
                        new_weights[idx] = current_weights[idx] - remaining_capacity
                break

        # 归一化（确保权重和为1）
        total_weight = np.sum(new_weights)
        if total_weight > 1e-10:
            new_weights = new_weights / total_weight

        return new_weights


class EventDrivenPerformanceCalculator:
    """事件驱动绩效计算器"""

    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def calculate_returns(
        self,
        portfolio_weights: np.ndarray,
        etf_returns: np.ndarray,
        transaction_costs: np.ndarray,
    ) -> np.ndarray:
        """
        计算组合收益（扣除成本）

        注意: T+1延迟已在持仓构建中处理
        """
        # 计算持仓收益
        portfolio_gross_returns = np.sum(portfolio_weights * etf_returns, axis=1)

        # 扣除交易成本
        portfolio_value = 1.0
        cost_ratio = transaction_costs / portfolio_value
        portfolio_net_returns = portfolio_gross_returns - cost_ratio

        return portfolio_net_returns

    def calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """计算绩效指标"""
        if len(returns) < 30:
            return {}

        # 年化收益率
        annual_return = np.prod(1 + returns) ** (252 / len(returns)) - 1

        # 年化波动率
        annual_vol = np.std(returns) * np.sqrt(252)

        # Sharpe比率
        sharpe = (
            (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0
        )

        # 最大回撤
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 胜率
        win_rate = np.mean(returns > 0)

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_return": np.prod(1 + returns) - 1,
            "calmar_ratio": calmar,
        }
