"""
信号到持仓的构建模块

将WFO生成的因子权重信号转换为实际持仓权重
- 信号标准化
- Top-N选择
- 权重归一化
- 交易成本扣除
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .trading_cost_model import AShareETFTradingCost


class PortfolioConstructor:
    """信号到持仓的构建器"""

    def __init__(self, top_n: int = 5, trading_cost_model: AShareETFTradingCost = None):
        self.top_n = top_n
        self.cost_model = trading_cost_model or AShareETFTradingCost()

    def construct_portfolio(
        self, factor_signals: np.ndarray, etf_prices: np.ndarray, etf_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建投资组合持仓

        Args:
            factor_signals: 因子信号 (T, N)
            etf_prices: ETF价格 (T, N)
            etf_names: ETF名称列表

        Returns:
            portfolio_weights: 持仓权重 (T, N)
            transaction_costs: 交易成本 (T,)

        注意:
            🔧 修复前视偏差 - 使用T-1信号构建T时刻持仓
        """
        T, N = factor_signals.shape
        portfolio_weights = np.zeros((T, N))
        transaction_costs = np.zeros(T)

        # 当前持仓（用于计算换手）
        current_weights = np.zeros(N)

        for t in range(T):
            # 🔧 修复: 使用T-1信号，避免前视偏差
            if t == 0:
                # 第一天无历史信号，空仓
                portfolio_weights[t] = current_weights
                continue

            # 使用t-1时刻的信号
            signals_t = factor_signals[t - 1]
            valid_mask = ~np.isnan(signals_t)

            if not np.any(valid_mask):
                portfolio_weights[t] = current_weights
                continue

            # 2. 选择Top-N ETF
            valid_signals = signals_t[valid_mask]
            valid_indices = np.where(valid_mask)[0]

            # 按信号降序排序
            sorted_indices = valid_indices[np.argsort(valid_signals)[::-1]]
            top_indices = sorted_indices[: self.top_n]

            # 3. 等权重分配
            if len(top_indices) > 0:
                weight_per_etf = 1.0 / len(top_indices)

                # 4. 计算交易成本
                new_weights = np.zeros(N)
                new_weights[top_indices] = weight_per_etf

                # 计算换手
                turnover = np.sum(np.abs(new_weights - current_weights))

                if turnover > 1e-10:  # 有换手
                    # 🔧 修复: 使用归一化资本，避免成本爆炸
                    portfolio_value = 1.0
                    trade_value = portfolio_value * turnover

                    # 计算成本
                    cost = self.cost_model.calculate_cost(trade_value)
                    transaction_costs[t] = cost

                    # 更新持仓
                    current_weights = new_weights.copy()

            portfolio_weights[t] = current_weights

        return portfolio_weights, transaction_costs


class PerformanceCalculator:
    """绩效计算模块"""

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

        Args:
            portfolio_weights: 持仓权重 (T, N)
            etf_returns: ETF收益率 (T, N)
            transaction_costs: 交易成本 (T,)

        Returns:
            portfolio_returns: 组合净收益 (T,)

        注意:
            🔧 修复成本率计算 - 使用稳定的资本基数
        """
        # 计算持仓收益
        portfolio_gross_returns = np.sum(portfolio_weights * etf_returns, axis=1)

        # 🔧 修复: 使用稳定的资本基数，避免分母崩溃
        portfolio_value = 1.0
        cost_ratio = transaction_costs / portfolio_value

        # 扣除交易成本
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

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_return": np.prod(1 + returns) - 1,
        }
