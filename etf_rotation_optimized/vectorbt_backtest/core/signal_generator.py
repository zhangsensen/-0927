#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化信号生成器 - 纯向量化实现，避免未来函数
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class VectorizedSignalGenerator:
    """向量化信号生成器"""

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_composite_score(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        factor_weights: Dict[str, float],
        method: str = "weighted_sum",
    ) -> pd.DataFrame:
        """
        生成综合得分（完全向量化，无未来函数）

        Args:
            factors_dict: {factor_name: DataFrame}
            factor_weights: {factor_name: weight}
            method: 'weighted_sum' 或 'rank_average'

        Returns:
            composite_score: DataFrame(index=date, columns=etf)
        """
        # 提取相关因子
        selected_factors = [f for f in factor_weights.keys() if f in factors_dict]
        if not selected_factors:
            raise ValueError("没有有效的因子")

        # 获取共同形状
        first_df = factors_dict[selected_factors[0]]
        composite = pd.DataFrame(0.0, index=first_df.index, columns=first_df.columns)

        if method == "weighted_sum":
            # 加权求和
            for factor_name in selected_factors:
                weight = factor_weights[factor_name]
                composite += factors_dict[factor_name].fillna(0) * weight

        elif method == "rank_average":
            # 排名平均（每个因子先排名，再加权平均）
            for factor_name in selected_factors:
                weight = factor_weights[factor_name]
                # 每日横截面排名（避免未来函数）
                ranks = factors_dict[factor_name].rank(
                    axis=1, method="average", na_option="keep"
                )
                composite += ranks.fillna(0) * weight

        else:
            raise ValueError(f"未知方法: {method}")

        return composite

    def generate_topn_signals(
        self, composite_score: pd.DataFrame, top_n: int, rebalance_freq: int
    ) -> pd.DataFrame:
        """
        生成Top-N交易信号（完全向量化，无嵌套循环）

        Args:
            composite_score: 综合得分DataFrame
            top_n: 持仓数量
            rebalance_freq: 调仓频率（天）

        Returns:
            signals: DataFrame(index=date, columns=etf)，值为目标权重（0或1/top_n）
        """
        n_dates, n_etfs = composite_score.shape
        signals = pd.DataFrame(
            0.0, index=composite_score.index, columns=composite_score.columns
        )

        # 生成调仓日期索引
        rebalance_indices = list(range(0, n_dates, rebalance_freq))
        if rebalance_indices[-1] != n_dates - 1:
            rebalance_indices.append(n_dates - 1)

        # 向量化选择Top-N（无内层循环）
        for i, next_i in zip(rebalance_indices[:-1], rebalance_indices[1:]):
            # 当日得分
            scores = composite_score.iloc[i]
            valid_scores = scores.dropna()

            if len(valid_scores) == 0:
                continue

            # 选出Top-N
            actual_top_n = min(top_n, len(valid_scores))
            top_etfs = valid_scores.nlargest(actual_top_n).index
            weight = 1.0 / actual_top_n

            # 使用loc批量赋值（向量化）
            signals.loc[signals.index[i:next_i], top_etfs] = weight

        return signals

    def backtest_portfolio(
        self,
        signals: pd.DataFrame,
        close_df: pd.DataFrame,
        init_cash: float = 1000000,
        fees: float = 0.0005,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        完全向量化回测（无循环）

        Args:
            signals: 目标权重信号（0-1之间）
            close_df: 收盘价
            init_cash: 初始资金
            fees: 双边费用率

        Returns:
            equity_curve: 权益曲线Series
            holdings: 持仓DataFrame（股数）
        """
        # 计算收益率（向量化）
        returns = close_df.pct_change(fill_method=None).fillna(0)

        # 计算权重变化（向量化）
        weight_changes = signals.diff().fillna(signals)
        turnover = weight_changes.abs().sum(axis=1)

        # 计算持仓收益（向量化）
        # 策略收益 = 权重 * 收益率的加权和
        portfolio_returns = (signals.shift(1) * returns).sum(axis=1)

        # 扣除交易成本（向量化）
        cost_drag = turnover * fees
        net_returns = portfolio_returns - cost_drag

        # 计算权益曲线（向量化）
        equity_curve = init_cash * (1 + net_returns).cumprod()
        equity_curve.iloc[0] = init_cash  # 第一天初始资金

        # 持仓（简化版，不计算具体股数）
        holdings = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        return equity_curve, holdings

    def calculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        计算性能指标

        Returns:
            metrics: 指标字典
        """
        # 收益率
        returns = equity_curve.pct_change().dropna()

        # 总收益
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # 年化收益
        n_days = len(equity_curve)
        n_years = n_days / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = equity_curve.expanding().max()
        drawdowns = (equity_curve - cummax) / cummax
        max_drawdown = drawdowns.min()

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # 胜率
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "trading_days": int(n_days),
        }
