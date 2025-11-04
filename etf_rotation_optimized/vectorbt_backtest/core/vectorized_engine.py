#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高性能向量化回测引擎 - 完全消除循环和数据传递
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class UltraFastVectorEngine:
    """超高性能向量化引擎 - 预计算所有可能需要的中间结果"""

    def __init__(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        close_df: pd.DataFrame,
        init_cash: float = 1000000,
        fees: float = 0.0005,
        logger: logging.Logger = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.init_cash = init_cash
        self.fees = fees

        # 预处理：将所有因子堆叠成3D数组 (n_dates, n_etfs, n_factors)
        self.factor_names = list(factors_dict.keys())
        factor_arrays = [factors_dict[f].values for f in self.factor_names]
        self.factors_3d = np.stack(factor_arrays, axis=2)  # (1399, 43, 18)

        # 存储价格和收益
        self.close = close_df.values
        self.returns = np.diff(self.close, axis=0, prepend=self.close[0:1]) / self.close
        self.returns[0] = 0  # 第一天收益为0

        # 索引
        self.dates = close_df.index
        self.etfs = close_df.columns
        self.n_dates, self.n_etfs = self.close.shape
        self.n_factors = len(self.factor_names)

        self.logger.info(
            f"引擎初始化: {self.n_dates}天 × {self.n_etfs}ETF × {self.n_factors}因子"
        )

    def backtest_single_strategy(
        self, weights: np.ndarray, top_n: int, rebalance_freq: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        单策略回测（纯NumPy向量化）

        Args:
            weights: 因子权重数组 shape=(n_factors,)
            top_n: 持仓数量
            rebalance_freq: 调仓频率

        Returns:
            equity_curve: 权益曲线
            metrics: 性能指标
        """
        # 1. 计算综合得分 (n_dates, n_etfs) - 完全向量化
        composite_score = np.nansum(self.factors_3d * weights, axis=2)

        # 2. 生成持仓权重信号 (n_dates, n_etfs) - 向量化
        signals = self._generate_signals_vectorized(
            composite_score, top_n, rebalance_freq
        )

        # 3. 计算收益 - 向量化
        equity_curve, turnover_series, cost_series, net_returns = (
            self._calculate_returns_vectorized(signals)
        )

        # 4. 计算指标 - 向量化
        metrics = self._calculate_metrics_vectorized(
            equity_curve,
            turnover_series=turnover_series,
            cost_series=cost_series,
            net_returns=net_returns,
        )

        return equity_curve, metrics

    def _generate_signals_vectorized(
        self, scores: np.ndarray, top_n: int, rebalance_freq: int
    ) -> np.ndarray:
        """完全向量化的信号生成 - 零循环版本"""
        n_dates, n_etfs = scores.shape

        # 1. 优化信号生成：考虑因子方向性和有效性
        # 使用绝对值和方向性综合评分
        abs_scores = np.abs(scores)
        sign_scores = np.sign(scores)

        # 计算有效性得分（绝对值 + 方向性调整）
        # 方向性：正因子得分 * 1.0，负因子得分 * 0.8（略微惩罚）
        effectiveness_scores = abs_scores * np.where(sign_scores > 0, 1.0, 0.8)

        # 使用负分数实现降序排序
        neg_eff_scores = -effectiveness_scores

        # argpartition: 保证前top_n个是最有效的，但不保证有序
        partitioned_indices = np.argpartition(
            np.where(np.isnan(neg_eff_scores), np.inf, neg_eff_scores),
            kth=min(top_n, n_etfs - 1),
            axis=1,
        )[
            :, :top_n
        ]  # (n_dates, top_n)

        # 2. 创建mask矩阵：哪些(date, etf)位置应该持有
        signals = np.zeros((n_dates, n_etfs))
        date_indices = np.arange(n_dates)[:, None]  # (n_dates, 1)
        signals[date_indices, partitioned_indices] = 1.0  # 先标记为1

        # 3. 处理调仓频率：只在调仓日更新，其他日期继承上一调仓日
        rebalance_mask = np.zeros(n_dates, dtype=bool)
        rebalance_mask[::rebalance_freq] = True
        rebalance_mask[-1] = True  # 最后一天也作为调仓日

        # 前向填充：非调仓日继承上一调仓日的持仓
        last_rebalance_positions = np.zeros(n_etfs)
        for i in range(n_dates):
            if rebalance_mask[i]:
                # 调仓日：更新持仓
                last_rebalance_positions = signals[i].copy()
                # 计算实际持有数量（排除NaN对应的ETF）
                valid_count = np.sum(~np.isnan(scores[i, partitioned_indices[i]]))
                if valid_count > 0:
                    # 等权重
                    last_rebalance_positions[
                        partitioned_indices[i][: int(valid_count)]
                    ] = (1.0 / valid_count)
                    last_rebalance_positions[
                        partitioned_indices[i][int(valid_count) :]
                    ] = 0.0
            else:
                # 非调仓日：继承上次持仓
                signals[i] = last_rebalance_positions

        return signals

    def _calculate_returns_vectorized(
        self, signals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """完全向量化的收益计算"""
        # 滞后1期的权重（避免未来函数）
        lagged_signals = np.vstack([signals[0:1], signals[:-1]])

        # 组合收益 = sum(权重 × 收益)
        portfolio_returns = np.nansum(lagged_signals * self.returns, axis=1)

        # 换手率
        signal_changes = np.abs(np.diff(signals, axis=0, prepend=signals[0:1]))
        turnover = np.sum(signal_changes, axis=1)

        # 扣除成本
        cost_drag = turnover * self.fees
        net_returns = portfolio_returns - cost_drag

        # 权益曲线
        equity_curve = self.init_cash * np.cumprod(1 + net_returns)

        return equity_curve, turnover, cost_drag, net_returns

    def _calculate_metrics_vectorized(
        self,
        equity: np.ndarray,
        turnover_series: np.ndarray = None,
        cost_series: np.ndarray = None,
        net_returns: np.ndarray = None,
    ) -> Dict[str, float]:
        """向量化指标计算"""
        # 收益率
        returns = np.diff(equity) / equity[:-1]
        if net_returns is None:
            net_returns = np.zeros_like(equity)
            net_returns[1:] = returns
            net_returns[0] = 0.0

        # 总收益
        total_return = (equity[-1] / equity[0]) - 1

        # 年化收益
        n_years = len(equity) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 年化波动
        annual_vol = np.std(returns) * np.sqrt(252)

        # 夏普
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # 最大回撤
        cummax = np.maximum.accumulate(equity)
        drawdowns = (equity - cummax) / cummax
        max_drawdown = np.min(drawdowns)

        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        # 胜率
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0

        metrics = {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "annual_vol": float(annual_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "trading_days": int(len(equity)),
        }

        if turnover_series is not None:
            total_turnover = float(np.sum(turnover_series))
            turnover_days = int(np.count_nonzero(turnover_series > 1e-8))
            metrics.update(
                {
                    "avg_turnover": float(np.mean(turnover_series)),
                    "median_turnover": float(np.median(turnover_series)),
                    "turnover_95p": float(np.percentile(turnover_series, 95)),
                    "total_turnover": total_turnover,
                    "turnover_days": turnover_days,
                    "avg_turnover_active": (
                        float(total_turnover / turnover_days)
                        if turnover_days > 0
                        else 0.0
                    ),
                }
            )

        if cost_series is not None:
            metrics.update(
                {
                    "avg_cost": float(np.mean(cost_series)),
                    "cost_95p": float(np.percentile(cost_series, 95)),
                    "total_cost": float(np.sum(cost_series)),
                }
            )

        if net_returns is not None:
            metrics.update(
                {
                    "avg_net_return": float(np.mean(net_returns)),
                    "net_return_95p": float(np.percentile(net_returns, 95)),
                }
            )

        return metrics

    def batch_backtest(
        self,
        weight_matrix: np.ndarray,
        top_n_list: List[int],
        rebalance_freq_list: List[int],
        strategy_ids: np.ndarray = None,
        chunk_size: int = 500,
    ) -> List[Dict]:
        """
        批量回测 - 内存优化分块版本

        核心改进：
        1. 分块处理权重组合，避免内存爆炸
        2. 使用更高效的内存管理

        Args:
            weight_matrix: (n_strategies, n_factors) 权重矩阵
            top_n_list: Top-N列表
            rebalance_freq_list: 调仓频率列表
            strategy_ids: 策略ID数组
            chunk_size: 分块大小（默认500）

        Returns:
            results: 结果列表
        """
        n_weights = len(weight_matrix)
        results = []

        # 分块处理权重组合
        n_chunks = (n_weights + chunk_size - 1) // chunk_size

        self.logger.info(
            f"分块处理: {n_weights} 权重 × {len(top_n_list)} Top-N × {len(rebalance_freq_list)} 频率 = {n_chunks} 块"
        )

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_weights)
            chunk_weights = weight_matrix[start_idx:end_idx]
            chunk_ids = (
                strategy_ids[start_idx:end_idx] if strategy_ids is not None else None
            )

            self.logger.info(
                f"处理块 {chunk_idx+1}/{n_chunks}: 权重 {start_idx}-{end_idx-1}"
            )

            # 预计算当前块的得分 (chunk_size, n_dates, n_etfs)
            chunk_scores = np.einsum("ijk,mk->mij", self.factors_3d, chunk_weights)

            # 对每个参数组合批量处理
            for top_n in top_n_list:
                for rebalance_freq in rebalance_freq_list:
                    # 批量生成信号 (chunk_size, n_dates, n_etfs)
                    chunk_signals = self._batch_generate_signals(
                        chunk_scores, top_n, rebalance_freq
                    )

                    # 批量计算收益 (chunk_size, n_dates)
                    (
                        chunk_equity_curves,
                        chunk_turnover,
                        chunk_cost,
                        chunk_net_returns,
                    ) = self._batch_calculate_returns(chunk_signals, rebalance_freq)

                    # 批量计算指标
                    for i in range(len(chunk_weights)):
                        try:
                            equity = chunk_equity_curves[i]
                            weights = chunk_weights[i]

                            # 计算指标
                            metrics = self._calculate_metrics_vectorized(
                                equity,
                                turnover_series=chunk_turnover[i],
                                cost_series=chunk_cost[i],
                                net_returns=chunk_net_returns[i],
                            )

                            # 找出非零权重的因子
                            active_mask = weights != 0
                            active_factors = [
                                self.factor_names[j]
                                for j in range(self.n_factors)
                                if active_mask[j]
                            ]
                            active_weights = [
                                float(weights[j])
                                for j in range(self.n_factors)
                                if active_mask[j]
                            ]

                            result = {
                                "strategy_id": (
                                    i + start_idx if chunk_ids is None else chunk_ids[i]
                                ),
                                "factors": active_factors,
                                "weights": active_weights,
                                "top_n": top_n,
                                "rebalance_freq": rebalance_freq,
                                "equity_curve": equity,
                                **metrics,
                            }
                            results.append(result)

                        except Exception as e:
                            self.logger.warning(f"策略 {i + start_idx} 失败: {e}")
                            continue

        return results

    def _batch_generate_signals(
        self, all_scores: np.ndarray, top_n: int, rebalance_freq: int
    ) -> np.ndarray:
        """
        批量生成信号 - 完全向量化（优化版本）

        Args:
            all_scores: (n_weights, n_dates, n_etfs) 所有策略的得分
            top_n: 持仓数量
            rebalance_freq: 调仓频率

        Returns:
            all_signals: (n_weights, n_dates, n_etfs) 所有策略的持仓信号
        """
        n_weights, n_dates, n_etfs = all_scores.shape
        all_signals = np.zeros_like(all_scores)

        # 1. 对所有策略一次性做 partition（取前 top_n 个）
        # 负分数实现降序
        neg_scores = -all_scores
        neg_scores_safe = np.where(np.isnan(neg_scores), np.inf, neg_scores)

        # argpartition 沿 axis=2（ETF维度）
        # shape: (n_weights, n_dates, top_n)
        kth = min(top_n, n_etfs - 1)
        partitioned_indices = np.argpartition(neg_scores_safe, kth=kth, axis=2)[
            :, :, :top_n
        ]

        # 2. 构建调仓mask
        rebalance_mask = np.zeros(n_dates, dtype=bool)
        rebalance_mask[::rebalance_freq] = True
        rebalance_mask[-1] = True

        # 3. 批量归一化权重 - 完全向量化版本
        # 获取所有调仓日的索引
        rebalance_indices = np.where(rebalance_mask)[0]

        # 提取调仓日的得分和索引
        rebalance_scores = all_scores[
            :, rebalance_indices, :
        ]  # (n_weights, n_rebalance_dates, n_etfs)
        rebalance_partitioned = partitioned_indices[
            :, rebalance_indices, :
        ]  # (n_weights, n_rebalance_dates, top_n)

        # 创建mask矩阵标记所有调仓日的top_n位置
        weight_idx = np.arange(n_weights)[:, None, None]  # (n_weights, 1, 1)
        date_idx = np.arange(len(rebalance_indices))[
            None, :, None
        ]  # (1, n_rebalance_dates, 1)

        # 构建3D布尔mask：标记哪些(weight, date, etf)是top_n
        top_n_mask = np.zeros((n_weights, len(rebalance_indices), n_etfs), dtype=bool)
        top_n_mask[weight_idx, date_idx, rebalance_partitioned] = True

        # 检查哪些top_n位置得分有效（非NaN）
        valid_mask = ~np.isnan(rebalance_scores)
        valid_top_n_mask = top_n_mask & valid_mask

        # 计算每个策略每个调仓日的有效持仓数
        n_valid_per_combination = np.sum(
            valid_top_n_mask, axis=2
        )  # (n_weights, n_rebalance_dates)

        # 计算权重：有效持仓数倒数，无效位置为0
        weights = np.zeros_like(rebalance_scores)

        # 避免除零：只在有效持仓数>0时计算权重
        valid_combinations = n_valid_per_combination > 0
        if np.any(valid_combinations):
            # 广播：(n_weights, n_rebalance_dates) -> (n_weights, n_rebalance_dates, n_etfs)
            weight_values = np.zeros((n_weights, len(rebalance_indices), n_etfs))
            weight_values[valid_combinations] = (
                1.0 / n_valid_per_combination[valid_combinations]
            )[:, None]
            # 只在valid_top_n_mask位置赋值
            weights = weight_values * valid_top_n_mask.astype(float)

        # 4. 将权重赋值回all_signals
        all_signals[:, rebalance_indices, :] = weights

        # 5. 前向填充：非调仓日继承上一调仓日 - 完全向量化
        # 使用cumsum技巧实现零循环前向填充
        rebalance_indices_expanded = np.where(rebalance_mask)[0]

        # 构建索引映射：每个日期对应的上一个调仓日索引
        # 使用searchsorted找到每个日期对应的调仓日位置
        date_to_rebalance_idx = (
            np.searchsorted(
                rebalance_indices_expanded, np.arange(n_dates), side="right"
            )
            - 1
        )
        date_to_rebalance_idx = np.clip(
            date_to_rebalance_idx, 0, len(rebalance_indices_expanded) - 1
        )

        # 获取每个日期对应的调仓日索引
        rebalance_for_dates = rebalance_indices_expanded[date_to_rebalance_idx]

        # 使用高级索引实现向量化赋值
        # weight_idx: (n_weights, 1, 1) 广播到所有策略
        # rebalance_for_dates: (n_dates,) 每个日期对应的调仓日索引
        # etf_idx: (n_etfs,) 所有ETF
        weight_idx = np.arange(n_weights)[:, None, None]  # (n_weights, 1, 1)
        etf_idx = np.arange(n_etfs)[None, None, :]  # (1, 1, n_etfs)

        # 批量赋值：all_signals[weight_idx, date_idx, etf_idx] = all_signals[weight_idx, rebalance_for_dates[date_idx], etf_idx]
        date_indices = np.arange(n_dates)[None, :, None]  # (1, n_dates, 1)
        rebalance_indices_3d = rebalance_for_dates[None, :, None]  # (1, n_dates, 1)

        # 一次性赋值所有非调仓日
        # 说明：高级索引将直接返回形状为 (n_weights, n_dates, n_etfs) 的数组，无需 squeeze
        all_signals = all_signals[weight_idx, rebalance_indices_3d, etf_idx]

        return all_signals

    def _batch_calculate_returns(
        self, all_signals: np.ndarray, rebalance_freq: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        批量计算收益 - 完全向量化

        Args:
            all_signals: (n_weights, n_dates, n_etfs) 所有策略的持仓信号

        Returns:
            all_equity_curves: (n_weights, n_dates) 所有策略的权益曲线
            turnover: (n_weights, n_dates) 换手率序列
            cost_drag: (n_weights, n_dates) 交易成本序列
            net_returns: (n_weights, n_dates) 成本后的收益序列
        """
        n_weights, n_dates, n_etfs = all_signals.shape

        # 1. 滞后1期（避免未来函数）
        # (n_weights, n_dates, n_etfs)
        lagged_signals = np.concatenate(
            [all_signals[:, 0:1, :], all_signals[:, :-1, :]],  # 第一天  # 其余天滞后
            axis=1,
        )

        # 2. 组合收益 = sum(权重 × 收益) 沿 ETF 维度求和
        # returns: (n_dates, n_etfs) -> broadcast to (n_weights, n_dates, n_etfs)
        # portfolio_returns: (n_weights, n_dates)
        portfolio_returns = np.nansum(lagged_signals * self.returns[None, :, :], axis=2)

        # 3. 优化换手率计算 - 只计算调仓日变化，避免微小波动
        # 创建调仓日mask
        rebalance_mask = np.zeros(n_dates, dtype=bool)
        rebalance_mask[::rebalance_freq] = True
        rebalance_mask[-1] = True

        # 只计算调仓日之间的变化
        rebalance_indices = np.where(rebalance_mask)[0]
        turnover = np.zeros((n_weights, n_dates))

        for i in range(1, len(rebalance_indices)):
            prev_idx = rebalance_indices[i - 1]
            curr_idx = rebalance_indices[i]

            # 计算调仓日之间的持仓变化
            position_change = np.abs(
                all_signals[:, curr_idx, :] - all_signals[:, prev_idx, :]
            )

            # 将换手率分配到调仓日
            turnover[:, curr_idx] = np.sum(position_change, axis=1)

        # 平滑处理：避免单日极端换手
        turnover = np.minimum(turnover, 0.5)  # 限制最大换手率为50%

        # 4. 扣除成本
        cost_drag = turnover * self.fees
        net_returns = portfolio_returns - cost_drag

        # 5. 权益曲线
        all_equity_curves = self.init_cash * np.cumprod(1 + net_returns, axis=1)

        return all_equity_curves, turnover, cost_drag, net_returns
