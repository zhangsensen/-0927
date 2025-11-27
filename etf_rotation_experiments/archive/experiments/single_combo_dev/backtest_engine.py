#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 真实回测引擎

在理论估算基础上，提供基于逐日路径的真实回测实现。
支持动态仓位和移动止损机制。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging


class Phase2BacktestEngine:
    """
    Phase 2 真实回测引擎
    
    功能：
    1. 动态仓位回测：根据信号强度和一致性动态调整仓位
    2. 移动止损回测：支持单ETF和组合级别止损
    3. 联合回测：同时应用动态仓位和止损
    
    设计原则：
    - 保持与理论估算模型的接口一致性
    - 逐日计算，记录完整路径
    - 输出可与理论估算对比的指标
    """
    
    def __init__(self, position_optimizer, signal_optimizer=None):
        """
        参数:
            position_optimizer: PositionOptimizer 实例
            signal_optimizer: SignalStrengthOptimizer 实例（可选）
        """
        self.position_opt = position_optimizer
        self.signal_opt = signal_optimizer
        self.logger = logging.getLogger(__name__)
    
    def run_dynamic_position_backtest(
        self,
        baseline_returns: pd.Series,
        high_confidence_days_ratio: float = 0.6,
        position_levels: List[Tuple[float, float]] = None
    ) -> Dict:
        """
        运行动态仓位回测（真实逐日路径）
        
        参数:
            baseline_returns: 基线日收益率序列（满仓情况下）
            high_confidence_days_ratio: 高置信度日期占比（用于模拟信号分布）
            position_levels: 仓位映射规则
        
        返回:
            回测结果字典，包含调整后收益/Sharpe/回撤等
        
        实现说明：
        - 由于无真实因子数据，这里模拟信号分布：
          高置信度日（收益率Top X%）+ 随机噪声
        - 每日调用 apply_dynamic_position 获取实际仓位
        - 计算真实净值曲线和回测指标
        """
        if position_levels is None:
            position_levels = [(0.5, 0.5), (0.7, 0.7), (0.9, 1.0)]
        
        n_days = len(baseline_returns)
        dates = baseline_returns.index
        
        # 改进的信号分布模拟：
        # 不再简单基于收益率排名，而是随机生成信号，并控制高置信度日占比
        np.random.seed(42)  # 固定随机种子保证可重复性
        
        # 根据high_confidence_days_ratio生成信号分布
        n_high_conf = int(n_days * high_confidence_days_ratio)
        
        # 高置信度日：signal_strength 和 consistency_ratio 都较高
        high_conf_signal = np.random.uniform(0.7, 1.0, n_high_conf)
        high_conf_consistency = np.random.uniform(0.7, 1.0, n_high_conf)
        
        # 低置信度日：signal_strength 和 consistency_ratio 都较低
        low_conf_signal = np.random.uniform(0.0, 0.5, n_days - n_high_conf)
        low_conf_consistency = np.random.uniform(0.0, 0.5, n_days - n_high_conf)
        
        # 合并并随机打乱（让高低置信度日随机分布）
        signal_strength = np.concatenate([high_conf_signal, low_conf_signal])
        consistency_ratio = np.concatenate([high_conf_consistency, low_conf_consistency])
        
        # 打乱顺序
        shuffle_idx = np.random.permutation(n_days)
        signal_strength = signal_strength[shuffle_idx]
        consistency_ratio = consistency_ratio[shuffle_idx]
        
        signal_strength = pd.Series(signal_strength, index=dates)
        consistency_ratio = pd.Series(consistency_ratio, index=dates)
        
        # 逐日计算仓位和收益
        positions = []
        confidences = []
        daily_returns = []
        
        for i in range(n_days):
            result = self.position_opt.apply_dynamic_position(
                signal_strength=signal_strength.iloc[i],
                consistency_ratio=consistency_ratio.iloc[i],
                position_levels=position_levels
            )
            
            position = result['position']
            confidence = result['confidence']
            
            # 实际收益 = 基线收益 × 实际仓位
            actual_return = baseline_returns.iloc[i] * position
            
            positions.append(position)
            confidences.append(confidence)
            daily_returns.append(actual_return)
        
        # 计算回测指标
        daily_returns = pd.Series(daily_returns, index=dates)
        cumulative_returns = (1 + daily_returns).cumprod()
        
        # 计算性能指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        returns_std = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / returns_std if returns_std > 0 else 0
        
        # 最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_dd = drawdowns.min()
        
        # 平均仓位
        avg_position = np.mean(positions)
        
        # 高置信度日占比（实际）
        actual_high_conf_ratio = np.sum(np.array(confidences) >= 0.9) / n_days
        
        return {
            'type': 'real_backtest',
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'positions': pd.Series(positions, index=dates),
            'confidences': pd.Series(confidences, index=dates),
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'returns_std': returns_std,
            'avg_position': avg_position,
            'actual_high_conf_ratio': actual_high_conf_ratio,
            'n_days': n_days
        }
    
    def run_trailing_stop_backtest(
        self,
        baseline_returns: pd.Series,
        etf_stop: float = 0.05,
        portfolio_stop: float = 0.10
    ) -> Dict:
        """
        运行移动止损回测（真实逐日路径）
        
        参数:
            baseline_returns: 基线日收益率序列（无止损情况下）
            etf_stop: 单ETF止损阈值
            portfolio_stop: 组合止损阈值
        
        返回:
            回测结果字典
        
        实现说明：
        - 每日计算持仓收益（相对初始买入价）
        - 调用 apply_trailing_stop 判断是否触发
        - 若触发，次日平仓（收益率为0），冷却期后再买入
        """
        n_days = len(baseline_returns)
        dates = baseline_returns.index
        
        # 初始化状态
        is_holding = True
        buy_price = 1.0  # 初始价格
        current_price = 1.0
        cooldown_days = 0  # 冷却期计数
        
        daily_returns = []
        stop_events = []
        holding_returns = []
        
        for i in range(n_days):
            if cooldown_days > 0:
                # 冷却期，不持仓
                daily_returns.append(0.0)
                holding_returns.append(0.0)
                cooldown_days -= 1
                
                # 冷却期结束，重新买入
                if cooldown_days == 0:
                    is_holding = True
                    buy_price = current_price
                    self.logger.debug(f"Date {dates[i]}: 冷却期结束，重新买入")
            
            elif is_holding:
                # 持仓状态
                base_ret = baseline_returns.iloc[i]
                current_price = current_price * (1 + base_ret)
                holding_return = current_price / buy_price - 1
                
                daily_returns.append(base_ret)
                holding_returns.append(holding_return)
                
                # 检查止损
                stop_result = self.position_opt.apply_trailing_stop(
                    holding_return=holding_return,
                    etf_stop=etf_stop,
                    portfolio_stop=portfolio_stop
                )
                
                if stop_result['triggered']:
                    # 触发止损
                    stop_events.append({
                        'date': dates[i],
                        'holding_return': holding_return,
                        'reason': stop_result['reason']
                    })
                    
                    is_holding = False
                    cooldown_days = 5  # 5日冷却期
                    self.logger.debug(f"Date {dates[i]}: {stop_result['reason']}")
        
        # 计算回测指标
        daily_returns = pd.Series(daily_returns, index=dates)
        cumulative_returns = (1 + daily_returns).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        returns_std = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / returns_std if returns_std > 0 else 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_dd = drawdowns.min()
        
        # 止损统计
        n_stops = len(stop_events)
        stop_rate = n_stops / (n_days / 252)  # 每年止损次数
        
        return {
            'type': 'real_backtest',
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'holding_returns': pd.Series(holding_returns, index=dates),
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'returns_std': returns_std,
            'stop_events': stop_events,
            'n_stops': n_stops,
            'stop_rate': stop_rate,
            'n_days': n_days
        }
    
    def run_combined_backtest(
        self,
        baseline_returns: pd.Series,
        high_confidence_days_ratio: float = 0.6,
        etf_stop: float = 0.05,
        portfolio_stop: float = 0.10,
        position_levels: List[Tuple[float, float]] = None
    ) -> Dict:
        """
        运行联合回测（动态仓位 + 移动止损）
        
        参数:
            baseline_returns: 基线日收益率序列
            high_confidence_days_ratio: 高置信度日期占比
            etf_stop: 单ETF止损阈值
            portfolio_stop: 组合止损阈值
            position_levels: 仓位映射规则
        
        返回:
            回测结果字典
        """
        if position_levels is None:
            position_levels = [(0.5, 0.5), (0.7, 0.7), (0.9, 1.0)]
        
        n_days = len(baseline_returns)
        dates = baseline_returns.index
        
        # 改进的信号分布模拟（与动态仓位回测保持一致）
        np.random.seed(42)
        n_high_conf = int(n_days * high_confidence_days_ratio)
        
        high_conf_signal = np.random.uniform(0.7, 1.0, n_high_conf)
        high_conf_consistency = np.random.uniform(0.7, 1.0, n_high_conf)
        low_conf_signal = np.random.uniform(0.0, 0.5, n_days - n_high_conf)
        low_conf_consistency = np.random.uniform(0.0, 0.5, n_days - n_high_conf)
        
        signal_strength = np.concatenate([high_conf_signal, low_conf_signal])
        consistency_ratio = np.concatenate([high_conf_consistency, low_conf_consistency])
        
        shuffle_idx = np.random.permutation(n_days)
        signal_strength = pd.Series(signal_strength[shuffle_idx], index=dates)
        consistency_ratio = pd.Series(consistency_ratio[shuffle_idx], index=dates)
        
        # 初始化状态
        is_holding = True
        buy_price = 1.0
        current_price = 1.0
        cooldown_days = 0
        
        daily_returns = []
        positions = []
        stop_events = []
        
        for i in range(n_days):
            if cooldown_days > 0:
                # 冷却期
                daily_returns.append(0.0)
                positions.append(0.0)
                cooldown_days -= 1
                
                if cooldown_days == 0:
                    is_holding = True
                    buy_price = current_price
            
            elif is_holding:
                # 计算动态仓位
                pos_result = self.position_opt.apply_dynamic_position(
                    signal_strength=signal_strength.iloc[i],
                    consistency_ratio=consistency_ratio.iloc[i],
                    position_levels=position_levels
                )
                position = pos_result['position']
                
                # 实际收益 = 基线收益 × 仓位
                base_ret = baseline_returns.iloc[i]
                actual_ret = base_ret * position
                current_price = current_price * (1 + actual_ret)
                holding_return = current_price / buy_price - 1
                
                daily_returns.append(actual_ret)
                positions.append(position)
                
                # 检查止损
                stop_result = self.position_opt.apply_trailing_stop(
                    holding_return=holding_return,
                    etf_stop=etf_stop,
                    portfolio_stop=portfolio_stop
                )
                
                if stop_result['triggered']:
                    stop_events.append({
                        'date': dates[i],
                        'holding_return': holding_return,
                        'reason': stop_result['reason']
                    })
                    is_holding = False
                    cooldown_days = 5
        
        # 计算回测指标
        daily_returns = pd.Series(daily_returns, index=dates)
        cumulative_returns = (1 + daily_returns).cumprod()
        
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        returns_std = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / returns_std if returns_std > 0 else 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_dd = drawdowns.min()
        
        return {
            'type': 'real_backtest',
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'positions': pd.Series(positions, index=dates),
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'returns_std': returns_std,
            'avg_position': np.mean([p for p in positions if p > 0]),
            'stop_events': stop_events,
            'n_stops': len(stop_events),
            'n_days': n_days
        }
    
    @staticmethod
    def generate_baseline_returns(
        annual_return: float,
        sharpe: float,
        n_days: int = 756,  # 3年交易日
        seed: int = None
    ) -> pd.Series:
        """
        生成基线日收益率序列（用于模拟回测）
        
        参数:
            annual_return: 年化收益率
            sharpe: Sharpe比率
            n_days: 交易日数
            seed: 随机种子
        
        返回:
            日收益率序列
        
        实现：
        - 根据年化收益和Sharpe反推日收益均值和标准差
        - 生成正态分布的日收益序列
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 日化参数
        daily_return_mean = annual_return / 252
        annual_std = annual_return / sharpe if sharpe > 0 else 0.15
        daily_std = annual_std / np.sqrt(252)
        
        # 生成日收益
        daily_returns = np.random.normal(daily_return_mean, daily_std, n_days)
        
        # 生成日期索引
        start_date = pd.Timestamp('2021-01-01')
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        
        return pd.Series(daily_returns, index=dates)
