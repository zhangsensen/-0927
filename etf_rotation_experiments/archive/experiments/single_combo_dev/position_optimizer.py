#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓位优化器
用于实现动态仓位映射和止损机制
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PositionOptimizer:
    """仓位优化器，支持动态仓位和止损机制"""
    
    def __init__(self, combo_config: Dict):
        """
        初始化
        
        Args:
            combo_config: 组合配置字典，包含因子列表
        """
        self.combo = combo_config.get('combo', '')
        self.factors = [f.strip() for f in self.combo.split('+')]
        self.n_factors = len(self.factors)
    
    def apply_dynamic_position(
        self, 
        signal_strength: float = 0.8,
        consistency_ratio: float = 1.0,
        position_levels: List[Tuple[float, float]] = None
    ) -> Dict:
        """
        应用动态仓位映射（可用于逐日回测）
        
        Args:
            signal_strength: 信号强度（0-1），由趋势强度/RSI等因子归一化得到
            consistency_ratio: 方向一致性比例（0-1），多因子方向一致的占比
            position_levels: 仓位映射规则，默认为 [(0.5, 0.5), (0.7, 0.7), (0.9, 1.0)]
                           格式：(置信度阈值, 对应仓位)，阈值升序排列
        
        Returns:
            字典，包含置信度分数和建议仓位
        
        金融直觉：
        - 综合置信度 = (信号强度 + 方向一致性) / 2
        - 置信度越高，仓位越高；最低保持30%底仓避免完全空仓
        - 可直接用于逐日回测：每日计算 signal_strength 和 consistency_ratio，
          然后调用此函数获取建议仓位
        
        回测接入建议：
        - 在调仓日，遍历候选 ETF 池，为每个 ETF 计算信号强度和一致性
        - 调用 apply_dynamic_position 获取建议仓位
        - 根据仓位权重分配资金（归一化后使总仓位 <= 1.0）
        """
        if position_levels is None:
            position_levels = [
                (0.5, 0.5),   # 低置信度 → 50%仓位
                (0.7, 0.7),   # 中置信度 → 70%仓位
                (0.9, 1.0),   # 高置信度 → 满仓
            ]
        
        # 计算综合置信度
        confidence = (signal_strength + consistency_ratio) / 2
        
        # 映射到仓位
        position = 0.3  # 最低仓位
        for threshold, pos in position_levels:
            if confidence >= threshold:
                position = pos
        
        return {
            'confidence': confidence,
            'signal_strength': signal_strength,
            'consistency_ratio': consistency_ratio,
            'position': position
        }
    
    def apply_trailing_stop(
        self,
        holding_return: float,
        etf_stop: float = 0.05,
        portfolio_stop: float = 0.10
    ) -> Dict:
        """
        应用移动止损机制（可用于逐日回测）
        
        Args:
            holding_return: 当前持仓收益率（浮动盈亏，相对买入价）
            etf_stop: 单ETF止损阈值（默认5%），适用于个股级别
            portfolio_stop: 组合级别止损阈值（默认10%），适用于整体组合
        
        Returns:
            字典，包含是否触发止损和止损原因
        
        金融直觉：
        - 单ETF止损（3%-7%）：防止个别标的极端下跌拖累组合
        - 组合止损（8%-12%）：防止系统性风险导致整体大幅回撤
        - 两层止损机制：先触发个股止损（快速止损），后触发组合止损（保底）
        
        回测接入建议：
        - 每日盘后计算每个持仓 ETF 的浮动盈亏 holding_return
        - 调用 apply_trailing_stop 判断是否触发止损
        - 若触发，次日开盘平仓该 ETF（或整个组合）
        - 建议实现：在 backtest 主循环中加入止损检查逻辑
        """
        etf_triggered = holding_return <= -etf_stop
        portfolio_triggered = holding_return <= -portfolio_stop
        
        triggered = etf_triggered or portfolio_triggered
        reason = None
        if etf_triggered:
            reason = f'单ETF止损触发 (收益{holding_return:.1%} < -{etf_stop:.0%})'
        elif portfolio_triggered:
            reason = f'组合止损触发 (收益{holding_return:.1%} < -{portfolio_stop:.0%})'
        
        return {
            'triggered': triggered,
            'etf_stop_triggered': etf_triggered,
            'portfolio_stop_triggered': portfolio_triggered,
            'reason': reason,
            'holding_return': holding_return
        }
    
    def estimate_dynamic_position_impact(
        self,
        baseline_sharpe: float,
        baseline_return: float,
        baseline_dd: float,
        high_confidence_days_ratio: float = 0.6
    ) -> Dict:
        """
        估算动态仓位的影响（理论模型，增强版）
        
        Args:
            baseline_sharpe: 基线Sharpe（满仓回测结果）
            baseline_return: 基线年化收益（满仓回测结果）
            baseline_dd: 基线最大回撤（满仓回测结果）
            high_confidence_days_ratio: 高置信度日期占比（0-1），满仓日占比
        
        Returns:
            估算结果字典，包含调整后收益/Sharpe/回撤等
        
        金融直觉与假设说明：
        1. **信号分布假设**（可根据历史数据校准）：
           - 高置信度日（满仓）：占比 high_confidence_days_ratio
           - 中置信度日（70%仓）：占比 (1 - high_conf) * 0.75
           - 低置信度日（50%仓）：占比 (1 - high_conf) * 0.25
        
        2. **收益质量差异假设**（关键创新点）：
           - 高置信度日收益率 +10%（信号更可靠，捕捉强趋势）
           - 中置信度日收益率 ±0%（中性）
           - 低置信度日收益率 -10%（信号弱，容易被噪声干扰）
           - 总收益 = 加权收益 × 平均仓位
        
        3. **回撤改善假设**（经验值，可调参）：
           - 低置信度日贡献 40% 的回撤（胜率低，易踩坑）
           - 降低仓位可避免这部分回撤：dd_reduction = 0.4 * (1 - avg_pos) * |dd|
        
        4. **波动率假设**（金融工程常用）：
           - 仓位降低 → 波动率按 sqrt(仓位) 缩放
           - Sharpe 提升 = baseline_sharpe / sqrt(avg_position)
        
        5. **理论 vs 实际**：
           - 本函数为参数敏感性分析，不依赖逐日路径
           - 迁移到回测时，需在每日 loop 中调用 apply_dynamic_position
             获取实际仓位，再计算真实盈亏曲线
        """
        # 假设信号分布：高置信度X%，中置信度Y%，低置信度Z%
        mid_confidence_ratio = (1 - high_confidence_days_ratio) * 0.75
        low_confidence_ratio = (1 - high_confidence_days_ratio) * 0.25
        # 平均仓位 = 高*1.0 + 中*0.7 + 低*0.5
        avg_position = (
            high_confidence_days_ratio * 1.0 +
            mid_confidence_ratio * 0.7 +
            low_confidence_ratio * 0.5
        )
        # 收益加权假设：高置信度日收益略优(+10%)，低置信度略差(-10%)，中置信度不变
        weighted_return = (
            baseline_return * (
                high_confidence_days_ratio * 1.10 +
                mid_confidence_ratio * 1.00 +
                low_confidence_ratio * 0.90
            )
        )
        # 总收益按仓位缩放
        adjusted_return = weighted_return * avg_position
        # 回撤改善效果 = 避免低置信度日期的满仓亏损
        # 假设低置信度日期的胜率更低，贡献了40%的回撤
        dd_reduction = 0.40 * (1 - avg_position) * abs(baseline_dd)
        adjusted_dd = baseline_dd + dd_reduction  # 回撤变小（负数变大）
        # Sharpe改善 = 波动率降低效果（仓位降低 → 波动降低）
        vol_reduction_factor = avg_position ** 0.5
        sharpe_boost = 1 / vol_reduction_factor
        adjusted_sharpe = baseline_sharpe * sharpe_boost
        return {
            'avg_position': avg_position,
            'adjusted_return': adjusted_return,
            'adjusted_sharpe': adjusted_sharpe,
            'adjusted_dd': adjusted_dd,
            'dd_reduction': dd_reduction,
            'return_loss': baseline_return - adjusted_return,
            'sharpe_boost_pct': (adjusted_sharpe / baseline_sharpe - 1) * 100
        }
    
    def estimate_trailing_stop_impact(
        self,
        baseline_sharpe: float,
        baseline_return: float,
        baseline_dd: float,
        etf_stop: float = 0.05,
        portfolio_stop: float = 0.10
    ) -> Dict:
        """
        估算移动止损的影响（理论模型，参数敏感）
        
        Args:
            baseline_sharpe: 基线Sharpe（无止损回测结果）
            baseline_return: 基线年化收益（无止损回测结果）
            baseline_dd: 基线最大回撤（无止损回测结果）
            etf_stop: 单ETF止损阈值（0.03-0.07，越小越紧）
            portfolio_stop: 组合止损阈值（0.08-0.12，越小越紧）
        
        Returns:
            估算结果字典，包含调整后收益/Sharpe/回撤/紧度等
        
        金融直觉与假设说明：
        1. **止损紧度定义**（核心创新，让参数真正"动起来"）：
           - tightness = 0.5 * (ref_etf/etf_stop + ref_portfolio/portfolio_stop)
           - 参考值：ref_etf=5%, ref_portfolio=10%（行业经验值）
           - tightness ∈ [0, 1]，越大越紧
           - 例：(3%, 8%) → tightness≈0.92（非常紧）
                (5%, 10%) → tightness=1.0（参考值）
                (7%, 12%) → tightness≈0.79（宽松）
        
        2. **回撤改善模型**（线性映射，可调参）：
           - dd_protection_ratio = 0.3 + 0.4 * tightness（30%-70%区间）
           - 直觉：止损越紧，越早止损，避免的极端回撤越多
           - 但不是100%，因为：
             (a) 市场有缺口风险，止损可能滑点
             (b) 部分回撤是系统性的，止损也无法避免
        
        3. **收益损失模型**（线性映射，经验值）：
           - return_cost_pct = 5 + 10 * tightness（5%-15%区间）
           - 直觉：止损越紧，误杀越多（错失反弹、频繁止损再买入）
           - 交易成本和心理成本也会增加
        
        4. **Sharpe 提升上限**（防止极端值）：
           - vol_reduction = dd_improvement / |baseline_dd|
           - sharpe_boost = 1 + min(max(vol_reduction*0.3 - return_cost*0.5, 0), 0.3)
           - 最多提升 30%，最少 0%（避免 Sharpe 翻倍这种不合理结果）
        
        5. **理论 vs 实际**：
           - 本函数为敏感性分析，用于快速评估不同止损参数的大致影响
           - 迁移到回测时，需在每日 loop 中：
             (a) 计算每个持仓的 holding_return
             (b) 调用 apply_trailing_stop 判断是否止损
             (c) 若触发，平仓并记录止损事件
             (d) 统计实际回撤、收益、止损次数等
        
        6. **参数调优建议**：
           - 从宽松到紧：(7%, 12%) → (5%, 10%) → (3%, 8%)
           - 观察 score = dd_improvement - return_cost*0.01，选择最优平衡点
           - 可结合历史回撤分布，选择能覆盖 80-90% 回撤的阈值
        """
        # 参考值（行业常用/经验值）
        ref_etf = 0.05  # 5%
        ref_portfolio = 0.10  # 10%
        # 计算“紧度”tightness，越大越紧，范围[0,1]
        tightness = 0.5 * (ref_etf / max(etf_stop, 1e-4) + ref_portfolio / max(portfolio_stop, 1e-4))
        tightness = max(0.0, min(tightness, 1.0))
        # 回撤保护比例，30%-70%区间，紧→大
        dd_protection_ratio = 0.3 + 0.4 * tightness
        dd_improvement = abs(baseline_dd) * dd_protection_ratio
        # 收益损失，5%-15%区间，紧→大
        return_cost_pct = 5 + 10 * tightness
        return_cost = return_cost_pct / 100
        adjusted_return = baseline_return * (1 - return_cost)
        # Sharpe提升，假设波动率与回撤成正比，提升幅度受限于回撤改善和收益损失
        vol_reduction = dd_improvement / abs(baseline_dd) if abs(baseline_dd) > 1e-6 else 0.0
        # Sharpe提升因子，最多提升30%，最少提升0%
        sharpe_boost_factor = 1 + min(max(vol_reduction * 0.3 - return_cost * 0.5, 0), 0.3)
        adjusted_sharpe = baseline_sharpe * sharpe_boost_factor
        # 调整后回撤
        adjusted_dd = baseline_dd + dd_improvement
        return {
            'etf_stop': etf_stop,
            'portfolio_stop': portfolio_stop,
            'adjusted_return': adjusted_return,
            'adjusted_sharpe': adjusted_sharpe,
            'adjusted_dd': adjusted_dd,
            'dd_improvement': dd_improvement,
            'return_cost_pct': return_cost_pct,
            'sharpe_boost_pct': (adjusted_sharpe / baseline_sharpe - 1) * 100,
            'tightness': tightness,
            'dd_protection_ratio': dd_protection_ratio
        }
