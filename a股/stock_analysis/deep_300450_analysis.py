#!/usr/bin/env python3
"""
300450深度技术分析 - 详解止损止盈设置的技术原理
深入分析支撑阻力位、波动性、市场微观结构等技术因素
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
sys.path.append('/Users/zhangshenshen/深度量化0927')

from factor_system.factor_generation.enhanced_factor_calculator import EnhancedFactorCalculator
from factor_system.factor_generation.enhanced_factor_calculator import TimeFrame


class Deep300450Analyzer:
    """300450深度技术分析器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data_dir = "/Users/zhangshenshen/深度量化0927/a股"

        print(f"🔍 300450深度技术分析器初始化")
        print(f"   股票代码: {stock_code}")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并准备数据，计算技术指标"""
        print("正在加载300450数据并计算技术指标...")

        # 读取日线数据
        data_file = f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1d_2025-10-09.csv"

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        # 读取并处理数据
        df = pd.read_csv(data_file, skiprows=1)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 转换数值列
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 清理数据
        df = df.dropna()

        # 转换为小写列名
        df.columns = [col.lower() for col in df.columns]

        # 计算技术指标
        calculator = EnhancedFactorCalculator()
        factors_df = calculator.calculate_comprehensive_factors(df, TimeFrame.DAILY)

        print(f"数据加载完成: {len(df)}条记录")
        print(f"技术指标计算完成: {len(factors_df.columns)}个指标")

        return df, factors_df

    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """计算成交量分布图，寻找关键支撑阻力位"""
        print("正在计算成交量分布图...")

        current_price = df['close'].iloc[-1]
        high_60 = df['high'].tail(60).max()
        low_60 = df['low'].tail(60).min()

        # 创建价格区间
        price_levels = np.linspace(low_60, high_60, 50)  # 50个价格区间

        volume_profile = []
        for price in price_levels:
            # 找到该价格附近的成交量
            nearby_volume = df[(df['close'] >= price - 0.5) & (df['close'] <= price + 0.5)]['volume'].sum()
            volume_profile.append({'price': price, 'volume': nearby_volume})

        # 找到成交量密集区
        sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)

        # 支撑位：在当前价格下方的高成交量区
        support_levels = []
        for level in sorted_profile[:10]:
            if level['price'] < current_price and level['volume'] > 0:
                support_levels.append(level['price'])

        # 阻力位：在当前价格上方的高成交量区
        resistance_levels = []
        for level in sorted_profile[:10]:
            if level['price'] > current_price and level['volume'] > 0:
                resistance_levels.append(level['price'])

        return {
            'volume_profile': volume_profile,
            'support_levels': support_levels[:5],  # 前5个支撑位
            'resistance_levels': resistance_levels[:5],  # 前5个阻力位
            'high_volume_areas': sorted_profile[:5]  # 前5个高成交量区
        }

    def analyze_market_structure(self, df: pd.DataFrame, factors_df: pd.DataFrame) -> Dict:
        """分析市场微观结构"""
        print("正在分析市场微观结构...")

        latest = df.iloc[-1]
        current_price = latest['close']

        # 1. 计算关键价格水平
        # 近期高点
        recent_highs = []
        for i in range(10, len(df)-10):
            if df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max():
                recent_highs.append(df['high'].iloc[i])

        # 近期低点
        recent_lows = []
        for i in range(10, len(df)-10):
            if df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min():
                recent_lows.append(df['low'].iloc[i])

        # 2. 趋势线分析
        # 上升趋势线
        uptrend_line_price = self._calculate_uptrend_line(df)
        # 下降趋势线
        downtrend_line_price = self._calculate_downtrend_line(df)

        # 3. 移动平均线分析
        ma_slopes = {}
        for period in [20, 30, 40, 60]:
            ma_col = f'MA{period}'
            if ma_col in factors_df.columns:
                ma_series = factors_df[ma_col].dropna()
                if len(ma_series) >= 5:
                    # 计算MA斜率
                    slope = (ma_series.iloc[-1] - ma_series.iloc[-5]) / 5
                    ma_slopes[ma_col] = slope

        # 4. 波动性分析
        atr_values = factors_df[[col for col in factors_df.columns if 'ATR' in col]].iloc[-1]
        avg_atr = atr_values.mean()

        # 5. 价格通道
        bb_upper_cols = [col for col in factors_df.columns if 'BB_20_2.0_Upper' in col]
        bb_lower_cols = [col for col in factors_df.columns if 'BB_20_2.0_Lower' in col]

        bb_upper = factors_df[bb_upper_cols].iloc[-1].iloc[0] if bb_upper_cols else current_price * 1.1
        bb_lower = factors_df[bb_lower_cols].iloc[-1].iloc[0] if bb_lower_cols else current_price * 0.9

        return {
            'current_price': current_price,
            'recent_highs': sorted(recent_highs, reverse=True)[:5],
            'recent_lows': sorted(recent_lows)[:5],
            'uptrend_line': uptrend_line_price,
            'downtrend_line': downtrend_line_price,
            'ma_slopes': ma_slopes,
            'avg_atr': avg_atr,
            'bollinger_bands': {'upper': bb_upper, 'lower': bb_lower}
        }

    def _calculate_uptrend_line(self, df: pd.DataFrame) -> Optional[float]:
        """计算上升趋势线价格"""
        # 找到最近3个月的重要低点
        lows = []
        for i in range(10, len(df)-10):
            if df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min():
                lows.append((df.index[i], df['low'].iloc[i]))

        if len(lows) >= 2:
            # 选择最近的两个低点
            point1 = lows[-2]
            point2 = lows[-1]

            # 计算趋势线
            x1, y1 = point1[0].toordinal(), point1[1]
            x2, y2 = point2[0].toordinal(), point2[1]

            # 计算当前日期对应的趋势线价格
            current_x = df.index[-1].toordinal()
            slope = (y2 - y1) / (x2 - x1)
            trend_price = y1 + slope * (current_x - x1)

            return trend_price
        return None

    def _calculate_downtrend_line(self, df: pd.DataFrame) -> Optional[float]:
        """计算下降趋势线价格"""
        # 找到最近3个月的重要高点
        highs = []
        for i in range(10, len(df)-10):
            if df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max():
                highs.append((df.index[i], df['high'].iloc[i]))

        if len(highs) >= 2:
            # 选择最近的两个高点
            point1 = highs[-2]
            point2 = highs[-1]

            # 计算趋势线
            x1, y1 = point1[0].toordinal(), point1[1]
            x2, y2 = point2[0].toordinal(), point2[1]

            # 计算当前日期对应的趋势线价格
            current_x = df.index[-1].toordinal()
            slope = (y2 - y1) / (x2 - x1)
            trend_price = y1 + slope * (current_x - x1)

            return trend_price
        return None

    def analyze_risk_reward_scenarios(self, df: pd.DataFrame, factors_df: pd.DataFrame, market_structure: Dict, volume_profile: Dict) -> Dict:
        """分析风险回报场景"""
        print("正在分析风险回报场景...")

        current_price = market_structure['current_price']
        avg_atr = market_structure['avg_atr']

        scenarios = {
            'bullish': {},
            'bearish': {},
            'neutral': {},
            'risk_levels': {}
        }

        # 1. 牛市场景
        # 找到最近的阻力位
        nearest_resistance = None
        if market_structure['recent_highs']:
            nearest_resistance = min([h for h in market_structure['recent_highs'] if h > current_price], default=current_price*1.1)

        if volume_profile['resistance_levels']:
            nearest_resistance = min(volume_profile['resistance_levels'])

        scenarios['bullish'] = {
            'target_price': nearest_resistance if nearest_resistance else current_price * 1.1,
            'potential_return': (nearest_resistance / current_price - 1) * 100 if nearest_resistance else 10,
            'probability': 'high' if self._is_uptrend_confirmed(market_structure) else 'medium'
        }

        # 2. 熊市场景
        # 找到最近的支撑位
        nearest_support = None
        if market_structure['recent_lows']:
            nearest_support = max([l for l in market_structure['recent_lows'] if l < current_price], default=current_price*0.9)

        if volume_profile['support_levels']:
            nearest_support = max(volume_profile['support_levels'])

        scenarios['bearish'] = {
            'target_price': nearest_support if nearest_support else current_price * 0.9,
            'potential_return': (nearest_support / current_price - 1) * 100 if nearest_support else -10,
            'probability': 'high' if self._is_downtrend_confirmed(market_structure) else 'medium'
        }

        # 3. 震荡场景
        scenarios['neutral'] = {
            'target_range': f"{nearest_support:.2f} - {nearest_resistance:.2f}" if nearest_support and nearest_resistance else f"{current_price*0.95:.2f} - {current_price*1.05:.2f}",
            'expected_return': 0,
            'probability': 'medium'
        }

        # 4. 风险等级计算
        # 波动性风险
        volatility_risk = 'low'
        if avg_atr / current_price > 0.05:  # 5%以上
            volatility_risk = 'medium'
        if avg_atr / current_price > 0.08:  # 8%以上
            volatility_risk = 'high'

        # 趋势风险
        trend_risk = 'low'
        if not self._is_uptrend_confirmed(market_structure):
            trend_risk = 'medium'

        # 成交量风险
        volume_risk = 'low'
        recent_volume = df['volume'].tail(5).mean()
        avg_volume = df['volume'].tail(20).mean()
        if recent_volume < avg_volume * 0.7:
            volume_risk = 'medium'

        scenarios['risk_levels'] = {
            'volatility_risk': volatility_risk,
            'trend_risk': trend_risk,
            'volume_risk': volume_risk,
            'overall_risk': self._calculate_overall_risk(volatility_risk, trend_risk, volume_risk)
        }

        return scenarios

    def _is_uptrend_confirmed(self, market_structure: Dict) -> bool:
        """判断是否确认上升趋势"""
        # MA20在MA60上方
        ma20 = market_structure['ma_slopes'].get('MA20', 0)
        ma60 = market_structure['ma_slopes'].get('MA60', 0)

        # 价格在上升趋势线上方
        uptrend_line = market_structure['uptrend_line']
        current_price = market_structure['current_price']

        uptrend_confirmed = False
        if uptrend_line and current_price > uptrend_line:
            uptrend_confirmed = True

        # 均线多头排列
        ma_aligned = ma20 > 0 and ma60 > 0

        return uptrend_confirmed and ma_aligned

    def _is_downtrend_confirmed(self, market_structure: Dict) -> bool:
        """判断是否确认下降趋势"""
        # MA20在MA60下方
        ma20 = market_structure['ma_slopes'].get('MA20', 0)
        ma60 = market_structure['ma_slopes'].get('MA60', 0)

        # 价格在下降趋势线下方
        downtrend_line = market_structure['downtrend_line']
        current_price = market_structure['current_price']

        downtrend_confirmed = False
        if downtrend_line and current_price < downtrend_line:
            downtrend_confirmed = True

        # 均线空头排列
        ma_aligned = ma20 < 0 and ma60 < 0

        return downtrend_confirmed and ma_aligned

    def _calculate_overall_risk(self, volatility_risk: str, trend_risk: str, volume_risk: str) -> str:
        """计算总体风险等级"""
        risk_scores = {
            'low': 1,
            'medium': 2,
            'high': 3
        }

        total_score = (risk_scores[volatility_risk] +
                      risk_scores[trend_risk] +
                      risk_scores[volume_risk])

        if total_score <= 3:
            return 'low'
        elif total_score <= 6:
            return 'medium'
        else:
            return 'high'

    def explain_stop_loss_logic(self, df: pd.DataFrame, factors_df: pd.DataFrame, market_structure: Dict, volume_profile: Dict) -> Dict:
        """解释止损逻辑和背后的技术原理"""
        print("正在分析止损逻辑...")

        current_price = market_structure['current_price']
        avg_atr = market_structure['avg_atr']

        stop_loss_explanation = {
            'recommended_stop': 0,
            'reasoning': [],
            'technical_factors': [],
            'risk_considerations': [],
            'alternative_stops': []
        }

        # 1. 基于市场结构的止损
        structure_based_stop = None

        # 寻找关键支撑位
        potential_stops = []

        # a) 近期重要低点
        if market_structure['recent_lows']:
            important_lows = [low for low in market_structure['recent_lows'] if low < current_price]
            if important_lows:
                nearest_low = max(important_lows)
                potential_stops.append({
                    'price': nearest_low,
                    'type': '近期低点',
                    'strength': 'high',
                    'distance_pct': (nearest_low / current_price - 1) * 100
                })

        # b) 上升趋势线
        if market_structure['uptrend_line'] and market_structure['uptrend_line'] < current_price:
            potential_stops.append({
                'price': market_structure['uptrend_line'],
                'type': '上升趋势线',
                'strength': 'medium',
                'distance_pct': (market_structure['uptrend_line'] / current_price - 1) * 100
            })

        # c) 移动平均线支撑
        ma_stops = []
        for period in [20, 30, 40]:
            ma_col = f'MA{period}'
            if ma_col in factors_df.columns:
                ma_value = factors_df[ma_col].iloc[-1]
                if ma_value < current_price and ma_value > 0:
                    ma_stops.append({
                        'price': ma_value,
                        'type': f'MA{period}支撑',
                        'strength': 'medium',
                        'distance_pct': (ma_value / current_price - 1) * 100
                    })

        # d) 成交量密集区支撑
        if volume_profile['support_levels']:
            volume_support = max(volume_profile['support_levels'])
            potential_stops.append({
                'price': volume_support,
                'type': '成交量密集区',
                'strength': 'high',
                'distance_pct': (volume_support / current_price - 1) * 100
            })

        # e) 布林带下轨
        if 'BB_Lower_20_2' in factors_df.columns:
            bb_lower = factors_df['BB_Lower_20_2'].iloc[-1]
            if bb_lower < current_price:
                potential_stops.append({
                    'price': bb_lower,
                    'type': '布林带下轨',
                    'strength': 'medium',
                    'distance_pct': (bb_lower / current_price - 1) * 100
                })

        # 按强度和距离排序
        potential_stops.sort(key=lambda x: (x['strength'], -abs(x['distance_pct'])), reverse=True)

        if potential_stops:
            selected_stop = potential_stops[0]
            stop_loss_explanation['recommended_stop'] = selected_stop['price']

            stop_loss_explanation['reasoning'].append(f"选择{selected_stop['type']}作为止损位")
            stop_loss_explanation['reasoning'].append(f"该位置强度：{selected_stop['strength']}")
            stop_loss_explanation['reasoning'].append(f"距离当前价格：{selected_stop['distance_pct']:+.2f}%")

            # 技术因素
            if selected_stop['type'] == '近期低点':
                stop_loss_explanation['technical_factors'].append("近期低点代表市场参与者认可的心理支撑位")
                stop_loss_explanation['technical_factors'].append("跌破此位可能引发更多止损和抛售")

            elif selected_stop['type'] == '上升趋势线':
                stop_loss_explanation['technical_factors'].append("上升趋势线是多头趋势的重要支撑")
                stop_loss_explanation['technical_factors'].append("跌破趋势线意味着趋势可能反转")

            elif selected_stop['type'] == '成交量密集区':
                stop_loss_explanation['technical_factors'].append("成交量密集区有大量成交，代表重要的换手区域")
                stop_loss_explanation['technical_factors'].append("此区域通常有较强支撑作用")

            # 风险考量
            volatility_ratio = avg_atr / current_price
            if volatility_ratio > 0.08:  # 8%以上波动
                stop_loss_explanation['risk_considerations'].append(f"高波动股票（ATR比率{volatility_ratio*100:.1f}%）需要较大止损空间")
                stop_loss_explanation['risk_considerations'].append("小止损容易被市场噪音触发")

            # ATR倍数建议
            atr_multiples = [2.0, 2.5, 3.0]
            for multiple in atr_multiples:
                atr_stop = current_price - multiple * avg_atr
                if atr_stop > 0:
                    stop_loss_explanation['alternative_stops'].append({
                        'price': atr_stop,
                        'type': f'{multiple}x ATR',
                        'distance_pct': (atr_stop / current_price - 1) * 100
                    })

        return stop_loss_explanation

    def explain_take_profit_logic(self, df: pd.DataFrame, factors_df: pd.DataFrame, market_structure: Dict, volume_profile: Dict) -> Dict:
        """解释止盈逻辑和背后的技术原理"""
        print("正在分析止盈逻辑...")

        current_price = market_structure['current_price']

        take_profit_explanation = {
            'targets': [],
            'reasoning': [],
            'technical_factors': [],
            'probability_analysis': {}
        }

        # 寻找潜在目标位
        potential_targets = []

        # 1. 近期高点
        if market_structure['recent_highs']:
            recent_high_targets = [high for high in market_structure['recent_highs'] if high > current_price]
            if recent_high_targets:
                nearest_high = min(recent_high_targets)
                potential_targets.append({
                    'price': nearest_high,
                    'type': '近期高点',
                    'strength': 'high',
                    'profit_pct': (nearest_high / current_price - 1) * 100,
                    'priority': 1
                })

        # 2. 成交量密集区阻力
        if volume_profile['resistance_levels']:
            for i, resistance in enumerate(volume_profile['resistance_levels'][:3]):
                potential_targets.append({
                    'price': resistance,
                    'type': '成交量密集区阻力',
                    'strength': 'high',
                    'profit_pct': (resistance / current_price - 1) * 100,
                    'priority': i + 2
                })

        # 3. 技术指标目标
        # RSI目标区
        rsi_cols = [col for col in factors_df.columns if 'RSI' in col]
        if rsi_cols:
            current_rsi = factors_df['RSI14'].iloc[-1] if 'RSI14' in factors_df.columns else 50

            # RSI超买目标（通常在70-80）
            if current_rsi < 70:
                # 估算RSI达到70的价格（简化计算）
                rsi_70_target = current_price * 1.08  # 简化估算
                potential_targets.append({
                    'price': rsi_70_target,
                    'type': 'RSI超买区',
                    'strength': 'medium',
                    'profit_pct': 8.0,
                    'priority': 4
                })

        # 4. 斐波那契扩展目标
        recent_swing_low = min(market_structure['recent_lows'][:3]) if market_structure['recent_lows'] else current_price * 0.9
        recent_swing_high = max(market_structure['recent_highs'][:3]) if market_structure['recent_highs'] else current_price * 1.1

        swing_range = recent_swing_high - recent_swing_low
        fib_levels = [1.272, 1.618, 2.0, 2.618]  # 斐波那契扩展

        for level in fib_levels:
            fib_target = current_price + level * swing_range
            potential_targets.append({
                'price': fib_target,
                'type': f'斐波那契{level}',
                'strength': 'medium',
                'profit_pct': (fib_target / current_price - 1) * 100,
                'priority': 5
            })

        # 5. 布林带上轨目标
        if 'BB_Upper_20_2' in factors_df.columns:
            bb_upper = factors_df['BB_Upper_20_2'].iloc[-1]
            if bb_upper > current_price:
                potential_targets.append({
                    'price': bb_upper,
                    'type': '布林带上轨',
                    'strength': 'medium',
                    'profit_pct': (bb_upper / current_price - 1) * 100,
                    'priority': 6
                })

        # 排序目标位
        potential_targets.sort(key=lambda x: x['profit_pct'])

        # 选择前4个目标位
        selected_targets = potential_targets[:4]

        for i, target in enumerate(selected_targets):
            take_profit_explanation['targets'].append({
                'target': i + 1,
                'price': target['price'],
                'profit_pct': target['profit_pct'],
                'type': target['type'],
                'strength': target['strength'],
                'position': '1/4'
            })

        # 解释选择逻辑
        take_profit_explanation['reasoning'].append("分批止盈策略基于以下原则：")
        take_profit_explanation['reasoning'].append("1. 第一目标：最近的关键阻力位，获利概率最高")
        take_profit_explanation['reasoning'].append("2. 第二目标：成交量密集区，可能遇到较强阻力")
        take_profit_explanation['reasoning'].append("3. 第三目标：技术指标目标区（如RSI超买）")
        take_profit_explanation['reasoning'].append("4. 第四目标：斐波那契扩展或布林带上轨，理论目标位")

        # 技术因素
        take_profit_explanation['technical_factors'].append("分批止盈可以：")
        take_profit_explanation['technical_factors'].append("• 锁定部分利润，降低风险")
        take_profit_explanation['technical_factors'].append("• 防止趋势反转带来的利润回吐")
        take_profit_explanation['technical_factors'].append("• 保留部分仓位，捕捉更大涨幅")
        take_profit_explanation['technical_factors'].append("• 根据市场反馈调整后续策略")

        # 概率分析
        take_profit_explanation['probability_analysis'] = {
            'target1_probability': 'high',  # 最近阻力位
            'target2_probability': 'medium-high',  # 成交量密集区
            'target3_probability': 'medium',  # 技术指标目标
            'target4_probability': 'low-medium'  # 理论目标
        }

        return take_profit_explanation

    def generate_deep_analysis_report(self, df: pd.DataFrame, factors_df: pd.DataFrame) -> str:
        """生成深度分析报告"""
        print("正在生成300450深度分析报告...")

        # 计算各项分析
        volume_profile = self.calculate_volume_profile(df)
        market_structure = self.analyze_market_structure(df, factors_df)
        risk_scenarios = self.analyze_risk_reward_scenarios(df, factors_df, market_structure, volume_profile)
        stop_loss_explanation = self.explain_stop_loss_logic(df, factors_df, market_structure, volume_profile)
        take_profit_explanation = self.explain_take_profit_logic(df, factors_df, market_structure, volume_profile)

        current_price = market_structure['current_price']

        # 计算基础指标
        change_1d = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        change_5d = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        change_20d = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    300450 深度技术分析报告                                 ║
║                  详解止损止盈设置的技术原理与市场逻辑                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
💰 当前价格: ¥{current_price:.2f}
📊 分析深度: 市场微观结构 + 成交量分布 + 风险回报场景

📈 价格表现:
────────────────────────────────────────────────────────────────────────────
  近1日:   {change_1d:+7.2f}%
  近5日:   {change_5d:+7.2f}%
  近20日:  {change_20d:+7.2f}%

🎯 市场微观结构分析:
────────────────────────────────────────────────────────────────────────────

📍 关键价格水平:
  当前价格: ¥{current_price:.2f}
  近期高点: {', '.join([f'¥{h:.2f}' for h in market_structure['recent_highs'][:3]])}
  近期低点: {', '.join([f'¥{l:.2f}' for l in market_structure['recent_lows'][:3]])}

📈 趋势线分析:
  上升趋势线: {f"¥{market_structure['uptrend_line']:.2f}" if market_structure['uptrend_line'] is not None else '未确认'}
  下降趋势线: {f"¥{market_structure['downtrend_line']:.2f}" if market_structure['downtrend_line'] is not None else '未确认'}

📊 均线斜率:"""

        for ma, slope in market_structure['ma_slopes'].items():
            direction = "上升" if slope > 0 else "下降" if slope < 0 else "横盘"
            report += f"\n  {ma}: {direction} (斜率: {slope:+.4f})"

        report += f"""
  布林带通道: ¥{market_structure['bollinger_bands']['lower']:.2f} - ¥{market_structure['bollinger_bands']['upper']:.2f}
  ATR波动率: ¥{market_structure['avg_atr']:.2f} ({market_structure['avg_atr']/current_price*100:.1f}%)

💎 成交量分布图分析:
────────────────────────────────────────────────────────────────────────────

📍 高成交量区域 (前5个):
"""

        for i, area in enumerate(volume_profile['high_volume_areas'][:5], 1):
            position = "支撑" if area['price'] < current_price else "阻力"
            distance = abs(area['price'] - current_price) / current_price * 100
            report += f"\n  区域{i}: ¥{area['price']:.2f} ({position}位，距离{distance:+.1f}%)"

        report += f"""

📊 成交量支撑位: {', '.join([f'¥{s:.2f}' for s in volume_profile['support_levels'][:3]])}
📊 成交量阻力位: {', '.join([f'¥{r:.2f}' for r in volume_profile['resistance_levels'][:3]])}

⚖️ 风险回报场景分析:
────────────────────────────────────────────────────────────────────────────

🚀 牛市场景:
  目标价格: ¥{risk_scenarios['bullish']['target_price']:.2f}
  潜在收益: +{risk_scenarios['bullish']['potential_return']:.1f}%
  成功概率: {risk_scenarios['bullish']['probability']}

🐻 熊市场景:
  目标价格: ¥{risk_scenarios['bearish']['target_price']:.2f}
  潜在收益: {risk_scenarios['bearish']['potential_return']:+.1f}%
  成功概率: {risk_scenarios['bearish']['probability']}

📊 震荡场景:
  目标区间: {risk_scenarios['neutral']['target_range']}
  预期收益: {risk_scenarios['neutral']['expected_return']:+.1f}%
  成功概率: {risk_scenarios['neutral']['probability']}

🎲 风险等级评估:
  波动性风险: {risk_scenarios['risk_levels']['volatility_risk']}
  趋势风险: {risk_scenarios['risk_levels']['trend_risk']}
  成交量风险: {risk_scenarios['risk_levels']['volume_risk']}
  综合风险: {risk_scenarios['risk_levels']['overall_risk']}

💸 止损设置深度解析:
────────────────────────────────────────────────────────────────────────────

🎯 推荐止损位: ¥{stop_loss_explanation['recommended_stop']:.2f}
📊 止损幅度: {(stop_loss_explanation['recommended_stop']/current_price-1)*100:+.2f}%

🔍 止损选择逻辑:
"""

        for reason in stop_loss_explanation['reasoning']:
            report += f"  • {reason}\n"

        report += f"""
🛡️ 技术支撑因素:
"""

        for factor in stop_loss_explanation['technical_factors']:
            report += f"  • {factor}\n"

        if stop_loss_explanation['risk_considerations']:
            report += f"""
⚠️ 风险考量:
"""
            for risk in stop_loss_explanation['risk_considerations']:
                report += f"  • {risk}\n"

        if stop_loss_explanation['alternative_stops']:
            report += f"""
🔄 备选止损位:
"""
            for alt_stop in stop_loss_explanation['alternative_stops']:
                report += f"  • {alt_stop['type']}: ¥{alt_stop['price']:.2f} ({alt_stop['distance_pct']:+.2f}%)\n"

        report += f"""
🎯 止盈设置深度解析:
────────────────────────────────────────────────────────────────────────────"""

        for target in take_profit_explanation['targets']:
            report += f"""
  🎯 目标{target['target']}: ¥{target['price']:.2f} (+{target['profit_pct']:+.2f}%)
     类型: {target['type']}
     强度: {target['strength']}
     建议减仓: {target['position']}"""

        report += f"""

💡 止盈策略逻辑:
"""

        for reason in take_profit_explanation['reasoning']:
            report += f"  {reason}\n"

        report += f"""
🔧 技术执行因素:
"""

        for factor in take_profit_explanation['technical_factors']:
            report += f"  {factor}\n"

        report += f"""
📊 到达概率分析:
"""

        for target, prob in take_profit_explanation['probability_analysis'].items():
            report += f"  {target}: {prob}\n"

        # 风险回报分析
        if stop_loss_explanation['recommended_stop'] and take_profit_explanation['targets']:
            risk = abs((stop_loss_explanation['recommended_stop'] / current_price - 1) * 100)
            first_target = take_profit_explanation['targets'][0]
            reward = first_target['profit_pct']

            report += f"""
📈 风险回报比分析:
  单笔最大风险: {risk:.2f}%
  第一目标收益: {reward:.2f}%
  风险回报比: 1:{reward/risk:.2f}
  推荐仓位: {'较大(20-30%)' if reward/risk > 2 else '适中(15-20%)' if reward/risk > 1.5 else '较小(10-15%)'}

💡 深度操作建议:
────────────────────────────────────────────────────────────────────────────

🟢 建议买入策略:
  • 立即在¥{current_price:.2f}附近建仓
  • 严格设置止损¥{stop_loss_explanation['recommended_stop']:.2f}
  • 分批止盈，锁定利润
  • 密切关注成交量变化

📊 仓位管理:
  • 初始仓位: 15-20%（基于风险回报比）
  • 加仓条件: 突破¥{take_profit_explanation['targets'][0]['price']:.2f}且放量
  • 减仓时机: 按计划分批止盈
  • 最大仓位: 不超过30%

⏰ 执行时机:
  • 当前市场: {'强势' if risk_scenarios['bullish']['probability'] == 'high' else '中性'}
  • 建议行动: 立即执行
  • 持有周期: 2-6周
  • 重点关注: 成交量确认、趋势延续

⚠️ 特别风险提示:
────────────────────────────────────────────────────────────────────────────
  • 高波动股票需严格止损，单笔损失控制在{risk:.1f}%以内
  • 创业板股票波动性大，需做好心理准备
  • 密切关注市场情绪和政策变化
  • 分批操作，避免一次性重仓
  • 止损是纪律，不是建议

🔍 监控指标:
  • 成交量变化：需放量确认上涨
  • RSI指标：关注是否进入超买区
  • 均线系统：保持多头排列
  • 市场情绪：创业板整体走势

═════════════════════════════════════════════════════════════════════════════════
                      300450深度分析完成
═════════════════════════════════════════════════════════════════════════════════"""

        print("300450深度分析报告生成完成")
        return report


def main():
    """主函数"""
    stock_code = "300450.SZ"

    print("🔍 开始300450深度技术分析...")
    print("💎 详解止损止盈设置的技术原理")

    try:
        # 初始化分析器
        analyzer = Deep300450Analyzer(stock_code)

        # 加载数据并计算指标
        df, factors_df = analyzer.load_and_prepare_data()

        # 执行深度分析
        volume_profile = analyzer.calculate_volume_profile(df)
        market_structure = analyzer.analyze_market_structure(df, factors_df)
        risk_scenarios = analyzer.analyze_risk_reward_scenarios(df, factors_df, market_structure, volume_profile)
        stop_loss_logic = analyzer.explain_stop_loss_logic(df, factors_df, market_structure, volume_profile)
        take_profit_logic = analyzer.explain_take_profit_logic(df, factors_df, market_structure, volume_profile)

        # 生成深度报告
        report = analyzer.generate_deep_analysis_report(df, factors_df)

        # 输出报告
        print(report)

        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_deep_analysis_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📝 300450深度分析报告已保存到: {report_file}")

        # 保存分析数据
        analysis_data = {
            'stock_code': stock_code,
            'analysis_time': datetime.now().isoformat(),
            'volume_profile': volume_profile,
            'market_structure': market_structure,
            'risk_scenarios': risk_scenarios,
            'stop_loss_logic': stop_loss_explanation,
            'take_profit_logic': take_profit_logic,
            'data_points': len(df),
            'indicators_count': len(factors_df.columns)
        }

        json_file = report_file.replace('.txt', '_analysis_data.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"📊 300450分析数据已保存到: {json_file}")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()