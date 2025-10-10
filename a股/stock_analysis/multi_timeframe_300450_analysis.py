#!/usr/bin/env python3
"""
300450多时间框架分析 - 日线与小时线交叉分析
结合日线和小时线数据进行综合技术指标计算和分析
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
sys.path.append('/Users/zhangshenshen/深度量化0927')

from factor_system.factor_generation.enhanced_factor_calculator import EnhancedFactorCalculator
from factor_system.factor_generation.enhanced_factor_calculator import TimeFrame


class MultiTimeframe300450Analyzer:
    """300450多时间框架分析器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data_dir = "/Users/zhangshenshen/深度量化0927/a股"

        print(f"🔍 300450多时间框架分析器初始化")
        print(f"   股票代码: {stock_code}")
        print(f"   分析模式: 日线 + 小时线 交叉验证")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并准备日线和小时线数据"""
        print("正在加载300450日线和小时线数据...")

        # 读取日线数据
        daily_file = f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1d_2025-10-09.csv"
        if not os.path.exists(daily_file):
            raise FileNotFoundError(f"日线数据文件不存在: {daily_file}")

        df_daily = pd.read_csv(daily_file, skiprows=1)
        df_daily.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df_daily['Date'] = pd.to_datetime(df_daily['Date'])
        df_daily.set_index('Date', inplace=True)
        df_daily.columns = [col.lower() for col in df_daily.columns]

        # 读取小时线数据
        hourly_file = f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1h_2025-10-09.csv"
        if not os.path.exists(hourly_file):
            raise FileNotFoundError(f"小时线数据文件不存在: {hourly_file}")

        df_hourly = pd.read_csv(hourly_file)
        df_hourly['Datetime'] = pd.to_datetime(df_hourly['Datetime'])
        df_hourly.set_index('Datetime', inplace=True)
        df_hourly.columns = [col.lower() for col in df_hourly.columns]

        print(f"✅ 日线数据加载完成: {len(df_daily)}条记录")
        print(f"✅ 小时线数据加载完成: {len(df_hourly)}条记录")

        return df_daily, df_hourly

    def calculate_multi_timeframe_indicators(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算多时间框架技术指标"""
        print("正在计算多时间框架技术指标...")

        calculator = EnhancedFactorCalculator()

        # 计算日线指标
        print("📊 计算日线技术指标...")
        daily_factors = calculator.calculate_comprehensive_factors(df_daily, TimeFrame.DAILY)

        # 计算小时线指标
        print("📊 计算小时线技术指标...")
        hourly_factors = calculator.calculate_comprehensive_factors(df_hourly, TimeFrame.DAILY)  # 使用DAILY参数计算小时线指标

        if daily_factors is not None and hourly_factors is not None:
            print(f"✅ 日线指标计算完成: {daily_factors.shape[1]}个指标")
            print(f"✅ 小时线指标计算完成: {hourly_factors.shape[1]}个指标")
        else:
            raise ValueError("技术指标计算失败")

        return daily_factors, hourly_factors

    def analyze_timeframe_alignment(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                  daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """分析时间框架对齐情况"""
        print("正在分析时间框架对齐...")

        # 获取最新数据
        latest_daily_price = df_daily['close'].iloc[-1]
        latest_hourly_price = df_hourly['close'].iloc[-1]

        # 计算价格差异
        price_diff = latest_hourly_price - latest_daily_price
        price_diff_pct = (price_diff / latest_daily_price) * 100

        # 获取关键指标对齐情况
        daily_rsi = daily_factors[[col for col in daily_factors.columns if 'RSI_14' in col]].iloc[-1].iloc[0] if [col for col in daily_factors.columns if 'RSI_14' in col] else 50
        hourly_rsi = hourly_factors[[col for col in hourly_factors.columns if 'RSI_14' in col]].iloc[-1].iloc[0] if [col for col in hourly_factors.columns if 'RSI_14' in col] else 50

        # MACD对齐分析
        daily_macd_cols = [col for col in daily_factors.columns if 'MACD_12_26_9' in col and 'MACD' in col and 'Signal' not in col and 'Histogram' not in col]
        hourly_macd_cols = [col for col in hourly_factors.columns if 'MACD_12_26_9' in col and 'MACD' in col and 'Signal' not in col and 'Histogram' not in col]

        daily_macd = daily_factors[daily_macd_cols].iloc[-1].iloc[0] if daily_macd_cols else 0
        hourly_macd = hourly_factors[hourly_macd_cols].iloc[-1].iloc[0] if hourly_macd_cols else 0

        # 趋势对齐分析
        daily_ma20 = daily_factors[[col for col in daily_factors.columns if 'MA_20' in col]].iloc[-1].iloc[0] if [col for col in daily_factors.columns if 'MA_20' in col] else latest_daily_price
        hourly_ma20 = hourly_factors[[col for col in hourly_factors.columns if 'MA_20' in col]].iloc[-1].iloc[0] if [col for col in hourly_factors.columns if 'MA_20' in col] else latest_hourly_price

        return {
            'price_alignment': {
                'daily_price': latest_daily_price,
                'hourly_price': latest_hourly_price,
                'price_diff': price_diff,
                'price_diff_pct': price_diff_pct
            },
            'rsi_alignment': {
                'daily_rsi': daily_rsi,
                'hourly_rsi': hourly_rsi,
                'rsi_diff': hourly_rsi - daily_rsi
            },
            'macd_alignment': {
                'daily_macd': daily_macd,
                'hourly_macd': hourly_macd,
                'macd_diff': hourly_macd - daily_macd
            },
            'trend_alignment': {
                'daily_above_ma20': latest_daily_price > daily_ma20,
                'hourly_above_ma20': latest_hourly_price > hourly_ma20,
                'trend_consistent': (latest_daily_price > daily_ma20) == (latest_hourly_price > hourly_ma20)
            }
        }

    def detect_multi_timeframe_signals(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                     daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame,
                                     alignment: Dict) -> Dict:
        """检测多时间框架交易信号"""
        print("正在检测多时间框架交易信号...")

        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'signal_strength': 0,
            'confidence_level': 'medium'
        }

        # 1. 趋势一致性信号
        if alignment['trend_alignment']['trend_consistent']:
            if alignment['trend_alignment']['daily_above_ma20']:
                signals['bullish_signals'].append("日线和小时线均位于MA20上方 - 趋势一致向上")
                signals['signal_strength'] += 2
            else:
                signals['bearish_signals'].append("日线和小时线均位于MA20下方 - 趋势一致向下")
                signals['signal_strength'] -= 2
        else:
            signals['neutral_signals'].append("日线和小时线趋势不一致 - 等待方向明确")

        # 2. RSI对齐分析
        daily_rsi = alignment['rsi_alignment']['daily_rsi']
        hourly_rsi = alignment['rsi_alignment']['hourly_rsi']

        if daily_rsi > 50 and hourly_rsi > 50:
            signals['bullish_signals'].append(f"日线RSI({daily_rsi:.1f})和小时线RSI({hourly_rsi:.1f})均强势")
            signals['signal_strength'] += 1.5
        elif daily_rsi < 50 and hourly_rsi < 50:
            signals['bearish_signals'].append(f"日线RSI({daily_rsi:.1f})和小时线RSI({hourly_rsi:.1f})均弱势")
            signals['signal_strength'] -= 1.5
        else:
            signals['neutral_signals'].append("RSI指标在不同时间框架出现分歧")

        # 3. MACD对齐分析
        daily_macd = alignment['macd_alignment']['daily_macd']
        hourly_macd = alignment['macd_alignment']['hourly_macd']

        if daily_macd > 0 and hourly_macd > 0:
            signals['bullish_signals'].append("日线和小时线MACD均位于零轴上方")
            signals['signal_strength'] += 1
        elif daily_macd < 0 and hourly_macd < 0:
            signals['bearish_signals'].append("日线和小时线MACD均位于零轴下方")
            signals['signal_strength'] -= 1

        # 4. 价格动量分析
        hourly_momentum = self.calculate_momentum_score(df_hourly)
        daily_momentum = self.calculate_momentum_score(df_daily)

        if hourly_momentum > 0.6 and daily_momentum > 0.6:
            signals['bullish_signals'].append("多时间框架动量均强劲")
            signals['signal_strength'] += 1
        elif hourly_momentum < 0.4 and daily_momentum < 0.4:
            signals['bearish_signals'].append("多时间框架动量均疲软")
            signals['signal_strength'] -= 1

        # 5. 成交量确认
        recent_hourly_volume = df_hourly['volume'].tail(24).mean()  # 最近24小时平均成交量
        historical_hourly_volume = df_hourly['volume'].mean()
        volume_ratio = recent_hourly_volume / historical_hourly_volume

        if volume_ratio > 1.5:
            signals['bullish_signals'].append(f"小时线成交量放大({volume_ratio:.1f}倍)")
            signals['signal_strength'] += 0.5

        # 确定信号强度和置信度
        if signals['signal_strength'] >= 3:
            signals['confidence_level'] = 'high'
        elif signals['signal_strength'] <= -3:
            signals['confidence_level'] = 'high'
        elif abs(signals['signal_strength']) >= 1.5:
            signals['confidence_level'] = 'medium'
        else:
            signals['confidence_level'] = 'low'

        return signals

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """计算动量得分"""
        if len(df) < 10:
            return 0.5

        # 计算不同周期的收益率
        r1 = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) >= 2 else 0
        r5 = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0
        r10 = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1) if len(df) >= 11 else 0

        # 计算波动率
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.02

        # 综合动量得分
        momentum_score = (r1 * 0.5 + r5 * 0.3 + r10 * 0.2) / (volatility + 0.01)
        momentum_score = max(0, min(1, (momentum_score + 0.5)))  # 标准化到[0,1]

        return momentum_score

    def generate_comprehensive_signals(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                    daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """生成综合交易信号"""
        print("正在生成综合交易信号...")

        alignment = self.analyze_timeframe_alignment(df_daily, df_hourly, daily_factors, hourly_factors)
        signals = self.detect_multi_timeframe_signals(df_daily, df_hourly, daily_factors, hourly_factors, alignment)

        # 生成具体操作建议
        current_price = df_hourly['close'].iloc[-1]

        # 计算支撑阻力位
        recent_highs = df_daily['high'].tail(20).nlargest(5).tolist()
        recent_lows = df_daily['low'].tail(20).nsmallest(5).tolist()

        # ATR计算（基于日线）
        if len(df_daily) >= 14:
            high_low = df_daily['high'] - df_daily['low']
            high_close = abs(df_daily['high'] - df_daily['close'].shift())
            low_close = abs(df_daily['low'] - df_daily['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
        else:
            atr = df_daily['close'].iloc[-1] * 0.02  # 默认2%

        # 综合建议
        if signals['signal_strength'] >= 2 and signals['confidence_level'] in ['high', 'medium']:
            recommendation = 'strong_buy'
        elif signals['signal_strength'] >= 1:
            recommendation = 'buy'
        elif signals['signal_strength'] <= -2 and signals['confidence_level'] in ['high', 'medium']:
            recommendation = 'strong_sell'
        elif signals['signal_strength'] <= -1:
            recommendation = 'sell'
        else:
            recommendation = 'hold'

        return {
            'recommendation': recommendation,
            'current_price': current_price,
            'signal_strength': signals['signal_strength'],
            'confidence_level': signals['confidence_level'],
            'signals': signals,
            'technical_levels': {
                'support_levels': recent_lows[:3],
                'resistance_levels': recent_highs[:3],
                'atr': atr
            },
            'timeframe_alignment': alignment
        }

    def generate_comprehensive_report(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                   daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> str:
        """生成综合分析报告"""
        print("正在生成300450多时间框架综合分析报告...")

        comprehensive_signals = self.generate_comprehensive_signals(df_daily, df_hourly, daily_factors, hourly_factors)

        current_price = comprehensive_signals['current_price']
        recommendation = comprehensive_signals['recommendation']
        signal_strength = comprehensive_signals['signal_strength']
        confidence = comprehensive_signals['confidence_level']
        signals = comprehensive_signals['signals']
        levels = comprehensive_signals['technical_levels']
        alignment = comprehensive_signals['timeframe_alignment']

        # 计算价格表现
        daily_change_1d = (df_daily['close'].iloc[-1] / df_daily['close'].iloc[-2] - 1) * 100 if len(df_daily) >= 2 else 0
        daily_change_5d = (df_daily['close'].iloc[-1] / df_daily['close'].iloc[-6] - 1) * 100 if len(df_daily) >= 6 else 0

        hourly_change_1h = (df_hourly['close'].iloc[-1] / df_hourly['close'].iloc[-2] - 1) * 100 if len(df_hourly) >= 2 else 0
        hourly_change_4h = (df_hourly['close'].iloc[-1] / df_hourly['close'].iloc[-5] - 1) * 100 if len(df_hourly) >= 5 else 0

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                300450 多时间框架综合分析报告                                ║
║                    日线与小时线交叉验证分析                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
💰 当前价格: ¥{current_price:.2f}
🎯 综合评级: {self.get_recommendation_text(recommendation)}
📊 信号强度: {signal_strength:+.1f}
🔍 置信度: {confidence.upper()}

📈 价格表现对比:
────────────────────────────────────────────────────────────────────────────
日线数据:
  近1日:   {daily_change_1d:+7.2f}%
  近5日:   {daily_change_5d:+7.2f}%

小时线数据:
  近1小时: {hourly_change_1h:+7.2f}%
  近4小时: {hourly_change_4h:+7.2f}%

🎯 多时间框架对齐分析:
────────────────────────────────────────────────────────────────────────────

价格对齐情况:
  日线收盘价: ¥{alignment['price_alignment']['daily_price']:.2f}
  小时线最新价: ¥{alignment['price_alignment']['hourly_price']:.2f}
  价格差异: ¥{alignment['price_alignment']['price_diff']:+.2f} ({alignment['price_alignment']['price_diff_pct']:+.2f}%)

RSI指标对齐:
  日线RSI: {alignment['rsi_alignment']['daily_rsi']:.1f}
  小时线RSI: {alignment['rsi_alignment']['hourly_rsi']:.1f}
  RSI差异: {alignment['rsi_alignment']['rsi_diff']:+.1f}

MACD指标对齐:
  日线MACD: {alignment['macd_alignment']['daily_macd']:+.4f}
  小时线MACD: {alignment['macd_alignment']['hourly_macd']:+.4f}
  MACD差异: {alignment['macd_alignment']['macd_diff']:+.4f}

趋势一致性:
  日线vs MA20: {'上方' if alignment['trend_alignment']['daily_above_ma20'] else '下方'}
  小时线vs MA20: {'上方' if alignment['trend_alignment']['hourly_above_ma20'] else '下方'}
  趋势一致: {'是' if alignment['trend_alignment']['trend_consistent'] else '否'}

🚨 多时间框架信号分析:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加看涨信号
        if signals['bullish_signals']:
            report += "\n🟢 看涨信号:\n"
            for signal in signals['bullish_signals']:
                report += f"  ✓ {signal}\n"

        # 添加看跌信号
        if signals['bearish_signals']:
            report += "\n🔴 看跌信号:\n"
            for signal in signals['bearish_signals']:
                report += f"  ✗ {signal}\n"

        # 添加中性信号
        if signals['neutral_signals']:
            report += "\n🟡 中性信号:\n"
            for signal in signals['neutral_signals']:
                report += f"  • {signal}\n"

        report += f"""
📊 关键技术位:
────────────────────────────────────────────────────────────────────────────
支撑位: {', '.join([f'¥{level:.2f}' for level in levels['support_levels'][:3]])}
阻力位: {', '.join([f'¥{level:.2f}' for level in levels['resistance_levels'][:3]])}
ATR: ¥{levels['atr']:.2f} ({(levels['atr']/current_price)*100:.1f}%)

💎 综合操作建议:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加操作建议
        if recommendation in ['strong_buy', 'buy']:
            report += f"""
🟢 建议买入: {self.get_recommendation_text(recommendation)}

📍 入场策略:
  • 当前价位¥{current_price:.2f}附近可考虑建仓
  • 建议分批买入，避免一次性重仓
  • 首次仓位建议10-15%

🛡️ 风险控制:
  • 止损位: ¥{current_price - levels['atr']*2:.2f} (-{(levels['atr']*2/current_price)*100:.1f}%)
  • 基于多时间框架一致性设置止损
  • 小时线跌破关键支撑时考虑减仓

🎯 盈利目标:
  • 第一目标: ¥{current_price + levels['atr']*3:.2f} (+{(levels['atr']*3/current_price)*100:+.1f}%)
  • 第二目标: ¥{current_price + levels['atr']*5:.2f} (+{(levels['atr']*5/current_price)*100:+.1f}%)
  • 分批止盈，锁定利润
"""
        elif recommendation in ['strong_sell', 'sell']:
            report += f"""
🔴 建议卖出: {self.get_recommendation_text(recommendation)}

📍 出场策略:
  • 当前价位¥{current_price:.2f}附近建议减仓
  • 多时间框架信号均偏弱
  • 建议分批卖出，控制风险

🎯 目标价位:
  • 第一目标: ¥{current_price - levels['atr']*2:.2f} (-{(levels['atr']*2/current_price)*100:.1f}%)
  • 第二目标: ¥{current_price - levels['atr']*4:.2f} (-{(levels['atr']*4/current_price)*100:.1f}%)
"""
        else:
            report += f"""
🟡 建议观望: {self.get_recommendation_text(recommendation)}

📍 观望策略:
  • 多时间框架信号不一致，等待明确方向
  • 可考虑小仓位试探
  • 密切关注信号变化

⏰ 观察要点:
  • 小时线突破近期区间可能带来方向选择
  • 日线趋势确认后再加重仓
  • 成交量配合情况
"""

        report += f"""
📋 监控要点:
────────────────────────────────────────────────────────────────────────────
1. 小时线RSI是否突破50中轴
2. 日线和小时线MACD是否同步转强/转弱
3. 成交量是否配合价格突破
4. 多时间框架趋势一致性变化

⚠️ 风险提示:
────────────────────────────────────────────────────────────────────────────
• 多时间框架分析提高信号可靠性，但不能完全消除风险
• 创业板股票波动较大，请严格止损
• 建议与其他分析方法结合使用
• 关注市场整体环境和政策变化

═════════════════════════════════════════════════════════════════════════════════
                      300450多时间框架分析完成
═════════════════════════════════════════════════════════════════════════════════
"""

        return report

    def get_recommendation_text(self, recommendation: str) -> str:
        """获取推荐文本"""
        mapping = {
            'strong_buy': '强烈买入',
            'buy': '买入',
            'hold': '持有',
            'sell': '卖出',
            'strong_sell': '强烈卖出'
        }
        return mapping.get(recommendation, '观望')


def main():
    """主函数"""
    print("🔍 开始300450多时间框架分析...")
    print("💎 日线与小时线交叉验证分析")

    analyzer = MultiTimeframe300450Analyzer("300450.SZ")

    try:
        # 加载数据
        df_daily, df_hourly = analyzer.load_and_prepare_data()

        # 计算技术指标
        daily_factors, hourly_factors = analyzer.calculate_multi_timeframe_indicators(df_daily, df_hourly)

        # 生成综合报告
        report = analyzer.generate_comprehensive_report(df_daily, df_hourly, daily_factors, hourly_factors)

        # 输出报告
        print(report)

        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_multi_timeframe_analysis_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 报告已保存: {report_file}")

        # 保存分析数据
        data_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_multi_timeframe_data_{timestamp}.json"
        import json

        analysis_data = {
            'analysis_time': datetime.now().isoformat(),
            'daily_data_shape': df_daily.shape,
            'hourly_data_shape': df_hourly.shape,
            'daily_factors_shape': daily_factors.shape,
            'hourly_factors_shape': hourly_factors.shape,
            'latest_signals': analyzer.generate_comprehensive_signals(df_daily, df_hourly, daily_factors, hourly_factors)
        }

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"📊 分析数据已保存: {data_file}")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()