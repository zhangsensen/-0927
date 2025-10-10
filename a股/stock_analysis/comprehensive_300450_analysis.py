#!/usr/bin/env python3
"""
300450综合分析报告 - 结合多时间框架与增强因子分析
整合日线、小时线数据和EnhancedFactorCalculator的综合分析
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


class Comprehensive300450Analyzer:
    """300450综合分析器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data_dir = "/Users/zhangshenshen/深度量化0927/a股"

        print(f"🔍 300450综合分析器初始化")
        print(f"   股票代码: {stock_code}")
        print(f"   分析模式: 多时间框架 + 增强因子 + 综合技术分析")

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载所有可用数据"""
        print("正在加载300450完整数据集...")

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

        print(f"✅ 日线数据加载完成: {len(df_daily)}条记录 ({df_daily.index[0]} 到 {df_daily.index[-1]})")
        print(f"✅ 小时线数据加载完成: {len(df_hourly)}条记录 ({df_hourly.index[0]} 到 {df_hourly.index[-1]})")

        return df_daily, df_hourly

    def calculate_comprehensive_indicators(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """计算全面的技术指标"""
        print("正在计算综合技术指标...")

        calculator = EnhancedFactorCalculator()

        # 计算日线指标
        print("📊 计算日线综合技术指标...")
        daily_factors = calculator.calculate_comprehensive_factors(df_daily, TimeFrame.DAILY)

        # 计算小时线指标
        print("📊 计算小时线综合技术指标...")
        hourly_factors = calculator.calculate_comprehensive_factors(df_hourly, TimeFrame.DAILY)

        if daily_factors is not None and hourly_factors is not None:
            print(f"✅ 日线指标计算完成: {daily_factors.shape[1]}个指标")
            print(f"✅ 小时线指标计算完成: {hourly_factors.shape[1]}个指标")
        else:
            raise ValueError("技术指标计算失败")

        return daily_factors, hourly_factors

    def perform_comprehensive_analysis(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                    daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """执行综合分析"""
        print("正在执行综合技术分析...")

        analysis = {
            'price_analysis': self.analyze_price_behavior(df_daily, df_hourly),
            'trend_analysis': self.analyze_trend_patterns(df_daily, df_hourly, daily_factors, hourly_factors),
            'momentum_analysis': self.analyze_momentum_indicators(daily_factors, hourly_factors),
            'volatility_analysis': self.analyze_volatility_patterns(df_daily, df_hourly, daily_factors, hourly_factors),
            'volume_analysis': self.analyze_volume_patterns(df_daily, df_hourly),
            'pattern_analysis': self.analyze_chart_patterns(df_daily, df_hourly),
            'support_resistance': self.identify_support_resistance(df_daily, df_hourly),
            'risk_assessment': self.assess_risk_factors(df_daily, df_hourly, daily_factors, hourly_factors),
            'multi_timeframe_signals': self.generate_multi_timeframe_signals(df_daily, df_hourly, daily_factors, hourly_factors)
        }

        # 计算综合评级
        analysis['overall_rating'] = self.calculate_overall_rating(analysis)

        return analysis

    def analyze_price_behavior(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """分析价格行为"""
        try:
            # 日线价格分析
            daily_latest = df_daily.iloc[-1]
            daily_prev = df_daily.iloc[-2] if len(df_daily) >= 2 else daily_latest
            daily_change = (daily_latest['close'] / daily_prev['close'] - 1) * 100

            # 小时线价格分析
            hourly_latest = df_hourly.iloc[-1]
            hourly_prev = df_hourly.iloc[-2] if len(df_hourly) >= 2 else hourly_latest
            hourly_change = (hourly_latest['close'] / hourly_prev['close'] - 1) * 100

            # 多周期表现
            daily_5d = (daily_latest['close'] / df_daily['close'].iloc[-6] - 1) * 100 if len(df_daily) >= 6 else 0
            daily_20d = (daily_latest['close'] / df_daily['close'].iloc[-21] - 1) * 100 if len(df_daily) >= 21 else 0

            # 价格位置分析
            daily_high_20d = df_daily['high'].tail(20).max()
            daily_low_20d = df_daily['low'].tail(20).min()
            price_position = (daily_latest['close'] - daily_low_20d) / (daily_high_20d - daily_low_20d) * 100

            return {
                'daily_price': daily_latest['close'],
                'hourly_price': hourly_latest['close'],
                'daily_change_1d': daily_change,
                'hourly_change_1h': hourly_change,
                'daily_change_5d': daily_5d,
                'daily_change_20d': daily_20d,
                'price_position_20d': price_position,
                'daily_range_20d': (daily_high_20d, daily_low_20d),
                'price_momentum': self.calculate_price_momentum(df_daily, df_hourly)
            }
        except Exception as e:
            print(f"❌ 价格行为分析失败: {e}")
            return {}

    def analyze_trend_patterns(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                             daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """分析趋势模式"""
        try:
            latest_daily = df_daily.iloc[-1]
            latest_hourly = df_hourly.iloc[-1]

            # 移动平均趋势分析
            ma_trends = {}
            for ma_period in [5, 10, 20, 30, 60]:
                ma_col = f"MA_{ma_period}"
                if ma_col in daily_factors.columns:
                    ma_value = daily_factors[ma_col].iloc[-1]
                    ma_trends[f"daily_ma{ma_period}"] = {
                        'value': ma_value,
                        'above': latest_daily['close'] > ma_value,
                        'slope': self.calculate_ma_slope(daily_factors, ma_col, 5)
                    }

            # 趋势强度分析
            trend_strength = self.calculate_trend_strength(df_daily)
            trend_consistency = self.analyze_trend_consistency(df_daily, df_hourly)

            return {
                'ma_trends': ma_trends,
                'trend_strength': trend_strength,
                'trend_consistency': trend_consistency,
                'primary_trend': self.determine_primary_trend(ma_trends, trend_strength),
                'trend_duration': self.calculate_trend_duration(df_daily)
            }
        except Exception as e:
            print(f"❌ 趋势模式分析失败: {e}")
            return {}

    def analyze_momentum_indicators(self, daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """分析动量指标"""
        try:
            momentum_indicators = {}

            # RSI分析
            rsi_indicators = {}
            for rsi_period in [14, 20, 30, 60]:
                rsi_cols = [col for col in daily_factors.columns if f'RSI_{rsi_period}' in col]
                if rsi_cols:
                    rsi_indicators[f"daily_rsi_{rsi_period}"] = daily_factors[rsi_cols[0]].iloc[-1]

            for rsi_period in [14, 20]:
                rsi_cols = [col for col in hourly_factors.columns if f'RSI_{rsi_period}' in col]
                if rsi_cols:
                    rsi_indicators[f"hourly_rsi_{rsi_period}"] = hourly_factors[rsi_cols[0]].iloc[-1]

            momentum_indicators['rsi'] = rsi_indicators

            # MACD分析
            macd_indicators = {}
            macd_daily_cols = [col for col in daily_factors.columns if 'MACD' in col and 'Signal' not in col and 'Histogram' not in col]
            if macd_daily_cols:
                macd_indicators['daily_macd'] = daily_factors[macd_daily_cols[0]].iloc[-1]

            macd_hourly_cols = [col for col in hourly_factors.columns if 'MACD' in col and 'Signal' not in col and 'Histogram' not in col]
            if macd_hourly_cols:
                macd_indicators['hourly_macd'] = hourly_factors[macd_hourly_cols[0]].iloc[-1]

            momentum_indicators['macd'] = macd_indicators

            # 随机指标分析
            stoch_indicators = {}
            stoch_daily_cols = [col for col in daily_factors.columns if 'STOCH' in col and '%K' in col]
            if stoch_daily_cols:
                stoch_indicators['daily_stoch_k'] = daily_factors[stoch_daily_cols[0]].iloc[-1]

            momentum_indicators['stoch'] = stoch_indicators

            # 综合动量评级
            momentum_indicators['overall_momentum'] = self.assess_overall_momentum(momentum_indicators)

            return momentum_indicators
        except Exception as e:
            print(f"❌ 动量指标分析失败: {e}")
            return {}

    def analyze_volatility_patterns(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                  daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """分析波动性模式"""
        try:
            volatility_analysis = {}

            # ATR分析
            atr_cols = [col for col in daily_factors.columns if 'ATR' in col]
            if atr_cols:
                latest_atr = daily_factors[atr_cols[0]].iloc[-1]
                current_price = df_daily['close'].iloc[-1]
                atr_percentage = (latest_atr / current_price) * 100

                volatility_analysis['daily_atr'] = {
                    'value': latest_atr,
                    'percentage': atr_percentage,
                    'historical_avg': daily_factors[atr_cols[0]].mean(),
                    'volatility_level': self.classify_volatility(atr_percentage)
                }

            # 布林带分析
            bb_cols = [col for col in daily_factors.columns if 'BB_20_2' in col]
            if bb_cols:
                bb_upper_cols = [col for col in bb_cols if 'Upper' in col]
                bb_lower_cols = [col for col in bb_cols if 'Lower' in col]

                if bb_upper_cols and bb_lower_cols:
                    bb_width = (daily_factors[bb_upper_cols[0]].iloc[-1] - daily_factors[bb_lower_cols[0]].iloc[-1]) / current_price * 100
                    volatility_analysis['bollinger_width'] = bb_width

            # 价格波动分析
            daily_returns = df_daily['close'].pct_change().dropna()
            hourly_returns = df_hourly['close'].pct_change().dropna()

            volatility_analysis['price_volatility'] = {
                'daily_std': daily_returns.std() * np.sqrt(252),  # 年化波动率
                'hourly_std': hourly_returns.std() * np.sqrt(365 * 24),  # 年化波动率
                'daily_current_vol': daily_returns.tail(20).std() * np.sqrt(252),
                'volatility_trend': self.analyze_volatility_trend(daily_returns)
            }

            return volatility_analysis
        except Exception as e:
            print(f"❌ 波动性分析失败: {e}")
            return {}

    def analyze_volume_patterns(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """分析成交量模式"""
        try:
            volume_analysis = {}

            # 日线成交量分析
            latest_daily_volume = df_daily['volume'].iloc[-1]
            daily_avg_volume = df_daily['volume'].tail(20).mean()
            daily_volume_ratio = latest_daily_volume / daily_avg_volume

            # 小时线成交量分析
            latest_hourly_volume = df_hourly['volume'].iloc[-1]
            hourly_avg_volume = df_hourly['volume'].tail(24).mean()  # 最近24小时
            hourly_volume_ratio = latest_hourly_volume / hourly_avg_volume

            # 成交量趋势
            daily_volume_trend = self.analyze_volume_trend(df_daily['volume'])
            hourly_volume_trend = self.analyze_volume_trend(df_hourly['volume'])

            # 价格成交量关系
            price_volume_correlation = self.calculate_price_volume_correlation(df_daily)

            volume_analysis = {
                'daily_volume': {
                    'latest': latest_daily_volume,
                    'avg_20d': daily_avg_volume,
                    'ratio': daily_volume_ratio,
                    'trend': daily_volume_trend
                },
                'hourly_volume': {
                    'latest': latest_hourly_volume,
                    'avg_24h': hourly_avg_volume,
                    'ratio': hourly_volume_ratio,
                    'trend': hourly_volume_trend
                },
                'price_volume_correlation': price_volume_correlation,
                'volume_signal': self.assess_volume_signal(daily_volume_ratio, hourly_volume_ratio)
            }

            return volume_analysis
        except Exception as e:
            print(f"❌ 成交量分析失败: {e}")
            return {}

    def analyze_chart_patterns(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """分析图表形态"""
        try:
            patterns = {}

            # 支撑阻力突破模式
            patterns['breakout_patterns'] = self.identify_breakout_patterns(df_daily)

            # 整理形态
            patterns['consolidation_patterns'] = self.identify_consolidation_patterns(df_daily)

            # 反转形态
            patterns['reversal_patterns'] = self.identify_reversal_patterns(df_daily)

            # 持续形态
            patterns['continuation_patterns'] = self.identify_continuation_patterns(df_daily)

            return patterns
        except Exception as e:
            print(f"❌ 图表形态分析失败: {e}")
            return {}

    def identify_support_resistance(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """识别支撑阻力位"""
        try:
            # 基于历史高低点的支撑阻力
            recent_highs = df_daily['high'].tail(60).nlargest(10)
            recent_lows = df_daily['low'].tail(60).nsmallest(10)

            # 基于成交量的支撑阻力
            volume_levels = self.identify_volume_based_levels(df_daily)

            # 基于技术指标的支撑阻力
            current_price = df_daily['close'].iloc[-1]

            support_resistance = {
                'key_resistance_levels': recent_highs.head(5).tolist(),
                'key_support_levels': recent_lows.head(5).tolist(),
                'volume_based_levels': volume_levels,
                'nearest_resistance': self.find_nearest_resistance(current_price, recent_highs.head(5).tolist()),
                'nearest_support': self.find_nearest_support(current_price, recent_lows.head(5).tolist()),
                'price_position': self.assess_price_position(current_price, recent_lows.head(5).tolist(), recent_highs.head(5).tolist())
            }

            return support_resistance
        except Exception as e:
            print(f"❌ 支撑阻力分析失败: {e}")
            return {}

    def assess_risk_factors(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                          daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """评估风险因素"""
        try:
            risk_factors = {}

            # 波动性风险
            daily_returns = df_daily['close'].pct_change().dropna()
            volatility_risk = daily_returns.std() * np.sqrt(252)
            risk_factors['volatility_risk'] = {
                'annualized_volatility': volatility_risk,
                'risk_level': self.classify_risk_level(volatility_risk),
                'max_drawdown': self.calculate_max_drawdown(df_daily)
            }

            # 流动性风险
            avg_volume = df_daily['volume'].tail(20).mean()
            current_volume = df_daily['volume'].iloc[-1]
            liquidity_risk = current_volume / avg_volume
            risk_factors['liquidity_risk'] = {
                'volume_ratio': liquidity_risk,
                'risk_level': self.classify_liquidity_risk(liquidity_risk)
            }

            # 趋势风险
            trend_risk = self.assess_trend_risk(df_daily, daily_factors)
            risk_factors['trend_risk'] = trend_risk

            # 综合风险评级
            risk_factors['overall_risk'] = self.calculate_overall_risk(risk_factors)

            return risk_factors
        except Exception as e:
            print(f"❌ 风险评估失败: {e}")
            return {}

    def generate_multi_timeframe_signals(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                      daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame) -> Dict:
        """生成多时间框架信号"""
        try:
            signals = {
                'daily_signals': self.generate_timeframe_signals(df_daily, daily_factors, '日线'),
                'hourly_signals': self.generate_timeframe_signals(df_hourly, hourly_factors, '小时线'),
                'combined_signals': {},
                'signal_strength': 0,
                'confidence_level': 'medium'
            }

            # 整合多时间框架信号
            daily_bullish = len([s for s in signals['daily_signals'] if s['type'] == 'bullish'])
            daily_bearish = len([s for s in signals['daily_signals'] if s['type'] == 'bearish'])
            hourly_bullish = len([s for s in signals['hourly_signals'] if s['type'] == 'bullish'])
            hourly_bearish = len([s for s in signals['hourly_signals'] if s['type'] == 'bearish'])

            total_bullish = daily_bullish + hourly_bullish
            total_bearish = daily_bearish + hourly_bearish

            signals['combined_signals'] = {
                'total_bullish': total_bullish,
                'total_bearish': total_bearish,
                'net_signal': total_bullish - total_bearish,
                'timeframe_agreement': self.check_timeframe_agreement(daily_bullish - daily_bearish, hourly_bullish - hourly_bearish)
            }

            # 计算信号强度
            signals['signal_strength'] = (total_bullish - total_bearish) * 0.5

            # 确定置信度
            if abs(signals['signal_strength']) >= 2 and signals['combined_signals']['timeframe_agreement']:
                signals['confidence_level'] = 'high'
            elif abs(signals['signal_strength']) >= 1:
                signals['confidence_level'] = 'medium'
            else:
                signals['confidence_level'] = 'low'

            return signals
        except Exception as e:
            print(f"❌ 多时间框架信号生成失败: {e}")
            return {}

    def generate_timeframe_signals(self, df: pd.DataFrame, factors: pd.DataFrame, timeframe_name: str) -> List[Dict]:
        """生成单个时间框架的信号"""
        signals = []
        try:
            latest_price = df['close'].iloc[-1]

            # RSI信号
            rsi_cols = [col for col in factors.columns if 'RSI_14' in col]
            if rsi_cols:
                rsi_value = factors[rsi_cols[0]].iloc[-1]
                if rsi_value > 70:
                    signals.append({'type': 'bearish', 'indicator': 'RSI', 'value': rsi_value, 'message': f'{timeframe_name}RSI超买'})
                elif rsi_value < 30:
                    signals.append({'type': 'bullish', 'indicator': 'RSI', 'value': rsi_value, 'message': f'{timeframe_name}RSI超卖'})
                elif rsi_value > 50:
                    signals.append({'type': 'bullish', 'indicator': 'RSI', 'value': rsi_value, 'message': f'{timeframe_name}RSI强势'})

            # MACD信号
            macd_cols = [col for col in factors.columns if 'MACD' in col and 'Signal' not in col and 'Histogram' not in col]
            signal_cols = [col for col in factors.columns if 'MACD' in col and 'Signal' in col]

            if macd_cols and signal_cols:
                macd_value = factors[macd_cols[0]].iloc[-1]
                signal_value = factors[signal_cols[0]].iloc[-1]

                if macd_value > signal_value and macd_value > 0:
                    signals.append({'type': 'bullish', 'indicator': 'MACD', 'value': macd_value, 'message': f'{timeframe_name}MACD多头'})
                elif macd_value < signal_value and macd_value < 0:
                    signals.append({'type': 'bearish', 'indicator': 'MACD', 'value': macd_value, 'message': f'{timeframe_name}MACD空头'})

            # 移动平均信号
            ma20_cols = [col for col in factors.columns if 'MA_20' in col]
            if ma20_cols:
                ma20_value = factors[ma20_cols[0]].iloc[-1]
                if latest_price > ma20_value:
                    signals.append({'type': 'bullish', 'indicator': 'MA20', 'value': ma20_value, 'message': f'{timeframe_name}价格位于MA20上方'})
                else:
                    signals.append({'type': 'bearish', 'indicator': 'MA20', 'value': ma20_value, 'message': f'{timeframe_name}价格位于MA20下方'})

        except Exception as e:
            print(f"❌ {timeframe_name}信号生成失败: {e}")

        return signals

    # 辅助方法
    def calculate_price_momentum(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """计算价格动量"""
        try:
            daily_momentum = {}
            hourly_momentum = {}

            # 日线动量
            for period in [1, 5, 10, 20]:
                if len(df_daily) > period:
                    momentum = (df_daily['close'].iloc[-1] / df_daily['close'].iloc[-period-1] - 1) * 100
                    daily_momentum[f'{period}d'] = momentum

            # 小时线动量
            for period in [1, 4, 8, 24]:
                if len(df_hourly) > period:
                    momentum = (df_hourly['close'].iloc[-1] / df_hourly['close'].iloc[-period-1] - 1) * 100
                    hourly_momentum[f'{period}h'] = momentum

            return {'daily': daily_momentum, 'hourly': hourly_momentum}
        except:
            return {'daily': {}, 'hourly': {}}

    def calculate_ma_slope(self, factors: pd.DataFrame, ma_col: str, periods: int) -> float:
        """计算移动平均斜率"""
        try:
            if len(factors) > periods:
                recent_ma = factors[ma_col].tail(periods)
                slope = (recent_ma.iloc[-1] - recent_ma.iloc[0]) / periods
                return slope
            return 0
        except:
            return 0

    def calculate_trend_strength(self, df: pd.DataFrame) -> Dict:
        """计算趋势强度"""
        try:
            if len(df) < 20:
                return {'strength': 0, 'direction': 'unknown'}

            # 使用线性回归计算趋势强度
            x = np.arange(len(df.tail(20)))
            y = df['close'].tail(20).values
            slope, intercept = np.polyfit(x, y, 1)

            # 计算R²
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                'strength': abs(slope) * r_squared,
                'direction': 'upward' if slope > 0 else 'downward',
                'r_squared': r_squared
            }
        except:
            return {'strength': 0, 'direction': 'unknown', 'r_squared': 0}

    def analyze_trend_consistency(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame) -> Dict:
        """分析趋势一致性"""
        try:
            # 简化的趋势一致性分析
            daily_trend = df_daily['close'].tail(5).is_monotonic_increasing or df_daily['close'].tail(5).is_monotonic_decreasing
            hourly_trend = df_hourly['close'].tail(24).is_monotonic_increasing or df_hourly['close'].tail(24).is_monotonic_decreasing

            return {
                'daily_consistent': daily_trend,
                'hourly_consistent': hourly_trend,
                'overall_consistent': daily_trend and hourly_trend
            }
        except:
            return {'daily_consistent': False, 'hourly_consistent': False, 'overall_consistent': False}

    def determine_primary_trend(self, ma_trends: Dict, trend_strength: Dict) -> str:
        """确定主要趋势"""
        try:
            bullish_count = sum(1 for trend in ma_trends.values() if trend.get('above', False))
            total_count = len(ma_trends)

            if bullish_count > total_count * 0.6:
                return 'uptrend'
            elif bullish_count < total_count * 0.4:
                return 'downtrend'
            else:
                return 'sideways'
        except:
            return 'unknown'

    def calculate_trend_duration(self, df: pd.DataFrame) -> Dict:
        """计算趋势持续时间"""
        try:
            # 简化版本：计算当前价格相对于MA20的位置持续时间
            if len(df) < 20:
                return {'duration': 0, 'type': 'unknown'}

            current_price = df['close'].iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]

            # 计算价格在MA20上方或下方的天数
            above_ma20 = df['close'] > df['close'].rolling(20).mean()
            current_position = above_ma20.iloc[-1]

            # 计算连续天数
            duration = 0
            for i in range(len(above_ma20) - 1, -1, -1):
                if above_ma20.iloc[i] == current_position:
                    duration += 1
                else:
                    break

            return {
                'duration': duration,
                'type': 'above_ma20' if current_position else 'below_ma20',
                'current_position': 'above' if current_position else 'below'
            }
        except:
            return {'duration': 0, 'type': 'unknown'}

    def assess_overall_momentum(self, momentum_indicators: Dict) -> str:
        """评估整体动量"""
        try:
            bullish_signals = 0
            bearish_signals = 0

            # RSI动量
            rsi_data = momentum_indicators.get('rsi', {})
            for rsi_key, rsi_value in rsi_data.items():
                if isinstance(rsi_value, (int, float)):
                    if rsi_value > 50:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1

            # MACD动量
            macd_data = momentum_indicators.get('macd', {})
            for macd_key, macd_value in macd_data.items():
                if isinstance(macd_value, (int, float)):
                    if macd_value > 0:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1

            if bullish_signals > bearish_signals:
                return 'bullish'
            elif bearish_signals > bullish_signals:
                return 'bearish'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def classify_volatility(self, atr_percentage: float) -> str:
        """分类波动性"""
        if atr_percentage > 5:
            return 'very_high'
        elif atr_percentage > 3:
            return 'high'
        elif atr_percentage > 2:
            return 'medium'
        else:
            return 'low'

    def analyze_volatility_trend(self, returns: pd.Series) -> str:
        """分析波动性趋势"""
        try:
            if len(returns) < 20:
                return 'insufficient_data'

            recent_vol = returns.tail(10).std()
            historical_vol = returns.head(20).std()

            if recent_vol > historical_vol * 1.2:
                return 'increasing'
            elif recent_vol < historical_vol * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'

    def analyze_volume_trend(self, volume_series: pd.Series) -> str:
        """分析成交量趋势"""
        try:
            if len(volume_series) < 10:
                return 'insufficient_data'

            recent_avg = volume_series.tail(5).mean()
            historical_avg = volume_series.head(10).mean()

            if recent_avg > historical_avg * 1.2:
                return 'increasing'
            elif recent_avg < historical_avg * 0.8:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'unknown'

    def calculate_price_volume_correlation(self, df: pd.DataFrame) -> float:
        """计算价格成交量相关性"""
        try:
            if len(df) < 20:
                return 0

            price_changes = df['close'].pct_change().tail(20).dropna()
            volume_changes = df['volume'].pct_change().tail(20).dropna()

            if len(price_changes) == len(volume_changes):
                correlation = price_changes.corr(volume_changes)
                return correlation if not np.isnan(correlation) else 0
            return 0
        except:
            return 0

    def assess_volume_signal(self, daily_ratio: float, hourly_ratio: float) -> str:
        """评估成交量信号"""
        if daily_ratio > 1.5 and hourly_ratio > 1.5:
            return 'strong_increase'
        elif daily_ratio > 1.2 or hourly_ratio > 1.2:
            return 'moderate_increase'
        elif daily_ratio < 0.8 and hourly_ratio < 0.8:
            return 'decrease'
        else:
            return 'normal'

    def identify_breakout_patterns(self, df: pd.DataFrame) -> List[str]:
        """识别突破形态"""
        patterns = []
        try:
            if len(df) < 20:
                return patterns

            # 简化的突破识别
            recent_high = df['high'].tail(20).max()
            current_price = df['close'].iloc[-1]

            if current_price > recent_high * 0.98:
                patterns.append('near_resistance_breakout')

            recent_low = df['low'].tail(20).min()
            if current_price < recent_low * 1.02:
                patterns.append('near_support_breakdown')

        except:
            pass

        return patterns

    def identify_consolidation_patterns(self, df: pd.DataFrame) -> List[str]:
        """识别整理形态"""
        patterns = []
        try:
            if len(df) < 20:
                return patterns

            # 检查价格是否在窄幅区间内整理
            recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
            current_price = df['close'].iloc[-1]

            if recent_range / current_price < 0.05:  # 5%以内
                patterns.append('tight_consolidation')
            elif recent_range / current_price < 0.10:  # 10%以内
                patterns.append('wide_consolidation')

        except:
            pass

        return patterns

    def identify_reversal_patterns(self, df: pd.DataFrame) -> List[str]:
        """识别反转形态"""
        patterns = []
        try:
            if len(df) < 10:
                return patterns

            # 简化的反转形态识别
            recent_closes = df['close'].tail(10)

            # 检查可能的双顶/双底
            if len(recent_closes) >= 6:
                peaks = []
                for i in range(1, len(recent_closes) - 1):
                    if recent_closes.iloc[i] > recent_closes.iloc[i-1] and recent_closes.iloc[i] > recent_closes.iloc[i+1]:
                        peaks.append(recent_closes.iloc[i])

                if len(peaks) >= 2 and abs(peaks[-1] - peaks[-2]) / peaks[-2] < 0.02:
                    patterns.append('potential_double_top')

        except:
            pass

        return patterns

    def identify_continuation_patterns(self, df: pd.DataFrame) -> List[str]:
        """识别持续形态"""
        patterns = []
        try:
            if len(df) < 10:
                return patterns

            # 检查旗形或三角整理
            recent_highs = df['high'].tail(10)
            recent_lows = df['low'].tail(10)

            # 如果高点逐渐下降，低点逐渐上升，可能是收敛三角形
            if (recent_highs.iloc[-3:] - recent_highs.iloc[:3]).mean() < 0 and \
               (recent_lows.iloc[-3:] - recent_lows.iloc[:3]).mean() > 0:
                patterns.append('converging_triangle')

        except:
            pass

        return patterns

    def identify_volume_based_levels(self, df: pd.DataFrame) -> List[float]:
        """识别基于成交量的价位"""
        try:
            if len(df) < 20:
                return []

            # 计算成交量加权价格
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            vwap_levels = df['vwap'].dropna().tail(5).tolist()

            return vwap_levels
        except:
            return []

    def find_nearest_resistance(self, current_price: float, resistance_levels: List[float]) -> Optional[float]:
        """找到最近的阻力位"""
        try:
            above_levels = [level for level in resistance_levels if level > current_price]
            return min(above_levels) if above_levels else None
        except:
            return None

    def find_nearest_support(self, current_price: float, support_levels: List[float]) -> Optional[float]:
        """找到最近的支撑位"""
        try:
            below_levels = [level for level in support_levels if level < current_price]
            return max(below_levels) if below_levels else None
        except:
            return None

    def assess_price_position(self, current_price: float, support_levels: List[float], resistance_levels: List[float]) -> str:
        """评估价格位置"""
        try:
            nearest_support = self.find_nearest_support(current_price, support_levels)
            nearest_resistance = self.find_nearest_resistance(current_price, resistance_levels)

            if nearest_support and nearest_resistance:
                support_distance = (current_price - nearest_support) / nearest_support * 100
                resistance_distance = (nearest_resistance - current_price) / nearest_resistance * 100

                if support_distance < resistance_distance:
                    return 'closer_to_support'
                else:
                    return 'closer_to_resistance'

            return 'middle'
        except:
            return 'unknown'

    def calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """计算最大回撤"""
        try:
            if len(df) < 2:
                return 0

            rolling_max = df['close'].expanding().max()
            drawdown = (df['close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            return max_drawdown * 100  # 转换为百分比
        except:
            return 0

    def classify_risk_level(self, volatility: float) -> str:
        """分类风险等级"""
        if volatility > 0.4:
            return 'very_high'
        elif volatility > 0.3:
            return 'high'
        elif volatility > 0.2:
            return 'medium'
        else:
            return 'low'

    def classify_liquidity_risk(self, volume_ratio: float) -> str:
        """分类流动性风险"""
        if volume_ratio < 0.5:
            return 'high'
        elif volume_ratio < 0.8:
            return 'medium'
        else:
            return 'low'

    def assess_trend_risk(self, df: pd.DataFrame, factors: pd.DataFrame) -> Dict:
        """评估趋势风险"""
        try:
            # 简化的趋势风险评估
            if len(df) < 10:
                return {'risk_level': 'unknown', 'score': 0}

            current_price = df['close'].iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price

            # 价格偏离MA20的程度
            deviation = abs(current_price - ma20) / ma20

            if deviation > 0.1:  # 偏离超过10%
                risk_level = 'high'
                risk_score = deviation * 100
            elif deviation > 0.05:  # 偏离超过5%
                risk_level = 'medium'
                risk_score = deviation * 50
            else:
                risk_level = 'low'
                risk_score = deviation * 20

            return {
                'risk_level': risk_level,
                'score': risk_score,
                'ma20_deviation': deviation * 100
            }
        except:
            return {'risk_level': 'unknown', 'score': 0}

    def calculate_overall_risk(self, risk_factors: Dict) -> Dict:
        """计算综合风险"""
        try:
            risk_scores = []

            # 波动性风险评分
            vol_risk = risk_factors.get('volatility_risk', {})
            if 'max_drawdown' in vol_risk:
                risk_scores.append(abs(vol_risk['max_drawdown']))

            # 趋势风险评分
            trend_risk = risk_factors.get('trend_risk', {})
            if 'score' in trend_risk:
                risk_scores.append(trend_risk['score'])

            # 流动性风险评分
            liquidity_risk = risk_factors.get('liquidity_risk', {})
            if 'volume_ratio' in liquidity_risk:
                volume_ratio = liquidity_risk['volume_ratio']
                if volume_ratio < 0.5:
                    risk_scores.append(50)
                elif volume_ratio < 0.8:
                    risk_scores.append(25)
                else:
                    risk_scores.append(10)

            if risk_scores:
                avg_risk_score = np.mean(risk_scores)

                if avg_risk_score > 30:
                    risk_level = 'high'
                elif avg_risk_score > 15:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'

                return {
                    'risk_level': risk_level,
                    'risk_score': avg_risk_score,
                    'component_scores': risk_scores
                }
            else:
                return {'risk_level': 'unknown', 'risk_score': 0}
        except:
            return {'risk_level': 'unknown', 'risk_score': 0}

    def check_timeframe_agreement(self, daily_signal: int, hourly_signal: int) -> bool:
        """检查时间框架一致性"""
        return (daily_signal > 0 and hourly_signal > 0) or (daily_signal < 0 and hourly_signal < 0)

    def calculate_overall_rating(self, analysis: Dict) -> Dict:
        """计算综合评级"""
        try:
            rating_score = 0
            rating_factors = []

            # 价格动量评分
            price_analysis = analysis.get('price_analysis', {})
            if 'daily_change_20d' in price_analysis:
                momentum_20d = price_analysis['daily_change_20d']
                rating_score += momentum_20d * 0.3
                rating_factors.append(f'20日动量: {momentum_20d:+.1f}%')

            # 趋势评分
            trend_analysis = analysis.get('trend_analysis', {})
            primary_trend = trend_analysis.get('primary_trend', 'unknown')
            if primary_trend == 'uptrend':
                rating_score += 2
                rating_factors.append('趋势: 上升')
            elif primary_trend == 'downtrend':
                rating_score -= 2
                rating_factors.append('趋势: 下降')

            # 多时间框架信号评分
            mt_signals = analysis.get('multi_timeframe_signals', {})
            signal_strength = mt_signals.get('signal_strength', 0)
            rating_score += signal_strength * 0.5
            rating_factors.append(f'信号强度: {signal_strength:+.1f}')

            # 成交量评分
            volume_analysis = analysis.get('volume_analysis', {})
            volume_signal = volume_analysis.get('volume_signal', 'normal')
            if volume_signal in ['strong_increase', 'moderate_increase']:
                rating_score += 1
                rating_factors.append('成交量: 放大')
            elif volume_signal == 'decrease':
                rating_score -= 1
                rating_factors.append('成交量: 萎缩')

            # 确定评级
            if rating_score >= 3:
                rating = 'strong_buy'
                rating_text = '强烈买入'
            elif rating_score >= 1:
                rating = 'buy'
                rating_text = '买入'
            elif rating_score <= -3:
                rating = 'strong_sell'
                rating_text = '强烈卖出'
            elif rating_score <= -1:
                rating = 'sell'
                rating_text = '卖出'
            else:
                rating = 'hold'
                rating_text = '持有观望'

            return {
                'rating': rating,
                'rating_text': rating_text,
                'rating_score': rating_score,
                'rating_factors': rating_factors,
                'confidence': mt_signals.get('confidence_level', 'medium')
            }
        except Exception as e:
            print(f"❌ 综合评级计算失败: {e}")
            return {
                'rating': 'hold',
                'rating_text': '持有观望',
                'rating_score': 0,
                'rating_factors': [],
                'confidence': 'low'
            }

    def generate_comprehensive_report(self, df_daily: pd.DataFrame, df_hourly: pd.DataFrame,
                                    daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame,
                                    analysis: Dict) -> str:
        """生成综合分析报告"""
        print("正在生成300450综合分析报告...")

        current_price = analysis['price_analysis'].get('daily_price', df_daily['close'].iloc[-1])
        overall_rating = analysis['overall_rating']

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    300450 综合技术分析报告                                   ║
║                  多时间框架 + 增强因子 + 深度技术分析                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
💰 当前价格: ¥{current_price:.2f}
🎯 综合评级: {overall_rating['rating_text']}
📊 评级得分: {overall_rating['rating_score']:+.2f}
🔍 置信度: {overall_rating['confidence'].upper()}

📈 价格表现分析:
────────────────────────────────────────────────────────────────────────────
  日线近1日:   {analysis['price_analysis'].get('daily_change_1d', 0):+7.2f}%
  日线近5日:   {analysis['price_analysis'].get('daily_change_5d', 0):+7.2f}%
  日线近20日:  {analysis['price_analysis'].get('daily_change_20d', 0):+7.2f}%
  小时线近1小时: {analysis['price_analysis'].get('hourly_change_1h', 0):+7.2f}%

📍 价格位置:
  20日区间位置: {analysis['price_analysis'].get('price_position_20d', 0):.1f}% (0=最低, 100=最高)
  20日价格区间: ¥{analysis['price_analysis'].get('daily_range_20d', (0, 0))[0]:.2f} - ¥{analysis['price_analysis'].get('daily_range_20d', (0, 0))[1]:.2f}

🎯 趋势分析:
────────────────────────────────────────────────────────────────────────────
主要趋势: {analysis['trend_analysis'].get('primary_trend', 'unknown')}
趋势强度: {analysis['trend_analysis'].get('trend_strength', {}).get('strength', 0):.3f}
趋势方向: {analysis['trend_analysis'].get('trend_strength', {}).get('direction', 'unknown')}
趋势持续性: {analysis['trend_analysis'].get('trend_consistency', {}).get('overall_consistent', False)}

📊 移动平均趋势:
"""

        # 添加移动平均趋势信息
        ma_trends = analysis['trend_analysis'].get('ma_trends', {})
        for ma_name, ma_data in ma_trends.items():
            if ma_data:
                direction = "↑" if ma_data.get('above', False) else "↓"
                slope = ma_data.get('slope', 0)
                report += f"  {ma_name}: {direction} ¥{ma_data.get('value', 0):.2f} (斜率: {slope:+.4f})\n"

        report += f"""
🚀 动量指标分析:
────────────────────────────────────────────────────────────────────────────
整体动量: {analysis['momentum_analysis'].get('overall_momentum', 'unknown')}

RSI指标:
"""

        # 添加RSI信息
        rsi_data = analysis['momentum_analysis'].get('rsi', {})
        for rsi_name, rsi_value in rsi_data.items():
            if isinstance(rsi_value, (int, float)):
                report += f"  {rsi_name}: {rsi_value:.1f}\n"

        # 添加MACD信息
        macd_data = analysis['momentum_analysis'].get('macd', {})
        if macd_data:
            report += "\nMACD指标:\n"
            for macd_name, macd_value in macd_data.items():
                if isinstance(macd_value, (int, float)):
                    report += f"  {macd_name}: {macd_value:+.4f}\n"

        report += f"""
📊 波动性分析:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加波动性信息
        vol_analysis = analysis['volatility_analysis']
        if 'daily_atr' in vol_analysis:
            atr_data = vol_analysis['daily_atr']
            report += f"""
ATR指标:
  当前ATR: ¥{atr_data.get('value', 0):.2f} ({atr_data.get('percentage', 0):.1f}%)
  历史平均: ¥{atr_data.get('historical_avg', 0):.2f}
  波动等级: {atr_data.get('volatility_level', 'unknown')}
"""

        if 'bollinger_width' in vol_analysis:
            report += f"  布林带宽度: {vol_analysis['bollinger_width']:.2f}%\n"

        if 'price_volatility' in vol_analysis:
            vol_data = vol_analysis['price_volatility']
            report += f"""
价格波动率:
  日线年化波动率: {vol_data.get('daily_std', 0):.1%}
  当前波动率: {vol_data.get('daily_current_vol', 0):.1%}
  波动趋势: {vol_data.get('volatility_trend', 'unknown')}
"""

        report += f"""
💰 成交量分析:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加成交量信息
        vol_analysis = analysis['volume_analysis']
        if 'daily_volume' in vol_analysis:
            daily_vol = vol_analysis['daily_volume']
            report += f"""
日线成交量:
  最新成交量: {daily_vol.get('latest', 0):,.0f}
  20日平均: {daily_vol.get('avg_20d', 0):,.0f}
  成交量比率: {daily_vol.get('ratio', 0):.2f}x
  成交量趋势: {daily_vol.get('trend', 'unknown')}
"""

        if 'hourly_volume' in vol_analysis:
            hourly_vol = vol_analysis['hourly_volume']
            report += f"""
小时线成交量:
  最新成交量: {hourly_vol.get('latest', 0):,.0f}
  24小时平均: {hourly_vol.get('avg_24h', 0):,.0f}
  成交量比率: {hourly_vol.get('ratio', 0):.2f}x
  成交量趋势: {hourly_vol.get('trend', 'unknown')}
"""

        report += f"""
成交量信号: {vol_analysis.get('volume_signal', 'normal')}
价格成交量相关性: {vol_analysis.get('price_volume_correlation', 0):.3f}

🎯 支撑阻力位:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加支撑阻力信息
        sr_analysis = analysis['support_resistance']
        if 'key_resistance_levels' in sr_analysis:
            resistance_levels = sr_analysis['key_resistance_levels'][:3]
            report += f"关键阻力位: {', '.join([f'¥{level:.2f}' for level in resistance_levels])}\n"

        if 'key_support_levels' in sr_analysis:
            support_levels = sr_analysis['key_support_levels'][:3]
            report += f"关键支撑位: {', '.join([f'¥{level:.2f}' for level in support_levels])}\n"

        if 'nearest_resistance' in sr_analysis:
            nearest_res = sr_analysis['nearest_resistance']
            nearest_sup = sr_analysis['nearest_support']
            report += f"最近阻力位: {f'¥{nearest_res:.2f}' if nearest_res else 'N/A'}\n"
            report += f"最近支撑位: {f'¥{nearest_sup:.2f}' if nearest_sup else 'N/A'}\n"

        report += f"价格位置: {sr_analysis.get('price_position', 'unknown')}\n"

        report += f"""
⚠️ 风险评估:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加风险信息
        risk_analysis = analysis['risk_assessment']
        if 'volatility_risk' in risk_analysis:
            vol_risk = risk_analysis['volatility_risk']
            report += f"""
波动性风险:
  年化波动率: {vol_risk.get('annualized_volatility', 0):.1%}
  最大回撤: {vol_risk.get('max_drawdown', 0):.1f}%
  风险等级: {vol_risk.get('risk_level', 'unknown')}
"""

        if 'trend_risk' in risk_analysis:
            trend_risk = risk_analysis['trend_risk']
            report += f"""
趋势风险:
  风险等级: {trend_risk.get('risk_level', 'unknown')}
  MA20偏离度: {trend_risk.get('ma20_deviation', 0):.1f}%
"""

        if 'overall_risk' in risk_analysis:
            overall_risk = risk_analysis['overall_risk']
            report += f"""
综合风险:
  风险等级: {overall_risk.get('risk_level', 'unknown')}
  风险得分: {overall_risk.get('risk_score', 0):.1f}
"""

        report += f"""
🚨 多时间框架信号:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加多时间框架信号
        mt_signals = analysis['multi_timeframe_signals']
        combined_signals = mt_signals.get('combined_signals', {})

        report += f"""
信号汇总:
  看涨信号数: {combined_signals.get('total_bullish', 0)}
  看跌信号数: {combined_signals.get('total_bearish', 0)}
  净信号强度: {combined_signals.get('net_signal', 0)}
  时间框架一致性: {'✅' if combined_signals.get('timeframe_agreement', False) else '❌'}
  信号强度: {mt_signals.get('signal_strength', 0):+.1f}
  置信度: {mt_signals.get('confidence_level', 'medium').upper()}
"""

        # 添加具体信号
        daily_signals = mt_signals.get('daily_signals', [])
        hourly_signals = mt_signals.get('hourly_signals', [])

        if daily_signals:
            report += "\n日线信号:\n"
            for signal in daily_signals:
                signal_icon = "🟢" if signal['type'] == 'bullish' else "🔴"
                report += f"  {signal_icon} {signal.get('message', '')} ({signal.get('indicator', '')}: {signal.get('value', 0):.2f})\n"

        if hourly_signals:
            report += "\n小时线信号:\n"
            for signal in hourly_signals:
                signal_icon = "🟢" if signal['type'] == 'bullish' else "🔴"
                report += f"  {signal_icon} {signal.get('message', '')} ({signal.get('indicator', '')}: {signal.get('value', 0):.2f})\n"

        report += f"""
💎 综合操作建议:
────────────────────────────────────────────────────────────────────────────
🎯 评级: {overall_rating['rating_text']} (得分: {overall_rating['rating_score']:+.2f})
🔍 评级依据: {', '.join(overall_rating['rating_factors'])}
"""

        # 添加操作建议
        rating = overall_rating['rating']
        if rating in ['strong_buy', 'buy']:
            report += f"""
🟢 建议策略: {overall_rating['rating_text']}

📍 入场建议:
  • 当前价位¥{current_price:.2f}附近可考虑建仓
  • 建议分批买入，首次仓位10-15%
  • 严格止损于最近支撑位下方

🛡️ 风险控制:
  • 止损位: ¥{analysis['support_resistance'].get('nearest_support', current_price * 0.95):.2f}
  • 密切关注多时间框架信号一致性
  • 控制单笔损失在可接受范围内

🎯 盈利目标:
  • 第一目标: ¥{analysis['support_resistance'].get('nearest_resistance', current_price * 1.05):.2f}
  • 第二目标: ¥{analysis['support_resistance'].get('key_resistance_levels', [current_price * 1.1])[0] if analysis['support_resistance'].get('key_resistance_levels') else current_price * 1.1:.2f}
  • 分批止盈，锁定利润
"""
        elif rating in ['strong_sell', 'sell']:
            report += f"""
🔴 建议策略: {overall_rating['rating_text']}

📍 出场建议:
  • 当前价位¥{current_price:.2f}建议减仓
  • 多时间框架信号显示负面趋势
  • 建议分批卖出，控制风险

🎯 目标价位:
  • 第一目标: ¥{analysis['support_resistance'].get('nearest_support', current_price * 0.95):.2f}
  • 第二目标: ¥{analysis['support_resistance'].get('key_support_levels', [current_price * 0.9])[0] if analysis['support_resistance'].get('key_support_levels') else current_price * 0.9:.2f}
"""
        else:
            report += f"""
🟡 建议策略: {overall_rating['rating_text']}

📍 观望建议:
  • 多时间框架信号不一致，等待明确方向
  • 可考虑小仓位试探市场反应
  • 密切关注信号变化和成交量配合

⏰ 观察要点:
  • 多时间框架信号改善
  • 成交量放大确认方向
  • 关键技术位突破或跌破
"""

        report += f"""
📋 监控要点:
────────────────────────────────────────────────────────────────────────────
1. 小时线RSI是否突破50中轴确认动量
2. 日线和小时线MACD是否同步转强/转弱
3. 成交量是否配合价格突破关键位
4. 多时间框架趋势一致性的变化
5. 支撑阻力位的测试情况

⚠️ 特别风险提示:
────────────────────────────────────────────────────────────────────────────
• 创业板股票波动性较大，请严格做好风险管理
• 综合分析基于历史数据，未来可能发生变化
• 建议与其他分析方法结合使用，相互验证
• 密切关注市场整体环境和政策变化
• 止损是投资纪律的重要组成部分，必须严格执行

═════════════════════════════════════════════════════════════════════════════════
                      300450综合技术分析完成
═════════════════════════════════════════════════════════════════════════════════
"""

        return report


def main():
    """主函数"""
    print("🔍 开始300450综合技术分析...")
    print("💎 多时间框架 + 增强因子 + 深度分析")

    analyzer = Comprehensive300450Analyzer("300450.SZ")

    try:
        # 加载所有数据
        df_daily, df_hourly = analyzer.load_all_data()

        # 计算综合技术指标
        daily_factors, hourly_factors = analyzer.calculate_comprehensive_indicators(df_daily, df_hourly)

        # 执行综合分析
        analysis = analyzer.perform_comprehensive_analysis(df_daily, df_hourly, daily_factors, hourly_factors)

        # 生成综合报告
        report = analyzer.generate_comprehensive_report(df_daily, df_hourly, daily_factors, hourly_factors, analysis)

        # 输出报告
        print(report)

        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_comprehensive_analysis_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n📄 综合分析报告已保存: {report_file}")

        # 保存分析数据
        data_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_comprehensive_data_{timestamp}.json"
        import json

        analysis_data = {
            'analysis_time': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_multi_timeframe',
            'data_shapes': {
                'daily_data': df_daily.shape,
                'hourly_data': df_hourly.shape,
                'daily_factors': daily_factors.shape,
                'hourly_factors': hourly_factors.shape
            },
            'overall_rating': analysis['overall_rating'],
            'risk_assessment': analysis['risk_assessment'],
            'multi_timeframe_signals': analysis['multi_timeframe_signals'],
            'key_insights': {
                'primary_trend': analysis['trend_analysis'].get('primary_trend'),
                'overall_momentum': analysis['momentum_analysis'].get('overall_momentum'),
                'volatility_level': analysis['volatility_analysis'].get('daily_atr', {}).get('volatility_level'),
                'volume_signal': analysis['volume_analysis'].get('volume_signal')
            }
        }

        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"📊 综合分析数据已保存: {data_file}")

        # 输出关键摘要
        print("\n" + "="*80)
        print("📋 关键分析摘要:")
        print("="*80)
        print(f"🎯 综合评级: {analysis['overall_rating']['rating_text']}")
        print(f"📊 评级得分: {analysis['overall_rating']['rating_score']:+.2f}")
        print(f"🔍 置信度: {analysis['overall_rating']['confidence'].upper()}")
        print(f"📈 主要趋势: {analysis['trend_analysis'].get('primary_trend', 'unknown')}")
        print(f"🚀 整体动量: {analysis['momentum_analysis'].get('overall_momentum', 'unknown')}")
        print(f"⚠️ 综合风险: {analysis['risk_assessment'].get('overall_risk', {}).get('risk_level', 'unknown')}")
        print(f"💰 当前价格: ¥{analysis['price_analysis'].get('daily_price', 0):.2f}")
        print("="*80)

    except Exception as e:
        print(f"❌ 综合分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()