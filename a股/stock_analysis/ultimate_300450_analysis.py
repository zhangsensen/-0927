#!/usr/bin/env python3
"""
300450终极技术分析 - 使用65个专业技术指标
整合EnhancedFactorCalculator进行深度分析，提供专业级交易建议
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append("/Users/zhangshenshen/深度量化0927")

# 导入EnhancedFactorCalculator和相关枚举
from factor_system.factor_generation.enhanced_factor_calculator import (
    EnhancedFactorCalculator,
    TimeFrame,
)


class Ultimate300450Analyzer:
    """300450终极技术分析器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data_dir = "/Users/zhangshenshen/深度量化0927/a股"

        print(f"🚀 300450终极技术分析器初始化完成")
        print(f"   股票代码: {stock_code}")

    def load_and_prepare_data(self) -> pd.DataFrame:
        """加载并准备数据，转换为EnhancedFactorCalculator需要的格式"""
        print("正在加载并准备300450数据...")

        # 读取日线数据
        data_file = (
            f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1d_2025-10-09.csv"
        )

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据文件不存在: {data_file}")

        # 读取并处理数据
        df = pd.read_csv(data_file, skiprows=1)
        df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        # 转换数值列
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 清理数据
        df = df.dropna()

        print(f"数据加载完成: {len(df)}条记录")
        print(f"时间范围: {df.index.min().date()} 到 {df.index.max().date()}")
        print(f"最新价格: ¥{df['Close'].iloc[-1]:.2f}")

        # 转换为EnhancedFactorCalculator需要的格式 (小写列名)
        df.columns = [col.lower() for col in df.columns]

        return df

    def calculate_factors_with_calculator(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用EnhancedFactorCalculator正确计算技术指标"""
        print("正在使用EnhancedFactorCalculator计算300450技术指标...")

        try:
            # 初始化计算器
            calculator = EnhancedFactorCalculator()

            # 计算日线技术指标
            print("  调用calculate_comprehensive_factors方法...")
            factors_df = calculator.calculate_comprehensive_factors(df, TimeFrame.DAILY)

            if factors_df is not None:
                print(f"✅ 成功计算 {len(factors_df.columns)} 个技术指标")
                print(
                    f"  指标数据范围: {factors_df.index.min().date()} 到 {factors_df.index.max().date()}"
                )

                # 显示一些关键指标
                print("  关键指标示例:")
                key_indicators = {
                    "RSI": [col for col in factors_df.columns if "RSI14" in col],
                    "MACD": [
                        col
                        for col in factors_df.columns
                        if "MACD" in col and "Signal" not in col
                    ],
                    "MA20": [col for col in factors_df.columns if "MA20" in col],
                    "BB": [col for col in factors_df.columns if "BB_Upper_20_2" in col],
                    "ATR": [col for col in factors_df.columns if "ATR14" in col],
                }

                for indicator_type, indicator_list in key_indicators.items():
                    if indicator_list:
                        col = indicator_list[0]
                        latest_value = factors_df[col].iloc[-1]
                        if not pd.isna(latest_value):
                            print(f"    {indicator_type}: {latest_value:.4f}")

                return factors_df
            else:
                print("❌ calculate_comprehensive_factors返回None")
                return None

        except Exception as e:
            print(f"❌ EnhancedFactorCalculator计算失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def analyze_comprehensive_signals(
        self, df: pd.DataFrame, factors_df: pd.DataFrame
    ) -> Dict:
        """综合分析65个技术指标信号"""
        print("正在进行300450综合信号分析...")

        latest = df.iloc[-1]
        prev_1 = df.iloc[-2] if len(df) >= 2 else latest
        prev_3 = df.iloc[-4] if len(df) >= 4 else latest
        prev_5 = df.iloc[-6] if len(df) >= 6 else latest
        prev_10 = df.iloc[-11] if len(df) >= 11 else latest

        current_price = latest["close"]

        signals = {
            "current_price": current_price,
            "buy_signals": [],
            "sell_signals": [],
            "neutral_signals": [],
            "overall_score": 0,
            "signal_strength": "weak",
            "buy_confidence": 0,
            "sell_confidence": 0,
            "technical_summary": {},
        }

        # 1. 趋势分析 (权重: 30%)
        trend_score, trend_signals = self._analyze_trend_signals(
            df, factors_df, latest, prev_1, prev_5, prev_10
        )
        signals["buy_signals"].extend(trend_signals["buy"])
        signals["sell_signals"].extend(trend_signals["sell"])
        trend_weight = 0.3

        # 2. 动量分析 (权重: 25%)
        momentum_score, momentum_signals = self._analyze_momentum_signals(
            factors_df, latest, prev_1, prev_3, prev_5
        )
        signals["buy_signals"].extend(momentum_signals["buy"])
        signals["sell_signals"].extend(momentum_signals["sell"])
        momentum_weight = 0.25

        # 3. 超买超卖分析 (权重: 20%)
        oscillator_score, oscillator_signals = self._analyze_oscillator_signals(
            factors_df, latest, prev_1
        )
        signals["buy_signals"].extend(oscillator_signals["buy"])
        signals["sell_signals"].extend(oscillator_signals["sell"])
        oscillator_weight = 0.2

        # 4. 成交量分析 (权重: 15%)
        volume_score, volume_signals = self._analyze_volume_signals(
            df, latest, prev_1, prev_5
        )
        signals["buy_signals"].extend(volume_signals["buy"])
        signals["sell_signals"].extend(volume_signals["sell"])
        signals["neutral_signals"].extend(volume_signals["neutral"])
        volume_weight = 0.15

        # 5. 波动性分析 (权重: 10%)
        volatility_score, volatility_signals = self._analyze_volatility_signals(
            factors_df, latest, prev_5
        )
        signals["neutral_signals"].extend(volatility_signals["neutral"])
        volatility_weight = 0.1

        # 计算加权得分
        weighted_score = (
            trend_score * trend_weight
            + momentum_score * momentum_weight
            + oscillator_score * oscillator_weight
            + volume_score * volume_weight
            + volatility_score * volatility_weight
        )

        signals["overall_score"] = weighted_score * 100  # 转换为百分制

        # 计算置信度
        signals["buy_confidence"] = min(
            len([s for s in signals["buy_signals"] if s["strength"] >= 7]) / 10, 1.0
        )
        signals["sell_confidence"] = min(
            len([s for s in signals["sell_signals"] if s["strength"] >= 7]) / 10, 1.0
        )

        # 确定信号强度
        abs_score = abs(signals["overall_score"])
        if abs_score >= 70:
            signals["signal_strength"] = "very_strong"
        elif abs_score >= 50:
            signals["signal_strength"] = "strong"
        elif abs_score >= 30:
            signals["signal_strength"] = "moderate"
        elif abs_score >= 15:
            signals["signal_strength"] = "weak"
        else:
            signals["signal_strength"] = "very_weak"

        # 确定综合信号
        if signals["overall_score"] >= 50:
            signals["overall_signal"] = "强烈买入"
        elif signals["overall_score"] >= 25:
            signals["overall_signal"] = "买入"
        elif signals["overall_score"] >= 10:
            signals["overall_signal"] = "偏向买入"
        elif signals["overall_score"] <= -50:
            signals["overall_signal"] = "强烈卖出"
        elif signals["overall_score"] <= -25:
            signals["overall_signal"] = "卖出"
        elif signals["overall_score"] <= -10:
            signals["overall_signal"] = "偏向卖出"
        else:
            signals["overall_signal"] = "观望"

        # 技术摘要
        signals["technical_summary"] = {
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "oscillator_score": oscillator_score,
            "volume_score": volume_score,
            "volatility_score": volatility_score,
        }

        print(f"  300450综合信号: {signals['overall_signal']}")
        print(f"  信号强度: {signals['signal_strength']}")
        print(f"  综合得分: {signals['overall_score']:+.1f}")
        print(f"  买入信心: {signals['buy_confidence']:.1%}")
        print(f"  卖出信心: {signals['sell_confidence']:.1%}")

        return signals

    def _analyze_trend_signals(
        self,
        df: pd.DataFrame,
        factors_df: pd.DataFrame,
        latest,
        prev_1,
        prev_5,
        prev_10,
    ) -> tuple:
        """分析趋势信号"""
        score = 0
        buy_signals = []
        sell_signals = []

        current_price = latest["close"]

        # 移动平均线分析
        ma_cols = [
            col
            for col in factors_df.columns
            if col.startswith("MA") and col[2:].isdigit()
        ]
        ma_cols.sort(key=lambda x: int(x[2:]))  # 按周期排序

        mas_above = []
        mas_below = []

        for ma_col in ma_cols:
            ma_value = factors_df[ma_col].iloc[-1]
            ma_period = int(ma_col[2:])

            if not pd.isna(ma_value):
                if current_price > ma_value:
                    mas_above.append(ma_period)
                    score += 2
                    if ma_period <= 20:  # 短期均线权重更高
                        score += 1
                else:
                    mas_below.append(ma_period)
                    score -= 2
                    if ma_period <= 20:
                        score -= 1

        # 均线多头排列
        if len(mas_above) >= 4 and mas_above == sorted(mas_above):
            buy_signals.append(
                {
                    "signal": f'均线多头排列(MA{",".join(map(str, mas_above[:4]))})',
                    "strength": 9,
                    "reason": "多条均线呈多头排列，趋势强劲",
                }
            )
            score += 12
        elif len(mas_below) >= 4 and mas_below == sorted(mas_below):
            sell_signals.append(
                {
                    "signal": f'均线空头排列(MA{",".join(map(str, mas_below[:4]))})',
                    "strength": 9,
                    "reason": "多条均线呈空头排列，趋势疲软",
                }
            )
            score -= 12

        # 均线交叉信号
        if "MA10" in factors_df.columns and "MA20" in factors_df.columns:
            ma10_latest = factors_df["MA10"].iloc[-1]
            ma20_latest = factors_df["MA20"].iloc[-1]
            ma10_prev = factors_df["MA10"].iloc[-2]
            ma20_prev = factors_df["MA20"].iloc[-2]

            if not (
                pd.isna(ma10_latest)
                or pd.isna(ma20_latest)
                or pd.isna(ma10_prev)
                or pd.isna(ma20_prev)
            ):
                # 金叉
                if ma10_latest > ma20_latest and ma10_prev <= ma20_prev:
                    buy_signals.append(
                        {
                            "signal": "MA10/MA20金叉",
                            "strength": 8,
                            "reason": "10日均线上穿20日均线，短期趋势转强",
                        }
                    )
                    score += 10
                # 死叉
                elif ma10_latest < ma20_latest and ma10_prev >= ma20_prev:
                    sell_signals.append(
                        {
                            "signal": "MA10/MA20死叉",
                            "strength": 8,
                            "reason": "10日均线下穿20日均线，短期趋势转弱",
                        }
                    )
                    score -= 10

        # EMA趋势分析
        ema_cols = [
            col
            for col in factors_df.columns
            if col.startswith("EMA") and col[3:].isdigit()
        ]
        if ema_cols:
            ema_above = 0
            for ema_col in ema_cols:
                ema_value = factors_df[ema_col].iloc[-1]
                if not pd.isna(ema_value) and current_price > ema_value:
                    ema_above += 1

            if ema_above >= len(ema_cols) * 0.7:  # 70%以上的EMA在价格下方
                buy_signals.append(
                    {
                        "signal": f"EMA多头支撑({ema_above}/{len(ema_cols)})",
                        "strength": 6,
                        "reason": "多数EMA支撑价格，上涨趋势确认",
                    }
                )
                score += 8
            elif ema_above <= len(ema_cols) * 0.3:  # 30%以下的EMA在价格下方
                sell_signals.append(
                    {
                        "signal": f"EMA空头压制({len(ema_cols)-ema_above}/{len(ema_cols)})",
                        "strength": 6,
                        "reason": "多数EMA压制价格，下跌趋势确认",
                    }
                )
                score -= 8

        # MACD趋势
        macd_cols = [
            col for col in factors_df.columns if "MACD" in col and "Signal" not in col
        ]
        if macd_cols:
            for macd_col in macd_cols:
                macd_value = factors_df[macd_col].iloc[-1]
                signal_col = macd_col.replace("MACD", "MACD_Signal")
                if signal_col in factors_df.columns:
                    signal_value = factors_df[signal_col].iloc[-1]
                    if not pd.isna(macd_value) and not pd.isna(signal_value):
                        if macd_value > signal_value > 0:
                            score += 6
                        elif macd_value < signal_value < 0:
                            score -= 6

        return score / 20, {"buy": buy_signals, "sell": sell_signals}  # 归一化到[-1, 1]

    def _analyze_momentum_signals(
        self, factors_df: pd.DataFrame, latest, prev_1, prev_3, prev_5
    ) -> tuple:
        """分析动量信号"""
        score = 0
        buy_signals = []
        sell_signals = []

        # RSI动量分析
        rsi_cols = [col for col in factors_df.columns if "RSI" in col]
        for rsi_col in rsi_cols:
            rsi_value = factors_df[rsi_col].iloc[-1]
            if not pd.isna(rsi_value):
                if rsi_value > 50:
                    score += 3
                    if rsi_value > 60:
                        buy_signals.append(
                            {
                                "signal": f"{rsi_col}强势({rsi_value:.1f})",
                                "strength": 5,
                                "reason": f"{rsi_col}显示强势动量",
                            }
                        )
                        score += 2
                else:
                    score -= 3
                    if rsi_value < 40:
                        sell_signals.append(
                            {
                                "signal": f"{rsi_col}弱势({rsi_value:.1f})",
                                "strength": 5,
                                "reason": f"{rsi_col}显示弱势动量",
                            }
                        )
                        score -= 2

        # Momentum指标分析
        momentum_cols = [col for col in factors_df.columns if "Momentum" in col]
        for mom_col in momentum_cols:
            mom_value = factors_df[mom_col].iloc[-1]
            if not pd.isna(mom_value):
                if mom_value > 0:
                    score += 2
                    if mom_value > 1:  # 假设Momentum单位是元
                        buy_signals.append(
                            {
                                "signal": f"{mom_col}正向({mom_value:.2f})",
                                "strength": 4,
                                "reason": f"{mom_col}显示上涨动量",
                            }
                        )
                        score += 2
                else:
                    score -= 2
                    if mom_value < -1:
                        sell_signals.append(
                            {
                                "signal": f"{mom_col}负向({mom_value:.2f})",
                                "strength": 4,
                                "reason": f"{mom_col}显示下跌动量",
                            }
                        )
                        score -= 2

        # CCI指标
        if "CCI14" in factors_df.columns:
            cci_value = factors_df["CCI14"].iloc[-1]
            if not pd.isna(cci_value):
                if cci_value > 100:
                    buy_signals.append(
                        {
                            "signal": f"CCI14超买({cci_value:.1f})",
                            "strength": 4,
                            "reason": "CCI显示强势超买状态",
                        }
                    )
                    score += 4
                elif cci_value < -100:
                    sell_signals.append(
                        {
                            "signal": f"CCI14超卖({cci_value:.1f})",
                            "strength": 4,
                            "reason": "CCI显示超卖状态",
                        }
                    )
                    score -= 4

        return score / 25, {"buy": buy_signals, "sell": sell_signals}  # 归一化到[-1, 1]

    def _analyze_oscillator_signals(
        self, factors_df: pd.DataFrame, latest, prev_1
    ) -> tuple:
        """分析超买超卖信号"""
        score = 0
        buy_signals = []
        sell_signals = []

        # Williams %R分析
        if "WILLR14" in factors_df.columns:
            willr_value = factors_df["WILLR14"].iloc[-1]
            if not pd.isna(willr_value):
                if willr_value < -80:  # 超卖
                    buy_signals.append(
                        {
                            "signal": f"Williams%R超卖({willr_value:.1f})",
                            "strength": 7,
                            "reason": "Williams%R低于-80，超卖反弹机会",
                        }
                    )
                    score += 8
                elif willr_value > -20:  # 超买
                    sell_signals.append(
                        {
                            "signal": f"Williams%R超买({willr_value:.1f})",
                            "strength": 7,
                            "reason": "Williams%R高于-20，超买回调风险",
                        }
                    )
                    score -= 8

        # 随机指标分析
        stoch_k_cols = [
            col for col in factors_df.columns if "STOCH" in col and "K" in col
        ]
        for stoch_k_col in stoch_k_cols:
            stoch_k_value = factors_df[stoch_k_col].iloc[-1]
            if not pd.isna(stoch_k_value):
                if stoch_k_value < 20:
                    buy_signals.append(
                        {
                            "signal": f"{stoch_k_col}超卖({stoch_k_value:.1f})",
                            "strength": 6,
                            "reason": f"{stoch_k_col}低于20，超卖区域",
                        }
                    )
                    score += 6
                elif stoch_k_value > 80:
                    sell_signals.append(
                        {
                            "signal": f"{stoch_k_col}超买({stoch_k_value:.1f})",
                            "strength": 6,
                            "reason": f"{stoch_k_col}高于80，超买区域",
                        }
                    )
                    score -= 6

        # 布林带分析
        bb_upper_cols = [col for col in factors_df.columns if "BB_Upper_20_2" in col]
        bb_lower_cols = [col for col in factors_df.columns if "BB_Lower_20_2" in col]

        if bb_upper_cols and bb_lower_cols:
            bb_upper = factors_df[bb_upper_cols[0]].iloc[-1]
            bb_lower = factors_df[bb_lower_cols[0]].iloc[-1]
            current_price = latest["close"]

            if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                # 突破上轨
                if current_price > bb_upper:
                    sell_signals.append(
                        {
                            "signal": "突破布林上轨",
                            "strength": 6,
                            "reason": "价格突破布林带上轨，超买回调风险",
                        }
                    )
                    score -= 7
                # 跌破下轨
                elif current_price < bb_lower:
                    buy_signals.append(
                        {
                            "signal": "跌破布林下轨",
                            "strength": 6,
                            "reason": "价格跌破布林带下轨，超卖反弹机会",
                        }
                    )
                    score += 7

        return score / 20, {"buy": buy_signals, "sell": sell_signals}  # 归一化到[-1, 1]

    def _analyze_volume_signals(
        self, df: pd.DataFrame, latest, prev_1, prev_5
    ) -> tuple:
        """分析成交量信号"""
        score = 0
        buy_signals = []
        sell_signals = []
        neutral_signals = []

        current_volume = latest["volume"]
        prev_volume = prev_1["volume"]
        avg_volume_5 = df["volume"].tail(5).mean()
        avg_volume_20 = df["volume"].tail(20).mean()

        # 成交量比率
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1

        # 放量分析
        if volume_ratio > 2.0:
            if latest["close"] > prev_1["close"]:
                buy_signals.append(
                    {
                        "signal": f"放量上涨({volume_ratio:.1f}倍)",
                        "strength": 7,
                        "reason": "成交量放大2倍以上且价格上涨，确认突破",
                    }
                )
                score += 8
            else:
                sell_signals.append(
                    {
                        "signal": f"放量下跌({volume_ratio:.1f}倍)",
                        "strength": 6,
                        "reason": "成交量放大但价格下跌，确认下跌",
                    }
                )
                score -= 7
        elif volume_ratio < 0.5:
            neutral_signals.append(
                {
                    "signal": f"缩量整理({volume_ratio:.1f}倍)",
                    "strength": 3,
                    "reason": "成交量萎缩，观望为主",
                }
            )

        # OBV分析
        obv_cols = [col for col in df.columns if "OBV" in col]
        if obv_cols:
            obv_value = df[obv_cols[0]].iloc[-1]
            obv_prev_5 = df[obv_cols[0]].iloc[-6] if len(df) >= 6 else obv_value

            if not pd.isna(obv_value) and not pd.isna(obv_prev_5):
                obv_trend = (
                    (obv_value - obv_prev_5) / abs(obv_prev_5) if obv_prev_5 != 0 else 0
                )
                price_trend = (latest["close"] - prev_5["close"]) / prev_5["close"]

                # 量价齐升
                if obv_trend > 0.02 and price_trend > 0.01:
                    buy_signals.append(
                        {
                            "signal": "OBV量价齐升",
                            "strength": 6,
                            "reason": "OBV和价格同步上升，健康上涨",
                        }
                    )
                    score += 7
                # 量价背离
                elif obv_trend < -0.02 and price_trend > 0.01:
                    sell_signals.append(
                        {
                            "signal": "OBV价量背离",
                            "strength": 7,
                            "reason": "价格上涨但OBV下降，警惕出货",
                        }
                    )
                    score -= 8

        return score / 15, {
            "buy": buy_signals,
            "sell": sell_signals,
            "neutral": neutral_signals,
        }  # 归一化到[-1, 1]

    def _analyze_volatility_signals(
        self, factors_df: pd.DataFrame, latest, prev_5
    ) -> tuple:
        """分析波动性信号"""
        score = 0
        neutral_signals = []

        # ATR分析
        atr_cols = [col for col in factors_df.columns if "ATR" in col]
        if atr_cols:
            atr_value = factors_df[atr_cols[0]].iloc[-1]
            current_price = latest["close"]

            if not pd.isna(atr_value) and current_price > 0:
                atr_pct = atr_value / current_price * 100

                if atr_pct > 5:  # 高波动
                    neutral_signals.append(
                        {
                            "signal": f"高波动风险(ATR {atr_pct:.1f}%)",
                            "strength": 4,
                            "reason": "波动性过高，谨慎操作",
                        }
                    )
                    score -= 3  # 高波动轻微扣分
                elif atr_pct < 1.5:  # 低波动
                    neutral_signals.append(
                        {
                            "signal": f"低波动机会(ATR {atr_pct:.1f}%)",
                            "strength": 3,
                            "reason": "波动性较低，适合建仓",
                        }
                    )
                    score += 2  # 低波动轻微加分

        return score / 10, {"neutral": neutral_signals}  # 归一化到[-1, 1]

    def calculate_trading_points(self, df: pd.DataFrame, signals: dict) -> dict:
        """计算精确交易点位"""
        print("正在计算300450交易点位...")

        current_price = signals["current_price"]
        overall_signal = signals["overall_signal"]
        signal_strength = signals["signal_strength"]

        trading_points = {
            "current_price": current_price,
            "signal": overall_signal,
            "strength": signal_strength,
            "entry_points": [],
            "stop_loss": None,
            "take_profits": [],
            "risk_management": {},
        }

        # 计算支撑位
        support_levels = self._calculate_support_levels(df)
        # 计算阻力位
        resistance_levels = self._calculate_resistance_levels(df)

        # 根据信号确定入场点
        if "买入" in overall_signal:
            if signal_strength in ["very_strong", "strong"]:
                trading_points["entry_points"].append(
                    {
                        "price": current_price,
                        "action": "立即买入",
                        "reason": f"{overall_signal}，信号{signal_strength}",
                        "position_size": "medium",
                        "urgency": "high",
                    }
                )
            else:
                # 等待小幅回调
                pullback_price = current_price * 0.98
                trading_points["entry_points"].append(
                    {
                        "price": pullback_price,
                        "action": "回调买入",
                        "reason": "信号中等强度，等待更优入场价格",
                        "position_size": "small",
                        "urgency": "medium",
                    }
                )

            # 止损位设置
            if support_levels:
                trading_points["stop_loss"] = support_levels[0]
            else:
                # 使用ATR设置止损
                atr_cols = [col for col in df.columns if "ATR14" in col]
                if atr_cols:
                    atr_value = df[atr_cols[0]].iloc[-1]
                    trading_points["stop_loss"] = current_price - 2 * atr_value
                else:
                    trading_points["stop_loss"] = current_price * 0.95

            # 止盈位设置
            if resistance_levels:
                take_profits = []
                for i, resistance in enumerate(resistance_levels[:4]):
                    profit_pct = (resistance / current_price - 1) * 100
                    take_profits.append(
                        {
                            "price": resistance,
                            "profit_pct": profit_pct,
                            "position": "1/4",
                            "priority": i + 1,
                        }
                    )
                trading_points["take_profits"] = take_profits
            else:
                # 默认止盈位
                default_profits = [1.03, 1.06, 1.10, 1.15]
                take_profits = []
                for i, multiplier in enumerate(default_profits):
                    profit_pct = (multiplier - 1) * 100
                    take_profits.append(
                        {
                            "price": current_price * multiplier,
                            "profit_pct": profit_pct,
                            "position": "1/4",
                            "priority": i + 1,
                        }
                    )
                trading_points["take_profits"] = take_profits

        elif "卖出" in overall_signal:
            # 卖出逻辑
            if signal_strength in ["very_strong", "strong"]:
                trading_points["entry_points"].append(
                    {
                        "price": current_price,
                        "action": "立即卖出",
                        "reason": f"{overall_signal}，信号{signal_strength}",
                        "position_size": "medium",
                        "urgency": "high",
                    }
                )

            # 止损位（回补点）
            if resistance_levels:
                trading_points["stop_loss"] = resistance_levels[0]
            else:
                trading_points["stop_loss"] = current_price * 1.05

        else:  # 观望
            trading_points["entry_points"].append(
                {
                    "price": current_price,
                    "action": "观望",
                    "reason": "信号不明确，等待方向选择",
                    "condition": f"突破{resistance_levels[0] if resistance_levels else current_price*1.05:.2f}或跌破{support_levels[0] if support_levels else current_price*0.95:.2f}",
                }
            )

        # 风险管理计算
        risk_management = self._calculate_risk_management(trading_points, signals)
        trading_points["risk_management"] = risk_management

        return trading_points

    def _calculate_support_levels(self, df: pd.DataFrame) -> list:
        """计算支撑位"""
        current_price = df["close"].iloc[-1]
        supports = []

        # 1. 近期重要低点
        for i in range(5, len(df) - 5):
            if df["low"].iloc[i] == df["low"].iloc[i - 5 : i + 5].min():
                if df["low"].iloc[i] < current_price:
                    supports.append(df["low"].iloc[i])

        # 2. 移动平均线支撑
        ma_cols = [
            col for col in df.columns if col.startswith("MA") and col[2:].isdigit()
        ]
        for ma_col in sorted(ma_cols, key=lambda x: int(x[2:]), reverse=True):
            ma_value = df[ma_col].iloc[-1]
            if not pd.isna(ma_value) and ma_value < current_price:
                supports.append(ma_value)

        # 3. 布林带下轨支撑
        bb_lower_cols = [col for col in df.columns if "BB_Lower_20_2" in col]
        for bb_col in bb_lower_cols:
            bb_value = df[bb_col].iloc[-1]
            if not pd.isna(bb_value) and bb_value < current_price:
                supports.append(bb_value)

        # 去重并排序（从高到低）
        supports = sorted(list(set(supports)), reverse=True)
        return supports[:6]

    def _calculate_resistance_levels(self, df: pd.DataFrame) -> list:
        """计算阻力位"""
        current_price = df["close"].iloc[-1]
        resistances = []

        # 1. 近期重要高点
        for i in range(5, len(df) - 5):
            if df["high"].iloc[i] == df["high"].iloc[i - 5 : i + 5].max():
                if df["high"].iloc[i] > current_price:
                    resistances.append(df["high"].iloc[i])

        # 2. 移动平均线阻力
        ma_cols = [
            col for col in df.columns if col.startswith("MA") and col[2:].isdigit()
        ]
        for ma_col in sorted(ma_cols, key=lambda x: int(x[2:])):
            ma_value = df[ma_col].iloc[-1]
            if not pd.isna(ma_value) and ma_value > current_price:
                resistances.append(ma_value)

        # 3. 布林带上轨阻力
        bb_upper_cols = [col for col in df.columns if "BB_Upper_20_2" in col]
        for bb_col in bb_upper_cols:
            bb_value = df[bb_col].iloc[-1]
            if not pd.isna(bb_value) and bb_value > current_price:
                resistances.append(bb_value)

        # 去重并排序（从低到高）
        resistances = sorted(list(set(resistances)))
        return resistances[:6]

    def _calculate_risk_management(self, trading_points: dict, signals: dict) -> dict:
        """计算风险管理参数"""
        current_price = trading_points["current_price"]
        stop_loss = trading_points["stop_loss"]
        take_profits = trading_points["take_profits"]

        risk_mgmt = {}

        if stop_loss:
            stop_loss_pct = (stop_loss / current_price - 1) * 100
            risk_mgmt["stop_loss_price"] = stop_loss
            risk_mgmt["stop_loss_pct"] = stop_loss_pct
            risk_mgmt["max_loss_per_trade"] = abs(stop_loss_pct)

            # 根据信号强度调整风险水平
            signal_strength = signals["signal_strength"]
            if signal_strength in ["very_strong", "strong"]:
                risk_mgmt["risk_level"] = "medium"
                risk_mgmt["recommended_position"] = "medium (15-25%)"
            elif signal_strength == "moderate":
                risk_mgmt["risk_level"] = "low"
                risk_mgmt["recommended_position"] = "small (10-15%)"
            else:
                risk_mgmt["risk_level"] = "very_low"
                risk_mgmt["recommended_position"] = "minimal (5-10%)"

        if take_profits:
            first_target = take_profits[0]
            if stop_loss:
                risk = abs(stop_loss_pct)
                reward = first_target["profit_pct"]
                risk_mgmt["risk_reward_ratio"] = reward / risk if risk > 0 else 0

        risk_mgmt["time_stop"] = (
            "20个交易日"
            if signals["signal_strength"] in ["weak", "very_weak"]
            else "40个交易日"
        )

        return risk_mgmt

    def generate_ultimate_report(
        self,
        df: pd.DataFrame,
        factors_df: pd.DataFrame,
        signals: dict,
        trading_points: dict,
    ) -> str:
        """生成300450终极分析报告"""
        print("正在生成300450终极分析报告...")

        current_price = signals["current_price"]
        overall_signal = signals["overall_signal"]
        signal_strength = signals["signal_strength"]

        # 计算价格表现
        change_1d = (
            (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
            if len(df) >= 2
            else 0
        )
        change_5d = (
            (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
            if len(df) >= 6
            else 0
        )
        change_20d = (
            (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100
            if len(df) >= 21
            else 0
        )

        # 波动率和风险指标
        returns = df["close"].pct_change().dropna()
        volatility_20 = returns.tail(20).std() * np.sqrt(252) * 100
        max_drawdown = (
            (df["close"].expanding().max() - df["close"])
            / df["close"].expanding().max()
        ).max() * 100

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                      300450 终极技术分析报告                               ║
║                    基于65个专业技术指标的精准分析                           ║
╚════════════════════════════════════════════════════════════════════════════════╝

📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
💰 当前价格: ¥{current_price:.2f}
🎯 综合信号: 【{overall_signal}】
📊 信号强度: {signal_strength}
📈 信号得分: {signals['overall_score']:+.1f}分 (满分100分)
🎲 买入信心: {signals['buy_confidence']:.1%} | 卖出信心: {signals['sell_confidence']:.1%}

📊 价格表现:
────────────────────────────────────────────────────────────────────────────
  近1日:   {change_1d:+7.2f}%
  近5日:   {change_5d:+7.2f}%
  近20日:  {change_20d:+7.2f}%
  年化波动率: {volatility_20:6.1f}%
  最大回撤:   {max_drawdown:6.1f}%

🎯 技术指标维度分析:
────────────────────────────────────────────────────────────────────────────
  趋势强度:     {signals['technical_summary']['trend_score']:+.2f}
  动量强度:     {signals['technical_summary']['momentum_score']:+.2f}
  超买超卖:     {signals['technical_summary']['oscillator_score']:+.2f}
  成交量信号:   {signals['technical_summary']['volume_score']:+.2f}
  波动性:       {signals['technical_summary']['volatility_score']:+.2f}"""

        # 显示买入信号
        if signals["buy_signals"]:
            report += f"""

🟢 买入信号 ({len(signals['buy_signals'])}个):
────────────────────────────────────────────────────────────────────────────"""
            for signal in signals["buy_signals"][:8]:
                report += f"""
  ✅ {signal['signal']} (强度: {signal['strength']}/10)
     理由: {signal['reason']}"""

        # 显示卖出信号
        if signals["sell_signals"]:
            report += f"""

🔴 卖出信号 ({len(signals['sell_signals'])}个):
────────────────────────────────────────────────────────────────────────────"""
            for signal in signals["sell_signals"][:8]:
                report += f"""
  ❌ {signal['signal']} (强度: {signal['strength']}/10)
     理由: {signal['reason']}"""

        # 显示中性信号
        if signals["neutral_signals"]:
            report += f"""

🟡 中性信号 ({len(signals['neutral_signals'])}个):
────────────────────────────────────────────────────────────────────────────"""
            for signal in signals["neutral_signals"][:3]:
                report += f"""
  ⚠️  {signal['signal']}
     理由: {signal['reason']}"""

        # 交易点位建议
        report += f"""

💼 300450交易执行建议:
────────────────────────────────────────────────────────────────────────────"""

        if trading_points["entry_points"]:
            for entry in trading_points["entry_points"]:
                report += f"""
  📍 {entry['action']}:
     入场价位: ¥{entry['price']:.2f}
     仓位大小: {entry.get('position_size', 'medium')}
     紧急程度: {entry.get('urgency', 'normal')}
     执行理由: {entry['reason']}"""
                if "condition" in entry:
                    report += f"""
     触发条件: {entry['condition']}"""

        # 风险管理
        risk_mgmt = trading_points["risk_management"]
        if risk_mgmt:
            report += f"""

🛡️ 风险管理:
────────────────────────────────────────────────────────────────────────────"""
            if "stop_loss_price" in risk_mgmt:
                report += f"""
  💸 止损价位: ¥{risk_mgmt['stop_loss_price']:.2f} ({risk_mgmt['stop_loss_pct']:+.2f}%)
  ⚠️ 最大风险: {risk_mgmt['max_loss_per_trade']:.2f}% 每笔交易
  📊 建议仓位: {risk_mgmt.get('recommended_position', 'medium')}
  ⏰ 时间止损: {risk_mgmt.get('time_stop', '40个交易日')}"""

            if "risk_reward_ratio" in risk_mgmt:
                report += f"""
  📈 风险回报比: 1:{risk_mgmt['risk_reward_ratio']:.2f}"""

        # 止盈策略
        if trading_points["take_profits"]:
            report += f"""

🎯 止盈策略:
────────────────────────────────────────────────────────────────────────────"""
            for tp in trading_points["take_profits"]:
                report += f"""
  🎯 目标{tp['priority']}: ¥{tp['price']:.2f} ({tp['profit_pct']:+.2f}%) - 减仓{tp['position']}"""

        # 关键技术位
        report += f"""

🎯 关键技术位:
────────────────────────────────────────────────────────────────────────────"""

        # 显示关键均线
        ma20 = (
            df[[col for col in df.columns if "MA20" in col]].iloc[-1].iloc[0]
            if any("MA20" in col for col in df.columns)
            else None
        )
        ma60 = (
            df[[col for col in df.columns if "MA60" in col]].iloc[-1].iloc[0]
            if any("MA60" in col for col in df.columns)
            else None
        )

        if ma20:
            ma20_signal = "🟢 站上" if current_price > ma20 else "🔴 跌破"
            report += f"""
  📍 MA20均线: ¥{ma20:.2f} ({ma20_signal})"""
        if ma60:
            ma60_signal = "🟢 站上" if current_price > ma60 else "🔴 跌破"
            report += f"""
  📍 MA60均线: ¥{ma60:.2f} ({ma60_signal})"""

        # 布林带信息
        bb_upper = (
            factors_df[[col for col in factors_df.columns if "BB_Upper_20_2" in col]]
            .iloc[-1]
            .iloc[0]
            if any("BB_Upper_20_2" in col for col in factors_df.columns)
            else None
        )
        bb_lower = (
            factors_df[[col for col in factors_df.columns if "BB_Lower_20_2" in col]]
            .iloc[-1]
            .iloc[0]
            if any("BB_Lower_20_2" in col for col in factors_df.columns)
            else None
        )

        if bb_upper and bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
            report += f"""
  📍 布林带: 上轨¥{bb_upper:.2f} 下轨¥{bb_lower:.2f}
  📍 当前位置: 布林带{bb_position:.1f}%位置"""

        # RSI值
        rsi14 = (
            factors_df[[col for col in factors_df.columns if "RSI14" in col]]
            .iloc[-1]
            .iloc[0]
            if any("RSI14" in col for col in factors_df.columns)
            else None
        )
        if rsi14:
            rsi_status = "超买" if rsi14 > 70 else "超卖" if rsi14 < 30 else "正常"
            report += f"""
  📍 RSI(14): {rsi14:.1f} ({rsi_status})"""

        # 执行总结
        report += f"""

💡 执行总结:
────────────────────────────────────────────────────────────────────────────"""

        if "买入" in overall_signal:
            report += f"""
  🟢 建议操作: 买入
  📊 依据: {len(signals['buy_signals'])}个买入信号 vs {len(signals['sell_signals'])}个卖出信号
  🎯 策略: {'立即建仓' if signal_strength in ['very_strong', 'strong'] else '分批建仓'}
  ⚠️ 风险: 严格执行止损¥{trading_points.get('stop_loss', 0):.2f}"""

        elif "卖出" in overall_signal:
            report += f"""
  🔴 建议操作: 卖出
  📊 依据: {len(signals['sell_signals'])}个卖出信号 vs {len(signals['buy_signals'])}个买入信号
  🎯 策略: {'立即减仓' if signal_strength in ['very_strong', 'strong'] else '分批减仓'}
  ⚠️ 风险: 控制仓位，防范进一步下跌"""

        else:
            report += f"""
  🟡 建议操作: 观望
  📊 依据: 多空信号不明确，等待方向选择
  🎯 策略: 保持耐心，等待明确信号
  ⚠️ 关注: 关键技术位突破"""

        report += f"""

⚠️ 重要风险提示:
────────────────────────────────────────────────────────────────────────────
  • 本分析基于{len(df)}个交易日数据和65个专业技术指标
  • 技术分析仅作为参考工具，不构成投资建议
  • 股市投资存在风险，请根据自身风险承受能力谨慎决策
  • 建议结合基本面分析、市场环境和资金面综合判断
  • 严格执行止损纪律是长期生存的关键
  • 分批建仓、分批止盈，避免情绪化交易
  • 密切关注市场情绪、政策变化和公司公告
  • 定期评估投资组合，及时调整策略

📊 分析工具: EnhancedFactorCalculator + 65专业指标
🔢 数据精度: 基于{len(df)}个交易日OHLCV数据
⏰ 分析时效: {datetime.now().strftime('%Y-%m-%d %H:%M')}，建议24小时内参考

═════════════════════════════════════════════════════════════════════════════════
                        300450终极分析完成
═════════════════════════════════════════════════════════════════════════════════"""

        print("300450终极报告生成完成")
        return report


def main():
    """主函数"""
    stock_code = "300450.SZ"

    print("🚀 开始300450终极技术分析...")
    print("📊 基于65个EnhancedFactorCalculator技术指标")

    try:
        # 初始化分析器
        analyzer = Ultimate300450Analyzer(stock_code)

        # 1. 加载数据
        df = analyzer.load_and_prepare_data()

        # 2. 使用EnhancedFactorCalculator计算指标
        factors_df = analyzer.calculate_factors_with_calculator(df)

        if factors_df is not None:
            # 3. 综合信号分析
            signals = analyzer.analyze_comprehensive_signals(df, factors_df)

            # 4. 计算交易点位
            trading_points = analyzer.calculate_trading_points(df, signals)

            # 5. 生成终极报告
            report = analyzer.generate_ultimate_report(
                df, factors_df, signals, trading_points
            )

            # 输出报告
            print(report)

            # 保存报告和数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存报告
            report_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_ultimate_analysis_{timestamp}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📝 300450终极报告已保存到: {report_file}")

            # 保存完整数据
            combined_df = df.join(factors_df, how="left")
            combined_file = (
                f"/Users/zhangshenshen/深度量化0927/a股/300450_factors_{timestamp}.csv"
            )
            combined_df.to_csv(combined_file)
            print(f"📊 300450技术指标数据已保存到: {combined_file}")

            # 保存交易数据
            trading_data = {
                "stock_code": stock_code,
                "analysis_time": datetime.now().isoformat(),
                "signals": signals,
                "trading_points": trading_points,
                "data_points": len(df),
                "indicators_count": len(factors_df.columns),
            }

            json_file = report_file.replace(".txt", "_trading_data.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(trading_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"📊 300450交易数据已保存到: {json_file}")

        else:
            print("❌ 无法计算技术指标")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
