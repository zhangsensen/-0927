#!/usr/bin/env python3
"""
300450 FactorEngine指标因子分析
结合FactorEngine进行多时间框架技术因子计算和综合分析
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 添加项目路径
import sys

sys.path.append("/Users/zhangshenshen/深度量化0927")

from factor_system.factor_engine import api
from factor_system.factor_engine.settings import get_research_config


class FactorEngine300450Analyzer:
    """300450 FactorEngine分析器"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.data_dir = "/Users/zhangshenshen/深度量化0927/a股"

        print(f"🔍 300450 FactorEngine分析器初始化")
        print(f"   股票代码: {stock_code}")
        print(f"   分析模式: FactorEngine指标因子 + 多时间框架")

    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并准备日线和小时线数据"""
        print("正在加载300450日线和小时线数据...")

        # 读取日线数据
        daily_file = (
            f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1d_2025-10-09.csv"
        )
        if not os.path.exists(daily_file):
            raise FileNotFoundError(f"日线数据文件不存在: {daily_file}")

        df_daily = pd.read_csv(daily_file, skiprows=1)
        df_daily.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        df_daily.set_index("Date", inplace=True)
        df_daily.columns = [col.lower() for col in df_daily.columns]

        # 读取小时线数据
        hourly_file = (
            f"{self.data_dir}/{self.stock_code}/{self.stock_code}_1h_2025-10-09.csv"
        )
        if not os.path.exists(hourly_file):
            raise FileNotFoundError(f"小时线数据文件不存在: {hourly_file}")

        df_hourly = pd.read_csv(hourly_file)
        df_hourly["Datetime"] = pd.to_datetime(df_hourly["Datetime"])
        df_hourly.set_index("Datetime", inplace=True)
        df_hourly.columns = [col.lower() for col in df_hourly.columns]

        print(f"✅ 日线数据加载完成: {len(df_daily)}条记录")
        print(f"✅ 小时线数据加载完成: {len(df_hourly)}条记录")

        return df_daily, df_hourly

    def calculate_factor_engine_indicators(
        self, df: pd.DataFrame, timeframe: str
    ) -> pd.DataFrame:
        """使用FactorEngine计算技术指标因子"""
        print(f"正在使用FactorEngine计算{timeframe}技术因子...")

        # 设置环境变量
        import os

        os.environ["FACTOR_ENGINE_RAW_DATA_DIR"] = (
            "/Users/zhangshenshen/深度量化0927/raw"
        )

        # 获取研究配置
        settings = get_research_config()

        # 准备数据格式 - FactorEngine需要标准格式
        factor_df = df.copy()

        # 确保数据格式正确
        factor_df.columns = [col.upper() for col in factor_df.columns]

        # 计算单个因子
        available_factors = api.list_available_factors()
        print(f"✅ 可用因子数量: {len(available_factors)}")

        # 计算关键因子 (使用FactorEngine实际可用的因子名称)
        key_factors = [
            "RSI14",
            "RSI10",
            "RSI7",
            "STOCH_14_20",
            "STOCH_10_14",
            "STOCH_7_10",
            "WILLR14",
            "WILLR9",
            "CCI14",
            "CCI10",
            "CCI20",
            "ATR14",
            "ATR10",
            "ATR7",
            "MSTD20",
            "MSTD15",
            "MSTD10",
            "FIXLB10",
            "FIXLB8",
            "FIXLB5",
        ]

        factor_results = {}
        success_count = 0

        for factor_name in key_factors:
            try:
                if factor_name in available_factors:
                    print(f"📊 计算因子: {factor_name}")
                    result = api.calculate_single_factor(
                        factor_id=factor_name,
                        symbol=self.stock_code,
                        timeframe=timeframe,
                        start_date=df.index[0],
                        end_date=df.index[-1],
                        data=factor_df,
                    )

                    if result is not None and not result.empty:
                        factor_results[factor_name] = result
                        success_count += 1
                        print(f"✅ {factor_name} 计算成功")
                    else:
                        print(f"❌ {factor_name} 计算结果为空")
                else:
                    print(f"⚠️ {factor_name} 不可用")
            except Exception as e:
                print(f"❌ {factor_name} 计算失败: {str(e)[:50]}...")

        print(f"✅ 成功计算因子: {success_count}/{len(key_factors)}")

        if factor_results:
            # 合并所有因子结果
            combined_factors = pd.concat(factor_results.values(), axis=1, join="outer")

            # 添加原始价格数据
            combined_factors["CLOSE"] = factor_df["CLOSE"]
            combined_factors["HIGH"] = factor_df["HIGH"]
            combined_factors["LOW"] = factor_df["LOW"]
            combined_factors["OPEN"] = factor_df["OPEN"]
            combined_factors["VOLUME"] = factor_df["VOLUME"]

            return combined_factors
        else:
            raise ValueError("没有成功计算任何因子")

    def analyze_factor_signals(
        self, daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame
    ) -> Dict:
        """分析因子信号"""
        print("正在分析因子信号...")

        signals = {
            "daily_signals": {},
            "hourly_signals": {},
            "combined_signals": {},
            "factor_alignment": {},
            "signal_strength": 0,
            "confidence_level": "medium",
        }

        # 分析日线因子信号
        signals["daily_signals"] = self.analyze_single_timeframe_factors(
            daily_factors, "日线"
        )

        # 分析小时线因子信号
        signals["hourly_signals"] = self.analyze_single_timeframe_factors(
            hourly_factors, "小时线"
        )

        # 分析因子对齐情况
        signals["factor_alignment"] = self.analyze_factor_alignment(
            daily_factors, hourly_factors
        )

        # 生成综合信号
        signals["combined_signals"] = self.generate_combined_signals(signals)

        # 计算信号强度
        signals["signal_strength"] = self.calculate_overall_signal_strength(signals)
        signals["confidence_level"] = self.determine_confidence_level(signals)

        return signals

    def analyze_single_timeframe_factors(
        self, factors_df: pd.DataFrame, timeframe_name: str
    ) -> Dict:
        """分析单个时间框架的因子信号"""
        signals = {
            "trend_signals": [],
            "momentum_signals": [],
            "volatility_signals": [],
            "volume_signals": [],
            "pattern_signals": [],
            "overall_score": 0,
        }

        try:
            latest_data = factors_df.iloc[-1]
            prev_data = factors_df.iloc[-2] if len(factors_df) >= 2 else latest_data

            # 趋势信号分析
            if "CLOSE" in latest_data and "SMA_20" in latest_data:
                if latest_data["CLOSE"] > latest_data.get("SMA_20", 0):
                    signals["trend_signals"].append(f"{timeframe_name}价格位于MA20上方")
                    signals["overall_score"] += 1
                else:
                    signals["trend_signals"].append(f"{timeframe_name}价格位于MA20下方")
                    signals["overall_score"] -= 1

            # 动量信号分析
            if "RSI_14" in latest_data:
                rsi_value = latest_data["RSI_14"]
                if rsi_value > 70:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}RSI超买({rsi_value:.1f})"
                    )
                    signals["overall_score"] -= 0.5
                elif rsi_value < 30:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}RSI超卖({rsi_value:.1f})"
                    )
                    signals["overall_score"] += 0.5
                elif rsi_value > 50:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}RSI强势({rsi_value:.1f})"
                    )
                    signals["overall_score"] += 0.3
                else:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}RSI弱势({rsi_value:.1f})"
                    )
                    signals["overall_score"] -= 0.3

            # MACD信号分析
            if "MACD_12_26_9" in latest_data and "MACD_12_26_9_SIGNAL" in latest_data:
                macd = latest_data["MACD_12_26_9"]
                signal = latest_data["MACD_12_26_9_SIGNAL"]
                if macd > signal and macd > 0:
                    signals["momentum_signals"].append(f"{timeframe_name}MACD多头排列")
                    signals["overall_score"] += 1
                elif macd < signal and macd < 0:
                    signals["momentum_signals"].append(f"{timeframe_name}MACD空头排列")
                    signals["overall_score"] -= 1
                elif macd > signal:
                    signals["momentum_signals"].append(f"{timeframe_name}MACD金叉")
                    signals["overall_score"] += 0.5
                else:
                    signals["momentum_signals"].append(f"{timeframe_name}MACD死叉")
                    signals["overall_score"] -= 0.5

            # 随机指标信号分析
            if (
                "STOCH_14_3_3_SLOWK" in latest_data
                and "STOCH_14_3_3_SLOWD" in latest_data
            ):
                k = latest_data["STOCH_14_3_3_SLOWK"]
                d = latest_data["STOCH_14_3_3_SLOWD"]
                if k > d and k < 80:
                    signals["momentum_signals"].append(f"{timeframe_name}KDJ金叉")
                    signals["overall_score"] += 0.5
                elif k < d and k > 20:
                    signals["momentum_signals"].append(f"{timeframe_name}KDJ死叉")
                    signals["overall_score"] -= 0.5
                elif k > 80:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}KDJ超买({k:.1f})"
                    )
                    signals["overall_score"] -= 0.3
                elif k < 20:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}KDJ超卖({k:.1f})"
                    )
                    signals["overall_score"] += 0.3

            # 威廉指标信号分析
            if "WILLR_14" in latest_data:
                willr = latest_data["WILLR_14"]
                if willr > -20:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}Williams超买({willr:.1f})"
                    )
                    signals["overall_score"] -= 0.3
                elif willr < -80:
                    signals["momentum_signals"].append(
                        f"{timeframe_name}Williams超卖({willr:.1f})"
                    )
                    signals["overall_score"] += 0.3

            # 布林带信号分析
            if (
                "BBANDS_20_2_UPPER" in latest_data
                and "BBANDS_20_2_LOWER" in latest_data
            ):
                upper = latest_data["BBANDS_20_2_UPPER"]
                lower = latest_data["BBANDS_20_2_LOWER"]
                close = latest_data["CLOSE"]

                if close > upper:
                    signals["volatility_signals"].append(
                        f"{timeframe_name}突破布林上轨"
                    )
                    signals["overall_score"] -= 0.5
                elif close < lower:
                    signals["volatility_signals"].append(
                        f"{timeframe_name}跌破布林下轨"
                    )
                    signals["overall_score"] += 0.5
                else:
                    signals["volatility_signals"].append(
                        f"{timeframe_name}价格在布林带内"
                    )

            # ADX趋势强度分析
            if "ADX_14" in latest_data:
                adx = latest_data["ADX_14"]
                if adx > 25:
                    signals["trend_signals"].append(
                        f"{timeframe_name}ADX强趋势({adx:.1f})"
                    )
                elif adx > 20:
                    signals["trend_signals"].append(
                        f"{timeframe_name}ADX中等趋势({adx:.1f})"
                    )
                else:
                    signals["trend_signals"].append(
                        f"{timeframe_name}ADX弱趋势({adx:.1f})"
                    )

            # 成交量信号分析（如果有成交量数据）
            if "VOLUME" in latest_data:
                current_volume = latest_data["VOLUME"]
                if len(factors_df) >= 20:
                    avg_volume = factors_df["VOLUME"].tail(20).mean()
                    volume_ratio = current_volume / avg_volume

                    if volume_ratio > 2.0:
                        signals["volume_signals"].append(
                            f"{timeframe_name}成交量显著放大({volume_ratio:.1f}倍)"
                        )
                        signals["overall_score"] += 0.5
                    elif volume_ratio > 1.5:
                        signals["volume_signals"].append(
                            f"{timeframe_name}成交量放大({volume_ratio:.1f}倍)"
                        )
                        signals["overall_score"] += 0.3
                    elif volume_ratio < 0.5:
                        signals["volume_signals"].append(
                            f"{timeframe_name}成交量萎缩({volume_ratio:.1f}倍)"
                        )

        except Exception as e:
            print(f"❌ {timeframe_name}因子分析失败: {e}")

        return signals

    def analyze_factor_alignment(
        self, daily_factors: pd.DataFrame, hourly_factors: pd.DataFrame
    ) -> Dict:
        """分析因子对齐情况"""
        alignment = {
            "trend_alignment": "unknown",
            "momentum_alignment": "unknown",
            "volatility_alignment": "unknown",
            "overall_alignment_score": 0,
            "aligned_factors": [],
            "conflicting_factors": [],
        }

        try:
            # 比较关键因子对齐情况
            daily_latest = daily_factors.iloc[-1]
            hourly_latest = hourly_factors.iloc[-1]

            # RSI对齐
            if "RSI_14" in daily_latest and "RSI_14" in hourly_latest:
                daily_rsi = daily_latest["RSI_14"]
                hourly_rsi = hourly_latest["RSI_14"]

                if (daily_rsi > 50 and hourly_rsi > 50) or (
                    daily_rsi < 50 and hourly_rsi < 50
                ):
                    alignment["aligned_factors"].append(
                        f"RSI一致(日线:{daily_rsi:.1f}, 小时线:{hourly_rsi:.1f})"
                    )
                    alignment["overall_alignment_score"] += 1
                else:
                    alignment["conflicting_factors"].append(
                        f"RSI分歧(日线:{daily_rsi:.1f}, 小时线:{hourly_rsi:.1f})"
                    )

            # MACD对齐
            if "MACD_12_26_9" in daily_latest and "MACD_12_26_9" in hourly_latest:
                daily_macd = daily_latest["MACD_12_26_9"]
                hourly_macd = hourly_latest["MACD_12_26_9"]

                if (daily_macd > 0 and hourly_macd > 0) or (
                    daily_macd < 0 and hourly_macd < 0
                ):
                    alignment["aligned_factors"].append(
                        f"MACD一致(日线:{daily_macd:+.4f}, 小时线:{hourly_macd:+.4f})"
                    )
                    alignment["overall_alignment_score"] += 1
                else:
                    alignment["conflicting_factors"].append(
                        f"MACD分歧(日线:{daily_macd:+.4f}, 小时线:{hourly_macd:+.4f})"
                    )

            # 趋势对齐
            daily_above_ma20 = (
                "SMA_20" in daily_latest
                and daily_latest["CLOSE"] > daily_latest["SMA_20"]
            )
            hourly_above_ma20 = (
                "SMA_20" in hourly_latest
                and hourly_latest["CLOSE"] > hourly_latest["SMA_20"]
            )

            if daily_above_ma20 == hourly_above_ma20:
                trend_dir = "上方" if daily_above_ma20 else "下方"
                alignment["aligned_factors"].append(f"趋势一致(均位于MA20{trend_dir})")
                alignment["overall_alignment_score"] += 1
            else:
                alignment["conflicting_factors"].append(f"趋势分歧(日线vs小时线)")

        except Exception as e:
            print(f"❌ 因子对齐分析失败: {e}")

        return alignment

    def generate_combined_signals(self, signals: Dict) -> Dict:
        """生成综合信号"""
        combined = {
            "bullish_signals": [],
            "bearish_signals": [],
            "neutral_signals": [],
            "key_insights": [],
        }

        # 收集所有看涨信号
        for signal_type in ["trend_signals", "momentum_signals", "volume_signals"]:
            daily_signals = signals["daily_signals"].get(signal_type, [])
            hourly_signals = signals["hourly_signals"].get(signal_type, [])

            for signal in daily_signals + hourly_signals:
                if any(
                    keyword in signal
                    for keyword in ["上方", "强势", "金叉", "放大", "突破"]
                ):
                    combined["bullish_signals"].append(signal)
                elif any(
                    keyword in signal
                    for keyword in ["下方", "弱势", "死叉", "萎缩", "跌破"]
                ):
                    combined["bearish_signals"].append(signal)
                else:
                    combined["neutral_signals"].append(signal)

        # 添加因子对齐信号
        alignment = signals["factor_alignment"]
        if alignment["overall_alignment_score"] >= 2:
            combined["bullish_signals"].append("多因子高度一致")
        elif alignment["overall_alignment_score"] <= -1:
            combined["bearish_signals"].append("多因子分歧严重")

        return combined

    def calculate_overall_signal_strength(self, signals: Dict) -> float:
        """计算整体信号强度"""
        daily_score = signals["daily_signals"].get("overall_score", 0)
        hourly_score = signals["hourly_signals"].get("overall_score", 0)
        alignment_score = signals["factor_alignment"].get("overall_alignment_score", 0)

        # 加权计算
        total_strength = daily_score * 0.4 + hourly_score * 0.4 + alignment_score * 0.2

        return total_strength

    def determine_confidence_level(self, signals: Dict) -> str:
        """确定置信度"""
        strength = signals["signal_strength"]
        alignment_score = signals["factor_alignment"].get("overall_alignment_score", 0)

        if abs(strength) >= 2 and alignment_score >= 1:
            return "high"
        elif abs(strength) >= 1:
            return "medium"
        else:
            return "low"

    def generate_comprehensive_analysis(
        self,
        df_daily: pd.DataFrame,
        df_hourly: pd.DataFrame,
        daily_factors: pd.DataFrame,
        hourly_factors: pd.DataFrame,
        signals: Dict,
    ) -> Dict:
        """生成综合分析"""
        print("正在生成FactorEngine综合分析...")

        current_price = (
            df_hourly["close"].iloc[-1]
            if "close" in df_hourly.columns
            else df_hourly["CLOSE"].iloc[-1]
        )

        # 基于信号强度确定推荐
        strength = signals["signal_strength"]
        confidence = signals["confidence_level"]

        if strength >= 2 and confidence in ["high", "medium"]:
            recommendation = "strong_buy"
        elif strength >= 1:
            recommendation = "buy"
        elif strength <= -2 and confidence in ["high", "medium"]:
            recommendation = "strong_sell"
        elif strength <= -1:
            recommendation = "sell"
        else:
            recommendation = "hold"

        # 计算技术位
        technical_levels = self.calculate_technical_levels(df_daily, daily_factors)

        return {
            "recommendation": recommendation,
            "current_price": current_price,
            "signal_strength": strength,
            "confidence_level": confidence,
            "signals": signals,
            "technical_levels": technical_levels,
            "factor_summary": self.generate_factor_summary(signals),
        }

    def calculate_technical_levels(
        self, df: pd.DataFrame, factors: pd.DataFrame
    ) -> Dict:
        """计算技术位"""
        try:
            current_price = (
                df["close"].iloc[-1] if "close" in df.columns else df["CLOSE"].iloc[-1]
            )

            # 支撑阻力位
            recent_highs = (
                df["high"].tail(20).nlargest(5)
                if "high" in df.columns
                else factors["HIGH"].tail(20).nlargest(5)
            )
            recent_lows = (
                df["low"].tail(20).nsmallest(5)
                if "low" in df.columns
                else factors["LOW"].tail(20).nsmallest(5)
            )

            # ATR计算
            if len(df) >= 14:
                high_low = (
                    (df["high"] - df["low"])
                    if "high" in df.columns
                    else (factors["HIGH"] - factors["LOW"])
                )
                atr = high_low.rolling(14).mean().iloc[-1]
            else:
                atr = current_price * 0.02

            return {
                "support_levels": recent_lows.head(3).tolist(),
                "resistance_levels": recent_highs.head(3).tolist(),
                "atr": atr,
                "current_price": current_price,
            }
        except Exception as e:
            print(f"❌ 技术位计算失败: {e}")
            return {
                "support_levels": [current_price * 0.95],
                "resistance_levels": [current_price * 1.05],
                "atr": current_price * 0.02,
                "current_price": current_price,
            }

    def generate_factor_summary(self, signals: Dict) -> Dict:
        """生成因子摘要"""
        return {
            "total_aligned_factors": len(
                signals["factor_alignment"]["aligned_factors"]
            ),
            "total_conflicting_factors": len(
                signals["factor_alignment"]["conflicting_factors"]
            ),
            "alignment_score": signals["factor_alignment"]["overall_alignment_score"],
            "key_aligned_factors": signals["factor_alignment"]["aligned_factors"][:3],
            "key_conflicting_factors": signals["factor_alignment"][
                "conflicting_factors"
            ][:3],
        }

    def generate_factor_engine_report(
        self,
        df_daily: pd.DataFrame,
        df_hourly: pd.DataFrame,
        daily_factors: pd.DataFrame,
        hourly_factors: pd.DataFrame,
        analysis: Dict,
    ) -> str:
        """生成FactorEngine分析报告"""
        print("正在生成FactorEngine综合分析报告...")

        current_price = analysis["current_price"]
        recommendation = analysis["recommendation"]
        strength = analysis["signal_strength"]
        confidence = analysis["confidence_level"]
        signals = analysis["signals"]
        levels = analysis["technical_levels"]
        factor_summary = analysis["factor_summary"]

        # 计算价格表现
        daily_change_1d = (
            (df_daily["close"].iloc[-1] / df_daily["close"].iloc[-2] - 1) * 100
            if len(df_daily) >= 2
            else 0
        )
        hourly_change_1h = (
            (df_hourly["close"].iloc[-1] / df_hourly["close"].iloc[-2] - 1) * 100
            if len(df_hourly) >= 2
            else 0
        )

        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                300450 FactorEngine指标因子分析报告                           ║
║                    多时间框架因子交叉验证与综合分析                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📅 分析时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
💰 当前价格: ¥{current_price:.2f}
🎯 FactorEngine评级: {self.get_recommendation_text(recommendation)}
📊 信号强度: {strength:+.1f}
🔍 置信度: {confidence.upper()}
🧮 因子对齐度: {factor_summary['alignment_score']:+.1f}

📈 价格表现:
────────────────────────────────────────────────────────────────────────────
  日线近1日:   {daily_change_1d:+7.2f}%
  小时线近1小时: {hourly_change_1h:+7.2f}%

🧩 FactorEngine因子分析:
────────────────────────────────────────────────────────────────────────────

📊 因子对齐情况:
  对齐因子数: {factor_summary['total_aligned_factors']}
  冲突因子数: {factor_summary['total_conflicting_factors']}
  对齐得分: {factor_summary['alignment_score']:+.1f}

"""

        # 添加关键对齐因子
        if factor_summary["key_aligned_factors"]:
            report += "✅ 主要对齐因子:\n"
            for factor in factor_summary["key_aligned_factors"]:
                report += f"  ✓ {factor}\n"

        # 添加关键冲突因子
        if factor_summary["key_conflicting_factors"]:
            report += "\n⚠️ 主要冲突因子:\n"
            for factor in factor_summary["key_conflicting_factors"]:
                report += f"  • {factor}\n"

        report += f"""
📈 日线因子信号分析:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加日线信号
        for signal_type, signal_list in signals["daily_signals"].items():
            if signal_list and signal_type != "overall_score":
                report += f"\n{self.get_signal_type_name(signal_type)}:\n"
                for signal in signal_list:
                    report += f"  {signal}\n"

        report += f"""
📊 小时线因子信号分析:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加小时线信号
        for signal_type, signal_list in signals["hourly_signals"].items():
            if signal_list and signal_type != "overall_score":
                report += f"\n{self.get_signal_type_name(signal_type)}:\n"
                for signal in signal_list:
                    report += f"  {signal}\n"

        report += f"""
🎯 综合信号汇总:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加看涨信号
        if signals["combined_signals"]["bullish_signals"]:
            report += "\n🟢 看涨信号:\n"
            for signal in signals["combined_signals"]["bullish_signals"]:
                report += f"  ✓ {signal}\n"

        # 添加看跌信号
        if signals["combined_signals"]["bearish_signals"]:
            report += "\n🔴 看跌信号:\n"
            for signal in signals["combined_signals"]["bearish_signals"]:
                report += f"  ✗ {signal}\n"

        # 添加中性信号
        if signals["combined_signals"]["neutral_signals"]:
            report += "\n🟡 中性信号:\n"
            for signal in signals["combined_signals"]["neutral_signals"]:
                report += f"  • {signal}\n"

        report += f"""
📊 关键技术位 (FactorEngine计算):
────────────────────────────────────────────────────────────────────────────
支撑位: {', '.join([f'¥{level:.2f}' for level in levels['support_levels'][:3]])}
阻力位: {', '.join([f'¥{level:.2f}' for level in levels['resistance_levels'][:3]])}
ATR: ¥{levels['atr']:.2f} ({(levels['atr']/current_price)*100:.1f}%)

💎 FactorEngine操作建议:
────────────────────────────────────────────────────────────────────────────
"""

        # 添加操作建议
        if recommendation in ["strong_buy", "buy"]:
            report += f"""
🟢 建议买入: {self.get_recommendation_text(recommendation)}

📍 FactorEngine入场策略:
  • 当前价位¥{current_price:.2f}可考虑建仓
  • 基于多因子对齐分析，信号{confidence}
  • 建议仓位10-15%，控制风险

🛡️ 因子风险控制:
  • 止损位: ¥{current_price - levels['atr']*2:.2f} (-{(levels['atr']*2/current_price)*100:.1f}%)
  • 密切关注因子对齐变化
  • 小时线因子转向时及时调整

🎯 盈利目标:
  • 第一目标: ¥{current_price + levels['atr']*3:.2f} (+{(levels['atr']*3/current_price)*100:+.1f}%)
  • 第二目标: ¥{current_price + levels['atr']*5:.2f} (+{(levels['atr']*5/current_price)*100:+.1f}%)
"""
        elif recommendation in ["strong_sell", "sell"]:
            report += f"""
🔴 建议卖出: {self.get_recommendation_text(recommendation)}

📍 FactorEngine出场策略:
  • 当前价位¥{current_price:.2f}建议减仓
  • 多因子显示负面信号
  • 建议分批卖出

🎯 目标价位:
  • 第一目标: ¥{current_price - levels['atr']*2:.2f} (-{(levels['atr']*2/current_price)*100:.1f}%)
  • 第二目标: ¥{current_price - levels['atr']*4:.2f} (-{(levels['atr']*4/current_price)*100:.1f}%)
"""
        else:
            report += f"""
🟡 建议观望: {self.get_recommendation_text(recommendation)}

📍 FactorEngine观望策略:
  • 因子信号不一致，等待明确方向
  • 关注因子对齐度变化
  • 可考虑小仓位试探

⏰ 观察要点:
  • 多时间框架因子对齐改善
  • 关键因子突破临界值
  • 成交量配合因子信号
"""

        report += f"""
🧮 FactorEngine技术要点:
────────────────────────────────────────────────────────────────────────────
1. 因子对齐度是关键指标，≥1表示一致性较好
2. 多时间框架因子交叉验证提高信号可靠性
3. 结合量价因子确认突破有效性
4. 动量因子领先趋势因子变化

⚠️ FactorEngine风险提示:
────────────────────────────────────────────────────────────────────────────
• 因子分析基于历史数据，未来可能发生变化
• 创业板股票波动较大，请严格止损
• 建议与其他分析方法结合使用
• 密切关注因子对齐度的实时变化

═════════════════════════════════════════════════════════════════════════════════
                      300450 FactorEngine分析完成
═════════════════════════════════════════════════════════════════════════════════
"""

        return report

    def get_recommendation_text(self, recommendation: str) -> str:
        """获取推荐文本"""
        mapping = {
            "strong_buy": "强烈买入",
            "buy": "买入",
            "hold": "持有观望",
            "sell": "卖出",
            "strong_sell": "强烈卖出",
        }
        return mapping.get(recommendation, "观望")

    def get_signal_type_name(self, signal_type: str) -> str:
        """获取信号类型名称"""
        mapping = {
            "trend_signals": "趋势因子",
            "momentum_signals": "动量因子",
            "volatility_signals": "波动率因子",
            "volume_signals": "成交量因子",
            "pattern_signals": "形态因子",
        }
        return mapping.get(signal_type, signal_type)


def main():
    """主函数"""
    print("🔍 开始300450 FactorEngine指标因子分析...")
    print("💎 多时间框架因子交叉验证分析")

    analyzer = FactorEngine300450Analyzer("300450.SZ")

    try:
        # 加载数据
        df_daily, df_hourly = analyzer.load_and_prepare_data()

        # 使用FactorEngine计算指标因子
        daily_factors = analyzer.calculate_factor_engine_indicators(df_daily, "daily")
        hourly_factors = analyzer.calculate_factor_engine_indicators(df_hourly, "1h")

        # 分析因子信号
        signals = analyzer.analyze_factor_signals(daily_factors, hourly_factors)

        # 生成综合分析
        analysis = analyzer.generate_comprehensive_analysis(
            df_daily, df_hourly, daily_factors, hourly_factors, signals
        )

        # 生成报告
        report = analyzer.generate_factor_engine_report(
            df_daily, df_hourly, daily_factors, hourly_factors, analysis
        )

        # 输出报告
        print(report)

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_factor_engine_analysis_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n📄 FactorEngine报告已保存: {report_file}")

        # 保存分析数据
        data_file = f"/Users/zhangshenshen/深度量化0927/a股/300450_factor_engine_data_{timestamp}.json"
        import json

        analysis_data = {
            "analysis_time": datetime.now().isoformat(),
            "factor_engine_analysis": True,
            "daily_data_shape": df_daily.shape,
            "hourly_data_shape": df_hourly.shape,
            "daily_factors_shape": daily_factors.shape,
            "hourly_factors_shape": hourly_factors.shape,
            "signal_strength": analysis["signal_strength"],
            "confidence_level": analysis["confidence_level"],
            "recommendation": analysis["recommendation"],
            "factor_summary": analysis["factor_summary"],
        }

        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"🧮 FactorEngine数据已保存: {data_file}")

    except Exception as e:
        print(f"❌ FactorEngine分析失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
