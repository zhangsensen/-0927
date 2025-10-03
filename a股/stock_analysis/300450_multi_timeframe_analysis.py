#!/usr/bin/env python3
"""
300450.SZ 先导智能多时间框架分析
结合日线和小时线的综合技术分析
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def load_stock_data(daily_file, hourly_file):
    """加载日线和小时线数据"""
    print("正在加载多时间框架数据...")

    # 加载日线数据
    daily_df = pd.read_csv(daily_file, skiprows=2)
    daily_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df.set_index("Date", inplace=True)
    daily_df.sort_index(inplace=True)

    # 加载小时线数据
    hourly_df = pd.read_csv(hourly_file, skiprows=2)
    hourly_df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    hourly_df["Date"] = pd.to_datetime(hourly_df["Date"])
    hourly_df.set_index("Date", inplace=True)
    hourly_df.sort_index(inplace=True)

    # 确保数值类型正确
    for df in [daily_df, hourly_df]:
        for col in ["Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(
        f"日线数据: {len(daily_df)} 条记录 ({daily_df.index.min()} 到 {daily_df.index.max()})"
    )
    print(
        f"小时线数据: {len(hourly_df)} 条记录 ({hourly_df.index.min()} 到 {hourly_df.index.max()})"
    )
    print(f"最新价格: {daily_df['Close'].iloc[-1]:.2f}")

    return daily_df, hourly_df


def calculate_rsi(prices, period=14):
    """计算RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(df):
    """计算MACD"""
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_bollinger_bands(df, period=20):
    """计算布林带"""
    bb_middle = df["Close"].rolling(window=period).mean()
    bb_std = df["Close"].rolling(window=period).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    return bb_upper, bb_middle, bb_lower


def calculate_technical_indicators(daily_df, hourly_df):
    """计算多时间框架技术指标"""
    print("正在计算多时间框架技术指标...")

    # 日线指标
    daily_df["MA5"] = daily_df["Close"].rolling(window=5).mean()
    daily_df["MA10"] = daily_df["Close"].rolling(window=10).mean()
    daily_df["MA20"] = daily_df["Close"].rolling(window=20).mean()
    daily_df["MA30"] = daily_df["Close"].rolling(window=30).mean()
    daily_df["MA60"] = daily_df["Close"].rolling(window=60).mean()
    daily_df["RSI"] = calculate_rsi(daily_df["Close"])
    daily_df["MACD"], daily_df["Signal"], daily_df["MACD_Hist"] = calculate_macd(
        daily_df
    )
    daily_df["BB_Upper"], daily_df["BB_Middle"], daily_df["BB_Lower"] = (
        calculate_bollinger_bands(daily_df)
    )

    # 小时线指标
    hourly_df["MA20"] = hourly_df["Close"].rolling(window=20).mean()  # 20小时均线
    hourly_df["MA60"] = hourly_df["Close"].rolling(window=60).mean()  # 60小时均线
    hourly_df["RSI"] = calculate_rsi(hourly_df["Close"])
    hourly_df["MACD"], hourly_df["Signal"], hourly_df["MACD_Hist"] = calculate_macd(
        hourly_df
    )
    hourly_df["BB_Upper"], hourly_df["BB_Middle"], hourly_df["BB_Lower"] = (
        calculate_bollinger_bands(hourly_df, period=50)
    )

    # 成交量指标
    daily_df["Volume_MA5"] = daily_df["Volume"].rolling(window=5).mean()
    daily_df["Volume_MA20"] = daily_df["Volume"].rolling(window=20).mean()
    hourly_df["Volume_MA20"] = hourly_df["Volume"].rolling(window=20).mean()

    print("技术指标计算完成")
    return daily_df, hourly_df


def analyze_timeframe_alignment(daily_df, hourly_df):
    """分析多时间框架对齐情况"""
    latest_daily = daily_df.iloc[-1]
    latest_hourly = hourly_df.iloc[-1]

    # 日线状态
    daily_macd_state = (
        "金叉" if latest_daily["MACD"] > latest_daily["Signal"] else "死叉"
    )
    daily_rsi_state = (
        "超买"
        if latest_daily["RSI"] > 70
        else "超卖" if latest_daily["RSI"] < 30 else "正常"
    )
    daily_ma_trend = "多头" if latest_daily["MA5"] > latest_daily["MA20"] else "空头"

    # 小时线状态
    hourly_macd_state = (
        "金叉" if latest_hourly["MACD"] > latest_hourly["Signal"] else "死叉"
    )
    hourly_rsi_state = (
        "超买"
        if latest_hourly["RSI"] > 70
        else "超卖" if latest_hourly["RSI"] < 30 else "正常"
    )
    hourly_ma_trend = (
        "多头" if latest_hourly["MA20"] > latest_hourly["MA60"] else "空头"
    )

    # 时间框架对齐评分
    alignment_score = 0
    alignment_signals = []

    if daily_macd_state == hourly_macd_state:
        alignment_score += 2
        alignment_signals.append(f"MACD时间框架一致({daily_macd_state})")
    else:
        alignment_signals.append(
            f"MACD时间框架分歧(日线{daily_macd_state}, 小时线{hourly_macd_state})"
        )

    if daily_ma_trend == hourly_ma_trend:
        alignment_score += 2
        alignment_signals.append(f"均线趋势一致({daily_ma_trend})")
    else:
        alignment_signals.append(
            f"均线趋势分歧(日线{daily_ma_trend}, 小时线{hourly_ma_trend})"
        )

    # RSI协同性
    if (daily_rsi_state == "超买" and hourly_rsi_state == "超买") or (
        daily_rsi_state == "超卖" and hourly_rsi_state == "超卖"
    ):
        alignment_score += 1
        alignment_signals.append("RSI超买超卖状态一致")

    return {
        "daily": {
            "price": latest_daily["Close"],
            "macd": daily_macd_state,
            "rsi": daily_rsi_state,
            "rsi_value": latest_daily["RSI"],
            "ma_trend": daily_ma_trend,
        },
        "hourly": {
            "price": latest_hourly["Close"],
            "macd": hourly_macd_state,
            "rsi": hourly_rsi_state,
            "rsi_value": latest_hourly["RSI"],
            "ma_trend": hourly_ma_trend,
        },
        "alignment_score": alignment_score,
        "alignment_signals": alignment_signals,
        "alignment_strength": (
            "强" if alignment_score >= 4 else "中等" if alignment_score >= 2 else "弱"
        ),
    }


def find_key_levels(daily_df, hourly_df):
    """寻找关键支撑阻力位"""
    current_price = daily_df["Close"].iloc[-1]

    # 日线关键位
    daily_high_20d = daily_df["High"].tail(20).max()
    daily_low_20d = daily_df["Low"].tail(20).min()
    daily_high_50d = daily_df["High"].tail(50).max()
    daily_low_50d = daily_df["Low"].tail(50).min()

    # 小时线关键位（最近5天）
    recent_hourly = hourly_df.tail(120)  # 最近5天的小时数据
    hourly_high = recent_hourly["High"].max()
    hourly_low = recent_hourly["Low"].min()

    # 斐波那契回撤位（日线）
    max_price = daily_df["High"].max()
    min_price = daily_df["Low"].min()
    diff = max_price - min_price
    fib_382 = max_price - diff * 0.382
    fib_500 = max_price - diff * 0.5
    fib_618 = max_price - diff * 0.618

    return {
        "daily": {
            "resistance_20d": daily_high_20d,
            "support_20d": daily_low_20d,
            "resistance_50d": daily_high_50d,
            "support_50d": daily_low_50d,
        },
        "hourly": {"resistance_5d": hourly_high, "support_5d": hourly_low},
        "fibonacci": {"38.2%": fib_382, "50.0%": fib_500, "61.8%": fib_618},
    }


def analyze_intraday_momentum(hourly_df):
    """分析日内动量"""
    recent_hours = hourly_df.tail(24)  # 最近24小时

    # 计算小时动量
    price_changes = recent_hours["Close"].pct_change().dropna()
    momentum_score = price_changes.sum()

    # 成交量动量
    volume_ma = recent_hours["Volume"].mean()
    current_volume = recent_hours["Volume"].iloc[-1]
    volume_momentum = current_volume / volume_ma

    # 价格相对位置
    recent_high = recent_hours["High"].max()
    recent_low = recent_hours["Low"].min()
    price_position = (recent_hours["Close"].iloc[-1] - recent_low) / (
        recent_high - recent_low
    )

    return {
        "momentum_score": momentum_score,
        "volume_momentum": volume_momentum,
        "price_position": price_position,
        "momentum_direction": (
            "向上" if momentum_score > 0 else "向下" if momentum_score < 0 else "横盘"
        ),
    }


def generate_multi_timeframe_signals(
    daily_df, hourly_df, alignment, key_levels, momentum
):
    """生成多时间框架交易信号"""
    signals = []
    buy_signals = 0
    sell_signals = 0

    current_price = daily_df["Close"].iloc[-1]

    # 时间框架对齐信号
    if alignment["alignment_score"] >= 4:
        signals.append("🟢 多时间框架高度一致，趋势可靠性高")
        buy_signals += 2
    elif alignment["alignment_score"] <= 1:
        signals.append("🔴 多时间框架严重分歧，观望为主")
        sell_signals += 1

    # 日线趋势信号
    if (
        alignment["daily"]["ma_trend"] == "多头"
        and alignment["daily"]["macd"] == "金叉"
    ):
        signals.append("🟢 日线趋势向上，中期看多")
        buy_signals += 2
    elif (
        alignment["daily"]["ma_trend"] == "空头"
        and alignment["daily"]["macd"] == "死叉"
    ):
        signals.append("🔴 日线趋势向下，中期看空")
        sell_signals += 2

    # 小时线动量信号
    if (
        alignment["hourly"]["macd"] == "金叉"
        and momentum["momentum_direction"] == "向上"
    ):
        signals.append("🟡 小时线动量向上，短期偏多")
        buy_signals += 1
    elif (
        alignment["hourly"]["macd"] == "死叉"
        and momentum["momentum_direction"] == "向下"
    ):
        signals.append("🟡 小时线动量向下，短期偏空")
        sell_signals += 1

    # RSI信号
    if alignment["daily"]["rsi_value"] < 35 and alignment["hourly"]["rsi_value"] < 35:
        signals.append("🟢 多时间框架RSI超卖，反弹机会")
        buy_signals += 2
    elif alignment["daily"]["rsi_value"] > 65 and alignment["hourly"]["rsi_value"] > 65:
        signals.append("🔴 多时间框架RSI超买，回调风险")
        sell_signals += 2

    # 成交量确认
    if momentum["volume_momentum"] > 1.5:
        signals.append("🟢 成交量显著放大，趋势确认")
        buy_signals += 1
    elif momentum["volume_momentum"] < 0.5:
        signals.append("🔴 成交量萎缩，趋势减弱")
        sell_signals += 1

    # 综合判断
    net_signal = buy_signals - sell_signals

    if net_signal >= 3:
        recommendation = "强烈买入"
        action = "积极建仓"
        confidence = "高"
    elif net_signal >= 1:
        recommendation = "买入"
        action = "分批建仓"
        confidence = "中等"
    elif net_signal <= -3:
        recommendation = "强烈卖出"
        action = "果断减仓"
        confidence = "高"
    elif net_signal <= -1:
        recommendation = "卖出"
        action = "逐步减仓"
        confidence = "中等"
    else:
        recommendation = "观望"
        action = "等待信号"
        confidence = "低"

    # 关键价位
    resistance_levels = [
        key_levels["hourly"]["resistance_5d"],
        key_levels["daily"]["resistance_20d"],
        current_price * 1.03,
        current_price * 1.05,
        key_levels["daily"]["resistance_50d"],
        key_levels["fibonacci"]["38.2%"],
    ]

    support_levels = [
        key_levels["hourly"]["support_5d"],
        key_levels["daily"]["support_20d"],
        current_price * 0.97,
        current_price * 0.95,
        key_levels["daily"]["support_50d"],
        key_levels["fibonacci"]["61.8%"],
    ]

    return {
        "recommendation": recommendation,
        "action": action,
        "confidence": confidence,
        "net_signal": net_signal,
        "signals": signals,
        "resistance_levels": sorted(
            [x for x in resistance_levels if x > current_price]
        ),
        "support_levels": sorted(
            [x for x in support_levels if x < current_price], reverse=True
        ),
    }


def main():
    """主分析函数"""
    print("=" * 80)
    print("300450.SZ 先导智能 - 多时间框架技术分析报告")
    print("=" * 80)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 加载数据
    daily_file = (
        "/Users/zhangshenshen/深度量化0927/a股/300450.SZ/300450.SZ_1d_2025-09-30.csv"
    )
    hourly_file = (
        "/Users/zhangshenshen/深度量化0927/a股/300450.SZ/300450.SZ_1h_2025-09-30.csv"
    )

    daily_df, hourly_df = load_stock_data(daily_file, hourly_file)

    # 计算技术指标
    daily_df, hourly_df = calculate_technical_indicators(daily_df, hourly_df)

    # 分析时间框架对齐
    alignment = analyze_timeframe_alignment(daily_df, hourly_df)

    # 寻找关键价位
    key_levels = find_key_levels(daily_df, hourly_df)

    # 分析日内动量
    momentum = analyze_intraday_momentum(hourly_df)

    # 生成多时间框架信号
    signals = generate_multi_timeframe_signals(
        daily_df, hourly_df, alignment, key_levels, momentum
    )

    # 输出分析结果
    print("\n" + "=" * 80)
    print("📊 多时间框架状态对比")
    print("=" * 80)
    print("📈 日线状态:")
    print(f"   价格: {alignment['daily']['price']:.2f}元")
    print(f"   MACD: {alignment['daily']['macd']}")
    print(
        f"   RSI: {alignment['daily']['rsi_value']:.1f} ({alignment['daily']['rsi']})"
    )
    print(f"   均线趋势: {alignment['daily']['ma_trend']}")

    print("\n🕒 小时线状态:")
    print(f"   价格: {alignment['hourly']['price']:.2f}元")
    print(f"   MACD: {alignment['hourly']['macd']}")
    print(
        f"   RSI: {alignment['hourly']['rsi_value']:.1f} ({alignment['hourly']['rsi']})"
    )
    print(f"   均线趋势: {alignment['hourly']['ma_trend']}")

    print(
        f"\n🔗 时间框架对齐强度: {alignment['alignment_strength']} (评分: {alignment['alignment_score']}/5)"
    )

    print("\n" + "=" * 80)
    print("⚡ 日内动量分析")
    print("=" * 80)
    print(f"📈 动量方向: {momentum['momentum_direction']}")
    print(f"📊 动量分数: {momentum['momentum_score']:.3f}")
    print(f"📈 成交量动量: {momentum['volume_momentum']:.2f}x")
    print(f"📍 价格位置: {momentum['price_position']:.2%} (相对于24小时区间)")

    print("\n" + "=" * 80)
    print("🎯 关键支撑阻力位")
    print("=" * 80)
    print("📊 日线关键位:")
    print(f"   20日阻力: {key_levels['daily']['resistance_20d']:.2f}元")
    print(f"   20日支撑: {key_levels['daily']['support_20d']:.2f}元")
    print(f"   50日阻力: {key_levels['daily']['resistance_50d']:.2f}元")
    print(f"   50日支撑: {key_levels['daily']['support_50d']:.2f}元")

    print("\n📊 小时线关键位 (最近5天):")
    print(f"   阻力位: {key_levels['hourly']['resistance_5d']:.2f}元")
    print(f"   支撑位: {key_levels['hourly']['support_5d']:.2f}元")

    print("\n📊 斐波那契回撤位:")
    print(f"   38.2%: {key_levels['fibonacci']['38.2%']:.2f}元")
    print(f"   50.0%: {key_levels['fibonacci']['50.0%']:.2f}元")
    print(f"   61.8%: {key_levels['fibonacci']['61.8%']:.2f}元")

    print("\n" + "=" * 80)
    print("💡 多时间框架交易信号")
    print("=" * 80)
    print(f"🎯 综合建议: {signals['recommendation']}")
    print(f"📋 操作建议: {signals['action']}")
    print(f"🔒 信号强度: {signals['confidence']}")
    print(f"📊 净信号: {signals['net_signal']:+d}")

    print("\n📈 详细信号:")
    for i, signal in enumerate(signals["signals"], 1):
        print(f"   {i}. {signal}")

    print(f"\n🎯 压力位 (从小到大):")
    for i, level in enumerate(signals["resistance_levels"][:4], 1):
        print(f"   压力位{i}: {level:.2f}元")

    print(f"\n🎯 支撑位 (从大到小):")
    for i, level in enumerate(signals["support_levels"][:4], 1):
        print(f"   支撑位{i}: {level:.2f}元")

    print("\n" + "=" * 80)
    print("📝 时间框架协同分析")
    print("=" * 80)
    print("🔗 时间框架对齐信号:")
    for signal in alignment["alignment_signals"]:
        print(f"   • {signal}")

    # 策略建议
    print("\n" + "=" * 80)
    print("🎯 策略建议")
    print("=" * 80)

    current_price = daily_df["Close"].iloc[-1]

    if signals["recommendation"] in ["强烈买入", "买入"]:
        print("🟢 入场策略:")
        print(
            f"   • 建议价格区间: {signals['support_levels'][0]:.2f} - {current_price:.2f}元"
        )
        print(f"   • 分批建仓，控制单笔仓位不超过20%")
        print(f"   • 优先在支撑位附近布局")

        print("\n🟢 出场策略:")
        print(f"   • 第一目标: {signals['resistance_levels'][0]:.2f}元")
        print(f"   • 第二目标: {signals['resistance_levels'][1]:.2f}元")
        print(f"   • 止损位: {signals['support_levels'][-1]:.2f}元")

    elif signals["recommendation"] in ["强烈卖出", "卖出"]:
        print("🔴 减仓策略:")
        print(f"   • 建议在压力位 {signals['resistance_levels'][0]:.2f}元附近减仓")
        print(f"   • 分批减仓，保留核心仓位")
        print(f"   • 关注支撑位反弹机会")

    else:
        print("🟡 观望策略:")
        print(f"   • 等待更明确信号")
        print(
            f"   • 关注价格在 {signals['support_levels'][0]:.2f} - {signals['resistance_levels'][0]:.2f}元区间的突破"
        )
        print(f"   • 当前适合高抛低吸")

    print(f"\n⚠️ 风险提示:")
    print(f"   • 该股年化波动率较高，建议轻仓操作")
    print(f"   • 严格设置止损，控制单笔损失在3%以内")
    print(f"   • 密切关注时间框架信号变化")

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
