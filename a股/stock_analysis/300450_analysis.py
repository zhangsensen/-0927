#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
300450.SZ 先导智能技术分析
短期走势分析框架
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


def load_stock_data(file_path):
    """加载股票数据"""
    print(f"正在加载数据文件: {file_path}")

    # 读取数据，跳过前两行（标题行和重复行）
    df = pd.read_csv(file_path, skiprows=2)

    # 重命名列
    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    # 转换日期格式
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # 确保数值类型正确
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 按日期排序
    df.sort_index(inplace=True)

    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"最新价格: {df['Close'].iloc[-1]:.2f}")

    return df


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


def calculate_technical_indicators(df):
    """计算技术指标"""
    print("正在计算技术指标...")

    # 移动平均线
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # RSI
    df["RSI"] = calculate_rsi(df["Close"])

    # MACD
    df["MACD"], df["Signal"], df["MACD_Hist"] = calculate_macd(df)

    # 布林带
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = calculate_bollinger_bands(df)

    # 成交量移动平均
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()

    # 波动率
    df["Volatility"] = (
        df["Close"].rolling(window=20).std()
        / df["Close"].rolling(window=20).mean()
        * 100
    )

    # KDJ指标
    df["K"], df["D"], df["J"] = calculate_kdj(df)

    print("技术指标计算完成")
    return df


def calculate_kdj(df, period=9):
    """计算KDJ指标"""
    low_list = df["Low"].rolling(window=period).min()
    high_list = df["High"].rolling(window=period).max()
    rsv = (df["Close"] - low_list) / (high_list - low_list) * 100

    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d

    return k, d, j


def analyze_current_state(df):
    """分析当前市场状态"""
    latest = df.iloc[-1]

    # RSI状态
    rsi = latest["RSI"]
    if rsi > 70:
        rsi_state = "超买"
    elif rsi < 30:
        rsi_state = "超卖"
    else:
        rsi_state = "正常"

    # MACD状态
    if latest["MACD"] > latest["Signal"]:
        macd_state = "金叉状态"
    else:
        macd_state = "死叉状态"

    # KDJ状态
    kdj_signal = ""
    if latest["K"] > latest["D"]:
        kdj_signal = "KDJ金叉"
    else:
        kdj_signal = "KDJ死叉"

    # 均线排列
    if latest["MA5"] > latest["MA10"] > latest["MA20"] > latest["MA60"]:
        ma_arrangement = "完美多头排列"
    elif latest["MA5"] < latest["MA10"] < latest["MA20"] < latest["MA60"]:
        ma_arrangement = "完美空头排列"
    else:
        ma_arrangement = "均线纠缠"

    # 布林带位置
    bb_position = (latest["Close"] - latest["BB_Lower"]) / (
        latest["BB_Upper"] - latest["BB_Lower"]
    )
    if bb_position > 0.8:
        bb_state = "接近上轨"
    elif bb_position < 0.2:
        bb_state = "接近下轨"
    else:
        bb_state = "中轨附近"

    return {
        "Price": latest["Close"],
        "RSI": rsi,
        "RSI_State": rsi_state,
        "MACD_State": macd_state,
        "KDJ_Signal": kdj_signal,
        "MA_Arrangement": ma_arrangement,
        "BB_State": bb_state,
        "BB_Position": bb_position,
        "Price_vs_MA5": latest["Close"] > latest["MA5"],
        "Price_vs_MA10": latest["Close"] > latest["MA10"],
        "Price_vs_MA20": latest["Close"] > latest["MA20"],
        "Price_vs_MA60": latest["Close"] > latest["MA60"],
    }


def find_support_resistance(df):
    """寻找支撑阻力位"""
    max_price = df["High"].max()
    min_price = df["Low"].min()
    current_price = df["Close"].iloc[-1]

    # 斐波那契回撤位
    diff = max_price - min_price
    fib_382 = max_price - diff * 0.382
    fib_500 = max_price - diff * 0.5
    fib_618 = max_price - diff * 0.618

    # 寻找近期高点和低点作为阻力支撑
    recent_highs = df["High"].tail(30).nlargest(3)
    recent_lows = df["Low"].tail(30).nsmallest(3)

    return {
        "max_price": max_price,
        "min_price": min_price,
        "fibonacci": {"38.2%": fib_382, "50.0%": fib_500, "61.8%": fib_618},
        "resistance_levels": recent_highs.values.tolist(),
        "support_levels": recent_lows.values.tolist(),
    }


def calculate_performance_metrics(df):
    """计算性能指标"""
    returns = df["Close"].pct_change().dropna()

    # 总收益率
    total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

    # 年化波动率
    annualized_volatility = returns.std() * np.sqrt(252) * 100

    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min() * 100

    # 夏普比率（假设无风险利率为3%）
    risk_free_rate = 0.03
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))

    # 上涨天数占比
    up_days_ratio = (returns > 0).mean() * 100

    # 期间涨跌幅
    price_change_pct = (
        (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
    ) * 100

    return {
        "Total_Return": total_return,
        "Annualized_Volatility": annualized_volatility,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Ratio": sharpe_ratio,
        "Up_Days_Ratio": up_days_ratio,
        "Current_Price": df["Close"].iloc[-1],
        "Price_Change_Pct": price_change_pct,
    }


def analyze_short_term_trend(df):
    """分析短期趋势"""
    # 5日趋势
    recent_5d = df["Close"].tail(5)
    trend_5d = "上涨" if recent_5d.iloc[-1] > recent_5d.iloc[0] else "下跌"

    # 10日趋势
    recent_10d = df["Close"].tail(10)
    trend_10d = "上涨" if recent_10d.iloc[-1] > recent_10d.iloc[0] else "下跌"

    # 20日趋势
    recent_20d = df["Close"].tail(20)
    trend_20d = "上涨" if recent_20d.iloc[-1] > recent_20d.iloc[0] else "下跌"

    # 计算趋势强度
    ma5_slope = (df["MA5"].iloc[-1] - df["MA5"].iloc[-5]) / 5
    ma10_slope = (df["MA10"].iloc[-1] - df["MA10"].iloc[-10]) / 10

    return {
        "trend_5d": trend_5d,
        "trend_10d": trend_10d,
        "trend_20d": trend_20d,
        "ma5_slope": ma5_slope,
        "ma10_slope": ma10_slope,
        "trend_strength": (
            "强" if abs(ma5_slope) > 0.5 else "中等" if abs(ma5_slope) > 0.2 else "弱"
        ),
    }


def generate_trading_recommendation(
    df, metrics, current_state, indicators, trend_analysis
):
    """生成交易建议"""
    signals = []
    score = 0

    # RSI信号
    if current_state["RSI"] > 70:
        signals.append("RSI超买 - 谨慎")
    elif current_state["RSI"] < 30:
        signals.append("RSI超卖 - 关注")
        score += 1

    # MACD信号
    if current_state["MACD_State"] == "金叉状态":
        signals.append("MACD金叉 - 偏多")
        score += 1
    else:
        signals.append("MACD死叉 - 偏空")

    # KDJ信号
    if current_state["KDJ_Signal"] == "KDJ金叉":
        signals.append("KDJ金叉 - 短多")
        score += 1

    # 均线信号
    if current_state["MA_Arrangement"] == "完美多头排列":
        signals.append("均线完美多头 - 强势")
        score += 2
    elif current_state["MA_Arrangement"] == "均线纠缠":
        signals.append("均线纠缠 - 震荡")

    # 价格相对均线位置
    if current_state["Price_vs_MA5"] and current_state["Price_vs_MA10"]:
        signals.append("价格站上短期均线 - 偏多")
        score += 1

    # 布林带信号
    if current_state["BB_State"] == "接近上轨":
        signals.append("布林带上轨 - 注意回调")
        score -= 1
    elif current_state["BB_State"] == "接近下轨":
        signals.append("布林带下轨 - 反弹机会")
        score += 1

    # 趋势信号
    if trend_analysis["trend_5d"] == "上涨":
        signals.append("5日趋势向上 - 短多")
        score += 1

    # 综合判断
    if score >= 3:
        recommendation = "买入"
        action = "积极买入"
        risk_level = "中低风险"
    elif score >= 1:
        recommendation = "持有"
        action = "持仓观望"
        risk_level = "中等风险"
    else:
        recommendation = "卖出"
        action = "减仓观望"
        risk_level = "高风险"

    # 目标价位和止损位
    current_price = current_state["Price"]

    # 压力位
    resistance_levels = [
        current_price * 1.03,  # 短期压力
        current_price * 1.05,  # 中期压力
        indicators["fibonacci"]["38.2%"],  # 斐波那契阻力
        (
            max(indicators["resistance_levels"])
            if indicators["resistance_levels"]
            else current_price * 1.08
        ),
    ]

    # 支撑位
    support_levels = [
        current_price * 0.97,  # 短期支撑
        current_price * 0.95,  # 中期支撑
        indicators["fibonacci"]["61.8%"],  # 斐波那契支撑
        (
            min(indicators["support_levels"])
            if indicators["support_levels"]
            else current_price * 0.92
        ),
    ]

    return {
        "Recommendation": recommendation,
        "Action": action,
        "Risk_Level": risk_level,
        "Score": score,
        "Signals": signals,
        "Targets": {
            "resistance_levels": sorted(resistance_levels),
            "support_levels": sorted(support_levels, reverse=True),
            "fibonacci": indicators["fibonacci"],
        },
    }


def main():
    """主分析函数"""
    print("=" * 60)
    print("300450.SZ 先导智能 - 短期走势技术分析报告")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 加载数据
    daily_file = (
        "/Users/zhangshenshen/深度量化0927/a股/300450.SZ/300450.SZ_1d_2025-09-30.csv"
    )

    df = load_stock_data(daily_file)

    # 计算技术指标
    df_with_indicators = calculate_technical_indicators(df)

    # 计算性能指标
    metrics = calculate_performance_metrics(df_with_indicators)

    # 寻找支撑阻力位
    indicators = find_support_resistance(df_with_indicators)

    # 分析当前状态
    current_state = analyze_current_state(df_with_indicators)

    # 分析短期趋势
    trend_analysis = analyze_short_term_trend(df_with_indicators)

    # 生成交易建议
    recommendation = generate_trading_recommendation(
        df_with_indicators, metrics, current_state, indicators, trend_analysis
    )

    # 输出分析结果
    print("\n" + "=" * 60)
    print("📊 性能指标")
    print("=" * 60)
    print(f"📈 总收益率: {metrics['Total_Return']:.2f}%")
    print(f"📊 年化波动率: {metrics['Annualized_Volatility']:.2f}%")
    print(f"📉 最大回撤: {metrics['Max_Drawdown']:.2f}%")
    print(f"⚡ 夏普比率: {metrics['Sharpe_Ratio']:.2f}")
    print(f"📈 上涨天数占比: {metrics['Up_Days_Ratio']:.1f}%")
    print(f"💰 当前价格: {metrics['Current_Price']:.2f}元")
    print(f"📊 期间涨跌幅: {metrics['Price_Change_Pct']:.2f}%")

    print("\n" + "=" * 60)
    print("📈 当前市场状态")
    print("=" * 60)
    print(f"💰 当前价格: {current_state['Price']:.2f}元")
    print(f"📊 RSI指标: {current_state['RSI']:.1f} ({current_state['RSI_State']})")
    print(f"📈 MACD状态: {current_state['MACD_State']}")
    print(f"📊 KDJ状态: {current_state['KDJ_Signal']}")
    print(f"📊 均线排列: {current_state['MA_Arrangement']}")
    print(
        f"📈 布林带位置: {current_state['BB_State']} ({current_state['BB_Position']:.2f})"
    )
    print(f"🔺 价格相对MA5: {'在上方' if current_state['Price_vs_MA5'] else '在下方'}")
    print(
        f"🔺 价格相对MA10: {'在上方' if current_state['Price_vs_MA10'] else '在下方'}"
    )
    print(
        f"🔺 价格相对MA20: {'在上方' if current_state['Price_vs_MA20'] else '在下方'}"
    )

    print("\n" + "=" * 60)
    print("📊 短期趋势分析")
    print("=" * 60)
    print(f"📈 5日趋势: {trend_analysis['trend_5d']}")
    print(f"📈 10日趋势: {trend_analysis['trend_10d']}")
    print(f"📈 20日趋势: {trend_analysis['trend_20d']}")
    print(f"📊 趋势强度: {trend_analysis['trend_strength']}")
    print(f"📈 MA5斜率: {trend_analysis['ma5_slope']:.4f}")
    print(f"📈 MA10斜率: {trend_analysis['ma10_slope']:.4f}")

    print("\n" + "=" * 60)
    print("🎯 支撑阻力位")
    print("=" * 60)
    print("📊 斐波那契回撤位:")
    print(f"   38.2%: {indicators['fibonacci']['38.2%']:.2f}元")
    print(f"   50.0%: {indicators['fibonacci']['50.0%']:.2f}元")
    print(f"   61.8%: {indicators['fibonacci']['61.8%']:.2f}元")
    print(f"📊 期间最高价: {indicators['max_price']:.2f}元")
    print(f"📊 期间最低价: {indicators['min_price']:.2f}元")

    print("\n" + "=" * 60)
    print("💡 交易建议")
    print("=" * 60)
    print(f"🎯 综合建议: {recommendation['Recommendation']}")
    print(f"📋 操作建议: {recommendation['Action']}")
    print(f"⚠️ 风险等级: {recommendation['Risk_Level']}")
    print(f"📊 技术评分: {recommendation['Score']}/7分")

    print("\n📊 技术信号:")
    for i, signal in enumerate(recommendation["Signals"], 1):
        print(f"   {i}. {signal}")

    print("\n🎯 压力位 (从小到大):")
    for i, level in enumerate(recommendation["Targets"]["resistance_levels"], 1):
        print(f"   压力位{i}: {level:.2f}元")

    print("\n🎯 支撑位 (从大到小):")
    for i, level in enumerate(recommendation["Targets"]["support_levels"], 1):
        print(f"   支撑位{i}: {level:.2f}元")

    # 详细分析报告
    print("\n" + "=" * 60)
    print("📝 详细分析报告")
    print("=" * 60)

    # 成交量分析
    recent_volume = df_with_indicators["Volume"].tail(5).mean()
    historical_volume = df_with_indicators["Volume"].mean()
    volume_ratio = recent_volume / historical_volume

    print(
        f"📊 成交量分析: 近期平均成交量 {recent_volume:.0f} vs 历史平均 {historical_volume:.0f}"
    )
    print(
        f"📈 成交量活跃度: {'放量' if volume_ratio > 1.2 else '缩量' if volume_ratio < 0.8 else '正常'}"
    )

    # 风险评估
    volatility_level = metrics["Annualized_Volatility"]
    if volatility_level > 50:
        risk_assessment = "高风险"
    elif volatility_level > 30:
        risk_assessment = "中风险"
    else:
        risk_assessment = "低风险"

    print(f"⚠️ 波动率风险评估: {risk_assessment} (年化波动率: {volatility_level:.2f}%)")

    # 关键技术位置
    latest = df_with_indicators.iloc[-1]
    print(f"📊 当前价格位置:")
    print(
        f"   距离MA5: {((latest['Close'] - latest['MA5']) / latest['MA5'] * 100):+.2f}%"
    )
    print(
        f"   距离MA20: {((latest['Close'] - latest['MA20']) / latest['MA20'] * 100):+.2f}%"
    )
    print(
        f"   距离布林带上轨: {((latest['Close'] - latest['BB_Upper']) / latest['BB_Upper'] * 100):+.2f}%"
    )
    print(
        f"   距离布林带下轨: {((latest['Close'] - latest['BB_Lower']) / latest['BB_Lower'] * 100):+.2f}%"
    )

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
