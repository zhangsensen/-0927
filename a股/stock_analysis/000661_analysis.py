#!/usr/bin/env python3
"""
000661.SZ 远望谷技术分析
使用300450分析框架分析000661数据
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

    print("技术指标计算完成")
    return df


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
        "MA_Arrangement": ma_arrangement,
        "BB_State": bb_state,
        "BB_Position": bb_position,
        "Price_vs_MA5": latest["Close"] > latest["MA5"],
        "Price_vs_MA20": latest["Close"] > latest["MA20"],
        "Price_vs_MA60": latest["Close"] > latest["MA60"],
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

    return {
        "max_price": max_price,
        "min_price": min_price,
        "fibonacci": {"38.2%": fib_382, "50.0%": fib_500, "61.8%": fib_618},
    }


def generate_trading_recommendation(df, metrics, current_state, indicators):
    """生成交易建议"""
    signals = []

    # RSI信号
    if current_state["RSI"] > 70:
        signals.append("RSI: 超买")
    elif current_state["RSI"] < 30:
        signals.append("RSI: 超卖")

    # MACD信号
    if current_state["MACD_State"] == "金叉状态":
        signals.append("MACD: 多头信号")
    else:
        signals.append("MACD: 空头信号")

    # 均线信号
    if current_state["MA_Arrangement"] == "完美多头排列":
        signals.append("均线: 完美多头排列")
    elif current_state["MA_Arrangement"] == "完美空头排列":
        signals.append("均线: 完美空头排列")

    # 综合评分
    score = 0
    if current_state["MACD_State"] == "金叉状态":
        score += 1
    if current_state["MA_Arrangement"] == "完美多头排列":
        score += 1
    if 30 <= current_state["RSI"] <= 70:
        score += 1

    if score >= 2:
        recommendation = "买入"
        action = "建议买入"
    elif score <= 1:
        recommendation = "卖出"
        action = "建议卖出"
    else:
        recommendation = "持有"
        action = "观望等待"

    # 目标价位
    current_price = current_state["Price"]
    take_profit = current_price * 1.05
    stop_loss = current_price * 0.95

    return {
        "Recommendation": recommendation,
        "Action": action,
        "Signals": signals,
        "Targets": {
            "resistance_1": take_profit,
            "resistance_2": take_profit * 1.05,
            "support_1": stop_loss,
            "support_2": stop_loss * 0.95,
            "fib_382": indicators["fibonacci"]["38.2%"],
            "fib_500": indicators["fibonacci"]["50.0%"],
            "fib_618": indicators["fibonacci"]["61.8%"],
        },
    }


def main():
    """主分析函数"""
    print("=" * 60)
    print("000661.SZ 长春高新 - 技术分析报告")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 加载数据
    daily_file = (
        "/Users/zhangshenshen/深度量化0927/a股/000661.SZ/000661.SZ_1d_2025-09-28.csv"
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

    # 生成交易建议
    recommendation = generate_trading_recommendation(
        df_with_indicators, metrics, current_state, indicators
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
    print(f"📊 均线排列: {current_state['MA_Arrangement']}")
    print(
        f"📈 布林带位置: {current_state['BB_State']} ({current_state['BB_Position']:.2f})"
    )
    print(f"🔺 价格相对MA5: {'在上方' if current_state['Price_vs_MA5'] else '在下方'}")
    print(
        f"🔺 价格相对MA20: {'在上方' if current_state['Price_vs_MA20'] else '在下方'}"
    )
    print(
        f"🔺 价格相对MA60: {'在上方' if current_state['Price_vs_MA60'] else '在下方'}"
    )

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
    print("\n📊 技术信号:")
    for i, signal in enumerate(recommendation["Signals"], 1):
        print(f"   {i}. {signal}")

    print("\n🎯 关键价位:")
    print(f"   📈 阻力位1: {recommendation['Targets']['resistance_1']:.2f}元")
    print(f"   📈 阻力位2: {recommendation['Targets']['resistance_2']:.2f}元")
    print(f"   📉 支撑位1: {recommendation['Targets']['support_1']:.2f}元")
    print(f"   📉 支撑位2: {recommendation['Targets']['support_2']:.2f}元")
    print(f"   📊 斐波那契支撑: {recommendation['Targets']['fib_382']:.2f}元")

    # 生成详细分析报告
    print("\n" + "=" * 60)
    print("📝 详细分析报告")
    print("=" * 60)

    # 趋势分析
    recent_trend = df_with_indicators["Close"].tail(20)
    trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]

    if trend_slope > 0:
        trend_direction = "上升趋势"
    elif trend_slope < 0:
        trend_direction = "下降趋势"
    else:
        trend_direction = "横盘整理"

    print(f"📈 短期趋势: {trend_direction}")
    print(
        f"📊 趋势强度: {'强' if abs(trend_slope) > 0.5 else '中等' if abs(trend_slope) > 0.2 else '弱'}"
    )

    # 成交量分析
    recent_volume = df_with_indicators["Volume"].tail(10).mean()
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
        risk_level = "高风险"
    elif volatility_level > 30:
        risk_level = "中风险"
    else:
        risk_level = "低风险"

    print(f"⚠️ 风险评估: {risk_level} (年化波动率: {volatility_level:.2f}%)")

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
