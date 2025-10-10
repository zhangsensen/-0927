#!/usr/bin/env python3
"""
通用股票技术分析脚本
使用最新下载的数据进行中短期技术分析
支持任意股票代码的分析
"""

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def generate_stock_name(stock_code):
    """根据股票代码生成股票名称"""
    # 存储概念股票代码映射
    stock_names = {
        "000021.SZ": "长城开发",
        "001309.SZ": "德明利",
        "002049.SZ": "紫光国微",
        "002156.SZ": "通富微电",
        "300223.SZ": "北京君正",
        "300661.SZ": "圣邦股份",
        "300782.SZ": "卓胜微",
        "301308.SZ": "江波龙",
        "603986.SS": "兆易创新",
        "688008.SS": "澜起科技",
        "688123.SS": "聚辰股份",
        "688200.SS": "华峰测控",
        "688516.SS": "奥普特",
        "688525.SS": "佰维存储",
        "688766.SS": "普冉股份",
        "688981.SS": "中芯国际",
        # 其他股票
        "300450.SZ": "先导智能",
        "002074.SZ": "国轩高科",
        "000661.SZ": "远望谷",
        "000001.SZ": "平安银行",
        "000002.SZ": "万科A",
        "600000.SH": "浦发银行",
        "600036.SH": "招商银行",
        "000858.SZ": "五粮液",
        "600519.SH": "贵州茅台",
        "000895.SZ": "双汇发展",
        "600276.SH": "恒瑞医药",
    }
    return stock_names.get(stock_code, stock_code)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="通用股票技术分析工具")
    parser.add_argument("stock_code", help="股票代码 (例如: 300450.SZ)")
    parser.add_argument(
        "--data-dir",
        default="/Users/zhangshenshen/深度量化0927/a股",
        help="数据文件目录路径",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/zhangshenshen/深度量化0927",
        help="分析报告输出目录",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="数据文件日期 (默认: 今天)",
    )
    return parser.parse_args()


def load_stock_data(file_path):
    """加载股票数据，跳过文件头部的重复行"""
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


def calculate_technical_indicators(df, stock_code=None):
    """
    计算技术指标 - 使用统一因子引擎适配器

    架构优化：
    - 使用AShareFactorAdapter统一获取技术指标
    - 消除300行重复的指标计算代码
    - 利用FactorEngine的缓存机制提升性能
    - 输入：标准DataFrame (Date索引, OHLCV列)
    - 输出：添加技术指标列的DataFrame
    """
    import sys
    from pathlib import Path

    # 添加项目根目录
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if stock_code is None:
        print("⚠️  需要提供stock_code参数来使用因子引擎")
        return df

    print(f"正在为{stock_code}计算技术指标...")

    try:
        # 导入A股因子适配器
        from factor_adapter import AShareFactorAdapter

        # 初始化适配器
        adapter = AShareFactorAdapter(data_dir=project_root)

        # 重置索引以适配适配器
        df_with_timestamp = df.reset_index()
        df_with_timestamp = df_with_timestamp.rename(columns={'Date': 'timestamp'})

        # 使用适配器添加技术指标
        df_with_indicators = adapter.add_indicators_to_dataframe(df_with_timestamp, stock_code)

        # 恢复原始索引格式
        df_with_indicators = df_with_indicators.set_index('timestamp')
        df_with_indicators.index.name = 'Date'

        # 获取缓存统计
        cache_stats = adapter.get_cache_stats()
        print(f"✅ 技术指标计算完成，缓存命中率: {cache_stats.get('memory_hit_rate', 0):.1%}")

        return df_with_indicators

    except Exception as e:
        print(f"❌ 因子引擎计算失败，回退到基础指标: {e}")

        # 回退：只计算最基础的指标
        print("正在计算基础技术指标...")

        # 基础移动平均线
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()

        # 基础RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        # 基础MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        print("✅ 基础指标计算完成")

        return df


# ============================================================
# 以下手工指标计算函数已废弃
# 全部由统一因子引擎替代：AShareFactorAdapter
# ============================================================
# 删除：calculate_rsi_wilders, calculate_kdj, calculate_williams_r,
#      calculate_atr, calculate_momentum, calculate_cci, calculate_trix,
#      calculate_dpo, calculate_mfi, calculate_adx, calculate_vortex
# 总计删除: ~200行重复代码
# ============================================================


def detect_macd_divergence(df, lookback=20):
    """检测MACD背离"""
    divergence = pd.Series(0, index=df.index, dtype=int)

    for i in range(lookback, len(df)):
        # 获取价格和MACD的局部高点
        price_highs = df["High"].iloc[i - lookback : i]
        macd_highs = df["MACD"].iloc[i - lookback : i]

        # 检测顶背离（价格创新高但MACD没有）
        if len(price_highs) >= 2:
            price_curr = price_highs.iloc[-1]
            price_prev = (
                price_highs.iloc[-5:-1].max()
                if len(price_highs) > 5
                else price_highs.iloc[:-1].max()
            )

            macd_curr = macd_highs.iloc[-1]
            macd_prev = (
                macd_highs.iloc[-5:-1].max()
                if len(macd_highs) > 5
                else macd_highs.iloc[:-1].max()
            )

            if price_curr > price_prev and macd_curr < macd_prev:
                divergence.iloc[i] = -1  # 顶背离

        # 检测底背离（价格创新低但MACD没有）
        price_lows = df["Low"].iloc[i - lookback : i]
        macd_lows = df["MACD"].iloc[i - lookback : i]

        if len(price_lows) >= 2:
            price_curr = price_lows.iloc[-1]
            price_prev = (
                price_lows.iloc[-5:-1].min()
                if len(price_lows) > 5
                else price_lows.iloc[:-1].min()
            )

            macd_curr = macd_lows.iloc[-1]
            macd_prev = (
                macd_lows.iloc[-5:-1].min()
                if len(macd_lows) > 5
                else macd_lows.iloc[:-1].min()
            )

            if price_curr < price_prev and macd_curr > macd_prev:
                divergence.iloc[i] = 1  # 底背离

    return divergence


def find_support_resistance(df, window=20):
    """寻找支撑阻力位 - 增强版本"""
    print("正在计算支撑阻力位...")

    # 传统方法
    highs = df["High"].rolling(window=window, center=True).max()
    lows = df["Low"].rolling(window=window, center=True).min()

    resistance_levels = highs.quantile([0.8, 0.85, 0.9, 0.95])
    support_levels = lows.quantile([0.05, 0.1, 0.15, 0.2])

    # 斐波那契回撤位
    max_price = df["High"].max()
    min_price = df["Low"].min()

    fib_382 = max_price - (max_price - min_price) * 0.382
    fib_500 = max_price - (max_price - min_price) * 0.5
    fib_618 = max_price - (max_price - min_price) * 0.618

    # 斐波那契扩展位
    fib_127 = max_price + (max_price - min_price) * 0.272
    fib_161 = max_price + (max_price - min_price) * 0.618

    # 使用聚类算法识别动态支撑阻力位（已删除cluster_support_resistance）
    # TODO: 如需高级支撑阻力位，可重新实现或使用factor_engine的信号因子
    clustered_resistance, clustered_support = {}, {}

    # 计算枢轴点 (Pivot Points)
    pivot_highs = df["High"].rolling(window=5, center=True).max()
    pivot_lows = df["Low"].rolling(window=5, center=True).min()

    # 找到最近的枢轴点
    current_price = df["Close"].iloc[-1]
    nearest_resistance_high = (
        pivot_highs[pivot_highs > current_price].min()
        if (pivot_highs > current_price).any()
        else current_price * 1.05
    )
    nearest_support_low = (
        pivot_lows[pivot_lows < current_price].max()
        if (pivot_lows < current_price).any()
        else current_price * 0.95
    )

    # 计算枢轴点 (Pivot Points)
    current_pivot = (
        df["High"].iloc[-1] + df["Low"].iloc[-1] + df["Close"].iloc[-1]
    ) / 3
    pivot_r1 = 2 * current_pivot - df["Low"].iloc[-1]
    pivot_r2 = current_pivot + (df["High"].iloc[-1] - df["Low"].iloc[-1])
    pivot_s1 = 2 * current_pivot - df["High"].iloc[-1]
    pivot_s2 = current_pivot - (df["High"].iloc[-1] - df["Low"].iloc[-1])

    print("支撑阻力位计算完成")
    return {
        "resistance": resistance_levels,
        "support": support_levels,
        "fibonacci": {
            "38.2%": fib_382,
            "50.0%": fib_500,
            "61.8%": fib_618,
            "127.2%": fib_127,
            "161.8%": fib_161,
        },
        "clustered_resistance": clustered_resistance,
        "clustered_support": clustered_support,
        "nearest_resistance": nearest_resistance_high,
        "nearest_support": nearest_support_low,
        "max_price": max_price,
        "min_price": min_price,
        "pivot_points": {
            "PP": current_pivot,
            "R1": pivot_r1,
            "R2": pivot_r2,
            "S1": pivot_s1,
            "S2": pivot_s2,
        },
    }


def calculate_performance_metrics(df):
    """计算性能指标"""
    print("正在计算性能指标...")

    returns = df["Close"].pct_change()

    # 计算累计收益
    df["Cumulative_Return"] = (1 + returns).cumprod() - 1

    # 计算年化波动率
    annualized_volatility = returns.std() * np.sqrt(252) * 100

    # 计算最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() * 100

    # 计算夏普比率（假设无风险利率为2%）
    risk_free_rate = 0.02
    excess_returns = returns.mean() * 252 - risk_free_rate
    sharpe_ratio = excess_returns / (returns.std() * np.sqrt(252))

    # 计算上涨天数占比
    up_days = (returns > 0).sum()
    total_days = len(returns) - 1  # 减去第一个NaN值
    up_days_ratio = up_days / total_days * 100

    metrics = {
        "Total_Return": df["Cumulative_Return"].iloc[-1] * 100,
        "Annualized_Volatility": annualized_volatility,
        "Max_Drawdown": max_drawdown,
        "Sharpe_Ratio": sharpe_ratio,
        "Up_Days_Ratio": up_days_ratio,
        "Current_Price": df["Close"].iloc[-1],
        "Price_Change_Pct": (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100,
    }

    print("性能指标计算完成")
    return metrics


def analyze_current_state(df, indicators):
    """分析当前市场状态"""
    print("正在分析当前市场状态...")

    current_price = df["Close"].iloc[-1]
    current_rsi = df["RSI"].iloc[-1]
    current_macd = df["MACD"].iloc[-1]
    current_signal = df["Signal"].iloc[-1]

    # 价格相对位置
    price_vs_ma5 = current_price > df["MA5"].iloc[-1]
    price_vs_ma20 = current_price > df["MA20"].iloc[-1]
    price_vs_ma60 = current_price > df["MA60"].iloc[-1]

    # 均线排列
    ma_arrangement = ""
    if (
        df["MA5"].iloc[-1]
        > df["MA10"].iloc[-1]
        > df["MA20"].iloc[-1]
        > df["MA60"].iloc[-1]
    ):
        ma_arrangement = "完美多头排列"
    elif df["MA5"].iloc[-1] > df["MA10"].iloc[-1] > df["MA20"].iloc[-1]:
        ma_arrangement = "短期多头排列"
    elif df["MA20"].iloc[-1] > df["MA60"].iloc[-1]:
        ma_arrangement = "中期多头排列"
    else:
        ma_arrangement = "均线混乱"

    # RSI状态
    if current_rsi > 70:
        rsi_state = "超买"
    elif current_rsi < 30:
        rsi_state = "超卖"
    elif current_rsi > 50:
        rsi_state = "强势区域"
    else:
        rsi_state = "弱势区域"

    # MACD状态
    if current_macd > current_signal:
        macd_state = "金叉状态"
    else:
        macd_state = "死叉状态"

    # 价格相对布林带
    bb_position = (current_price - df["BB_Lower"].iloc[-1]) / (
        df["BB_Upper"].iloc[-1] - df["BB_Lower"].iloc[-1]
    )

    if bb_position > 0.8:
        bb_state = "接近上轨"
    elif bb_position < 0.2:
        bb_state = "接近下轨"
    else:
        bb_state = "中间区域"

    return {
        "Price": current_price,
        "RSI": current_rsi,
        "MACD_State": macd_state,
        "MA_Arrangement": ma_arrangement,
        "RSI_State": rsi_state,
        "BB_State": bb_state,
        "Price_vs_MA5": price_vs_ma5,
        "Price_vs_MA20": price_vs_ma20,
        "Price_vs_MA60": price_vs_ma60,
        "BB_Position": bb_position,
    }


def generate_trading_recommendation(df, metrics, current_state, indicators):
    """生成增强的交易建议 - 整合所有技术指标"""
    print("正在生成增强的交易建议...")

    current_price = current_state["Price"]

    # ===== 1. 动量指标分析 =====
    momentum_score = 0
    momentum_signals = {}

    # RSI信号 (使用Wilders平滑RSI)
    current_rsi = df["RSI_Wilders"].iloc[-1]
    if current_rsi > 70:
        momentum_score -= 2
        momentum_signals["rsi"] = "超买"
    elif current_rsi < 30:
        momentum_score += 2
        momentum_signals["rsi"] = "超卖"
    elif current_rsi > 50:
        momentum_score += 1
        momentum_signals["rsi"] = "强势"
    else:
        momentum_score -= 1
        momentum_signals["rsi"] = "弱势"

    # MACD信号
    current_macd_hist = df["MACD_Hist"].iloc[-1]
    current_macd = df["MACD"].iloc[-1]
    if current_macd > 0 and current_macd_hist > 0:
        momentum_score += 2
        momentum_signals["macd"] = "多头加速"
    elif current_macd > 0 and current_macd_hist < 0:
        momentum_score += 1
        momentum_signals["macd"] = "多头回调"
    elif current_macd < 0 and current_macd_hist < 0:
        momentum_score -= 2
        momentum_signals["macd"] = "空头加速"
    else:
        momentum_score -= 1
        momentum_signals["macd"] = "空头反弹"

    # KDJ信号
    k_val = df["KDJ_K"].iloc[-1]
    d_val = df["KDJ_D"].iloc[-1]
    j_val = df["KDJ_J"].iloc[-1]
    
    if k_val > 80 and d_val > 80:
        momentum_score -= 1
        momentum_signals["kdj"] = "超买区"
    elif k_val < 20 and d_val < 20:
        momentum_score += 1
        momentum_signals["kdj"] = "超卖区"
    elif k_val > d_val and j_val > k_val:
        momentum_score += 1
        momentum_signals["kdj"] = "金叉向上"
    elif k_val < d_val and j_val < k_val:
        momentum_score -= 1
        momentum_signals["kdj"] = "死叉向下"
    else:
        momentum_signals["kdj"] = "震荡"

    # Williams %R信号
    williams_r = df["Williams_R"].iloc[-1]
    if williams_r > -20:
        momentum_score -= 1
        momentum_signals["williams_r"] = "超买"
    elif williams_r < -80:
        momentum_score += 1
        momentum_signals["williams_r"] = "超卖"
    else:
        momentum_signals["williams_r"] = "正常"

    # CCI信号
    cci_val = df["CCI"].iloc[-1]
    if cci_val > 200:
        momentum_score -= 1
        momentum_signals["cci"] = "极度超买"
    elif cci_val < -200:
        momentum_score += 1
        momentum_signals["cci"] = "极度超卖"
    else:
        momentum_signals["cci"] = "正常区间"

    # ===== 2. 趋势强度分析 =====
    trend_score = 0
    trend_signals = {}

    # ADX趋势强度
    adx_val = df["ADX"].iloc[-1]
    di_plus = df["DI_plus"].iloc[-1]
    di_minus = df["DI_minus"].iloc[-1]
    
    if adx_val > 35:
        trend_score += 2
        trend_signals["adx_strength"] = "强趋势"
    elif adx_val > 25:
        trend_score += 1
        trend_signals["adx_strength"] = "中等趋势"
    else:
        trend_signals["adx_strength"] = "弱趋势/震荡"
    
    # ADX方向
    if di_plus > di_minus:
        if adx_val > 25:
            trend_score += 2
            trend_signals["adx_direction"] = "上升趋势"
        else:
            trend_score += 1
            trend_signals["adx_direction"] = "微弱上升"
    else:
        if adx_val > 25:
            trend_score -= 2
            trend_signals["adx_direction"] = "下降趋势"
        else:
            trend_score -= 1
            trend_signals["adx_direction"] = "微弱下降"

    # Vortex指标（暂未实现，降级处理）
    trend_signals["vortex"] = "未计算"

    # 均线系统评分
    if current_state["MA_Arrangement"] == "完美多头排列":
        trend_score += 3
        trend_signals["ma_system"] = "完美多头排列"
    elif "多头排列" in current_state["MA_Arrangement"]:
        trend_score += 1
        trend_signals["ma_system"] = "短期多头"
    elif current_state["MA_Arrangement"] == "均线混乱":
        trend_signals["ma_system"] = "均线缠绕"
    else:
        trend_score -= 1
        trend_signals["ma_system"] = "短期空头"

    # ===== 3. 波动率分析 =====
    volatility_score = 0
    volatility_signals = {}

    # 布林带位置
    bb_position = current_state["BB_Position"]
    bb_width = (df["BB_Upper"].iloc[-1] - df["BB_Lower"].iloc[-1]) / df[
        "BB_Middle"
    ].iloc[-1]

    if bb_position > 0.8:
        volatility_score -= 1
        volatility_signals["bb_position"] = "接近上轨"
    elif bb_position < 0.2:
        volatility_score += 1
        volatility_signals["bb_position"] = "接近下轨"
    else:
        volatility_signals["bb_position"] = "中轨区域"

    # 布林带宽度（波动率）
    if bb_width > 0.15:  # 高波动率
        volatility_signals["bb_width"] = "高波动"
    elif bb_width < 0.05:  # 低波动率
        volatility_signals["bb_width"] = "低波动"
    else:
        volatility_signals["bb_width"] = "正常波动"

    # ATR波动率
    current_atr = df["ATR"].iloc[-1]
    atr_ratio = current_atr / current_price
    if atr_ratio > 0.03:
        volatility_score -= 1
        volatility_signals["atr_volatility"] = "高波动"
    else:
        volatility_signals["atr_volatility"] = "正常波动"

    # ===== 4. 成交量确认 =====
    volume_score = 0
    volume_signals = {}

    # 成交量比率
    volume_ratio = df["Volume_Ratio"].iloc[-1]
    if volume_ratio > 1.5:
        volume_score += 1
        volume_signals["volume_strength"] = "放量"
    elif volume_ratio < 0.5:
        volume_score -= 1
        volume_signals["volume_strength"] = "缩量"
    else:
        volume_signals["volume_strength"] = "正常量"

    # MFI资金流量指标
    mfi_val = df["MFI"].iloc[-1]
    if mfi_val > 80:
        volume_score -= 1
        volume_signals["mfi"] = "资金流入过度"
    elif mfi_val < 20:
        volume_score += 1
        volume_signals["mfi"] = "资金流入不足"
    else:
        volume_signals["mfi"] = "资金流量正常"

    # ===== 5. 综合评分计算 =====
    total_score = momentum_score + trend_score + volatility_score + volume_score

    # ===== 6. 支撑阻力位分析 =====
    # 价格相对位置
    nearest_resistance = indicators.get("nearest_resistance", current_price * 1.05)
    nearest_support = indicators.get("nearest_support", current_price * 0.95)

    if nearest_resistance and nearest_support:
        price_position = (current_price - nearest_support) / (
            nearest_resistance - nearest_support
        )
        if price_position > 0.8:
            position_signal = "接近阻力位"
            total_score -= 1
        elif price_position < 0.2:
            position_signal = "接近支撑位"
            total_score += 1
        else:
            position_signal = "中间区域"
    else:
        position_signal = "位置不明"

    # ===== 7. 生成交易建议 =====
    # 信心度调整
    confidence_adjustment = min(abs(total_score) / 10, 1.0)

    if total_score >= 6:
        recommendation = "强烈买入"
        action = "积极建仓"
        risk_level = "低风险"
    elif total_score >= 3:
        recommendation = "买入"
        action = "逢低买入"
        risk_level = "中等风险"
    elif total_score <= -6:
        recommendation = "强烈卖出"
        action = "果断减仓"
        risk_level = "低风险"
    elif total_score <= -3:
        recommendation = "卖出"
        action = "逢高减仓"
        risk_level = "中等风险"
    else:
        recommendation = "持有"
        action = "观望等待"
        risk_level = "观望"

    # ===== 8. 风险管理参数 =====
    # 基于ATR设置止损
    if current_atr:
        if recommendation in ["强烈买入", "买入"]:
            stop_loss = max(
                nearest_support if nearest_support else current_price * 0.95,
                current_price - 2 * current_atr,
            )
            take_profit = min(
                nearest_resistance if nearest_resistance else current_price * 1.05,
                current_price + 3 * current_atr,
            )
        else:
            stop_loss = min(
                nearest_resistance if nearest_resistance else current_price * 1.05,
                current_price + 2 * current_atr,
            )
            take_profit = max(
                nearest_support if nearest_support else current_price * 0.95,
                current_price - 3 * current_atr,
            )
    else:
        # 传统百分比止损
        if recommendation in ["强烈买入", "买入"]:
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.10
        else:
            stop_loss = current_price * 1.05
            take_profit = current_price * 0.90

    # 仓位建议
    if total_score >= 6:
        position_size = 80  # 80%仓位
    elif total_score >= 3:
        position_size = 50  # 50%仓位
    elif total_score <= -6:
        position_size = 10  # 10%仓位
    elif total_score <= -3:
        position_size = 30  # 30%仓位
    else:
        position_size = 40  # 40%仓位

    # 生成信号列表用于显示
    signals = []
    signals.extend(
        [f"RSI: {momentum_signals['rsi']}", f"MACD: {momentum_signals['macd']}"]
    )
    signals.extend(
        [
            f"ADX: {trend_signals['adx_strength']} {trend_signals['adx_direction']}",
            f"均线: {trend_signals['ma_system']}",
        ]
    )
    if volume_signals["volume_strength"] != "正常量":
        signals.append(f"成交量: {volume_signals['volume_strength']}")
    if position_signal != "中间区域":
        signals.append(f"位置: {position_signal}")

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": current_price,
        "analysis_scores": {
            "momentum_score": momentum_score,
            "trend_score": trend_score,
            "volatility_score": volatility_score,
            "volume_score": volume_score,
            "total_score": total_score,
        },
        "technical_signals": {
            "momentum_signals": momentum_signals,
            "trend_signals": trend_signals,
            "volatility_signals": volatility_signals,
            "volume_signals": volume_signals,
            "position_signal": position_signal,
        },
        "recommendation": {
            "action": recommendation,
            "detail": action,
            "risk_level": risk_level,
            "confidence": confidence_adjustment * 100,
        },
        "support_resistance": indicators,
        "risk_management": {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
            "risk_reward_ratio": abs(take_profit - current_price)
            / abs(current_price - stop_loss),
        },
        "key_indicators": {
            "RSI": current_rsi,
            "MACD": current_macd,
            "ADX": adx_val,
            "ATR": current_atr,
            "BB_Position": bb_position,
        },
        "Signals": signals,  # 保持兼容性
        "Targets": {  # 保持兼容性
            "resistance_1": take_profit,
            "resistance_2": take_profit * 1.05,
            "support_1": stop_loss,
            "support_2": stop_loss * 0.95,
            "fib_382": indicators["fibonacci"]["38.2%"],
            "fib_500": indicators["fibonacci"]["50.0%"],
            "fib_618": indicators["fibonacci"]["61.8%"],
        },
        "Current_Price": current_price,  # 保持兼容性
        "Recommendation": recommendation,  # 保持兼容性
        "Action": action,  # 保持兼容性
    }


def main():
    """主分析函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="通用股票技术分析工具")
    parser.add_argument(
        "stock_code", nargs="?", default="300450.SZ", help="股票代码 (例如: 300450.SZ)"
    )
    parser.add_argument(
        "--data-dir",
        default="/Users/zhangshenshen/深度量化0927/a股",
        help="数据文件目录路径",
    )
    parser.add_argument(
        "--output-dir",
        default="/Users/zhangshenshen/深度量化0927",
        help="分析报告输出目录",
    )
    args = parser.parse_args()

    stock_code = args.stock_code
    stock_name = generate_stock_name(stock_code)

    print("=" * 60)
    print(f"{stock_code} {stock_name} - 技术分析报告")
    print("=" * 60)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 构建数据文件路径（使用最新可用文件）
    import glob
    stock_dir = f"{args.data_dir}/{stock_code}"
    daily_files = glob.glob(f"{stock_dir}/{stock_code}_1d_*.csv")
    if not daily_files:
        print(f"❌ 未找到日线数据: {stock_dir}/{stock_code}_1d_*.csv")
        return
    daily_file = sorted(daily_files)[-1]  # 使用最新文件
    hourly_file = daily_file.replace('_1d_', '_1h_')

    # 检查数据文件是否存在
    if not os.path.exists(daily_file):
        print(f"❌ 数据文件不存在: {daily_file}")
        return

    df = load_stock_data(daily_file)

    # 计算技术指标
    df_with_indicators = calculate_technical_indicators(df)

    # 计算性能指标
    metrics = calculate_performance_metrics(df_with_indicators)

    # 寻找支撑阻力位
    indicators = find_support_resistance(df_with_indicators)

    # 分析当前状态
    current_state = analyze_current_state(df_with_indicators, indicators)

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
    if volatility_level > 40:
        risk_level = "高风险"
    elif volatility_level > 25:
        risk_level = "中等风险"
    else:
        risk_level = "低风险"

    print(f"⚠️ 风险评估: {risk_level} (年化波动率: {volatility_level:.2f}%)")

    # 保存分析结果
    report_content = f"""
# {stock_code} {stock_name} - 技术分析报告

**分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 性能指标
- **总收益率**: {metrics['Total_Return']:.2f}%
- **年化波动率**: {metrics['Annualized_Volatility']:.2f}%
- **最大回撤**: {metrics['Max_Drawdown']:.2f}%
- **夏普比率**: {metrics['Sharpe_Ratio']:.2f}
- **上涨天数占比**: {metrics['Up_Days_Ratio']:.1f}%
- **当前价格**: {metrics['Current_Price']:.2f}元
- **期间涨跌幅**: {metrics['Price_Change_Pct']:.2f}%

## 📈 当前市场状态
- **当前价格**: {current_state['Price']:.2f}元
- **RSI指标**: {current_state['RSI']:.1f} ({current_state['RSI_State']})
- **MACD状态**: {current_state['MACD_State']}
- **均线排列**: {current_state['MA_Arrangement']}
- **布林带位置**: {current_state['BB_State']} ({current_state['BB_Position']:.2f})

## 🎯 支撑阻力位
- **斐波那契38.2%**: {indicators['fibonacci']['38.2%']:.2f}元
- **斐波那契50.0%**: {indicators['fibonacci']['50.0%']:.2f}元
- **斐波那契61.8%**: {indicators['fibonacci']['61.8%']:.2f}元
- **期间最高价**: {indicators['max_price']:.2f}元
- **期间最低价**: {indicators['min_price']:.2f}元

## 💡 交易建议
- **综合建议**: {recommendation['Recommendation']}
- **操作建议**: {recommendation['Action']}

### 技术信号
{chr(10).join(f"- {signal}" for signal in recommendation['Signals'])}

### 关键价位
- **阻力位1**: {recommendation['Targets']['resistance_1']:.2f}元
- **阻力位2**: {recommendation['Targets']['resistance_2']:.2f}元
- **支撑位1**: {recommendation['Targets']['support_1']:.2f}元
- **支撑位2**: {recommendation['Targets']['support_2']:.2f}元

## 📝 详细分析
- **短期趋势**: {trend_direction}
- **趋势强度**: {'强' if abs(trend_slope) > 0.5 else '中等' if abs(trend_slope) > 0.2 else '弱'}
- **成交量活跃度**: {'放量' if volume_ratio > 1.2 else '缩量' if volume_ratio < 0.8 else '正常'}
- **风险评估**: {risk_level} (年化波动率: {volatility_level:.2f}%)

**免责声明**: 本分析报告仅供参考，不构成投资建议。投资者应根据自身风险承受能力和投资目标做出独立决策。
"""

    # 保存报告
    report_file = f"{args.output_dir}/{stock_code}_技术分析报告.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n📄 详细分析报告已保存至: {report_file}")
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
