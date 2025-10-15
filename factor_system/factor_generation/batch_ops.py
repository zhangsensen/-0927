#!/usr/bin/env python3
"""
批量向量化操作 - VectorBT批量计算封装
消除逐参数闭包，一次计算多窗口
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)


def execute_ma_batch(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """批量计算移动平均线

    Args:
        price: 价格序列
        windows: 窗口列表

    Returns:
        DataFrame，每列为一个窗口的MA结果
    """
    if not windows:
        return pd.DataFrame(index=price.index)

    # VectorBT批量计算
    ma_result = vbt.MA.run(price, window=windows)

    # 提取结果并重命名
    result_df = pd.DataFrame(index=price.index)
    for i, window in enumerate(windows):
        if len(windows) == 1:
            result_df[f"MA{window}"] = ma_result.ma
        else:
            result_df[f"MA{window}"] = ma_result.ma.iloc[:, i]

    logger.debug(f"批量MA计算完成: {len(windows)} 个窗口")
    return result_df


def execute_ema_batch(price: pd.Series, spans: List[int]) -> pd.DataFrame:
    """批量计算指数移动平均线

    Args:
        price: 价格序列
        spans: 跨度列表

    Returns:
        DataFrame，每列为一个跨度的EMA结果
    """
    if not spans:
        return pd.DataFrame(index=price.index)

    result_df = pd.DataFrame(index=price.index)

    # pandas EMA批量计算（向量化）
    for span in spans:
        result_df[f"EMA{span}"] = price.ewm(span=span, adjust=False).mean()

    logger.debug(f"批量EMA计算完成: {len(spans)} 个跨度")
    return result_df


def execute_mstd_batch(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """批量计算移动标准差

    Args:
        price: 价格序列
        windows: 窗口列表

    Returns:
        DataFrame，每列为一个窗口的MSTD结果
    """
    if not windows:
        return pd.DataFrame(index=price.index)

    # VectorBT批量计算
    mstd_result = vbt.MSTD.run(price, window=windows)

    result_df = pd.DataFrame(index=price.index)
    for i, window in enumerate(windows):
        if len(windows) == 1:
            result_df[f"MSTD{window}"] = mstd_result.mstd
        else:
            result_df[f"MSTD{window}"] = mstd_result.mstd.iloc[:, i]

    logger.debug(f"批量MSTD计算完成: {len(windows)} 个窗口")
    return result_df


def execute_atr_batch(
    high: pd.Series, low: pd.Series, close: pd.Series, windows: List[int]
) -> pd.DataFrame:
    """批量计算平均真实范围

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        windows: 窗口列表

    Returns:
        DataFrame，每列为一个窗口的ATR结果
    """
    if not windows:
        return pd.DataFrame(index=close.index)

    result_df = pd.DataFrame(index=close.index)

    # VectorBT批量计算
    atr_result = vbt.ATR.run(high, low, close, window=windows)

    for i, window in enumerate(windows):
        if len(windows) == 1:
            result_df[f"ATR{window}"] = atr_result.atr
        else:
            result_df[f"ATR{window}"] = atr_result.atr.iloc[:, i]

    logger.debug(f"批量ATR计算完成: {len(windows)} 个窗口")
    return result_df


def execute_rolling_stats_batch(
    price: pd.Series, windows: List[int], stat_type: str
) -> pd.DataFrame:
    """批量计算滚动统计指标（FMAX/FMEAN/FMIN/FSTD）

    Args:
        price: 价格序列
        windows: 窗口列表
        stat_type: 统计类型 (max/mean/min/std)

    Returns:
        DataFrame，每列为一个窗口的统计结果
    """
    if not windows:
        return pd.DataFrame(index=price.index)

    result_df = pd.DataFrame(index=price.index)
    stat_func = getattr(price.rolling, stat_type)

    for window in windows:
        result_df[f"F{stat_type.upper()}{window}"] = stat_func(window=window)

    logger.debug(f"批量{stat_type.upper()}计算完成: {len(windows)} 个窗口")
    return result_df


def execute_position_batch(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """批量计算价格位置指标

    Args:
        price: 价格序列
        windows: 窗口列表

    Returns:
        DataFrame，每列为一个窗口的Position结果
    """
    if not windows:
        return pd.DataFrame(index=price.index)

    result_df = pd.DataFrame(index=price.index)

    for window in windows:
        rolling_min = price.rolling(window=window).min()
        rolling_max = price.rolling(window=window).max()
        position = (price - rolling_min) / (rolling_max - rolling_min + 1e-8)
        result_df[f"Position{window}"] = position

    logger.debug(f"批量Position计算完成: {len(windows)} 个窗口")
    return result_df


def execute_trend_batch(price: pd.Series, windows: List[int]) -> pd.DataFrame:
    """批量计算趋势强度指标

    Args:
        price: 价格序列
        windows: 窗口列表

    Returns:
        DataFrame，每列为一个窗口的Trend结果
    """
    if not windows:
        return pd.DataFrame(index=price.index)

    result_df = pd.DataFrame(index=price.index)

    for window in windows:
        rolling_mean = price.rolling(window=window).mean()
        rolling_std = price.rolling(window=window).std()
        trend = (price - rolling_mean) / (rolling_std + 1e-8)
        result_df[f"Trend{window}"] = trend

    logger.debug(f"批量Trend计算完成: {len(windows)} 个窗口")
    return result_df


def execute_momentum_batch(price: pd.Series, periods: List[int]) -> pd.DataFrame:
    """批量计算动量指标

    Args:
        price: 价格序列
        periods: 周期列表

    Returns:
        DataFrame，每列为一个周期的Momentum结果
    """
    if not periods:
        return pd.DataFrame(index=price.index)

    result_df = pd.DataFrame(index=price.index)

    for period in periods:
        momentum = price / price.shift(period) - 1
        result_df[f"Momentum{period}"] = momentum

    logger.debug(f"批量Momentum计算完成: {len(periods)} 个周期")
    return result_df


class BatchExecutor:
    """批量执行器 - 统一调度接口"""

    @staticmethod
    def execute(
        indicator_name: str, data: Dict[str, pd.Series], params: Dict[str, List[Any]]
    ) -> pd.DataFrame:
        """执行批量计算

        Args:
            indicator_name: 指标名称
            data: 数据字典 (price/high/low/volume等)
            params: 参数字典

        Returns:
            计算结果DataFrame
        """
        price = data.get("price") if "price" in data else data.get("close")

        if indicator_name == "MA":
            return execute_ma_batch(price, params.get("window", []))
        elif indicator_name == "EMA":
            return execute_ema_batch(price, params.get("span", []))
        elif indicator_name == "MSTD":
            return execute_mstd_batch(price, params.get("window", []))
        elif indicator_name == "ATR":
            return execute_atr_batch(
                data["high"], data["low"], price, params.get("window", [])
            )
        elif indicator_name == "Position":
            return execute_position_batch(price, params.get("window", []))
        elif indicator_name == "Trend":
            return execute_trend_batch(price, params.get("window", []))
        elif indicator_name == "Momentum":
            return execute_momentum_batch(price, params.get("period", []))
        elif indicator_name.startswith("F"):
            stat_type = indicator_name[1:].lower()
            return execute_rolling_stats_batch(
                price, params.get("window", []), stat_type
            )
        else:
            logger.warning(f"未知批量指标: {indicator_name}")
            return pd.DataFrame(index=price.index)
