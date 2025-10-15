"""
数据处理工具函数

Linus准则：
- 简洁、向量化、无冗余
- 固定schema，统一时区
"""

from __future__ import annotations

import pandas as pd


def resample_to_daily(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    将分钟级价格数据重采样为日线

    Args:
        df: 包含datetime索引和价格列的DataFrame
        price_col: 价格列名（默认'close'）

    Returns:
        日线重采样后的DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # 重采样到日线：取每日最后一个值（收盘价）
    daily = df.resample("D").last()

    # 移除全NaN行（非交易日）
    daily = daily.dropna(how="all")

    return daily


def align_money_flow_with_price(
    money_flow_df: pd.DataFrame, price_df: pd.DataFrame, price_col: str = "close"
) -> pd.DataFrame:
    """
    对齐资金流数据和价格数据

    Args:
        money_flow_df: 资金流数据（日线，已设置DatetimeIndex）
        price_df: 价格数据（可能是分钟级，需要重采样）
        price_col: 价格列名

    Returns:
        合并后的DataFrame，包含资金流和价格数据
    """
    # 如果价格数据不是日线，重采样
    if isinstance(price_df.index, pd.DatetimeIndex):
        # 检查是否是日内数据（有时分秒）
        if price_df.index[0].hour != 0 or price_df.index[0].minute != 0:
            # 重采样到日线
            price_daily = resample_to_daily(price_df, price_col)
        else:
            price_daily = price_df
    else:
        raise ValueError("price_df must have DatetimeIndex")

    # 合并数据（左连接，保留所有资金流日期）
    merged = money_flow_df.join(price_daily[[price_col]], how="left")

    return merged
