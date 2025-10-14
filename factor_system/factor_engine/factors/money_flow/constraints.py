"""
资金流硬约束因子

包含：
- 跳空信号（gap_sig）
- 四段小时K
- 尾盘抢筹比率（tail30_ratio）
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_gap_signal(
    data: pd.DataFrame, threshold_sigma: float = 1.8, window: int = 20
) -> pd.Series:
    """
    跳空信号 - 向量化

    gap_pct = open_t / close_{t-1} - 1
    标准化：20D滚动σ，阈值1.8σ

    Returns:
        三值信号 {-1, 0, 1}
    """
    if "open" not in data.columns or "close" not in data.columns:
        return pd.Series(0, index=data.index, name="gap_sig")

    # 向量化计算跳空
    gap_pct = data["open"] / data["close"].shift(1) - 1

    # 20D滚动标准差
    gap_std = gap_pct.rolling(window).std()
    gap_zscore = gap_pct / np.maximum(gap_std, 1e-6)

    # 三值信号
    signal = np.zeros(len(data))
    signal[gap_zscore > threshold_sigma] = 1
    signal[gap_zscore < -threshold_sigma] = -1

    # 排除除权日（TODO: 需要复权标记）
    # signal[data['is_ex_dividend']] = 0

    # 排除一字板（TODO: 需要涨跌停标记）
    # signal[data['is_limit']] = 0

    return pd.Series(signal, index=data.index, name="gap_sig")


def calculate_tail30_ratio(
    minute_data: pd.DataFrame, include_auction: bool = False, zscore_window: int = 60
) -> pd.Series:
    """
    尾盘抢筹比率 - 基于实际分钟数据

    tail30_vol_ratio = sum(volume[14:30-15:00]) / sum(all session volume)

    Args:
        minute_data: 分钟级数据，必须包含datetime索引和volume列
        include_auction: 是否包含收盘竞价（14:57-15:00）
        zscore_window: Z-score标准化窗口

    Returns:
        尾盘成交占比的zscore序列
    """
    if minute_data.empty or "volume" not in minute_data.columns:
        return pd.Series(dtype=float, name="tail30_ratio")

    # 确保索引为DatetimeIndex
    if not isinstance(minute_data.index, pd.DatetimeIndex):
        if "datetime" in minute_data.columns:
            minute_data = minute_data.set_index("datetime")
        else:
            raise ValueError("minute_data must have datetime index")

    daily_groups = minute_data.groupby(minute_data.index.date)
    results = {}

    for date, day_data in daily_groups:
        # 定义尾盘时段
        if include_auction:
            tail_start = pd.Timestamp.combine(date, pd.Timestamp("14:30").time())
            tail_end = pd.Timestamp.combine(date, pd.Timestamp("15:00").time())
        else:
            # 排除收盘竞价
            tail_start = pd.Timestamp.combine(date, pd.Timestamp("14:30").time())
            tail_end = pd.Timestamp.combine(date, pd.Timestamp("14:57").time())

        tail_mask = (day_data.index >= tail_start) & (day_data.index <= tail_end)
        tail_volume = day_data.loc[tail_mask, "volume"].sum()
        total_volume = day_data["volume"].sum()

        ratio = tail_volume / total_volume if total_volume > 0 else 0
        results[pd.Timestamp(date)] = ratio

    # 转换为Series
    ratio_series = pd.Series(results, name="tail30_ratio")

    # Z-score标准化
    if len(ratio_series) >= zscore_window:
        ratio_mean = ratio_series.rolling(window=zscore_window, min_periods=1).mean()
        ratio_std = ratio_series.rolling(window=zscore_window, min_periods=1).std()
        ratio_zscore = (ratio_series - ratio_mean) / np.maximum(ratio_std, 1e-6)
        return ratio_zscore
    else:
        return ratio_series


def generate_hourly_4seg(minute_data: pd.DataFrame) -> pd.DataFrame:
    """
    四段小时K - 基于实际分钟数据

    固定四段：
    - 09:30-10:29
    - 10:30-11:29
    - 13:00-13:59
    - 14:00-14:59

    Args:
        minute_data: 分钟级数据，必须包含datetime索引和OHLCV列

    Returns:
        四段小时K DataFrame，每天4条记录
    """
    if minute_data.empty:
        return pd.DataFrame()

    # 确保索引为DatetimeIndex
    if not isinstance(minute_data.index, pd.DatetimeIndex):
        if "datetime" in minute_data.columns:
            minute_data = minute_data.set_index("datetime")
        else:
            raise ValueError("minute_data must have datetime index")

    # 定义四个时段
    segments = [
        ("09:30", "10:29", "seg1_morning_early"),
        ("10:30", "11:29", "seg2_morning_late"),
        ("13:00", "13:59", "seg3_afternoon_early"),
        ("14:00", "14:59", "seg4_afternoon_late"),
    ]

    all_segments = []
    daily_groups = minute_data.groupby(minute_data.index.date)

    for date, day_data in daily_groups:
        for start_time, end_time, seg_name in segments:
            start_dt = pd.Timestamp.combine(date, pd.Timestamp(start_time).time())
            end_dt = pd.Timestamp.combine(date, pd.Timestamp(end_time).time())

            seg_data = day_data[(day_data.index >= start_dt) & (day_data.index <= end_dt)]

            if not seg_data.empty:
                segment_info = {
                    "datetime": end_dt,  # 使用段结束时间
                    "date": pd.Timestamp(date),
                    "segment_name": seg_name,
                    "segment_period": f"{start_time}-{end_time}",
                    "open": seg_data["open"].iloc[0],
                    "high": seg_data["high"].max(),
                    "low": seg_data["low"].min(),
                    "close": seg_data["close"].iloc[-1],
                    "volume": seg_data["volume"].sum(),
                    "turnover": seg_data["turnover"].sum(),
                    "bar_count": len(seg_data),  # 该时段的分钟K线数量
                }
                all_segments.append(segment_info)

    result = pd.DataFrame(all_segments)
    if not result.empty:
        result.set_index("datetime", inplace=True)

    return result


def apply_tradability_constraints(
    data: pd.DataFrame, gap_sig: pd.Series, tail30_ratio: pd.Series
) -> pd.DataFrame:
    """
    应用交易性约束

    Args:
        data: 原始数据
        gap_sig: 跳空信号
        tail30_ratio: 尾盘抢筹比率

    Returns:
        带约束标记的DataFrame
    """
    data = data.copy()
    data["gap_sig"] = gap_sig
    data["tail30_ratio"] = tail30_ratio

    # 跳空触发标记
    data["gap_triggered"] = (gap_sig != 0).astype(int)

    # 尾盘抢筹触发标记（阈值可配置）
    data["tail30_triggered"] = (np.abs(tail30_ratio) > 1.5).astype(int)

    return data
