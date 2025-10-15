"""
日内资金流因子

基于分钟级数据的日内因子：
- 成交量爆发
- 价格突破确认
- 日内动量
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor


class IntradayVolumeSurge(BaseFactor):
    """
    日内成交量爆发因子

    计算每日成交量相对于日均值的倍数
    """

    factor_id = "IntradayVolumeSurge"
    version = "v1.0"
    category = "money_flow_intraday"
    description = "日内成交量爆发倍数"

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        super().__init__(lookback_days=lookback_days)

    def calculate(self, minute_data: pd.DataFrame) -> pd.Series:
        """
        计算日内成交量爆发

        Args:
            minute_data: 分钟级数据

        Returns:
            每日成交量爆发倍数
        """
        if minute_data.empty or "volume" not in minute_data.columns:
            return pd.Series(dtype=float, name=self.factor_id)

        # 按日聚合成交量
        daily_volume = minute_data.groupby(minute_data.index.date)["volume"].sum()

        # 计算移动平均
        volume_ma = daily_volume.rolling(
            window=self.lookback_days, min_periods=1
        ).mean()

        # 计算爆发倍数
        surge_ratio = daily_volume / np.maximum(volume_ma, 1e-6)

        surge_ratio.index = pd.to_datetime(surge_ratio.index)
        return surge_ratio.rename(self.factor_id)


class IntradayPriceBreakout(BaseFactor):
    """
    日内价格突破确认因子

    检测价格突破前高/前低并持续的情况
    """

    factor_id = "IntradayPriceBreakout"
    version = "v1.0"
    category = "money_flow_intraday"
    description = "日内价格突破确认"

    def __init__(self, lookback_days: int = 5, confirmation_minutes: int = 30):
        self.lookback_days = lookback_days
        self.confirmation_minutes = confirmation_minutes
        super().__init__(
            lookback_days=lookback_days, confirmation_minutes=confirmation_minutes
        )

    def calculate(self, minute_data: pd.DataFrame) -> pd.Series:
        """
        计算日内价格突破

        Args:
            minute_data: 分钟级数据

        Returns:
            突破信号序列（1=向上突破，-1=向下突破，0=无突破）
        """
        if minute_data.empty:
            return pd.Series(dtype=float, name=self.factor_id)

        required_cols = ["high", "low", "close"]
        if not all(col in minute_data.columns for col in required_cols):
            return pd.Series(dtype=float, name=self.factor_id)

        daily_groups = minute_data.groupby(minute_data.index.date)
        breakout_signals = {}

        for date, day_data in daily_groups:
            if len(day_data) < self.confirmation_minutes:
                breakout_signals[pd.Timestamp(date)] = 0
                continue

            # 获取前N日的最高价和最低价
            prev_dates = [
                d
                for d in daily_groups.groups.keys()
                if d < date and (date - d).days <= self.lookback_days
            ]

            if not prev_dates:
                breakout_signals[pd.Timestamp(date)] = 0
                continue

            prev_data = pd.concat([daily_groups.get_group(d) for d in prev_dates])
            prev_high = prev_data["high"].max()
            prev_low = prev_data["low"].min()

            # 检查当日是否突破
            day_high = day_data["high"].max()
            day_low = day_data["low"].min()
            day_close = day_data["close"].iloc[-1]

            # 向上突破：突破前高且收盘在前高之上
            if day_high > prev_high and day_close > prev_high:
                breakout_signals[pd.Timestamp(date)] = 1
            # 向下突破：突破前低且收盘在前低之下
            elif day_low < prev_low and day_close < prev_low:
                breakout_signals[pd.Timestamp(date)] = -1
            else:
                breakout_signals[pd.Timestamp(date)] = 0

        return pd.Series(breakout_signals, name=self.factor_id)


class IntradayMomentum(BaseFactor):
    """
    日内动量因子

    计算日内不同时段的价格动量
    """

    factor_id = "IntradayMomentum"
    version = "v1.0"
    category = "money_flow_intraday"
    description = "日内动量（上午vs下午）"

    def calculate(self, minute_data: pd.DataFrame) -> pd.Series:
        """
        计算日内动量

        Args:
            minute_data: 分钟级数据

        Returns:
            日内动量序列（下午收益 - 上午收益）
        """
        if minute_data.empty or "close" not in minute_data.columns:
            return pd.Series(dtype=float, name=self.factor_id)

        daily_groups = minute_data.groupby(minute_data.index.date)
        momentum_values = {}

        for date, day_data in daily_groups:
            # 分割上午和下午数据
            morning_mask = day_data.index.time < pd.Timestamp("12:00").time()
            afternoon_mask = day_data.index.time >= pd.Timestamp("13:00").time()

            morning_data = day_data[morning_mask]
            afternoon_data = day_data[afternoon_mask]

            if morning_data.empty or afternoon_data.empty:
                momentum_values[pd.Timestamp(date)] = 0
                continue

            # 计算上午和下午的收益率
            morning_ret = (
                morning_data["close"].iloc[-1] / morning_data["close"].iloc[0] - 1
            )
            afternoon_ret = (
                afternoon_data["close"].iloc[-1] / afternoon_data["close"].iloc[0] - 1
            )

            # 动量 = 下午收益 - 上午收益
            momentum = afternoon_ret - morning_ret
            momentum_values[pd.Timestamp(date)] = momentum

        return pd.Series(momentum_values, name=self.factor_id)
