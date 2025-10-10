"""
SMA - Simple Moving Average"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


class SMA(BaseFactor):
    """
    SMA - Simple Moving Average

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    简单移动平均
    """

    factor_id = "SMA"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "overlap"
    description = "简单移动平均（VectorBT实现）"

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算SMA - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            SMA values
        """
        close = data["close"]

        # 使用VectorBT适配器计算SMA，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        sma = adapter.calculate_sma(close, window=self.period)

        return sma
