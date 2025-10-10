"""
EMA - Exponential Moving Average"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter

class EMA(BaseFactor):
    """
    EMA - Exponential Moving Average

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    指数移动平均
    """

    factor_id = "EMA"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "overlap"
    description = "指数移动平均（VectorBT实现）"

    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算EMA - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            EMA values
        """
        close = data['close']

        # 使用VectorBT适配器计算EMA，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        ema = adapter.calculate_ema(close, window=self.period)

        return ema
