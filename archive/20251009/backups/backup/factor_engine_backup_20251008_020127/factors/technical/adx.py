"""
ADX - Average Directional Movement Index"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


class ADX(BaseFactor):
    """
    ADX - Average Directional Movement Index

    平均趋向指标（VectorBT实现，与factor_generation完全一致）
    """

    factor_id = "ADX"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "technical"
    description = "平均趋向指标（VectorBT实现）"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ADX - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            ADX values
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 使用VectorBT适配器计算ADX，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        adx = adapter.calculate_adx(high, low, close, timeperiod=self.period)

        return adx
