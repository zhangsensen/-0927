"""CCI - Commodity Channel Index"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


class CCI(BaseFactor):
    """
    CCI - Commodity Channel Index

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    计算公式:
    CCI = (Typical Price - SMA(Typical Price)) / (0.015 * Mean Deviation)
    """

    factor_id = "CCI"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "technical"
    description = "商品通道指标（VectorBT实现）"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算CCI - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            CCI values
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 使用VectorBT适配器计算CCI，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        cci = adapter.calculate_cci(high, low, close, timeperiod=self.period)

        return cci
