"""ADOSC - 技术指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


class ADOSC(BaseFactor):
    """
    ADOSC - 技术指标

    基于VectorBT实现，确保与factor_generation完全一致
    """

    factor_id = "ADOSC"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "technical"
    description = "ADOSC技术指标（VectorBT实现）"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ADOSC - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            ADOSC values
        """
        # 提取所需数据列
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        # 使用VectorBT适配器计算，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        return adapter.calculate_adosc(high, low, close, volume)