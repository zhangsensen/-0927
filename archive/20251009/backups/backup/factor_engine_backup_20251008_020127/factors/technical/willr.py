"""WILLR - 威廉指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter

class WILLR(BaseFactor):
    """
    WILLR - Williams %R

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    计算公式:
    %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)
    """

    factor_id = "WILLR"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "technical"
    description = "威廉指标（VectorBT实现）"

    def __init__(self, period: int = 14):
        """
        初始化WILLR

        Args:
            period: 计算周期，默认14
        """
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算WILLR - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            WILLR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # 使用VectorBT适配器计算WILLR，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        willr = adapter.calculate_willr(high, low, close, timeperiod=self.period)

        return willr
