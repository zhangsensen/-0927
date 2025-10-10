"""

ROCP - Rate of Change Percentage"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class ROCP(BaseFactor):
    """
    ROCP - Rate of Change Percentage

    变动速率百分比
    """

    factor_id = "ROCP"
    version = "v1.0"
    category = "technical"
    description = "变动速率百分比"

    def __init__(self, period: int = 10):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算ROCP"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.ROCP(close, timeperiod=self.period)
        return result
