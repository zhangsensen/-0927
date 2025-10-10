"""

AROONOSC - Aroon Oscillator"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class AROONOSC(BaseFactor):
    """
    AROONOSC - Aroon Oscillator

    阿隆振荡器
    """

    factor_id = "AROONOSC"
    version = "v1.0"
    category = "technical"
    description = "阿隆振荡器"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算AROONOSC"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        result = talib.AROONOSC(high, low, close, timeperiod=self.period)
        return result
