"""
T3 - Triple Exponential Moving Average T3"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class T3(BaseFactor):
    """
    T3 - Triple Exponential Moving Average T3

    T3移动平均
    """

    factor_id = "T3"
    version = "v1.0"
    category = "overlap"
    description = "T3移动平均"

    def __init__(self, period: int = 5, vfactor: float = 0.7):
        self.period = period
        self.vfactor = vfactor
        super().__init__(period=period, vfactor=vfactor)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算T3"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.T3(close, timeperiod=self.period, vfactor=self.vfactor)
        return result
