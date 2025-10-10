"""

TRIX - Triple Exponential Moving Average"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class TRIX(BaseFactor):
    """
    TRIX - Triple Exponential Moving Average

    三重指数移动平均
    """

    factor_id = "TRIX"
    version = "v1.0"
    category = "technical"
    description = "三重指数移动平均"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算TRIX"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.TRIX(close, timeperiod=self.period)
        return result
