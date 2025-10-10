"""
VAR - Variance"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class VAR(BaseFactor):
    """
    VAR - Variance

    方差
    """

    factor_id = "VAR"
    version = "v1.0"
    category = "statistic"
    description = "方差"

    def __init__(self, period: int = 5, nbdev: float = 1.0):
        self.period = period
        self.nbdev = nbdev
        super().__init__(timeperiod=period, nbdev=nbdev)
        self.period = period
        self.nbdev = nbdev

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算VAR"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.VAR(close, timeperiod=self.period, nbdev=nbdev)
        return result
