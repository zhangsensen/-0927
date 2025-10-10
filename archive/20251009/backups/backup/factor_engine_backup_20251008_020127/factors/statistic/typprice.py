"""
TYPPRICE - Typical Price"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class TYPPRICE(BaseFactor):
    """TYPPRICE - Typical Price"""

    factor_id = "TYPPRICE"
    version = "v1.0"
    category = "statistic"
    description = "典型价格"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算TYPPRICE"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        result = talib.TYPPRICE(high, low, close)
        return result
