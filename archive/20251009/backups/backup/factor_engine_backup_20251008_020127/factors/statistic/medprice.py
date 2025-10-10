"""
MEDPRICE - Median Price"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class MEDPRICE(BaseFactor):
    """MEDPRICE - Median Price"""

    factor_id = "MEDPRICE"
    version = "v1.0"
    category = "statistic"
    description = "中位价"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算MEDPRICE"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        high = data["high"]
        low = data["low"]

        result = talib.MEDPRICE(high, low)
        return result
