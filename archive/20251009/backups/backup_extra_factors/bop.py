"""
BOP - Balance Of Power"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class BOP(BaseFactor):
    """BOP - Balance Of Power"""

    factor_id = "BOP"
    version = "v1.0"
    category = "technical"
    description = "力量平衡"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算BOP"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        open_price = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        result = talib.BOP(open_price, high, low, close)
        return result
