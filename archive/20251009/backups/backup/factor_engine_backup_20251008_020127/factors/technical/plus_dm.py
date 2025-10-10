"""

PLUS_DM - Plus Directional Movement"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class PLUS_DM(BaseFactor):
    """
    PLUS_DM - Plus Directional Movement

    上升方向运动
    """

    factor_id = "PLUS_DM"
    version = "v1.0"
    category = "technical"
    description = "上升方向运动"

    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算PLUS_DM"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        result = talib.PLUS_DM(high, low, close, timeperiod=self.period)
        return result
