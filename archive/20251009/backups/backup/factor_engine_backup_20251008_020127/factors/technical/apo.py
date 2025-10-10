"""

APO - Absolute Price Oscillator"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class APO(BaseFactor):
    """
    APO - Absolute Price Oscillator

    绝对价格振荡器
    """

    factor_id = "APO"
    version = "v1.0"
    category = "technical"
    description = "绝对价格振荡器"

    def __init__(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0):
        super().__init__(fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.matype = matype

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算APO"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.APO(
            close,
            fastperiod=self.fastperiod,
            slowperiod=self.slowperiod,
            matype=self.matype,
        )
        return result
