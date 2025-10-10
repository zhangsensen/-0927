"""
CDL3STARSINSOUTH - Three Stars In The South"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class CDL3STARSINSOUTH(BaseFactor):
    """
    CDL3STARSINSOUTH - Three Stars In The South

    南方三星

    返回值: 100=看涨, 0=无形态, -100=看跌
    """

    factor_id = "CDL3STARSINSOUTH"
    version = "v1.0"
    category = "pattern"
    description = "南方三星"

    def __init__(self):
        super().__init__()

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib识别K线形态"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        open_price = data["open"]
        high = data["high"]
        low = data["low"]
        close = data["close"]

        pattern = talib.CDL3STARSINSOUTH(open_price, high, low, close)
        return pattern
