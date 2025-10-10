"""
CDLENGULFING - Engulfing Pattern"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class CDLENGULFING(BaseFactor):
    """
    CDLENGULFING - Engulfing Pattern

    吞噬模式

    返回值: 100=看涨, 0=无形态, -100=看跌
    """

    factor_id = "CDLENGULFING"
    version = "v1.0"
    category = "pattern"
    description = "吞噬模式"

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

        pattern = talib.CDLENGULFING(open_price, high, low, close)
        return pattern
