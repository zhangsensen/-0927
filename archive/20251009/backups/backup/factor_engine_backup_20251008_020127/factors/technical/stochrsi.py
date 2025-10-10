"""
STOCHRSI - Stochastic RSI"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class STOCHRSI(BaseFactor):
    """
    STOCHRSI - Stochastic RSI

    随机RSI
    """

    factor_id = "STOCHRSI"
    version = "v1.0"
    category = "technical"
    description = "随机RSI"

    def __init__(
        self, timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3
    ):
        super().__init__(
            timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period
        )
        self.timeperiod = timeperiod
        self.fastk_period = fastk_period
        self.fastd_period = fastd_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算STOCHRSI"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = fastk, fastd = talib.STOCHRSI(
            close,
            timeperiod=timeperiod,
            fastk_period=fastk_period,
            fastd_period=fastd_period,
        )
        return fastk  # 返回%K
        return result
