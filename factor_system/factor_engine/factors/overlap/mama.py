"""
MAMA - MESA Adaptive Moving Average"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class MAMA(BaseFactor):
    """
    MAMA - MESA Adaptive Moving Average

    MESA自适应移动平均
    """

    factor_id = "MAMA"
    version = "v1.0"
    category = "overlap"
    description = "MESA自适应移动平均"

    def __init__(self, fastlimit: float = 0.5, slowlimit: float = 0.05):
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit
        super().__init__(fastlimit=fastlimit, slowlimit=slowlimit)
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算MAMA"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        mama, fama = talib.MAMA(
            close, fastlimit=self.fastlimit, slowlimit=self.slowlimit
        )
        return mama  # 返回MAMA线
