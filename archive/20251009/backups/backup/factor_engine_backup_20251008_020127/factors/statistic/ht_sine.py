"""
HT_SINE - Hilbert Transform - Sine"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class HT_SINE(BaseFactor):
    """HT_SINE - Hilbert Transform - Sine"""

    factor_id = "HT_SINE"
    version = "v1.0"
    category = "statistic"
    description = "希尔伯特变换正弦波"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算HT_SINE"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        sine, leadsine = talib.HT_SINE(close)
        return sine
