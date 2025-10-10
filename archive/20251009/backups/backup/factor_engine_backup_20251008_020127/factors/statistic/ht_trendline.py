"""
HT_TRENDLINE - Hilbert Transform - Trendline"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class HT_TRENDLINE(BaseFactor):
    """HT_TRENDLINE - Hilbert Transform - Trendline"""

    factor_id = "HT_TRENDLINE"
    version = "v1.0"
    category = "statistic"
    description = "希尔伯特变换趋势线"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算HT_TRENDLINE"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        close = data["close"]

        result = talib.HT_TRENDLINE(close)
        return result
