"""
SAREXT - Parabolic SAR Extended"""

from __future__ import annotations

import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class SAREXT(BaseFactor):
    """SAREXT - Parabolic SAR Extended"""

    factor_id = "SAREXT"
    version = "v1.0"
    category = "overlap"
    description = "扩展抛物线转向"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算SAREXT"""
        if not HAS_TALIB:
            import numpy as np

            return pd.Series(np.nan, index=data.index)

        high = data["high"]
        low = data["low"]

        result = talib.SAREXT(high, low)
        return result
