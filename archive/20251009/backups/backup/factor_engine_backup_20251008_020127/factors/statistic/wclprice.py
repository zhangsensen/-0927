"""
WCLPRICE - Weighted Close Price"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class WCLPRICE(BaseFactor):
    """WCLPRICE - Weighted Close Price"""
    
    factor_id = "WCLPRICE"
    version = "v1.0"
    category = "statistic"
    description = "加权收盘价"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算WCLPRICE"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = talib.WCLPRICE(high, low, close)
        return result
