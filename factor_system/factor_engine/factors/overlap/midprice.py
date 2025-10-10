"""
MIDPRICE - Midpoint Price over period"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class MIDPRICE(BaseFactor):
    """
    MIDPRICE - Midpoint Price over period
    
    中间价
    """
    
    factor_id = "MIDPRICE"
    version = "v1.0"
    category = "overlap"
    description = "中间价"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算MIDPRICE"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = talib.MIDPRICE(high, low, timeperiod=self.period)
        return result
