"""
BETA - Beta"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class BETA(BaseFactor):
    """
    BETA - Beta
    
    贝塔系数
    """
    
    factor_id = "BETA"
    version = "v1.0"
    category = "statistic"
    description = "贝塔系数"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算BETA"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        
        result = talib.BETA(high, low, timeperiod=self.period)
        return result
