"""
CORREL - Pearson's Correlation Coefficient"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class CORREL(BaseFactor):
    """
    CORREL - Pearson's Correlation Coefficient
    
    皮尔逊相关系数
    """
    
    factor_id = "CORREL"
    version = "v1.0"
    category = "statistic"
    description = "皮尔逊相关系数"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算CORREL"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.CORREL(close, timeperiod=self.period)
        return result
