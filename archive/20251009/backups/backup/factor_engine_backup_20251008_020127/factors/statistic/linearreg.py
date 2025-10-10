"""
LINEARREG - Linear Regression"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class LINEARREG(BaseFactor):
    """
    LINEARREG - Linear Regression
    
    线性回归
    """
    
    factor_id = "LINEARREG"
    version = "v1.0"
    category = "statistic"
    description = "线性回归"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算LINEARREG"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.LINEARREG(close, timeperiod=self.period)
        return result
