"""
LINEARREG_ANGLE - Linear Regression Angle"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class LINEARREG_ANGLE(BaseFactor):
    """
    LINEARREG_ANGLE - Linear Regression Angle
    
    线性回归角度
    """
    
    factor_id = "LINEARREG_ANGLE"
    version = "v1.0"
    category = "statistic"
    description = "线性回归角度"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算LINEARREG_ANGLE"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.LINEARREG_ANGLE(close, timeperiod=self.period)
        return result
