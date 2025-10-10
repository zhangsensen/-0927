"""

MINUS_DM - Minus Directional Movement"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class MINUS_DM(BaseFactor):
    """
    MINUS_DM - Minus Directional Movement
    
    下降方向运动
    """
    
    factor_id = "MINUS_DM"
    version = "v1.0"
    category = "technical"
    description = "下降方向运动"
    
    def __init__(self, period: int = 14):
        super().__init__(period=period)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算MINUS_DM"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data["high"]
        low = data["low"]
        close = data['close']
        
        
        result = talib.MINUS_DM(high, low, close, timeperiod=self.period)
        return result
