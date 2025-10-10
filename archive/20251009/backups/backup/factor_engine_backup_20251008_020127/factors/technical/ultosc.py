"""

ULTOSC - Ultimate Oscillator"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class ULTOSC(BaseFactor):
    """
    ULTOSC - Ultimate Oscillator
    
    终极振荡器
    """
    
    factor_id = "ULTOSC"
    version = "v1.0"
    category = "technical"
    description = "终极振荡器"
    
    def __init__(self, timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28):
        super().__init__(timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        self.timeperiod1 = timeperiod1
        self.timeperiod2 = timeperiod2
        self.timeperiod3 = timeperiod3
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算ULTOSC"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data["high"]
        low = data["low"]
        close = data['close']
        
        
        result = talib.ULTOSC(high, low, close, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        return result
