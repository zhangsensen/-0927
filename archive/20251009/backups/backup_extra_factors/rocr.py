"""

ROCR - Rate of Change Ratio"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class ROCR(BaseFactor):
    """
    ROCR - Rate of Change Ratio
    
    变动速率比率
    """
    
    factor_id = "ROCR"
    version = "v1.0"
    category = "technical"
    description = "变动速率比率"
    
    def __init__(self, period: int = 10):
        super().__init__(period=period)
        self.period = period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算ROCR"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        
        
        close = data['close']
        
        
        result = talib.ROCR(close, timeperiod=self.period)
        return result
