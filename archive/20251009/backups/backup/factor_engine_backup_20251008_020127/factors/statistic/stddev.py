"""
STDDEV - Standard Deviation"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class STDDEV(BaseFactor):
    """
    STDDEV - Standard Deviation
    
    标准差
    """
    
    factor_id = "STDDEV"
    version = "v1.0"
    category = "statistic"
    description = "标准差"
    
    def __init__(self, period: int = 5, nbdev: float = 1.0):
        self.period = period
        self.nbdev = nbdev
        super().__init__(timeperiod=period, nbdev=nbdev)
        self.period = period
        self.nbdev = nbdev
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算STDDEV"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.STDDEV(close, timeperiod=self.period, nbdev=nbdev)
        return result
