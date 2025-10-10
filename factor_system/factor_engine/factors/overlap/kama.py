"""
KAMA - Kaufman Adaptive Moving Average"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class KAMA(BaseFactor):
    """
    KAMA - Kaufman Adaptive Moving Average
    
    考夫曼自适应移动平均
    """
    
    factor_id = "KAMA"
    version = "v1.0"
    category = "overlap"
    description = "考夫曼自适应移动平均"
    
    def __init__(self, period: int = 20):
        self.period = period
        super().__init__(period=period)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算KAMA"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.KAMA(close, timeperiod=self.period)
        return result
