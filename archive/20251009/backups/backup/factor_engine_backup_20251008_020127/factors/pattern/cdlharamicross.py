"""
CDLHARAMICROSS - Harami Cross Pattern"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class CDLHARAMICROSS(BaseFactor):
    """
    CDLHARAMICROSS - Harami Cross Pattern
    
    十字孕线模式
    
    返回值: 100=看涨, 0=无形态, -100=看跌
    """
    
    factor_id = "CDLHARAMICROSS"
    version = "v1.0"
    category = "pattern"
    description = "十字孕线模式"
    
    def __init__(self):
        super().__init__()
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib识别K线形态"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        pattern = talib.CDLHARAMICROSS(open_price, high, low, close)
        return pattern
