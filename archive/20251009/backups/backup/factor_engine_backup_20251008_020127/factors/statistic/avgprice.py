"""
AVGPRICE - Average Price"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class AVGPRICE(BaseFactor):
    """AVGPRICE - Average Price"""
    
    factor_id = "AVGPRICE"
    version = "v1.0"
    category = "statistic"
    description = "平均价格"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算AVGPRICE"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = talib.AVGPRICE(open_price, high, low, close)
        return result
