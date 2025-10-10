"""
HT_DCPHASE - Hilbert Transform - DC Phase"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class HT_DCPHASE(BaseFactor):
    """HT_DCPHASE - Hilbert Transform - DC Phase"""
    
    factor_id = "HT_DCPHASE"
    version = "v1.0"
    category = "statistic"
    description = "希尔伯特变换DC相位"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算HT_DCPHASE"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        result = talib.HT_DCPHASE(close)
        return result
