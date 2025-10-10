"""
HT_PHASOR - Hilbert Transform - Phasor"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class HT_PHASOR(BaseFactor):
    """HT_PHASOR - Hilbert Transform - Phasor"""
    
    factor_id = "HT_PHASOR"
    version = "v1.0"
    category = "statistic"
    description = "希尔伯特变换相量"
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算HT_PHASOR"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        close = data['close']
        
        inphase, quadrature = talib.HT_PHASOR(close)
        return inphase
