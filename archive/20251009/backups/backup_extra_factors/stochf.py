"""
STOCHF - Stochastic Fast"""

from __future__ import annotations

import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

from factor_system.factor_engine.core.base_factor import BaseFactor


class STOCHF(BaseFactor):
    """
    STOCHF - Stochastic Fast
    
    快速随机指标
    """
    
    factor_id = "STOCHF"
    version = "v1.0"
    category = "technical"
    description = "快速随机指标"
    
    def __init__(self, fastk_period: int = 5, fastd_period: int = 3):
        super().__init__(fastk_period=fastk_period, fastd_period=fastd_period)
        self.fastk_period = fastk_period
        self.fastd_period = fastd_period
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """使用TA-Lib计算STOCHF"""
        if not HAS_TALIB:
            import numpy as np
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        result = fastk, fastd = talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=fastd_period)
        return fastk  # 返回%K
        return result
