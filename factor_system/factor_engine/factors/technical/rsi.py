"""RSI - 相对强弱指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


class RSI(BaseFactor):
    """
    RSI - Relative Strength Index

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    计算公式:
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))
    """

    factor_id = "RSI"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "相对强弱指标（共享计算器实现）"

    def __init__(self, period: int = 14, timeperiod: int = None, **kwargs):
        """
        初始化RSI

        Args:
            period: 计算周期，默认14
            timeperiod: 兼容参数（与period相同）
        """
        # 兼容两种参数命名
        if timeperiod is not None:
            period = timeperiod
        super().__init__(period=period, **kwargs)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算RSI - 使用共享计算器确保全系统一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            RSI values
        """
        close = data["close"]

        # 使用共享计算器计算RSI，确保与factor_generation、hk_midfreq完全一致
        rsi = SHARED_CALCULATORS.calculate_rsi(close, period=self.period)

        return rsi
