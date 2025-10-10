"""ROC - 技术指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


class ROC(BaseFactor):
    """
    ROC - Rate of Change

    基于共享计算器实现，确保与factor_generation完全一致
    """

    factor_id = "ROC"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "ROC技术指标（共享计算器实现）"

    def __init__(self, period: int = 14):
        super().__init__(timeperiod=period)
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ROC - 使用共享计算器确保与factor_generation一致

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            ROC values
        """
        # 提取所需数据列
        close = data["close"]

        # 使用共享计算器计算，确保与factor_generation完全一致
        return SHARED_CALCULATORS.calculate_roc(close, period=self.period)
