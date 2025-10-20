"""STOCH - 随机指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


class STOCH(BaseFactor):
    """
    STOCH - Stochastic Oscillator

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    计算公式:
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA(%K, period)
    """

    factor_id = "STOCH"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "随机指标（共享计算器实现）"

    def __init__(
        self,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ):
        """
        初始化STOCH

        Args:
            fastk_period: Fast %K周期，默认14
            slowk_period: Slow %K平滑周期，默认3
            slowd_period: %D平滑周期，默认3
        """
        super().__init__(
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period,
        )
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算STOCH %K - 使用共享计算器确保全系统一致

        Args:
            data: DataFrame with 'high', 'low', 'close' columns

        Returns:
            STOCH %K values (Slow %K)
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 使用共享计算器计算STOCH，确保与factor_generation、hk_midfreq完全一致
        stoch_result = SHARED_CALCULATORS.calculate_stoch(
            high,
            low,
            close,
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowd_period=self.slowd_period,
        )

        return stoch_result["slowk"]

