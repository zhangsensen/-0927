"""
BBANDS - Bollinger Bands"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.factor_engine.core.vectorbt_adapter import get_vectorbt_adapter


class BBANDS(BaseFactor):
    """
    BBANDS - Bollinger Bands

    布林带（VectorBT实现，与factor_generation完全一致）
    """

    factor_id = "BBANDS"
    version = "v2.0"  # 升级版本，基于VectorBT
    category = "overlap"
    description = "布林带（VectorBT实现）"

    def __init__(self, period: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0):
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        super().__init__(timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算BBANDS - 使用VectorBT确保与factor_generation一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            BBANDS中轨值
        """
        close = data["close"]

        # 使用VectorBT适配器计算BBANDS，确保与factor_generation完全一致
        adapter = get_vectorbt_adapter()
        bbands_result = adapter.calculate_bbands(
            close, window=self.period, nbdevup=self.nbdevup, nbdevdn=self.nbdevdn
        )

        # 返回中轨
        return bbands_result["BBANDS_MIDDLE"]
