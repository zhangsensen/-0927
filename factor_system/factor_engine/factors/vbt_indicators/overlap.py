"""
重叠指标（布林带等） - 基于VectorBT实现
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ...core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class BBANDS(BaseFactor):
    """布林带指标"""
    factor_id = "BBANDS"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # 返回布林带宽度作为综合指标
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            bb_width = (result.upper_band - result.lower_band) / result.middle_band
            return bb_width.rename("BBANDS")
        except Exception as e:
            logger.error(f"计算BBANDS失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BBANDS")


class BB_10_2_0_Upper(BaseFactor):
    """10日2倍标准差布林带上轨"""
    factor_id = "BB_10_2.0_Upper"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=10, alpha=2.0)
            return result.upper_band.rename("BB_10_2.0_Upper")
        except Exception as e:
            logger.error(f"计算BB_10_2.0_Upper失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_10_2.0_Upper")


class BB_10_2_0_Middle(BaseFactor):
    """10日2倍标准差布林带中轨"""
    factor_id = "BB_10_2.0_Middle"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=10, alpha=2.0)
            return result.middle_band.rename("BB_10_2.0_Middle")
        except Exception as e:
            logger.error(f"计算BB_10_2.0_Middle失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_10_2.0_Middle")


class BB_10_2_0_Lower(BaseFactor):
    """10日2倍标准差布林带下轨"""
    factor_id = "BB_10_2.0_Lower"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=10, alpha=2.0)
            return result.lower_band.rename("BB_10_2.0_Lower")
        except Exception as e:
            logger.error(f"计算BB_10_2.0_Lower失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_10_2.0_Lower")


class BB_10_2_0_Width(BaseFactor):
    """10日2倍标准差布林带宽度"""
    factor_id = "BB_10_2.0_Width"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=10, alpha=2.0)
            bb_width = (result.upper_band - result.lower_band) / result.middle_band
            return bb_width.rename("BB_10_2.0_Width")
        except Exception as e:
            logger.error(f"计算BB_10_2.0_Width失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_10_2.0_Width")


class BB_15_2_0_Upper(BaseFactor):
    """15日2倍标准差布林带上轨"""
    factor_id = "BB_15_2.0_Upper"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=15, alpha=2.0)
            return result.upper_band.rename("BB_15_2.0_Upper")
        except Exception as e:
            logger.error(f"计算BB_15_2.0_Upper失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_15_2.0_Upper")


class BB_15_2_0_Middle(BaseFactor):
    """15日2倍标准差布林带中轨"""
    factor_id = "BB_15_2.0_Middle"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=15, alpha=2.0)
            return result.middle_band.rename("BB_15_2.0_Middle")
        except Exception as e:
            logger.error(f"计算BB_15_2.0_Middle失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_15_2.0_Middle")


class BB_15_2_0_Lower(BaseFactor):
    """15日2倍标准差布林带下轨"""
    factor_id = "BB_15_2.0_Lower"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=15, alpha=2.0)
            return result.lower_band.rename("BB_15_2.0_Lower")
        except Exception as e:
            logger.error(f"计算BB_15_2.0_Lower失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_15_2.0_Lower")


class BB_15_2_0_Width(BaseFactor):
    """15日2倍标准差布林带宽度"""
    factor_id = "BB_15_2.0_Width"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=15, alpha=2.0)
            bb_width = (result.upper_band - result.lower_band) / result.middle_band
            return bb_width.rename("BB_15_2.0_Width")
        except Exception as e:
            logger.error(f"计算BB_15_2.0_Width失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_15_2.0_Width")


class BB_20_2_0_Upper(BaseFactor):
    """20日2倍标准差布林带上轨"""
    factor_id = "BB_20_2.0_Upper"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            return result.upper_band.rename("BB_20_2.0_Upper")
        except Exception as e:
            logger.error(f"计算BB_20_2.0_Upper失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_20_2.0_Upper")


class BB_20_2_0_Middle(BaseFactor):
    """20日2倍标准差布林带中轨"""
    factor_id = "BB_20_2.0_Middle"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            return result.middle_band.rename("BB_20_2.0_Middle")
        except Exception as e:
            logger.error(f"计算BB_20_2.0_Middle失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_20_2.0_Middle")


class BB_20_2_0_Lower(BaseFactor):
    """20日2倍标准差布林带下轨"""
    factor_id = "BB_20_2.0_Lower"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            return result.lower_band.rename("BB_20_2.0_Lower")
        except Exception as e:
            logger.error(f"计算BB_20_2_0_Lower失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_20_2_0_Lower")


class BB_20_2_0_Width(BaseFactor):
    """20日2倍标准差布林带宽度"""
    factor_id = "BB_20_2.0_Width"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            bb_width = (result.upper_band - result.lower_band) / result.middle_band
            return bb_width.rename("BB_20_2.0_Width")
        except Exception as e:
            logger.error(f"计算BB_20_2.0_Width失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BB_20_2.0_Width")


class BOLB_20(BaseFactor):
    """20日布林带位置（收盘价相对于布林带的位置）"""
    factor_id = "BOLB_20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.BBANDS.run(data["close"], window=20, alpha=2.0)
            # 计算价格在布林带中的位置：0在下轨，1在上轨
            position = (data["close"] - result.lower_band) / (result.upper_band - result.lower_band)
            return position.rename("BOLB_20")
        except Exception as e:
            logger.error(f"计算BOLB_20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="BOLB_20")