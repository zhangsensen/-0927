"""
波动率类指标 - 基于VectorBT实现
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ...core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class ATR(BaseFactor):
    """平均真实波幅"""
    factor_id = "ATR"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.ATR.run(data["high"], data["low"], data["close"], window=14).atr.rename("ATR")
        except Exception as e:
            logger.error(f"计算ATR失败: {e}")
            return pd.Series(np.nan, index=data.index, name="ATR")


class ATR7(BaseFactor):
    """7日平均真实波幅"""
    factor_id = "ATR7"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.ATR.run(data["high"], data["low"], data["close"], window=7).atr.rename("ATR7")
        except Exception as e:
            logger.error(f"计算ATR7失败: {e}")
            return pd.Series(np.nan, index=data.index, name="ATR7")


class ATR10(BaseFactor):
    """10日平均真实波幅"""
    factor_id = "ATR10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.ATR.run(data["high"], data["low"], data["close"], window=10).atr.rename("ATR10")
        except Exception as e:
            logger.error(f"计算ATR10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="ATR10")


class ATR14(BaseFactor):
    """14日平均真实波幅"""
    factor_id = "ATR14"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.ATR.run(data["high"], data["low"], data["close"], window=14).atr.rename("ATR14")
        except Exception as e:
            logger.error(f"计算ATR14失败: {e}")
            return pd.Series(np.nan, index=data.index, name="ATR14")


class ATR20(BaseFactor):
    """20日平均真实波幅"""
    factor_id = "ATR20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.ATR.run(data["high"], data["low"], data["close"], window=20).atr.rename("ATR20")
        except Exception as e:
            logger.error(f"计算ATR20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="ATR20")


class MSTD5(BaseFactor):
    """5日移动标准差"""
    factor_id = "MSTD5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MSTD.run(data["close"], window=5).mstd.rename("MSTD5")
        except Exception as e:
            logger.error(f"计算MSTD5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MSTD5")


class MSTD10(BaseFactor):
    """10日移动标准差"""
    factor_id = "MSTD10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MSTD.run(data["close"], window=10).mstd.rename("MSTD10")
        except Exception as e:
            logger.error(f"计算MSTD10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MSTD10")


class MSTD15(BaseFactor):
    """15日移动标准差"""
    factor_id = "MSTD15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MSTD.run(data["close"], window=15).mstd.rename("MSTD15")
        except Exception as e:
            logger.error(f"计算MSTD15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MSTD15")


class MSTD20(BaseFactor):
    """20日移动标准差"""
    factor_id = "MSTD20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MSTD.run(data["close"], window=20).mstd.rename("MSTD20")
        except Exception as e:
            logger.error(f"计算MSTD20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MSTD20")


class FSTD5(BaseFactor):
    """5日固定窗口标准差"""
    factor_id = "FSTD5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].rolling(window=5).std().rename("FSTD5")
        except Exception as e:
            logger.error(f"计算FSTD5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="FSTD5")


class FSTD10(BaseFactor):
    """10日固定窗口标准差"""
    factor_id = "FSTD10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].rolling(window=10).std().rename("FSTD10")
        except Exception as e:
            logger.error(f"计算FSTD10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="FSTD10")


class FSTD15(BaseFactor):
    """15日固定窗口标准差"""
    factor_id = "FSTD15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].rolling(window=15).std().rename("FSTD15")
        except Exception as e:
            logger.error(f"计算FSTD15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="FSTD15")


class FSTD20(BaseFactor):
    """20日固定窗口标准差"""
    factor_id = "FSTD20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].rolling(window=20).std().rename("FSTD20")
        except Exception as e:
            logger.error(f"计算FSTD20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="FSTD20")