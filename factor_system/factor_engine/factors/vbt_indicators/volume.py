"""
成交量类指标 - 基于VectorBT实现
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ...core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class OBV(BaseFactor):
    """能量潮指标"""
    factor_id = "OBV"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.OBV.run(data["close"], data["volume"]).obv.rename("OBV")
        except Exception as e:
            logger.error(f"计算OBV失败: {e}")
            return pd.Series(np.nan, index=data.index, name="OBV")


class OBV_SMA5(BaseFactor):
    """OBV 5日简单移动平均"""
    factor_id = "OBV_SMA_5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            obv = vbt.OBV.run(data["close"], data["volume"]).obv
            return obv.rolling(window=5).mean().rename("OBV_SMA_5")
        except Exception as e:
            logger.error(f"计算OBV_SMA_5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="OBV_SMA_5")


class OBV_SMA10(BaseFactor):
    """OBV 10日简单移动平均"""
    factor_id = "OBV_SMA_10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            obv = vbt.OBV.run(data["close"], data["volume"]).obv
            return obv.rolling(window=10).mean().rename("OBV_SMA_10")
        except Exception as e:
            logger.error(f"计算OBV_SMA_10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="OBV_SMA_10")


class OBV_SMA15(BaseFactor):
    """OBV 15日简单移动平均"""
    factor_id = "OBV_SMA_15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            obv = vbt.OBV.run(data["close"], data["volume"]).obv
            return obv.rolling(window=15).mean().rename("OBV_SMA_15")
        except Exception as e:
            logger.error(f"计算OBV_SMA_15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="OBV_SMA_15")


class OBV_SMA20(BaseFactor):
    """OBV 20日简单移动平均"""
    factor_id = "OBV_SMA_20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            obv = vbt.OBV.run(data["close"], data["volume"]).obv
            return obv.rolling(window=20).mean().rename("OBV_SMA_20")
        except Exception as e:
            logger.error(f"计算OBV_SMA_20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="OBV_SMA_20")


class VWAP5(BaseFactor):
    """5日成交量加权平均价"""
    factor_id = "VWAP5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # VWAP = (Price * Volume)的累计和 / Volume的累计和
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=5).sum() / data["volume"].rolling(window=5).sum()
            return vwap.rename("VWAP5")
        except Exception as e:
            logger.error(f"计算VWAP5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP5")


class VWAP10(BaseFactor):
    """10日成交量加权平均价"""
    factor_id = "VWAP10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=10).sum() / data["volume"].rolling(window=10).sum()
            return vwap.rename("VWAP10")
        except Exception as e:
            logger.error(f"计算VWAP10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP10")


class VWAP15(BaseFactor):
    """15日成交量加权平均价"""
    factor_id = "VWAP15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=15).sum() / data["volume"].rolling(window=15).sum()
            return vwap.rename("VWAP15")
        except Exception as e:
            logger.error(f"计算VWAP15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP15")


class VWAP20(BaseFactor):
    """20日成交量加权平均价"""
    factor_id = "VWAP20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=20).sum() / data["volume"].rolling(window=20).sum()
            return vwap.rename("VWAP20")
        except Exception as e:
            logger.error(f"计算VWAP20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP20")


class VWAP25(BaseFactor):
    """25日成交量加权平均价"""
    factor_id = "VWAP25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=25).sum() / data["volume"].rolling(window=25).sum()
            return vwap.rename("VWAP25")
        except Exception as e:
            logger.error(f"计算VWAP25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP25")


class VWAP30(BaseFactor):
    """30日成交量加权平均价"""
    factor_id = "VWAP30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).rolling(window=30).sum() / data["volume"].rolling(window=30).sum()
            return vwap.rename("VWAP30")
        except Exception as e:
            logger.error(f"计算VWAP30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="VWAP30")


class Volume_Ratio10(BaseFactor):
    """10日成交量比率"""
    factor_id = "Volume_Ratio10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            volume_ma = data["volume"].rolling(window=10).mean()
            return (data["volume"] / volume_ma).rename("Volume_Ratio10")
        except Exception as e:
            logger.error(f"计算Volume_Ratio10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Ratio10")


class Volume_Ratio15(BaseFactor):
    """15日成交量比率"""
    factor_id = "Volume_Ratio15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            volume_ma = data["volume"].rolling(window=15).mean()
            return (data["volume"] / volume_ma).rename("Volume_Ratio15")
        except Exception as e:
            logger.error(f"计算Volume_Ratio15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Ratio15")


class Volume_Ratio20(BaseFactor):
    """20日成交量比率"""
    factor_id = "Volume_Ratio20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            volume_ma = data["volume"].rolling(window=20).mean()
            return (data["volume"] / volume_ma).rename("Volume_Ratio20")
        except Exception as e:
            logger.error(f"计算Volume_Ratio20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Ratio20")


class Volume_Ratio25(BaseFactor):
    """25日成交量比率"""
    factor_id = "Volume_Ratio25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            volume_ma = data["volume"].rolling(window=25).mean()
            return (data["volume"] / volume_ma).rename("Volume_Ratio25")
        except Exception as e:
            logger.error(f"计算Volume_Ratio25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Ratio25")


class Volume_Ratio30(BaseFactor):
    """30日成交量比率"""
    factor_id = "Volume_Ratio30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            volume_ma = data["volume"].rolling(window=30).mean()
            return (data["volume"] / volume_ma).rename("Volume_Ratio30")
        except Exception as e:
            logger.error(f"计算Volume_Ratio30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Ratio30")


class Volume_Momentum10(BaseFactor):
    """10日成交量动量"""
    factor_id = "Volume_Momentum10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff(10).rename("Volume_Momentum10")
        except Exception as e:
            logger.error(f"计算Volume_Momentum10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Momentum10")


class Volume_Momentum15(BaseFactor):
    """15日成交量动量"""
    factor_id = "Volume_Momentum15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff(15).rename("Volume_Momentum15")
        except Exception as e:
            logger.error(f"计算Volume_Momentum15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Momentum15")


class Volume_Momentum20(BaseFactor):
    """20日成交量动量"""
    factor_id = "Volume_Momentum20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff(20).rename("Volume_Momentum20")
        except Exception as e:
            logger.error(f"计算Volume_Momentum20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Momentum20")


class Volume_Momentum25(BaseFactor):
    """25日成交量动量"""
    factor_id = "Volume_Momentum25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff(25).rename("Volume_Momentum25")
        except Exception as e:
            logger.error(f"计算Volume_Momentum25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Momentum25")


class Volume_Momentum30(BaseFactor):
    """30日成交量动量"""
    factor_id = "Volume_Momentum30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["volume"].diff(30).rename("Volume_Momentum30")
        except Exception as e:
            logger.error(f"计算Volume_Momentum30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Volume_Momentum30")