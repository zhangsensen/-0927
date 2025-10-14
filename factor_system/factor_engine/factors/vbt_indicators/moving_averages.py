"""
移动平均类指标 - 基于VectorBT实现
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ...core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class MA3(BaseFactor):
    """3日简单移动平均"""
    factor_id = "MA3"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=3).ma.rename("MA3")
        except Exception as e:
            logger.error(f"计算MA3失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA3")


class MA5(BaseFactor):
    """5日简单移动平均"""
    factor_id = "MA5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=5).ma.rename("MA5")
        except Exception as e:
            logger.error(f"计算MA5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA5")


class MA8(BaseFactor):
    """8日简单移动平均"""
    factor_id = "MA8"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=8).ma.rename("MA8")
        except Exception as e:
            logger.error(f"计算MA8失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA8")


class MA10(BaseFactor):
    """10日简单移动平均"""
    factor_id = "MA10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=10).ma.rename("MA10")
        except Exception as e:
            logger.error(f"计算MA10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA10")


class MA12(BaseFactor):
    """12日简单移动平均"""
    factor_id = "MA12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=12).ma.rename("MA12")
        except Exception as e:
            logger.error(f"计算MA12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA12")


class MA15(BaseFactor):
    """15日简单移动平均"""
    factor_id = "MA15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=15).ma.rename("MA15")
        except Exception as e:
            logger.error(f"计算MA15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA15")


class MA20(BaseFactor):
    """20日简单移动平均"""
    factor_id = "MA20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=20).ma.rename("MA20")
        except Exception as e:
            logger.error(f"计算MA20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA20")


class MA25(BaseFactor):
    """25日简单移动平均"""
    factor_id = "MA25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=25).ma.rename("MA25")
        except Exception as e:
            logger.error(f"计算MA25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA25")


class MA30(BaseFactor):
    """30日简单移动平均"""
    factor_id = "MA30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=30).ma.rename("MA30")
        except Exception as e:
            logger.error(f"计算MA30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA30")


class MA40(BaseFactor):
    """40日简单移动平均"""
    factor_id = "MA40"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=40).ma.rename("MA40")
        except Exception as e:
            logger.error(f"计算MA40失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA40")


class MA50(BaseFactor):
    """50日简单移动平均"""
    factor_id = "MA50"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=50).ma.rename("MA50")
        except Exception as e:
            logger.error(f"计算MA50失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA50")


class MA60(BaseFactor):
    """60日简单移动平均"""
    factor_id = "MA60"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=60).ma.rename("MA60")
        except Exception as e:
            logger.error(f"计算MA60失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA60")


class MA80(BaseFactor):
    """80日简单移动平均"""
    factor_id = "MA80"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=80).ma.rename("MA80")
        except Exception as e:
            logger.error(f"计算MA80失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA80")


class MA100(BaseFactor):
    """100日简单移动平均"""
    factor_id = "MA100"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=100).ma.rename("MA100")
        except Exception as e:
            logger.error(f"计算MA100失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA100")


class MA120(BaseFactor):
    """120日简单移动平均"""
    factor_id = "MA120"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=120).ma.rename("MA120")
        except Exception as e:
            logger.error(f"计算MA120失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA120")


class MA150(BaseFactor):
    """150日简单移动平均"""
    factor_id = "MA150"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=150).ma.rename("MA150")
        except Exception as e:
            logger.error(f"计算MA150失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA150")


class MA200(BaseFactor):
    """200日简单移动平均"""
    factor_id = "MA200"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MA.run(data["close"], window=200).ma.rename("MA200")
        except Exception as e:
            logger.error(f"计算MA200失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MA200")


class EMA3(BaseFactor):
    """3日指数移动平均"""
    factor_id = "EMA3"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=3, adjust=False).mean().rename("EMA3")
        except Exception as e:
            logger.error(f"计算EMA3失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA3")


class EMA5(BaseFactor):
    """5日指数移动平均"""
    factor_id = "EMA5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=5, adjust=False).mean().rename("EMA5")
        except Exception as e:
            logger.error(f"计算EMA5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA5")


class EMA8(BaseFactor):
    """8日指数移动平均"""
    factor_id = "EMA8"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=8, adjust=False).mean().rename("EMA8")
        except Exception as e:
            logger.error(f"计算EMA8失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA8")


class EMA12(BaseFactor):
    """12日指数移动平均"""
    factor_id = "EMA12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=12, adjust=False).mean().rename("EMA12")
        except Exception as e:
            logger.error(f"计算EMA12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA12")


class EMA15(BaseFactor):
    """15日指数移动平均"""
    factor_id = "EMA15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=15, adjust=False).mean().rename("EMA15")
        except Exception as e:
            logger.error(f"计算EMA15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA15")


class EMA20(BaseFactor):
    """20日指数移动平均"""
    factor_id = "EMA20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=20, adjust=False).mean().rename("EMA20")
        except Exception as e:
            logger.error(f"计算EMA20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA20")


class EMA26(BaseFactor):
    """26日指数移动平均"""
    factor_id = "EMA26"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=26, adjust=False).mean().rename("EMA26")
        except Exception as e:
            logger.error(f"计算EMA26失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA26")


class EMA30(BaseFactor):
    """30日指数移动平均"""
    factor_id = "EMA30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=30, adjust=False).mean().rename("EMA30")
        except Exception as e:
            logger.error(f"计算EMA30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA30")


class EMA40(BaseFactor):
    """40日指数移动平均"""
    factor_id = "EMA40"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=40, adjust=False).mean().rename("EMA40")
        except Exception as e:
            logger.error(f"计算EMA40失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA40")


class EMA50(BaseFactor):
    """50日指数移动平均"""
    factor_id = "EMA50"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=50, adjust=False).mean().rename("EMA50")
        except Exception as e:
            logger.error(f"计算EMA50失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA50")


class EMA60(BaseFactor):
    """60日指数移动平均"""
    factor_id = "EMA60"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].ewm(span=60, adjust=False).mean().rename("EMA60")
        except Exception as e:
            logger.error(f"计算EMA60失败: {e}")
            return pd.Series(np.nan, index=data.index, name="EMA60")


# 高级移动平均
class DEMA(BaseFactor):
    """双指数移动平均"""
    factor_id = "DEMA"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # 使用VectorBT的EMA实现DEMA
            ema1 = data["close"].ewm(span=20, adjust=False).mean()
            ema2 = ema1.ewm(span=20, adjust=False).mean()
            return (2 * ema1 - ema2).rename("DEMA")
        except Exception as e:
            logger.error(f"计算DEMA失败: {e}")
            return pd.Series(np.nan, index=data.index, name="DEMA")


class TEMA(BaseFactor):
    """三指数移动平均"""
    factor_id = "TEMA"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # 使用VectorBT的EMA实现TEMA
            ema1 = data["close"].ewm(span=20, adjust=False).mean()
            ema2 = ema1.ewm(span=20, adjust=False).mean()
            ema3 = ema2.ewm(span=20, adjust=False).mean()
            return (3 * ema1 - 3 * ema2 + ema3).rename("TEMA")
        except Exception as e:
            logger.error(f"计算TEMA失败: {e}")
            return pd.Series(np.nan, index=data.index, name="TEMA")


class KAMA(BaseFactor):
    """考夫曼自适应移动平均"""
    factor_id = "KAMA"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # 简化版KAMA实现
            close = data["close"]
            change = close.diff().abs()
            volatility = change.rolling(window=10).sum()
            er = change.rolling(window=10).sum() / volatility
            sc = (er * (2.0/3.0 - 2.0/31.0) + 2.0/31.0) ** 2
            return close.ewm(alpha=sc, adjust=False).mean().rename("KAMA")
        except Exception as e:
            logger.error(f"计算KAMA失败: {e}")
            return pd.Series(np.nan, index=data.index, name="KAMA")


class WMA10(BaseFactor):
    """10日加权移动平均"""
    factor_id = "WMA10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # 使用VectorBT的WMA
            return vbt.WMA.run(data["close"], window=10).wma.rename("WMA10")
        except Exception as e:
            logger.error(f"计算WMA10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="WMA10")


class WMA20(BaseFactor):
    """20日加权移动平均"""
    factor_id = "WMA20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.WMA.run(data["close"], window=20).wma.rename("WMA20")
        except Exception as e:
            logger.error(f"计算WMA20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="WMA20")


class WMA5(BaseFactor):
    """5日加权移动平均"""
    factor_id = "WMA5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.WMA.run(data["close"], window=5).wma.rename("WMA5")
        except Exception as e:
            logger.error(f"计算WMA5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="WMA5")


class TRIMA(BaseFactor):
    """三角移动平均"""
    factor_id = "TRIMA"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # TRIMA是SMA的SMA
            sma1 = data["close"].rolling(window=15).mean()
            return sma1.rolling(window=15).mean().rename("TRIMA")
        except Exception as e:
            logger.error(f"计算TRIMA失败: {e}")
            return pd.Series(np.nan, index=data.index, name="TRIMA")