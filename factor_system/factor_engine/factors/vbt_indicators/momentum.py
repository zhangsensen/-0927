"""
动量类指标 - 基于VectorBT实现
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ...core.base_factor import BaseFactor

logger = logging.getLogger(__name__)


class RSI(BaseFactor):
    """相对强弱指数"""
    factor_id = "RSI"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=14).rsi.rename("RSI")
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI")


class RSI3(BaseFactor):
    """3日相对强弱指数"""
    factor_id = "RSI3"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=3).rsi.rename("RSI3")
        except Exception as e:
            logger.error(f"计算RSI3失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI3")


class RSI6(BaseFactor):
    """6日相对强弱指数"""
    factor_id = "RSI6"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=6).rsi.rename("RSI6")
        except Exception as e:
            logger.error(f"计算RSI6失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI6")


class RSI9(BaseFactor):
    """9日相对强弱指数"""
    factor_id = "RSI9"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=9).rsi.rename("RSI9")
        except Exception as e:
            logger.error(f"计算RSI9失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI9")


class RSI12(BaseFactor):
    """12日相对强弱指数"""
    factor_id = "RSI12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=12).rsi.rename("RSI12")
        except Exception as e:
            logger.error(f"计算RSI12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI12")


class RSI14(BaseFactor):
    """14日相对强弱指数"""
    factor_id = "RSI14"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=14).rsi.rename("RSI14")
        except Exception as e:
            logger.error(f"计算RSI14失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI14")


class RSI18(BaseFactor):
    """18日相对强弱指数"""
    factor_id = "RSI18"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=18).rsi.rename("RSI18")
        except Exception as e:
            logger.error(f"计算RSI18失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI18")


class RSI21(BaseFactor):
    """21日相对强弱指数"""
    factor_id = "RSI21"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=21).rsi.rename("RSI21")
        except Exception as e:
            logger.error(f"计算RSI21失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI21")


class RSI25(BaseFactor):
    """25日相对强弱指数"""
    factor_id = "RSI25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=25).rsi.rename("RSI25")
        except Exception as e:
            logger.error(f"计算RSI25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI25")


class RSI30(BaseFactor):
    """30日相对强弱指数"""
    factor_id = "RSI30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.RSI.run(data["close"], window=30).rsi.rename("RSI30")
        except Exception as e:
            logger.error(f"计算RSI30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="RSI30")


class MACD(BaseFactor):
    """MACD指标"""
    factor_id = "MACD"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MACD.run(data["close"], fast=12, slow=26, signal=9).macd.rename("MACD")
        except Exception as e:
            logger.error(f"计算MACD失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MACD")


class MACD_SIGNAL(BaseFactor):
    """MACD信号线"""
    factor_id = "MACD_SIGNAL"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MACD.run(data["close"], fast=12, slow=26, signal=9).signal.rename("MACD_SIGNAL")
        except Exception as e:
            logger.error(f"计算MACD_SIGNAL失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MACD_SIGNAL")


class MACD_HIST(BaseFactor):
    """MACD柱状图"""
    factor_id = "MACD_HIST"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.MACD.run(data["close"], fast=12, slow=26, signal=9).histogram.rename("MACD_HIST")
        except Exception as e:
            logger.error(f"计算MACD_HIST失败: {e}")
            return pd.Series(np.nan, index=data.index, name="MACD_HIST")


class STOCH(BaseFactor):
    """随机指标"""
    factor_id = "STOCH"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = vbt.STOCH.run(data["high"], data["low"], data["close"],
                                   k_window=14, d_window=3, d_smooth_window=3)
            return result.stoch_k.rename("STOCH")
        except Exception as e:
            logger.error(f"计算STOCH失败: {e}")
            return pd.Series(np.nan, index=data.index, name="STOCH")


class WILLR(BaseFactor):
    """威廉指标"""
    factor_id = "WILLR"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.WILLR.run(data["high"], data["low"], data["close"], window=14).willr.rename("WILLR")
        except Exception as e:
            logger.error(f"计算WILLR失败: {e}")
            return pd.Series(np.nan, index=data.index, name="WILLR")


class WILLR14(BaseFactor):
    """14日威廉指标"""
    factor_id = "WILLR14"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return vbt.WILLR.run(data["high"], data["low"], data["close"], window=14).willr.rename("WILLR14")
        except Exception as e:
            logger.error(f"计算WILLR14失败: {e}")
            return pd.Series(np.nan, index=data.index, name="WILLR14")


class CCI(BaseFactor):
    """商品通道指标"""
    factor_id = "CCI"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            # CCI需要自定义计算
            tp = (data["high"] + data["low"] + data["close"]) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            return (tp - sma_tp) / (0.015 * mad).rename("CCI")
        except Exception as e:
            logger.error(f"计算CCI失败: {e}")
            return pd.Series(np.nan, index=data.index, name="CCI")


class CCI14(BaseFactor):
    """14日商品通道指标"""
    factor_id = "CCI14"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            tp = (data["high"] + data["low"] + data["close"]) / 3
            sma_tp = tp.rolling(window=14).mean()
            mad = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
            return (tp - sma_tp) / (0.015 * mad).rename("CCI14")
        except Exception as e:
            logger.error(f"计算CCI14失败: {e}")
            return pd.Series(np.nan, index=data.index, name="CCI14")


class Momentum1(BaseFactor):
    """1期动量"""
    factor_id = "Momentum1"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(1).rename("Momentum1")
        except Exception as e:
            logger.error(f"计算Momentum1失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum1")


class Momentum3(BaseFactor):
    """3期动量"""
    factor_id = "Momentum3"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(3).rename("Momentum3")
        except Exception as e:
            logger.error(f"计算Momentum3失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum3")


class Momentum5(BaseFactor):
    """5期动量"""
    factor_id = "Momentum5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(5).rename("Momentum5")
        except Exception as e:
            logger.error(f"计算Momentum5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum5")


class Momentum8(BaseFactor):
    """8期动量"""
    factor_id = "Momentum8"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(8).rename("Momentum8")
        except Exception as e:
            logger.error(f"计算Momentum8失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum8")


class Momentum10(BaseFactor):
    """10期动量"""
    factor_id = "Momentum10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(10).rename("Momentum10")
        except Exception as e:
            logger.error(f"计算Momentum10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum10")


class Momentum12(BaseFactor):
    """12期动量"""
    factor_id = "Momentum12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(12).rename("Momentum12")
        except Exception as e:
            logger.error(f"计算Momentum12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum12")


class Momentum15(BaseFactor):
    """15期动量"""
    factor_id = "Momentum15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(15).rename("Momentum15")
        except Exception as e:
            logger.error(f"计算Momentum15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum15")


class Momentum20(BaseFactor):
    """20期动量"""
    factor_id = "Momentum20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return data["close"].diff(20).rename("Momentum20")
        except Exception as e:
            logger.error(f"计算Momentum20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Momentum20")


class Position5(BaseFactor):
    """5期位置"""
    factor_id = "Position5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(5).min()) / (data["close"].rolling(5).max() - data["close"].rolling(5).min()).rename("Position5")
        except Exception as e:
            logger.error(f"计算Position5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position5")


class Position8(BaseFactor):
    """8期位置"""
    factor_id = "Position8"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(8).min()) / (data["close"].rolling(8).max() - data["close"].rolling(8).min()).rename("Position8")
        except Exception as e:
            logger.error(f"计算Position8失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position8")


class Position10(BaseFactor):
    """10期位置"""
    factor_id = "Position10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(10).min()) / (data["close"].rolling(10).max() - data["close"].rolling(10).min()).rename("Position10")
        except Exception as e:
            logger.error(f"计算Position10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position10")


class Position12(BaseFactor):
    """12期位置"""
    factor_id = "Position12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(12).min()) / (data["close"].rolling(12).max() - data["close"].rolling(12).min()).rename("Position12")
        except Exception as e:
            logger.error(f"计算Position12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position12")


class Position15(BaseFactor):
    """15期位置"""
    factor_id = "Position15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(15).min()) / (data["close"].rolling(15).max() - data["close"].rolling(15).min()).rename("Position15")
        except Exception as e:
            logger.error(f"计算Position15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position15")


class Position20(BaseFactor):
    """20期位置"""
    factor_id = "Position20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(20).min()) / (data["close"].rolling(20).max() - data["close"].rolling(20).min()).rename("Position20")
        except Exception as e:
            logger.error(f"计算Position20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position20")


class Position25(BaseFactor):
    """25期位置"""
    factor_id = "Position25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(25).min()) / (data["close"].rolling(25).max() - data["close"].rolling(25).min()).rename("Position25")
        except Exception as e:
            logger.error(f"计算Position25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position25")


class Position30(BaseFactor):
    """30期位置"""
    factor_id = "Position30"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] - data["close"].rolling(30).min()) / (data["close"].rolling(30).max() - data["close"].rolling(30).min()).rename("Position30")
        except Exception as e:
            logger.error(f"计算Position30失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Position30")


class Trend5(BaseFactor):
    """5期趋势"""
    factor_id = "Trend5"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(5)).astype(int).rename("Trend5")
        except Exception as e:
            logger.error(f"计算Trend5失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend5")


class Trend8(BaseFactor):
    """8期趋势"""
    factor_id = "Trend8"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(8)).astype(int).rename("Trend8")
        except Exception as e:
            logger.error(f"计算Trend8失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend8")


class Trend10(BaseFactor):
    """10期趋势"""
    factor_id = "Trend10"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(10)).astype(int).rename("Trend10")
        except Exception as e:
            logger.error(f"计算Trend10失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend10")


class Trend12(BaseFactor):
    """12期趋势"""
    factor_id = "Trend12"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(12)).astype(int).rename("Trend12")
        except Exception as e:
            logger.error(f"计算Trend12失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend12")


class Trend15(BaseFactor):
    """15期趋势"""
    factor_id = "Trend15"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(15)).astype(int).rename("Trend15")
        except Exception as e:
            logger.error(f"计算Trend15失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend15")


class Trend20(BaseFactor):
    """20期趋势"""
    factor_id = "Trend20"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(20)).astype(int).rename("Trend20")
        except Exception as e:
            logger.error(f"计算Trend20失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend20")


class Trend25(BaseFactor):
    """25期趋势"""
    factor_id = "Trend25"
    category = "vbt_technical"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        try:
            return (data["close"] > data["close"].shift(25)).astype(int).rename("Trend25")
        except Exception as e:
            logger.error(f"计算Trend25失败: {e}")
            return pd.Series(np.nan, index=data.index, name="Trend25")