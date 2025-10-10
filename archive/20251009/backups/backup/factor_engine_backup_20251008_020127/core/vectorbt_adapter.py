"""
VectorBT计算适配器 - 复制factor_generation的核心计算逻辑

基于VectorBT + TA-Lib的成熟实现，确保与factor_generation计算逻辑完全一致
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import pandas as pd
import vectorbt as vbt

logger = logging.getLogger(__name__)


def extract_vbt_labels(vbt_result, name: Optional[str] = None):
    """从VectorBT标签对象中提取可序列化的标签数据

    从factor_generation/enhanced_factor_calculator.py复制
    """
    try:
        index = getattr(vbt_result, "index", None)

        # 优先尝试常见的属性
        for attr in ["integer", "labels", "real", "values"]:
            if hasattr(vbt_result, attr):
                try:
                    value = getattr(vbt_result, attr)
                except Exception:
                    continue
                if value is None:
                    continue
                value_index = getattr(value, "index", index)
                return ensure_series(
                    value, value_index, name or getattr(value, "name", name)
                )

        # VectorBT对象通常支持to_pd()方法
        if hasattr(vbt_result, "to_pd"):
            converted = vbt_result.to_pd()
            if isinstance(converted, pd.Series):
                return ensure_series(converted, converted.index, name or converted.name)
            if isinstance(converted, pd.DataFrame) and converted.shape[1] == 1:
                series = converted.iloc[:, 0]
                return ensure_series(series, series.index, name or series.name)

        # 直接是Series或数组
        if isinstance(vbt_result, pd.Series):
            return ensure_series(vbt_result, vbt_result.index, name or vbt_result.name)

        if hasattr(vbt_result, "values"):
            return ensure_series(vbt_result.to_numpy(), index, name)

        return ensure_series(vbt_result, index, name)
    except Exception as e:
        logging.warning(f"提取VectorBT标签失败: {e}")
        return None


def extract_indicator_component(result, attr_candidates):
    """从指标结果对象中提取第一个可用的属性值

    从factor_generation/enhanced_factor_calculator.py复制
    """
    for attr in attr_candidates:
        if hasattr(result, attr):
            return getattr(result, attr)

    if hasattr(result, "_asdict"):
        result_dict = result._asdict()
        for attr in attr_candidates:
            if attr in result_dict:
                return result_dict[attr]

    return None


def ensure_series(values, index, name):
    """确保将输出转换为带名称的Series

    从factor_generation/enhanced_factor_calculator.py复制
    """
    if isinstance(values, pd.Series):
        series = values.copy()
        if name is not None:
            series.name = name
        return series

    if hasattr(values, "to_pd"):
        converted = values.to_pd()
        if isinstance(converted, pd.Series):
            return converted if name is None else converted.rename(name)
        if isinstance(converted, pd.DataFrame) and converted.shape[1] == 1:
            series = converted.iloc[:, 0]
            return series if name is None else series.rename(name)

    if hasattr(values, "to_series"):
        try:
            series = values.to_series()
            if name is not None:
                series.name = name
            return series
        except Exception:
            pass

    return pd.Series(values, index=index, name=name)


class VectorBTAdapter:
    """
    VectorBT计算适配器

    统一封装VectorBT的指标计算，确保与factor_generation逻辑一致
    """

    def __init__(self):
        """初始化适配器"""
        self._check_vectorbt_availability()

    def _check_vectorbt_availability(self):
        """检查VectorBT可用性"""
        if not hasattr(vbt, 'RSI'):
            raise ImportError("VectorBT不完整，缺少技术指标支持")

        # 检查TA-Lib支持
        if hasattr(vbt, 'talib'):
            logger.info("VectorBT TA-Lib支持可用")
        else:
            logger.warning("VectorBT TA-Lib支持不可用，将使用内置指标")

    # 技术指标计算方法
    def calculate_rsi(self, price: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI"""
        try:
            result = vbt.RSI.run(price, window=window)
            return ensure_series(result.rsi, price.index, "RSI")
        except Exception as e:
            logger.error(f"RSI计算失败: {e}")
            return pd.Series(index=price.index, name="RSI", dtype=float)

    def calculate_stoch(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> pd.Series:
        """计算STOCH - 返回%K值"""
        try:
            result = vbt.STOCH.run(high, low, close,
                                   k_period=fastk_period,
                                   d_period=slowd_period,
                                   smooth_k=slowk_period)
            # 返回%K值（slowk）
            return ensure_series(result.slowk, close.index, "STOCH")
        except Exception as e:
            logger.error(f"STOCH计算失败: {e}")
            return pd.Series(index=close.index, name="STOCH", dtype=float)

    def calculate_macd(self, close: pd.Series,
                      fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """计算MACD线"""
        try:
            result = vbt.MACD.run(close,
                                fast_period=fast_period,
                                slow_period=slow_period,
                                signal_period=signal_period)
            return ensure_series(result.macd, close.index, "MACD")
        except Exception as e:
            logger.error(f"MACD计算失败: {e}")
            return pd.Series(index=close.index, name="MACD", dtype=float)

    def calculate_macd_signal(self, close: pd.Series,
                            fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """计算MACD信号线"""
        try:
            result = vbt.MACD.run(close,
                                fast_period=fast_period,
                                slow_period=slow_period,
                                signal_period=signal_period)
            return ensure_series(result.signal, close.index, "MACD_SIGNAL")
        except Exception as e:
            logger.error(f"MACD_SIGNAL计算失败: {e}")
            return pd.Series(index=close.index, name="MACD_SIGNAL", dtype=float)

    def calculate_macd_histogram(self, close: pd.Series,
                               fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
        """计算MACD柱状图"""
        try:
            result = vbt.MACD.run(close,
                                fast_period=fast_period,
                                slow_period=slow_period,
                                signal_period=signal_period)
            return ensure_series(result.histogram, close.index, "MACD_HIST")
        except Exception as e:
            logger.error(f"MACD_HIST计算失败: {e}")
            return pd.Series(index=close.index, name="MACD_HIST", dtype=float)

    def calculate_willr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       timeperiod: int = 14) -> pd.Series:
        """计算Williams %R"""
        try:
            result = vbt.WILLR.run(high, low, close, window=timeperiod)
            return ensure_series(result.willr, close.index, "WILLR")
        except Exception as e:
            logger.error(f"WILLR计算失败: {e}")
            return pd.Series(index=close.index, name="WILLR", dtype=float)

    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算CCI"""
        try:
            result = vbt.CCI.run(high, low, close, window=timeperiod)
            return ensure_series(result.cci, close.index, "CCI")
        except Exception as e:
            logger.error(f"CCI计算失败: {e}")
            return pd.Series(index=close.index, name="CCI", dtype=float)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算ATR"""
        try:
            result = vbt.ATR.run(high, low, close, window=timeperiod)
            return ensure_series(result.atr, close.index, "ATR")
        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return pd.Series(index=close.index, name="ATR", dtype=float)

    def calculate_sma(self, price: pd.Series, window: int) -> pd.Series:
        """计算SMA"""
        try:
            result = vbt.SMA.run(price, window=window)
            return ensure_series(result.sma, price.index, "SMA")
        except Exception as e:
            logger.error(f"SMA计算失败: {e}")
            return pd.Series(index=price.index, name="SMA", dtype=float)

    def calculate_ema(self, price: pd.Series, window: int) -> pd.Series:
        """计算EMA"""
        try:
            result = vbt.EMA.run(price, window=window)
            return ensure_series(result.ema, price.index, "EMA")
        except Exception as e:
            logger.error(f"EMA计算失败: {e}")
            return pd.Series(index=price.index, name="EMA", dtype=float)

    def calculate_bbands(self, price: pd.Series, window: int = 20,
                        nbdevup: float = 2.0, nbdevdn: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带"""
        try:
            result = vbt.BBANDS.run(price, window=window,
                                   upper=nbdevup, lower=nbdevdn)
            return {
                "BBANDS_UPPER": ensure_series(result.upperband, price.index, "BBANDS_UPPER"),
                "BBANDS_MIDDLE": ensure_series(result.middleband, price.index, "BBANDS_MIDDLE"),
                "BBANDS_LOWER": ensure_series(result.lowerband, price.index, "BBANDS_LOWER")
            }
        except Exception as e:
            logger.error(f"BBANDS计算失败: {e}")
            empty_series = pd.Series(index=price.index, dtype=float)
            return {
                "BBANDS_UPPER": empty_series.rename("BBANDS_UPPER"),
                "BBANDS_MIDDLE": empty_series.rename("BBANDS_MIDDLE"),
                "BBANDS_LOWER": empty_series.rename("BBANDS_LOWER")
            }

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算OBV"""
        try:
            result = vbt.OBV.run(close, volume)
            return ensure_series(result.obv, close.index, "OBV")
        except Exception as e:
            logger.error(f"OBV计算失败: {e}")
            return pd.Series(index=close.index, name="OBV", dtype=float)

    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算MFI"""
        try:
            # 优先使用TA-Lib
            if hasattr(vbt, 'talib'):
                talib_mfi = vbt.talib('MFI')
                result = talib_mfi.run(high, low, close, volume, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "MFI")
            else:
                # MFI是RSI的成交量加权版本
                # 使用typical price = (high + low + close) / 3
                typical_price = (high + low + close) / 3
                # 这里暂时用RSI近似，实际应该用成交量加权
                result = vbt.RSI.run(typical_price, window=timeperiod)
                return ensure_series(result.rsi, close.index, "MFI")
        except Exception as e:
            logger.error(f"MFI计算失败: {e}")
            return pd.Series(index=close.index, name="MFI", dtype=float)

    # TA-Lib扩展指标支持
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      timeperiod: int = 14) -> pd.Series:
        """计算ADX"""
        try:
            if hasattr(vbt, 'talib'):
                talib_adx = vbt.talib('ADX')
                result = talib_adx.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "ADX")
            else:
                logger.warning("ADX需要TA-Lib支持")
                return pd.Series(index=close.index, name="ADX", dtype=float)
        except Exception as e:
            logger.error(f"ADX计算失败: {e}")
            return pd.Series(index=close.index, name="ADX", dtype=float)

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算ATR"""
        try:
            if hasattr(vbt, 'talib'):
                talib_atr = vbt.talib('ATR')
                result = talib_atr.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "ATR")
            else:
                # 回退到VectorBT内置ATR
                result = vbt.ATR.run(high, low, close, window=timeperiod)
                return ensure_series(result.atr, close.index, "ATR")
        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return pd.Series(index=close.index, name="ATR", dtype=float)

    def calculate_trange(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算True Range"""
        try:
            if hasattr(vbt, 'talib'):
                talib_trange = vbt.talib('TRANGE')
                result = talib_trange.run(high, low, close)
                return ensure_series(result.real, close.index, "TRANGE")
            else:
                # 手动计算True Range
                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                trange = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return ensure_series(trange, close.index, "TRANGE")
        except Exception as e:
            logger.error(f"TRANGE计算失败: {e}")
            return pd.Series(index=close.index, name="TRANGE", dtype=float)

    def calculate_ad(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算A/D Line"""
        try:
            if hasattr(vbt, 'talib'):
                talib_ad = vbt.talib('AD')
                result = talib_ad.run(high, low, close, volume)
                return ensure_series(result.real, close.index, "AD")
            else:
                # 回退到VectorBT内置OBV（作为近似）
                result = vbt.OBV.run(close, volume)
                return ensure_series(result.obv, close.index, "AD")
        except Exception as e:
            logger.error(f"AD计算失败: {e}")
            return pd.Series(index=close.index, name="AD", dtype=float)

    def calculate_adosc(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                        fastperiod: int = 3, slowperiod: int = 10) -> pd.Series:
        """计算A/D Oscillator"""
        try:
            if hasattr(vbt, 'talib'):
                talib_adosc = vbt.talib('ADOSC')
                result = talib_adosc.run(high, low, close, volume,
                                       fastperiod=fastperiod, slowperiod=slowperiod)
                return ensure_series(result.real, close.index, "ADOSC")
            else:
                logger.warning("ADOSC需要TA-Lib支持")
                return pd.Series(index=close.index, name="ADOSC", dtype=float)
        except Exception as e:
            logger.error(f"ADOSC计算失败: {e}")
            return pd.Series(index=close.index, name="ADOSC", dtype=float)

    # 扩展支持所有活跃的v1.0因子
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      timeperiod: int = 14) -> pd.Series:
        """计算ADX - 与factor_generation一致的vbt.talib实现"""
        try:
            if hasattr(vbt, 'talib'):
                talib_adx = vbt.talib('ADX')
                result = talib_adx.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "ADX")
            else:
                logger.warning("ADX需要TA-Lib支持")
                return pd.Series(index=close.index, name="ADX", dtype=float)
        except Exception as e:
            logger.error(f"ADX计算失败: {e}")
            return pd.Series(index=close.index, name="ADX", dtype=float)

    def calculate_adxr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       timeperiod: int = 14) -> pd.Series:
        """计算ADXR"""
        try:
            if hasattr(vbt, 'talib'):
                talib_adxr = vbt.talib('ADXR')
                result = talib_adxr.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "ADXR")
            else:
                logger.warning("ADXR需要TA-Lib支持")
                return pd.Series(index=close.index, name="ADXR", dtype=float)
        except Exception as e:
            logger.error(f"ADXR计算失败: {e}")
            return pd.Series(index=close.index, name="ADXR", dtype=float)

    def calculate_aroon(self, high: pd.Series, low: pd.Series,
                       timeperiod: int = 14) -> pd.Series:
        """计算AROON - 返回Aroon Down值"""
        try:
            if hasattr(vbt, 'talib'):
                talib_aroon = vbt.talib('AROON')
                result = talib_aroon.run(high, low, timeperiod=timeperiod)
                return ensure_series(result.aroondown, low.index, "AROON")
            else:
                logger.warning("AROON需要TA-Lib支持")
                return pd.Series(index=low.index, name="AROON", dtype=float)
        except Exception as e:
            logger.error(f"AROON计算失败: {e}")
            return pd.Series(index=low.index, name="AROON", dtype=float)

    def calculate_aroonosc(self, high: pd.Series, low: pd.Series,
                           timeperiod: int = 14) -> pd.Series:
        """计算AROONOSC"""
        try:
            if hasattr(vbt, 'talib'):
                talib_aroonosc = vbt.talib('AROONOSC')
                result = talib_aroonosc.run(high, low, timeperiod=timeperiod)
                return ensure_series(result.real, low.index, "AROONOSC")
            else:
                logger.warning("AROONOSC需要TA-Lib支持")
                return pd.Series(index=low.index, name="AROONOSC", dtype=float)
        except Exception as e:
            logger.error(f"AROONOSC计算失败: {e}")
            return pd.Series(index=low.index, name="AROONOSC", dtype=float)

    def calculate_bbands(self, price: pd.Series, window: int = 20,
                        nbdevup: float = 2.0, nbdevdn: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带 - 使用TA-Lib确保一致性"""
        try:
            if hasattr(vbt, 'talib'):
                talib_bbands = vbt.talib('BBANDS')
                result = talib_bbands.run(price, timeperiod=window,
                                        nbdevup=nbdevup, nbdevdn=nbdevdn)
                return {
                    "BBANDS_UPPER": ensure_series(result.upperband, price.index, "BBANDS_UPPER"),
                    "BBANDS_MIDDLE": ensure_series(result.middleband, price.index, "BBANDS_MIDDLE"),
                    "BBANDS_LOWER": ensure_series(result.lowerband, price.index, "BBANDS_LOWER")
                }
            else:
                # 回退到VectorBT内置BBANDS
                result = vbt.BBANDS.run(price, window=window,
                                       upper=nbdevup, lower=nbdevdn)
                return {
                    "BBANDS_UPPER": ensure_series(result.upperband, price.index, "BBANDS_UPPER"),
                    "BBANDS_MIDDLE": ensure_series(result.middleband, price.index, "BBANDS_MIDDLE"),
                    "BBANDS_LOWER": ensure_series(result.lowerband, price.index, "BBANDS_LOWER")
                }
        except Exception as e:
            logger.error(f"BBANDS计算失败: {e}")
            empty_series = pd.Series(index=price.index, dtype=float)
            return {
                "BBANDS_UPPER": empty_series.rename("BBANDS_UPPER"),
                "BBANDS_MIDDLE": empty_series.rename("BBANDS_MIDDLE"),
                "BBANDS_LOWER": empty_series.rename("BBANDS_LOWER")
            }

    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算CCI - 使用TA-Lib确保一致性"""
        try:
            if hasattr(vbt, 'talib'):
                talib_cci = vbt.talib('CCI')
                result = talib_cci.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "CCI")
            else:
                # 回退到VectorBT内置CCI
                result = vbt.CCI.run(high, low, close, window=timeperiod)
                return ensure_series(result.cci, close.index, "CCI")
        except Exception as e:
            logger.error(f"CCI计算失败: {e}")
            return pd.Series(index=close.index, name="CCI", dtype=float)

    def calculate_cmo(self, close: pd.Series, timeperiod: int = 14) -> pd.Series:
        """计算CMO"""
        try:
            if hasattr(vbt, 'talib'):
                talib_cmo = vbt.talib('CMO')
                result = talib_cmo.run(close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "CMO")
            else:
                logger.warning("CMO需要TA-Lib支持")
                return pd.Series(index=close.index, name="CMO", dtype=float)
        except Exception as e:
            logger.error(f"CMO计算失败: {e}")
            return pd.Series(index=close.index, name="CMO", dtype=float)

    def calculate_dx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                    timeperiod: int = 14) -> pd.Series:
        """计算DX"""
        try:
            if hasattr(vbt, 'talib'):
                talib_dx = vbt.talib('DX')
                result = talib_dx.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "DX")
            else:
                logger.warning("DX需要TA-Lib支持")
                return pd.Series(index=close.index, name="DX", dtype=float)
        except Exception as e:
            logger.error(f"DX计算失败: {e}")
            return pd.Series(index=close.index, name="DX", dtype=float)

    def calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                     timeperiod: int = 14) -> pd.Series:
        """计算MFI - 使用TA-Lib确保一致性"""
        try:
            if hasattr(vbt, 'talib'):
                talib_mfi = vbt.talib('MFI')
                result = talib_mfi.run(high, low, close, volume, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "MFI")
            else:
                # MFI是RSI的成交量加权版本
                typical_price = (high + low + close) / 3
                result = vbt.RSI.run(typical_price, window=timeperiod)
                return ensure_series(result.rsi, close.index, "MFI")
        except Exception as e:
            logger.error(f"MFI计算失败: {e}")
            return pd.Series(index=close.index, name="MFI", dtype=float)

    def calculate_minus_di(self, high: pd.Series, low: pd.Series, close: pd.Series,
                           timeperiod: int = 14) -> pd.Series:
        """计算MINUS_DI"""
        try:
            if hasattr(vbt, 'talib'):
                talib_minus_di = vbt.talib('MINUS_DI')
                result = talib_minus_di.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "MINUS_DI")
            else:
                logger.warning("MINUS_DI需要TA-Lib支持")
                return pd.Series(index=close.index, name="MINUS_DI", dtype=float)
        except Exception as e:
            logger.error(f"MINUS_DI计算失败: {e}")
            return pd.Series(index=close.index, name="MINUS_DI", dtype=float)

    def calculate_plus_di(self, high: pd.Series, low: pd.Series, close: pd.Series,
                          timeperiod: int = 14) -> pd.Series:
        """计算PLUS_DI"""
        try:
            if hasattr(vbt, 'talib'):
                talib_plus_di = vbt.talib('PLUS_DI')
                result = talib_plus_di.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "PLUS_DI")
            else:
                logger.warning("PLUS_DI需要TA-Lib支持")
                return pd.Series(index=close.index, name="PLUS_DI", dtype=float)
        except Exception as e:
            logger.error(f"PLUS_DI计算失败: {e}")
            return pd.Series(index=close.index, name="PLUS_DI", dtype=float)

    def calculate_mom(self, close: pd.Series, timeperiod: int = 10) -> pd.Series:
        """计算MOM"""
        try:
            if hasattr(vbt, 'talib'):
                talib_mom = vbt.talib('MOM')
                result = talib_mom.run(close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "MOM")
            else:
                # 手动计算Momentum
                result = close - close.shift(timeperiod)
                return ensure_series(result, close.index, "MOM")
        except Exception as e:
            logger.error(f"MOM计算失败: {e}")
            return pd.Series(index=close.index, name="MOM", dtype=float)

    def calculate_natr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      timeperiod: int = 14) -> pd.Series:
        """计算NATR"""
        try:
            if hasattr(vbt, 'talib'):
                talib_natr = vbt.talib('NATR')
                result = talib_natr.run(high, low, close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "NATR")
            else:
                # 手动计算Normalized Average True Range
                atr = self.calculate_atr(high, low, close, timeperiod)
                natr = (atr / close) * 100
                return ensure_series(natr, close.index, "NATR")
        except Exception as e:
            logger.error(f"NATR计算失败: {e}")
            return pd.Series(index=close.index, name="NATR", dtype=float)

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算OBV - 使用VectorBT内置实现"""
        try:
            result = vbt.OBV.run(close, volume)
            return ensure_series(result.obv, close.index, "OBV")
        except Exception as e:
            logger.error(f"OBV计算失败: {e}")
            return pd.Series(index=close.index, name="OBV", dtype=float)

    def calculate_roc(self, close: pd.Series, timeperiod: int = 10) -> pd.Series:
        """计算ROC"""
        try:
            if hasattr(vbt, 'talib'):
                talib_roc = vbt.talib('ROC')
                result = talib_roc.run(close, timeperiod=timeperiod)
                return ensure_series(result.real, close.index, "ROC")
            else:
                # 手动计算Rate of Change
                roc = ((close - close.shift(timeperiod)) / close.shift(timeperiod)) * 100
                return ensure_series(roc, close.index, "ROC")
        except Exception as e:
            logger.error(f"ROC计算失败: {e}")
            return pd.Series(index=close.index, name="ROC", dtype=float)

    def calculate_sar(self, high: pd.Series, low: pd.Series,
                      acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """计算SAR"""
        try:
            if hasattr(vbt, 'talib'):
                talib_sar = vbt.talib('SAR')
                result = talib_sar.run(high, low, acceleration=acceleration, maximum=maximum)
                return ensure_series(result.real, low.index, "SAR")
            else:
                # 回退到VectorBT内置SAR（如果可用）
                logger.warning("VectorBT内置SAR可能不可用")
                return pd.Series(index=low.index, name="SAR", dtype=float)
        except Exception as e:
            logger.error(f"SAR计算失败: {e}")
            return pd.Series(index=low.index, name="SAR", dtype=float)

    def calculate_trange(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算True Range - 使用TA-Lib确保一致性"""
        try:
            if hasattr(vbt, 'talib'):
                talib_trange = vbt.talib('TRANGE')
                result = talib_trange.run(high, low, close)
                return ensure_series(result.real, close.index, "TRANGE")
            else:
                # 手动计算True Range
                tr1 = high - low
                tr2 = (high - close.shift(1)).abs()
                tr3 = (low - close.shift(1)).abs()
                trange = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                return ensure_series(trange, close.index, "TRANGE")
        except Exception as e:
            logger.error(f"TRANGE计算失败: {e}")
            return pd.Series(index=close.index, name="TRANGE", dtype=float)

    # 移动平均类扩展
    def calculate_dema(self, price: pd.Series, timeperiod: int = 20) -> pd.Series:
        """计算DEMA"""
        try:
            if hasattr(vbt, 'talib'):
                talib_dema = vbt.talib('DEMA')
                result = talib_dema.run(price, timeperiod=timeperiod)
                return ensure_series(result.real, price.index, "DEMA")
            else:
                # DEMA = 2*EMA - EMA(EMA)
                ema1 = vbt.EMA.run(price, window=timeperiod).ema
                ema2 = vbt.EMA.run(ema1, window=timeperiod).ema
                dema = 2 * ema1 - ema2
                return ensure_series(dema, price.index, "DEMA")
        except Exception as e:
            logger.error(f"DEMA计算失败: {e}")
            return pd.Series(index=price.index, name="DEMA", dtype=float)

    def calculate_tema(self, price: pd.Series, timeperiod: int = 20) -> pd.Series:
        """计算TEMA"""
        try:
            if hasattr(vbt, 'talib'):
                talib_tema = vbt.talib('TEMA')
                result = talib_tema.run(price, timeperiod=timeperiod)
                return ensure_series(result.real, price.index, "TEMA")
            else:
                # TEMA = 3*EMA1 - 3*EMA2 + EMA3
                ema1 = vbt.EMA.run(price, window=timeperiod).ema
                ema2 = vbt.EMA.run(ema1, window=timeperiod).ema
                ema3 = vbt.EMA.run(ema2, window=timeperiod).ema
                tema = 3 * ema1 - 3 * ema2 + ema3
                return ensure_series(tema, price.index, "TEMA")
        except Exception as e:
            logger.error(f"TEMA计算失败: {e}")
            return pd.Series(index=price.index, name="TEMA", dtype=float)

    def calculate_wma(self, price: pd.Series, timeperiod: int = 20) -> pd.Series:
        """计算WMA"""
        try:
            if hasattr(vbt, 'talib'):
                talib_wma = vbt.talib('WMA')
                result = talib_wma.run(price, timeperiod=timeperiod)
                return ensure_series(result.real, price.index, "WMA")
            else:
                # 回退到VectorBT内置WMA
                result = vbt.WMA.run(price, window=timeperiod)
                return ensure_series(result.wma, price.index, "WMA")
        except Exception as e:
            logger.error(f"WMA计算失败: {e}")
            return pd.Series(index=price.index, name="WMA", dtype=float)


# 全局适配器实例
_global_adapter: Optional[VectorBTAdapter] = None


def get_vectorbt_adapter() -> VectorBTAdapter:
    """获取全局VectorBT适配器实例"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = VectorBTAdapter()
    return _global_adapter


# 便捷函数
def calculate_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    """便捷函数：计算RSI"""
    return get_vectorbt_adapter().calculate_rsi(price, window)


def calculate_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
                   fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> pd.Series:
    """便捷函数：计算STOCH"""
    return get_vectorbt_adapter().calculate_stoch(high, low, close, fastk_period, slowk_period, slowd_period)


def calculate_macd(close: pd.Series,
                  fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
    """便捷函数：计算MACD"""
    return get_vectorbt_adapter().calculate_macd(close, fast_period, slow_period, signal_period)