"""
共享因子计算器 - 统一计算逻辑

⚠️ 重要约束:
1. 所有因子计算必须通过这个模块
2. 禁止在其他地方重新实现因子计算
3. 修改因子计算逻辑只能在这里修改

使用场景:
- factor_generation: 批量生成因子时间序列
- factor_screening: 筛选优秀因子
- factor_engine: 按需计算因子
- hk_midfreq: 回测时生成信号

目标: 确保因子生成、筛选、回测的计算逻辑100%一致
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import talib

    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib未安装，将使用纯Python实现（性能较低）")

logger = logging.getLogger(__name__)


class SharedFactorCalculators:
    """
    统一因子计算器

    所有因子计算的唯一入口
    """

    def __init__(self):
        """初始化共享计算器"""
        self.has_talib = HAS_TALIB
        if not HAS_TALIB:
            logger.warning(
                "⚠️ TA-Lib未安装! 将使用纯Python实现。"
                "建议安装: conda install -c conda-forge ta-lib"
            )

    # ========== 技术指标 - Momentum ==========

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI - Relative Strength Index

        Args:
            close: 收盘价序列
            period: 周期，默认14

        Returns:
            RSI值序列 (0-100)

        算法: 使用TA-Lib标准实现（EMA平滑）
        """
        if not self.has_talib:
            logger.warning("RSI计算回退到纯Python实现")
            return self._rsi_python(close, period)

        try:
            rsi = talib.RSI(close, timeperiod=period)
            return rsi
        except Exception as e:
            logger.error(f"RSI计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_stoch(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """
        STOCH - Stochastic Oscillator

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            fastk_period: Fast %K周期，默认14
            slowk_period: Slow %K平滑周期，默认3
            slowd_period: %D平滑周期，默认3

        Returns:
            {'slowk': Series, 'slowd': Series}

        算法: 使用TA-Lib标准实现（SMA平滑）
        """
        if not self.has_talib:
            logger.warning("STOCH计算回退到纯Python实现")
            return self._stoch_python(
                high, low, close, fastk_period, slowk_period, slowd_period
            )

        try:
            slowk, slowd = talib.STOCH(
                high,
                low,
                close,
                fastk_period=fastk_period,
                slowk_matype=0,  # SMA
                slowk_period=slowk_period,
                slowd_matype=0,  # SMA
                slowd_period=slowd_period,
            )
            return {"slowk": slowk, "slowd": slowd}
        except Exception as e:
            logger.error(f"STOCH计算失败: {e}")
            return {
                "slowk": pd.Series(np.nan, index=close.index),
                "slowd": pd.Series(np.nan, index=close.index),
            }

    def calculate_willr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14,
    ) -> pd.Series:
        """
        WILLR - Williams' %R

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            timeperiod: 周期，默认14

        Returns:
            WILLR值序列 (-100 to 0)

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("WILLR计算回退到纯Python实现")
            return self._willr_python(high, low, close, timeperiod)

        try:
            willr = talib.WILLR(high, low, close, timeperiod=timeperiod)
            return willr
        except Exception as e:
            logger.error(f"WILLR计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_macd(
        self,
        close: pd.Series,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> Dict[str, pd.Series]:
        """
        MACD - Moving Average Convergence Divergence

        Args:
            close: 收盘价序列
            fastperiod: 快速EMA周期，默认12
            slowperiod: 慢速EMA周期，默认26
            signalperiod: 信号线周期，默认9

        Returns:
            {'macd': Series, 'signal': Series, 'hist': Series}

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("MACD计算回退到纯Python实现")
            return self._macd_python(close, fastperiod, slowperiod, signalperiod)

        try:
            macd, signal, hist = talib.MACD(
                close,
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
            )
            return {"macd": macd, "signal": signal, "hist": hist}
        except Exception as e:
            logger.error(f"MACD计算失败: {e}")
            return {
                "macd": pd.Series(np.nan, index=close.index),
                "signal": pd.Series(np.nan, index=close.index),
                "hist": pd.Series(np.nan, index=close.index),
            }

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        ADX - Average Directional Movement Index

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期，默认14

        Returns:
            ADX值序列

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("ADX计算回退到纯Python实现")
            return self._adx_python(high, low, close, period)

        try:
            adx = talib.ADX(high, low, close, timeperiod=period)
            return adx
        except Exception as e:
            logger.error(f"ADX计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14,
    ) -> pd.Series:
        """
        ATR - Average True Range

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            timeperiod: 周期，默认14

        Returns:
            ATR值序列

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("ATR计算回退到纯Python实现")
            return self._atr_python(high, low, close, timeperiod)

        try:
            atr = talib.ATR(high, low, close, timeperiod=timeperiod)
            return atr
        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_bbands(
        self,
        close: pd.Series,
        period: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
    ) -> Dict[str, pd.Series]:
        """
        BBANDS - Bollinger Bands

        Args:
            close: 收盘价序列
            period: 周期，默认20
            nbdevup: 上轨标准差倍数，默认2.0
            nbdevdn: 下轨标准差倍数，默认2.0

        Returns:
            {'upper': Series, 'middle': Series, 'lower': Series}

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("BBANDS计算回退到纯Python实现")
            return self._bbands_python(close, period, nbdevup, nbdevdn)

        try:
            upper, middle, lower = talib.BBANDS(
                close,
                timeperiod=period,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
                matype=0,  # SMA
            )
            return {"upper": upper, "middle": middle, "lower": lower}
        except Exception as e:
            logger.error(f"BBANDS计算失败: {e}")
            return {
                "upper": pd.Series(np.nan, index=close.index),
                "middle": pd.Series(np.nan, index=close.index),
                "lower": pd.Series(np.nan, index=close.index),
            }

    def calculate_plus_di(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        PLUS_DI - Plus Directional Indicator

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14

        Returns:
            PLUS_DI值序列

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("PLUS_DI计算回退到纯Python实现")
            return self._plus_di_python(high, low, close, period)

        try:
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
            return plus_di
        except Exception as e:
            logger.error(f"PLUS_DI计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_trange(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """
        TRANGE - True Range

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            True Range值序列

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("TRANGE计算回退到纯Python实现")
            return self._trange_python(high, low, close)

        try:
            trange = talib.TRANGE(high, low, close)
            return trange
        except Exception as e:
            logger.error(f"TRANGE计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_roc(
        self,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        ROC - Rate of Change

        Args:
            close: 收盘价序列
            period: 计算周期，默认14

        Returns:
            ROC值序列（百分比）

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("ROC计算回退到纯Python实现")
            return self._roc_python(close, period)

        try:
            roc = talib.ROC(close, timeperiod=period)
            return roc
        except Exception as e:
            logger.error(f"ROC计算失败: {e}")
            return pd.Series(np.nan, index=close.index)

    def calculate_candlestick_pattern(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pattern_name: str,
    ) -> pd.Series:
        """
        K线模式识别

        Args:
            open_price: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            pattern_name: 模式名称 (如 'CDL2CROWS', 'CDL3BLACKCROWS' 等)

        Returns:
            模式识别结果Series，100表示看涨，-100表示看跌，0表示无模式
        """
        if self.has_talib:
            try:
                # 使用TA-Lib进行K线模式识别
                talib_function = getattr(talib, pattern_name)
                result = talib_function(
                    open_price.values, high.values, low.values, close.values
                )
                return pd.Series(result, index=open_price.index, name=pattern_name)
            except Exception as e:
                logger.warning(f"TA-Lib {pattern_name} 计算失败: {e}")
                return pd.Series(0, index=open_price.index, name=pattern_name)
        else:
            # 纯Python实现的简单K线模式识别（只实现几个常见模式）
            return self._calculate_candlestick_pattern_fallback(
                open_price, high, low, close, pattern_name
            )

    def _calculate_candlestick_pattern_fallback(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        pattern_name: str,
    ) -> pd.Series:
        """纯Python实现的K线模式识别备用方案"""
        result = pd.Series(0, index=open_price.index, name=pattern_name)

        if pattern_name == "CDL2CROWS":
            # 两只乌鸦 - 连续两根阴线，第二根阴线在第一根阴线实体内
            prev_close = close.shift(1)
            prev_open = open_price.shift(1)
            prev2_close = close.shift(2)
            prev2_open = open_price.shift(2)

            # 前两根为阳线，当前为阴线
            condition1 = close < open_price
            condition2 = prev_close > prev_open
            condition3 = prev2_close > prev2_open
            # 当前阴线在第一根阳线实体内
            condition4 = close < prev_close
            condition5 = open > prev_close

            result = np.where(
                condition1 & condition2 & condition3 & condition4 & condition5, -100, 0
            )

        elif pattern_name == "CDL3BLACKCROWS":
            # 三只乌鸦 - 连续三根阴线，每根阴线收盘价低于前一根
            prev_close = close.shift(1)
            prev_open = open_price.shift(1)
            prev2_close = close.shift(2)
            prev2_open = open_price.shift(2)

            condition1 = close < open_price  # 当前为阴线
            condition2 = prev_close < prev_open  # 前一天为阴线
            condition3 = prev2_close < prev2_open  # 前两天为阴线
            condition4 = close < prev_close  # 收盘价低于前一天

            result = np.where(
                condition1 & condition2 & condition3 & condition4, -100, 0
            )

        elif pattern_name == "CDLHAMMER":
            # 锤子线 - 长下影线，小实体，实体在顶部
            body_size = abs(close - open_price)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low

            # 下影线长度大于实体2倍，上影线很小
            condition1 = lower_shadow > 2 * body_size
            condition2 = upper_shadow < body_size * 0.1

            result = np.where(condition1 & condition2, 100, 0)

        elif pattern_name == "CDLSHOOTINGSTAR":
            # 射击之星 - 长上影线，小实体，实体在底部
            body_size = abs(close - open_price)
            upper_shadow = high - np.maximum(close, open_price)
            lower_shadow = np.minimum(close, open_price) - low

            # 上影线长度大于实体2倍，下影线很小
            condition1 = upper_shadow > 2 * body_size
            condition2 = lower_shadow < body_size * 0.1

            result = np.where(condition1 & condition2, -100, 0)

        return pd.Series(result, index=open_price.index, name=pattern_name)

    # ========== 纯Python实现（备用） ==========

    def _rsi_python(self, close: pd.Series, period: int) -> pd.Series:
        """RSI纯Python实现"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _stoch_python(self, high, low, close, fastk_period, slowk_period, slowd_period):
        """STOCH纯Python实现"""
        lowest_low = low.rolling(window=fastk_period, min_periods=fastk_period).min()
        highest_high = high.rolling(window=fastk_period, min_periods=fastk_period).max()

        fastk = (
            100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1e-10)
        )
        slowk = fastk.rolling(window=slowk_period, min_periods=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period, min_periods=slowd_period).mean()

        return {"slowk": slowk, "slowd": slowd}

    def _willr_python(self, high, low, close, period):
        """WILLR纯Python实现"""
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()

        willr = (
            -100
            * (highest_high - close)
            / (highest_high - lowest_low).replace(0, 1e-10)
        )
        return willr

    def _macd_python(self, close, fastperiod, slowperiod, signalperiod):
        """MACD纯Python实现"""
        ema_fast = close.ewm(span=fastperiod, adjust=False).mean()
        ema_slow = close.ewm(span=slowperiod, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal

        return {"macd": macd, "signal": signal, "hist": hist}

    def _adx_python(self, high, low, close, period):
        """ADX纯Python实现（简化版）"""
        logger.warning("ADX纯Python实现未完全实现，返回NaN")
        return pd.Series(np.nan, index=close.index)

    def _atr_python(self, high, low, close, period):
        """ATR纯Python实现"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return atr

    def _bbands_python(self, close, period, nbdevup, nbdevdn):
        """BBANDS纯Python实现"""
        middle = close.rolling(window=period, min_periods=period).mean()
        std = close.rolling(window=period, min_periods=period).std()

        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)

        return {"upper": upper, "middle": middle, "lower": lower}

    def _plus_di_python(self, high, low, close, period):
        """PLUS_DI纯Python实现"""
        # 计算True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # 计算方向性移动
        plus_dm = high.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (high.diff() > low.diff()), 0)

        # 平滑处理
        atr = tr.rolling(window=period, min_periods=period).sum()
        plus_di_smooth = plus_dm.rolling(window=period, min_periods=period).sum()

        # 计算PLUS_DI
        plus_di = 100 * (plus_di_smooth / atr).replace(0, np.nan)
        return plus_di

    def _trange_python(self, high, low, close):
        """TRANGE纯Python实现"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr

    def _roc_python(self, close, period):
        """ROC纯Python实现"""
        roc = (close - close.shift(period)) / close.shift(period) * 100
        return roc


# 全局单例 - 确保所有地方使用同一个实例
SHARED_CALCULATORS = SharedFactorCalculators()


# ========== 便捷函数 ==========


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI计算便捷函数"""
    return SHARED_CALCULATORS.calculate_rsi(close, period)


def calculate_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> Dict[str, pd.Series]:
    """STOCH计算便捷函数"""
    return SHARED_CALCULATORS.calculate_stoch(
        high, low, close, fastk_period, slowk_period, slowd_period
    )


def calculate_willr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """WILLR计算便捷函数"""
    return SHARED_CALCULATORS.calculate_willr(high, low, close, timeperiod)


def calculate_macd(
    close: pd.Series,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Dict[str, pd.Series]:
    """MACD计算便捷函数"""
    return SHARED_CALCULATORS.calculate_macd(
        close, fastperiod, slowperiod, signalperiod
    )


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX计算便捷函数"""
    return SHARED_CALCULATORS.calculate_adx(high, low, close, period)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """ATR计算便捷函数"""
    return SHARED_CALCULATORS.calculate_atr(high, low, close, timeperiod)


def calculate_bbands(
    close: pd.Series,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> Dict[str, pd.Series]:
    """BBANDS计算便捷函数"""
    return SHARED_CALCULATORS.calculate_bbands(close, period, nbdevup, nbdevdn)


def calculate_plus_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """PLUS_DI计算便捷函数"""
    return SHARED_CALCULATORS.calculate_plus_di(high, low, close, period)


def calculate_trange(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """TRANGE计算便捷函数"""
    return SHARED_CALCULATORS.calculate_trange(high, low, close)


def calculate_roc(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ROC计算便捷函数"""
    return SHARED_CALCULATORS.calculate_roc(close, period)


def calculate_candlestick_pattern(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    pattern_name: str,
) -> pd.Series:
    """K线模式识别计算便捷函数"""
    return SHARED_CALCULATORS.calculate_candlestick_pattern(
        open_price, high, low, close, pattern_name
    )

