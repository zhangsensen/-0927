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
        period: int = 14,
    ) -> pd.Series:
        """
        WILLR - Williams' %R

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期，默认14

        Returns:
            WILLR值序列 (-100 to 0)

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("WILLR计算回退到纯Python实现")
            return self._willr_python(high, low, close, period)

        try:
            willr = talib.WILLR(high, low, close, timeperiod=period)
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
        period: int = 14,
    ) -> pd.Series:
        """
        ATR - Average True Range

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期，默认14

        Returns:
            ATR值序列

        算法: 使用TA-Lib标准实现
        """
        if not self.has_talib:
            logger.warning("ATR计算回退到纯Python实现")
            return self._atr_python(high, low, close, period)

        try:
            atr = talib.ATR(high, low, close, timeperiod=period)
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
    period: int = 14,
) -> pd.Series:
    """WILLR计算便捷函数"""
    return SHARED_CALCULATORS.calculate_willr(high, low, close, period)


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
    period: int = 14,
) -> pd.Series:
    """ATR计算便捷函数"""
    return SHARED_CALCULATORS.calculate_atr(high, low, close, period)


def calculate_bbands(
    close: pd.Series,
    period: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
) -> Dict[str, pd.Series]:
    """BBANDS计算便捷函数"""
    return SHARED_CALCULATORS.calculate_bbands(close, period, nbdevup, nbdevdn)
