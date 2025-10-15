#!/usr/bin/env python3
"""VectorBT指标适配器 - 统一接入vbt/pandas_ta/talib指标到因子引擎"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

logger = logging.getLogger(__name__)


class VBTIndicatorAdapter:
    """VectorBT指标适配器 - 统一适配层"""

    def __init__(self, price_field: str = "close", engine_version: str = "1.0.0"):
        self.price_field = price_field
        self.engine_version = engine_version
        self.indicators_computed = 0

    def compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有VBT/pandas_ta/talib指标

        Args:
            df: DataFrame with [date, open, high, low, close, volume]

        Returns:
            DataFrame with all indicators
        """
        logger.info(f"VBT适配器开始计算，输入: {df.shape}")

        # 提取价格数据
        open_price = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        factors = {}

        # ===== 1. VectorBT内置指标 =====
        try:
            factors.update(self._compute_vbt_indicators(df, close, high, low, volume))
        except Exception as e:
            logger.warning(f"VBT指标计算失败: {e}")

        # ===== 2. TA-Lib指标（完整版） =====
        try:
            factors.update(
                self._compute_talib_indicators(open_price, high, low, close, volume)
            )
        except Exception as e:
            logger.warning(f"TA-Lib指标计算失败: {e}")

        # ===== 3. 自定义统计指标 =====
        try:
            factors.update(self._compute_custom_indicators(close, high, low, volume))
        except Exception as e:
            logger.warning(f"自定义指标计算失败: {e}")

        # 转换为DataFrame
        result_df = pd.DataFrame(factors, index=df.index)

        # 添加date列
        if "date" in df.columns:
            result_df["date"] = df["date"].values

        self.indicators_computed = len(factors)
        logger.info(f"✅ VBT适配器完成: {self.indicators_computed} 个指标")

        return result_df

    def _compute_vbt_indicators(
        self,
        df: pd.DataFrame,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """计算VectorBT内置指标"""
        factors = {}

        # MA系列
        for window in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 252]:
            try:
                ma = vbt.MA.run(close, window=window, short_name="ma")
                factors[f"VBT_MA_{window}"] = ma.ma.values
            except:
                pass

        # EMA系列
        for span in [5, 10, 12, 20, 26, 30, 40, 50, 60, 80, 100, 120]:
            try:
                ema = vbt.MA.run(close, window=span, ewm=True, short_name="ema")
                factors[f"VBT_EMA_{span}"] = ema.ma.values
            except:
                pass

        # MACD
        for fast, slow, signal in [(12, 26, 9), (16, 34, 7), (20, 42, 8), (5, 35, 5)]:
            try:
                macd = vbt.MACD.run(
                    close, fast_window=fast, slow_window=slow, signal_window=signal
                )
                factors[f"VBT_MACD_{fast}_{slow}_{signal}"] = macd.macd.values
                factors[f"VBT_MACD_SIGNAL_{fast}_{slow}_{signal}"] = macd.signal.values
                factors[f"VBT_MACD_HIST_{fast}_{slow}_{signal}"] = macd.hist.values
            except:
                pass

        # RSI
        for window in [6, 9, 12, 14, 20, 24, 30, 60]:
            try:
                rsi = vbt.RSI.run(close, window=window)
                factors[f"VBT_RSI_{window}"] = rsi.rsi.values
            except:
                pass

        # BBANDS
        for window in [10, 15, 20, 25, 30, 40, 50]:
            for alpha in [1.5, 2.0, 2.5]:
                try:
                    bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
                    factors[f"VBT_BB_UPPER_{window}_{alpha}"] = bb.upper.values
                    factors[f"VBT_BB_MIDDLE_{window}_{alpha}"] = bb.middle.values
                    factors[f"VBT_BB_LOWER_{window}_{alpha}"] = bb.lower.values
                    factors[f"VBT_BB_WIDTH_{window}_{alpha}"] = bb.bandwidth.values
                    factors[f"VBT_BB_PERCENT_{window}_{alpha}"] = bb.percent.values
                except:
                    pass

        # STOCH
        for k_window in [5, 9, 14, 20]:
            for d_window in [3, 5]:
                try:
                    stoch = vbt.STOCH.run(
                        high, low, close, k_window=k_window, d_window=d_window
                    )
                    factors[f"VBT_STOCH_K_{k_window}_{d_window}"] = (
                        stoch.percent_k.values
                    )
                    factors[f"VBT_STOCH_D_{k_window}_{d_window}"] = (
                        stoch.percent_d.values
                    )
                except:
                    pass

        # ATR
        for window in [7, 10, 14, 20, 30, 60]:
            try:
                atr = vbt.ATR.run(high, low, close, window=window)
                factors[f"VBT_ATR_{window}"] = atr.atr.values
            except:
                pass

        # OBV
        try:
            obv = vbt.OBV.run(close, volume)
            factors["VBT_OBV"] = obv.obv.values
        except:
            pass

        logger.info(f"VBT内置指标: {len(factors)} 个")
        return factors

    def _compute_talib_indicators(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """计算TA-Lib完整指标集"""
        factors = {}

        # Overlap Studies (重叠指标)
        for period in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 252]:
            factors[f"TA_SMA_{period}"] = talib.SMA(close, timeperiod=period)
            factors[f"TA_EMA_{period}"] = talib.EMA(close, timeperiod=period)

        for period in [10, 20, 30, 40, 50]:
            factors[f"TA_WMA_{period}"] = talib.WMA(close, timeperiod=period)
            factors[f"TA_DEMA_{period}"] = talib.DEMA(close, timeperiod=period)
            factors[f"TA_TEMA_{period}"] = talib.TEMA(close, timeperiod=period)
            factors[f"TA_TRIMA_{period}"] = talib.TRIMA(close, timeperiod=period)

        for period in [10, 20, 30]:
            factors[f"TA_KAMA_{period}"] = talib.KAMA(close, timeperiod=period)

        # MAMA
        mama, fama = talib.MAMA(close)
        factors["TA_MAMA"] = mama
        factors["TA_FAMA"] = fama

        # Momentum Indicators (动量指标)
        for fast, slow, signal in [(12, 26, 9), (16, 34, 7), (20, 42, 8), (5, 35, 5)]:
            macd, signal_line, hist = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            factors[f"TA_MACD_{fast}_{slow}_{signal}"] = macd
            factors[f"TA_MACD_SIGNAL_{fast}_{slow}_{signal}"] = signal_line
            factors[f"TA_MACD_HIST_{fast}_{slow}_{signal}"] = hist

        for period in [6, 9, 12, 14, 20, 24, 30, 60]:
            factors[f"TA_RSI_{period}"] = talib.RSI(close, timeperiod=period)

        for period in [10, 14, 20, 30]:
            factors[f"TA_MOM_{period}"] = talib.MOM(close, timeperiod=period)
            factors[f"TA_ROC_{period}"] = talib.ROC(close, timeperiod=period)
            factors[f"TA_ROCP_{period}"] = talib.ROCP(close, timeperiod=period)
            factors[f"TA_ROCR_{period}"] = talib.ROCR(close, timeperiod=period)

        # Volatility Indicators (波动率指标)
        for period in [10, 15, 20, 25, 30, 40, 50]:
            for nbdev in [1.5, 2.0, 2.5]:
                upper, middle, lower = talib.BBANDS(
                    close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
                )
                factors[f"TA_BB_UPPER_{period}_{nbdev}"] = upper
                factors[f"TA_BB_MIDDLE_{period}_{nbdev}"] = middle
                factors[f"TA_BB_LOWER_{period}_{nbdev}"] = lower

        for period in [7, 10, 14, 20, 30, 60]:
            factors[f"TA_ATR_{period}"] = talib.ATR(high, low, close, timeperiod=period)
            factors[f"TA_NATR_{period}"] = talib.NATR(
                high, low, close, timeperiod=period
            )
            factors[f"TA_TRANGE_{period}"] = talib.TRANGE(high, low, close)

        # Volume Indicators (成交量指标)
        factors["TA_OBV"] = talib.OBV(close, volume)
        factors["TA_AD"] = talib.AD(high, low, close, volume)
        factors["TA_ADOSC"] = talib.ADOSC(
            high, low, close, volume, fastperiod=3, slowperiod=10
        )

        # Cycle Indicators (周期指标)
        factors["TA_HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
        factors["TA_HT_DCPHASE"] = talib.HT_DCPHASE(close)
        factors["TA_HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

        # Price Transform (价格变换)
        factors["TA_AVGPRICE"] = talib.AVGPRICE(open_price, high, low, close)
        factors["TA_MEDPRICE"] = talib.MEDPRICE(high, low)
        factors["TA_TYPPRICE"] = talib.TYPPRICE(high, low, close)
        factors["TA_WCLPRICE"] = talib.WCLPRICE(high, low, close)

        # Volatility Indicators (更多波动率)
        for period in [10, 20, 30, 40, 60]:
            factors[f"TA_STDDEV_{period}"] = talib.STDDEV(
                close, timeperiod=period, nbdev=1
            )
            factors[f"TA_VAR_{period}"] = talib.VAR(close, timeperiod=period, nbdev=1)

        # Pattern Recognition (形态识别 - 选择性添加)
        factors["TA_CDL_DOJI"] = talib.CDLDOJI(open_price, high, low, close)
        factors["TA_CDL_HAMMER"] = talib.CDLHAMMER(open_price, high, low, close)
        factors["TA_CDL_ENGULFING"] = talib.CDLENGULFING(open_price, high, low, close)
        factors["TA_CDL_MORNINGSTAR"] = talib.CDLMORNINGSTAR(
            open_price, high, low, close
        )
        factors["TA_CDL_EVENINGSTAR"] = talib.CDLEVENINGSTAR(
            open_price, high, low, close
        )

        logger.info(f"TA-Lib指标: {len(factors)} 个")
        return factors

    def _compute_custom_indicators(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """计算自定义统计指标"""
        factors = {}

        # 收益率系列
        for period in [1, 2, 3, 5, 10, 20, 30, 60]:
            ret = np.roll(close, -period) / close - 1
            factors[f"RETURN_{period}"] = ret

        # 波动率系列
        for window in [5, 10, 20, 30, 60]:
            log_returns = pd.Series(close).pct_change()
            vol = log_returns.rolling(window).std()
            factors[f"VOLATILITY_{window}"] = vol.values

        # 价格位置
        for window in [10, 20, 30, 60]:
            rolling_high = pd.Series(high).rolling(window).max()
            rolling_low = pd.Series(low).rolling(window).min()
            factors[f"PRICE_POSITION_{window}"] = (close - rolling_low) / (
                rolling_high - rolling_low + 1e-10
            )

        # 成交量比率
        for window in [5, 10, 20, 30]:
            vol_ma = pd.Series(volume).rolling(window).mean()
            factors[f"VOLUME_RATIO_{window}"] = volume / (vol_ma + 1e-10)

        # 动量指标
        for window in [5, 10, 20, 30]:
            factors[f"MOMENTUM_{window}"] = close / np.roll(close, window) - 1

        logger.info(f"自定义指标: {len(factors)} 个")
        return factors
