#!/usr/bin/env python3
"""简单因子计算器 - 直接使用TA-Lib计算ETF日线因子"""

import logging

import numpy as np
import pandas as pd
import talib

logger = logging.getLogger(__name__)


class SimpleFactorCalculator:
    """简单因子计算器 - 154个技术指标"""

    def __init__(self):
        self.factor_count = 0

    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume]

        Returns:
            DataFrame with all factors
        """
        logger.info(f"计算因子，输入数据: {df.shape}")

        # 提取价格数据
        open_price = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        factors = {}

        # 1. 移动平均线（MA）
        for period in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120]:
            factors[f"SMA_{period}"] = talib.SMA(close, timeperiod=period)

        # 2. 指数移动平均（EMA）
        for period in [5, 10, 12, 20, 26, 30, 40, 50, 60]:
            factors[f"EMA_{period}"] = talib.EMA(close, timeperiod=period)

        # 3. MACD
        for fast, slow, signal in [(12, 26, 9), (16, 34, 7), (20, 42, 8)]:
            macd, signal_line, hist = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            factors[f"MACD_{fast}_{slow}_{signal}"] = macd
            factors[f"MACD_SIGNAL_{fast}_{slow}_{signal}"] = signal_line
            factors[f"MACD_HIST_{fast}_{slow}_{signal}"] = hist

        # 4. RSI
        for period in [6, 12, 14, 20, 30, 60]:
            factors[f"RSI_{period}"] = talib.RSI(close, timeperiod=period)

        # 5. 布林带（BB）
        for period in [10, 20, 30, 40, 50]:
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=period, nbdevup=2, nbdevdn=2
            )
            factors[f"BB_UPPER_{period}"] = upper
            factors[f"BB_MIDDLE_{period}"] = middle
            factors[f"BB_LOWER_{period}"] = lower
            factors[f"BB_WIDTH_{period}"] = (upper - lower) / middle

        # 6. 随机指标（STOCH）
        for fastk, slowk, slowd in [(5, 3, 3), (14, 3, 3), (20, 5, 5)]:
            slowk_val, slowd_val = talib.STOCH(
                high,
                low,
                close,
                fastk_period=fastk,
                slowk_period=slowk,
                slowd_period=slowd,
            )
            factors[f"STOCH_K_{fastk}_{slowk}_{slowd}"] = slowk_val
            factors[f"STOCH_D_{fastk}_{slowk}_{slowd}"] = slowd_val

        # 7. ATR
        for period in [7, 14, 20, 30, 60]:
            factors[f"ATR_{period}"] = talib.ATR(high, low, close, timeperiod=period)

        # 8. ADX
        for period in [14, 20, 30]:
            factors[f"ADX_{period}"] = talib.ADX(high, low, close, timeperiod=period)

        # 9. CCI
        for period in [14, 20, 30]:
            factors[f"CCI_{period}"] = talib.CCI(high, low, close, timeperiod=period)

        # 10. Williams %R
        for period in [14, 20, 30]:
            factors[f"WILLR_{period}"] = talib.WILLR(
                high, low, close, timeperiod=period
            )

        # 11. MOM
        for period in [10, 20, 30]:
            factors[f"MOM_{period}"] = talib.MOM(close, timeperiod=period)

        # 12. ROC
        for period in [10, 20, 30]:
            factors[f"ROC_{period}"] = talib.ROC(close, timeperiod=period)

        # 13. OBV
        factors["OBV"] = talib.OBV(close, volume)

        # 14. AD
        factors["AD"] = talib.AD(high, low, close, volume)

        # 15. ADOSC
        factors["ADOSC"] = talib.ADOSC(
            high, low, close, volume, fastperiod=3, slowperiod=10
        )

        # 16. TRIX
        for period in [15, 20, 30]:
            factors[f"TRIX_{period}"] = talib.TRIX(close, timeperiod=period)

        # 17. ULTOSC
        factors["ULTOSC"] = talib.ULTOSC(
            high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28
        )

        # 18. AROON
        for period in [14, 20, 30]:
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
            factors[f"AROON_DOWN_{period}"] = aroon_down
            factors[f"AROON_UP_{period}"] = aroon_up

        # 19. AROONOSC
        for period in [14, 20, 30]:
            factors[f"AROONOSC_{period}"] = talib.AROONOSC(high, low, timeperiod=period)

        # 20. BOP
        factors["BOP"] = talib.BOP(open_price, high, low, close)

        # 21. PLUS_DI / MINUS_DI
        for period in [14, 20]:
            factors[f"PLUS_DI_{period}"] = talib.PLUS_DI(
                high, low, close, timeperiod=period
            )
            factors[f"MINUS_DI_{period}"] = talib.MINUS_DI(
                high, low, close, timeperiod=period
            )

        # 22. STDDEV
        for period in [10, 20, 30, 40, 60]:
            factors[f"STDDEV_{period}"] = talib.STDDEV(
                close, timeperiod=period, nbdev=1
            )

        # 23. VAR
        for period in [10, 20, 30]:
            factors[f"VAR_{period}"] = talib.VAR(close, timeperiod=period, nbdev=1)

        # 24. LINEARREG
        for period in [14, 20, 30]:
            factors[f"LINEARREG_{period}"] = talib.LINEARREG(close, timeperiod=period)

        # 25. TSF
        for period in [14, 20, 30]:
            factors[f"TSF_{period}"] = talib.TSF(close, timeperiod=period)

        # 转换为DataFrame
        result_df = pd.DataFrame(factors, index=df.index)

        # 添加date列
        if "date" in df.columns:
            result_df["date"] = df["date"].values

        self.factor_count = len(factors)
        logger.info(f"✅ 计算完成: {self.factor_count} 个因子")

        return result_df
