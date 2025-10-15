#!/usr/bin/env python3
"""VBT适配器 - 生产级实现

Linus原则：
1. T+1强制：所有因子先shift(1)
2. min_history显式：window+1，不足=NaN
3. 价格口径单一：price_field统一
4. cache_key唯一：factor_id+params+price_field+engine_version
5. 元数据回传：family/bucket/min_history/required_fields
"""

import hashlib
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import talib
import vectorbt as vbt

logger = logging.getLogger(__name__)


class VBTIndicatorAdapter:
    """VBT适配器 - 生产级"""

    def __init__(self, price_field: str = "close", engine_version: str = "1.0.1"):
        self.price_field = price_field
        self.engine_version = engine_version
        self.indicators_computed = 0
        self.metadata = {}  # factor_id -> {min_history, family, bucket}

    def compute_all_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        计算所有指标（T+1安全）

        Args:
            df: DataFrame with [date, open, high, low, close, volume]

        Returns:
            (factors_df, metadata_dict)
        """
        logger.info(f"VBT适配器开始计算，输入: {df.shape}")

        # 提取价格数据
        open_price = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        factors = {}

        # 1. VBT内置指标
        try:
            vbt_factors = self._compute_vbt_indicators(close, high, low, volume)
            factors.update(vbt_factors)
        except Exception as e:
            logger.warning(f"VBT指标失败: {e}")

        # 2. TA-Lib完整指标
        try:
            talib_factors = self._compute_talib_indicators(
                open_price, high, low, close, volume
            )
            factors.update(talib_factors)
        except Exception as e:
            logger.warning(f"TA-Lib指标失败: {e}")

        # 3. 自定义统计指标
        try:
            custom_factors = self._compute_custom_indicators(close, high, low, volume)
            factors.update(custom_factors)
        except Exception as e:
            logger.warning(f"自定义指标失败: {e}")

        # 转换为DataFrame
        result_df = pd.DataFrame(factors, index=df.index)

        # 添加date列
        if "date" in df.columns:
            result_df["date"] = df["date"].values

        self.indicators_computed = len(factors)
        logger.info(f"✅ 计算完成: {self.indicators_computed} 个指标")

        return result_df, self.metadata

    def _apply_t1_shift(
        self, series: np.ndarray, factor_id: str, min_history: int
    ) -> np.ndarray:
        """
        应用T+1 shift（强制）

        精确实现：
        1. 前 min_history 位置 NaN（窗口不足）
        2. 整体右移 1 位（T+1），首位置 NaN

        Args:
            series: 原始序列
            factor_id: 因子ID
            min_history: 最小历史长度

        Returns:
            T+1安全的序列
        """
        n = len(series)
        result = np.full(n, np.nan, dtype=np.float64)

        # 前 min_history 位置 NaN（窗口不足）
        # 从 min_history 开始，复制 series[min_history-1:] 到 result[min_history:]
        # 即 result[min_history] = series[min_history-1]（右移1位）
        if n > min_history:
            result[min_history:] = series[min_history - 1 : n - 1]

        return result

    def _register_metadata(
        self,
        factor_id: str,
        min_history: int,
        family: str,
        bucket: str,
        params: Dict = None,
    ):
        """注册因子元数据"""
        self.metadata[factor_id] = {
            "min_history": min_history,
            "family": family,
            "bucket": bucket,
            "params": params or {},
            "price_field": self.price_field,
            "engine_version": self.engine_version,
            "cache_key": self._generate_cache_key(factor_id, min_history, params),
        }

    def _generate_cache_key(
        self, factor_id: str, min_history: int, params: Dict = None
    ) -> str:
        """生成唯一缓存键"""
        key_parts = [
            factor_id,
            str(min_history),
            self.price_field,
            self.engine_version,
            str(sorted((params or {}).items())),
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _compute_vbt_indicators(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """计算VBT指标（T+1安全）"""
        factors = {}

        # MA系列
        for window in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 252]:
            try:
                ma = vbt.MA.run(close, window=window, short_name="ma")
                factor_id = f"VBT_MA_{window}"
                min_history = window + 1
                factors[factor_id] = self._apply_t1_shift(
                    ma.ma.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id, min_history, "VBT", "trend", {"window": window}
                )
            except Exception:
                pass

        # EMA系列
        for span in [5, 10, 12, 20, 26, 30, 40, 50, 60, 80, 100, 120]:
            try:
                ema = vbt.MA.run(close, window=span, ewm=True, short_name="ema")
                factor_id = f"VBT_EMA_{span}"
                min_history = span + 1
                factors[factor_id] = self._apply_t1_shift(
                    ema.ma.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id, min_history, "VBT", "trend", {"span": span}
                )
            except Exception:
                pass

        # MACD
        for fast, slow, signal in [(12, 26, 9), (16, 34, 7), (20, 42, 8), (5, 35, 5)]:
            try:
                macd_obj = vbt.MACD.run(
                    close, fast_window=fast, slow_window=slow, signal_window=signal
                )
                min_history = slow + signal + 1

                factor_id = f"VBT_MACD_{fast}_{slow}_{signal}"
                factors[factor_id] = self._apply_t1_shift(
                    macd_obj.macd.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id,
                    min_history,
                    "VBT",
                    "momentum",
                    {"fast": fast, "slow": slow, "signal": signal},
                )

                factor_id = f"VBT_MACD_SIGNAL_{fast}_{slow}_{signal}"
                factors[factor_id] = self._apply_t1_shift(
                    macd_obj.signal.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id,
                    min_history,
                    "VBT",
                    "momentum",
                    {"fast": fast, "slow": slow, "signal": signal},
                )

                factor_id = f"VBT_MACD_HIST_{fast}_{slow}_{signal}"
                factors[factor_id] = self._apply_t1_shift(
                    macd_obj.hist.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id,
                    min_history,
                    "VBT",
                    "momentum",
                    {"fast": fast, "slow": slow, "signal": signal},
                )
            except Exception:
                pass

        # RSI
        for window in [6, 9, 12, 14, 20, 24, 30, 60]:
            try:
                rsi = vbt.RSI.run(close, window=window)
                factor_id = f"VBT_RSI_{window}"
                min_history = window + 1
                factors[factor_id] = self._apply_t1_shift(
                    rsi.rsi.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id, min_history, "VBT", "momentum", {"window": window}
                )
            except Exception:
                pass

        # BBANDS
        for window in [10, 15, 20, 25, 30, 40, 50]:
            for alpha in [1.5, 2.0, 2.5]:
                try:
                    bb = vbt.BBANDS.run(close, window=window, alpha=alpha)
                    min_history = window + 1

                    for name, series in [
                        ("UPPER", bb.upper),
                        ("MIDDLE", bb.middle),
                        ("LOWER", bb.lower),
                        ("WIDTH", bb.bandwidth),
                        ("PERCENT", bb.percent),
                    ]:
                        factor_id = f"VBT_BB_{name}_{window}_{alpha}"
                        factors[factor_id] = self._apply_t1_shift(
                            series.values, factor_id, min_history
                        )
                        self._register_metadata(
                            factor_id,
                            min_history,
                            "VBT",
                            "volatility",
                            {"window": window, "alpha": alpha},
                        )
                except Exception:
                    pass

        # STOCH
        for k_window in [5, 9, 14, 20]:
            for d_window in [3, 5]:
                try:
                    stoch = vbt.STOCH.run(
                        high, low, close, k_window=k_window, d_window=d_window
                    )
                    min_history = k_window + d_window + 1

                    factor_id = f"VBT_STOCH_K_{k_window}_{d_window}"
                    factors[factor_id] = self._apply_t1_shift(
                        stoch.percent_k.values, factor_id, min_history
                    )
                    self._register_metadata(
                        factor_id,
                        min_history,
                        "VBT",
                        "momentum",
                        {"k_window": k_window, "d_window": d_window},
                    )

                    factor_id = f"VBT_STOCH_D_{k_window}_{d_window}"
                    factors[factor_id] = self._apply_t1_shift(
                        stoch.percent_d.values, factor_id, min_history
                    )
                    self._register_metadata(
                        factor_id,
                        min_history,
                        "VBT",
                        "momentum",
                        {"k_window": k_window, "d_window": d_window},
                    )
                except Exception:
                    pass

        # ATR
        for window in [7, 10, 14, 20, 30, 60]:
            try:
                atr = vbt.ATR.run(high, low, close, window=window)
                factor_id = f"VBT_ATR_{window}"
                min_history = window + 1
                factors[factor_id] = self._apply_t1_shift(
                    atr.atr.values, factor_id, min_history
                )
                self._register_metadata(
                    factor_id, min_history, "VBT", "volatility", {"window": window}
                )
            except Exception:
                pass

        # OBV
        try:
            obv = vbt.OBV.run(close, volume)
            factor_id = "VBT_OBV"
            min_history = 2
            factors[factor_id] = self._apply_t1_shift(
                obv.obv.values, factor_id, min_history
            )
            self._register_metadata(factor_id, min_history, "VBT", "volume", {})
        except Exception:
            pass

        logger.info(f"VBT指标: {len(factors)} 个")
        return factors

    def _compute_talib_indicators(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """计算TA-Lib指标（T+1安全）"""
        factors = {}

        # SMA/EMA
        for period in [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 252]:
            factor_id = f"TA_SMA_{period}"
            min_history = period + 1
            factors[factor_id] = self._apply_t1_shift(
                talib.SMA(close, timeperiod=period), factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "TA-Lib", "overlap", {"period": period}
            )

            factor_id = f"TA_EMA_{period}"
            factors[factor_id] = self._apply_t1_shift(
                talib.EMA(close, timeperiod=period), factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "TA-Lib", "overlap", {"period": period}
            )

        # MACD
        for fast, slow, signal in [(12, 26, 9), (16, 34, 7), (20, 42, 8), (5, 35, 5)]:
            macd, signal_line, hist = talib.MACD(
                close, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            min_history = slow + signal + 1

            factor_id = f"TA_MACD_{fast}_{slow}_{signal}"
            factors[factor_id] = self._apply_t1_shift(macd, factor_id, min_history)
            self._register_metadata(
                factor_id,
                min_history,
                "TA-Lib",
                "momentum",
                {"fast": fast, "slow": slow, "signal": signal},
            )

            factor_id = f"TA_MACD_SIGNAL_{fast}_{slow}_{signal}"
            factors[factor_id] = self._apply_t1_shift(
                signal_line, factor_id, min_history
            )
            self._register_metadata(
                factor_id,
                min_history,
                "TA-Lib",
                "momentum",
                {"fast": fast, "slow": slow, "signal": signal},
            )

            factor_id = f"TA_MACD_HIST_{fast}_{slow}_{signal}"
            factors[factor_id] = self._apply_t1_shift(hist, factor_id, min_history)
            self._register_metadata(
                factor_id,
                min_history,
                "TA-Lib",
                "momentum",
                {"fast": fast, "slow": slow, "signal": signal},
            )

        # RSI
        for period in [6, 9, 12, 14, 20, 24, 30, 60]:
            factor_id = f"TA_RSI_{period}"
            min_history = period + 1
            factors[factor_id] = self._apply_t1_shift(
                talib.RSI(close, timeperiod=period), factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "TA-Lib", "momentum", {"period": period}
            )

        # ATR
        for period in [7, 10, 14, 20, 30, 60]:
            factor_id = f"TA_ATR_{period}"
            min_history = period + 1
            factors[factor_id] = self._apply_t1_shift(
                talib.ATR(high, low, close, timeperiod=period), factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "TA-Lib", "volatility", {"period": period}
            )

        # BBANDS
        for period in [10, 15, 20, 25, 30, 40, 50]:
            for nbdev in [1.5, 2.0, 2.5]:
                upper, middle, lower = talib.BBANDS(
                    close, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev
                )
                min_history = period + 1

                factor_id = f"TA_BB_UPPER_{period}_{nbdev}"
                factors[factor_id] = self._apply_t1_shift(upper, factor_id, min_history)
                self._register_metadata(
                    factor_id,
                    min_history,
                    "TA-Lib",
                    "volatility",
                    {"period": period, "nbdev": nbdev},
                )

                factor_id = f"TA_BB_MIDDLE_{period}_{nbdev}"
                factors[factor_id] = self._apply_t1_shift(
                    middle, factor_id, min_history
                )
                self._register_metadata(
                    factor_id,
                    min_history,
                    "TA-Lib",
                    "volatility",
                    {"period": period, "nbdev": nbdev},
                )

                factor_id = f"TA_BB_LOWER_{period}_{nbdev}"
                factors[factor_id] = self._apply_t1_shift(lower, factor_id, min_history)
                self._register_metadata(
                    factor_id,
                    min_history,
                    "TA-Lib",
                    "volatility",
                    {"period": period, "nbdev": nbdev},
                )

        # OBV
        factor_id = "TA_OBV"
        min_history = 2
        factors[factor_id] = self._apply_t1_shift(
            talib.OBV(close, volume), factor_id, min_history
        )
        self._register_metadata(factor_id, min_history, "TA-Lib", "volume", {})

        logger.info(f"TA-Lib指标: {len(factors)} 个")
        return factors

    def _compute_custom_indicators(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """计算自定义指标（T+1安全）"""
        factors = {}

        # 收益率系列（严格使用过去收益，避免未来泄露）
        for period in [1, 2, 3, 5, 10, 20, 30, 60]:
            factor_id = f"RETURN_{period}"
            min_history = period + 1
            # 使用过去period收益：close / close.shift(period) - 1
            # 显式移位 + 首部 NaN（避免 np.roll 环回）
            past = np.concatenate([np.full(period, np.nan), close[:-period]])
            ret = close / past - 1
            factors[factor_id] = self._apply_t1_shift(ret, factor_id, min_history)
            self._register_metadata(
                factor_id,
                min_history,
                "Custom",
                "return",
                {"period": period, "lookback": True},
            )

        # 波动率系列
        for window in [5, 10, 20, 30, 60]:
            factor_id = f"VOLATILITY_{window}"
            min_history = window + 1
            log_returns = pd.Series(close).pct_change()
            vol = log_returns.rolling(window).std()
            factors[factor_id] = self._apply_t1_shift(
                vol.values, factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "Custom", "volatility", {"window": window}
            )

        # 价格位置
        for window in [10, 20, 30, 60]:
            factor_id = f"PRICE_POSITION_{window}"
            min_history = window + 1
            rolling_high = pd.Series(high).rolling(window).max()
            rolling_low = pd.Series(low).rolling(window).min()
            pos = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
            factors[factor_id] = self._apply_t1_shift(
                pos.values, factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "Custom", "position", {"window": window}
            )

        # 成交量比率
        for window in [5, 10, 20, 30]:
            factor_id = f"VOLUME_RATIO_{window}"
            min_history = window + 1
            vol_ma = pd.Series(volume).rolling(window).mean()
            ratio = volume / (vol_ma + 1e-10)
            factors[factor_id] = self._apply_t1_shift(
                ratio.values, factor_id, min_history
            )
            self._register_metadata(
                factor_id, min_history, "Custom", "volume", {"window": window}
            )

        # 动量指标
        for window in [5, 10, 20, 30]:
            factor_id = f"MOMENTUM_{window}"
            min_history = window + 1
            # 显式移位 + 首部 NaN（避免 np.roll 环回）
            past = np.concatenate([np.full(window, np.nan), close[:-window]])
            mom = close / past - 1
            factors[factor_id] = self._apply_t1_shift(mom, factor_id, min_history)
            self._register_metadata(
                factor_id, min_history, "Custom", "momentum", {"window": window}
            )

        logger.info(f"自定义指标: {len(factors)} 个")
        return factors
