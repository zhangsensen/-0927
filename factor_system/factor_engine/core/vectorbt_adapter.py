"""
VectorBT计算适配器 - 精简版本

只包含factor_generation中实际存在的因子：
- RSI
- MACD
- STOCH
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt

    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False
    vbt = None

logger = logging.getLogger(__name__)


def ensure_series(values, index, name: Optional[str] = None) -> pd.Series:
    """确保返回值为Series"""
    if isinstance(values, pd.Series):
        series = values.copy()
        if name is not None:
            series.name = name
        return series
    elif isinstance(values, (np.ndarray, list)):
        try:
            series = pd.Series(values, index=index, name=name)
            return series
        except Exception:
            pass

    return pd.Series(values, index=index, name=name)


class VectorBTAdapter:
    """
    VectorBT计算适配器 - 精简版本

    只支持factor_generation中存在的因子
    """

    def __init__(self):
        """初始化适配器"""
        self._check_vectorbt_availability()

    def _check_vectorbt_availability(self):
        """检查VectorBT可用性"""
        if not HAS_VECTORBT:
            raise ImportError("VectorBT未安装")

        # 检查必要的指标
        if not hasattr(vbt, "RSI"):
            raise ImportError("VectorBT不完整，缺少RSI支持")

        # 检查TA-Lib支持
        if hasattr(vbt, "talib"):
            logger.info("VectorBT TA-Lib支持可用")
        else:
            logger.warning("VectorBT TA-Lib支持不可用，将使用内置指标")

    # 只支持factor_generation中存在的因子
    def calculate_rsi(self, price: pd.Series, timeperiod: int = 14) -> pd.Series:
        """计算RSI"""
        try:
            # 使用TA-Lib确保与共享计算器一致
            import talib

            result = talib.RSI(price, timeperiod=timeperiod)
            return ensure_series(result, price.index, "RSI")
        except Exception as e:
            logger.error(f"RSI计算失败: {e}")
            return pd.Series(index=price.index, name="RSI", dtype=float)

    def calculate_stoch(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> pd.Series:
        """计算STOCH - 返回%K值"""
        try:
            # 使用TA-Lib确保与共享计算器一致
            import talib

            result = talib.STOCH(
                high,
                low,
                close,
                fastk_period=fastk_period,
                slowk_period=slowk_period,
                slowd_period=slowd_period,
            )
            # 返回slowk值（与共享计算器一致）
            return ensure_series(result[0], close.index, "STOCH")
        except Exception as e:
            logger.error(f"STOCH计算失败: {e}")
            return pd.Series(index=close.index, name="STOCH", dtype=float)

    def calculate_macd(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.Series:
        """计算MACD线"""
        try:
            # 使用TA-Lib确保与共享计算器一致
            import talib

            result = talib.MACD(
                close,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )
            return ensure_series(result[0], close.index, "MACD")
        except Exception as e:
            logger.error(f"MACD计算失败: {e}")
            return pd.Series(index=close.index, name="MACD", dtype=float)

    def calculate_macd_signal(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.Series:
        """计算MACD信号线"""
        try:
            # 使用TA-Lib确保与共享计算器一致
            import talib

            result = talib.MACD(
                close,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )
            return ensure_series(result[1], close.index, "MACD_SIGNAL")
        except Exception as e:
            logger.error(f"MACD_SIGNAL计算失败: {e}")
            return pd.Series(index=close.index, name="MACD_SIGNAL", dtype=float)

    def calculate_macd_histogram(
        self,
        close: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.Series:
        """计算MACD柱状图"""
        try:
            # 使用TA-Lib确保与共享计算器一致
            import talib

            result = talib.MACD(
                close,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )
            return ensure_series(result[2], close.index, "MACD_HIST")
        except Exception as e:
            logger.error(f"MACD_HIST计算失败: {e}")
            return pd.Series(index=close.index, name="MACD_HIST", dtype=float)


# 全局实例
_adapter_instance = None


def get_vectorbt_adapter() -> VectorBTAdapter:
    """获取VectorBT适配器实例"""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = VectorBTAdapter()
    return _adapter_instance


# 便捷函数
def calculate_rsi(price: pd.Series, timeperiod: int = 14) -> pd.Series:
    """RSI计算便捷函数"""
    adapter = get_vectorbt_adapter()
    return adapter.calculate_rsi(price, timeperiod=timeperiod)


def calculate_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> pd.Series:
    """STOCH计算便捷函数"""
    adapter = get_vectorbt_adapter()
    return adapter.calculate_stoch(
        high, low, close, fastk_period, slowk_period, slowd_period
    )


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD计算便捷函数"""
    adapter = get_vectorbt_adapter()
    return adapter.calculate_macd(close, fast_period, slow_period, signal_period)


def calculate_macd_signal(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD信号线计算便捷函数"""
    adapter = get_vectorbt_adapter()
    return adapter.calculate_macd_signal(close, fast_period, slow_period, signal_period)


def calculate_macd_histogram(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """MACD柱状图计算便捷函数"""
    adapter = get_vectorbt_adapter()
    return adapter.calculate_macd_histogram(
        close, fast_period, slow_period, signal_period
    )
