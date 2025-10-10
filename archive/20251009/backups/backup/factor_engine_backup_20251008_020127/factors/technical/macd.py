"""MACD - 移动平均收敛散度指标"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor
from factor_system.shared.factor_calculators import SHARED_CALCULATORS


class MACD(BaseFactor):
    """
    MACD - Moving Average Convergence Divergence

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    计算公式:
    1. EMA_fast = EMA(close, fast_period)
    2. EMA_slow = EMA(close, slow_period)
    3. MACD = EMA_fast - EMA_slow
    4. Signal = EMA(MACD, signal_period)
    5. Histogram = MACD - Signal

    标准参数: (12, 26, 9)
    """

    factor_id = "MACD"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "MACD线（快速EMA - 慢速EMA）"

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        """
        初始化MACD

        Args:
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线EMA周期，默认9
        """
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MACD线 - 使用共享计算器确保全系统一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            MACD values
        """
        close = data["close"]

        # 使用共享计算器计算MACD，确保与factor_generation、hk_midfreq完全一致
        macd_result = SHARED_CALCULATORS.calculate_macd(
            close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period,
        )

        return macd_result["macd"]


class MACDSignal(BaseFactor):
    """
    MACD Signal Line - MACD信号线

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致
    对MACD线进行EMA平滑得到信号线
    """

    factor_id = "MACD_SIGNAL"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "MACD信号线（MACD的EMA）"

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MACD信号线 - 使用共享计算器确保全系统一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            Signal line values
        """
        close = data["close"]

        # 使用共享计算器计算MACD信号线，确保与factor_generation、hk_midfreq完全一致
        macd_result = SHARED_CALCULATORS.calculate_macd(
            close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period,
        )

        return macd_result["signal"]


class MACDHistogram(BaseFactor):
    """
    MACD Histogram - MACD柱状图

    基于VectorBT实现，确保与factor_generation计算逻辑完全一致

    MACD柱 = MACD线 - 信号线
    用于判断MACD的强弱和金叉/死叉
    """

    factor_id = "MACD_HIST"
    version = "v3.0"  # 升级版本，基于共享计算器
    category = "technical"
    description = "MACD柱状图（MACD - Signal）"

    def __init__(
        self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MACD柱状图 - 使用共享计算器确保全系统一致

        Args:
            data: DataFrame with 'close' column

        Returns:
            Histogram values (positive = bullish, negative = bearish)
        """
        close = data["close"]

        # 使用共享计算器计算MACD柱状图，确保与factor_generation、hk_midfreq完全一致
        macd_result = SHARED_CALCULATORS.calculate_macd(
            close,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period,
        )

        return macd_result["hist"]
