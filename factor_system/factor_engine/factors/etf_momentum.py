"""ETF轮动专用长周期动量因子

所有因子严格遵循T+1安全原则：
- 在T时刻，只能使用T-1及之前的数据
- 价格口径：优先adj_close，回退close
- 最小样本约束：不足则返回NaN
"""

from __future__ import annotations

import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor


class Momentum63(BaseFactor):
    """63日动量（3个月收益率）"""

    factor_id = "Momentum63"
    category = "momentum"
    description = "63日价格动量（3个月收益率）"
    min_history = 64  # 需要至少64个交易日（63 + 1）

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算63日动量，T+1安全"""
        # 最小样本检查
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        # 价格口径：优先adj_close，回退close
        price_col = "adj_close" if "adj_close" in data.columns else "close"

        # 正确的动量计算：避免未来函数
        # 在T时间，计算[T-63, T-1]区间的收益率
        close_t_minus_1 = data[price_col].shift(1)  # 昨天收盘价
        close_63_days_ago = data[price_col].shift(64)  # 64天前收盘价（T-1时刻的63天前）
        return (close_t_minus_1 - close_63_days_ago) / close_63_days_ago


class Momentum126(BaseFactor):
    """126日动量（6个月收益率）"""

    factor_id = "Momentum126"
    category = "momentum"
    description = "126日价格动量（6个月收益率）"
    min_history = 127  # 需要至少127个交易日

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算126日动量，T+1安全"""
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        price_col = "adj_close" if "adj_close" in data.columns else "close"
        close_t_minus_1 = data[price_col].shift(1)
        close_126_days_ago = data[price_col].shift(127)
        return (close_t_minus_1 - close_126_days_ago) / close_126_days_ago


class Momentum252(BaseFactor):
    """252日动量（12个月收益率）"""

    factor_id = "Momentum252"
    category = "momentum"
    description = "252日价格动量（12个月收益率）"
    min_history = 253  # 需要至少253个交易日

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算252日动量，T+1安全"""
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        price_col = "adj_close" if "adj_close" in data.columns else "close"
        close_t_minus_1 = data[price_col].shift(1)
        close_252_days_ago = data[price_col].shift(253)
        return (close_t_minus_1 - close_252_days_ago) / close_252_days_ago


class VOLATILITY_120D(BaseFactor):
    """120日波动率"""

    factor_id = "VOLATILITY_120D"
    category = "volatility"
    description = "120日收益率标准差（年化）"
    min_history = 121  # 需要至少121个交易日

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算120日波动率，T+1安全"""
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        price_col = "adj_close" if "adj_close" in data.columns else "close"
        close_t_minus_1 = data[price_col].shift(1)
        returns = close_t_minus_1.pct_change()
        volatility = returns.rolling(window=120).std()
        return volatility * (252**0.5)


class MOM_ACCEL(BaseFactor):
    """动量加速度（短期动量 - 长期动量）"""

    factor_id = "MOM_ACCEL"
    category = "momentum"
    description = "动量加速度（63日动量 - 252日动量）"
    min_history = 253  # 需要至少253个交易日（取决于长期动量）

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算动量加速度，T+1安全"""
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        price_col = "adj_close" if "adj_close" in data.columns else "close"
        close_t_minus_1 = data[price_col].shift(1)
        close_63_days_ago = data[price_col].shift(64)
        close_252_days_ago = data[price_col].shift(253)

        mom_short = (close_t_minus_1 - close_63_days_ago) / close_63_days_ago
        mom_long = (close_t_minus_1 - close_252_days_ago) / close_252_days_ago
        return mom_short - mom_long


class DRAWDOWN_63D(BaseFactor):
    """63日最大回撤"""

    factor_id = "DRAWDOWN_63D"
    category = "risk"
    description = "63日最大回撤（负值）"
    min_history = 64  # 需要至少64个交易日

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """计算63日最大回撤，T+1安全"""
        if len(data) < self.min_history:
            return pd.Series(index=data.index, dtype=float)

        price_col = "adj_close" if "adj_close" in data.columns else "close"
        close_t_minus_1 = data[price_col].shift(1)
        rolling_max = close_t_minus_1.rolling(window=63).max()
        drawdown = (close_t_minus_1 - rolling_max) / rolling_max
        return drawdown
