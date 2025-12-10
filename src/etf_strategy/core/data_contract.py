"""
数据契约 | Data Contract

统一的数据质量标准和验证规则
- 快速失败，不静默
- 明确的NaN阈值
- 严格的schema验证

作者: Linus Fix
日期: 2025-10-28
"""

import numpy as np
import pandas as pd


class DataContract:
    """数据质量契约"""

    # 质量阈值
    # 注：46 个 ETF 上市时间不同，早期 NaN 率较高是正常的
    # 2020-01-01 至 2025-12-08 期间，NaN 率约 15%（部分 ETF 在 2021-2023 年上市）
    MAX_NAN_RATIO_OHLCV = 0.20  # OHLCV 最大 NaN 率 20%（放宽以适配不同上市时间）
    MAX_NAN_RATIO_FACTOR = 0.3  # 因子最大NaN率30%
    MIN_TRADING_DAYS = 100  # 最小交易日数

    @staticmethod
    def validate_ohlcv(ohlcv: dict) -> None:
        """
        验证OHLCV数据

        Args:
            ohlcv: {'open': DataFrame, 'high': DataFrame, ...}

        Raises:
            ValueError: 数据不符合契约
        """
        required_cols = ["open", "high", "low", "close", "volume"]

        for col in required_cols:
            if col not in ohlcv:
                raise ValueError(f"缺失必需列: {col}")

            df = ohlcv[col]

            # 检查形状
            if df.empty:
                raise ValueError(f"{col} 数据为空")

            # 检查NaN率
            nan_ratio = df.isna().sum().sum() / df.size
            if nan_ratio > DataContract.MAX_NAN_RATIO_OHLCV:
                raise ValueError(
                    f"{col} NaN率={nan_ratio:.1%} 超过{DataContract.MAX_NAN_RATIO_OHLCV:.0%}阈值"
                )

            # 检查交易日数
            if len(df) < DataContract.MIN_TRADING_DAYS:
                raise ValueError(
                    f"{col} 交易日数={len(df)} 少于最小要求{DataContract.MIN_TRADING_DAYS}"
                )

    @staticmethod
    def validate_factor(
        factor: pd.DataFrame, name: str, max_nan_ratio: float = None
    ) -> None:
        """
        验证单个因子

        Args:
            factor: 因子DataFrame (日期 × 标的)
            name: 因子名称
            max_nan_ratio: 最大NaN率（None使用默认值）

        Raises:
            ValueError: 因子不符合契约
        """
        if max_nan_ratio is None:
            max_nan_ratio = DataContract.MAX_NAN_RATIO_FACTOR

        if factor.empty:
            raise ValueError(f"因子{name}数据为空")

        nan_ratio = factor.isna().sum().sum() / factor.size

        if nan_ratio > max_nan_ratio:
            raise ValueError(
                f"因子{name} NaN率={nan_ratio:.1%} 超过{max_nan_ratio:.0%}阈值"
            )

    @staticmethod
    def validate_alignment(factors: np.ndarray, returns: np.ndarray) -> None:
        """
        验证因子和收益对齐

        Args:
            factors: (T, N, K) 因子矩阵
            returns: (T, N) 收益矩阵

        Raises:
            ValueError: 对齐不正确
        """
        if factors.shape[0] != returns.shape[0]:
            raise ValueError(
                f"因子时间维度({factors.shape[0]}) != 收益时间维度({returns.shape[0]})"
            )

        if factors.shape[1] != returns.shape[1]:
            raise ValueError(
                f"因子标的维度({factors.shape[1]}) != 收益标的维度({returns.shape[1]})"
            )

        if factors.shape[0] < 2:
            raise ValueError(f"数据长度({factors.shape[0]}) < 2，无法进行T-1对齐")


def align_factor_to_return(
    factors: np.ndarray, returns: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    因子T-1 → 收益T对齐

    Args:
        factors: (T, N, K) 因子矩阵
        returns: (T, N) 收益矩阵

    Returns:
        (factors_aligned, returns_aligned)
        - factors_aligned: (T-1, N, K) 因子[0:T-1]
        - returns_aligned: (T-1, N) 收益[1:T]

    Raises:
        ValueError: 数据不符合对齐要求
    """
    # 验证输入
    DataContract.validate_alignment(factors, returns)

    # 对齐：因子[t] 预测 收益[t+1]
    factors_aligned = factors[:-1]  # [0, T-1)
    returns_aligned = returns[1:]  # [1, T)

    return factors_aligned, returns_aligned
