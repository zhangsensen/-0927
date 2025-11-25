"""
Numba加速IC计算器 | Numba-Accelerated IC Calculator

使用Numba JIT编译加速IC计算，性能提升10-50倍

作者: Linus Performance
日期: 2025-10-28
"""

import logging

import numba
import numpy as np

logger = logging.getLogger(__name__)


@numba.jit(nopython=True, parallel=False, cache=True)
def compute_spearman_ic_numba(signals: np.ndarray, returns: np.ndarray) -> float:
    """
    Numba加速的Spearman IC计算（单个信号）

    Args:
        signals: (T, N) 信号矩阵
        returns: (T, N) 收益矩阵

    Returns:
        平均IC
    """
    T, N = signals.shape
    ic_sum = 0.0
    valid_days = 0

    for t in range(T):
        signal_t = signals[t, :]
        return_t = returns[t, :]

        # 去除NaN
        mask = ~(np.isnan(signal_t) | np.isnan(return_t))
        n_valid = np.sum(mask)

        # [P0修复] 提高样本数阈值: 2→30,过滤噪声IC
        if n_valid >= 30:
            s = signal_t[mask]
            r = return_t[mask]

            # 计算秩
            s_rank = np.argsort(np.argsort(s)).astype(np.float64)
            r_rank = np.argsort(np.argsort(r)).astype(np.float64)

            # Spearman相关系数
            s_mean = np.mean(s_rank)
            r_mean = np.mean(r_rank)

            numerator = np.sum((s_rank - s_mean) * (r_rank - r_mean))
            s_std = np.sqrt(np.sum((s_rank - s_mean) ** 2))
            r_std = np.sqrt(np.sum((r_rank - r_mean) ** 2))

            if s_std > 0 and r_std > 0:
                ic = numerator / (s_std * r_std)
                if not np.isnan(ic):
                    ic_sum += ic
                    valid_days += 1

    if valid_days > 0:
        return ic_sum / valid_days
    return 0.0


@numba.jit(nopython=True, parallel=False, cache=True)
def compute_multiple_ics_numba(
    all_signals: np.ndarray, returns: np.ndarray
) -> np.ndarray:
    """
    Numba加速的批量IC计算（多个信号）

    Args:
        all_signals: (n_combos, T, N) 多个信号
        returns: (T, N) 收益矩阵

    Returns:
        (n_combos,) IC数组
    """
    n_combos = all_signals.shape[0]
    ics = np.zeros(n_combos)

    for i in numba.prange(n_combos):  # 并行循环
        ics[i] = compute_spearman_ic_numba(all_signals[i], returns)

    return ics


class ICCalculatorNumba:
    """Numba加速的IC计算器"""

    @staticmethod
    def compute_ic(signals: np.ndarray, returns: np.ndarray) -> float:
        """
        计算单个信号的IC

        Args:
            signals: (T, N) 信号矩阵
            returns: (T, N) 收益矩阵

        Returns:
            平均IC
        """
        return compute_spearman_ic_numba(signals, returns)

    @staticmethod
    def compute_batch_ics(all_signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """
        批量计算多个信号的IC

        Args:
            all_signals: (n_combos, T, N) 多个信号
            returns: (T, N) 收益矩阵

        Returns:
            (n_combos,) IC数组
        """
        return compute_multiple_ics_numba(all_signals, returns)
