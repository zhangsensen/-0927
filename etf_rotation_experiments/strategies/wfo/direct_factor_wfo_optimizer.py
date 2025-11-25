"""简化版直接因子级WFO优化器。

该实现去除了历史遗留的多阶段组合搜索逻辑，专注于：
1. 生成滑动窗口
2. 计算每个窗口的因子IC
3. 依据阈值选择因子并分配权重
4. 评估OOS期的IC与IR

整个流程以NumPy/Numba为核心，避免Python层的重循环。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import njit

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from core.ic_calculator_numba import compute_spearman_ic_numba

logger = logging.getLogger(__name__)


@njit(cache=True)
def _compute_factor_ics_windows(factors: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """计算每个窗口、每个因子的IC。"""
    n_windows, t_steps, n_assets, n_factors = factors.shape
    out = np.zeros((n_windows, n_factors))

    for w in range(n_windows):
        ret_window = returns[w]
        for f in range(n_factors):
            signal = factors[w, :, :, f]
            out[w, f] = compute_spearman_ic_numba(signal, ret_window)

    return out


@njit(cache=True)
def _compute_signal_metrics(
    signals: np.ndarray, returns: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为每个窗口计算平均IC、IR以及正IC占比。"""
    n_windows, t_steps, n_assets = signals.shape
    mean_ic = np.zeros(n_windows)
    ir = np.zeros(n_windows)
    positive_rate = np.zeros(n_windows)

    for w in range(n_windows):
        signal_window = signals[w]
        return_window = returns[w]

        ic_buffer = np.empty(t_steps)
        valid_days = 0

        for t in range(t_steps):
            signal_t = signal_window[t]
            return_t = return_window[t]

            mask = ~(np.isnan(signal_t) | np.isnan(return_t))
            if np.sum(mask) <= 2:
                continue

            sig = signal_t[mask]
            ret = return_t[mask]

            sig_rank = np.argsort(np.argsort(sig)).astype(np.float64)
            ret_rank = np.argsort(np.argsort(ret)).astype(np.float64)

            sig_mean = np.mean(sig_rank)
            ret_mean = np.mean(ret_rank)

            sig_centered = sig_rank - sig_mean
            ret_centered = ret_rank - ret_mean

            sig_std = np.sqrt(np.sum(sig_centered**2))
            ret_std = np.sqrt(np.sum(ret_centered**2))

            if sig_std <= 0.0 or ret_std <= 0.0:
                continue

            ic_value = np.sum(sig_centered * ret_centered) / (sig_std * ret_std)
            if not np.isnan(ic_value):
                ic_buffer[valid_days] = ic_value
                valid_days += 1

        if valid_days == 0:
            mean_ic[w] = 0.0
            ir[w] = 0.0
            positive_rate[w] = 0.0
        else:
            daily_ic = ic_buffer[:valid_days]
            ic_mean = np.mean(daily_ic)
            ic_std = np.std(daily_ic)

            mean_ic[w] = ic_mean
            ir[w] = ic_mean / ic_std if ic_std > 1e-12 else 0.0
            positive_rate[w] = np.sum(daily_ic > 0.0) / valid_days

    return mean_ic, ir, positive_rate


@dataclass
class DirectFactorWindowResult:
    """单个WFO窗口的统计结果。"""

    window_index: int
    is_start: int
    is_end: int
    oos_start: int
    oos_end: int
    selected_factors: List[str]
    factor_weights: Dict[str, float]
    is_ic_scores: Dict[str, float]
    oos_factor_ics: Dict[str, float]
    oos_ic: float
    oos_ir: float
    positive_rate: float
    baseline_ic: float
    baseline_ir: float


class DirectFactorWFOOptimizer:
    """直接因子级WFO优化器（精简版）。"""

    def __init__(
        self,
        factor_weighting: str = "ic_weighted",
        min_factor_ic: float = 0.01,
        ic_floor: float = 0.0,
        verbose: bool = True,
    ) -> None:
        if factor_weighting not in {"equal", "ic_weighted"}:
            raise ValueError("factor_weighting 仅支持 'equal' 或 'ic_weighted'")

        self.factor_weighting = factor_weighting
        self.min_factor_ic = float(min_factor_ic)
        self.ic_floor = float(ic_floor)
        self.verbose = verbose

    def run_wfo(
        self,
        factors_data: np.ndarray,
        returns: np.ndarray,
        factor_names: List[str],
        is_period: int,
        oos_period: int,
        step_size: int,
    ) -> Tuple[List[DirectFactorWindowResult], pd.DataFrame]:
        """执行Walk-Forward优化。"""
        if factors_data.ndim != 3:
            raise ValueError("factors_data 需为 (T, N, F) 结构")
        if returns.ndim != 2:
            raise ValueError("returns 需为 (T, N) 结构")
        if factors_data.shape[0] != returns.shape[0]:
            raise ValueError("因子与收益在时间维度上未对齐")
        if factors_data.shape[2] != len(factor_names):
            raise ValueError("因子名称数量与数据维度不一致")

        total_length = factors_data.shape[0]
        window_length = is_period + oos_period
        if window_length + step_size > total_length + step_size:
            raise ValueError("样本长度不足以构建任何窗口")

        starts = np.arange(0, total_length - window_length + 1, step_size, dtype=int)
        n_windows = starts.size
        n_assets = factors_data.shape[1]
        n_factors = factors_data.shape[2]

        if self.verbose:
            logger.info(
                "WFO窗口: %d 个 | IS=%d | OOS=%d | 步长=%d",
                n_windows,
                is_period,
                oos_period,
                step_size,
            )

        # 生成滑动窗口数据 (W, window_length, N, F)
        index_matrix = starts[:, None] + np.arange(window_length)[None, :]
        window_factors = factors_data[index_matrix]
        window_returns = returns[index_matrix]

        # 切分 IS/OOS 并做 T-1 对齐
        is_factors = window_factors[:, :is_period]
        oos_factors = window_factors[:, is_period:]
        is_returns = window_returns[:, :is_period]
        oos_returns = window_returns[:, is_period:]

        is_factors_aligned = is_factors[:, :-1]
        is_returns_aligned = is_returns[:, 1:]
        oos_factors_aligned = oos_factors[:, :-1]
        oos_returns_aligned = oos_returns[:, 1:]

        # 计算 IS/OOS 因子 IC
        is_ic_matrix = _compute_factor_ics_windows(
            is_factors_aligned, is_returns_aligned
        )
        oos_ic_matrix = _compute_factor_ics_windows(
            oos_factors_aligned, oos_returns_aligned
        )

        # 依据阈值选择因子并生成权重
        final_masks = np.zeros((n_windows, n_factors), dtype=np.bool_)
        weights = np.zeros((n_windows, n_factors), dtype=np.float64)

        for w in range(n_windows):
            mask = is_ic_matrix[w] > self.min_factor_ic
            if not np.any(mask):
                mask[np.argmax(is_ic_matrix[w])] = True
            final_masks[w] = mask

            if self.factor_weighting == "equal":
                weights[w, mask] = 1.0 / np.sum(mask)
            else:
                raw = np.maximum(is_ic_matrix[w, mask], self.ic_floor)
                total = np.sum(raw)
                if total <= 1e-12:
                    weights[w, mask] = 1.0 / np.sum(mask)
                else:
                    weights[w, mask] = raw / total

        # 计算加权信号
        weighted_signals = np.einsum(
            "wtnf,wf->wtn", oos_factors_aligned, weights, optimize=True
        )

        # 计算 OOS 指标
        oos_mean_ic, oos_ir, positive_rate = _compute_signal_metrics(
            weighted_signals, oos_returns_aligned
        )

        baseline_signals = np.ones(
            (n_windows, oos_returns_aligned.shape[1], n_assets), dtype=np.float64
        )
        baseline_ic, baseline_ir, _ = _compute_signal_metrics(
            baseline_signals, oos_returns_aligned
        )

        results: List[DirectFactorWindowResult] = []
        for idx, start in enumerate(starts):
            is_start = int(start)
            is_end = int(start + is_period)
            oos_start = is_end
            oos_end = int(oos_start + oos_period)

            selected_indices = np.where(final_masks[idx])[0]
            selected_names = [factor_names[i] for i in selected_indices]

            window_result = DirectFactorWindowResult(
                window_index=int(idx),
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
                selected_factors=selected_names,
                factor_weights={
                    factor_names[i]: float(weights[idx, i]) for i in selected_indices
                },
                is_ic_scores={
                    factor_names[i]: float(is_ic_matrix[idx, i])
                    for i in range(n_factors)
                },
                oos_factor_ics={
                    factor_names[i]: float(oos_ic_matrix[idx, i])
                    for i in range(n_factors)
                },
                oos_ic=float(oos_mean_ic[idx]),
                oos_ir=float(oos_ir[idx]),
                positive_rate=float(positive_rate[idx]),
                baseline_ic=float(baseline_ic[idx]),
                baseline_ir=float(baseline_ir[idx]),
            )
            results.append(window_result)

        summary_df = self._summarize(results)

        if self.verbose and not summary_df.empty:
            logger.info(
                "平均OOS IC=%.4f | 平均IR=%.3f | 平均正IC率=%.1f%%",
                summary_df["oos_ic"].mean(),
                summary_df["oos_ir"].mean(),
                summary_df["positive_rate"].mean() * 100.0,
            )

        return results, summary_df

    @staticmethod
    def _summarize(results: List[DirectFactorWindowResult]) -> pd.DataFrame:
        records = []
        for r in results:
            records.append(
                {
                    "window_index": r.window_index,
                    "is_start": r.is_start,
                    "is_end": r.is_end,
                    "oos_start": r.oos_start,
                    "oos_end": r.oos_end,
                    "n_selected_factors": len(r.selected_factors),
                    "selected_factors": ",".join(r.selected_factors),
                    "oos_ic": r.oos_ic,
                    "oos_ir": r.oos_ir,
                    "positive_rate": r.positive_rate,
                    "baseline_ic": r.baseline_ic,
                    "baseline_ir": r.baseline_ir,
                }
            )

        return pd.DataFrame.from_records(records)
