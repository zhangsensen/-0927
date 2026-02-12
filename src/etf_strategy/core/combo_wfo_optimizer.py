"""Combo-level WFO Optimizer."""

from __future__ import annotations

import logging
import os
import time as _time
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .hysteresis import apply_hysteresis
from .ic_calculator_numba import compute_multiple_ics_numba, compute_spearman_ic_numba
from .wfo_realbt_calibrator import WFORealBacktestCalibrator

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _get_optimal_n_jobs() -> int:
    """获取最优并行核心数 (单CCD内8核共享L3，避免跨CCD延迟)

    Also caps NUMBA_NUM_THREADS to avoid oversubscription when
    Numba parallel=True kernels (ic_calculator_numba) run inside
    joblib workers.
    """
    env_jobs = os.getenv("JOBLIB_N_JOBS")
    if env_jobs:
        n_jobs = int(env_jobs)
    else:
        # 9950X 双CCD: 单CCD 8核共享32MB L3, 跨CCD有额外延迟
        cpu_count = os.cpu_count() or 8
        physical_cores = cpu_count // 2
        n_jobs = min(physical_cores, 8)

    # Prevent Numba thread oversubscription: n_jobs workers each spawn
    # NUMBA_NUM_THREADS threads for parallel=True kernels.
    # Cap so total threads <= logical CPU count.
    if not os.getenv("NUMBA_NUM_THREADS"):
        cpu_count = os.cpu_count() or 8
        numba_threads = max(1, cpu_count // n_jobs)
        os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)

    return n_jobs


def warmup_numba_kernels() -> None:
    """用微型 dummy 数据预热所有 @njit 函数的 JIT 编译 (仅首次运行受益)"""
    t0 = _time.time()
    T, N, F = 10, 5, 3
    dummy_factors = np.random.randn(T, N, F).astype(np.float64)
    dummy_signal = np.random.randn(T, N).astype(np.float64)
    dummy_returns = np.random.randn(T, N).astype(np.float64)
    dummy_weights = np.ones(F, dtype=np.float64) / F
    dummy_exposures = np.ones(T, dtype=np.float64)

    _compute_combo_signal(dummy_factors, dummy_weights)
    _compute_rebalanced_ic(dummy_signal, dummy_returns, 3)
    dummy_cost_arr = np.full(N, 0.0002, dtype=np.float64)
    _compute_rebalanced_return(
        dummy_signal, dummy_returns, dummy_exposures, 3, 2, dummy_cost_arr
    )
    _compute_rebalanced_return_stats(
        dummy_signal, dummy_returns, dummy_exposures, 3, 2, dummy_cost_arr,
        False, 0.0, 0, 0,
    )
    compute_spearman_ic_numba(dummy_signal, dummy_returns)
    compute_multiple_ics_numba(dummy_factors.transpose(2, 0, 1), dummy_returns)

    elapsed = _time.time() - t0
    logger.info(f"Numba warmup completed in {elapsed:.1f}s")


@dataclass
class ComboWFOConfig:
    combo_sizes: List[int]
    is_period: int
    oos_period: int
    step_size: int
    n_jobs: int = -1  # -1 表示自动检测
    verbose: int = 1
    enable_fdr: bool = True
    fdr_alpha: float = 0.05
    complexity_penalty_lambda: float = 0.01
    delta_rank: float = 0.0  # Exp4: hysteresis rank01 gap threshold (0 = disabled)
    min_hold_days: int = 0  # Exp4: minimum holding period (0 = disabled)
    use_bucket_constraints: bool = False  # 跨桶约束开关
    bucket_min_buckets: int = 3  # 最少覆盖桶数
    bucket_max_per_bucket: int = 2  # 每桶最多选几个因子
    max_parent_occurrence: int = 0  # 父因子最大重复 (0=不限制)

    def __post_init__(self):
        """初始化后自动调整 n_jobs"""
        if self.n_jobs == -1:
            self.n_jobs = _get_optimal_n_jobs()


@njit(cache=True)
def _compute_combo_signal(factors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    T, N, F = factors.shape
    signal = np.zeros((T, N))

    for t in range(T):
        for n in range(N):
            s = 0.0
            w_sum = 0.0
            for f in range(F):
                val = factors[t, n, f]
                if not np.isnan(val):
                    s += val * weights[f]
                    w_sum += weights[f]
            if w_sum > 0:
                signal[t, n] = s / w_sum
            else:
                signal[t, n] = np.nan

    return signal


@njit(cache=True)
def _compute_rebalanced_ic(
    signal: np.ndarray, returns: np.ndarray, rebalance_freq: int
) -> Tuple[float, float, float]:
    """
    使用 Signal(T-1) 评估未来 T→T+freq 的收益，避免同日偷价。

    - 信号取自 start_idx-1（日 t-1 的收盘已知）
    - 收益累积从 start_idx 开始（即 T 日的收益属于下一期）
    - 循环从 rebalance_freq 开始以确保有上一日信号
    """
    T, N = signal.shape
    ic_buffer = np.zeros(T)
    valid_periods = 0

    for start_idx in range(rebalance_freq, T, rebalance_freq):
        end_idx = min(start_idx + rebalance_freq, T)
        if end_idx - start_idx < 2:  # 至少2天才有意义
            continue
        sig = signal[start_idx - 1]
        cumret = np.ones(N)
        for t in range(start_idx, end_idx):
            ret = returns[t]
            for n in range(N):
                if not np.isnan(ret[n]):
                    cumret[n] *= 1 + ret[n]
        cumret = cumret - 1.0
        mask = ~(np.isnan(sig) | np.isnan(cumret))
        if np.sum(mask) <= 2:
            continue
        s = sig[mask]
        r = cumret[mask]
        s_rank = np.argsort(np.argsort(s)).astype(np.float64)
        r_rank = np.argsort(np.argsort(r)).astype(np.float64)
        s_mean = np.mean(s_rank)
        r_mean = np.mean(r_rank)
        s_centered = s_rank - s_mean
        r_centered = r_rank - r_mean
        s_std = np.sqrt(np.sum(s_centered**2))
        r_std = np.sqrt(np.sum(r_centered**2))
        if s_std > 0 and r_std > 0:
            ic = np.sum(s_centered * r_centered) / (s_std * r_std)
            if not np.isnan(ic):
                ic_buffer[valid_periods] = ic
                valid_periods += 1
    if valid_periods == 0:
        return 0.0, 0.0, 0.0
    daily_ic = ic_buffer[:valid_periods]
    mean_ic = np.mean(daily_ic)
    ic_std = np.std(daily_ic)
    ir = mean_ic / ic_std if ic_std > 1e-12 else 0.0
    positive_rate = np.sum(daily_ic > 0) / valid_periods
    return mean_ic, ir, positive_rate


@njit(cache=True)
def _compute_rebalanced_return(
    signal: np.ndarray,
    returns: np.ndarray,
    exposures: np.ndarray,
    rebalance_freq: int,
    top_k: int,
    cost_arr: np.ndarray,
    use_t1_open: bool = False,
    delta_rank: float = 0.0,
    min_hold_days: int = 0,
) -> float:
    """
    计算滚动 OOS 收益率 (模拟真实交易)

    参数:
        signal: (T, N) 因子信号
        returns: (T, N) 收益率
        exposures: (T,) 逐日风险暴露系数 (0~1)，用于模拟部分仓位/停跑
        rebalance_freq: 调仓周期 (天)
        top_k: 持仓数量
        cost_arr: (N,) 每个 ETF 的单边成本 (decimal)
        delta_rank: rank01 gap threshold for swap (0 = disabled)
        min_hold_days: minimum holding days before sell (0 = disabled)

    返回:
        累计收益率 (如 0.50 = 50%)
    """
    ret, _, _ = _compute_rebalanced_return_stats(
        signal, returns, exposures, rebalance_freq, top_k,
        cost_arr, use_t1_open, delta_rank, min_hold_days, 0,
    )
    return ret


@njit(cache=True)
def _compute_rebalanced_return_stats(
    signal: np.ndarray,
    returns: np.ndarray,
    exposures: np.ndarray,
    rebalance_freq: int,
    top_k: int,
    cost_arr: np.ndarray,
    use_t1_open: bool = False,
    delta_rank: float = 0.0,
    min_hold_days: int = 0,
    warmup_length: int = 0,
):
    """
    计算滚动收益率 + Sharpe + MaxDD (模拟真实交易, 含 IS 暖启动)

    参数:
        signal: (T, N) 因子信号 (IS+OOS 拼接, 或仅 OOS)
        returns: (T, N) 收益率
        exposures: (T,) 逐日风险暴露系数
        rebalance_freq: 调仓周期 (天)
        top_k: 持仓数量
        cost_arr: (N,) 每个 ETF 的单边成本
        use_t1_open: T1_OPEN 执行模式
        delta_rank: hysteresis rank01 gap threshold
        min_hold_days: minimum holding period
        warmup_length: IS 暖启动长度; 前 warmup_length 天仅更新持仓状态,
                       不计入收益/Sharpe/MDD. 0 = 无暖启动.

    返回:
        (total_return, sharpe_ratio, max_drawdown)
        - total_return: 累计收益率 (仅 OOS 部分)
        - sharpe_ratio: 年化 Sharpe (基于逐期收益, 仅 OOS 部分)
        - max_drawdown: 最大回撤 (正值, 如 0.10 = 10%)
    """
    T, N = signal.shape
    if exposures.shape[0] != T:
        return 0.0, 0.0, 0.0

    equity = 1.0
    prev_positions = np.zeros(N, dtype=np.int64)
    hold_days_wfo = np.zeros(N, dtype=np.int64)

    # Collect per-period returns for Sharpe/MDD (OOS only)
    max_periods = T // max(rebalance_freq, 1) + 1
    period_returns = np.empty(max_periods, dtype=np.float64)
    n_oos_periods = 0
    peak_equity = 1.0
    max_dd = 0.0

    for start_idx in range(rebalance_freq, T, rebalance_freq):
        end_idx = min(start_idx + rebalance_freq, T)
        if end_idx - start_idx < 2:
            continue
        sig = signal[start_idx - 1]

        # Exp4: increment hold days
        for n in range(N):
            if prev_positions[n] == 1:
                hold_days_wfo[n] += rebalance_freq

        valid_mask = ~np.isnan(sig)
        valid_count = np.sum(valid_mask)
        if valid_count < top_k:
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_signals = sig[valid_indices]
        sorted_idx = np.argsort(-valid_signals)
        top_indices = valid_indices[sorted_idx[:top_k]]

        new_positions = np.zeros(N, dtype=np.int64)
        if delta_rank > 0.0 or min_hold_days > 0:
            h_mask = np.zeros(N, dtype=np.bool_)
            for n in range(N):
                h_mask[n] = prev_positions[n] == 1
            target_mask = apply_hysteresis(
                sig, h_mask, hold_days_wfo, top_indices,
                top_k, delta_rank, min_hold_days,
            )
            for n in range(N):
                if target_mask[n]:
                    new_positions[n] = 1
        else:
            for idx in top_indices:
                new_positions[idx] = 1

        # Update hold days
        for n in range(N):
            if prev_positions[n] == 0 and new_positions[n] == 1:
                hold_days_wfo[n] = 1
            elif prev_positions[n] == 1 and new_positions[n] == 0:
                hold_days_wfo[n] = 0

        # --- OOS scoring: only accumulate equity after warmup ---
        is_oos = start_idx >= warmup_length

        if is_oos:
            # Commission cost
            exp_trade = exposures[start_idx]
            if np.isnan(exp_trade):
                exp_trade = 1.0
            commission_cost = 0.0
            for n in range(N):
                if prev_positions[n] != new_positions[n]:
                    commission_cost += cost_arr[n] * exp_trade / top_k

            # Period return (daily compounding)
            cum_port = 1.0
            ret_start = start_idx + 1 if use_t1_open else start_idx
            for t in range(ret_start, end_idx):
                exp_t = exposures[t]
                if np.isnan(exp_t):
                    exp_t = 1.0
                day_ret_sum = 0.0
                day_cnt = 0
                for n in range(N):
                    if new_positions[n] == 1:
                        ret = returns[t, n]
                        if not np.isnan(ret):
                            day_ret_sum += ret
                            day_cnt += 1
                if day_cnt > 0:
                    avg_ret = day_ret_sum / day_cnt
                    cum_port *= 1.0 + exp_t * avg_ret

            period_ret = cum_port - 1.0 - commission_cost
            equity *= 1.0 + period_ret

            # Track for Sharpe
            period_returns[n_oos_periods] = period_ret
            n_oos_periods += 1

            # Track MDD
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_dd:
                max_dd = dd

        prev_positions = new_positions

    total_return = equity - 1.0

    # Compute Sharpe from period returns
    if n_oos_periods >= 2:
        pr = period_returns[:n_oos_periods]
        mean_r = 0.0
        for i in range(n_oos_periods):
            mean_r += pr[i]
        mean_r /= n_oos_periods
        var_r = 0.0
        for i in range(n_oos_periods):
            var_r += (pr[i] - mean_r) ** 2
        var_r /= n_oos_periods
        std_r = var_r**0.5
        if std_r > 1e-12:
            # Annualize: assume ~252/rebalance_freq periods per year
            periods_per_year = 252.0 / rebalance_freq
            sharpe = (mean_r / std_r) * (periods_per_year**0.5)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return total_return, sharpe, max_dd


class ComboWFOOptimizer:
    def __init__(
        self,
        combo_sizes: List[int] = [2, 3, 4, 5],
        is_period: int = 252,
        oos_period: int = 60,
        step_size: int = 20,
        n_jobs: int = -1,
        verbose: int = 1,
        enable_fdr: bool = True,
        fdr_alpha: float = 0.05,
        complexity_penalty_lambda: float = 0.01,
        rebalance_frequencies: List[int] = None,
        use_t1_open: bool = False,
        delta_rank: float = 0.0,
        min_hold_days: int = 0,
        use_bucket_constraints: bool = False,
        bucket_min_buckets: int = 3,
        bucket_max_per_bucket: int = 2,
        max_parent_occurrence: int = 0,
    ):
        self.use_t1_open = use_t1_open
        self.config = ComboWFOConfig(
            combo_sizes=combo_sizes,
            is_period=is_period,
            oos_period=oos_period,
            step_size=step_size,
            n_jobs=n_jobs,
            verbose=verbose,
            enable_fdr=enable_fdr,
            fdr_alpha=fdr_alpha,
            complexity_penalty_lambda=complexity_penalty_lambda,
            delta_rank=delta_rank,
            min_hold_days=min_hold_days,
            use_bucket_constraints=use_bucket_constraints,
            bucket_min_buckets=bucket_min_buckets,
            bucket_max_per_bucket=bucket_max_per_bucket,
            max_parent_occurrence=max_parent_occurrence,
        )
        self.rebalance_frequencies = (
            rebalance_frequencies if rebalance_frequencies else [5, 10, 15, 20, 25, 30]
        )

    def _generate_combos(
        self, factor_names: List[str], n_factors: int
    ) -> List[Tuple[int, ...]]:
        """生成因子组合 (索引 tuple).

        当 use_bucket_constraints=True 时, 使用跨桶约束剪枝;
        否则使用原始无约束枚举.
        """
        all_combos: List[Tuple[int, ...]] = []

        if self.config.use_bucket_constraints:
            from etf_strategy.core.factor_buckets import generate_cross_bucket_combos

            name_to_idx = {name: i for i, name in enumerate(factor_names)}
            for size in self.config.combo_sizes:
                # size=2 不可能覆盖3桶, 用 min_buckets=min(size, target)
                effective_min = min(size, self.config.bucket_min_buckets)
                name_combos = generate_cross_bucket_combos(
                    factor_names,
                    combo_size=size,
                    min_buckets=effective_min,
                    max_per_bucket=self.config.bucket_max_per_bucket,
                    max_parent_occurrence=self.config.max_parent_occurrence,
                )
                idx_combos = [
                    tuple(name_to_idx[n] for n in nc) for nc in name_combos
                ]
                all_combos.extend(idx_combos)
                logger.info(
                    f"  {size}-factor combos: {len(idx_combos)} (cross-bucket, "
                    f"min_buckets={effective_min})"
                )
        else:
            for size in self.config.combo_sizes:
                combos = list(combinations(range(n_factors), size))
                all_combos.extend(combos)
                logger.info(f"  {size}-factor combos: {len(combos)}")

        return all_combos

    def _generate_windows(self, total_days: int):
        windows = []
        offset = 0
        while True:
            is_start = offset
            is_end = offset + self.config.is_period
            oos_start = is_end
            oos_end = oos_start + self.config.oos_period
            if oos_end > total_days:
                break
            windows.append(((is_start, is_end), (oos_start, oos_end)))
            offset += self.config.step_size
        return windows

    def _test_combo_single_window(
        self,
        combo_indices,
        factors_is,
        returns_is,
        factors_oos,
        returns_oos,
        exposures_is,
        exposures_oos,
        pos_size: int = 2,
        cost_arr: np.ndarray | None = None,
        factor_ics: np.ndarray | None = None,
    ):
        """
        测试单个 WFO 窗口的因子组合表现

        返回:
            best_score: 最佳 IC
            best_ir: 最佳 IR
            best_pos_rate: 最佳正率
            best_freq: 最佳调仓频率
            oos_return: OOS 收益率 (含 hysteresis + IS 暖启动)
            oos_sharpe: OOS Sharpe ratio (年化)
            oos_maxdd: OOS 最大回撤 (正值)
        """
        n_factors = len(combo_indices)
        if factor_ics is not None:
            # 使用预计算的IC（避免combo间重复计算）
            is_ics = np.array([factor_ics[f_idx] for f_idx in combo_indices])
        else:
            is_ics = np.zeros(n_factors)
            for i, f_idx in enumerate(combo_indices):
                is_ics[i] = compute_spearman_ic_numba(
                    factors_is[:, :, f_idx], returns_is
                )
        abs_ics = np.abs(is_ics)
        if abs_ics.sum() > 0:
            weights = abs_ics / abs_ics.sum()
        else:
            weights = np.ones(n_factors) / n_factors

        factors_oos_combo = factors_oos[:, :, combo_indices]
        signal_oos = _compute_combo_signal(factors_oos_combo, weights)

        best_score = -999.0
        best_freq = 10
        best_ir = 0.0
        best_pos_rate = 0.0

        for freq in self.rebalance_frequencies:
            mean_ic, ir, pos_rate = _compute_rebalanced_ic(
                signal_oos, returns_oos, freq
            )
            if mean_ic > best_score:
                best_score = mean_ic
                best_freq = freq
                best_ir = ir
                best_pos_rate = pos_rate

        # Compute OOS return + Sharpe + MaxDD with IS warm-up for hysteresis
        N_oos = returns_oos.shape[1]
        if cost_arr is None:
            cost_arr_local = np.full(N_oos, 0.0002, dtype=np.float64)
        else:
            cost_arr_local = cost_arr

        use_warmup = (
            self.config.delta_rank > 0.0 or self.config.min_hold_days > 0
        )

        if use_warmup:
            # Concatenate IS+OOS for hysteresis warm-up
            factors_is_combo = factors_is[:, :, combo_indices]
            signal_is = _compute_combo_signal(factors_is_combo, weights)
            signal_full = np.concatenate((signal_is, signal_oos), axis=0)
            returns_full = np.concatenate((returns_is, returns_oos), axis=0)
            exposures_full = np.concatenate((exposures_is, exposures_oos), axis=0)
            warmup_length = len(returns_is)
        else:
            signal_full = signal_oos
            returns_full = returns_oos
            exposures_full = exposures_oos
            warmup_length = 0

        oos_return, oos_sharpe, oos_maxdd = _compute_rebalanced_return_stats(
            signal_full,
            returns_full,
            exposures_full,
            best_freq,
            pos_size,
            cost_arr_local,
            self.use_t1_open,
            self.config.delta_rank,
            self.config.min_hold_days,
            warmup_length,
        )

        return (
            best_score, best_ir, best_pos_rate, best_freq,
            oos_return, oos_sharpe, oos_maxdd,
        )

    def _test_combo_impl(
        self,
        combo_indices,
        factors_data,
        returns,
        exposures,
        windows,
        pos_size: int = 2,
        cost_arr: np.ndarray | None = None,
        precomputed_ics: np.ndarray | None = None,
    ):
        oos_ic_list = []
        oos_ir_list = []
        positive_rate_list = []
        best_freq_list = []
        oos_return_list = []
        oos_sharpe_list = []
        oos_maxdd_list = []

        for w_idx, (is_range, oos_range) in enumerate(windows):
            is_start, is_end = is_range
            oos_start, oos_end = oos_range
            factors_is = factors_data[is_start:is_end]
            returns_is = returns[is_start:is_end]
            factors_oos = factors_data[oos_start:oos_end]
            returns_oos = returns[oos_start:oos_end]
            exposures_is = exposures[is_start:is_end]
            exposures_oos = exposures[oos_start:oos_end]

            window_ics = (
                precomputed_ics[w_idx] if precomputed_ics is not None else None
            )

            res = self._test_combo_single_window(
                list(combo_indices),
                factors_is,
                returns_is,
                factors_oos,
                returns_oos,
                exposures_is,
                exposures_oos,
                pos_size,
                cost_arr,
                factor_ics=window_ics,
            )

            oos_ic_list.append(res[0])
            oos_ir_list.append(res[1])
            positive_rate_list.append(res[2])
            best_freq_list.append(res[3])
            oos_return_list.append(res[4])
            oos_sharpe_list.append(res[5])
            oos_maxdd_list.append(res[6])

        return {
            "combo_indices": combo_indices,
            "oos_ic_list": oos_ic_list,
            "oos_ir_list": oos_ir_list,
            "positive_rate_list": positive_rate_list,
            "best_freq_list": best_freq_list,
            "oos_return_list": oos_return_list,
            "oos_sharpe_list": oos_sharpe_list,
            "oos_maxdd_list": oos_maxdd_list,
        }

    def _test_combo_batch(
        self,
        combo_batch,
        factors_data,
        returns,
        exposures,
        windows,
        pos_size: int = 2,
        cost_arr: np.ndarray | None = None,
        precomputed_ics: np.ndarray | None = None,
    ):
        """批量处理多个 combo，减少 joblib IPC 开销"""
        return [
            self._test_combo_impl(
                combo, factors_data, returns, exposures,
                windows, pos_size, cost_arr, precomputed_ics,
            )
            for combo in combo_batch
        ]

    def _calc_stability_score(
        self, oos_ic_list, oos_ir_list, positive_rate_list, combo_size
    ):
        mean_ic = np.mean(oos_ic_list)
        mean_ir = np.mean(oos_ir_list)
        mean_pos_rate = np.mean(positive_rate_list)
        ic_std = np.std(oos_ic_list)
        base_score = 0.5 * mean_ic + 0.3 * mean_ir + 0.2 * mean_pos_rate
        stability_bonus = -0.1 * ic_std
        complexity_penalty = -self.config.complexity_penalty_lambda * combo_size
        final_score = base_score + stability_bonus + complexity_penalty
        return final_score

    def _apply_fdr_correction(self, results_df):
        def calc_pvalue(row):
            ics = np.array(row["oos_ic_list"])
            if len(ics) < 2:
                return 1.0
            _, p_val = stats.ttest_1samp(ics, 0.0, alternative="greater")
            return p_val

        results_df["p_value"] = results_df.apply(calc_pvalue, axis=1)
        _, q_values, _, _ = multipletests(
            results_df["p_value"], alpha=self.config.fdr_alpha, method="fdr_bh"
        )
        results_df["q_value"] = q_values
        results_df["is_significant"] = q_values < self.config.fdr_alpha
        return results_df

    def run_combo_search(
        self,
        factors_data,
        returns,
        factor_names,
        top_n=100,
        pos_size: int = 2,
        commission_rate: float = 0.0002,
        cost_arr: np.ndarray | None = None,
        exposures: np.ndarray | None = None,
    ):
        """
        执行因子组合搜索

        参数:
            factors_data: (T, N, F) 因子数据
            returns: (T, N) 收益率数据
            factor_names: 因子名称列表
            top_n: 返回 Top N 组合
            pos_size: 持仓数量 (用于计算 OOS 收益)
            commission_rate: 手续费率 (legacy fallback, 当 cost_arr=None 时使用)
            cost_arr: (N,) per-ETF 单边成本数组, 覆盖 commission_rate
            exposures: (T,) 逐日暴露（regime gate），None 表示全 1.0
        """
        T, N, F = factors_data.shape
        if exposures is None:
            exposures = np.ones(T, dtype=np.float64)
        else:
            exposures = np.asarray(exposures, dtype=np.float64)
            if exposures.shape[0] != T:
                raise ValueError(
                    f"exposures length mismatch: expected {T}, got {exposures.shape[0]}"
                )
        # Exp2: 构建 per-ETF 成本数组 (fallback to uniform commission_rate)
        if cost_arr is None:
            cost_arr = np.full(N, commission_rate, dtype=np.float64)
        else:
            cost_arr = np.asarray(cost_arr, dtype=np.float64)

        logger.info(f"Data: {T} days x {N} ETFs x {F} factors")
        if self.config.delta_rank > 0 or self.config.min_hold_days > 0:
            logger.info(
                f"Hysteresis enabled in WFO OOS: delta_rank={self.config.delta_rank}, "
                f"min_hold_days={self.config.min_hold_days} (with IS warm-up)"
            )
        logger.info(f"Standard mode: IC-based optimization (pos_size={pos_size})")

        windows = self._generate_windows(T)
        logger.info(f"Generated {len(windows)} WFO windows")
        all_combos = self._generate_combos(factor_names, F)
        logger.info(f"Total: {len(all_combos)} combos")

        # 预计算所有因子的单因子IC（避免combo间重复计算）
        n_windows = len(windows)
        precomputed_ics = np.zeros((n_windows, F))
        for w_idx, (is_range, _) in enumerate(windows):
            is_start, is_end = is_range
            factors_is = factors_data[is_start:is_end]
            returns_is = returns[is_start:is_end]
            # 批量计算: (F, T, N) → (F,) IC值
            all_factor_signals = np.transpose(factors_is, (2, 0, 1))
            precomputed_ics[w_idx] = compute_multiple_ics_numba(
                all_factor_signals, returns_is
            )
        logger.info(
            f"Pre-computed {n_windows} x {F} = {n_windows * F} factor ICs"
        )

        # joblib 批处理: 减少 IPC 次数 (12,597 → ~126 批)
        BATCH_SIZE = 100
        combo_batches = [
            all_combos[i : i + BATCH_SIZE]
            for i in range(0, len(all_combos), BATCH_SIZE)
        ]
        logger.info(
            f"Parallel: n_jobs={self.config.n_jobs}, "
            f"batches={len(combo_batches)} (batch_size={BATCH_SIZE})"
        )

        results_nested = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
            delayed(self._test_combo_batch)(
                batch,
                factors_data,
                returns,
                exposures,
                windows,
                pos_size,
                cost_arr,
                precomputed_ics,
            )
            for batch in tqdm(
                combo_batches, desc="WFO combo eval", unit="batch", ncols=80
            )
        )
        results = [r for batch_results in results_nested for r in batch_results]

        records = []
        for res in results:
            combo_indices = res["combo_indices"]
            combo_str = " + ".join([factor_names[i] for i in combo_indices])
            oos_ic_list = res["oos_ic_list"]
            oos_ir_list = res["oos_ir_list"]
            positive_rate_list = res["positive_rate_list"]
            best_freq_list = res["best_freq_list"]
            oos_return_list = res["oos_return_list"]
            oos_sharpe_list = res["oos_sharpe_list"]
            oos_maxdd_list = res["oos_maxdd_list"]

            mean_oos_ic = np.mean(oos_ic_list)
            oos_ic_std = np.std(oos_ic_list)
            mean_oos_ir = np.mean(oos_ir_list)
            mean_positive_rate = np.mean(positive_rate_list)

            mean_oos_return = np.mean(oos_return_list)
            cum_oos_return = 1.0
            for r in oos_return_list:
                cum_oos_return *= 1.0 + r
            cum_oos_return -= 1.0

            mean_oos_sharpe = (
                np.mean(oos_sharpe_list) if oos_sharpe_list else 0.0
            )
            mean_oos_maxdd = (
                np.mean(oos_maxdd_list) if oos_maxdd_list else 0.0
            )

            freq_counter = Counter(best_freq_list)
            best_rebalance_freq = freq_counter.most_common(1)[0][0]

            stability_score = self._calc_stability_score(
                oos_ic_list, oos_ir_list, positive_rate_list, len(combo_indices)
            )
            records.append(
                {
                    "combo": combo_str,
                    "combo_size": len(combo_indices),
                    "mean_oos_ic": mean_oos_ic,
                    "oos_ic_std": oos_ic_std,
                    "oos_ic_ir": mean_oos_ir,
                    "positive_rate": mean_positive_rate,
                    "best_rebalance_freq": best_rebalance_freq,
                    "stability_score": stability_score,
                    "mean_oos_return": mean_oos_return,
                    "cum_oos_return": cum_oos_return,
                    "mean_oos_sharpe": mean_oos_sharpe,
                    "mean_oos_maxdd": mean_oos_maxdd,
                    "oos_ic_list": oos_ic_list,
                    "oos_ir_list": oos_ir_list,
                    "positive_rate_list": positive_rate_list,
                    "best_freq_list": best_freq_list,
                    "oos_return_list": oos_return_list,
                    "oos_sharpe_list": oos_sharpe_list,
                    "oos_maxdd_list": oos_maxdd_list,
                }
            )
        results_df = pd.DataFrame(records)
        if self.config.enable_fdr:
            logger.info("Applying FDR...")
            results_df = self._apply_fdr_correction(results_df)
        else:
            results_df["p_value"] = np.nan
            results_df["q_value"] = np.nan
            results_df["is_significant"] = True

        # Compute execution_score: normalized composite of return + sharpe + maxdd
        # This score reflects production execution (with hysteresis + IS warm-up)
        use_execution_scoring = (
            self.config.delta_rank > 0 or self.config.min_hold_days > 0
        )
        if use_execution_scoring:
            results_df = self._add_execution_score(results_df)

        # Sorting: execution_score (when available) > calibrated > IC
        calibrated_model_path = Path("results/calibrator_gbdt_full.joblib")
        use_calibrated = calibrated_model_path.exists()

        if use_execution_scoring and "execution_score" in results_df.columns:
            results_df = results_df.sort_values(
                by=["execution_score", "mean_oos_ic"],
                ascending=[False, False],
            ).reset_index(drop=True)
            sort_method = "execution_score (hysteresis-aware)"
        elif use_calibrated:
            try:
                calibrator = WFORealBacktestCalibrator.load(
                    calibrated_model_path
                )
                results_df = results_df.copy()
                results_df["calibrated_sharpe_pred"] = calibrator.predict(
                    results_df
                )
                results_df = results_df.sort_values(
                    by=["calibrated_sharpe_pred", "stability_score"],
                    ascending=[False, False],
                ).reset_index(drop=True)
                sort_method = "calibrated"
            except Exception as e:
                logger.warning(f"Calibrator failed, fallback to IC: {e}")
                results_df = results_df.sort_values(
                    by=["mean_oos_ic", "stability_score"],
                    ascending=[False, False],
                ).reset_index(drop=True)
                sort_method = "IC"
        else:
            results_df = results_df.sort_values(
                by=["mean_oos_ic", "stability_score"],
                ascending=[False, False],
            ).reset_index(drop=True)
            sort_method = "IC"

        top_combos = results_df.head(top_n).to_dict("records")
        logger.info(
            f"Found {len(top_combos)} top combos (sorted by {sort_method})"
        )
        return top_combos, results_df

    @staticmethod
    def _add_execution_score(df: pd.DataFrame) -> pd.DataFrame:
        """Compute execution_score from hysteresis-aware OOS metrics.

        Composite: 0.4 * norm(mean_oos_return) + 0.3 * norm(mean_oos_sharpe)
                   + 0.3 * norm(-mean_oos_maxdd)

        All components rank-normalized to [0, 1] to avoid scale dependency.
        """
        df = df.copy()
        n = len(df)
        if n < 2:
            df["execution_score"] = 0.0
            return df

        # Rank-normalize each component to [0, 1]
        def rank_normalize(series):
            ranks = series.rank(method="average", ascending=True)
            return (ranks - 1) / max(n - 1, 1)

        norm_ret = rank_normalize(df["mean_oos_return"])
        norm_sharpe = rank_normalize(df["mean_oos_sharpe"])
        # For maxdd, lower is better -> negate before ranking
        norm_mdd = rank_normalize(-df["mean_oos_maxdd"])

        df["execution_score"] = (
            0.4 * norm_ret + 0.3 * norm_sharpe + 0.3 * norm_mdd
        )
        return df
