"""Combo-level WFO Optimizer."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .ic_calculator_numba import compute_spearman_ic_numba

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ComboWFOConfig:
    combo_sizes: List[int]
    is_period: int
    oos_period: int
    step_size: int
    n_jobs: int = -1
    verbose: int = 1
    enable_fdr: bool = True
    fdr_alpha: float = 0.05
    complexity_penalty_lambda: float = 0.01


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
    T, N = signal.shape
    n_periods = (T + rebalance_freq - 1) // rebalance_freq
    ic_buffer = np.empty(n_periods)
    valid_periods = 0

    for i in range(n_periods):
        start_idx = i * rebalance_freq
        end_idx = min(start_idx + rebalance_freq, T)
        sig = signal[start_idx]
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
    ):
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
        )
        self.rebalance_frequencies = (
            rebalance_frequencies if rebalance_frequencies else [5, 10, 15, 20, 25, 30]
        )

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
        self, combo_indices, factors_is, returns_is, factors_oos, returns_oos
    ):
        n_factors = len(combo_indices)
        is_ics = np.zeros(n_factors)
        for i, f_idx in enumerate(combo_indices):
            is_ics[i] = compute_spearman_ic_numba(factors_is[:, :, f_idx], returns_is)
        abs_ics = np.abs(is_ics)
        if abs_ics.sum() > 0:
            weights = abs_ics / abs_ics.sum()
        else:
            weights = np.ones(n_factors) / n_factors
        best_ic = -999.0
        best_freq = 10
        best_ir = 0.0
        best_pos_rate = 0.0
        factors_oos_combo = factors_oos[:, :, combo_indices]
        for freq in self.rebalance_frequencies:
            signal_oos = _compute_combo_signal(factors_oos_combo, weights)
            mean_ic, ir, pos_rate = _compute_rebalanced_ic(
                signal_oos, returns_oos, freq
            )
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_freq = freq
                best_ir = ir
                best_pos_rate = pos_rate
        return best_ic, best_ir, best_pos_rate, best_freq

    def _test_combo_impl(self, combo_indices, factors_data, returns, windows):
        oos_ic_list = []
        oos_ir_list = []
        positive_rate_list = []
        best_freq_list = []
        for is_range, oos_range in windows:
            is_start, is_end = is_range
            oos_start, oos_end = oos_range
            factors_is = factors_data[is_start:is_end]
            returns_is = returns[is_start:is_end]
            factors_oos = factors_data[oos_start:oos_end]
            returns_oos = returns[oos_start:oos_end]
            oos_ic, oos_ir, pos_rate, best_freq = self._test_combo_single_window(
                list(combo_indices), factors_is, returns_is, factors_oos, returns_oos
            )
            oos_ic_list.append(oos_ic)
            oos_ir_list.append(oos_ir)
            positive_rate_list.append(pos_rate)
            best_freq_list.append(best_freq)
        return {
            "combo_indices": combo_indices,
            "oos_ic_list": oos_ic_list,
            "oos_ir_list": oos_ir_list,
            "positive_rate_list": positive_rate_list,
            "best_freq_list": best_freq_list,
        }

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
            t_stat, p_val = stats.ttest_1samp(ics, 0.0, alternative="greater")
            return p_val

        results_df["p_value"] = results_df.apply(calc_pvalue, axis=1)
        _, q_values, _, _ = multipletests(
            results_df["p_value"], alpha=self.config.fdr_alpha, method="fdr_bh"
        )
        results_df["q_value"] = q_values
        results_df["is_significant"] = q_values < self.config.fdr_alpha
        return results_df

    def run_combo_search(self, factors_data, returns, factor_names, top_n=100):
        T, N, F = factors_data.shape
        logger.info(f"Data: {T} days x {N} ETFs x {F} factors")
        windows = self._generate_windows(T)
        logger.info(f"Generated {len(windows)} WFO windows")
        all_combos = []
        for size in self.config.combo_sizes:
            combos = list(combinations(range(F), size))
            all_combos.extend(combos)
            logger.info(f"  {size}-factor combos: {len(combos)}")
        logger.info(f"Total: {len(all_combos)} combos")
        results = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
            delayed(self._test_combo_impl)(combo, factors_data, returns, windows)
            for combo in tqdm(all_combos, desc="WFO组合评估", unit="combo", ncols=80)
        )
        records = []
        for res in results:
            combo_indices = res["combo_indices"]
            combo_str = " + ".join([factor_names[i] for i in combo_indices])
            oos_ic_list = res["oos_ic_list"]
            oos_ir_list = res["oos_ir_list"]
            positive_rate_list = res["positive_rate_list"]
            best_freq_list = res["best_freq_list"]
            mean_oos_ic = np.mean(oos_ic_list)
            oos_ic_std = np.std(oos_ic_list)
            mean_oos_ir = np.mean(oos_ir_list)
            mean_positive_rate = np.mean(positive_rate_list)
            from collections import Counter

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
                    "oos_ic_list": oos_ic_list,
                    "oos_ir_list": oos_ir_list,
                    "positive_rate_list": positive_rate_list,
                    "best_freq_list": best_freq_list,
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
        # 改为按IC排序（IC越高越好），稳定性得分作为次要指标
        results_df = results_df.sort_values(
            by=["mean_oos_ic", "stability_score"], ascending=[False, False]
        ).reset_index(drop=True)
        top_combos = results_df.head(top_n).to_dict("records")
        logger.info(f"Found {len(top_combos)} top combos (sorted by IC)")
        return top_combos, results_df
