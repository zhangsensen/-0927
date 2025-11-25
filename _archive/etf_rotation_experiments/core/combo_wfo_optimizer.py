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
    scoring_strategy: str = "ic"
    oos_portfolio_size: int = 5


@njit(cache=True)
def _compute_combo_signal(factors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine factor cube into a single signal using signed weights."""

    T, N, F = factors.shape
    signal = np.full((T, N), np.nan)

    for t in range(T):
        for n in range(N):
            s = 0.0
            w_abs_sum = 0.0
            for f in range(F):
                val = factors[t, n, f]
                if not np.isnan(val):
                    w = weights[f]
                    if w != 0.0:
                        s += val * w
                        w_abs_sum += abs(w)
            if w_abs_sum > 0.0:
                signal[t, n] = s / w_abs_sum

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


@njit(cache=True)
def _compute_rebalanced_sharpe_stats(
    signal: np.ndarray,
    returns: np.ndarray,
    rebalance_freq: int,
    top_k: int,
) -> Tuple[float, float, float, int]:
    if rebalance_freq <= 0 or top_k <= 0:
        return 0.0, 0.0, 0.0, 0

    T, N = signal.shape
    if returns.shape[0] != T or N != returns.shape[1]:
        return 0.0, 0.0, 0.0, 0

    mean_ret = 0.0
    m2 = 0.0
    count = 0
    neg_inf = -1.0e12

    for start in range(0, T, rebalance_freq):
        end = start + rebalance_freq
        if end > T:
            end = T

        sig = signal[start]
        tmp = np.empty(N, dtype=np.float64)
        for n in range(N):
            val = sig[n]
            if np.isnan(val):
                tmp[n] = neg_inf
            else:
                tmp[n] = val

        order = np.argsort(tmp)
        valid_count = 0
        selected = np.empty(top_k, dtype=np.int64)
        for i in range(top_k):
            idx = order[N - 1 - i]
            if tmp[idx] == neg_inf:
                break
            selected[i] = idx
            valid_count += 1
        if valid_count == 0:
            continue

        for t in range(start, end):
            ret_sum = 0.0
            asset_count = 0
            for i in range(valid_count):
                idx = selected[i]
                r = returns[t, idx]
                if not np.isnan(r):
                    ret_sum += r
                    asset_count += 1
            if asset_count == 0:
                continue
            port_ret = ret_sum / asset_count
            count += 1
            delta = port_ret - mean_ret
            mean_ret += delta / count
            m2 += delta * (port_ret - mean_ret)

    if count <= 1:
        return 0.0, mean_ret, 0.0, count

    variance = m2 / (count - 1)
    if variance <= 1e-12:
        return 0.0, mean_ret, np.sqrt(variance), count
    daily_std = np.sqrt(variance)
    sharpe = np.sqrt(252.0) * mean_ret / daily_std
    return sharpe, mean_ret, daily_std, count


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
        scoring_strategy: str = "ic",
        scoring_position_size: int = 5,
    ):
        strategy_alias = {
            "oos_sharpe": "oos_sharpe_proxy",
            "oos_sharpe_proxy": "oos_sharpe_proxy",
            "oos_sharpe_true": "oos_sharpe_true",
            "oos_sharpe_compound": "oos_sharpe_compound",
            "compound_sharpe": "oos_sharpe_compound",
        }
        scoring_strategy = scoring_strategy.lower()
        scoring_strategy = strategy_alias.get(scoring_strategy, scoring_strategy)
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
            scoring_strategy=scoring_strategy,
            oos_portfolio_size=scoring_position_size,
        )
        self.rebalance_frequencies = (
            rebalance_frequencies if rebalance_frequencies else [5, 10, 15, 20, 25, 30]
        )
        allowed_strategies = {"ic", "oos_sharpe_proxy", "oos_sharpe_true", "oos_sharpe_compound"}
        if scoring_strategy not in allowed_strategies:
            raise ValueError(f"Unsupported scoring_strategy: {scoring_strategy}")

    def _simulate_portfolio_returns(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
        rebalance_freq: int,
        top_k: int,
    ) -> List[float]:
        """Simulate equal-weight top-k portfolio returns for the given signal window."""

        if rebalance_freq <= 0 or top_k <= 0:
            return []

        if signal.shape != returns.shape:
            return []

        T, N = signal.shape
        daily_returns: List[float] = []
        neg_inf = -1.0e12

        for start in range(0, T, rebalance_freq):
            end = min(start + rebalance_freq, T)

            sig = signal[start]
            filled = np.where(np.isnan(sig), neg_inf, sig)
            order = np.argsort(filled)

            selected: List[int] = []
            for idx in reversed(order):
                if filled[idx] == neg_inf:
                    continue
                selected.append(int(idx))
                if len(selected) == top_k:
                    break

            if not selected:
                continue

            for t in range(start, end):
                day_rets = returns[t, selected]
                valid_mask = ~np.isnan(day_rets)
                if not np.any(valid_mask):
                    continue
                port_ret = float(np.mean(day_rets[valid_mask]))
                daily_returns.append(port_ret)

        return daily_returns

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
        if factors_is.shape[0] <= 1 or factors_oos.shape[0] <= 1:
            default_freq = self.rebalance_frequencies[0]
            return 0.0, 0.0, 0.0, default_freq, 0.0, 0.0, 0.0, 0

        factors_is_aligned = factors_is[:-1]
        returns_is_aligned = returns_is[1:]
        factors_oos_aligned = factors_oos[:-1]
        returns_oos_aligned = returns_oos[1:]

        is_ics = np.zeros(n_factors)
        for i, f_idx in enumerate(combo_indices):
            is_slice = factors_is_aligned[:, :, f_idx]
            is_ics[i] = compute_spearman_ic_numba(is_slice, returns_is_aligned)

        abs_sum = np.sum(np.abs(is_ics))
        if abs_sum > 0.0:
            weights = is_ics / abs_sum
        else:
            weights = np.ones(n_factors) / n_factors

        best_ic = -999.0
        best_freq = self.rebalance_frequencies[0]
        best_ir = 0.0
        best_pos_rate = 0.0
        factors_oos_combo = factors_oos_aligned[:, :, combo_indices]
        signal_oos = _compute_combo_signal(factors_oos_combo, weights)
        for freq in self.rebalance_frequencies:
            mean_ic, ir, pos_rate = _compute_rebalanced_ic(
                signal_oos, returns_oos_aligned, freq
            )
            if mean_ic > best_ic:
                best_ic = mean_ic
                best_freq = freq
                best_ir = ir
                best_pos_rate = pos_rate

        window_returns = self._simulate_portfolio_returns(
            signal_oos,
            returns_oos_aligned,
            best_freq,
            self.config.oos_portfolio_size,
        )

        sample_count = len(window_returns)
        if sample_count == 0:
            sharpe = 0.0
            mean_ret = 0.0
            std_ret = 0.0
        else:
            returns_arr = np.array(window_returns, dtype=float)
            mean_ret = float(np.mean(returns_arr))
            if sample_count > 1:
                std_ret = float(np.std(returns_arr, ddof=1))
            else:
                std_ret = 0.0
            if std_ret > 1e-12:
                sharpe = float(np.sqrt(252.0) * mean_ret / std_ret)
            else:
                sharpe = 0.0
        return (
            best_ic,
            best_ir,
            best_pos_rate,
            best_freq,
            sharpe,
            mean_ret,
            std_ret,
            sample_count,
            window_returns,
        )

    def _test_combo_impl(self, combo_indices, factors_data, returns, windows):
        oos_ic_list = []
        oos_ir_list = []
        positive_rate_list = []
        best_freq_list = []
        sharpe_list = []
        mean_ret_list = []
        std_ret_list = []
        sample_count_list = []
        portfolio_returns_history: List[List[float]] = []
        for is_range, oos_range in windows:
            is_start, is_end = is_range
            oos_start, oos_end = oos_range
            factors_is = factors_data[is_start:is_end]
            returns_is = returns[is_start:is_end]
            factors_oos = factors_data[oos_start:oos_end]
            returns_oos = returns[oos_start:oos_end]
            (
                oos_ic,
                oos_ir,
                pos_rate,
                best_freq,
                sharpe,
                mean_ret,
                std_ret,
                sample_count,
                window_returns,
            ) = self._test_combo_single_window(
                list(combo_indices), factors_is, returns_is, factors_oos, returns_oos
            )
            oos_ic_list.append(oos_ic)
            oos_ir_list.append(oos_ir)
            positive_rate_list.append(pos_rate)
            best_freq_list.append(best_freq)
            sharpe_list.append(sharpe)
            mean_ret_list.append(mean_ret)
            std_ret_list.append(std_ret)
            sample_count_list.append(sample_count)
            portfolio_returns_history.append(window_returns)

        flat_returns = [r for window_returns in portfolio_returns_history for r in window_returns]
        if len(flat_returns) == 0:
            compound_mean = 0.0
            compound_std = 0.0
            compound_sharpe = 0.0
        else:
            flat_array = np.array(flat_returns, dtype=float)
            compound_mean = float(np.mean(flat_array))
            if flat_array.size > 1:
                compound_std = float(np.std(flat_array, ddof=1))
            else:
                compound_std = 0.0
            if compound_std > 1e-12:
                compound_sharpe = float(np.sqrt(252.0) * compound_mean / compound_std)
            else:
                compound_sharpe = 0.0
        return {
            "combo_indices": combo_indices,
            "oos_ic_list": oos_ic_list,
            "oos_ir_list": oos_ir_list,
            "positive_rate_list": positive_rate_list,
            "best_freq_list": best_freq_list,
            "oos_sharpe_list": sharpe_list,
            "oos_daily_mean_list": mean_ret_list,
            "oos_daily_std_list": std_ret_list,
            "oos_sample_count_list": sample_count_list,
            "oos_compound_sharpe": compound_sharpe,
            "oos_compound_mean": compound_mean,
            "oos_compound_std": compound_std,
            "oos_compound_sample_count": len(flat_returns),
        }

    def _calc_stability_score(
        self, oos_ic_list, oos_ir_list, positive_rate_list, combo_size
    ):
        mean_ic = np.mean(oos_ic_list)
        mean_ir = np.mean(oos_ir_list)
        mean_pos_rate = np.mean(positive_rate_list)
        ic_std = np.std(oos_ic_list)
        
        # [P0修复] 降低IC权重,增加IR和稳定性权重,避免过度追求高IC
        # 原: 0.5*IC + 0.3*IR + 0.2*正率
        # 新: 0.3*IC + 0.3*IR + 0.2*正率 + 0.2*稳定性奖励
        base_score = 0.3 * mean_ic + 0.3 * mean_ir + 0.2 * mean_pos_rate
        
        # 增强稳定性奖励: 从-0.1*std改为-0.2*std
        stability_bonus = -0.2 * ic_std
        
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

    def run_combo_search(self, factors_data, returns, factor_names, top_n=5000):
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
        
        # 添加诊断日志：检查前几个组合的IC值
        logger.info("诊断：测试前3个组合的IC计算...")
        for i, combo in enumerate(all_combos[:3]):
            test_result = self._test_combo_impl(combo, factors_data, returns, windows)
            logger.info(f"  组合 {i+1} ({' + '.join([factor_names[j] for j in combo])}): "
                       f"mean_oos_ic={np.mean(test_result['oos_ic_list']):.4f}, "
                       f"oos_ic_list={test_result['oos_ic_list']}")
        
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
            oos_sharpe_list = res["oos_sharpe_list"]
            oos_mean_list = res["oos_daily_mean_list"]
            oos_std_list = res["oos_daily_std_list"]
            compound_sharpe = res.get("oos_compound_sharpe", 0.0)
            compound_mean = res.get("oos_compound_mean", 0.0)
            compound_std = res.get("oos_compound_std", 0.0)
            compound_sample_count = res.get("oos_compound_sample_count", 0)
            mean_oos_ic = np.mean(oos_ic_list)
            oos_ic_std = np.std(oos_ic_list)
            mean_oos_ir = np.mean(oos_ir_list)
            mean_positive_rate = np.mean(positive_rate_list)
            mean_oos_sharpe = np.mean(oos_sharpe_list) if len(oos_sharpe_list) else 0.0
            oos_sharpe_std = np.std(oos_sharpe_list) if len(oos_sharpe_list) else 0.0
            sample_count_list = res["oos_sample_count_list"]
            mean_oos_sample_count = np.mean(sample_count_list) if len(sample_count_list) else 0.0
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
                    "oos_sharpe_list": oos_sharpe_list,
                    "mean_oos_sharpe": mean_oos_sharpe,
                    "oos_sharpe_std": oos_sharpe_std,
                    "oos_daily_mean_list": oos_mean_list,
                    "oos_daily_std_list": oos_std_list,
                    "oos_sample_count_list": sample_count_list,
                    "mean_oos_sample_count": mean_oos_sample_count,
                    "oos_compound_sharpe": compound_sharpe,
                    "oos_compound_mean": compound_mean,
                    "oos_compound_std": compound_std,
                    "oos_compound_sample_count": compound_sample_count,
                }
            )
        results_df = pd.DataFrame(records)
        logger.info(f"结果DataFrame创建完成: {len(results_df)}行, 列: {list(results_df.columns)}")
        logger.info(f"mean_oos_ic统计: mean={results_df['mean_oos_ic'].mean():.4f}, std={results_df['mean_oos_ic'].std():.4f}, min={results_df['mean_oos_ic'].min():.4f}, max={results_df['mean_oos_ic'].max():.4f}")
        if "mean_oos_sharpe" in results_df.columns:
            logger.info(
                "mean_oos_sharpe统计: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                results_df["mean_oos_sharpe"].mean(),
                results_df["mean_oos_sharpe"].std(),
                results_df["mean_oos_sharpe"].min(),
                results_df["mean_oos_sharpe"].max(),
            )
        
        if self.config.enable_fdr:
            logger.info("Applying FDR...")
            results_df = self._apply_fdr_correction(results_df)
        else:
            results_df["p_value"] = np.nan
            results_df["q_value"] = np.nan
            results_df["is_significant"] = True
        # 轻量级 oos_sharpe_proxy：基于 IC 水平、IC 稳定性与正收益占比的标准化加权合成
        try:
            ic = results_df["mean_oos_ic"].values
            ir = results_df.get("oos_ic_ir", pd.Series(np.zeros(len(results_df)))).values
            pos = results_df.get("positive_rate", pd.Series(np.zeros(len(results_df)))).values
            ic_std = results_df.get("oos_ic_std", pd.Series(np.zeros(len(results_df)))).values

            def _z(x):
                m = np.nanmean(x)
                s = np.nanstd(x)
                s = s if s > 1e-12 else 1.0
                return (x - m) / s

            z_ic = _z(ic)
            z_ir = _z(ir)
            z_pos = _z(pos)
            z_icstd = _z(ic_std)
            # 权重：强调 IC 与 IR，同时奖励正收益占比，惩罚波动（ic_std）
            proxy = 0.5 * z_ic + 0.3 * z_ir + 0.2 * z_pos - 0.3 * z_icstd
            results_df["oos_sharpe_proxy"] = proxy
            logger.info("Computed lightweight oos_sharpe_proxy for %d combos", len(results_df))
        except Exception as e:
            logger.warning("Failed to compute oos_sharpe_proxy: %s", e)
        results_df = self._apply_scoring(results_df)
        top_combos = results_df.head(top_n).to_dict("records")
        logger.info(
            f"Found {len(top_combos)} top combos (sorted by {self.config.scoring_strategy})"
        )
        return top_combos, results_df

    def _apply_scoring(self, results_df: pd.DataFrame) -> pd.DataFrame:
        strategy = self.config.scoring_strategy
        if strategy == "oos_sharpe_proxy":
            if "oos_sharpe_proxy" not in results_df.columns:
                raise ValueError(
                    "oos_sharpe_proxy column is required for 'oos_sharpe_proxy' scoring strategy"
                )
            sort_columns = ["oos_sharpe_proxy", "stability_score", "mean_oos_ic"]
        elif strategy == "oos_sharpe_compound":
            if "oos_compound_sharpe" not in results_df.columns:
                raise ValueError(
                    "oos_compound_sharpe column is required for 'oos_sharpe_compound' scoring strategy"
                )
            sort_columns = ["oos_compound_sharpe", "stability_score", "mean_oos_ic"]
        elif strategy == "oos_sharpe_true":
            if "mean_oos_sharpe" not in results_df.columns:
                raise ValueError(
                    "mean_oos_sharpe column is required for 'oos_sharpe_true' scoring strategy"
                )
            sort_columns = ["mean_oos_sharpe", "stability_score", "oos_sharpe_proxy", "mean_oos_ic"]
        else:
            sort_columns = ["mean_oos_ic", "stability_score"]
        logger.info("Sorting combos using strategy=%s with sort columns=%s", strategy, sort_columns)
        return results_df.sort_values(
            by=sort_columns, ascending=[False] * len(sort_columns)
        ).reset_index(drop=True)
