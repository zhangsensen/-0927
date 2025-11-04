#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子有效性检验和筛选模块
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class FactorValidator:
    """因子有效性检验器"""

    def __init__(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        close_df: pd.DataFrame,
        logger: logging.Logger = None,
    ):
        self.factors_dict = factors_dict
        self.close_df = close_df
        self.logger = logger or logging.getLogger(__name__)
        self.returns = close_df.pct_change().iloc[1:]  # 收益率
        self.selection_trace: Dict[str, object] = {}

    def calculate_ic_series(self, factor_name: str, lag: int = 1) -> pd.Series:
        """计算单个因子的IC序列"""
        factor_data = self.factors_dict[factor_name].iloc[:-lag]  # 滞后处理
        forward_returns = self.returns.shift(-lag).iloc[lag:]  # 未来收益

        # 对齐数据
        common_index = factor_data.index.intersection(forward_returns.index)
        factor_data = factor_data.loc[common_index]
        forward_returns = forward_returns.loc[common_index]

        # 计算IC（按日期）
        ic_series = pd.Series(index=common_index, dtype=float)

        for date in common_index:
            factor_values = factor_data.loc[date]
            future_returns = forward_returns.loc[date]

            # 去除NaN
            valid_mask = ~(factor_values.isna() | future_returns.isna())
            if valid_mask.sum() > 5:  # 至少5个有效数据点
                ic_series[date] = factor_values[valid_mask].corr(
                    future_returns[valid_mask]
                )

        return ic_series

    def validate_all_factors(
        self, lag: int = 1, significance_threshold: float = 0.02
    ) -> Dict[str, Dict]:
        """检验所有因子的有效性"""
        factor_stats = {}

        for factor_name in self.factors_dict.keys():
            try:
                ic_series = self.calculate_ic_series(factor_name, lag)

                if len(ic_series) > 0:
                    ic_mean = float(ic_series.mean())
                    ic_std = float(ic_series.std())
                    if np.isnan(ic_std):
                        ic_std = 0.0
                    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
                    n_obs = len(ic_series)
                    t_stat = 0.0
                    p_value = 1.0
                    if ic_std > 0 and n_obs > 2:
                        t_stat = ic_mean / (ic_std / np.sqrt(n_obs))
                        # 双尾检验
                        p_value = float(2 * sp_stats.t.sf(np.abs(t_stat), df=n_obs - 1))

                    stats = {
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_ir": ic_ir,
                        "ic_positive_rate": float((ic_series > 0).mean()),
                        "ic_significant_rate": float(
                            (ic_series.abs() > significance_threshold).mean()
                        ),
                        "n_observations": int(n_obs),
                        "t_stat": float(t_stat),
                        "p_value": float(p_value),
                    }
                    factor_stats[factor_name] = stats

                    self.logger.info(
                        f"因子 {factor_name}: IC均值={stats['ic_mean']:.4f}, IR={stats['ic_ir']:.4f}, "
                        f"显著率={stats['ic_significant_rate']:.2%}"
                    )
                else:
                    self.logger.warning(f"因子 {factor_name}: 无有效IC数据")

            except Exception as e:
                self.logger.error(f"因子 {factor_name} 检验失败: {e}")

        return factor_stats

    def select_best_factors(
        self,
        factor_stats: Dict[str, Dict],
        min_ir: float = 0.03,
        min_positive_rate: float = 0.55,
        min_significant_rate: float = 0.1,
        min_abs_ic: float = 0.015,
        min_observations: int = 80,
        max_correlation: float = 0.8,
        fdr_q: float = 0.1,
        fallback_top_k: int = 20,
    ) -> List[str]:
        """筛选最佳因子集合"""
        total_factors = len(factor_stats)
        trace = {
            "total_factors": total_factors,
            "min_ir": min_ir,
            "min_positive_rate": min_positive_rate,
            "min_significant_rate": min_significant_rate,
            "min_abs_ic": min_abs_ic,
            "min_observations": min_observations,
            "fdr_q": fdr_q,
            "max_correlation": max_correlation,
            "fallback_top_k": fallback_top_k,
        }

        # 第一步：基于IC指标筛选
        valid_factors = []
        for factor_name, stats in factor_stats.items():
            if stats.get("n_observations", 0) < min_observations:
                continue
            if abs(stats.get("ic_mean", 0.0)) < min_abs_ic:
                continue
            if abs(stats.get("ic_ir", 0.0)) < min_ir:
                continue
            if stats.get("ic_positive_rate", 0.0) < min_positive_rate:
                continue
            if stats.get("ic_significant_rate", 0.0) < min_significant_rate:
                continue
            valid_factors.append(factor_name)

        trace["initial_pass"] = len(valid_factors)
        self.logger.info(f"初选因子: {len(valid_factors)}/{total_factors}")

        # 第二步：Benjamini-Hochberg FDR过滤
        if fdr_q is not None and fdr_q > 0 and valid_factors:
            p_values = pd.Series(
                {
                    name: factor_stats[name].get("p_value", np.nan)
                    for name in valid_factors
                }
            )
            p_values = p_values.dropna()
            if not p_values.empty:
                fdr_selected = self._apply_bh_fdr(p_values, fdr_q)
                valid_factors = [name for name in valid_factors if name in fdr_selected]

        trace["fdr_pass"] = len(valid_factors)
        self.logger.info(f"FDR过滤后因子: {len(valid_factors)}")

        fallback_used = False
        if not valid_factors:
            fallback_used = True
            fallback_pool = [
                name
                for name, stats in factor_stats.items()
                if stats.get("n_observations", 0) >= max(20, min_observations // 2)
            ] or list(factor_stats.keys())
            sorted_fallback = sorted(
                fallback_pool,
                key=lambda name: (
                    abs(factor_stats[name].get("ic_mean", 0.0)),
                    factor_stats[name].get("ic_positive_rate", 0.0),
                ),
                reverse=True,
            )
            valid_factors = sorted_fallback[:fallback_top_k]
            self.logger.warning(
                "FDR过滤后无因子满足要求，启用回退策略，保留前%d个候选",
                len(valid_factors),
            )

        trace["fallback_used"] = fallback_used
        trace["fallback_candidates"] = list(valid_factors) if fallback_used else []

        # 第三步：相关性去重
        if len(valid_factors) <= 1:
            selected_factors = list(valid_factors)
        else:
            factor_correlations = self._calculate_factor_correlations(valid_factors)
            selected_factors = []

            for factor in valid_factors:
                too_correlated = False
                for selected in selected_factors:
                    corr = abs(factor_correlations.loc[factor, selected])
                    if corr > max_correlation:
                        too_correlated = True
                        break

                if not too_correlated:
                    selected_factors.append(factor)

        trace["final_count"] = len(selected_factors)
        trace["final_factors"] = list(selected_factors)
        self.selection_trace = trace

        self.logger.info(
            "最终选择因子: %d/%d (FDR后:%d, 回退:%s)",
            len(selected_factors),
            total_factors,
            trace.get("fdr_pass", 0),
            "是" if fallback_used else "否",
        )

        return selected_factors

    def _calculate_factor_correlations(self, factor_names: List[str]) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        if not factor_names:
            return pd.DataFrame()
        # 对齐所有因子数据并拉平为列向量
        flattened = {}
        for name in factor_names:
            factor_values = self.factors_dict[name].values.reshape(-1)
            flattened[name] = factor_values

        factor_df = pd.DataFrame(flattened)
        correlation_matrix = factor_df.corr().fillna(0.0)

        return correlation_matrix

    @staticmethod
    def _apply_bh_fdr(p_values: pd.Series, q: float) -> List[str]:
        """应用Benjamini-Hochberg FDR过滤"""
        if p_values.empty:
            return list(p_values.index)

        sorted_p = p_values.sort_values()
        m = len(sorted_p)
        ranks = np.arange(1, m + 1)
        thresholds = (ranks / m) * q
        passed = sorted_p.values <= thresholds

        if not np.any(passed):
            return []

        max_idx = np.max(np.where(passed)[0])
        return list(sorted_p.index[: max_idx + 1])

    def get_factor_directions(self, factor_stats: Dict[str, Dict]) -> Dict[str, float]:
        """确定因子方向性"""
        directions = {}
        for factor_name, stats in factor_stats.items():
            # IC均值为正：正向因子；为负：负向因子
            directions[factor_name] = 1.0 if stats["ic_mean"] > 0 else -1.0

        return directions
