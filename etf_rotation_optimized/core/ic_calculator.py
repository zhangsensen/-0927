"""
IC 计算器兼容层 | IC Calculator Compatibility Wrapper

使用Numba加速的计算内核，保持旧API兼容性
仅实现Walk Forward Optimizer需要的接口

作者: Compatibility Layer
日期: 2025-01-17
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

# ICCalculatorNumba 已废弃（向量化实现已内置）


@dataclass
class ICStats:
    """IC 统计数据"""

    mean: float
    std: float
    ir: float
    t_stat: float
    p_value: float
    sharpe: float
    n_obs: int
    min: float
    max: float
    median: float
    skew: float
    kurtosis: float


class ICCalculator:
    """
    IC 计算器兼容包装器

    使用Numba加速后端，提供WalkForwardOptimizer需要的旧API
    """

    def __init__(self, verbose: bool = True):
        """初始化"""
        self.verbose = verbose
        self.ic_data = {}
        self.ic_stats = {}

    def compute_ic(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        method: str = "pearson",
        forward_periods: int = 1,
    ) -> Dict[str, pd.Series]:
        """
        计算 IC 时间序列

        参数:
            factors_dict: {factor_name: DataFrame(date × symbol)}
            returns_df: DataFrame(date × symbol)
            method: 'pearson' | 'spearman' (Numba只支持spearman)
            forward_periods: 前向天数

        返回:
            {factor_name: Series(date)} IC时间序列
        """
        if method not in {"pearson", "spearman"}:
            raise ValueError(f"不支持的方法: {method}")

        return self._compute_ic_vectorized(
            factors_dict=factors_dict,
            returns_df=returns_df,
            method=method,
            forward_periods=forward_periods,
        )

    def _compute_ic_vectorized(
        self,
        factors_dict: Dict[str, pd.DataFrame],
        returns_df: pd.DataFrame,
        method: str,
        forward_periods: int,
    ) -> Dict[str, pd.Series]:
        """
        彻底向量化的横截面IC计算（逐日、逐因子），消除日期循环

        实现:
        - pearson: 行内标准化后逐列相关 = nanmean(zx * zy)
        - spearman: 先做行内秩变换(近似处理ties)，再按pearson流程
        - 满窗与NaN: 使用nanmean/nanstd自动忽略NaN；有效样本<10的日子记为NaN
        """
        eps = 1e-12

        # 对齐前视期: 与旧逻辑一致，date的IC使用 forward_periods-1 天后的收益
        if forward_periods < 1:
            raise ValueError("forward_periods 必须>=1")
        fwd = forward_periods - 1
        ret_aligned = returns_df.shift(-fwd)

        # 组装三维因子矩阵 (F, T, N)
        factor_names = list(factors_dict.keys())
        if len(factor_names) == 0:
            return {}

        # 校验形状一致
        T, N = ret_aligned.shape
        factors_arr = np.stack(
            [
                factors_dict[name].to_numpy(dtype=float, copy=False)
                for name in factor_names
            ],
            axis=0,
        )  # (F, T, N)
        ret_arr = ret_aligned.to_numpy(dtype=float, copy=False)  # (T, N)

        if factors_arr.shape[1:] != ret_arr.shape:
            raise ValueError("factors 与 returns 形状不一致")

        F = factors_arr.shape[0]

        # spearman需要秩变换（按行/日对列进行排名）
        def _rowwise_rank_2d(a2d: np.ndarray) -> np.ndarray:
            """对形状(T, N)矩阵逐行做秩排名，NaN保留为NaN（ties使用序号秩近似）。"""
            T2, N2 = a2d.shape
            mask = np.isnan(a2d)
            filled = a2d.copy()
            filled[mask] = np.inf  # NaN放到末尾
            order = np.argsort(filled, axis=1, kind="mergesort")  # 稳定排序
            ranks = np.empty_like(order, dtype=float)
            row_idx = np.arange(T2)[:, None]
            ranks[row_idx, order] = np.arange(N2)[None, :]
            ranks[mask] = np.nan
            return ranks

        if method == "spearman":
            # 对每个因子逐行秩排名: 将(F,T,N) reshape 成 (F*T, N) 批量处理
            a2 = factors_arr.reshape(-1, N)
            ranks_a2 = _rowwise_rank_2d(a2)
            factors_arr = ranks_a2.reshape(F, T, N)

            # returns 只需做一次秩排名后在F维上重复
            ret_rank = _rowwise_rank_2d(ret_arr)
            ret_arr = np.broadcast_to(ret_rank[None, :, :], (F, T, N))
        else:
            # pearson: 直接使用原值，broadcast returns 到F维
            ret_arr = np.broadcast_to(ret_arr[None, :, :], (F, T, N))

        # 有效性掩码（两者均非NaN）
        valid_mask = (~np.isnan(factors_arr)) & (~np.isnan(ret_arr))  # (F,T,N)

        # 行内标准化（横截面Z-score）
        sig_mean = np.nanmean(factors_arr, axis=2, keepdims=True)
        sig_std = np.nanstd(factors_arr, axis=2, keepdims=True)
        ret_mean = np.nanmean(ret_arr, axis=2, keepdims=True)
        ret_std = np.nanstd(ret_arr, axis=2, keepdims=True)

        sig_norm = (factors_arr - sig_mean) / (sig_std + eps)
        ret_norm = (ret_arr - ret_mean) / (ret_std + eps)

        # 逐日横截面相关（皮尔逊或秩相关的皮尔逊等价）
        prod = sig_norm * ret_norm
        ic_matrix = np.nanmean(prod, axis=2)  # (F, T)

        # 有效样本数与std约束（防止除零/伪相关）
        valid_count = np.sum(valid_mask, axis=2)  # (F, T)
        sig_std2d = sig_std.squeeze(-1)  # (F, T)
        ret_std2d = ret_std.squeeze(-1)  # (F, T)
        ic_matrix = np.where(
            (valid_count >= 10) & (sig_std2d > eps) & (ret_std2d > eps),
            ic_matrix,
            np.nan,
        )

        # 返回 {factor: Series} - 零循环组装
        ic_df = pd.DataFrame(ic_matrix.T, index=returns_df.index, columns=factor_names)
        ic_dict = ic_df.to_dict("series")

        self.ic_data = ic_dict
        return ic_dict

    def compute_all_ic_stats(self) -> Dict[str, ICStats]:
        """
        计算所有因子的IC统计量

        返回:
            {factor_name: ICStats}
        """
        stats_dict = {}

        for factor_name, ic_series in self.ic_data.items():
            if len(ic_series) == 0:
                # 无数据
                stats_dict[factor_name] = ICStats(
                    mean=0.0,
                    std=0.0,
                    ir=0.0,
                    t_stat=0.0,
                    p_value=1.0,
                    sharpe=0.0,
                    n_obs=0,
                    min=0.0,
                    max=0.0,
                    median=0.0,
                    skew=0.0,
                    kurtosis=0.0,
                )
                continue

            ic_values = ic_series.values
            n = len(ic_values)

            mean_ic = np.mean(ic_values)
            std_ic = np.std(ic_values, ddof=1) if n > 1 else 0.0
            ir = mean_ic / std_ic if std_ic > 0 else 0.0

            # t检验
            if n > 1:
                t_stat = mean_ic / (std_ic / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
            else:
                t_stat = 0.0
                p_value = 1.0

            stats_dict[factor_name] = ICStats(
                mean=mean_ic,
                std=std_ic,
                ir=ir,
                t_stat=t_stat,
                p_value=p_value,
                sharpe=ir,  # IC Sharpe = IR
                n_obs=n,
                min=np.min(ic_values),
                max=np.max(ic_values),
                median=np.median(ic_values),
                skew=stats.skew(ic_values) if n > 2 else 0.0,
                kurtosis=stats.kurtosis(ic_values) if n > 3 else 0.0,
            )

        self.ic_stats = stats_dict
        return stats_dict
