"""
因子筛选器 | Factor Selector
================================================================================
Layer 4: 预组合阶段因子筛选。

步骤:
  1. 质量过滤: 去掉 quality_score < 阈值 或 passed=False
  2. 层次聚类去冗余: 截面 Spearman 相关矩阵 → hierarchical clustering
     → 同簇因子只保留质量评分最高的
  3. 上限约束: max_factors
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .quality import FactorQualityReport
from .registry import FactorEntry

logger = logging.getLogger(__name__)


class FactorSelector:
    """
    因子筛选器

    通过质量过滤 + 层次聚类去冗余，选出最终精选因子池。
    """

    def __init__(
        self,
        max_correlation: float = 0.7,
        min_quality_score: float = 2.0,
        max_factors: int = 40,
    ):
        """
        参数:
            max_correlation: 聚类阈值，|corr| > 此值的因子归入同簇
            min_quality_score: 最低质量评分
            max_factors: 最大因子数量
        """
        self.max_correlation = max_correlation
        self.min_quality_score = min_quality_score
        self.max_factors = max_factors

    def select(
        self,
        entries: List[FactorEntry],
        reports: Dict[str, FactorQualityReport],
        factors_dict: Dict[str, pd.DataFrame],
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        执行因子筛选。

        参数:
            entries: 所有因子条目
            reports: {factor_name: QualityReport}
            factors_dict: {factor_name: DataFrame}

        返回:
            (selected_names, correlation_matrix)
        """
        # Step 1: 质量过滤
        quality_passed = []
        for entry in entries:
            report = reports.get(entry.name)
            if report is None:
                continue
            if not report.passed:
                continue
            if report.quality_score < self.min_quality_score:
                continue
            if entry.name not in factors_dict:
                continue
            quality_passed.append(entry.name)

        logger.info("Quality filter: %d / %d passed", len(quality_passed), len(entries))

        if len(quality_passed) <= 1:
            return quality_passed, pd.DataFrame()

        # Step 2: 计算截面相关矩阵
        corr_matrix = self._compute_correlation_matrix(
            {name: factors_dict[name] for name in quality_passed}
        )

        # Step 3: 层次聚类去冗余
        selected = self._cluster_and_select(
            quality_passed, corr_matrix, reports
        )

        # Step 4: 上限约束 (按质量评分降序截断)
        if len(selected) > self.max_factors:
            selected = sorted(
                selected,
                key=lambda n: reports[n].quality_score,
                reverse=True,
            )[:self.max_factors]
            selected = sorted(selected)  # 恢复字母序

        logger.info("Final selection: %d factors", len(selected))
        return selected, corr_matrix

    @staticmethod
    def _compute_correlation_matrix(
        factors_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        计算因子间截面 Spearman 相关矩阵（向量化）。

        方法: 将每个因子的截面 rank 展平为 (T*N,) 向量，
        拼成 (T*N, n_factors) 矩阵，一次 np.corrcoef 完成。
        """
        names = sorted(factors_dict.keys())
        n = len(names)

        # 截面 rank 后展平为列向量
        rank_dfs = {}
        for name in names:
            rank_dfs[name] = factors_dict[name].rank(axis=1, pct=True)

        # 对齐 index
        common_idx = rank_dfs[names[0]].index
        for name in names[1:]:
            common_idx = common_idx.intersection(rank_dfs[name].index)

        # 拼成 (T*N, n_factors) 矩阵，NaN 用 0 填充（rank 后 NaN 很少）
        flat_cols = []
        for name in names:
            flat_cols.append(rank_dfs[name].loc[common_idx].values.ravel())

        mat = np.column_stack(flat_cols)  # (T*N, n_factors)

        # 用 pandas corr 自动处理 NaN (pairwise complete)
        corr_df = pd.DataFrame(mat, columns=names).corr(method="spearman")

        # 保底: 对角线=1, NaN→0
        corr_vals = corr_df.values
        np.fill_diagonal(corr_vals, 1.0)
        corr_vals = np.nan_to_num(corr_vals, nan=0.0)

        return pd.DataFrame(corr_vals, index=names, columns=names)

    def _cluster_and_select(
        self,
        names: List[str],
        corr_matrix: pd.DataFrame,
        reports: Dict[str, FactorQualityReport],
    ) -> List[str]:
        """层次聚类去冗余，每簇保留质量最高的因子"""
        n = len(names)
        if n <= 1:
            return names

        # 距离矩阵: 1 - |corr|
        abs_corr = corr_matrix.loc[names, names].values
        np.fill_diagonal(abs_corr, 1.0)
        dist = 1 - np.abs(abs_corr)

        # 确保距离矩阵对称且非负
        dist = np.maximum(dist, 0.0)
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)

        # condensed distance
        condensed = squareform(dist, checks=False)

        # 层次聚类
        Z = linkage(condensed, method="average")
        threshold = 1 - self.max_correlation  # e.g., 0.3
        clusters = fcluster(Z, t=threshold, criterion="distance")

        # 每簇保留质量评分最高的
        cluster_best = {}
        for i, (name, cluster_id) in enumerate(zip(names, clusters)):
            score = reports[name].quality_score if name in reports else 0.0
            if cluster_id not in cluster_best or score > cluster_best[cluster_id][1]:
                cluster_best[cluster_id] = (name, score)

        selected = sorted([v[0] for v in cluster_best.values()])
        logger.info("Clustering: %d clusters from %d factors (threshold=%.2f)",
                     len(cluster_best), n, self.max_correlation)
        return selected
