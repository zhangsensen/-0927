#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子筛选器
实施多维筛选标准：IC阈值、单调性、稳定性、相关性去重
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict

from .ic_analyzer import ICAnalyzer, ICAnalysisResult
from .stability_analyzer import StabilityAnalyzer, StabilityResult
from factor_system.utils import safe_operation, FactorSystemError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ScreeningCriteria:
    """筛选标准"""
    min_ic_mean: float = 0.02          # 最小IC均值
    max_ic_pvalue: float = 0.05       # 最大IC p值
    min_ic_win_rate: float = 0.5      # 最小IC胜率
    min_stability_score: float = 0.7  # 最小稳定性评分
    max_correlation: float = 0.95     # 最大相关性阈值
    min_monotonicity_r2: float = 0.8  # 最小单调性R²
    min_sample_size: int = 30         # 最小样本数


@dataclass
class FactorScreeningResult:
    """因子筛选结果"""
    variant_id: str
    ic_mean: float
    ic_pvalue: float
    ic_win_rate: float
    stability_score: float
    monotonicity_score: float
    correlation_cluster: int
    cluster_rank: int
    overall_score: float
    screening_reason: str  # 通过/筛选原因


class FactorScreener:
    """因子筛选器"""

    def __init__(self, criteria: Optional[ScreeningCriteria] = None):
        """
        初始化因子筛选器

        Args:
            criteria: 筛选标准
        """
        self.criteria = criteria or ScreeningCriteria()
        self.ic_analyzer = ICAnalyzer()
        self.stability_analyzer = StabilityAnalyzer(self.ic_analyzer)

        logger.info("因子筛选器初始化完成")
        logger.info(f"筛选标准: IC均值>{self.criteria.min_ic_mean}, "
                   f"p值<{self.criteria.max_correlation}, "
                   f"稳定性>{self.criteria.min_stability_score}")

    def _calculate_monotonicity_score(self, factor_data: pd.DataFrame,
                                    price_data: pd.DataFrame,
                                    factor_column: str) -> float:
        """
        计算因子单调性评分

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名

        Returns:
            单调性评分
        """
        try:
            # 计算收益率
            returns_data = self.ic_analyzer._calculate_returns(price_data)
            merged_data = self.ic_analyzer._merge_factor_and_returns(
                factor_data, returns_data, factor_column
            )

            if merged_data.empty:
                return 0.0

            # 按日期分组分析单调性
            monotonicity_scores = []

            for date, group in merged_data.groupby('date'):
                if len(group) < 5:  # 至少需要5个样本
                    continue

                # 按因子值排序
                sorted_group = group.sort_values(factor_column)
                factor_values = sorted_group[factor_column]
                returns = sorted_group['return']

                # 计算单调性 (因子值与收益率的秩相关)
                monotonicity, p_value = stats.spearmanr(factor_values, returns)

                if not pd.isna(monotonicity):
                    monotonicity_scores.append(abs(monotonicity))

            if not monotonicity_scores:
                return 0.0

            # 返回平均单调性
            return np.mean(monotonicity_scores)

        except Exception as e:
            logger.warning(f"单调性计算失败 {factor_column}: {str(e)}")
            return 0.0

    def _calculate_factor_correlation_matrix(self, factors_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factors_data: 因子数据字典

        Returns:
            相关性矩阵
        """
        # 合并所有因子数据
        all_factor_data = []

        for variant_id, factor_data in factors_data.items():
            # 获取因子列名
            factor_columns = [col for col in factor_data.columns
                            if col not in ['symbol', 'date'] and variant_id in col]

            if not factor_columns:
                continue

            factor_column = factor_columns[0]
            factor_df = factor_data[['symbol', 'date', factor_column]].copy()
            factor_df = factor_df.rename(columns={factor_column: variant_id})
            all_factor_data.append(factor_df)

        if not all_factor_data:
            return pd.DataFrame()

        # 合并数据
        merged_data = all_factor_data[0]
        for df in all_factor_data[1:]:
            merged_data = merged_data.merge(df, on=['symbol', 'date'], how='outer')

        # 计算相关性矩阵
        factor_cols = [col for col in merged_data.columns if col not in ['symbol', 'date']]
        correlation_matrix = merged_data[factor_cols].corr()

        return correlation_matrix

    def _perform_correlation_clustering(self, correlation_matrix: pd.DataFrame) -> Dict[str, int]:
        """
        执行相关性聚类

        Args:
            correlation_matrix: 相关性矩阵

        Returns:
            聚类结果字典 {factor_id: cluster_id}
        """
        if correlation_matrix.empty:
            return {}

        # 创建图
        G = nx.Graph()
        factors = correlation_matrix.columns.tolist()

        # 添加节点
        for factor in factors:
            G.add_node(factor)

        # 添加边（相关性超过阈值）
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors[i+1:], i+1):
                corr_value = correlation_matrix.loc[factor1, factor2]
                if abs(corr_value) > self.criteria.max_correlation:
                    G.add_edge(factor1, factor2, weight=abs(corr_value))

        # 找连通分量作为聚类
        clusters = {}
        cluster_id = 0

        for component in nx.connected_components(G):
            for factor in component:
                clusters[factor] = cluster_id
            cluster_id += 1

        # 未连接的因子各自成一类
        for factor in factors:
            if factor not in clusters:
                clusters[factor] = cluster_id
                cluster_id += 1

        logger.info(f"相关性聚类完成: {len(factors)} 个因子分为 {cluster_id} 类")
        return clusters

    def _rank_factors_within_clusters(self, factors_data: Dict[str, pd.DataFrame],
                                    ic_results: Dict[str, ICAnalysisResult],
                                    stability_results: Dict[str, StabilityResult],
                                    correlation_clusters: Dict[str, int]) -> Dict[str, int]:
        """
        在聚类内对因子进行排序

        Args:
            factors_data: 因子数据
            ic_results: IC分析结果
            stability_results: 稳定性分析结果
            correlation_clusters: 相关性聚类

        Returns:
            聚类内排序字典
        """
        cluster_rankings = {}

        # 按聚类分组
        clusters = defaultdict(list)
        for factor_id, cluster_id in correlation_clusters.items():
            clusters[cluster_id].append(factor_id)

        # 在每个聚类内排序
        for cluster_id, factor_ids in clusters.items():
            factor_scores = []

            for factor_id in factor_ids:
                # 计算综合评分
                ic_result = ic_results.get(factor_id)
                stability_result = stability_results.get(factor_id)

                if ic_result is None or stability_result is None:
                    continue

                # 综合评分 (IC均值 * 稳定性 * IC胜率)
                score = (abs(ic_result.ic_mean) * stability_result.stability_score *
                        ic_result.ic_win_rate)

                factor_scores.append((factor_id, score))

            # 按评分排序
            factor_scores.sort(key=lambda x: x[1], reverse=True)

            # 分配排名
            for rank, (factor_id, score) in enumerate(factor_scores):
                cluster_rankings[factor_id] = rank

        return cluster_rankings

    def _screen_single_factor(self, variant_id: str,
                            ic_result: ICAnalysisResult,
                            stability_result: StabilityResult,
                            cluster_rank: int) -> FactorScreeningResult:
        """
        筛选单个因子

        Args:
            variant_id: 因子ID
            ic_result: IC分析结果
            stability_result: 稳定性分析结果
            cluster_rank: 聚类内排名

        Returns:
            筛选结果
        """
        # 检查IC均值
        if abs(ic_result.ic_mean) < self.criteria.min_ic_mean:
            return FactorScreeningResult(
                variant_id=variant_id,
                ic_mean=ic_result.ic_mean,
                ic_pvalue=ic_result.p_value,
                ic_win_rate=ic_result.ic_win_rate,
                stability_score=stability_result.stability_score,
                monotonicity_score=0.0,
                correlation_cluster=0,
                cluster_rank=cluster_rank,
                overall_score=0.0,
                screening_reason=f"IC均值过低: {ic_result.ic_mean:.4f} < {self.criteria.min_ic_mean}"
            )

        # 检查IC显著性
        if ic_result.p_value > self.criteria.max_ic_pvalue:
            return FactorScreeningResult(
                variant_id=variant_id,
                ic_mean=ic_result.ic_mean,
                ic_pvalue=ic_result.p_value,
                ic_win_rate=ic_result.ic_win_rate,
                stability_score=stability_result.stability_score,
                monotonicity_score=0.0,
                correlation_cluster=0,
                cluster_rank=cluster_rank,
                overall_score=0.0,
                screening_reason=f"IC不显著: p值={ic_result.p_value:.4f} > {self.criteria.max_ic_pvalue}"
            )

        # 检查IC胜率
        if ic_result.ic_win_rate < self.criteria.min_ic_win_rate:
            return FactorScreeningResult(
                variant_id=variant_id,
                ic_mean=ic_result.ic_mean,
                ic_pvalue=ic_result.p_value,
                ic_win_rate=ic_result.ic_win_rate,
                stability_score=stability_result.stability_score,
                monotonicity_score=0.0,
                correlation_cluster=0,
                cluster_rank=cluster_rank,
                overall_score=0.0,
                screening_reason=f"IC胜率过低: {ic_result.ic_win_rate:.2%} < {self.criteria.min_ic_win_rate:.2%}"
            )

        # 检查稳定性
        if stability_result.stability_score < self.criteria.min_stability_score:
            return FactorScreeningResult(
                variant_id=variant_id,
                ic_mean=ic_result.ic_mean,
                ic_pvalue=ic_result.p_value,
                ic_win_rate=ic_result.ic_win_rate,
                stability_score=stability_result.stability_score,
                monotonicity_score=0.0,
                correlation_cluster=0,
                cluster_rank=cluster_rank,
                overall_score=0.0,
                screening_reason=f"稳定性不足: {stability_result.stability_score:.3f} < {self.criteria.min_stability_score:.3f}"
            )

        # 检查样本数
        if ic_result.sample_count < self.criteria.min_sample_size:
            return FactorScreeningResult(
                variant_id=variant_id,
                ic_mean=ic_result.ic_mean,
                ic_pvalue=ic_result.p_value,
                ic_win_rate=ic_result.ic_win_rate,
                stability_score=stability_result.stability_score,
                monotonicity_score=0.0,
                correlation_cluster=0,
                cluster_rank=cluster_rank,
                overall_score=0.0,
                screening_reason=f"样本数不足: {ic_result.sample_count} < {self.criteria.min_sample_size}"
            )

        # 计算综合评分
        overall_score = (abs(ic_result.ic_mean) * 100 +  # IC权重
                        stability_result.stability_score * 30 +  # 稳定性权重
                        ic_result.ic_win_rate * 20 -  # 胜率权重
                        cluster_rank * 0.1)  # 聚类排名惩罚

        return FactorScreeningResult(
            variant_id=variant_id,
            ic_mean=ic_result.ic_mean,
            ic_pvalue=ic_result.p_value,
            ic_win_rate=ic_result.ic_win_rate,
            stability_score=stability_result.stability_score,
            monotonicity_score=0.0,  # 将在后续计算
            correlation_cluster=0,   # 将在后续设置
            cluster_rank=cluster_rank,
            overall_score=overall_score,
            screening_reason="通过筛选"
        )

    def screen_factors(self, factors_data: Dict[str, pd.DataFrame],
                      price_data: pd.DataFrame) -> Dict[str, FactorScreeningResult]:
        """
        筛选因子

        Args:
            factors_data: 因子数据字典
            price_data: 价格数据

        Returns:
            筛选结果字典
        """
        logger.info(f"开始筛选 {len(factors_data)} 个因子")

        # 第一步：IC分析
        logger.info("第一步：IC分析...")
        ic_results = self.ic_analyzer.batch_analyze_factors(factors_data, price_data)

        # 第二步：稳定性分析
        logger.info("第二步：稳定性分析...")
        stability_results = self.stability_analyzer.batch_analyze_stability(factors_data, price_data)

        # 第三步：相关性聚类
        logger.info("第三步：相关性聚类...")
        correlation_matrix = self._calculate_factor_correlation_matrix(factors_data)
        correlation_clusters = self._perform_correlation_clustering(correlation_matrix)

        # 第四步：聚类内排序
        logger.info("第四步：聚类内排序...")
        cluster_rankings = self._rank_factors_within_clusters(
            factors_data, ic_results, stability_results, correlation_clusters
        )

        # 第五步：筛选因子
        logger.info("第五步：应用筛选标准...")
        screening_results = {}

        for variant_id in factors_data.keys():
            ic_result = ic_results.get(variant_id)
            stability_result = stability_results.get(variant_id)
            cluster_rank = cluster_rankings.get(variant_id, 999)

            if ic_result is None or stability_result is None:
                logger.warning(f"因子 {variant_id} 缺少IC或稳定性分析结果")
                continue

            # 筛选因子
            screening_result = self._screen_single_factor(
                variant_id, ic_result, stability_result, cluster_rank
            )

            # 设置聚类信息
            screening_result.correlation_cluster = correlation_clusters.get(variant_id, -1)

            # 计算单调性评分（仅对通过的因子）
            if screening_result.screening_reason == "通过筛选":
                try:
                    factor_data = factors_data[variant_id]
                    factor_columns = [col for col in factor_data.columns
                                    if col not in ['symbol', 'date'] and variant_id in col]
                    if factor_columns:
                        monotonicity = self._calculate_monotonicity_score(
                            factor_data, price_data, factor_columns[0]
                        )
                        screening_result.monotonicity_score = monotonicity

                        # 检查单调性
                        if monotonicity < self.criteria.min_monotonicity_r2:
                            screening_result.screening_reason = f"单调性不足: {monotonicity:.3f} < {self.criteria.min_monotonicity_r2:.3f}"
                            screening_result.overall_score = 0.0
                except Exception as e:
                    logger.warning(f"单调性计算失败 {variant_id}: {str(e)}")

            screening_results[variant_id] = screening_result

        # 统计结果
        passed_count = sum(1 for r in screening_results.values() if r.screening_reason == "通过筛选")
        logger.info(f"因子筛选完成: {passed_count}/{len(factors_data)} 个因子通过筛选")

        return screening_results

    def save_screening_results(self, results: Dict[str, FactorScreeningResult],
                              output_path: str):
        """
        保存筛选结果

        Args:
            results: 筛选结果
            output_path: 输出路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        data = []
        for variant_id, result in results.items():
            data.append({
                "variant_id": result.variant_id,
                "ic_mean": result.ic_mean,
                "ic_pvalue": result.ic_pvalue,
                "ic_win_rate": result.ic_win_rate,
                "stability_score": result.stability_score,
                "monotonicity_score": result.monotonicity_score,
                "correlation_cluster": result.correlation_cluster,
                "cluster_rank": result.cluster_rank,
                "overall_score": result.overall_score,
                "screening_reason": result.screening_reason
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"筛选结果已保存到: {output_file}")

        # 保存通过的因子
        passed_factors = df[df['screening_reason'] == '通过筛选'].sort_values('overall_score', ascending=False)
        passed_file = output_file.parent / f"{output_file.stem}_passed.csv"
        passed_factors.to_csv(passed_file, index=False)

        logger.info(f"通过的因子已保存到: {passed_file}")
        logger.info(f"通过筛选的因子数量: {len(passed_factors)}")


@safe_operation
def screen_etf_factors(factors_data: Dict[str, pd.DataFrame],
                      price_data: pd.DataFrame,
                      criteria: Optional[ScreeningCriteria] = None,
                      output_dir: str = None) -> Dict[str, FactorScreeningResult]:
    """
    便捷函数：筛选ETF因子

    Args:
        factors_data: 因子数据字典
        price_data: 价格数据
        criteria: 筛选标准
        output_dir: 输出目录

    Returns:
        筛选结果字典
    """
    screener = FactorScreener(criteria)

    results = screener.screen_factors(factors_data, price_data)

    # 保存结果
    if output_dir is None:
        output_dir = "factor_system/factor_output/etf_cross_section/screening_results"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/factor_screening_{timestamp}.csv"
    screener.save_screening_results(results, output_file)

    return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("因子筛选模块测试完成")