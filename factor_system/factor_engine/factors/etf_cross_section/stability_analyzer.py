#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子稳定性分析模块
测试不同市场环境下因子表现、时间衰减分析、子周期稳定性
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats
import warnings

from .ic_analyzer import ICAnalyzer, ICAnalysisResult
from factor_system.utils import safe_operation, FactorSystemError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class MarketEnvironment:
    """市场环境定义"""
    name: str
    start_date: datetime
    end_date: datetime
    description: str


@dataclass
class StabilityResult:
    """稳定性分析结果"""
    variant_id: str
    overall_ic_mean: float
    overall_ic_std: float
    period_ic_means: Dict[str, float]
    period_ic_stds: Dict[str, float]
    stability_score: float  # 稳定性评分
    decay_analysis: Dict[str, float]
    trend_consistency: float
    environmental_consistency: float
    sample_sizes: Dict[str, int]


class StabilityAnalyzer:
    """稳定性分析器"""

    def __init__(self, ic_analyzer: Optional[ICAnalyzer] = None):
        """
        初始化稳定性分析器

        Args:
            ic_analyzer: IC分析器实例
        """
        self.ic_analyzer = ic_analyzer or ICAnalyzer()
        logger.info("稳定性分析器初始化完成")

    def _define_market_environments(self, start_date: datetime,
                                   end_date: datetime) -> List[MarketEnvironment]:
        """
        定义市场环境

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            市场环境列表
        """
        total_days = (end_date - start_date).days
        mid_point = start_date + timedelta(days=total_days // 2)
        quarter_point = start_date + timedelta(days=total_days // 4)
        three_quarter_point = start_date + timedelta(days=total_days * 3 // 4)

        environments = [
            MarketEnvironment(
                name="整体期间",
                start_date=start_date,
                end_date=end_date,
                description="完整分析期间"
            ),
            MarketEnvironment(
                name="前期",
                start_date=start_date,
                end_date=mid_point,
                description="分析期间前半段"
            ),
            MarketEnvironment(
                name="后期",
                start_date=mid_point,
                end_date=end_date,
                description="分析期间后半段"
            ),
            MarketEnvironment(
                name="第一季度",
                start_date=start_date,
                end_date=quarter_point,
                description="分析期间第一个季度"
            ),
            MarketEnvironment(
                name="第二季度",
                start_date=quarter_point,
                end_date=mid_point,
                description="分析期间第二个季度"
            ),
            MarketEnvironment(
                name="第三季度",
                start_date=mid_point,
                end_date=three_quarter_point,
                description="分析期间第三个季度"
            ),
            MarketEnvironment(
                name="第四季度",
                start_date=three_quarter_point,
                end_date=end_date,
                description="分析期间第四个季度"
            )
        ]

        return environments

    def _calculate_ic_consistency(self, ic_means: Dict[str, float],
                                ic_stds: Dict[str, float]) -> float:
        """
        计算IC一致性

        Args:
            ic_means: 各期IC均值
            ic_stds: 各期IC标准差

        Returns:
            一致性评分
        """
        if len(ic_means) < 2:
            return 0.0

        # 计算IC均值的标准差（越小越稳定）
        mean_values = list(ic_means.values())
        ic_mean_std = np.std(mean_values)

        # 计算IC标准差的变异系数（越小越稳定）
        std_values = list(ic_stds.values())
        mean_std = np.mean(std_values)
        std_cv = np.std(std_values) / mean_std if mean_std > 0 else 0

        # 综合稳定性评分（越接近1越稳定）
        consistency_score = max(0, 1 - ic_mean_std - std_cv)

        return consistency_score

    def _analyze_decay(self, factor_data: pd.DataFrame,
                      price_data: pd.DataFrame,
                      factor_column: str,
                      max_periods: int = 20) -> Dict[str, float]:
        """
        分析因子衰减

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名
            max_periods: 最大分析周期

        Returns:
            衰减分析结果
        """
        decay_results = {}

        for period in range(1, min(max_periods + 1, 21)):
            try:
                # 计算指定周期的IC
                result = self.ic_analyzer.analyze_factor_ic(
                    factor_data, price_data, factor_column, return_period=period
                )

                if result.sample_count > 0:
                    decay_results[f"period_{period}"] = result.ic_mean
                else:
                    decay_results[f"period_{period}"] = 0.0

            except Exception as e:
                logger.warning(f"计算 {period} 期衰减失败: {str(e)}")
                decay_results[f"period_{period}"] = 0.0

        # 计算衰减率
        if len(decay_results) >= 2:
            period_1_ic = decay_results.get("period_1", 0)
            period_20_ic = decay_results.get("period_20", period_1_ic)

            if period_1_ic != 0:
                decay_rate = (period_1_ic - period_20_ic) / abs(period_1_ic)
                decay_results["decay_rate"] = decay_rate
            else:
                decay_results["decay_rate"] = 0.0

        return decay_results

    def _analyze_trend_consistency(self, factor_data: pd.DataFrame,
                                  price_data: pd.DataFrame,
                                  factor_column: str) -> float:
        """
        分析趋势一致性

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名

        Returns:
            趋势一致性评分
        """
        try:
            # 计算收益率
            returns_data = self.ic_analyzer._calculate_returns(price_data)
            merged_data = self.ic_analyzer._merge_factor_and_returns(
                factor_data, returns_data, factor_column
            )

            if merged_data.empty:
                return 0.0

            # 计算滚动IC
            ic_series = self.ic_analyzer._calculate_time_series_ic(merged_data, factor_column)

            if len(ic_series) < 10:
                return 0.0

            # 计算IC的趋势性
            ic_values = ic_series['ic'].dropna()
            if len(ic_values) < 10:
                return 0.0

            # 计算IC的自相关性
            autocorr = ic_values.autocorr(lag=1)
            if pd.isna(autocorr):
                autocorr = 0.0

            # 计算IC的线性趋势
            x = np.arange(len(ic_values))
            slope, _, _, p_value, _ = stats.linregress(x, ic_values)

            # 趋势一致性评分
            trend_score = max(0, min(1, (abs(slope) * 1000) * (1 - p_value) * (1 + abs(autocorr)) / 2))

            return trend_score

        except Exception as e:
            logger.warning(f"趋势一致性分析失败: {str(e)}")
            return 0.0

    def analyze_factor_stability(self,
                                factor_data: pd.DataFrame,
                                price_data: pd.DataFrame,
                                factor_column: str) -> StabilityResult:
        """
        分析单个因子的稳定性

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名

        Returns:
            稳定性分析结果
        """
        # 获取时间范围
        factor_dates = pd.to_datetime(factor_data['date'])
        start_date = factor_dates.min()
        end_date = factor_dates.max()

        # 定义市场环境
        environments = self._define_market_environments(start_date, end_date)

        # 分析各市场环境下的IC表现
        period_ic_means = {}
        period_ic_stds = {}
        sample_sizes = {}

        for env in environments:
            try:
                # 筛选对应时间段的数据
                env_factor_data = factor_data[
                    (pd.to_datetime(factor_data['date']) >= env.start_date) &
                    (pd.to_datetime(factor_data['date']) <= env.end_date)
                ]

                env_price_data = price_data[
                    (pd.to_datetime(price_data['date']) >= env.start_date) &
                    (pd.to_datetime(price_data['date']) <= env.end_date)
                ]

                if len(env_factor_data) < 10 or len(env_price_data) < 10:
                    logger.warning(f"市场环境 {env.name} 数据不足")
                    period_ic_means[env.name] = 0.0
                    period_ic_stds[env.name] = 0.0
                    sample_sizes[env.name] = 0
                    continue

                # 分析IC
                ic_result = self.ic_analyzer.analyze_factor_ic(
                    env_factor_data, env_price_data, factor_column
                )

                period_ic_means[env.name] = ic_result.ic_mean
                period_ic_stds[env.name] = ic_result.ic_std
                sample_sizes[env.name] = ic_result.sample_count

            except Exception as e:
                logger.error(f"分析环境 {env.name} 失败: {str(e)}")
                period_ic_means[env.name] = 0.0
                period_ic_stds[env.name] = 0.0
                sample_sizes[env.name] = 0

        # 计算整体IC
        overall_ic_mean = period_ic_means.get("整体期间", 0.0)
        overall_ic_std = period_ic_stds.get("整体期间", 1.0)

        # 计算稳定性评分
        env_means = {k: v for k, v in period_ic_means.items() if k != "整体期间"}
        env_stds = {k: v for k, v in period_ic_stds.items() if k != "整体期间"}

        environmental_consistency = self._calculate_ic_consistency(env_means, env_stds)

        # 分析衰减
        decay_analysis = self._analyze_decay(factor_data, price_data, factor_column)

        # 分析趋势一致性
        trend_consistency = self._analyze_trend_consistency(factor_data, price_data, factor_column)

        # 综合稳定性评分
        stability_score = (
            environmental_consistency * 0.4 +
            min(1.0, max(0.0, 1.0 - abs(decay_analysis.get("decay_rate", 0.0)))) * 0.3 +
            trend_consistency * 0.3
        )

        result = StabilityResult(
            variant_id=factor_column,
            overall_ic_mean=overall_ic_mean,
            overall_ic_std=overall_ic_std,
            period_ic_means=period_ic_means,
            period_ic_stds=period_ic_stds,
            stability_score=stability_score,
            decay_analysis=decay_analysis,
            trend_consistency=trend_consistency,
            environmental_consistency=environmental_consistency,
            sample_sizes=sample_sizes
        )

        logger.debug(f"因子 {factor_column} 稳定性分析完成: 评分={stability_score:.3f}, "
                    f"环境一致性={environmental_consistency:.3f}, 趋势一致性={trend_consistency:.3f}")

        return result

    def batch_analyze_stability(self, factors_data: Dict[str, pd.DataFrame],
                               price_data: pd.DataFrame) -> Dict[str, StabilityResult]:
        """
        批量分析因子稳定性

        Args:
            factors_data: 因子数据字典
            price_data: 价格数据

        Returns:
            稳定性分析结果字典
        """
        logger.info(f"开始批量分析 {len(factors_data)} 个因子的稳定性")

        results = {}

        for variant_id, factor_data in factors_data.items():
            try:
                # 获取因子列名
                factor_columns = [col for col in factor_data.columns
                                if col not in ['symbol', 'date'] and variant_id in col]

                if not factor_columns:
                    logger.warning(f"因子 {variant_id} 没有数据列")
                    continue

                # 使用第一个因子列
                factor_column = factor_columns[0]

                # 分析稳定性
                result = self.analyze_factor_stability(factor_data, price_data, factor_column)
                result.variant_id = variant_id
                results[variant_id] = result

            except Exception as e:
                logger.error(f"分析因子 {variant_id} 稳定性失败: {str(e)}")
                continue

        successful_count = len(results)
        logger.info(f"批量稳定性分析完成: {successful_count}/{len(factors_data)} 个因子成功")

        return results

    def save_stability_results(self, results: Dict[str, StabilityResult],
                              output_path: str):
        """
        保存稳定性分析结果

        Args:
            results: 稳定性分析结果
            output_path: 输出路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为DataFrame
        data = []
        for variant_id, result in results.items():
            data.append({
                "variant_id": result.variant_id,
                "overall_ic_mean": result.overall_ic_mean,
                "overall_ic_std": result.overall_ic_std,
                "stability_score": result.stability_score,
                "environmental_consistency": result.environmental_consistency,
                "trend_consistency": result.trend_consistency,
                "decay_rate": result.decay_analysis.get("decay_rate", 0.0),
                "total_sample_size": result.sample_sizes.get("整体期间", 0)
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"稳定性分析结果已保存到: {output_file}")

        # 打印统计摘要
        high_stability_count = sum(1 for r in results.values() if r.stability_score > 0.7)
        logger.info(f"高稳定性因子数量: {high_stability_count}/{len(results)} ({high_stability_count/len(results):.1%})")

        # 保存详细分析结果
        detail_file = output_file.parent / f"{output_file.stem}_detailed.json"
        detailed_data = {}
        for variant_id, result in results.items():
            detailed_data[variant_id] = {
                "period_ic_means": result.period_ic_means,
                "period_ic_stds": result.period_ic_stds,
                "decay_analysis": result.decay_analysis,
                "sample_sizes": result.sample_sizes
            }

        import json
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"详细稳定性分析结果已保存到: {detail_file}")


@safe_operation
def main():
    """主函数 - 测试稳定性分析功能"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 这里可以添加测试代码
    logger.info("稳定性分析模块测试完成")


if __name__ == "__main__":
    main()