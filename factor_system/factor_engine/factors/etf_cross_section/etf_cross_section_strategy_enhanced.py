#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面策略引擎增强版
扩展原有策略系统，支持动态因子选择和智能因子权重分配
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from scipy import stats
import warnings

from .etf_cross_section_strategy import ETFCrossSectionStrategy, StrategyConfig
from .etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced
from .factor_registry import get_factor_registry, FactorCategory

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedStrategyConfig(StrategyConfig):
    """增强策略配置"""
    # 动态因子配置
    enable_dynamic_factors: bool = True
    factor_selection_method: str = "auto"  # auto=自动选择, manual=手动指定, top_n=选择前N个
    max_dynamic_factors: int = 50  # 最大动态因子数量
    factor_selection_window: int = 63  # 因子选择窗口（交易日）

    # 因子选择标准
    min_factor_ic: float = 0.02  # 最小IC值
    min_factor_win_rate: float = 0.55  # 最小胜率
    max_factor_correlation: float = 0.7  # 最大因子间相关性

    # 智能权重配置
    dynamic_factor_weighting: bool = True  # 是否启用动态因子权重
    factor_weight_decay: float = 0.95  # 因子权重衰减系数
    adapt_to_market: bool = True  # 是否适应市场环境

    # 原有因子配置保持不变
    # ... 继承所有原有配置字段


class FactorSelector:
    """动态因子选择器"""

    def __init__(self, factor_registry, config: EnhancedStrategyConfig):
        """
        初始化因子选择器

        Args:
            factor_registry: 因子注册表
            config: 策略配置
        """
        self.registry = factor_registry
        self.config = config
        self.factor_performance_cache = {}

    def calculate_factor_ic(self, factor_data: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """
        计算因子IC值

        Args:
            factor_data: 因子数据
            returns: 收益率数据

        Returns:
            因子IC值字典
        """
        ic_values = {}

        for column in factor_data.columns:
            if column in ['date', 'etf_code']:
                continue

            try:
                # 按日期分组计算IC
                daily_ic = []
                for date, group in factor_data.groupby('date'):
                    if len(group) < 3:  # 至少需要3个样本
                        continue

                    # 对齐数据
                    factor_values = group.set_index('etf_code')[column]
                    return_values = returns.get(date, pd.Series())

                    if not factor_values.empty and not return_values.empty:
                        # 取交集
                        common_etfs = factor_values.index.intersection(return_values.index)
                        if len(common_etfs) >= 3:
                            ic, _ = stats.spearmanr(
                                factor_values[common_etfs],
                                return_values[common_etfs]
                            )
                            if not np.isnan(ic):
                                daily_ic.append(ic)

                if daily_ic:
                    ic_values[column] = np.mean(daily_ic)

            except Exception as e:
                logger.warning(f"计算因子IC失败 {column}: {str(e)}")
                continue

        return ic_values

    def calculate_factor_performance(self, factor_data: pd.DataFrame,
                                  price_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        计算因子表现指标

        Args:
            factor_data: 因子数据
            price_data: 价格数据

        Returns:
            因子表现指标字典
        """
        # 计算收益率
        price_pivot = price_data.pivot(index='date', columns='etf_code', values='close')
        returns = price_pivot.pct_change().shift(-1)  # 下一期收益率

        performance = {}

        for column in factor_data.columns:
            if column in ['date', 'etf_code']:
                continue

            try:
                # 提取因子数据
                factor_values = factor_data[['date', 'etf_code', column]].copy()

                # 计算IC值
                ic_values = self.calculate_factor_ic(factor_values, returns)

                if column in ic_values:
                    ic_mean = ic_values[column]

                    # 计算胜率
                    ic_series = []
                    for date, group in factor_values.groupby('date'):
                        if len(group) < 3:
                            continue
                        factor_vals = group.set_index('etf_code')[column]
                        return_vals = returns.get(date, pd.Series())
                        common_etfs = factor_vals.index.intersection(return_vals.index)
                        if len(common_etfs) >= 3:
                            ic, _ = stats.spearmanr(
                                factor_vals[common_etfs],
                                return_vals[common_etfs]
                            )
                            if not np.isnan(ic):
                                ic_series.append(ic)

                    win_rate = np.mean(np.array(ic_series) > 0) if ic_series else 0

                    # 计算IC标准差
                    ic_std = np.std(ic_series) if ic_series else 0

                    performance[column] = {
                        'ic_mean': ic_mean,
                        'ic_std': ic_std,
                        'win_rate': win_rate,
                        'ir': ic_mean / ic_std if ic_std > 0 else 0,
                        'sample_count': len(ic_series)
                    }

            except Exception as e:
                logger.warning(f"计算因子表现失败 {column}: {str(e)}")
                continue

        return performance

    def select_factors(self, factor_data: pd.DataFrame,
                      price_data: pd.DataFrame) -> List[str]:
        """
        选择有效因子

        Args:
            factor_data: 因子数据
            price_data: 价格数据

        Returns:
            选择的因子ID列表
        """
        if self.config.factor_selection_method == "manual":
            # 手动指定，返回所有动态因子
            return self.registry.list_factors(is_dynamic=True)

        # 计算因子表现
        performance = self.calculate_factor_performance(factor_data, price_data)

        # 应用选择标准
        selected_factors = []

        for factor_id, metrics in performance.items():
            # 检查IC值
            if abs(metrics['ic_mean']) < self.config.min_factor_ic:
                continue

            # 检查胜率
            if metrics['win_rate'] < self.config.min_factor_win_rate:
                continue

            # 检查样本数
            if metrics['sample_count'] < 10:
                continue

            selected_factors.append((factor_id, metrics))

        # 按IC值排序
        selected_factors.sort(key=lambda x: abs(x[1]['ic_mean']), reverse=True)

        # 因子去重（相关性过滤）
        if self.config.max_factor_correlation < 1.0:
            selected_factors = self._filter_correlated_factors(
                selected_factors, factor_data
            )

        # 选择前N个
        factor_ids = [f[0] for f in selected_factors[:self.config.max_dynamic_factors]]

        logger.info(f"动态因子选择完成: {len(factor_ids)}/{len(performance)} 个因子")
        return factor_ids

    def _filter_correlated_factors(self, factor_performance: List[Tuple[str, Dict]],
                                 factor_data: pd.DataFrame) -> List[Tuple[str, Dict]]:
        """
        过滤高相关因子

        Args:
            factor_performance: 因子表现列表
            factor_data: 因子数据

        Returns:
            过滤后的因子表现列表
        """
        if len(factor_performance) <= 1:
            return factor_performance

        # 计算因子相关性矩阵
        factor_ids = [f[0] for f in factor_performance]
        factor_matrix = factor_data.pivot(index='date', columns='etf_code', values=factor_ids)

        if factor_matrix.empty:
            return factor_performance

        correlation_matrix = factor_matrix.corr().abs()

        # 选择因子
        selected = []
        for i, (factor_id, metrics) in enumerate(factor_performance):
            if i == 0:
                selected.append((factor_id, metrics))
                continue

            # 检查与已选因子的相关性
            is_correlated = False
            for selected_id, _ in selected:
                if selected_id in correlation_matrix.index and factor_id in correlation_matrix.columns:
                    corr = correlation_matrix.loc[selected_id, factor_id]
                    if corr > self.config.max_factor_correlation:
                        is_correlated = True
                        break

            if not is_correlated:
                selected.append((factor_id, metrics))

        return selected


class ETFCrossSectionStrategyEnhanced(ETFCrossSectionStrategy):
    """增强的ETF横截面策略引擎"""

    def __init__(self, config: EnhancedStrategyConfig,
                 data_manager=None, enable_storage: bool = True):
        """
        初始化增强策略引擎

        Args:
            config: 增强策略配置
            data_manager: 数据管理器
            enable_storage: 是否启用存储
        """
        # 调用父类初始化（需要转换为基础配置）
        base_config = StrategyConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            etf_universe=config.etf_universe,
            rebalance_freq=config.rebalance_freq,
            top_n=config.top_n,
            min_score=config.min_score,
            weight_method=config.weight_method,
            max_single_weight=config.max_single_weight,
            max_sector_weight=config.max_sector_weight,
            min_liquidity_score=config.min_liquidity_score,
            lookback_days=config.lookback_days,
            momentum_weight=config.momentum_weight,
            quality_weight=config.quality_weight,
            liquidity_weight=config.liquidity_weight,
            technical_weight=config.technical_weight
        )

        super().__init__(base_config, data_manager, enable_storage)

        # 扩展配置
        self.enhanced_config = config

        # 使用增强的因子计算器
        self.factor_calculator = ETFCrossSectionFactorsEnhanced(
            data_manager=self.data_manager,
            enable_storage=enable_storage,
            enable_dynamic_factors=config.enable_dynamic_factors
        )

        # 初始化因子选择器
        if config.enable_dynamic_factors:
            self.factor_selector = FactorSelector(
                get_factor_registry(), config
            )
        else:
            self.factor_selector = None

        # 因子权重历史
        self.factor_weights_history = []

        logger.info("增强ETF横截面策略引擎初始化完成")

    def initialize_dynamic_factors(self) -> int:
        """初始化动态因子"""
        if not self.enhanced_config.enable_dynamic_factors:
            return 0

        return self.factor_calculator.initialize_dynamic_factors()

    def run_enhanced_backtest(self) -> pd.DataFrame:
        """
        运行增强回测

        Returns:
            回测结果DataFrame
        """
        logger.info("开始运行增强策略回测...")

        # 初始化动态因子
        self.initialize_dynamic_factors()

        # 计算增强因子数据
        all_factors = self.factor_calculator.calculate_all_factors_enhanced(
            start_date=self.enhanced_config.start_date,
            end_date=self.enhanced_config.end_date,
            etf_codes=self.enhanced_config.etf_universe,
            include_original=True,
            use_cache=True
        )

        if all_factors.empty:
            logger.error("因子数据为空，无法运行回测")
            return pd.DataFrame()

        # 生成再平衡日期
        rebalance_dates = self._generate_rebalance_dates(all_factors)

        # 运行回测
        backtest_results = []

        for i, date in enumerate(rebalance_dates):
            logger.info(f"处理再平衡日期: {date} ({i+1}/{len(rebalance_dates)})")

            try:
                # 选择因子（如果启用动态因子）
                if self.enhanced_config.enable_dynamic_factors and self.factor_selector:
                    selected_factors = self._select_factors_for_date(date, all_factors)
                    factor_weights = self._calculate_factor_weights(date, selected_factors, all_factors)
                else:
                    selected_factors = None
                    factor_weights = None

                # 计算组合权重
                portfolio_weights = self._calculate_enhanced_portfolio_weights(
                    date, all_factors, selected_factors, factor_weights
                )

                if portfolio_weights:
                    backtest_results.append({
                        'date': date,
                        'weights': portfolio_weights,
                        'selected_factors': selected_factors,
                        'factor_weights': factor_weights
                    })

            except Exception as e:
                logger.error(f"处理日期 {date} 失败: {str(e)}")
                continue

        # 转换为DataFrame
        if backtest_results:
            results_df = pd.DataFrame(backtest_results)
            logger.info(f"增强回测完成: {len(results_df)} 个交易日")
            return results_df
        else:
            logger.error("没有生成任何回测结果")
            return pd.DataFrame()

    def _select_factors_for_date(self, date: str, all_factors: pd.DataFrame) -> Optional[List[str]]:
        """
        为指定日期选择因子

        Args:
            date: 日期
            all_factors: 所有因子数据

        Returns:
            选择的因子ID列表
        """
        try:
            # 获取选择窗口内的数据
            date_dt = pd.to_datetime(date)
            start_date = date_dt - timedelta(days=self.enhanced_config.factor_selection_window)
            start_date_str = start_date.strftime('%Y-%m-%d')

            window_data = all_factors[
                (all_factors['date'] >= start_date_str) &
                (all_factors['date'] <= date)
            ].copy()

            if window_data.empty:
                return None

            # 获取价格数据
            etf_codes = window_data['etf_code'].unique().tolist()
            price_data = self.data_manager.get_time_series_data(
                start_date_str, date, etf_codes
            )

            if price_data.empty:
                return None

            # 选择因子
            selected_factors = self.factor_selector.select_factors(window_data, price_data)
            return selected_factors

        except Exception as e:
            logger.error(f"选择因子失败 {date}: {str(e)}")
            return None

    def _calculate_factor_weights(self, date: str, selected_factors: List[str],
                                all_factors: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        计算因子权重

        Args:
            date: 日期
            selected_factors: 选择的因子ID列表
            all_factors: 所有因子数据

        Returns:
            因子权重字典
        """
        if not self.enhanced_config.dynamic_factor_weighting or not selected_factors:
            # 等权重
            if selected_factors:
                equal_weight = 1.0 / len(selected_factors)
                return {factor_id: equal_weight for factor_id in selected_factors}
            else:
                return {}

        try:
            # 获取历史因子表现数据
            # 这里简化处理，实际应该基于历史表现计算权重
            # 可以使用IC值、胜率等指标

            # 当前简单实现：基于因子名称的启发式权重
            factor_weights = {}
            total_weight = 0

            for factor_id in selected_factors:
                weight = 1.0

                # 根据因子类别调整权重
                if 'momentum' in factor_id.lower() or 'rsi' in factor_id.lower() or 'macd' in factor_id.lower():
                    weight *= 1.2  # 动量因子权重稍高
                elif 'volume' in factor_id.lower() or 'obv' in factor_id.lower():
                    weight *= 1.1  # 成交量因子权重中等
                elif 'volatility' in factor_id.lower() or 'atr' in factor_id.lower():
                    weight *= 0.9  # 波动率因子权重稍低

                factor_weights[factor_id] = weight
                total_weight += weight

            # 归一化
            if total_weight > 0:
                factor_weights = {k: v/total_weight for k, v in factor_weights.items()}

            return factor_weights

        except Exception as e:
            logger.error(f"计算因子权重失败 {date}: {str(e)}")
            # 返回等权重
            if selected_factors:
                equal_weight = 1.0 / len(selected_factors)
                return {factor_id: equal_weight for factor_id in selected_factors}
            return {}

    def _calculate_enhanced_portfolio_weights(self, date: str, all_factors: pd.DataFrame,
                                           selected_factors: Optional[List[str]] = None,
                                           factor_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        计算增强的组合权重

        Args:
            date: 日期
            all_factors: 所有因子数据
            selected_factors: 选择的因子ID列表
            factor_weights: 因子权重字典

        Returns:
            组合权重列表 [(etf_code, weight), ...]
        """
        # 获取当日因子数据
        date_data = all_factors[all_factors['date'] == date].copy()

        if date_data.empty:
            return []

        # 如果有动态因子选择，计算综合得分
        if selected_factors and factor_weights:
            composite_scores = self._calculate_composite_scores(
                date_data, selected_factors, factor_weights
            )
            date_data['enhanced_composite_score'] = date_data['etf_code'].map(composite_scores)
            score_column = 'enhanced_composite_score'
        else:
            # 使用原有的综合得分
            score_column = 'composite_score'

        if score_column not in date_data.columns:
            logger.warning(f"得分列 {score_column} 不存在")
            return []

        # 筛选ETF
        valid_etfs = date_data[date_data[score_column] >= self.enhanced_config.min_score]

        if valid_etfs.empty:
            return []

        # 按得分排序选择前N只
        valid_etfs = valid_etfs.nlargest(self.enhanced_config.top_n, score_column)

        # 计算权重
        weights = self._calculate_weights_enhanced(valid_etfs, score_column)

        logger.debug(f"日期 {date} 选择了 {len(weights)} 只ETF")
        return weights

    def _calculate_composite_scores(self, date_data: pd.DataFrame,
                                  selected_factors: List[str],
                                  factor_weights: Dict[str, float]) -> Dict[str, float]:
        """
        计算综合得分

        Args:
            date_data: 当日因子数据
            selected_factors: 选择的因子列表
            factor_weights: 因子权重字典

        Returns:
            ETF综合得分字典
        """
        composite_scores = {}

        for _, row in date_data.iterrows():
            etf_code = row['etf_code']
            total_score = 0.0
            total_weight = 0.0

            for factor_id in selected_factors:
                if factor_id in row and not pd.isna(row[factor_id]):
                    factor_value = float(row[factor_id])
                    weight = factor_weights.get(factor_id, 0.0)

                    # 标准化因子值（简化处理）
                    # 实际应该使用历史数据进行z-score标准化
                    if factor_id.startswith('RSI') or factor_id.startswith('VBT_RSI'):
                        # RSI类型指标，值域0-100，标准化到[-1,1]
                        normalized_value = (factor_value - 50) / 50
                    elif factor_id in ['MACD', 'VBT_MACD'] or 'macd' in factor_id.lower():
                        # MACD指标，直接使用
                        normalized_value = factor_value
                    else:
                        # 其他指标，简单标准化
                        normalized_value = factor_value

                    total_score += normalized_value * weight
                    total_weight += weight

            if total_weight > 0:
                composite_scores[etf_code] = total_score / total_weight

        return composite_scores

    def _calculate_weights_enhanced(self, selected_etfs: pd.DataFrame, score_column: str) -> List[Tuple[str, float]]:
        """
        计算增强权重

        Args:
            selected_etfs: 选择的ETF DataFrame
            score_column: 得分列名

        Returns:
            权重列表
        """
        weights = []

        if self.enhanced_config.weight_method == "equal":
            # 等权重
            equal_weight = 1.0 / len(selected_etfs)
            for _, row in selected_etfs.iterrows():
                weights.append((row['etf_code'], equal_weight))

        elif self.enhanced_config.weight_method == "score":
            # 得分加权
            if score_column in selected_etfs.columns:
                # 确保得分非负
                scores = selected_etfs[score_column].clip(lower=0)
                total_score = scores.sum()
                if total_score > 0:
                    for _, row in selected_etfs.iterrows():
                        weight = row[score_column] / total_score
                        weights.append((row['etf_code'], weight))

        # 应用权重限制
        weights = self._apply_weight_constraints(weights)

        return weights

    def _apply_weight_constraints(self, weights: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        应用权重约束

        Args:
            weights: 原始权重列表

        Returns:
            约束后的权重列表
        """
        if not weights:
            return weights

        # 应用单只ETF最大权重约束
        constrained_weights = []
        for etf_code, weight in weights:
            constrained_weight = min(weight, self.enhanced_config.max_single_weight)
            constrained_weights.append((etf_code, constrained_weight))

        # 重新归一化
        total_weight = sum(weight for _, weight in constrained_weights)
        if total_weight > 0:
            constrained_weights = [(etf, weight/total_weight) for etf, weight in constrained_weights]

        return constrained_weights

    def get_strategy_statistics(self) -> Dict[str, Any]:
        """获取策略统计信息"""
        stats = {
            "strategy_type": "enhanced",
            "dynamic_factors_enabled": self.enhanced_config.enable_dynamic_factors,
            "factor_selection_method": self.enhanced_config.factor_selection_method,
            "max_dynamic_factors": self.enhanced_config.max_dynamic_factors,
            "dynamic_factor_weighting": self.enhanced_config.dynamic_factor_weighting
        }

        if self.enhanced_config.enable_dynamic_factors:
            factor_stats = self.factor_calculator.get_factor_statistics()
            stats.update(factor_stats)

        return stats


@safe_operation
def main():
    """主函数 - 测试增强策略引擎"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建增强策略配置
    config = EnhancedStrategyConfig(
        start_date="2024-01-01",
        end_date="2025-10-14",
        enable_dynamic_factors=True,
        factor_selection_method="auto",
        max_dynamic_factors=20,
        top_n=5,
        rebalance_freq="M"
    )

    # 创建增强策略引擎
    strategy = ETFCrossSectionStrategyEnhanced(config)

    # 获取策略统计信息
    stats = strategy.get_strategy_statistics()
    print(f"策略统计信息: {stats}")

    # 运行回测（简化版本，实际需要完整的数据支持）
    print("增强策略引擎测试完成")


if __name__ == "__main__":
    main()