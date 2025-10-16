#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子IC分析模块
计算信息系数(IC)、统计显著性检验、多空组合分析
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

from factor_system.utils import safe_operation, FactorSystemError

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ICAnalysisResult:
    """IC分析结果"""
    variant_id: str
    ic_mean: float
    ic_std: float
    ic_ir: float  # Information Ratio = IC_mean / IC_std
    ic_win_rate: float
    t_stat: float
    p_value: float
    is_significant: bool
    sample_count: int
    analysis_period: str


@dataclass
class LongShortResult:
    """多空组合分析结果"""
    variant_id: str
    long_return_mean: float
    short_return_mean: float
    long_short_return: float
    long_short_std: float
    long_short_sharpe: float
    win_rate: float
    max_drawdown: float
    calmar_ratio: float


class ICAnalyzer:
    """IC分析器"""

    def __init__(self, significance_level: float = 0.05):
        """
        初始化IC分析器

        Args:
            significance_level: 显著性水平
        """
        self.significance_level = significance_level
        logger.info(f"IC分析器初始化完成，显著性水平: {significance_level}")

    def _calculate_returns(self, price_data: pd.DataFrame, period: int = 1) -> pd.Series:
        """
        计算收益率

        Args:
            price_data: 价格数据 (symbol, date, close)
            period: 收益率周期

        Returns:
            收益率Series
        """
        # 转换为透视表格式
        price_pivot = price_data.pivot(index='date', columns='symbol', values='close')

        # 计算收益率
        returns = price_pivot.pct_change(periods=period)

        # 展平为Series
        returns_series = returns.stack()
        returns_series.name = 'return'

        return returns_series

    def _merge_factor_and_returns(self, factor_data: pd.DataFrame,
                                 returns_data: pd.Series,
                                 factor_column: str) -> pd.DataFrame:
        """
        合并因子数据和收益率数据

        Args:
            factor_data: 因子数据
            returns_data: 收益率数据
            factor_column: 因子列名

        Returns:
            合并后的数据
        """
        # 确保有必要的列
        required_cols = ['symbol', 'date', factor_column]
        if not all(col in factor_data.columns for col in required_cols):
            raise ValueError(f"因子数据缺少必要列: {required_cols}")

        # 创建因子数据副本
        factor_df = factor_data[['symbol', 'date', factor_column]].copy()

        # 设置索引以便合并
        factor_df = factor_df.set_index(['symbol', 'date'])
        returns_df = returns_data.to_frame('return')

        # 合并数据
        merged = factor_df.join(returns_df, how='inner')

        # 移除缺失值
        merged = merged.dropna()

        logger.debug(f"合并后数据: {len(merged)} 条记录")

        return merged

    def _calculate_single_ic(self, factor_values: pd.Series,
                           returns: pd.Series) -> Tuple[float, float, int]:
        """
        计算单个时间截面的IC

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            (IC值, p值, 样本数)
        """
        if len(factor_values) < 3:  # 至少需要3个样本
            return np.nan, np.nan, len(factor_values)

        try:
            # 计算Spearman相关系数
            ic, p_value = stats.spearmanr(factor_values, returns)
            return ic, p_value, len(factor_values)
        except Exception as e:
            logger.warning(f"IC计算失败: {str(e)}")
            return np.nan, np.nan, len(factor_values)

    def _calculate_time_series_ic(self, merged_data: pd.DataFrame,
                                factor_column: str) -> pd.DataFrame:
        """
        计算时间序列IC

        Args:
            merged_data: 合并后的数据
            factor_column: 因子列名

        Returns:
            IC时间序列数据
        """
        ic_results = []

        # 按日期分组计算IC
        for date, group in merged_data.groupby('date'):
            if len(group) >= 3:  # 至少需要3只ETF
                factor_values = group[factor_column]
                returns = group['return']

                ic, p_value, sample_count = self._calculate_single_ic(factor_values, returns)

                ic_results.append({
                    'date': date,
                    'ic': ic,
                    'p_value': p_value,
                    'sample_count': sample_count
                })

        return pd.DataFrame(ic_results)

    def analyze_factor_ic(self, factor_data: pd.DataFrame,
                         price_data: pd.DataFrame,
                         factor_column: str,
                         return_period: int = 1) -> ICAnalysisResult:
        """
        分析单个因子的IC

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名
            return_period: 收益率周期

        Returns:
            IC分析结果
        """
        # 计算收益率
        returns_data = self._calculate_returns(price_data, return_period)

        # 合并数据
        merged_data = self._merge_factor_and_returns(factor_data, returns_data, factor_column)

        if merged_data.empty:
            logger.warning(f"因子 {factor_column} 没有有效数据")
            return ICAnalysisResult(
                variant_id=factor_column,
                ic_mean=0, ic_std=0, ic_ir=0, ic_win_rate=0,
                t_stat=0, p_value=1, is_significant=False,
                sample_count=0, analysis_period=""
            )

        # 计算时间序列IC
        ic_series = self._calculate_time_series_ic(merged_data, factor_column)

        if ic_series.empty:
            logger.warning(f"因子 {factor_column} 无法计算IC时间序列")
            return ICAnalysisResult(
                variant_id=factor_column,
                ic_mean=0, ic_std=0, ic_ir=0, ic_win_rate=0,
                t_stat=0, p_value=1, is_significant=False,
                sample_count=0, analysis_period=""
            )

        # 移除无效IC值
        valid_ic = ic_series['ic'].dropna()

        if len(valid_ic) == 0:
            logger.warning(f"因子 {factor_column} 没有有效的IC值")
            return ICAnalysisResult(
                variant_id=factor_column,
                ic_mean=0, ic_std=0, ic_ir=0, ic_win_rate=0,
                t_stat=0, p_value=1, is_significant=False,
                sample_count=0, analysis_period=""
            )

        # 计算IC统计指标
        ic_mean = valid_ic.mean()
        ic_std = valid_ic.std()
        ic_ir = ic_mean / ic_std if ic_std != 0 else 0
        ic_win_rate = (valid_ic > 0).mean()

        # t检验
        t_stat, p_value = stats.ttest_1samp(valid_ic, 0)
        is_significant = p_value < self.significance_level

        # 分析周期
        start_date = ic_series['date'].min()
        end_date = ic_series['date'].max()
        analysis_period = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"

        result = ICAnalysisResult(
            variant_id=factor_column,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_win_rate=ic_win_rate,
            t_stat=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            sample_count=len(valid_ic),
            analysis_period=analysis_period
        )

        logger.debug(f"因子 {factor_column} IC分析完成: IC={ic_mean:.4f}, IR={ic_ir:.4f}, 显著={is_significant}")

        return result

    def analyze_long_short_portfolio(self, factor_data: pd.DataFrame,
                                   price_data: pd.DataFrame,
                                   factor_column: str,
                                   top_pct: float = 0.3,
                                   bottom_pct: float = 0.3,
                                   rebalance_freq: str = 'M') -> LongShortResult:
        """
        分析多空组合表现

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            factor_column: 因子列名
            top_pct: 多头组合比例
            bottom_pct: 空头组合比例
            rebalance_freq: 再平衡频率 ('M'=月度, 'W'=周度)

        Returns:
            多空组合分析结果
        """
        # 计算收益率
        returns_data = self._calculate_returns(price_data)

        # 合并数据
        merged_data = self._merge_factor_and_returns(factor_data, returns_data, factor_column)

        if merged_data.empty:
            return LongShortResult(
                variant_id=factor_column,
                long_return_mean=0, short_return_mean=0, long_short_return=0,
                long_short_std=0, long_short_sharpe=0, win_rate=0,
                max_drawdown=0, calmar_ratio=0
            )

        # 按再平衡频率分组
        if rebalance_freq == 'M':
            merged_data['rebalance_date'] = merged_data.index.get_level_values('date').to_period('M').to_timestamp()
        elif rebalance_freq == 'W':
            merged_data['rebalance_date'] = merged_data.index.get_level_values('date').to_period('W').to_timestamp()
        else:
            merged_data['rebalance_date'] = merged_data.index.get_level_values('date')

        # 计算每期的多空组合收益
        long_short_returns = []

        for rebalance_date, group in merged_data.groupby('rebalance_date'):
            if len(group) < 6:  # 至少需要6只ETF
                continue

            # 按因子值排序
            group_sorted = group.sort_values(factor_column)

            # 选择多头和空头
            n_stocks = len(group_sorted)
            n_long = max(1, int(n_stocks * top_pct))
            n_short = max(1, int(n_stocks * bottom_pct))

            long_stocks = group_sorted.tail(n_long).index.get_level_values('symbol')
            short_stocks = group_sorted.head(n_short).index.get_level_values('symbol')

            # 计算组合收益
            long_return = group.loc[group.index.get_level_values('symbol').isin(long_stocks), 'return'].mean()
            short_return = group.loc[group.index.get_level_values('symbol').isin(short_stocks), 'return'].mean()
            long_short_return = long_return - short_return

            long_short_returns.append({
                'date': rebalance_date,
                'long_return': long_return,
                'short_return': short_return,
                'long_short_return': long_short_return
            })

        if not long_short_returns:
            return LongShortResult(
                variant_id=factor_column,
                long_return_mean=0, short_return_mean=0, long_short_return=0,
                long_short_std=0, long_short_sharpe=0, win_rate=0,
                max_drawdown=0, calmar_ratio=0
            )

        # 计算统计指标
        ls_returns_df = pd.DataFrame(long_short_returns)

        long_return_mean = ls_returns_df['long_return'].mean()
        short_return_mean = ls_returns_df['short_return'].mean()
        long_short_return = ls_returns_df['long_short_return'].mean()
        long_short_std = ls_returns_df['long_short_return'].std()
        long_short_sharpe = long_short_return / long_short_std if long_short_std != 0 else 0
        win_rate = (ls_returns_df['long_short_return'] > 0).mean()

        # 计算最大回撤
        cumulative_returns = (1 + ls_returns_df['long_short_return']).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 计算Calmar比率
        calmar_ratio = long_short_return / abs(max_drawdown) if max_drawdown != 0 else 0

        result = LongShortResult(
            variant_id=factor_column,
            long_return_mean=long_return_mean,
            short_return_mean=short_return_mean,
            long_short_return=long_short_return,
            long_short_std=long_short_std,
            long_short_sharpe=long_short_sharpe,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio
        )

        logger.debug(f"因子 {factor_column} 多空分析完成: 多空收益={long_short_return:.4f}, "
                    f"夏普比率={long_short_sharpe:.4f}, 胜率={win_rate:.2%}")

        return result

    def batch_analyze_factors(self, factors_data: Dict[str, pd.DataFrame],
                             price_data: pd.DataFrame) -> Dict[str, ICAnalysisResult]:
        """
        批量分析因子IC

        Args:
            factors_data: 因子数据字典 {variant_id: factor_data}
            price_data: 价格数据

        Returns:
            IC分析结果字典
        """
        logger.info(f"开始批量分析 {len(factors_data)} 个因子的IC")

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

                # 分析IC
                result = self.analyze_factor_ic(factor_data, price_data, factor_column)
                result.variant_id = variant_id
                results[variant_id] = result

            except Exception as e:
                logger.error(f"分析因子 {variant_id} IC失败: {str(e)}")
                continue

        successful_count = len(results)
        logger.info(f"批量IC分析完成: {successful_count}/{len(factors_data)} 个因子成功")

        return results

    def save_ic_analysis_results(self, results: Dict[str, ICAnalysisResult],
                                output_path: str):
        """
        保存IC分析结果

        Args:
            results: IC分析结果
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
                "ic_std": result.ic_std,
                "ic_ir": result.ic_ir,
                "ic_win_rate": result.ic_win_rate,
                "t_stat": result.t_stat,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "sample_count": result.sample_count,
                "analysis_period": result.analysis_period
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)

        logger.info(f"IC分析结果已保存到: {output_file}")

        # 打印统计摘要
        significant_count = sum(1 for r in results.values() if r.is_significant)
        logger.info(f"显著因子数量: {significant_count}/{len(results)} ({significant_count/len(results):.1%})")


@safe_operation
def main():
    """主函数 - 测试IC分析功能"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 这里可以添加测试代码
    logger.info("IC分析模块测试完成")


if __name__ == "__main__":
    main()