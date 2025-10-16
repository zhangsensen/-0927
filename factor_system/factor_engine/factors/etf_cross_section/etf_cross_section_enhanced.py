#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子增强模块
扩展现有ETF横截面系统，支持800-1200个动态因子，保持原有架构完全不变
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from ..etf_cross_section import ETFCrossSectionFactors
from .factor_registry import get_factor_registry, FactorCategory
from .etf_factor_factory import ETFFactorFactory

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


class ETFCrossSectionFactorsEnhanced(ETFCrossSectionFactors):
    """增强的ETF横截面因子计算器"""

    def __init__(self, data_manager=None, enable_storage: bool = True,
                 enable_dynamic_factors: bool = True):
        """
        初始化增强的ETF横截面因子计算器

        Args:
            data_manager: ETF数据管理器
            enable_storage: 是否启用存储功能
            enable_dynamic_factors: 是否启用动态因子
        """
        # 调用父类初始化，保持原有功能完全不变
        super().__init__(data_manager, enable_storage)

        # 新增功能：动态因子支持
        self.enable_dynamic_factors = enable_dynamic_factors
        self.factor_registry = get_factor_registry()
        self.factor_factory = ETFFactorFactory()

        # 缓存动态因子
        self.dynamic_factor_cache = {}

        logger.info(f"增强ETF横截面因子计算器初始化完成")
        logger.info(f"动态因子支持: {'启用' if enable_dynamic_factors else '禁用'}")

    def initialize_dynamic_factors(self) -> int:
        """
        初始化动态因子（注册所有VBT+TA因子）

        Returns:
            成功注册的因子数量
        """
        if not self.enable_dynamic_factors:
            logger.warning("动态因子功能未启用")
            return 0

        logger.info("开始初始化动态因子...")

        # 清除现有动态因子
        self.factor_registry.clear_dynamic_factors()

        # 注册所有因子
        success_count = self.factor_factory.register_all_factors()

        logger.info(f"动态因子初始化完成: {success_count} 个因子已注册")
        return success_count

    def calculate_single_dynamic_factor(self,
                                     factor_id: str,
                                     price_data: pd.DataFrame,
                                     symbol: str) -> pd.Series:
        """
        计算单个动态因子

        Args:
            factor_id: 因子ID
            price_data: 价格数据
            symbol: ETF代码

        Returns:
            因子值Series
        """
        try:
            # 获取因子元数据
            factor_metadata = self.factor_registry.get_factor(factor_id)
            if factor_metadata is None:
                logger.error(f"因子未找到: {factor_id}")
                return pd.Series(0, index=price_data.index)

            # 获取该ETF的数据
            symbol_data = price_data[price_data['symbol'] == symbol].copy()
            if symbol_data.empty:
                logger.warning(f"ETF {symbol} 数据为空")
                return pd.Series(0, index=price_data['date'].unique())

            # 设置索引为日期
            symbol_data = symbol_data.set_index('date').sort_index()

            # 计算因子
            factor_function = factor_metadata.function
            factor_values = factor_function(symbol_data)

            # 确保返回正确索引的Series
            result = pd.Series(0, index=symbol_data.index)
            if isinstance(factor_values, pd.Series):
                result = factor_values.reindex(symbol_data.index, fill_value=0)
            elif isinstance(factor_values, np.ndarray):
                result = pd.Series(factor_values, index=symbol_data.index)
            else:
                result = pd.Series(factor_values, index=symbol_data.index)

            return result

        except Exception as e:
            logger.error(f"计算动态因子失败 {factor_id} - {symbol}: {str(e)}")
            return pd.Series(0, index=price_data['date'].unique())

    def calculate_dynamic_factors_batch(self,
                                      factor_ids: List[str],
                                      price_data: pd.DataFrame,
                                      symbols: List[str],
                                      parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量计算动态因子

        Args:
            factor_ids: 因子ID列表
            price_data: 价格数据
            symbols: ETF代码列表
            parallel: 是否并行计算

        Returns:
            因子数据字典 {factor_id: factor_dataframe}
        """
        if not factor_ids:
            return {}

        logger.info(f"开始批量计算 {len(factor_ids)} 个动态因子，{len(symbols)} 只ETF")

        results = {}
        total_tasks = len(factor_ids) * len(symbols)

        if parallel and total_tasks > 10:
            # 并行计算
            results = self._calculate_factors_parallel(factor_ids, price_data, symbols)
        else:
            # 串行计算
            results = self._calculate_factors_serial(factor_ids, price_data, symbols)

        successful_count = sum(1 for df in results.values() if not df.empty)
        logger.info(f"动态因子计算完成: {successful_count}/{len(factor_ids)} 个因子成功")

        return results

    def _calculate_factors_parallel(self,
                                   factor_ids: List[str],
                                   price_data: pd.DataFrame,
                                   symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """并行计算因子"""
        results = {}
        max_workers = min(mp.cpu_count(), 8)  # 限制最大进程数

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_key = {}

            for factor_id in factor_ids:
                for symbol in symbols:
                    key = f"{factor_id}_{symbol}"
                    future = executor.submit(
                        self.calculate_single_dynamic_factor,
                        factor_id, price_data, symbol
                    )
                    future_to_key[future] = (factor_id, symbol)

            # 收集结果
            for future in as_completed(future_to_key):
                factor_id, symbol = future_to_key[future]
                try:
                    factor_values = future.result()

                    # 累积到因子结果中
                    if factor_id not in results:
                        results[factor_id] = {}

                    # 添加日期列
                    factor_values.name = 'factor_value'
                    factor_df = factor_values.reset_index()
                    factor_df['symbol'] = symbol
                    factor_df = factor_df.rename(columns={'factor_value': factor_id})

                    if symbol not in results[factor_id]:
                        results[factor_id][symbol] = factor_df
                    else:
                        results[factor_id][symbol] = pd.concat([
                            results[factor_id][symbol], factor_df
                        ], ignore_index=True)

                except Exception as e:
                    logger.error(f"并行计算失败 {factor_id} - {symbol}: {str(e)}")

        # 合并每个因子的所有ETF数据
        final_results = {}
        for factor_id, symbol_data in results.items():
            if symbol_data:
                factor_df = pd.concat(symbol_data.values(), ignore_index=True)
                factor_df = factor_df.sort_values(['date', 'symbol']).reset_index(drop=True)
                final_results[factor_id] = factor_df

        return final_results

    def _calculate_factors_serial(self,
                                 factor_ids: List[str],
                                 price_data: pd.DataFrame,
                                 symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """串行计算因子"""
        results = {}

        for factor_id in factor_ids:
            symbol_data_list = []

            for symbol in symbols:
                try:
                    factor_values = self.calculate_single_dynamic_factor(
                        factor_id, price_data, symbol
                    )

                    # 转换为DataFrame
                    factor_df = factor_values.reset_index()
                    factor_df['symbol'] = symbol
                    factor_df = factor_df.rename(columns={0: factor_id})

                    symbol_data_list.append(factor_df)

                except Exception as e:
                    logger.error(f"串行计算失败 {factor_id} - {symbol}: {str(e)}")

            if symbol_data_list:
                factor_df = pd.concat(symbol_data_list, ignore_index=True)
                factor_df = factor_df.sort_values(['date', 'symbol']).reset_index(drop=True)
                results[factor_id] = factor_df

        return results

    def calculate_all_factors_enhanced(self,
                                     start_date: str,
                                     end_date: str,
                                     etf_codes: Optional[List[str]] = None,
                                     factor_categories: Optional[List[FactorCategory]] = None,
                                     include_original: bool = True,
                                     use_cache: bool = True) -> pd.DataFrame:
        """
        计算增强的所有因子（原有因子 + 动态因子）

        Args:
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表
            factor_categories: 因子类别过滤
            include_original: 是否包含原有32个因子
            use_cache: 是否使用缓存

        Returns:
            完整的因子DataFrame
        """
        logger.info(f"开始计算增强的ETF横截面因子: {start_date} ~ {end_date}")

        # 初始化动态因子（如果需要）
        if self.enable_dynamic_factors and not self.factor_registry.get_dynamic_factors():
            self.initialize_dynamic_factors()

        # 获取ETF列表
        if etf_codes is None:
            etf_codes = self.data_manager.get_etf_universe()

        # 获取价格数据
        price_data = self.data_manager.get_time_series_data(start_date, end_date, etf_codes)
        if price_data.empty:
            logger.error("未获取到价格数据")
            return pd.DataFrame()

        # 结果容器
        all_factors_dfs = []

        # 计算原有因子（如果需要）
        if include_original:
            logger.info("计算原有32个因子...")
            try:
                original_factors = self.calculate_all_factors(
                    start_date, end_date, etf_codes, use_cache, save_to_storage=False
                )
                if not original_factors.empty:
                    all_factors_dfs.append(original_factors)
                    logger.info(f"原有因子计算完成: {len(original_factors)} 条记录")
            except Exception as e:
                logger.error(f"原有因子计算失败: {str(e)}")

        # 计算动态因子（如果启用）
        if self.enable_dynamic_factors:
            logger.info("计算动态因子...")
            dynamic_factors = self._calculate_dynamic_factors_by_category(
                price_data, etf_codes, factor_categories, use_cache
            )

            if dynamic_factors:
                # 合并所有动态因子
                merged_dynamic = self._merge_dynamic_factors(dynamic_factors)
                if not merged_dynamic.empty:
                    all_factors_dfs.append(merged_dynamic)
                    logger.info(f"动态因子计算完成: {len(merged_dynamic)} 条记录，{len(dynamic_factors)} 个因子")

        # 合并所有因子
        if all_factors_dfs:
            final_result = self._merge_all_factors(all_factors_dfs)
            logger.info(f"增强因子计算完成: {len(final_result)} 条记录，{final_result['etf_code'].nunique()} 只ETF")
            return final_result
        else:
            logger.error("没有成功计算出任何因子")
            return pd.DataFrame()

    def _calculate_dynamic_factors_by_category(self,
                                             price_data: pd.DataFrame,
                                             etf_codes: List[str],
                                             categories: Optional[List[FactorCategory]] = None,
                                             use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """按类别计算动态因子"""
        # 获取要计算的因子ID
        factor_ids = self.factor_registry.list_factors()
        dynamic_factor_ids = self.factor_registry.list_factors(is_dynamic=True)

        if categories:
            # 按类别过滤
            filtered_factors = []
            for factor_id in dynamic_factor_ids:
                metadata = self.factor_registry.get_factor(factor_id)
                if metadata and metadata.category in categories:
                    filtered_factors.append(factor_id)
            factor_ids = filtered_factors
        else:
            factor_ids = dynamic_factor_ids

        logger.info(f"计划计算动态因子: {len(factor_ids)} 个")

        if not factor_ids:
            return {}

        # 检查缓存
        cache_key = f"dynamic_factors_{'_'.join(sorted(factor_ids))}_{price_data['date'].min()}_{price_data['date'].max()}"
        if use_cache and cache_key in self.dynamic_factor_cache:
            logger.info("从缓存加载动态因子")
            return self.dynamic_factor_cache[cache_key]

        # 批量计算
        results = self.calculate_dynamic_factors_batch(
            factor_ids, price_data, etf_codes, parallel=True
        )

        # 缓存结果
        if use_cache:
            self.dynamic_factor_cache[cache_key] = results

        return results

    def _merge_dynamic_factors(self, dynamic_factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """合并动态因子数据"""
        if not dynamic_factors:
            return pd.DataFrame()

        # 合并所有动态因子
        merged_df = None

        for factor_id, factor_df in dynamic_factors.items():
            if factor_df.empty:
                continue

            # 确保数据格式正确
            if 'date' not in factor_df.columns or 'symbol' not in factor_df.columns:
                continue

            # 重命名列以匹配原有格式
            factor_df = factor_df.rename(columns={'symbol': 'etf_code'})

            if merged_df is None:
                merged_df = factor_df
            else:
                merged_df = pd.merge(merged_df, factor_df, on=['date', 'etf_code'], how='outer')

        return merged_df if merged_df is not None else pd.DataFrame()

    def _merge_all_factors(self, factor_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合并所有因子数据"""
        if not factor_dfs:
            return pd.DataFrame()

        merged_df = factor_dfs[0].copy()

        for df in factor_dfs[1:]:
            if df.empty:
                continue
            merged_df = pd.merge(merged_df, df, on=['date', 'etf_code'], how='outer')

        return merged_df

    def get_available_factors(self, category: Optional[FactorCategory] = None,
                            include_original: bool = True) -> Dict[str, Any]:
        """
        获取可用因子列表

        Args:
            category: 因子类别过滤
            include_original: 是否包含原有因子

        Returns:
            因子信息字典
        """
        factors_info = {
            "original_factors": [],
            "dynamic_factors": [],
            "total_count": 0
        }

        # 原有因子
        if include_original:
            original_factor_names = [
                'momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d',
                'volatility_60d', 'max_drawdown_252d', 'quality_score', 'liquidity_score',
                'rsi_14d', 'macd_signal', 'bollinger_position', 'volume_ratio',
                'composite_score'  # 综合因子
            ]
            factors_info["original_factors"] = original_factor_names

        # 动态因子
        if self.enable_dynamic_factors:
            dynamic_factor_ids = self.factor_registry.list_factors(category=category)
            factors_info["dynamic_factors"] = dynamic_factor_ids

        factors_info["total_count"] = len(factors_info["original_factors"]) + len(factors_info["dynamic_factors"])

        return factors_info

    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子统计信息"""
        stats = {
            "original_factors": 32,
            "dynamic_factors": 0,
            "categories": {}
        }

        if self.enable_dynamic_factors:
            registry_stats = self.factor_registry.get_statistics()
            stats["dynamic_factors"] = registry_stats["dynamic_factors"]
            stats["categories"] = registry_stats["categories"]

        stats["total_factors"] = stats["original_factors"] + stats["dynamic_factors"]

        return stats


@safe_operation
def main():
    """主函数 - 测试增强的ETF横截面因子系统"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建增强的因子计算器
    enhanced_calculator = ETFCrossSectionFactorsEnhanced(enable_dynamic_factors=True)

    # 初始化动态因子
    factor_count = enhanced_calculator.initialize_dynamic_factors()
    print(f"初始化动态因子: {factor_count} 个")

    # 获取统计信息
    stats = enhanced_calculator.get_factor_statistics()
    print(f"因子统计: {stats}")

    # 获取可用因子
    available_factors = enhanced_calculator.get_available_factors()
    print(f"可用因子: {available_factors}")


if __name__ == "__main__":
    main()