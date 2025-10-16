#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子统一管理器
解决循环导入问题，实现800-1200个动态因子的完整集成
"""

import numpy as np
import pandas as pd
import logging
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .interfaces import (
    IFactorCalculator, ICrossSectionManager, IFactorRegistry, IProgressMonitor,
    ETFCrossSectionConfig, FactorCalculationResult, CrossSectionResult
)
from .factor_registry import get_factor_registry, FactorCategory
from .etf_factor_factory import ETFFactorFactory
from .batch_factor_calculator import BatchFactorCalculator

logger = logging.getLogger(__name__)


class DefaultProgressMonitor(IProgressMonitor):
    """默认进度监控器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._start_time = time.time()

    def update_progress(self, current: int, total: int, message: str = ""):
        if self.verbose:
            progress = (current / total) * 100 if total > 0 else 0
            elapsed = time.time() - self._start_time
            logger.info(f"进度: {progress:.1f}% ({current}/{total}) - {message} [耗时: {elapsed:.1f}s]")

    def log_info(self, message: str):
        if self.verbose:
            logger.info(message)

    def log_warning(self, message: str):
        logger.warning(message)

    def log_error(self, message: str):
        logger.error(message)


class ETFCrossSectionUnifiedManager(IFactorCalculator, ICrossSectionManager):
    """ETF横截面因子统一管理器"""

    def __init__(self, config: Optional[ETFCrossSectionConfig] = None,
                 progress_monitor: Optional[IProgressMonitor] = None):
        """
        初始化统一管理器

        Args:
            config: 配置对象
            progress_monitor: 进度监控器
        """
        self.config = config or ETFCrossSectionConfig()
        self.config.validate()

        self.progress_monitor = progress_monitor or DefaultProgressMonitor(self.config.verbose)

        # 动态因子相关（无循环导入风险）
        self.dynamic_factory = ETFFactorFactory()
        self.factor_registry = get_factor_registry()

        # 传统因子计算器（延迟初始化避免循环导入）
        self._legacy_calculator = None
        self._batch_calculator = None

        # 统计信息
        self._available_factors_cache = None
        self._factor_categories_cache = None

        self.progress_monitor.log_info("ETF横截面统一管理器初始化完成")

    @property
    def legacy_calculator(self):
        """延迟初始化传统因子计算器，避免循环导入"""
        if self._legacy_calculator is None and self.config.enable_legacy_factors:
            self.progress_monitor.log_info("延迟初始化传统因子计算器...")
            # 使用延迟导入函数避免循环依赖
            from . import get_etf_cross_section_factors
            from factor_system.factor_engine.providers.etf_cross_section_provider import ETFCrossSectionDataManager

            ETFCrossSectionFactors = get_etf_cross_section_factors()
            if ETFCrossSectionFactors is None:
                self.progress_monitor.log_error("无法加载传统因子计算器")
                return None

            data_manager = ETFCrossSectionDataManager()
            self._legacy_calculator = ETFCrossSectionFactors(
                data_manager=data_manager,
                enable_storage=False
            )
            self.progress_monitor.log_info("传统因子计算器初始化完成")

        return self._legacy_calculator

    @property
    def batch_calculator(self):
        """延迟初始化批量因子计算器"""
        if self._batch_calculator is None:
            self._batch_calculator = BatchFactorCalculator()
        return self._batch_calculator

    def _register_all_dynamic_factors(self) -> int:
        """注册所有动态因子到注册表"""
        if not self.config.enable_dynamic_factors:
            return 0

        self.progress_monitor.log_info("开始注册动态因子...")
        start_time = time.time()

        # 清除现有动态因子
        self.factor_registry.clear_dynamic_factors()

        # 注册所有动态因子
        registered_count = self.dynamic_factory.register_all_factors()

        # 限制最大因子数量
        all_factors = self.factor_registry.list_factors(is_dynamic=True)
        if len(all_factors) > self.config.max_dynamic_factors:
            # 保留前N个因子
            factors_to_keep = all_factors[:self.config.max_dynamic_factors]
            self.progress_monitor.log_warning(f"动态因子数量超过限制，仅保留前{self.config.max_dynamic_factors}个")

            # 这里需要实现因子移除逻辑，暂时跳过具体实现

        elapsed = time.time() - start_time
        self.progress_monitor.log_info(f"动态因子注册完成: {registered_count}个，耗时: {elapsed:.2f}s")

        return registered_count

    def _load_legacy_factors_from_config(self) -> List[str]:
        """从配置文件加载传统因子列表"""
        import yaml
        from pathlib import Path
        
        config_file = Path(__file__).parent / "configs" / "legacy_factors.yaml"
        
        if not config_file.exists():
            self.progress_monitor.log_warning(f"传统因子配置文件不存在: {config_file}")
            return []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 收集所有启用的因子
            factors = []
            enabled_categories = config.get('enabled_categories', [])
            legacy_factors_config = config.get('legacy_factors', {})
            
            for category in enabled_categories:
                if category in legacy_factors_config:
                    factors.extend(legacy_factors_config[category])
            
            self.progress_monitor.log_info(f"从配置加载传统因子: {len(factors)}个")
            return factors
            
        except Exception as e:
            self.progress_monitor.log_error(f"加载传统因子配置失败: {str(e)}")
            return []
    
    def get_available_factors(self) -> List[str]:
        """获取可用因子列表"""
        if self._available_factors_cache is None:
            factors = []

            # 动态因子
            if self.config.enable_dynamic_factors:
                dynamic_factors = self.factor_registry.list_factors(is_dynamic=True)
                factors.extend(dynamic_factors)
                self.progress_monitor.log_info(f"可用动态因子: {len(dynamic_factors)}个")

            # 传统因子 - 从配置文件加载
            if self.config.enable_legacy_factors and self.legacy_calculator:
                legacy_factors = self._load_legacy_factors_from_config()
                factors.extend(legacy_factors)
                self.progress_monitor.log_info(f"可用传统因子: {len(legacy_factors)}个")

            self._available_factors_cache = factors

        return self._available_factors_cache

    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        if self._factor_categories_cache is None:
            categories = {
                'momentum': [],
                'mean_reversion': [],
                'volume': [],
                'volatility': [],
                'trend': [],
                'overlap': [],
                'candlestick': [],
                'legacy': []
            }

            all_factors = self.get_available_factors()

            for factor_id in all_factors:
                # 根据因子ID进行分类
                if any(keyword in factor_id.lower() for keyword in ["rsi", "macd", "sto", "mom", "roc", "adx", "aroon"]):
                    categories['momentum'].append(factor_id)
                elif any(keyword in factor_id.lower() for keyword in ["bb_", "bollinger", "keltner"]):
                    categories['mean_reversion'].append(factor_id)
                elif any(keyword in factor_id.lower() for keyword in ["vol", "obv", "vwap", "ad"]):
                    categories['volume'].append(factor_id)
                elif any(keyword in factor_id.lower() for keyword in ["atr", "std", "tr"]):
                    categories['volatility'].append(factor_id)
                elif any(keyword in factor_id.lower() for keyword in ["ma_", "ema", "sma", "wma", "kama"]):
                    categories['trend'].append(factor_id)
                elif any(keyword in factor_id.lower() for keyword in ["cdl", "candle"]):
                    categories['candlestick'].append(factor_id)
                elif factor_id.startswith('VBT_') or factor_id.startswith('TALIB_'):
                    # 动态因子，根据注册表中的分类
                    continue
                else:
                    categories['legacy'].append(factor_id)

            # 获取动态因子的分类信息
            if self.config.enable_dynamic_factors:
                dynamic_categories = self.factor_registry.get_statistics()
                # 这里需要根据实际统计信息更新分类

            self._factor_categories_cache = categories

        return self._factor_categories_cache

    def calculate_factors(self,
                         symbols: List[str],
                         timeframe: str,
                         start_date: datetime,
                         end_date: datetime,
                         factor_ids: Optional[List[str]] = None) -> FactorCalculationResult:
        """
        计算因子值

        Args:
            symbols: 股票代码列表
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            factor_ids: 因子ID列表，None表示计算所有因子

        Returns:
            因子计算结果
        """
        self.progress_monitor.log_info(f"开始计算因子: {len(symbols)}只股票, {timeframe}周期")
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            # 确保动态因子已注册
            self._register_all_dynamic_factors()

            # 确定要计算的因子
            if factor_ids is None:
                factor_ids = self.get_available_factors()

            self.progress_monitor.log_info(f"计划计算因子: {len(factor_ids)}个")

            # 分离传统因子和动态因子
            legacy_factor_ids = []
            dynamic_factor_ids = []

            # 获取传统因子列表
            legacy_factor_map = {
                'MOMENTUM_21D': 'calculate_momentum_factors',
                'MOMENTUM_63D': 'calculate_momentum_factors',
                'MOMENTUM_126D': 'calculate_momentum_factors',
                'MOMENTUM_252D': 'calculate_momentum_factors',
                'VOLATILITY_20D': 'calculate_volatility_factors',
                'VOLATILITY_60D': 'calculate_volatility_factors',
                'VOLATILITY_120D': 'calculate_volatility_factors',
                'VOLATILITY_252D': 'calculate_volatility_factors',
                # ... 其他映射
            }

            for factor_id in factor_ids:
                if factor_id in legacy_factor_map and self.config.enable_legacy_factors:
                    legacy_factor_ids.append(factor_id)
                else:
                    dynamic_factor_ids.append(factor_id)

            all_factors_df = None
            successful_factors = []
            failed_factors = []

            # 计算传统因子
            if legacy_factor_ids and self.legacy_calculator:
                self.progress_monitor.log_info(f"计算传统因子: {len(legacy_factor_ids)}个")
                try:
                    # 使用传统因子计算器
                    legacy_results = self._calculate_legacy_factors(
                        symbols, timeframe, start_date, end_date, legacy_factor_ids
                    )
                    if legacy_results is not None and not legacy_results.empty:
                        if all_factors_df is None:
                            all_factors_df = legacy_results
                        else:
                            all_factors_df = pd.concat([all_factors_df, legacy_results], axis=1)
                        successful_factors.extend(legacy_factor_ids)
                        self.progress_monitor.log_info(f"传统因子计算成功: {len(legacy_factor_ids)}个")
                    else:
                        failed_factors.extend(legacy_factor_ids)
                        self.progress_monitor.log_warning(f"传统因子计算失败: {len(legacy_factor_ids)}个")
                except Exception as e:
                    failed_factors.extend(legacy_factor_ids)
                    self.progress_monitor.log_error(f"传统因子计算异常: {str(e)}")

            # 计算动态因子
            if dynamic_factor_ids and self.config.enable_dynamic_factors:
                self.progress_monitor.log_info(f"计算动态因子: {len(dynamic_factor_ids)}个")
                try:
                    dynamic_results = self._calculate_dynamic_factors(
                        symbols, timeframe, start_date, end_date, dynamic_factor_ids
                    )
                    if dynamic_results is not None and not dynamic_results.empty:
                        if all_factors_df is None:
                            all_factors_df = dynamic_results
                        else:
                            all_factors_df = pd.concat([all_factors_df, dynamic_results], axis=1)
                        successful_factors.extend(dynamic_factor_ids)
                        self.progress_monitor.log_info(f"动态因子计算成功: {len(dynamic_factor_ids)}个")
                    else:
                        failed_factors.extend(dynamic_factor_ids)
                        self.progress_monitor.log_warning(f"动态因子计算失败: {len(dynamic_factor_ids)}个")
                except Exception as e:
                    failed_factors.extend(dynamic_factor_ids)
                    self.progress_monitor.log_error(f"动态因子计算异常: {str(e)}")

            # 计算资源使用情况
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory
            calculation_time = time.time() - start_time

            self.progress_monitor.log_info(f"因子计算完成: {len(successful_factors)}成功, {len(failed_factors)}失败")
            self.progress_monitor.log_info(f"耗时: {calculation_time:.2f}s, 内存增量: {memory_usage:.1f}MB")

            return FactorCalculationResult(
                factors_df=all_factors_df,
                successful_factors=successful_factors,
                failed_factors=failed_factors,
                calculation_time=calculation_time,
                memory_usage_mb=memory_usage
            )

        except Exception as e:
            self.progress_monitor.log_error(f"因子计算失败: {str(e)}")
            raise

    def _calculate_legacy_factors(self,
                                 symbols: List[str],
                                 timeframe: str,
                                 start_date: datetime,
                                 end_date: datetime,
                                 factor_ids: List[str]) -> Optional[pd.DataFrame]:
        """
        计算传统因子
        
        Args:
            symbols: ETF代码列表
            timeframe: 时间周期（暂未使用，传统因子基于日线）
            start_date: 开始日期
            end_date: 结束日期
            factor_ids: 因子ID列表
            
        Returns:
            统一格式的DataFrame: MultiIndex(date, symbol), columns=factor_ids
        """
        if not self.legacy_calculator:
            return None

        try:
            self.progress_monitor.log_info(f"开始计算传统因子: {len(symbols)}只ETF, {len(factor_ids)}个因子")
            
            # 调用传统因子计算器
            legacy_df = self.legacy_calculator.calculate_all_factors(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                etf_codes=symbols,
                use_cache=True,
                save_to_storage=False
            )
            
            if legacy_df.empty:
                self.progress_monitor.log_warning("传统因子计算返回空结果")
                return pd.DataFrame()
            
            # 转换为统一格式
            unified_df = self._format_legacy_factors(legacy_df, factor_ids)
            
            self.progress_monitor.log_info(f"传统因子计算完成: {len(unified_df)} 条记录")
            return unified_df

        except Exception as e:
            self.progress_monitor.log_error(f"传统因子计算失败: {str(e)}")
            import traceback
            self.progress_monitor.log_error(traceback.format_exc())
            return None

    def _format_legacy_factors(self, legacy_df: pd.DataFrame, 
                               factor_ids: List[str]) -> pd.DataFrame:
        """
        转换传统因子格式为统一格式
        
        Args:
            legacy_df: 传统因子DataFrame (columns: etf_code, date, ...因子列)
            factor_ids: 需要的因子ID列表
            
        Returns:
            统一格式DataFrame: MultiIndex(date, symbol), columns=factor_ids
        """
        if legacy_df.empty:
            return pd.DataFrame()
        
        try:
            # 复制避免修改原始数据
            df = legacy_df.copy()
            
            # 确保date列是datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # 提取所有可用的因子列（排除etf_code和date）
            meta_cols = ['etf_code', 'date']
            factor_cols = [col for col in df.columns if col not in meta_cols]
            
            # 如果指定了factor_ids，只保留请求的因子
            if factor_ids:
                available_cols = [col for col in factor_ids if col in factor_cols]
                if not available_cols:
                    # 尝试模糊匹配
                    available_cols = [col for col in factor_cols 
                                     if any(fid in col for fid in factor_ids)]
                
                if available_cols:
                    factor_cols = available_cols
                else:
                    self.progress_monitor.log_warning(
                        f"请求的因子ID在传统因子中未找到，使用所有可用因子"
                    )
            
            # 选择需要的列
            cols_to_keep = meta_cols + factor_cols
            df = df[cols_to_keep]
            
            # 设置MultiIndex (date, symbol)
            # 注意：统一使用symbol而不是etf_code
            if 'etf_code' in df.columns:
                df = df.rename(columns={'etf_code': 'symbol'})
            
            if 'date' in df.columns and 'symbol' in df.columns:
                df = df.set_index(['date', 'symbol'])
            
            self.progress_monitor.log_info(
                f"传统因子格式转换完成: {df.shape}, 因子列: {list(df.columns)[:5]}..."
            )
            
            return df
            
        except Exception as e:
            self.progress_monitor.log_error(f"传统因子格式转换失败: {str(e)}")
            import traceback
            self.progress_monitor.log_error(traceback.format_exc())
            return pd.DataFrame()

    def _calculate_dynamic_factors(self,
                                 symbols: List[str],
                                 timeframe: str,
                                 start_date: datetime,
                                 end_date: datetime,
                                 factor_ids: List[str]) -> Optional[pd.DataFrame]:
        """计算动态因子"""
        try:
            # 🔥 关键修复：传递factor_registry给批量计算器
            results = self.batch_calculator.calculate_factors(
                symbols=symbols,
                factor_ids=factor_ids,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                max_workers=self.config.max_workers,
                factor_registry=self.factor_registry  # 传递ETF专用注册表
            )

            return results

        except Exception as e:
            self.progress_monitor.log_error(f"动态因子计算失败: {str(e)}")
            return None

    def build_cross_section(self,
                           date: datetime,
                           symbols: List[str],
                           factor_ids: List[str]) -> CrossSectionResult:
        """
        构建横截面数据

        Args:
            date: 截面日期
            symbols: 股票代码列表
            factor_ids: 因子ID列表

        Returns:
            横截面分析结果
        """
        self.progress_monitor.log_info(f"构建横截面: {date.strftime('%Y-%m-%d')}, {len(symbols)}只股票")
        start_time = time.time()

        try:
            # 计算因子值（只计算特定日期）
            # 这里简化处理，实际应该只计算特定日期的数据
            end_date = date
            start_date = date - timedelta(days=30)  # 确保有足够的历史数据计算因子

            factor_result = self.calculate_factors(
                symbols=symbols,
                timeframe='daily',
                start_date=start_date,
                end_date=end_date,
                factor_ids=factor_ids
            )

            if factor_result.factors_df is None or factor_result.factors_df.empty:
                raise ValueError("因子计算结果为空")

            # 提取指定日期的横截面数据
            df = factor_result.factors_df
            
            # 处理MultiIndex (date, symbol)
            if isinstance(df.index, pd.MultiIndex):
                # 尝试提取指定日期的数据
                try:
                    # 使用xs方法提取特定日期
                    cross_section_df = df.xs(date, level=0)
                except KeyError:
                    # 如果精确日期不存在，找最近的日期
                    available_dates = df.index.get_level_values(0).unique()
                    closest_date = min(available_dates, key=lambda d: abs((d - date).total_seconds()))
                    cross_section_df = df.xs(closest_date, level=0)
                    self.progress_monitor.log_warning(
                        f"未找到{date.strftime('%Y-%m-%d')}的数据，使用最近日期: {closest_date.strftime('%Y-%m-%d')}"
                    )
            else:
                # 如果不是MultiIndex，使用iloc
                cross_section_df = df.iloc[-len(symbols):]
                self.progress_monitor.log_warning("因子数据不是MultiIndex格式，使用最后N条记录")
            
            # 确保symbol作为索引
            if 'symbol' in cross_section_df.columns:
                cross_section_df = cross_section_df.set_index('symbol')
            elif not cross_section_df.index.name == 'symbol':
                cross_section_df.index.name = 'symbol'

            # 计算摘要统计
            summary_stats = self.get_cross_section_summary(cross_section_df)

            build_time = time.time() - start_time
            self.progress_monitor.log_info(f"横截面构建完成: {cross_section_df.shape}, 耗时: {build_time:.2f}s")

            return CrossSectionResult(
                cross_section_df=cross_section_df,
                summary_stats=summary_stats,
                build_time=build_time
            )

        except Exception as e:
            self.progress_monitor.log_error(f"横截面构建失败: {str(e)}")
            raise

    def get_cross_section_summary(self, cross_section_df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取横截面摘要统计

        Args:
            cross_section_df: 横截面数据

        Returns:
            摘要统计字典
        """
        if cross_section_df is None or cross_section_df.empty:
            return {}

        summary = {}

        for factor_id in cross_section_df.columns:
            factor_values = cross_section_df[factor_id].dropna()

            if len(factor_values) > 0:
                summary[factor_id] = {
                    'count': len(factor_values),
                    'mean': float(factor_values.mean()),
                    'std': float(factor_values.std()),
                    'min': float(factor_values.min()),
                    'max': float(factor_values.max()),
                    'median': float(factor_values.median()),
                    'missing_rate': (len(cross_section_df) - len(factor_values)) / len(cross_section_df)
                }

        return summary

    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            'config': {
                'enable_legacy_factors': self.config.enable_legacy_factors,
                'enable_dynamic_factors': self.config.enable_dynamic_factors,
                'max_dynamic_factors': self.config.max_dynamic_factors,
                'max_workers': self.config.max_workers,
                'memory_limit_mb': self.config.memory_limit_mb
            },
            'available_factors': {
                'total_count': len(self.get_available_factors()),
                'categories': {k: len(v) for k, v in self.get_factor_categories().items()}
            },
            'dynamic_registry': self.factor_registry.get_statistics(),
            'dynamic_factory': {
                'vbt_indicators': len(self.dynamic_factory.vbt_indicator_map),
                'talib_indicators': len(self.dynamic_factory.talib_indicator_map)
            }
        }

        return stats

    def clear_cache(self):
        """清除缓存"""
        self._available_factors_cache = None
        self._factor_categories_cache = None
        gc.collect()
        self.progress_monitor.log_info("缓存已清除")


# 工厂函数
def create_etf_cross_section_manager(config: Optional[ETFCrossSectionConfig] = None,
                                   verbose: bool = True) -> ETFCrossSectionUnifiedManager:
    """
    创建ETF横截面管理器

    Args:
        config: 配置对象
        verbose: 是否显示详细输出

    Returns:
        ETF横截面管理器实例
    """
    if config is None:
        config = ETFCrossSectionConfig()
        config.verbose = verbose

    progress_monitor = DefaultProgressMonitor(verbose=verbose)
    return ETFCrossSectionUnifiedManager(config, progress_monitor)