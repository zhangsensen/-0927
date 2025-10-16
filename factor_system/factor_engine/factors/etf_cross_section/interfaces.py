#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子系统接口定义
实现完全解耦的架构设计，避免循环依赖
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd


class IFactorCalculator(ABC):
    """因子计算器接口"""

    @abstractmethod
    def calculate_factors(self,
                         symbols: List[str],
                         timeframe: str,
                         start_date: datetime,
                         end_date: datetime,
                         factor_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算因子值

        Args:
            symbols: 股票代码列表
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            factor_ids: 因子ID列表，None表示计算所有因子

        Returns:
            因子值DataFrame，索引为日期，列为(symbol, factor_id)
        """
        pass

    @abstractmethod
    def get_available_factors(self) -> List[str]:
        """获取可用因子列表"""
        pass

    @abstractmethod
    def get_factor_categories(self) -> Dict[str, List[str]]:
        """获取因子分类"""
        pass


class ICrossSectionManager(ABC):
    """横截面管理器接口"""

    @abstractmethod
    def build_cross_section(self,
                           date: datetime,
                           symbols: List[str],
                           factor_ids: List[str]) -> pd.DataFrame:
        """
        构建横截面数据

        Args:
            date: 截面日期
            symbols: 股票代码列表
            factor_ids: 因子ID列表

        Returns:
            横截面DataFrame，索引为symbol，列为factor_id
        """
        pass

    @abstractmethod
    def get_cross_section_summary(self,
                                 cross_section_df: pd.DataFrame) -> Dict[str, Any]:
        """
        获取横截面摘要统计

        Args:
            cross_section_df: 横截面数据

        Returns:
            摘要统计字典
        """
        pass


class IFactorRegistry(ABC):
    """因子注册表接口"""

    @abstractmethod
    def register_factor(self,
                       factor_id: str,
                       function: callable,
                       parameters: Dict[str, Any],
                       category: str,
                       description: str,
                       is_dynamic: bool = False) -> bool:
        """注册因子"""
        pass

    @abstractmethod
    def get_factors(self,
                    category: Optional[str] = None,
                    is_dynamic: Optional[bool] = None) -> List[str]:
        """获取因子列表"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        pass


class IProgressMonitor(ABC):
    """进度监控接口"""

    @abstractmethod
    def update_progress(self, current: int, total: int, message: str = ""):
        """更新进度"""
        pass

    @abstractmethod
    def log_info(self, message: str):
        """记录信息"""
        pass

    @abstractmethod
    def log_warning(self, message: str):
        """记录警告"""
        pass

    @abstractmethod
    def log_error(self, message: str):
        """记录错误"""
        pass


class ETFCrossSectionConfig:
    """ETF横截面配置类"""

    def __init__(self):
        # 性能配置
        self.max_workers: int = -1  # 并行进程数，-1表示使用所有CPU
        self.chunk_size: int = 100  # 分块处理大小
        self.memory_limit_mb: int = 1024  # 内存限制(MB)

        # 因子配置
        self.enable_legacy_factors: bool = True  # 是否启用传统32因子
        self.enable_dynamic_factors: bool = True  # 是否启用动态因子
        self.max_dynamic_factors: int = 1200  # 最大动态因子数

        # 计算配置
        self.cache_enabled: bool = True  # 是否启用缓存
        self.cache_ttl_hours: int = 24  # 缓存过期时间(小时)

        # 输出配置
        self.verbose: bool = True  # 详细输出
        self.save_intermediate: bool = False  # 保存中间结果

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.max_workers < -1:
            raise ValueError("max_workers must be >= -1")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be > 0")
        return True


class FactorCalculationResult:
    """因子计算结果类"""

    def __init__(self,
                 factors_df: pd.DataFrame,
                 successful_factors: List[str],
                 failed_factors: List[str],
                 calculation_time: float,
                 memory_usage_mb: float):
        self.factors_df = factors_df
        self.successful_factors = successful_factors
        self.failed_factors = failed_factors
        self.calculation_time = calculation_time
        self.memory_usage_mb = memory_usage_mb
        self.total_factors = len(successful_factors) + len(failed_factors)
        self.success_rate = len(successful_factors) / self.total_factors if self.total_factors > 0 else 0

    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            'total_factors': self.total_factors,
            'successful_factors': len(self.successful_factors),
            'failed_factors': len(self.failed_factors),
            'success_rate': self.success_rate,
            'calculation_time': self.calculation_time,
            'memory_usage_mb': self.memory_usage_mb,
            'data_shape': self.factors_df.shape if self.factors_df is not None else None
        }


class CrossSectionResult:
    """横截面分析结果类"""

    def __init__(self,
                 cross_section_df: pd.DataFrame,
                 summary_stats: Dict[str, Any],
                 build_time: float):
        self.cross_section_df = cross_section_df
        self.summary_stats = summary_stats
        self.build_time = build_time
        self.num_stocks = len(cross_section_df) if cross_section_df is not None else 0
        self.num_factors = len(cross_section_df.columns) if cross_section_df is not None else 0

    def get_summary(self) -> Dict[str, Any]:
        """获取结果摘要"""
        return {
            'num_stocks': self.num_stocks,
            'num_factors': self.num_factors,
            'build_time': self.build_time,
            'summary_stats': self.summary_stats,
            'data_shape': self.cross_section_df.shape if self.cross_section_df is not None else None
        }