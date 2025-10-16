"""因子质量优化器 - 去重、筛选、性能优化

Linus原则: 数据驱动，性能优先，质量第一
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class FactorOptimizer:
    """因子质量优化器"""
    
    def __init__(
        self,
        correlation_threshold: float = 0.95,
        null_threshold: float = 0.10,
        zero_threshold: float = 0.50,
        volatility_threshold: float = 100.0,
    ):
        """初始化优化器
        
        Args:
            correlation_threshold: 相关性阈值
            null_threshold: 缺失率阈值
            zero_threshold: 零值率阈值
            volatility_threshold: 波动率阈值
        """
        self.correlation_threshold = correlation_threshold
        self.null_threshold = null_threshold
        self.zero_threshold = zero_threshold
        self.volatility_threshold = volatility_threshold
        
        self.removed_factors = {
            'high_correlation': [],
            'high_null': [],
            'high_zero': [],
            'high_volatility': []
        }
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型，节省50%内存
        
        Args:
            df: 原始DataFrame
            
        Returns:
            优化后的DataFrame
        """
        logger.info("优化数据类型...")
        
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # 优化float64 -> float32
        for col in df.select_dtypes(include=['float64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # 检查是否在float32范围内
            if pd.notna(col_min) and pd.notna(col_max):
                if col_min > -3.4e38 and col_max < 3.4e38:
                    df[col] = df[col].astype('float32')
        
        # 优化int64 -> int32
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        savings = original_memory - optimized_memory
        
        logger.info(f"  原始内存: {original_memory:.1f} MB")
        logger.info(f"  优化内存: {optimized_memory:.1f} MB")
        logger.info(f"  节省: {savings:.1f} MB ({savings/original_memory*100:.1f}%)")
        
        return df
    
    def remove_high_correlation_factors(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """去除高相关性因子
        
        Args:
            df: 因子DataFrame
            
        Returns:
            (去重后的DataFrame, 被移除的因子列表)
        """
        logger.info(f"去除高相关性因子 (阈值: {self.correlation_threshold})...")
        
        # 计算相关性矩阵
        corr_matrix = df.corr().abs()
        
        # 找到高相关因子对
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr_pairs = [
            (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            for i in range(len(corr_matrix.columns))
            for j in range(i+1, len(corr_matrix.columns))
            if corr_matrix.iloc[i, j] > self.correlation_threshold
        ]
        
        logger.info(f"  发现 {len(high_corr_pairs)} 个高相关因子对")
        
        # 贪心算法：每次移除与最多因子高相关的因子
        factors_to_remove = set()
        correlation_count = {}
        
        for f1, f2, _ in high_corr_pairs:
            correlation_count[f1] = correlation_count.get(f1, 0) + 1
            correlation_count[f2] = correlation_count.get(f2, 0) + 1
        
        # 按相关性数量排序
        sorted_factors = sorted(
            correlation_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 逐个移除，直到没有高相关对
        remaining_pairs = set(high_corr_pairs)
        for factor, _ in sorted_factors:
            if factor in factors_to_remove:
                continue
            
            # 检查移除此因子能消除多少高相关对
            pairs_to_remove = [
                p for p in remaining_pairs 
                if factor in (p[0], p[1])
            ]
            
            if pairs_to_remove:
                factors_to_remove.add(factor)
                remaining_pairs -= set(pairs_to_remove)
            
            if not remaining_pairs:
                break
        
        self.removed_factors['high_correlation'] = list(factors_to_remove)
        logger.info(f"  移除 {len(factors_to_remove)} 个高相关因子")
        
        return df.drop(columns=factors_to_remove), list(factors_to_remove)
    
    def remove_low_quality_factors(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """移除低质量因子
        
        Args:
            df: 因子DataFrame
            
        Returns:
            (清洗后的DataFrame, 被移除的因子字典)
        """
        logger.info("移除低质量因子...")
        
        # 1. 高缺失率因子
        null_rates = df.isnull().sum() / len(df)
        high_null_factors = null_rates[null_rates > self.null_threshold].index.tolist()
        self.removed_factors['high_null'] = high_null_factors
        logger.info(f"  高缺失率因子 (>{self.null_threshold*100:.0f}%): {len(high_null_factors)}个")
        
        # 2. 高零值率因子
        zero_rates = (df == 0).sum() / len(df)
        high_zero_factors = zero_rates[zero_rates > self.zero_threshold].index.tolist()
        self.removed_factors['high_zero'] = high_zero_factors
        logger.info(f"  高零值率因子 (>{self.zero_threshold*100:.0f}%): {len(high_zero_factors)}个")
        
        # 3. 高波动率因子
        volatility = df.std()
        high_vol_factors = volatility[volatility > self.volatility_threshold].index.tolist()
        self.removed_factors['high_volatility'] = high_vol_factors
        logger.info(f"  高波动率因子 (>{self.volatility_threshold}): {len(high_vol_factors)}个")
        
        # 合并所有要移除的因子
        all_removed = set(high_null_factors + high_zero_factors + high_vol_factors)
        logger.info(f"  总计移除: {len(all_removed)}个低质量因子")
        
        return df.drop(columns=all_removed), self.removed_factors
    
    def optimize_panel(
        self, 
        df: pd.DataFrame,
        remove_correlation: bool = True,
        remove_low_quality: bool = True,
        optimize_dtype: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """全面优化因子面板
        
        Args:
            df: 原始因子面板
            remove_correlation: 是否去除高相关因子
            remove_low_quality: 是否移除低质量因子
            optimize_dtype: 是否优化数据类型
            
        Returns:
            (优化后的DataFrame, 优化报告)
        """
        logger.info("=" * 60)
        logger.info("开始因子面板优化")
        logger.info("=" * 60)
        
        original_shape = df.shape
        optimized_df = df.copy()
        
        # 1. 移除低质量因子
        if remove_low_quality:
            optimized_df, removed_quality = self.remove_low_quality_factors(optimized_df)
        
        # 2. 去除高相关因子
        if remove_correlation:
            optimized_df, removed_corr = self.remove_high_correlation_factors(optimized_df)
        
        # 3. 优化数据类型
        if optimize_dtype:
            optimized_df = self.optimize_dtypes(optimized_df)
        
        # 生成优化报告
        report = {
            'original_shape': original_shape,
            'optimized_shape': optimized_df.shape,
            'removed_factors': {
                'total': original_shape[1] - optimized_df.shape[1],
                'high_correlation': len(self.removed_factors['high_correlation']),
                'high_null': len(self.removed_factors['high_null']),
                'high_zero': len(self.removed_factors['high_zero']),
                'high_volatility': len(self.removed_factors['high_volatility'])
            },
            'retention_rate': optimized_df.shape[1] / original_shape[1]
        }
        
        logger.info("=" * 60)
        logger.info("优化完成")
        logger.info("=" * 60)
        logger.info(f"原始形状: {original_shape}")
        logger.info(f"优化形状: {optimized_df.shape}")
        logger.info(f"保留率: {report['retention_rate']*100:.1f}%")
        logger.info(f"移除因子: {report['removed_factors']['total']}个")
        logger.info(f"  - 高相关: {report['removed_factors']['high_correlation']}个")
        logger.info(f"  - 高缺失: {report['removed_factors']['high_null']}个")
        logger.info(f"  - 高零值: {report['removed_factors']['high_zero']}个")
        logger.info(f"  - 高波动: {report['removed_factors']['high_volatility']}个")
        
        return optimized_df, report
