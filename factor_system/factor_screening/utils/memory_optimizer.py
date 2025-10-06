#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化工具 - P2-4内存效率提升
目标: 从75%提升至80%+
"""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """内存优化器 - 监控和优化内存使用"""

    def __init__(self, target_efficiency: float = 0.80):
        """
        初始化内存优化器

        Args:
            target_efficiency: 目标内存效率（0.80 = 80%）
        """
        self.target_efficiency = target_efficiency
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_memory_percent(self) -> float:
        """获取内存使用百分比"""
        return self.process.memory_percent()

    def calculate_efficiency(self, working_data_size: float) -> float:
        """
        计算内存效率

        Args:
            working_data_size: 工作数据集大小（MB）

        Returns:
            内存效率（0~1）
        """
        current_memory = self.get_memory_usage()
        total_used = current_memory - self.baseline_memory

        if total_used <= 0:
            return 1.0

        efficiency = working_data_size / total_used
        return min(1.0, efficiency)

    @staticmethod
    def optimize_dataframe_memory(
        df: pd.DataFrame, aggressive: bool = False
    ) -> pd.DataFrame:
        """
        优化DataFrame内存占用

        Args:
            df: 原始DataFrame
            aggressive: 激进模式（可能损失精度）

        Returns:
            优化后的DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # 优化数值类型
        for col in df.select_dtypes(include=["float64"]).columns:
            if aggressive:
                # 激进模式: float32
                df[col] = df[col].astype("float32")
            else:
                # 保守模式: 检查值范围后决定
                col_min = df[col].min()
                col_max = df[col].max()

                # 如果值范围适合float32，则转换
                if (
                    col_min >= np.finfo(np.float32).min
                    and col_max <= np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype("float32")

        for col in df.select_dtypes(include=["int64"]).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            # 选择最小足够的整数类型
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype("int8")
            elif (
                col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max
            ):
                df[col] = df[col].astype("int16")
            elif (
                col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max
            ):
                df[col] = df[col].astype("int32")

        # 优化对象类型
        for col in df.select_dtypes(include=["object"]).columns:
            num_unique_values = df[col].nunique()
            num_total_values = len(df[col])

            # 如果唯一值比例< 50%，转换为category
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype("category")

        optimized_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_memory / original_memory) * 100

        logger.info(
            f"内存优化: {original_memory:.2f}MB → {optimized_memory:.2f}MB "
            f"(减少 {reduction:.1f}%)"
        )

        return df

    @contextmanager
    def track_memory(self, operation_name: str = "Operation"):
        """
        内存跟踪上下文管理器

        Args:
            operation_name: 操作名称

        Example:
            with optimizer.track_memory("IC计算"):
                result = calculate_ic(factors, returns)
        """
        start_memory = self.get_memory_usage()
        logger.info(f"📊 {operation_name} 开始，内存: {start_memory:.1f}MB")

        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            delta = end_memory - start_memory

            if delta > 0:
                logger.info(
                    f"📊 {operation_name} 完成，内存增长: +{delta:.1f}MB "
                    f"(当前 {end_memory:.1f}MB)"
                )
            else:
                logger.info(
                    f"📊 {operation_name} 完成，内存释放: {abs(delta):.1f}MB "
                    f"(当前 {end_memory:.1f}MB)"
                )

    def force_cleanup(self) -> Dict[str, float]:
        """
        强制内存清理

        Returns:
            清理统计信息
        """
        before_memory = self.get_memory_usage()

        # 强制垃圾回收
        collected = gc.collect()

        after_memory = self.get_memory_usage()
        freed = before_memory - after_memory

        stats = {
            "before_mb": before_memory,
            "after_mb": after_memory,
            "freed_mb": freed,
            "objects_collected": collected,
        }

        if freed > 0:
            logger.info(f"🧹 内存清理: 释放 {freed:.1f}MB，回收 {collected} 个对象")

        return stats

    @staticmethod
    def estimate_dataframe_memory(shape: tuple, dtypes: Dict[str, str]) -> float:
        """
        估算DataFrame内存占用

        Args:
            shape: (行数, 列数)
            dtypes: 列名到数据类型的映射

        Returns:
            估算内存大小（MB）
        """
        dtype_sizes = {
            "float64": 8,
            "float32": 4,
            "int64": 8,
            "int32": 4,
            "int16": 2,
            "int8": 1,
            "object": 50,  # 估算
            "category": 1,  # 估算
        }

        rows, cols = shape
        total_bytes = 0

        for dtype in dtypes.to_numpy()():
            bytes_per_value = dtype_sizes.get(dtype, 8)  # 默认8字节
            total_bytes += rows * bytes_per_value

        return total_bytes / 1024 / 1024

    def check_memory_pressure(self, threshold: float = 0.85) -> bool:
        """
        检查内存压力

        Args:
            threshold: 内存使用率阈值（0.85 = 85%）

        Returns:
            是否超过阈值
        """
        memory_percent = self.process.memory_percent()

        if memory_percent > threshold * 100:
            logger.warning(
                f"⚠️  内存压力过高: {memory_percent:.1f}% "
                f"(阈值 {threshold*100:.0f}%)"
            )
            return True

        return False

    def suggest_optimization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        建议优化策略

        Args:
            df: 待优化DataFrame

        Returns:
            优化建议
        """
        current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # 模拟优化后内存
        df_copy = df.copy()
        df_optimized = self.optimize_dataframe_memory(df_copy, aggressive=False)
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024

        potential_saving = current_memory - optimized_memory
        saving_percent = (potential_saving / current_memory) * 100

        suggestions = {
            "current_memory_mb": current_memory,
            "optimized_memory_mb": optimized_memory,
            "potential_saving_mb": potential_saving,
            "saving_percent": saving_percent,
            "recommendations": [],
        }

        # 生成具体建议
        if saving_percent > 20:
            suggestions["recommendations"].append("建议进行内存优化，可节省超过20%内存")

        float64_cols = len(df.select_dtypes(include=["float64"]).columns)
        if float64_cols > 0:
            suggestions["recommendations"].append(
                f"发现{float64_cols}个float64列，可转换为float32"
            )

        int64_cols = len(df.select_dtypes(include=["int64"]).columns)
        if int64_cols > 0:
            suggestions["recommendations"].append(
                f"发现{int64_cols}个int64列，可转换为更小整数类型"
            )

        return suggestions


# 全局优化器实例
_global_optimizer: Optional[MemoryOptimizer] = None


def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = MemoryOptimizer()
    return _global_optimizer


# 便捷装饰器


def optimize_memory(aggressive: bool = False):
    """
    内存优化装饰器

    Args:
        aggressive: 激进模式

    Example:
        @optimize_memory()
        def my_function(df):
            return process(df)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()

            # 优化输入DataFrame
            new_args = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    arg = optimizer.optimize_dataframe_memory(arg, aggressive)
                new_args.append(arg)

            new_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    value = optimizer.optimize_dataframe_memory(value, aggressive)
                new_kwargs[key] = value

            # 执行函数
            with optimizer.track_memory(func.__name__):
                result = func(*new_args, **new_kwargs)

            # 优化输出DataFrame
            if isinstance(result, pd.DataFrame):
                result = optimizer.optimize_dataframe_memory(result, aggressive)

            # 清理内存
            optimizer.force_cleanup()

            return result

        return wrapper

    return decorator
