"""
性能监控器 | Performance Monitor

监控关键路径的性能指标:
- 执行时间
- 内存使用
- 数据量统计

作者: Linus Monitor
日期: 2025-10-28
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable

import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器"""

    @staticmethod
    def get_memory_usage() -> float:
        """获取当前内存使用（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    @contextmanager
    def timer(name: str):
        """
        计时上下文管理器

        用法:
            with PerformanceMonitor.timer("数据加载"):
                data = load_data()
        """
        start_time = time.time()
        start_mem = PerformanceMonitor.get_memory_usage()

        logger.info(f"⏱️  [{name}] 开始 (内存: {start_mem:.1f} MB)")

        try:
            yield
        finally:
            end_time = time.time()
            end_mem = PerformanceMonitor.get_memory_usage()
            elapsed = end_time - start_time
            mem_delta = end_mem - start_mem

            logger.info(
                f"⏱️  [{name}] 完成 - 耗时: {elapsed:.2f}s, "
                f"内存变化: {mem_delta:+.1f} MB (当前: {end_mem:.1f} MB)"
            )

    @staticmethod
    def monitor_function(func: Callable) -> Callable:
        """
        函数性能监控装饰器

        用法:
            @PerformanceMonitor.monitor_function
            def compute_factors(data):
                ...
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            with PerformanceMonitor.timer(func_name):
                result = func(*args, **kwargs)

            return result

        return wrapper


def log_data_stats(name: str, data, logger_instance=None):
    """
    记录数据统计信息

    Args:
        name: 数据名称
        data: 数据对象 (DataFrame, ndarray, dict等)
        logger_instance: 日志实例（可选）
    """
    if logger_instance is None:
        logger_instance = logger

    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        shape = data.shape
        nan_ratio = data.isna().sum().sum() / data.size
        logger_instance.info(
            f"📊 [{name}] DataFrame: {shape[0]} × {shape[1]}, NaN率: {nan_ratio:.2%}"
        )

    elif isinstance(data, np.ndarray):
        shape = data.shape
        nan_ratio = np.isnan(data).sum() / data.size
        logger_instance.info(f"📊 [{name}] ndarray: {shape}, NaN率: {nan_ratio:.2%}")

    elif isinstance(data, dict):
        logger_instance.info(f"📊 [{name}] dict: {len(data)} 项")
        if data and isinstance(next(iter(data.values())), (pd.DataFrame, np.ndarray)):
            first_key = next(iter(data.keys()))
            first_val = data[first_key]
            if isinstance(first_val, pd.DataFrame):
                logger_instance.info(f"   示例: {first_key} → {first_val.shape}")
            elif isinstance(first_val, np.ndarray):
                logger_instance.info(f"   示例: {first_key} → {first_val.shape}")

    else:
        logger_instance.info(f"📊 [{name}] {type(data).__name__}")
