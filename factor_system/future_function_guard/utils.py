#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 工具函数模块
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 提供通用的工具函数
- 缓存管理功能
- 文件处理和路径管理
- 日志记录和报告生成
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import CacheConfig
from .exceptions import CacheError


class SimpleCache:
    """简单的内存缓存实现"""

    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self._access_times: Dict[str, float] = {}  # key -> last access time

    def _is_expired(self, key: str) -> bool:
        """检查缓存项是否过期"""
        if key not in self._cache:
            return True

        _, timestamp = self._cache[key]
        return time.time() - timestamp > self.ttl_seconds

    def _evict_if_needed(self) -> None:
        """如果需要，淘汰最久未使用的项"""
        if len(self._cache) >= self.max_size:
            # 找到最久未访问的键
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            self._evict(oldest_key)

    def _evict(self, key: str) -> None:
        """删除指定的缓存项"""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        if key in self._cache and not self._is_expired(key):
            self._access_times[key] = time.time()
            return self._cache[key][0]
        return default

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        self._evict_if_needed()
        self._cache[key] = (value, time.time())
        self._access_times[key] = time.time()

    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_times.clear()

    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)

    def cleanup_expired(self) -> int:
        """清理过期项，返回清理的数量"""
        expired_keys = [key for key in self._cache.keys() if self._is_expired(key)]
        for key in expired_keys:
            self._evict(key)
        return len(expired_keys)


class FileCache:
    """文件系统缓存实现"""

    def __init__(self, cache_dir: Union[str, Path], config: CacheConfig):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名，避免文件名过长或包含特殊字符
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"

    def _is_expired(self, cache_path: Path) -> bool:
        """检查缓存文件是否过期"""
        if not cache_path.exists():
            return True

        mtime = cache_path.stat().st_mtime
        return time.time() - mtime > self.config.ttl_hours * 3600

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)

        if cache_path.exists() and not self._is_expired(cache_path):
            try:
                if self.config.compression_enabled:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
            except Exception as e:
                raise CacheError(f"Failed to load cache: {e}", cache_key=key)

        return default

    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)

        try:
            if self.config.compression_enabled:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(value, f, ensure_ascii=False, default=str)
        except Exception as e:
            raise CacheError(f"Failed to save cache: {e}", cache_key=key)

    def clear(self) -> None:
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                raise CacheError(f"Failed to delete cache file: {e}")

    def cleanup_expired(self) -> int:
        """清理过期文件，返回清理的数量"""
        cleaned_count = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            if self._is_expired(cache_file):
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                except Exception:
                    pass  # 忽略删除失败的文件
        return cleaned_count

    def get_size_info(self) -> Dict[str, Any]:
        """获取缓存大小信息"""
        total_size = 0
        file_count = 0

        for cache_file in self.cache_dir.glob("*.cache"):
            total_size += cache_file.stat().st_size
            file_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "cache_dir": str(self.cache_dir)
        }


def setup_logging(config) -> logging.Logger:
    """
    设置日志记录器

    Args:
        config: 日志配置

    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger("future_function_guard")
    logger.setLevel(getattr(logging, config.level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    formatter = logging.Formatter(config.format)

    # 控制台处理器
    if config.enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if config.file_path:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_time_series_data(data: Union[pd.DataFrame, pd.Series]) -> List[str]:
    """
    验证时间序列数据的基本质量

    Args:
        data: 时间序列数据（DataFrame或Series）

    Returns:
        发现的问题列表
    """
    issues = []

    # 处理不同类型的数据
    if isinstance(data, pd.Series):
        # 对于Series，转换为DataFrame以便统一处理
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        # 非pandas数据类型
        issues.append("数据不是pandas DataFrame或Series")
        return issues

    if df.empty:
        issues.append("数据为空")
        return issues

    # 检查时间索引
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("索引不是DatetimeIndex")

    # 检查时间序列单调性
    if not df.index.is_monotonic_increasing:
        issues.append("时间索引不是严格递增的")

    # 检查重复时间点
    if df.index.duplicated().any():
        duplicates = df.index.duplicated().sum()
        issues.append(f"存在{duplicates}个重复时间点")

    # 检查OHLCV数据完整性（仅对DataFrame）
    if isinstance(data, pd.DataFrame):
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"缺少必要列: {missing_columns}")

        # 检查数据完整性
        if all(col in df.columns for col in required_columns):
            # 检查负价格
            price_columns = ["open", "high", "low", "close"]
            for col in price_columns:
                if (df[col] <= 0).any():
                    issues.append(f"列{col}包含负价格")

            # 检查负成交量
            if (df["volume"] < 0).any():
                issues.append("成交量包含负值")

            # 检查OHLC逻辑
            logic_errors = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            )
            if logic_errors.any():
                issues.append("OHLC数据存在逻辑错误")

    return issues


def calculate_factor_statistics(
    factor_data: pd.Series,
    return_data: Optional[pd.Series] = None,
    horizon: int = 1
) -> Dict[str, Any]:
    """
    计算因子统计信息

    Args:
        factor_data: 因子数据
        return_data: 收益率数据（可选）
        horizon: 预测周期

    Returns:
        统计信息字典
    """
    stats = {}

    # 基础统计
    stats["count"] = len(factor_data)
    stats["missing_count"] = factor_data.isna().sum()
    stats["coverage"] = (factor_data.notna().mean() if len(factor_data) > 0 else 0)

    if factor_data.notna().sum() > 0:
        valid_data = factor_data.dropna()
        stats["mean"] = float(valid_data.mean())
        stats["std"] = float(valid_data.std())
        stats["min"] = float(valid_data.min())
        stats["max"] = float(valid_data.max())
        stats["q25"] = float(valid_data.quantile(0.25))
        stats["q50"] = float(valid_data.quantile(0.50))
        stats["q75"] = float(valid_data.quantile(0.75))

        # 分布统计
        try:
            from scipy import stats as scipy_stats
            stats["skewness"] = float(scipy_stats.skew(valid_data))
            stats["kurtosis"] = float(scipy_stats.kurtosis(valid_data))
        except ImportError:
            stats["skewness"] = None
            stats["kurtosis"] = None

        # IC计算
        if return_data is not None and len(return_data) == len(factor_data):
            try:
                # 对齐数据
                common_index = factor_data.notna() & return_data.notna()
                if common_index.sum() > 10:
                    aligned_factor = factor_data[common_index]
                    aligned_return = return_data[common_index].shift(horizon).dropna()
                    aligned_factor = aligned_factor.loc[aligned_return.index]

                    if len(aligned_factor) > 5:
                        ic = aligned_factor.corr(aligned_return)
                        stats["ic"] = float(ic) if not pd.isna(ic) else None
                        stats["ic_abs"] = float(abs(ic)) if not pd.isna(ic) else None
            except Exception:
                stats["ic"] = None
                stats["ic_abs"] = None

    return stats


def generate_timestamp() -> str:
    """生成标准时间戳字符串"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """
    格式化时间间隔

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值

    Returns:
        除法结果或默认值
    """
    try:
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
    except Exception:
        return default


def normalize_factor_name(name: str) -> str:
    """
    标准化因子名称

    Args:
        name: 原始因子名称

    Returns:
        标准化的因子名称
    """
    # 移除特殊字符，替换为下划线
    import re
    normalized = re.sub(r'[^\w\s-]', '_', name)
    # 替换多个连续的下划线
    normalized = re.sub(r'_+', '_', normalized)
    # 移除首尾下划线
    normalized = normalized.strip('_')
    return normalized


def create_directory_if_not_exists(directory: Union[str, Path]) -> Path:
    """
    创建目录（如果不存在）

    Args:
        directory: 目录路径

    Returns:
        Path对象
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    计算文件的MD5哈希值

    Args:
        file_path: 文件路径

    Returns:
        MD5哈希值
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def batch_processing(items: List[Any], batch_size: int = 100):
    """
    批量处理生成器

    Args:
        items: 要处理的项目列表
        batch_size: 批次大小

    Yields:
        批次数据
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def is_valid_time_range(start_date: datetime, end_date: datetime) -> bool:
    """
    验证时间范围是否有效

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        是否有效
    """
    return start_date < end_date and start_date.year >= 1990 and end_date.year <= 2100


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    深度合并两个字典

    Args:
        dict1: 第一个字典
        dict2: 第二个字典

    Returns:
        合并后的字典
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result