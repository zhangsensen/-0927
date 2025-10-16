"""缓存管理 - 内存+磁盘双层缓存"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from factor_system.factor_engine.core.registry import FactorRequest

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""

    memory_size_mb: int = 500  # 内存缓存大小(MB)
    disk_cache_dir: Path = Path("cache/factor_engine")
    ttl_hours: int = 24  # 缓存有效期(小时)
    enable_disk: bool = True
    enable_memory: bool = True
    max_ram_mb: int = 2048
    n_jobs: int = 1
    copy_mode: str = "view"  # 数据复制模式: view/copy/deepcopy


class LRUCache:
    """LRU内存缓存 - 线程安全版本"""

    def __init__(self, maxsize_mb: int):
        self.maxsize_bytes = maxsize_mb * 1024 * 1024
        self.cache: OrderedDict = OrderedDict()
        self.current_size = 0
        self._lock = threading.RLock()  # 可重入锁，支持递归调用

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存，根据copy_mode返回适当拷贝"""
        with self._lock:
            if key not in self.cache:
                return None

            # 移动到最后（最近使用）
            self.cache.move_to_end(key)
            data, timestamp = self.cache[key]

            # 检查是否过期
            if datetime.now() - timestamp > timedelta(hours=24):
                del self.cache[key]
                return None

            # 根据copy_mode返回适当的拷贝
            copy_mode = getattr(self, "copy_mode", "view")
            if copy_mode == "view":
                return data
            elif copy_mode == "copy":
                return data.copy()
            else:  # deepcopy
                return data.copy()

    def set(self, key: str, data: pd.DataFrame):
        """设置缓存（线程安全）"""
        # 估算数据大小
        data_size = data.memory_usage(deep=True).sum()

        # 如果数据太大，直接跳过
        if data_size > self.maxsize_bytes:
            logger.warning(f"数据太大，跳过内存缓存: {data_size / 1024 / 1024:.2f}MB")
            return

        with self._lock:
            try:
                # 检查key是否已存在，如果是则先减去旧数据大小
                if key in self.cache:
                    old_data, _ = self.cache[key]
                    old_size = old_data.memory_usage(deep=True).sum()
                    self.current_size -= old_size
                    logger.debug(f"替换现有缓存: {key}, 释放 {old_size / 1024:.2f}KB")

                # 清理空间
                while self.current_size + data_size > self.maxsize_bytes and self.cache:
                    old_key, (old_data, _) = self.cache.popitem(last=False)
                    old_size = old_data.memory_usage(deep=True).sum()
                    self.current_size -= old_size
                    logger.debug(f"LRU淘汰: {old_key}")

                # 添加新数据
                self.cache[key] = (data, datetime.now())
                self.current_size += data_size

            except Exception as e:
                # 确保异常情况下current_size不会出错
                logger.error(f"缓存设置失败: {key}, error={e}")
                # 重新计算current_size以保证一致性
                self._recalculate_size()

    def clear(self):
        """清空缓存（线程安全）"""
        with self._lock:
            self.cache.clear()
            self.current_size = 0

    def _recalculate_size(self):
        """重新计算缓存大小（用于异常恢复）"""
        try:
            total_size = 0
            for data, _ in self.cache.values():
                total_size += data.memory_usage(deep=True).sum()
            self.current_size = total_size
            logger.debug(f"重新计算缓存大小: {total_size / 1024 / 1024:.2f}MB")
        except Exception as e:
            logger.error(f"重新计算缓存大小失败: {e}")
            self.current_size = 0  # 重置为0，避免无限增长

    def set_copy_mode(self, mode: str):
        """设置复制模式"""
        with self._lock:
            if mode in ["view", "copy", "deepcopy"]:
                self.copy_mode = mode
            else:
                logger.warning(f"无效的copy_mode: {mode}，使用默认值 'view'")
                self.copy_mode = "view"

    def __len__(self) -> int:
        return len(self.cache)


class DiskCache:
    """磁盘缓存 - 简化版本"""

    def __init__(self, cache_dir: Path, ttl_hours: int):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存，返回深拷贝"""
        cache_file = self._get_cache_path(key)

        if not cache_file.exists():
            return None

        # 检查过期
        if self._is_expired(cache_file):
            cache_file.unlink()
            logger.debug(f"缓存过期: {key}")
            return None

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"磁盘缓存命中: {key}")

            # 根据copy_mode返回适当的拷贝
            copy_mode = getattr(self, "copy_mode", "view")
            if copy_mode == "view":
                return data
            elif copy_mode == "copy":
                return data.copy()
            else:  # deepcopy
                return data.copy()
        except Exception as e:
            logger.error(f"加载磁盘缓存失败: {e}")
            cache_file.unlink(missing_ok=True)  # 删除损坏的缓存文件
            return None

    def set(self, key: str, data: pd.DataFrame):
        """设置缓存"""
        cache_file = self._get_cache_path(key)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"写入磁盘缓存: {key}")
        except Exception as e:
            logger.error(f"写入磁盘缓存失败: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"

    def _is_expired(self, cache_file: Path) -> bool:
        """检查缓存是否过期"""
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - mtime > self.ttl

    def clear(self):
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class CacheManager:
    """双层缓存管理器 - 简化版本"""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # 初始化缓存
        self.memory_cache = (
            LRUCache(self.config.memory_size_mb) if self.config.enable_memory else None
        )
        self.disk_cache = (
            DiskCache(self.config.disk_cache_dir, self.config.ttl_hours)
            if self.config.enable_disk
            else None
        )

        # 设置copy_mode
        if self.memory_cache:
            self.memory_cache.set_copy_mode(self.config.copy_mode)
        if self.disk_cache:
            self.disk_cache.copy_mode = self.config.copy_mode

        # 统计信息
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
        }

    def get(
        self,
        factor_requests: Sequence["FactorRequest"],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        获取缓存的因子数据

        Returns:
            (cached_data, missing_factor_ids)
        """
        cache_key = self._build_key(
            factor_requests, symbols, timeframe, start_date, end_date
        )

        # 1. 尝试内存缓存
        if self.memory_cache:
            data = self.memory_cache.get(cache_key)
            if data is not None:
                self.stats["memory_hits"] += 1
                logger.debug(f"内存缓存命中: {cache_key}")
                return data, []

        # 2. 尝试磁盘缓存
        if self.disk_cache:
            data = self.disk_cache.get(cache_key)
            if data is not None:
                self.stats["disk_hits"] += 1
                # 加载到内存
                if self.memory_cache:
                    self.memory_cache.set(cache_key, data)
                logger.debug(f"磁盘缓存命中: {cache_key}")
                return data, []

        # 3. 缓存未命中
        self.stats["misses"] += 1
        missing_ids = [req.factor_id for req in factor_requests]
        return None, missing_ids

    def set(
        self,
        data: pd.DataFrame,
        factor_requests: Sequence["FactorRequest"],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ):
        """写入缓存"""
        cache_key = self._build_key(
            factor_requests, symbols, timeframe, start_date, end_date
        )

        # 写入内存
        if self.memory_cache:
            self.memory_cache.set(cache_key, data)

        # 写入磁盘
        if self.disk_cache:
            self.disk_cache.set(cache_key, data)

    def _build_key(
        self,
        factor_requests: Sequence["FactorRequest"],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """构建简化的缓存键"""
        # 简单的字符串格式，避免复杂的JSON序列化
        factor_parts = [
            req.cache_key()
            for req in sorted(factor_requests, key=lambda x: x.factor_id)
        ]
        symbol_str = ",".join(sorted(symbols))

        # 创建键的基础部分
        key_parts = [
            f"factors:{','.join(factor_parts)}",
            f"symbols:{symbol_str}",
            f"tf:{timeframe}",
            f"dates:{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
        ]

        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = sum(self.stats.values())
        hit_rate = (
            (self.stats["memory_hits"] + self.stats["disk_hits"]) / total
            if total > 0
            else 0
        )

        return {
            **self.stats,
            "total_requests": total,
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache) if self.memory_cache else 0,
        }

    def clear(self):
        """清空所有缓存"""
        if self.memory_cache:
            self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
        logger.info("缓存已清空")
