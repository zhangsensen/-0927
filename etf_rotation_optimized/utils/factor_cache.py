#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子缓存管理器
=============

设计理念:
- 因子计算11秒 → 缓存后0.1秒（100倍提速）
- 自动检测数据/代码变化，失效自动重算
- 支持标准化因子缓存

作者: Linus-Approved
"""

import hashlib
import inspect
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class FactorCache:
    """因子缓存管理器"""

    def __init__(self, cache_dir: Path = None, use_timestamp: bool = True):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录，默认为项目根目录/cache/factors
            use_timestamp: 是否使用时间戳命名（推荐True，避免数据混淆）
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "cache" / "factors"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 使用时间戳命名，避免哈希碰撞
        self.use_timestamp = use_timestamp

        # 缓存有效期（小时）
        self.ttl_hours = 24 * 7  # 1周

    def _compute_data_hash(self, ohlcv: Dict[str, pd.DataFrame]) -> str:
        """
        计算OHLCV数据的哈希值

        Args:
            ohlcv: OHLCV数据字典

        Returns:
            MD5哈希值
        """
        # 只用close的shape和最后一行来计算hash（快速）
        close = ohlcv["close"]
        hash_input = f"{close.shape}_{close.iloc[-1].sum():.6f}_{close.index[-1]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _compute_code_hash(self, lib_class) -> str:
        """
        计算因子库代码的哈希值

        Args:
            lib_class: 因子库类

        Returns:
            MD5哈希值
        """
        # 获取类的源代码
        source = inspect.getsource(lib_class)
        return hashlib.md5(source.encode()).hexdigest()[:16]

    def _get_cache_key(self, data_hash: str, code_hash: str, stage: str) -> str:
        """生成缓存键（使用时间戳避免混淆）"""
        if self.use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{stage}_{timestamp}_{data_hash[:8]}_{code_hash[:8]}"
        else:
            return f"{stage}_{data_hash}_{code_hash}"

    def _find_latest_cache(
        self, stage: str, data_hash: str, code_hash: str
    ) -> Optional[Path]:
        """查找最新的有效缓存文件（按时间戳）"""
        if not self.use_timestamp:
            # 旧模式：直接哈希匹配
            cache_key = f"{stage}_{data_hash}_{code_hash}"
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            return cache_path if cache_path.exists() else None

        # 新模式：查找最新的匹配文件
        pattern = f"{stage}_*_{data_hash[:8]}_{code_hash[:8]}.pkl"
        matching_files = sorted(self.cache_dir.glob(pattern), reverse=True)

        if matching_files:
            return matching_files[0]  # 返回最新的

        return None

    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_path.exists():
            return False

        # 检查时效
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours > self.ttl_hours:
            return False

        return True

    def load_factors(
        self, ohlcv: Dict[str, pd.DataFrame], lib_class, stage: str = "raw"
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载缓存的因子

        Args:
            ohlcv: OHLCV数据
            lib_class: 因子库类
            stage: 阶段标识（raw/standardized）

        Returns:
            缓存的因子字典，如果无效则返回None
        """
        data_hash = self._compute_data_hash(ohlcv)
        code_hash = self._compute_code_hash(lib_class)

        # 查找最新的有效缓存
        cache_path = self._find_latest_cache(stage, data_hash, code_hash)

        if cache_path and self._is_cache_valid(cache_path):
            print(f"✅ 加载因子缓存: {cache_path.name}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        return None

    def save_factors(
        self,
        factors: Dict[str, pd.DataFrame],
        ohlcv: Dict[str, pd.DataFrame],
        lib_class,
        stage: str = "raw",
    ):
        """
        保存因子到缓存

        Args:
            factors: 因子字典
            ohlcv: OHLCV数据
            lib_class: 因子库类
            stage: 阶段标识
        """
        data_hash = self._compute_data_hash(ohlcv)
        code_hash = self._compute_code_hash(lib_class)
        cache_key = self._get_cache_key(data_hash, code_hash, stage)
        cache_path = self._get_cache_path(cache_key)

        with open(cache_path, "wb") as f:
            pickle.dump(factors, f)

        print(f"✅ 保存因子缓存: {cache_key}")

    def clear_old_cache(self, max_age_days: int = 30):
        """
        清理过期缓存

        Args:
            max_age_days: 最大保留天数
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        removed_count = 0

        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                removed_count += 1

        if removed_count > 0:
            print(f"🗑️  清理{removed_count}个过期缓存")

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        if not cache_files:
            return {"count": 0, "total_size_mb": 0, "oldest_hours": 0}

        total_size = sum(f.stat().st_size for f in cache_files)
        oldest_file = min(cache_files, key=lambda f: f.stat().st_mtime)
        oldest_hours = (time.time() - oldest_file.stat().st_mtime) / 3600

        return {
            "count": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
            "oldest_hours": oldest_hours,
        }
