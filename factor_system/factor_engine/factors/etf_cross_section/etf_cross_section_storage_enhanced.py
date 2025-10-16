#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面存储系统增强版
扩展现有存储系统，支持800-1200个动态因子的高效存储和缓存
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import gzip
import json
from functools import lru_cache
import hashlib

from .etf_cross_section_storage import ETFCrossSectionStorage
from .factor_registry import get_factor_registry

from factor_system.utils import safe_operation, FactorSystemError

logger = logging.getLogger(__name__)


class ETFCrossSectionStorageEnhanced(ETFCrossSectionStorage):
    """增强的ETF横截面存储系统"""

    def __init__(self, base_dir: str = None, enable_compression: bool = True):
        """
        初始化增强存储系统

        Args:
            base_dir: 基础存储目录
            enable_compression: 是否启用压缩
        """
        # 调用父类初始化
        super().__init__(base_dir)

        # 扩展功能
        self.enable_compression = enable_compression
        self.factor_registry = get_factor_registry()

        # 创建新的存储目录
        self.dynamic_factors_dir = self.base_dir / "dynamic_factors"
        self.dynamic_factors_dir.mkdir(exist_ok=True)

        self.factor_cache_dir = self.base_dir / "factor_cache"
        self.factor_cache_dir.mkdir(exist_ok=True)

        self.factor_metadata_dir = self.base_dir / "factor_metadata"
        self.factor_metadata_dir.mkdir(exist_ok=True)

        # 内存缓存
        self._memory_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 3600  # 1小时缓存TTL

        logger.info(f"增强ETF横截面存储系统初始化完成")
        logger.info(f"压缩功能: {'启用' if enable_compression else '禁用'}")

    def save_dynamic_factors(self, factors_data: Dict[str, pd.DataFrame],
                           start_date: str, end_date: str,
                           etf_codes: List[str],
                           compress: Optional[bool] = None) -> bool:
        """
        保存动态因子数据

        Args:
            factors_data: 因子数据字典 {factor_id: dataframe}
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表
            compress: 是否压缩

        Returns:
            保存是否成功
        """
        try:
            # 生成存储键
            storage_key = self._generate_storage_key(start_date, end_date, etf_codes, factors_data.keys())

            # 保存路径
            file_name = f"{storage_key}.{'pkl.gz' if (compress if compress is not None else self.enable_compression) else 'pkl'}"
            file_path = self.dynamic_factors_dir / file_name

            # 准备保存数据
            save_data = {
                'factors_data': factors_data,
                'metadata': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'etf_codes': etf_codes,
                    'factor_ids': list(factors_data.keys()),
                    'total_factors': len(factors_data),
                    'created_at': datetime.now().isoformat(),
                    'data_size_mb': self._estimate_data_size(factors_data)
                }
            }

            # 保存数据
            if compress if compress is not None else self.enable_compression:
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"动态因子数据已保存: {file_path}")
            logger.info(f"因子数量: {len(factors_data)}, ETF数量: {len(etf_codes)}")

            # 更新内存缓存
            cache_key = f"dynamic_factors_{storage_key}"
            self._memory_cache[cache_key] = factors_data
            self._cache_timestamps[cache_key] = datetime.now().timestamp()

            return True

        except Exception as e:
            logger.error(f"保存动态因子数据失败: {str(e)}")
            return False

    def load_dynamic_factors(self, start_date: str, end_date: str,
                           etf_codes: List[str], factor_ids: Optional[List[str]] = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        加载动态因子数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表
            factor_ids: 因子ID列表（可选）

        Returns:
            因子数据字典或None
        """
        try:
            # 检查内存缓存
            cache_key = f"dynamic_factors_{self._generate_storage_key(start_date, end_date, etf_codes, factor_ids or [])}"
            if cache_key in self._memory_cache:
                if self._is_cache_valid(cache_key):
                    logger.debug("从内存缓存加载动态因子数据")
                    cached_data = self._memory_cache[cache_key]

                    # 如果指定了factor_ids，进行过滤
                    if factor_ids:
                        return {fid: df for fid, df in cached_data.items() if fid in factor_ids}
                    else:
                        return cached_data
                else:
                    # 缓存过期，清除
                    del self._memory_cache[cache_key]
                    del self._cache_timestamps[cache_key]

            # 尝试多个可能的存储键
            possible_keys = [
                self._generate_storage_key(start_date, end_date, etf_codes, factor_ids or []),
                self._generate_storage_key(start_date, end_date, etf_codes, []),
            ]

            for storage_key in possible_keys:
                # 尝试加载压缩文件
                for ext in ['pkl.gz', 'pkl']:
                    file_path = self.dynamic_factors_dir / f"{storage_key}.{ext}"
                    if file_path.exists():
                        try:
                            # 加载数据
                            if ext == 'pkl.gz':
                                with gzip.open(file_path, 'rb') as f:
                                    save_data = pickle.load(f)
                            else:
                                with open(file_path, 'rb') as f:
                                    save_data = pickle.load(f)

                            factors_data = save_data.get('factors_data', {})
                            metadata = save_data.get('metadata', {})

                            logger.info(f"从文件加载动态因子数据: {file_path}")
                            logger.info(f"因子数量: {len(factors_data)}, ETF数量: {len(etf_codes)}")

                            # 更新内存缓存
                            self._memory_cache[cache_key] = factors_data
                            self._cache_timestamps[cache_key] = datetime.now().timestamp()

                            # 如果指定了factor_ids，进行过滤
                            if factor_ids:
                                return {fid: df for fid, df in factors_data.items() if fid in factor_ids}
                            else:
                                return factors_data

                        except Exception as e:
                            logger.warning(f"加载文件失败 {file_path}: {str(e)}")
                            continue

            logger.warning("未找到匹配的动态因子数据")
            return None

        except Exception as e:
            logger.error(f"加载动态因子数据失败: {str(e)}")
            return None

    def save_factor_cache(self, cache_key: str, data: Any, ttl_hours: int = 24) -> bool:
        """
        保存因子缓存

        Args:
            cache_key: 缓存键
            data: 缓存数据
            ttl_hours: 生存时间（小时）

        Returns:
            保存是否成功
        """
        try:
            # 缓存文件路径
            file_name = f"cache_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl.gz"
            file_path = self.factor_cache_dir / file_name

            # 准备缓存数据
            cache_data = {
                'key': cache_key,
                'data': data,
                'created_at': datetime.now().isoformat(),
                'ttl_hours': ttl_hours,
                'expires_at': (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            }

            # 保存缓存
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 更新内存缓存
            self._memory_cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now().timestamp()

            logger.debug(f"因子缓存已保存: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"保存因子缓存失败 {cache_key}: {str(e)}")
            return False

    def load_factor_cache(self, cache_key: str) -> Optional[Any]:
        """
        加载因子缓存

        Args:
            cache_key: 缓存键

        Returns:
            缓存数据或None
        """
        try:
            # 检查内存缓存
            if cache_key in self._memory_cache:
                if self._is_cache_valid(cache_key):
                    logger.debug(f"从内存缓存加载: {cache_key}")
                    return self._memory_cache[cache_key]
                else:
                    del self._memory_cache[cache_key]
                    del self._cache_timestamps[cache_key]

            # 缓存文件路径
            file_name = f"cache_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl.gz"
            file_path = self.factor_cache_dir / file_name

            if not file_path.exists():
                return None

            # 加载缓存
            with gzip.open(file_path, 'rb') as f:
                cache_data = pickle.load(f)

            # 检查是否过期
            expires_at = datetime.fromisoformat(cache_data['expires_at'])
            if datetime.now() > expires_at:
                logger.debug(f"缓存已过期: {cache_key}")
                file_path.unlink()  # 删除过期缓存
                return None

            # 更新内存缓存
            self._memory_cache[cache_key] = cache_data['data']
            self._cache_timestamps[cache_key] = datetime.now().timestamp()

            logger.debug(f"从文件缓存加载: {cache_key}")
            return cache_data['data']

        except Exception as e:
            logger.error(f"加载因子缓存失败 {cache_key}: {str(e)}")
            return None

    def save_factor_metadata(self, factor_metadata: Dict[str, Any]) -> bool:
        """
        保存因子元数据

        Args:
            factor_metadata: 因子元数据字典

        Returns:
            保存是否成功
        """
        try:
            metadata_file = self.factor_metadata_dir / "factor_metadata.json"

            # 读取现有元数据
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = {}

            # 合并元数据
            existing_metadata.update(factor_metadata)

            # 保存元数据
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"因子元数据已保存: {len(factor_metadata)} 个因子")
            return True

        except Exception as e:
            logger.error(f"保存因子元数据失败: {str(e)}")
            return False

    def load_factor_metadata(self) -> Dict[str, Any]:
        """
        加载因子元数据

        Returns:
            因子元数据字典
        """
        try:
            metadata_file = self.factor_metadata_dir / "factor_metadata.json"

            if not metadata_file.exists():
                return {}

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            logger.debug(f"因子元数据已加载: {len(metadata)} 个因子")
            return metadata

        except Exception as e:
            logger.error(f"加载因子元数据失败: {str(e)}")
            return {}

    def cleanup_expired_cache(self) -> int:
        """
        清理过期缓存

        Returns:
            清理的文件数量
        """
        try:
            cleaned_count = 0

            # 清理文件缓存
            for cache_file in self.factor_cache_dir.glob("cache_*.pkl.gz"):
                try:
                    with gzip.open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)

                    expires_at = datetime.fromisoformat(cache_data['expires_at'])
                    if datetime.now() > expires_at:
                        cache_file.unlink()
                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"清理缓存文件失败 {cache_file}: {str(e)}")
                    # 尝试删除损坏的文件
                    try:
                        cache_file.unlink()
                        cleaned_count += 1
                    except:
                        pass

            # 清理内存缓存
            expired_keys = [
                key for key, timestamp in self._cache_timestamps.items()
                if datetime.now().timestamp() - timestamp > self._cache_ttl
            ]

            for key in expired_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]
                cleaned_count += 1

            logger.info(f"清理过期缓存完成: {cleaned_count} 个文件")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理过期缓存失败: {str(e)}")
            return 0

    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息

        Returns:
            存储统计信息字典
        """
        try:
            stats = {
                "dynamic_factors_files": 0,
                "dynamic_factors_size_mb": 0,
                "cache_files": 0,
                "cache_size_mb": 0,
                "memory_cache_items": len(self._memory_cache),
                "memory_cache_size_mb": self._estimate_memory_cache_size(),
                "compression_enabled": self.enable_compression
            }

            # 统计动态因子文件
            for file_path in self.dynamic_factors_dir.glob("*.pkl*"):
                stats["dynamic_factors_files"] += 1
                stats["dynamic_factors_size_mb"] += file_path.stat().st_size / (1024 * 1024)

            # 统计缓存文件
            for file_path in self.factor_cache_dir.glob("*.pkl*"):
                stats["cache_files"] += 1
                stats["cache_size_mb"] += file_path.stat().st_size / (1024 * 1024)

            return stats

        except Exception as e:
            logger.error(f"获取存储统计失败: {str(e)}")
            return {}

    def _generate_storage_key(self, start_date: str, end_date: str,
                            etf_codes: List[str], factor_ids: List[str]) -> str:
        """生成存储键"""
        # 排序以确保一致性
        etf_codes_sorted = sorted(etf_codes)
        factor_ids_sorted = sorted(factor_ids)

        # 生成键
        key_components = [
            start_date.replace('-', ''),
            end_date.replace('-', ''),
            hashlib.md5('_'.join(etf_codes_sorted).encode()).hexdigest()[:8],
            hashlib.md5('_'.join(factor_ids_sorted).encode()).hexdigest()[:8]
        ]

        return '_'.join(key_components)

    def _estimate_data_size(self, factors_data: Dict[str, pd.DataFrame]) -> float:
        """估算数据大小（MB）"""
        try:
            total_size = 0
            for df in factors_data.values():
                # 估算DataFrame大小
                total_size += df.memory_usage(deep=True).sum()
            return total_size / (1024 * 1024)  # 转换为MB
        except:
            return 0.0

    def _estimate_memory_cache_size(self) -> float:
        """估算内存缓存大小（MB）"""
        try:
            total_size = 0
            for data in self._memory_cache.values():
                if isinstance(data, pd.DataFrame):
                    total_size += data.memory_usage(deep=True).sum()
                elif isinstance(data, dict):
                    # 简化估算
                    total_size += len(str(data)) * 0.001  # 假设每个字符1KB
            return total_size / (1024 * 1024)  # 转换为MB
        except:
            return 0.0

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._cache_timestamps:
            return False

        timestamp = self._cache_timestamps[cache_key]
        return (datetime.now().timestamp() - timestamp) < self._cache_ttl

    @lru_cache(maxsize=128)
    def _get_cached_factor_info(self, factor_id: str) -> Dict[str, Any]:
        """获取缓存的因子信息"""
        metadata = self.factor_registry.get_factor(factor_id)
        if metadata:
            return {
                'factor_id': factor_id,
                'category': metadata.category.value,
                'description': metadata.description,
                'is_dynamic': metadata.is_dynamic
            }
        return {}

    def optimize_storage(self) -> Dict[str, int]:
        """
        优化存储

        Returns:
            优化结果统计
        """
        try:
            optimization_results = {
                "compressed_files": 0,
                "deleted_files": 0,
                "freed_space_mb": 0
            }

            # 清理过期缓存
            deleted_count = self.cleanup_expired_cache()
            optimization_results["deleted_files"] += deleted_count

            # 如果启用压缩，检查未压缩的文件
            if self.enable_compression:
                for file_path in self.dynamic_factors_dir.glob("*.pkl"):
                    # 检查是否存在对应的压缩文件
                    gz_path = file_path.with_suffix('.pkl.gz')
                    if not gz_path.exists():
                        try:
                            # 加载未压缩文件
                            with open(file_path, 'rb') as f:
                                data = pickle.load(f)

                            # 保存为压缩文件
                            with gzip.open(gz_path, 'wb') as f:
                                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                            # 删除原文件
                            original_size = file_path.stat().st_size
                            compressed_size = gz_path.stat().st_size
                            file_path.unlink()

                            optimization_results["compressed_files"] += 1
                            optimization_results["freed_space_mb"] += (original_size - compressed_size) / (1024 * 1024)

                            logger.info(f"压缩文件: {file_path.name}, 节省空间: {(original_size - compressed_size) / (1024 * 1024):.2f} MB")

                        except Exception as e:
                            logger.warning(f"压缩文件失败 {file_path}: {str(e)}")

            logger.info(f"存储优化完成: {optimization_results}")
            return optimization_results

        except Exception as e:
            logger.error(f"存储优化失败: {str(e)}")
            return {"error": str(e)}


@safe_operation
def main():
    """主函数 - 测试增强存储系统"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建增强存储系统
    storage = ETFCrossSectionStorageEnhanced(enable_compression=True)

    # 获取存储统计
    stats = storage.get_storage_statistics()
    print(f"存储统计: {stats}")

    # 优化存储
    optimization_results = storage.optimize_storage()
    print(f"优化结果: {optimization_results}")

    print("增强存储系统测试完成")


if __name__ == "__main__":
    main()