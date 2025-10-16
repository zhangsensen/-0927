#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面数据持久化存储模块
提供横截面数据的读写、缓存和管理功能
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import hashlib

from factor_system.utils import get_factor_output_dir

logger = logging.getLogger(__name__)


class ETFCrossSectionStorage:
    """ETF横截面数据存储管理器"""

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化存储管理器

        Args:
            base_dir: 基础存储目录，默认使用 factor_output/etf_cross_section
        """
        if base_dir is None:
            self.base_dir = get_factor_output_dir() / "etf_cross_section"
        else:
            self.base_dir = Path(base_dir)

        # 子目录
        self.daily_dir = self.base_dir / "daily"
        self.monthly_dir = self.base_dir / "monthly"
        self.factors_dir = self.base_dir / "factors"
        self.cache_dir = self.base_dir / "cache"
        self.metadata_dir = self.base_dir / "metadata"

        # 确保目录存在
        for dir_path in [self.daily_dir, self.monthly_dir, self.factors_dir, self.cache_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"ETF横截面存储初始化完成: {self.base_dir}")

    def _get_file_path(self, data_type: str, date: Union[str, datetime],
                      etf_code: Optional[str] = None,
                      suffix: str = "parquet") -> Path:
        """
        获取文件路径

        Args:
            data_type: 数据类型 (daily, monthly, factors, cache)
            date: 日期
            etf_code: ETF代码（可选）
            suffix: 文件后缀

        Returns:
            文件路径
        """
        if isinstance(date, datetime):
            date_str = date.strftime("%Y%m%d")
        else:
            date_str = date.replace("-", "")

        # 选择目录
        if data_type == "daily":
            dir_path = self.daily_dir
        elif data_type == "monthly":
            dir_path = self.monthly_dir
        elif data_type == "factors":
            dir_path = self.factors_dir
        elif data_type == "cache":
            dir_path = self.cache_dir
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

        # 按年月分区
        year_month = date_str[:6]  # YYYYMM

        if etf_code:
            # 单ETF文件
            filename = f"{etf_code}_{date_str}.{suffix}"
        else:
            # 横截面数据文件
            filename = f"cross_section_{year_month}.{suffix}"

        return dir_path / year_month / filename

    def save_cross_section_data(self,
                               cross_section_df: pd.DataFrame,
                               date: Union[str, datetime],
                               data_type: str = "daily") -> bool:
        """
        保存横截面数据

        Args:
            cross_section_df: 横截面数据DataFrame
            date: 日期
            data_type: 数据类型

        Returns:
            是否保存成功
        """
        try:
            if cross_section_df.empty:
                logger.warning("横截面数据为空，跳过保存")
                return False

            # 获取文件路径
            file_path = self._get_file_path(data_type, date, suffix="parquet")

            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存数据
            cross_section_df.to_parquet(file_path, index=False)

            logger.info(f"横截面数据已保存: {file_path}")
            return True

        except Exception as e:
            logger.error(f"保存横截面数据失败: {e}")
            return False

    def load_cross_section_data(self,
                               date: Union[str, datetime],
                               data_type: str = "daily") -> Optional[pd.DataFrame]:
        """
        加载横截面数据

        Args:
            date: 日期
            data_type: 数据类型

        Returns:
            横截面数据DataFrame
        """
        try:
            file_path = self._get_file_path(data_type, date, suffix="parquet")

            if not file_path.exists():
                logger.debug(f"横截面数据文件不存在: {file_path}")
                return None

            # 加载数据
            df = pd.read_parquet(file_path)

            logger.debug(f"横截面数据已加载: {file_path}, {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"加载横截面数据失败: {e}")
            return None

    def save_factor_data(self,
                         factors_df: pd.DataFrame,
                         etf_code: str,
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime]) -> bool:
        """
        保存单只ETF的因子数据

        Args:
            factors_df: 因子数据DataFrame
            etf_code: ETF代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            是否保存成功
        """
        try:
            if factors_df.empty:
                logger.warning(f"ETF {etf_code} 因子数据为空，跳过保存")
                return False

            # 生成文件名
            if isinstance(start_date, datetime):
                start_str = start_date.strftime("%Y%m%d")
            else:
                start_str = start_date.replace("-", "")

            if isinstance(end_date, datetime):
                end_str = end_date.strftime("%Y%m%d")
            else:
                end_str = end_date.replace("-", "")

            filename = f"{etf_code}_factors_{start_str}_{end_str}.parquet"
            file_path = self.factors_dir / filename

            # 保存数据
            factors_df.to_parquet(file_path, index=False)

            # 保存元数据
            metadata = {
                "etf_code": etf_code,
                "start_date": start_str,
                "end_date": end_str,
                "record_count": len(factors_df),
                "factor_columns": factors_df.columns.tolist(),
                "created_at": datetime.now().isoformat(),
                "data_hash": self._calculate_hash(factors_df)
            }

            metadata_file = self.metadata_dir / f"{etf_code}_factors_{start_str}_{end_str}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"ETF {etf_code} 因子数据已保存: {file_path}")
            return True

        except Exception as e:
            logger.error(f"保存ETF {etf_code} 因子数据失败: {e}")
            return False

    def load_factor_data(self,
                         etf_code: str,
                         start_date: Union[str, datetime],
                         end_date: Union[str, datetime]) -> Optional[pd.DataFrame]:
        """
        加载单只ETF的因子数据

        Args:
            etf_code: ETF代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            因子数据DataFrame
        """
        try:
            # 生成文件名
            if isinstance(start_date, datetime):
                start_str = start_date.strftime("%Y%m%d")
            else:
                start_str = start_date.replace("-", "")

            if isinstance(end_date, datetime):
                end_str = end_date.strftime("%Y%m%d")
            else:
                end_str = end_date.replace("-", "")

            filename = f"{etf_code}_factors_{start_str}_{end_str}.parquet"
            file_path = self.factors_dir / filename

            if not file_path.exists():
                logger.debug(f"ETF {etf_code} 因子数据文件不存在: {file_path}")
                return None

            # 加载数据
            df = pd.read_parquet(file_path)

            logger.debug(f"ETF {etf_code} 因子数据已加载: {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"加载ETF {etf_code} 因子数据失败: {e}")
            return None

    def save_composite_factors(self,
                              composite_df: pd.DataFrame,
                              etf_list: List[str],
                              start_date: Union[str, datetime],
                              end_date: Union[str, datetime]) -> bool:
        """
        保存综合因子数据

        Args:
            composite_df: 综合因子DataFrame
            etf_list: ETF列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            是否保存成功
        """
        try:
            if composite_df.empty:
                logger.warning("综合因子数据为空，跳过保存")
                return False

            # 生成文件名
            if isinstance(start_date, datetime):
                start_str = start_date.strftime("%Y%m%d")
            else:
                start_str = start_date.replace("-", "")

            if isinstance(end_date, datetime):
                end_str = end_date.strftime("%Y%m%d")
            else:
                end_str = end_date.replace("-", "")

            # 生成数据哈希用于去重
            data_hash = self._calculate_hash(composite_df)
            filename = f"composite_factors_{start_str}_{end_str}_{data_hash[:8]}.parquet"
            file_path = self.factors_dir / filename

            # 保存数据
            composite_df.to_parquet(file_path, index=False)

            # 保存元数据
            metadata = {
                "etf_count": len(etf_list),
                "etf_list": etf_list,
                "start_date": start_str,
                "end_date": end_str,
                "record_count": len(composite_df),
                "factor_columns": composite_df.columns.tolist(),
                "created_at": datetime.now().isoformat(),
                "data_hash": data_hash
            }

            metadata_file = self.metadata_dir / f"composite_factors_{start_str}_{end_str}_{data_hash[:8]}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"综合因子数据已保存: {file_path}")
            return True

        except Exception as e:
            logger.error(f"保存综合因子数据失败: {e}")
            return False

    def load_composite_factors(self,
                               start_date: Union[str, datetime],
                               end_date: Union[str, datetime]) -> Optional[pd.DataFrame]:
        """
        加载综合因子数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            综合因子DataFrame
        """
        try:
            # 查找匹配的文件
            if isinstance(start_date, datetime):
                start_str = start_date.strftime("%Y%m%d")
            else:
                start_str = start_date.replace("-", "")

            if isinstance(end_date, datetime):
                end_str = end_date.strftime("%Y%m%d")
            else:
                end_str = end_date.replace("-", "")

            # 查找文件模式
            pattern = f"composite_factors_{start_str}_{end_str}_*.parquet"
            matching_files = list(self.factors_dir.glob(pattern))

            if not matching_files:
                logger.debug(f"未找到匹配的综合因子数据: {pattern}")
                return None

            # 选择最新的文件
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

            # 加载数据
            df = pd.read_parquet(latest_file)

            logger.debug(f"综合因子数据已加载: {latest_file}, {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"加载综合因子数据失败: {e}")
            return None

    def save_cache(self, cache_key: str, data: pd.DataFrame, ttl_hours: int = 24) -> bool:
        """
        保存缓存数据

        Args:
            cache_key: 缓存键
            data: 缓存数据
            ttl_hours: 缓存有效期（小时）

        Returns:
            是否保存成功
        """
        try:
            if data.empty:
                logger.warning("缓存数据为空，跳过保存")
                return False

            # 生成缓存文件路径
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cache_{cache_hash}_{timestamp}.parquet"
            file_path = self.cache_dir / filename

            # 保存数据
            data.to_parquet(file_path, index=False)

            # 保存缓存元数据
            cache_metadata = {
                "cache_key": cache_key,
                "filename": filename,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
                "record_count": len(data),
                "data_hash": self._calculate_hash(data)
            }

            metadata_file = self.cache_dir / f"cache_{cache_hash}_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(cache_metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"缓存数据已保存: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"保存缓存数据失败: {e}")
            return False

    def load_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        加载缓存数据

        Args:
            cache_key: 缓存键

        Returns:
            缓存数据DataFrame
        """
        try:
            # 查找匹配的缓存文件
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
            pattern = f"cache_{cache_hash}_*.json"
            metadata_files = list(self.cache_dir.glob(pattern))

            if not metadata_files:
                logger.debug(f"未找到缓存: {cache_key}")
                return None

            # 检查缓存是否过期
            now = datetime.now()
            for metadata_file in metadata_files:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                expires_at = datetime.fromisoformat(metadata["expires_at"])
                if now < expires_at:
                    # 缓存未过期，加载数据
                    data_file = self.cache_dir / metadata["filename"]
                    if data_file.exists():
                        df = pd.read_parquet(data_file)
                        logger.debug(f"缓存数据已加载: {cache_key}")
                        return df
                else:
                    # 缓存已过期，删除文件
                    data_file = self.cache_dir / metadata["filename"]
                    if data_file.exists():
                        data_file.unlink()
                    metadata_file.unlink()

            logger.debug(f"缓存已过期或不存在: {cache_key}")
            return None

        except Exception as e:
            logger.error(f"加载缓存数据失败: {e}")
            return None

    def cleanup_expired_cache(self) -> int:
        """
        清理过期缓存

        Returns:
            清理的文件数量
        """
        try:
            now = datetime.now()
            cleaned_count = 0

            for metadata_file in self.cache_dir.glob("cache_*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if now >= expires_at:
                        # 删除过期的缓存文件
                        data_file = self.cache_dir / metadata["filename"]
                        if data_file.exists():
                            data_file.unlink()
                        metadata_file.unlink()
                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"清理缓存文件失败 {metadata_file}: {e}")

            logger.info(f"清理过期缓存完成，删除 {cleaned_count} 个文件")
            return cleaned_count

        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
            return 0

    def get_storage_info(self) -> Dict:
        """
        获取存储信息

        Returns:
            存储信息字典
        """
        try:
            info = {
                "base_directory": str(self.base_dir),
                "daily_files": len(list(self.daily_dir.rglob("*.parquet"))),
                "monthly_files": len(list(self.monthly_dir.rglob("*.parquet"))),
                "factors_files": len(list(self.factors_dir.glob("*.parquet"))),
                "cache_files": len(list(self.cache_dir.glob("*.parquet"))),
                "metadata_files": len(list(self.metadata_dir.glob("*.json"))),
                "total_size_mb": self._calculate_directory_size(self.base_dir) / (1024 * 1024)
            }

            return info

        except Exception as e:
            logger.error(f"获取存储信息失败: {e}")
            return {}

    def _calculate_hash(self, df: pd.DataFrame) -> str:
        """
        计算DataFrame的哈希值

        Args:
            df: DataFrame

        Returns:
            哈希值字符串
        """
        try:
            # 使用行数、列数和数据的哈希
            content = f"{len(df)}_{len(df.columns)}_{df.values.tobytes()}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(df.shape).encode()).hexdigest()

    def _calculate_directory_size(self, directory: Path) -> int:
        """
        计算目录大小（字节）

        Args:
            directory: 目录路径

        Returns:
            目录大小（字节）
        """
        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def list_available_dates(self, data_type: str = "daily") -> List[str]:
        """
        列出可用的日期

        Args:
            data_type: 数据类型

        Returns:
            日期列表
        """
        try:
            if data_type == "daily":
                dir_path = self.daily_dir
            elif data_type == "monthly":
                dir_path = self.monthly_dir
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")

            dates = set()
            for file_path in dir_path.rglob("*.parquet"):
                # 从文件名中提取日期
                parts = file_path.stem.split("_")
                if len(parts) >= 3 and parts[0] == "cross" and parts[1] == "section":
                    year_month = parts[2]
                    dates.add(year_month)

            return sorted(list(dates))

        except Exception as e:
            logger.error(f"列出可用日期失败: {e}")
            return []


# 便捷函数
def get_etf_cross_section_storage() -> ETFCrossSectionStorage:
    """获取ETF横截面存储管理器实例"""
    return ETFCrossSectionStorage()


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    storage = ETFCrossSectionStorage()

    # 显示存储信息
    info = storage.get_storage_info()
    print("ETF横截面存储信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 清理过期缓存
    cleaned = storage.cleanup_expired_cache()
    print(f"清理过期缓存: {cleaned} 个文件")