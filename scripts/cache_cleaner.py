#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面系统缓存清理工具
安全清理所有内存和存储缓存，确保系统从干净状态启动
"""

import argparse
import gc
import glob
import logging
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CacheCleaner:
    """缓存清理器"""

    def __init__(self, base_dir: str = None):
        """
        初始化缓存清理器

        Args:
            base_dir: 基础目录
        """
        if base_dir is None:
            # 获取项目根目录
            current_dir = Path(__file__).parent.parent
            self.base_dir = current_dir
        else:
            self.base_dir = Path(base_dir)

        self.backup_dir = (
            self.base_dir / "cache_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"缓存清理器初始化完成")
        logger.info(f"项目根目录: {self.base_dir}")
        logger.info(f"备份目录: {self.backup_dir}")

    def backup_data(self, source_path: Path, description: str) -> bool:
        """
        备份数据

        Args:
            source_path: 源路径
            description: 描述

        Returns:
            备份是否成功
        """
        try:
            if not source_path.exists():
                logger.debug(f"路径不存在，跳过备份: {source_path}")
                return True

            # 创建备份路径
            backup_path = self.backup_dir / description
            backup_path.mkdir(parents=True, exist_ok=True)

            if source_path.is_dir():
                # 备份目录
                target_path = backup_path / source_path.name
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source_path, target_path)
                logger.info(f"备份目录: {source_path} -> {target_path}")
            else:
                # 备份文件
                target_path = backup_path / source_path.name
                shutil.copy2(source_path, target_path)
                logger.info(f"备份文件: {source_path} -> {target_path}")

            return True

        except Exception as e:
            logger.error(f"备份失败 {source_path}: {str(e)}")
            return False

    def clear_memory_caches(self) -> bool:
        """
        清理内存缓存

        Returns:
            清理是否成功
        """
        try:
            logger.info("开始清理内存缓存...")

            # 清理Python模块缓存
            modules_to_clear = [
                "factor_system.factor_engine.factors.etf_cross_section.factor_registry",
                "factor_system.factor_engine.factors.etf_cross_section.etf_factor_factory",
                "factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_enhanced",
                "factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_strategy_enhanced",
                "factor_system.factor_engine.factors.etf_cross_section.etf_cross_section_storage_enhanced",
            ]

            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    logger.debug(f"清理模块缓存: {module_name}")

            # 强制垃圾回收
            logger.info("执行垃圾回收...")
            collected = gc.collect()
            logger.info(f"垃圾回收完成，回收对象数: {collected}")

            return True

        except Exception as e:
            logger.error(f"清理内存缓存失败: {str(e)}")
            return False

    def clear_storage_caches(self) -> bool:
        """
        清理存储缓存

        Returns:
            清理是否成功
        """
        try:
            logger.info("开始清理存储缓存...")

            # 查找ETF横截面相关缓存目录
            cache_patterns = [
                self.base_dir / "factor_system" / "factor_output" / "etf_cross_section",
                self.base_dir / "factor_system" / "factor_output" / "cross_section",
                self.base_dir / "raw" / "ETF" / "daily" / "cache",
                self.base_dir / "raw" / "ETF" / "moneyflow" / "cache",
                self.base_dir / "raw" / "ETF" / "moneyflow_market" / "cache",
            ]

            # 查找__pycache__目录
            pycache_patterns = [
                self.base_dir / "factor_system" / "factor_engine" / "__pycache__",
                self.base_dir
                / "factor_system"
                / "factor_engine"
                / "factors"
                / "__pycache__",
                self.base_dir
                / "factor_system"
                / "factor_engine"
                / "factors"
                / "etf_cross_section"
                / "__pycache__",
                self.base_dir
                / "factor_system"
                / "factor_engine"
                / "providers"
                / "__pycache__",
            ]

            # 查找临时文件
            temp_patterns = [
                self.base_dir / "*.tmp",
                self.base_dir / "*.temp",
                self.base_dir / ".pytest_cache",
                self.base_dir / ".mypy_cache",
            ]

            # 清理存储缓存
            cleared_count = 0
            cleared_size = 0

            for pattern in cache_patterns:
                if pattern.exists():
                    # 备份
                    self.backup_data(pattern, f"storage_cache/{pattern.name}")

                    # 获取大小
                    size = self._get_directory_size(pattern)
                    cleared_size += size
                    cleared_count += 1

                    # 清理
                    if pattern.is_dir():
                        # 清理目录内容但保留目录结构
                        for item in pattern.glob("*"):
                            if item.is_file():
                                item.unlink()
                                logger.debug(f"删除缓存文件: {item}")
                            elif item.is_dir():
                                shutil.rmtree(item)
                                logger.debug(f"删除缓存目录: {item}")
                    else:
                        pattern.unlink()
                        logger.debug(f"删除缓存文件: {pattern}")

            # 清理__pycache__目录
            for pattern in pycache_patterns:
                if pattern.exists():
                    self.backup_data(pattern, f"pycache/{pattern.name}")
                    size = self._get_directory_size(pattern)
                    cleared_size += size
                    cleared_count += 1

                    shutil.rmtree(pattern)
                    logger.debug(f"删除__pycache__目录: {pattern}")

            # 清理临时文件
            for pattern in temp_patterns:
                for item in self.base_dir.glob(pattern.name):
                    if item.exists():
                        self.backup_data(item, f"temp_files/{item.name}")
                        if item.is_file():
                            size = item.stat().st_size
                            cleared_size += size
                            cleared_count += 1
                            item.unlink()
                            logger.debug(f"删除临时文件: {item}")
                        elif item.is_dir():
                            size = self._get_directory_size(item)
                            cleared_size += size
                            cleared_count += 1
                            shutil.rmtree(item)
                            logger.debug(f"删除临时目录: {item}")

            logger.info(
                f"存储缓存清理完成: {cleared_count} 项, {cleared_size / (1024*1024):.2f} MB"
            )
            return True

        except Exception as e:
            logger.error(f"清理存储缓存失败: {str(e)}")
            return False

    def clear_dynamic_factor_caches(self) -> bool:
        """
        清理动态因子专用缓存

        Returns:
            清理是否成功
        """
        try:
            logger.info("开始清理动态因子缓存...")

            # 动态因子缓存目录
            dynamic_cache_patterns = [
                self.base_dir
                / "factor_system"
                / "factor_output"
                / "etf_cross_section"
                / "dynamic_factors",
                self.base_dir
                / "factor_system"
                / "factor_output"
                / "etf_cross_section"
                / "factor_cache",
                self.base_dir
                / "factor_system"
                / "factor_output"
                / "etf_cross_section"
                / "factor_metadata",
            ]

            cleared_count = 0
            cleared_size = 0

            for pattern in dynamic_cache_patterns:
                if pattern.exists():
                    # 备份
                    self.backup_data(pattern, f"dynamic_cache/{pattern.name}")

                    # 获取大小
                    size = self._get_directory_size(pattern)
                    cleared_size += size
                    cleared_count += 1

                    # 清理
                    if pattern.is_dir():
                        shutil.rmtree(pattern)
                        logger.debug(f"删除动态因子缓存目录: {pattern}")
                    else:
                        pattern.unlink()
                        logger.debug(f"删除动态因子缓存文件: {pattern}")

            logger.info(
                f"动态因子缓存清理完成: {cleared_count} 项, {cleared_size / (1024*1024):.2f} MB"
            )
            return True

        except Exception as e:
            logger.error(f"清理动态因子缓存失败: {str(e)}")
            return False

    def reset_global_registry(self) -> bool:
        """
        重置全局注册表

        Returns:
            重置是否成功
        """
        try:
            logger.info("开始重置全局注册表...")

            # 重置因子注册表全局实例
            registry_file = (
                self.base_dir
                / "factor_system"
                / "factor_engine"
                / "factors"
                / "etf_cross_section"
                / "factor_registry.py"
            )

            if registry_file.exists():
                # 备份注册表文件
                self.backup_data(registry_file, "registry/")

                # 重新导入模块以重置全局变量
                import importlib

                import factor_system.factor_engine.factors.etf_cross_section.factor_registry as registry_module

                # 重置全局注册表实例
                if hasattr(registry_module, "_global_registry"):
                    old_registry = registry_module._global_registry
                    registry_module._global_registry = (
                        registry_module.ETFFactorRegistry()
                    )
                    logger.info("全局因子注册表已重置")

                # 清理模块缓存
                importlib.reload(registry_module)

            return True

        except Exception as e:
            logger.error(f"重置全局注册表失败: {str(e)}")
            return False

    def _get_directory_size(self, path: Path) -> int:
        """
        获取目录大小

        Args:
            path: 路径

        Returns:
            大小（字节）
        """
        try:
            if path.is_file():
                return path.stat().st_size
            elif path.is_dir():
                total_size = 0
                for item in path.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                return total_size
            else:
                return 0
        except:
            return 0

    def run_full_cleanup(self) -> Dict[str, bool]:
        """
        运行完整清理流程

        Returns:
            清理结果字典
        """
        logger.info("=" * 60)
        logger.info("开始执行完整缓存清理流程")
        logger.info("=" * 60)

        results = {}

        try:
            # 步骤1：清理内存缓存
            logger.info("步骤1：清理内存缓存")
            results["memory_cache"] = self.clear_memory_caches()

            # 步骤2：清理存储缓存
            logger.info("步骤2：清理存储缓存")
            results["storage_cache"] = self.clear_storage_caches()

            # 步骤3：清理动态因子缓存
            logger.info("步骤3：清理动态因子缓存")
            results["dynamic_cache"] = self.clear_dynamic_factor_caches()

            # 步骤4：重置全局注册表
            logger.info("步骤4：重置全局注册表")
            results["registry_reset"] = self.reset_global_registry()

            # 最终垃圾回收
            logger.info("步骤5：最终垃圾回收")
            gc.collect()

            logger.info("=" * 60)
            logger.info("缓存清理流程完成")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"缓存清理流程失败: {str(e)}")
            results["error"] = str(e)
            return results

    def get_cleanup_summary(self) -> Dict[str, Any]:
        """
        获取清理摘要

        Returns:
            清理摘要字典
        """
        try:
            summary = {
                "backup_directory": str(self.backup_dir),
                "backup_exists": self.backup_dir.exists(),
                "backup_size_mb": self._get_directory_size(self.backup_dir)
                / (1024 * 1024),
                "timestamp": datetime.now().isoformat(),
                "base_directory": str(self.base_dir),
            }

            # 统计备份内容
            backup_contents = {}
            if self.backup_dir.exists():
                for item in self.backup_dir.iterdir():
                    if item.is_dir():
                        backup_contents[item.name] = {
                            "size_mb": self._get_directory_size(item) / (1024 * 1024),
                            "item_count": len(list(item.rglob("*"))),
                        }

            summary["backup_contents"] = backup_contents

            return summary

        except Exception as e:
            logger.error(f"获取清理摘要失败: {str(e)}")
            return {"error": str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ETF横截面系统缓存清理工具")
    parser.add_argument("--base-dir", help="项目根目录")
    parser.add_argument(
        "--dry-run", action="store_true", help="仅显示将要清理的内容，不实际执行"
    )
    parser.add_argument("--memory-only", action="store_true", help="仅清理内存缓存")
    parser.add_argument("--storage-only", action="store_true", help="仅清理存储缓存")

    args = parser.parse_args()

    # 创建缓存清理器
    cleaner = CacheCleaner(args.base_dir)

    if args.dry_run:
        logger.info("DRY RUN模式：仅显示将要清理的内容")
        # TODO: 实现dry run逻辑
        return

    if args.memory_only:
        results = {"memory_cache": cleaner.clear_memory_caches()}
    elif args.storage_only:
        results = {
            "storage_cache": cleaner.clear_storage_caches(),
            "dynamic_cache": cleaner.clear_dynamic_factor_caches(),
        }
    else:
        results = cleaner.run_full_cleanup()

    # 显示结果
    logger.info("\n清理结果:")
    for key, value in results.items():
        status = "✅ 成功" if value else "❌ 失败"
        logger.info(f"  {key}: {status}")

    # 显示摘要
    summary = cleaner.get_cleanup_summary()
    logger.info(f"\n备份信息: {summary['backup_directory']}")
    logger.info(f"备份大小: {summary['backup_size_mb']:.2f} MB")


if __name__ == "__main__":
    import argparse

    main()
