#!/usr/bin/env python3
"""
VectorBT配置管理 - 缓存与并行设置
"""

import logging
from typing import Any, Dict

import vectorbt as vbt

logger = logging.getLogger(__name__)


class VBTConfigManager:
    """VectorBT配置管理器"""

    def __init__(self):
        self.cache_enabled = False
        self.parallel_enabled = False
        self._original_settings = {}

    def configure(self, config: Dict[str, Any]):
        """配置VectorBT性能参数

        Args:
            config: 配置字典，包含:
                - enable_cache: 是否启用缓存
                - cache_mode: 缓存模式 (full/partial/off)
                - n_jobs: 并行任务数
                - chunked: 分块处理
        """
        # 保存原始设置
        self._save_original_settings()

        # 配置缓存
        if config.get("enable_cache", False):
            self._enable_cache(config.get("cache_mode", "full"))

        # 配置并行
        if config.get("n_jobs", 1) != 1:
            self._enable_parallel(
                config.get("n_jobs", -1), config.get("chunked", "auto")
            )

        logger.info(
            f"VectorBT配置完成: cache={self.cache_enabled}, parallel={self.parallel_enabled}"
        )

    def _save_original_settings(self):
        """保存原始设置"""
        try:
            self._original_settings = {
                "caching_enabled": vbt.settings.caching.enabled,
                "n_jobs": getattr(vbt.settings, "n_jobs", 1),
            }
        except Exception as e:
            logger.warning(f"保存原始设置失败: {e}")

    def _enable_cache(self, mode: str = "full"):
        """启用缓存

        Args:
            mode: 缓存模式 (full/partial/off)
        """
        try:
            if mode == "off":
                vbt.settings.caching.enabled = False
                self.cache_enabled = False
                logger.info("VectorBT缓存已禁用")
            else:
                vbt.settings.caching.enabled = True
                # 尝试设置缓存模式（如果API支持）
                if hasattr(vbt.settings.caching, "mode"):
                    vbt.settings.caching.mode = mode

                # 启用所有模块缓存
                if hasattr(vbt.settings.caching, "enabled_modules"):
                    vbt.settings.caching.enabled_modules = ["*"]

                self.cache_enabled = True
                logger.info(f"VectorBT缓存已启用: mode={mode}")
        except Exception as e:
            logger.warning(f"启用缓存失败: {e}")

    def _enable_parallel(self, n_jobs: int = -1, chunked: str = "auto"):
        """启用并行计算

        Args:
            n_jobs: 并行任务数 (-1=所有核心)
            chunked: 分块处理模式
        """
        try:
            # 设置并行任务数
            if hasattr(vbt.settings, "n_jobs"):
                vbt.settings.n_jobs = n_jobs
                logger.info(f"VectorBT并行已启用: n_jobs={n_jobs}")

            # 设置分块处理
            if hasattr(vbt.settings, "chunked"):
                vbt.settings.chunked = chunked
                logger.info(f"VectorBT分块处理: chunked={chunked}")

            self.parallel_enabled = True
        except Exception as e:
            logger.warning(f"启用并行失败: {e}")

    def restore(self):
        """恢复原始设置"""
        try:
            if "caching_enabled" in self._original_settings:
                vbt.settings.caching.enabled = self._original_settings[
                    "caching_enabled"
                ]

            if "n_jobs" in self._original_settings and hasattr(vbt.settings, "n_jobs"):
                vbt.settings.n_jobs = self._original_settings["n_jobs"]

            logger.info("VectorBT设置已恢复")
        except Exception as e:
            logger.warning(f"恢复设置失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计

        Returns:
            缓存统计字典
        """
        stats = {"enabled": self.cache_enabled, "hits": 0, "misses": 0, "hit_rate": 0.0}

        try:
            # 尝试获取缓存统计（如果API支持）
            if hasattr(vbt, "Cache") and hasattr(vbt.Cache, "stats"):
                cache_stats = vbt.Cache.stats()
                stats.update(cache_stats)
        except Exception as e:
            logger.debug(f"获取缓存统计失败: {e}")

        return stats


# 全局配置管理器实例
_config_manager = VBTConfigManager()


def configure_vbt(config: Dict[str, Any]):
    """配置VectorBT（全局函数）"""
    _config_manager.configure(config)


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计（全局函数）"""
    return _config_manager.get_cache_stats()


def restore_vbt_settings():
    """恢复VectorBT设置（全局函数）"""
    _config_manager.restore()
