#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Linus式项目路径管理 - 消灭硬编码路径

核心原则：
1. 所有路径基于 Path(__file__).resolve() 动态计算
2. 单一真相来源 - 项目根目录
3. 向后兼容 - 保留旧API但标记为deprecated
"""

from pathlib import Path
from typing import Optional


class ProjectPaths:
    """项目路径统一管理器"""

    # 🎯 单一真相来源：项目根目录
    _project_root: Optional[Path] = None

    @classmethod
    def get_project_root(cls) -> Path:
        """
        获取项目根目录（深度量化0927/）

        Returns:
            Path: 项目根目录绝对路径
        """
        if cls._project_root is None:
            # 从当前文件向上追溯到项目根目录
            current_file = Path(__file__).resolve()
            # factor_system/utils/project_paths.py -> factor_system -> 项目根
            cls._project_root = current_file.parent.parent.parent
        return cls._project_root

    @classmethod
    def get_raw_data_dir(cls, market: Optional[str] = None) -> Path:
        """
        获取原始数据目录

        Args:
            market: 市场代码（HK/US），可选

        Returns:
            Path: 原始数据目录路径
        """
        raw_dir = cls.get_project_root() / "raw"
        if market:
            return raw_dir / market.upper()
        return raw_dir

    @classmethod
    def get_factor_output_dir(cls, market: Optional[str] = None) -> Path:
        """
        获取因子输出目录

        Args:
            market: 市场代码（HK/US），可选

        Returns:
            Path: 因子输出目录路径
        """
        output_dir = cls.get_project_root() / "factor_system" / "factor_output"
        if market:
            return output_dir / market.upper()
        return output_dir

    @classmethod
    def get_screening_results_dir(cls) -> Path:
        """
        获取筛选结果目录

        Returns:
            Path: 筛选结果目录路径
        """
        return (
            cls.get_project_root()
            / "factor_system"
            / "factor_screening"
            / "screening_results"
        )

    @classmethod
    def get_logs_dir(cls, module: Optional[str] = None) -> Path:
        """
        获取日志目录

        Args:
            module: 模块名称（screening/generation/engine），可选

        Returns:
            Path: 日志目录路径
        """
        logs_dir = cls.get_project_root() / "logs"
        if module:
            return logs_dir / module
        return logs_dir

    @classmethod
    def get_cache_dir(cls, module: Optional[str] = None) -> Path:
        """
        获取缓存目录

        Args:
            module: 模块名称，可选

        Returns:
            Path: 缓存目录路径
        """
        cache_dir = cls.get_project_root() / "cache"
        if module:
            return cache_dir / module
        return cache_dir

    @classmethod
    def get_config_dir(cls, module: Optional[str] = None) -> Path:
        """
        获取配置目录

        Args:
            module: 模块名称，可选

        Returns:
            Path: 配置目录路径
        """
        if module:
            return cls.get_project_root() / "factor_system" / module / "configs"
        return cls.get_project_root() / "configs"

    @classmethod
    def ensure_directories(cls) -> None:
        """确保所有关键目录存在"""
        directories = [
            cls.get_raw_data_dir(),
            cls.get_factor_output_dir(),
            cls.get_screening_results_dir(),
            cls.get_logs_dir(),
            cls.get_cache_dir(),
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 🔧 便捷函数（向后兼容）
def get_project_root() -> Path:
    """获取项目根目录（便捷函数）"""
    return ProjectPaths.get_project_root()


def get_raw_data_dir(market: Optional[str] = None) -> Path:
    """获取原始数据目录（便捷函数）"""
    return ProjectPaths.get_raw_data_dir(market)


def get_factor_output_dir(market: Optional[str] = None) -> Path:
    """获取因子输出目录（便捷函数）"""
    return ProjectPaths.get_factor_output_dir(market)


def get_screening_results_dir() -> Path:
    """获取筛选结果目录（便捷函数）"""
    return ProjectPaths.get_screening_results_dir()


def get_logs_dir(module: Optional[str] = None) -> Path:
    """获取日志目录（便捷函数）"""
    return ProjectPaths.get_logs_dir(module)


def get_cache_dir(module: Optional[str] = None) -> Path:
    """获取缓存目录（便捷函数）"""
    return ProjectPaths.get_cache_dir(module)


def get_config_dir(module: Optional[str] = None) -> Path:
    """获取配置目录（便捷函数）"""
    return ProjectPaths.get_config_dir(module)


# 🔧 验证函数
def validate_project_structure() -> bool:
    """
    验证项目结构完整性

    Returns:
        bool: 项目结构是否完整
    """
    required_dirs = [
        "factor_system",
        "factor_system/factor_engine",
        "factor_system/factor_generation",
        "factor_system/factor_screening",
    ]

    project_root = ProjectPaths.get_project_root()
    for dir_name in required_dirs:
        if not (project_root / dir_name).exists():
            return False
    return True


if __name__ == "__main__":
    # 测试路径管理器
    print("🔧 项目路径管理器测试")
    print(f"项目根目录: {ProjectPaths.get_project_root()}")
    print(f"原始数据目录: {ProjectPaths.get_raw_data_dir()}")
    print(f"因子输出目录: {ProjectPaths.get_factor_output_dir()}")
    print(f"筛选结果目录: {ProjectPaths.get_screening_results_dir()}")
    print(f"日志目录: {ProjectPaths.get_logs_dir()}")
    print(f"缓存目录: {ProjectPaths.get_cache_dir()}")
    print(f"项目结构验证: {'✅ 通过' if validate_project_structure() else '❌ 失败'}")
