#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 统一导出接口
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 统一的模块导出接口
- 便捷的快速启动函数
- 版本信息和元数据
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# 版本信息
__version__ = "1.0.0"
__author__ = "量化首席工程师"
__email__ = "quant_engineer@example.com"
__description__ = "专业级未来函数防护组件，为量化交易系统提供多层次安全保障"

# 子模块（高级用法）
from . import health_monitor, runtime_validator, static_checker

# 核心类和配置
from .config import (
    AlertThreshold,
    CacheConfig,
    GuardConfig,
    HealthMonitorConfig,
    LoggingConfig,
    MonitoringLevel,
    RuntimeValidationConfig,
    StaticCheckConfig,
    StrictMode,
)

# 装饰器接口
from .decorators import (
    batch_safe,
    future_safe,
    monitor_factor_health,
    safe_development,
    safe_production,
    safe_research,
    safe_shift,
    validate_time_series,
)

# 异常类
from .exceptions import (
    CacheError,
    ConfigurationError,
    FutureFunctionDetectedError,
    FutureFunctionGuardError,
    HealthMonitorError,
    RuntimeValidationError,
    StaticCheckError,
    TimeSeriesSafetyError,
    create_error,
    format_exception_for_logging,
)
from .guard import FutureFunctionGuard

# 工具函数
from .utils import (
    FileCache,
    SimpleCache,
    batch_processing,
    calculate_factor_statistics,
    create_directory_if_not_exists,
    format_duration,
    generate_timestamp,
    get_file_hash,
    is_valid_time_range,
    merge_dicts,
    normalize_factor_name,
    safe_divide,
    setup_logging,
    validate_time_series_data,
)

# ==================== 便捷函数 ====================


def create_guard(
    mode: str = "auto", strict_mode: str = "warn_only", **kwargs
) -> FutureFunctionGuard:
    """
    创建未来函数防护组件的便捷函数

    Args:
        mode: 运行模式 (auto, development, research, production)
        strict_mode: 严格模式 (disabled, warn_only, enforced)
        **kwargs: 其他配置参数

    Returns:
        配置好的FutureFunctionGuard实例

    Examples:
        >>> # 使用默认配置
        >>> guard = create_guard()

        >>> # 使用预设环境
        >>> guard = create_guard(mode="production")

        >>> # 自定义配置
        >>> guard = create_guard(
        ...     mode="research",
        ...     strict_mode="warn_only",
        ...     correlation_threshold=0.95
        ... )
    """
    if mode in ["development", "research", "production"]:
        config = GuardConfig.preset(mode)
    else:
        config = GuardConfig.from_environment()

    # 应用自定义参数
    if strict_mode:
        config.runtime_validation.strict_mode = StrictMode(strict_mode)

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.runtime_validation, key):
            setattr(config.runtime_validation, key, value)
        elif hasattr(config.health_monitor, key):
            setattr(config.health_monitor, key, value)

    return FutureFunctionGuard(config)


def quick_check(target, mode: str = "auto", output_format: str = "text") -> str:
    """
    快速静态检查代码的便捷函数

    Args:
        target: 检查目标（文件、目录或文件列表）
        mode: 运行模式
        output_format: 输出格式 (text, json, markdown)

    Returns:
        检查报告

    Examples:
        >>> # 检查单个文件
        >>> report = quick_check("my_factor.py")

        >>> # 检查目录
        >>> report = quick_check("./src/", output_format="markdown")

        >>> # 检查多个文件
        >>> report = quick_check(["file1.py", "file2.py"])
    """
    guard = create_guard(mode=mode)
    return guard.generate_static_report(target, output_format)


def validate_factors(
    factor_data,
    factor_ids: Optional[list] = None,
    timeframe: str = "daily",
    mode: str = "auto",
) -> dict:
    """
    快速验证因子数据的便捷函数

    Args:
        factor_data: 因子数据 (Series或DataFrame)
        factor_ids: 因子ID列表
        timeframe: 时间框架
        mode: 运行模式

    Returns:
        验证结果

    Examples:
        >>> # 验证单个因子
        >>> result = validate_factors(factor_series, "my_factor")

        >>> # 验证多个因子
        >>> result = validate_factors(factor_df, timeframe="60min")
    """
    guard = create_guard(mode=mode)

    if isinstance(factor_data, (pd.Series, pd.DataFrame)):
        if isinstance(factor_data, pd.Series):
            factor_id = factor_ids[0] if factor_ids else "factor"
            return guard.validate_factor_calculation(factor_data, factor_id, timeframe)
        else:
            return guard.validate_time_series_data(factor_data, factor_ids, timeframe)
    else:
        raise ValueError("factor_data must be pandas Series or DataFrame")


def monitor_health(
    factor_data, factor_id: str, strict_mode: bool = False, mode: str = "auto"
) -> dict:
    """
    快速监控因子健康的便捷函数

    Args:
        factor_data: 因子数据
        factor_id: 因子ID
        strict_mode: 严格模式
        mode: 运行模式

    Returns:
        健康检查结果

    Examples:
        >>> result = monitor_health(factor_series, "my_factor")
        >>> print(f"Quality score: {result['quality_score']:.1f}")
    """
    guard = create_guard(mode=mode)
    return guard.check_factor_health(factor_data, factor_id, strict_mode)


def comprehensive_check(
    code_paths: Optional[list] = None,
    data_dict: Optional[dict] = None,
    mode: str = "auto",
) -> dict:
    """
    综合安全检查的便捷函数

    Args:
        code_paths: 代码路径列表
        data_dict: 数据字典 {name: data}
        mode: 运行模式

    Returns:
        综合检查结果

    Examples:
        >>> # 检查代码和数据
        >>> result = comprehensive_check(
        ...     code_paths=["./src/"],
        ...     data_dict={"factor1": factor_data}
        ... )
        >>> print(f"Overall status: {result['overall_status']}")
    """
    guard = create_guard(mode=mode)
    return guard.comprehensive_security_check(code_paths, data_dict)


# ==================== 环境预设 ====================


def development_guard(**kwargs) -> FutureFunctionGuard:
    """创建开发环境防护组件"""
    return create_guard(mode="development", **kwargs)


def research_guard(**kwargs) -> FutureFunctionGuard:
    """创建研究环境防护组件"""
    return create_guard(mode="research", **kwargs)


def production_guard(**kwargs) -> FutureFunctionGuard:
    """创建生产环境防护组件"""
    return create_guard(mode="production", **kwargs)


# ==================== 版本信息 ====================


def get_version() -> str:
    """获取版本信息"""
    return __version__


def get_info() -> dict:
    """获取组件详细信息"""
    return {
        "name": "future_function_guard",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "python_requires": ">=3.8",
        "dependencies": ["pandas>=1.3.0", "numpy>=1.20.0"],
        "optional_dependencies": ["scipy>=1.7.0"],  # 用于统计检查
    }


# ==================== 兼容性支持 ====================

# 为了向后兼容，提供一些别名
FutureGuard = FutureFunctionGuard
GuardConfig = GuardConfig
StaticChecker = static_checker.StaticChecker
RuntimeValidator = runtime_validator.RuntimeValidator
HealthMonitor = health_monitor.HealthMonitor

# ==================== 导出列表 ====================

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__description__",
    "get_version",
    "get_info",
    # 核心类
    "FutureFunctionGuard",
    "GuardConfig",
    # 配置类
    "StaticCheckConfig",
    "RuntimeValidationConfig",
    "HealthMonitorConfig",
    "CacheConfig",
    "LoggingConfig",
    "StrictMode",
    "MonitoringLevel",
    "AlertThreshold",
    # 异常类
    "FutureFunctionGuardError",
    "StaticCheckError",
    "RuntimeValidationError",
    "TimeSeriesSafetyError",
    "ConfigurationError",
    "HealthMonitorError",
    "CacheError",
    "FutureFunctionDetectedError",
    "create_error",
    "format_exception_for_logging",
    # 装饰器
    "future_safe",
    "safe_shift",
    "monitor_factor_health",
    "validate_time_series",
    "batch_safe",
    "safe_research",
    "safe_production",
    "safe_development",
    # 工具函数
    "SimpleCache",
    "FileCache",
    "setup_logging",
    "validate_time_series_data",
    "calculate_factor_statistics",
    "generate_timestamp",
    "format_duration",
    "safe_divide",
    "normalize_factor_name",
    "create_directory_if_not_exists",
    "get_file_hash",
    "batch_processing",
    "is_valid_time_range",
    "merge_dicts",
    # 便捷函数
    "create_guard",
    "quick_check",
    "validate_factors",
    "monitor_health",
    "comprehensive_check",
    # 环境预设
    "development_guard",
    "research_guard",
    "production_guard",
    # 子模块
    "static_checker",
    "runtime_validator",
    "health_monitor",
    # 向后兼容别名
    "FutureGuard",
    "StaticChecker",
    "RuntimeValidator",
    "HealthMonitor",
]

# ==================== 模块初始化 ====================


def _init_module():
    """模块初始化"""
    import logging

    # 设置模块级日志
    logger = logging.getLogger(__name__)
    logger.debug(f"FutureFunctionGuard v{__version__} initialized")

    # 检查可选依赖
    try:
        import scipy

        logger.debug("SciPy available - advanced statistical checks enabled")
    except ImportError:
        logger.debug("SciPy not available - some statistical checks will be skipped")

    try:
        import sklearn

        logger.debug("Scikit-learn available - advanced ML checks enabled")
    except ImportError:
        logger.debug("Scikit-learn not available - ML checks will be skipped")


# 执行初始化
_init_module()
