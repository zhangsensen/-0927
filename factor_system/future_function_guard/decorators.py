#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 装饰器接口
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 提供简洁的装饰器接口
- 自动化未来函数防护
- 透明的运行时验证
- 灵活的配置选项
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import GuardConfig, StrictMode, RuntimeValidationConfig
from .exceptions import FutureFunctionGuardError, RuntimeValidationError
from .runtime_validator import RuntimeValidator
from .utils import setup_logging


def future_safe(
    strict_mode: Optional[Union[str, StrictMode]] = None,
    check_coverage: bool = True,
    check_temporal_alignment: bool = True,
    check_statistical_properties: bool = True,
    correlation_threshold: Optional[float] = None,
    coverage_threshold: Optional[float] = None,
    config: Optional[GuardConfig] = None,
    logger_name: Optional[str] = None
):
    """
    未来函数安全装饰器

    Args:
        strict_mode: 严格模式设置
        check_coverage: 是否检查数据覆盖率
        check_temporal_alignment: 是否检查时间对齐
        check_statistical_properties: 是否检查统计特性
        correlation_threshold: 相关性阈值
        coverage_threshold: 覆盖率阈值
        config: 完整配置对象
        logger_name: 日志记录器名称

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置
            if config is not None:
                guard_config = config
            else:
                # 从环境获取或使用默认配置
                guard_config = GuardConfig.from_environment()

            # 应用装饰器参数覆盖
            runtime_config = guard_config.runtime_validation
            if strict_mode is not None:
                if isinstance(strict_mode, str):
                    runtime_config.strict_mode = StrictMode(strict_mode)
                else:
                    runtime_config.strict_mode = strict_mode

            if correlation_threshold is not None:
                runtime_config.correlation_threshold = correlation_threshold

            if coverage_threshold is not None:
                runtime_config.coverage_threshold = coverage_threshold

            # 设置日志
            logger = setup_logging(guard_config.logging)
            if logger_name:
                logger = logging.getLogger(logger_name)

            # 如果验证被禁用，直接执行函数
            if not guard_config.enabled or not runtime_config.enabled:
                return func(*args, **kwargs)

            # 创建运行时验证器
            validator = RuntimeValidator(runtime_config)

            try:
                # 执行原函数
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # 验证结果
                validation_issues = []

                # 检查返回结果的安全性
                if isinstance(result, (pd.Series, pd.DataFrame)):
                    validation_result = _validate_function_result(
                        result, validator, func.__name__, runtime_config
                    )
                    if not validation_result.is_valid:
                        validation_issues.append(validation_result.message)
                        if runtime_config.strict_mode == StrictMode.ENFORCED:
                            raise RuntimeValidationError(
                                f"Function {func.__name__} failed safety validation: {validation_result.message}",
                                validation_type="function_result",
                                function_name=func.__name__
                            )
                        elif runtime_config.strict_mode == StrictMode.WARN_ONLY:
                            logger.warning(f"Function {func.__name__} safety warning: {validation_result.message}")

                # 记录执行信息
                if guard_config.performance_monitoring:
                    logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")

                return result

            except Exception as e:
                logger.error(f"Function {func.__name__} execution failed: {e}")
                if isinstance(e, (RuntimeValidationError, FutureFunctionGuardError)):
                    raise
                else:
                    # 包装其他异常
                    raise FutureFunctionGuardError(
                        f"Function {func.__name__} execution error: {e}",
                        function_name=func.__name__,
                        cause=e
                    ) from e

        return wrapper
    return decorator


def safe_shift(
    max_periods: Optional[int] = None,
    allow_negative: bool = False,
    strict_mode: Optional[Union[str, StrictMode]] = None,
    config: Optional[GuardConfig] = None
):
    """
    安全shift装饰器

    Args:
        max_periods: 最大允许shift周期
        allow_negative: 是否允许负数shift
        strict_mode: 严格模式
        config: 配置对象

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置
            if config is not None:
                guard_config = config
            else:
                guard_config = GuardConfig.from_environment()

            if strict_mode is not None:
                if isinstance(strict_mode, str):
                    current_strict = StrictMode(strict_mode)
                else:
                    current_strict = strict_mode
            else:
                current_strict = guard_config.runtime_validation.strict_mode

            # 检查参数中的shift周期
            shift_periods = None

            # 从位置参数查找
            for arg in args:
                if isinstance(arg, (int, float)):
                    shift_periods = int(arg)
                    break

            # 从关键字参数查找
            if shift_periods is None:
                for key, value in kwargs.items():
                    if key.lower() in ['periods', 'n', 'shift', 'period'] and isinstance(value, (int, float)):
                        shift_periods = int(value)
                        break

            # 验证shift参数
            if shift_periods is not None:
                if not allow_negative and shift_periods < 0:
                    error_msg = f"Negative shift not allowed: {shift_periods}"
                    if current_strict == StrictMode.ENFORCED:
                        raise RuntimeValidationError(
                            error_msg,
                            validation_type="shift_validation",
                            periods=shift_periods
                        )
                    elif current_strict == StrictMode.WARN_ONLY:
                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"{func.__name__}: {error_msg}")

                if max_periods is not None and abs(shift_periods) > max_periods:
                    error_msg = f"Shift period {shift_periods} exceeds maximum {max_periods}"
                    if current_strict == StrictMode.ENFORCED:
                        raise RuntimeValidationError(
                            error_msg,
                            validation_type="shift_validation",
                            periods=shift_periods,
                            max_periods=max_periods
                        )
                    elif current_strict == StrictMode.WARN_ONLY:
                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"{func.__name__}: {error_msg}")

            # 执行原函数
            return func(*args, **kwargs)

        return wrapper
    return decorator


def monitor_factor_health(
    factor_id: Optional[str] = None,
    timeframe: Optional[str] = None,
    strict_mode: bool = False,
    config: Optional[GuardConfig] = None
):
    """
    因子健康监控装饰器

    Args:
        factor_id: 因子ID（可选，会从函数名推断）
        timeframe: 时间框架
        strict_mode: 严格模式
        config: 配置对象

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置
            if config is not None:
                guard_config = config
            else:
                guard_config = GuardConfig.from_environment()

            if not guard_config.enabled or not guard_config.health_monitor.enabled:
                return func(*args, **kwargs)

            # 推断因子ID
            inferred_factor_id = factor_id or func.__name__
            inferred_timeframe = timeframe or "daily"

            # 执行原函数
            result = func(*args, **kwargs)

            # 健康检查
            if isinstance(result, (pd.Series, pd.DataFrame)):
                from .health_monitor import HealthMonitor

                health_monitor = HealthMonitor(guard_config.health_monitor)

                if isinstance(result, pd.Series):
                    health_monitor.check_factor_health(result, inferred_factor_id, strict_mode)
                else:
                    # DataFrame情况，检查每一列
                    for col in result.columns:
                        health_monitor.check_factor_health(result[col], col, strict_mode)

            return result

        return wrapper
    return decorator


def validate_time_series(
    require_datetime_index: bool = True,
    check_monotonic: bool = True,
    check_duplicates: bool = True,
    min_length: Optional[int] = None,
    strict_mode: Optional[Union[str, StrictMode]] = None,
    config: Optional[GuardConfig] = None
):
    """
    时间序列验证装饰器

    Args:
        require_datetime_index: 是否要求DatetimeIndex
        check_monotonic: 是否检查单调性
        check_duplicates: 是否检查重复索引
        min_length: 最小长度要求
        strict_mode: 严格模式
        config: 配置对象

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 执行原函数
            result = func(*args, **kwargs)

            # 获取配置
            if config is not None:
                guard_config = config
            else:
                guard_config = GuardConfig.from_environment()

            if strict_mode is not None:
                if isinstance(strict_mode, str):
                    current_strict = StrictMode(strict_mode)
                else:
                    current_strict = strict_mode
            else:
                current_strict = guard_config.runtime_validation.strict_mode

            # 如果验证被禁用，直接返回
            if not guard_config.enabled or not guard_config.runtime_validation.enabled:
                return result

            # 验证时间序列数据
            if isinstance(result, (pd.Series, pd.DataFrame)):
                issues = []

                # 检查索引类型
                if require_datetime_index and not isinstance(result.index, pd.DatetimeIndex):
                    issues.append("Index is not DatetimeIndex")

                # 检查单调性
                if check_monotonic and not result.index.is_monotonic_increasing:
                    issues.append("Index is not monotonic increasing")

                # 检查重复索引
                if check_duplicates and result.index.duplicated().any():
                    dup_count = result.index.duplicated().sum()
                    issues.append(f"Found {dup_count} duplicate indices")

                # 检查最小长度
                if min_length is not None and len(result) < min_length:
                    issues.append(f"Length {len(result)} is less than required {min_length}")

                # 处理验证结果
                if issues:
                    error_msg = f"Time series validation failed: {'; '.join(issues)}"
                    if current_strict == StrictMode.ENFORCED:
                        raise RuntimeValidationError(
                            error_msg,
                            validation_type="time_series_validation",
                            issues=issues
                        )
                    elif current_strict == StrictMode.WARN_ONLY:
                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.warning(f"{func.__name__}: {error_msg}")

            return result

        return wrapper
    return decorator


def batch_safe(
    batch_size: Optional[int] = None,
    validate_batch: bool = True,
    aggregate_results: bool = True,
    config: Optional[GuardConfig] = None
):
    """
    批量处理安全装饰器

    Args:
        batch_size: 批次大小
        validate_batch: 是否验证每个批次
        aggregate_results: 是否聚合结果
        config: 配置对象

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取配置
            if config is not None:
                guard_config = config
            else:
                guard_config = GuardConfig.from_environment()

            # 查找批处理参数
            batch_param_name = None
            batch_data = None

            # 在参数中查找可能的数据列表
            for i, arg in enumerate(args):
                if isinstance(arg, (list, tuple)) and len(arg) > 1:
                    batch_data = arg
                    batch_param_name = f"arg_{i}"
                    break

            if batch_data is None:
                for key, value in kwargs.items():
                    if isinstance(value, (list, tuple)) and len(value) > 1:
                        batch_data = value
                        batch_param_name = key
                        break

            # 如果没有找到批处理数据，直接执行
            if batch_data is None:
                return func(*args, **kwargs)

            # 确定批次大小
            effective_batch_size = batch_size or len(batch_data)

            # 批量处理
            results = []
            validation_results = []

            for i in range(0, len(batch_data), effective_batch_size):
                batch = batch_data[i:i + effective_batch_size]

                # 创建新的参数
                new_args = list(args)
                new_kwargs = kwargs.copy()

                if batch_param_name.startswith('arg_'):
                    arg_index = int(batch_param_name.split('_')[1])
                    new_args[arg_index] = batch
                else:
                    new_kwargs[batch_param_name] = batch

                # 执行批次
                try:
                    batch_result = func(*new_args, **new_kwargs)
                    results.append(batch_result)

                    # 验证批次结果
                    if validate_batch and guard_config.enabled:
                        validation_result = _validate_function_result(
                            batch_result,
                            RuntimeValidator(guard_config.runtime_validation),
                            f"{func.__name__}_batch_{i // effective_batch_size}",
                            guard_config.runtime_validation
                        )
                        validation_results.append(validation_result)

                except Exception as e:
                    if guard_config.runtime_validation.strict_mode == StrictMode.ENFORCED:
                        raise
                    else:
                        import logging
                        logger = logging.getLogger(func.__module__)
                        logger.error(f"Batch {i // effective_batch_size} failed: {e}")
                        results.append(None)

            # 聚合结果
            if aggregate_results:
                try:
                    if all(isinstance(r, (pd.Series, pd.DataFrame)) and r is not None for r in results):
                        # pandas数据聚合
                        if isinstance(results[0], pd.Series):
                            aggregated = pd.concat(results, axis=0)
                        else:
                            aggregated = pd.concat(results, axis=0)
                        return aggregated
                    elif all(isinstance(r, (list, np.ndarray)) and r is not None for r in results):
                        # 数组聚合
                        if isinstance(results[0], np.ndarray):
                            return np.concatenate(results)
                        else:
                            return [item for sublist in results for item in sublist]
                    else:
                        return results
                except Exception as e:
                    import logging
                    logger = logging.getLogger(func.__module__)
                    logger.warning(f"Batch aggregation failed: {e}")
                    return results
            else:
                return results

        return wrapper
    return decorator


def _validate_function_result(
    result: Union[pd.Series, pd.DataFrame],
    validator: RuntimeValidator,
    function_name: str,
    config: RuntimeValidationConfig
) -> Any:
    """
    验证函数结果的安全性

    Args:
        result: 函数返回结果
        validator: 验证器实例
        function_name: 函数名称
        config: 配置

    Returns:
        验证结果
    """
    if isinstance(result, pd.Series):
        # 验证单个因子
        return validator.validate_factor_calculation(
            result, function_name, "unknown", None
        )
    elif isinstance(result, pd.DataFrame):
        # 验证因子面板
        factor_ids = list(result.columns)
        return validator.validate_batch_factors(
            result, factor_ids, "unknown", None
        )
    else:
        # 非时间序列数据，跳过验证
        from .runtime_validator import ValidationResult
        return ValidationResult.success(
            "function_result",
            message=f"Non-time-series result from {function_name}, validation skipped"
        )


# 便捷装饰器
def safe_research(func: Optional[Callable] = None) -> Union[Callable, Callable[[Callable], Callable]]:
    """研究环境安全装饰器"""
    def decorator(f: Callable) -> Callable:
        return future_safe(config=GuardConfig.preset("research"))(f)

    if func is None:
        return decorator
    else:
        return decorator(func)


def safe_production(func: Optional[Callable] = None) -> Union[Callable, Callable[[Callable], Callable]]:
    """生产环境安全装饰器"""
    def decorator(f: Callable) -> Callable:
        return future_safe(config=GuardConfig.preset("production"))(f)

    if func is None:
        return decorator
    else:
        return decorator(func)


def safe_development(func: Optional[Callable] = None) -> Union[Callable, Callable[[Callable], Callable]]:
    """开发环境安全装饰器"""
    def decorator(f: Callable) -> Callable:
        return future_safe(config=GuardConfig.preset("development"))(f)

    if func is None:
        return decorator
    else:
        return decorator(func)