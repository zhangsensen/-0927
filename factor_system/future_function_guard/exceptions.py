#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数防护组件 - 自定义异常类
作者：量化首席工程师
版本：1.0.0
日期：2025-10-17

功能：
- 定义未来函数防护相关的异常类
- 提供详细的错误信息和上下文
- 支持异常链和错误追踪
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FutureFunctionGuardError(Exception):
    """未来函数防护组件基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.cause = cause

    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        if self.cause:
            base_msg += f" (Caused by: {self.cause})"
        return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }


class StaticCheckError(FutureFunctionGuardError):
    """静态检查异常"""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        issue_type: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if file_path:
            context["file_path"] = file_path
        if line_number:
            context["line_number"] = line_number
        if issue_type:
            context["issue_type"] = issue_type

        super().__init__(
            message=message, error_code="STATIC_CHECK_ERROR", context=context, **kwargs
        )


class RuntimeValidationError(FutureFunctionGuardError):
    """运行时验证异常"""

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        data_shape: Optional[tuple] = None,
        factor_id: Optional[str] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        # 构建上下文
        final_context = context or {}
        if validation_type:
            final_context["validation_type"] = validation_type
        if data_shape:
            final_context["data_shape"] = data_shape
        if factor_id:
            final_context["factor_id"] = factor_id

        # 保存validation_type作为实例属性
        self.validation_type = validation_type

        super().__init__(
            message=message,
            error_code=error_code or "RUNTIME_VALIDATION_ERROR",
            context=final_context,
            cause=cause,
        )


class TimeSeriesSafetyError(RuntimeValidationError):
    """时间序列安全异常"""

    def __init__(
        self,
        message: str,
        violation_type: Optional[str] = None,
        timeframe: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if violation_type:
            context["violation_type"] = violation_type
        if timeframe:
            context["timeframe"] = timeframe

        super().__init__(
            message=message,
            validation_type="time_series_safety",
            error_code="TIME_SERIES_SAFETY_ERROR",
            context=context,
            **kwargs,
        )


class ConfigurationError(FutureFunctionGuardError):
    """配置错误异常"""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        # 构建上下文
        final_context = context or {}
        if config_key:
            final_context["config_key"] = config_key
        if config_value is not None:
            final_context["config_value"] = str(config_value)

        super().__init__(
            message=message,
            error_code=error_code or "CONFIGURATION_ERROR",
            context=final_context,
            cause=cause,
        )


class HealthMonitorError(FutureFunctionGuardError):
    """健康监控异常"""

    def __init__(
        self,
        message: str,
        health_check_type: Optional[str] = None,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        actual_value: Optional[float] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if health_check_type:
            context["health_check_type"] = health_check_type
        if metric_name:
            context["metric_name"] = metric_name
        if threshold is not None:
            context["threshold"] = threshold
        if actual_value is not None:
            context["actual_value"] = actual_value

        super().__init__(
            message=message,
            error_code="HEALTH_MONITOR_ERROR",
            context=context,
            **kwargs,
        )


class CacheError(FutureFunctionGuardError):
    """缓存操作异常"""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if cache_key:
            context["cache_key"] = cache_key
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message, error_code="CACHE_ERROR", context=context, **kwargs
        )


class FutureFunctionDetectedError(FutureFunctionGuardError):
    """检测到未来函数使用异常"""

    def __init__(
        self,
        message: str,
        function_name: Optional[str] = None,
        parameters: Optional[str] = None,
        severity: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if function_name:
            context["function_name"] = function_name
        if parameters:
            context["parameters"] = parameters
        if severity:
            context["severity"] = severity

        super().__init__(
            message=message,
            error_code="FUTURE_FUNCTION_DETECTED",
            context=context,
            **kwargs,
        )


# 异常映射，便于错误分类和处理
ERROR_MAPPING = {
    "static_check": StaticCheckError,
    "runtime_validation": RuntimeValidationError,
    "time_series_safety": TimeSeriesSafetyError,
    "configuration": ConfigurationError,
    "health_monitor": HealthMonitorError,
    "cache": CacheError,
    "future_function": FutureFunctionDetectedError,
}


def create_error(error_type: str, message: str, **kwargs) -> FutureFunctionGuardError:
    """
    工厂函数：根据错误类型创建相应的异常实例

    Args:
        error_type: 错误类型
        message: 错误信息
        **kwargs: 其他参数

    Returns:
        相应的异常实例
    """
    error_class = ERROR_MAPPING.get(error_type, FutureFunctionGuardError)
    return error_class(message=message, **kwargs)


def format_exception_for_logging(exception: Exception) -> Dict[str, Any]:
    """
    格式化异常信息，便于日志记录

    Args:
        exception: 异常实例

    Returns:
        格式化的异常信息
    """
    if isinstance(exception, FutureFunctionGuardError):
        return exception.to_dict()
    else:
        return {
            "error_type": type(exception).__name__,
            "error_code": "UNKNOWN_ERROR",
            "message": str(exception),
            "context": {},
            "cause": None,
        }
