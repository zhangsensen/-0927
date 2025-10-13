#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Linus式异常处理工具 - 防御性编程

核心原则：
1. 报错要快，输出要准
2. 异常信息必须包含上下文
3. 关键路径必须有防御性包装
"""

import functools
import logging
import traceback
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_operation(
    func: Callable[..., T],
    error_msg: Optional[str] = None,
    default_return: Optional[T] = None,
    raise_on_error: bool = True,
) -> Callable[..., Optional[T]]:
    """
    防御性操作包装器

    Args:
        func: 要包装的函数
        error_msg: 自定义错误消息
        default_return: 发生错误时的默认返回值
        raise_on_error: 是否在错误时抛出异常

    Returns:
        包装后的函数

    Examples:
        >>> @safe_operation
        ... def risky_calculation(x, y):
        ...     return x / y

        >>> result = risky_calculation(10, 0)  # 返回None并记录错误
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 构建详细错误消息
            msg = error_msg or f"操作失败: {func.__name__}"
            logger.error(f"{msg}: {e}")
            logger.debug(f"堆栈跟踪:\n{traceback.format_exc()}")

            if raise_on_error:
                raise
            return default_return

    return wrapper


def safe_io_operation(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """
    I/O操作专用包装器（文件读写、网络请求等）

    Args:
        func: I/O操作函数

    Returns:
        包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"文件不存在: {e}")
            raise
        except PermissionError as e:
            logger.error(f"权限不足: {e}")
            raise
        except IOError as e:
            logger.error(f"I/O错误: {e}")
            raise
        except Exception as e:
            logger.error(f"未知I/O错误: {e}")
            logger.debug(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise

    return wrapper


def safe_compute_operation(func: Callable[..., T]) -> Callable[..., Optional[T]]:
    """
    计算操作专用包装器（因子计算、统计分析等）

    Args:
        func: 计算操作函数

    Returns:
        包装后的函数
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError as e:
            logger.error(f"除零错误: {e}")
            raise ValueError("计算错误: 除数为零") from e
        except ValueError as e:
            logger.error(f"数值错误: {e}")
            raise
        except OverflowError as e:
            logger.error(f"数值溢出: {e}")
            raise
        except Exception as e:
            logger.error(f"计算失败: {func.__name__}: {e}")
            logger.debug(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise

    return wrapper


class FactorSystemError(Exception):
    """因子系统基础异常"""

    pass


class ConfigurationError(FactorSystemError):
    """配置错误"""

    pass


class DataValidationError(FactorSystemError):
    """数据验证错误"""

    pass


class CalculationError(FactorSystemError):
    """计算错误"""

    pass


class PathError(FactorSystemError):
    """路径错误"""

    pass


def validate_not_none(value: Optional[T], name: str) -> T:
    """
    验证值不为None

    Args:
        value: 要验证的值
        name: 参数名称

    Returns:
        验证后的值

    Raises:
        ValueError: 如果值为None
    """
    if value is None:
        raise ValueError(f"{name} 不能为None")
    return value


def validate_positive(value: float, name: str) -> float:
    """
    验证值为正数

    Args:
        value: 要验证的值
        name: 参数名称

    Returns:
        验证后的值

    Raises:
        ValueError: 如果值不是正数
    """
    if value <= 0:
        raise ValueError(f"{name} 必须为正数，当前值: {value}")
    return value


def validate_in_range(value: float, min_val: float, max_val: float, name: str) -> float:
    """
    验证值在指定范围内

    Args:
        value: 要验证的值
        min_val: 最小值
        max_val: 最大值
        name: 参数名称

    Returns:
        验证后的值

    Raises:
        ValueError: 如果值不在范围内
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{name} 必须在 [{min_val}, {max_val}] 范围内，当前值: {value}"
        )
    return value


if __name__ == "__main__":
    # 测试异常处理工具
    print("🔧 异常处理工具测试")

    @safe_operation
    def test_division(x: float, y: float) -> float:
        return x / y

    # 正常情况
    result = test_division(10, 2)
    print(f"正常计算: 10 / 2 = {result}")

    # 异常情况
    try:
        result = test_division(10, 0)
    except Exception as e:
        print(f"捕获异常: {e}")

    # 验证函数测试
    try:
        validate_positive(-1, "test_value")
    except ValueError as e:
        print(f"验证失败: {e}")

    print("✅ 异常处理工具测试完成")
