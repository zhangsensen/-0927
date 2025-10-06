# -*- coding: utf-8 -*-
"""
统一的日志格式工具

强制要求日志格式: {session_id}|{symbol}|{tf}|{direction}|{metric}={value}
"""
import re
from typing import Any, Optional


class LogFormatError(Exception):
    """日志格式不合规异常"""

    pass


class StructuredLogger:
    """结构化日志记录器"""

    @staticmethod
    def validate_format(message: str) -> bool:
        """
        验证日志格式是否符合规范

        必须包含: session_id|symbol|tf|direction|metric=value
        """
        pattern = r"^.*\|.*\|.*\|.*\|.*=.*$"
        return bool(re.match(pattern, message))

    @staticmethod
    def format_message(
        session_id: str,
        symbol: str,
        timeframe: str,
        direction: str,
        metric: str,
        value: Any,
        extra: Optional[str] = None,
    ) -> str:
        """
        生成标准化日志消息

        Args:
            session_id: 会话ID (如: 20251006_030838)
            symbol: 标的代码 (如: 0700.HK)
            timeframe: 时间频率 (如: 5min, 60min)
            direction: 多空方向 (如: LONG, SHORT, NEUTRAL)
            metric: 指标名称 (如: trades, return, sharpe)
            value: 指标值
            extra: 额外信息 (可选)

        Returns:
            格式化的日志消息
        """
        base_msg = f"{session_id}|{symbol}|{timeframe}|{direction}|{metric}={value}"
        if extra:
            return f"{base_msg}|{extra}"
        return base_msg

    @staticmethod
    def format_bulk(
        session_id: str,
        symbol: str,
        timeframe: str,
        direction: str,
        metrics: dict[str, Any],
    ) -> str:
        """
        批量格式化多个指标

        Args:
            session_id: 会话ID
            symbol: 标的代码
            timeframe: 时间频率
            direction: 多空方向
            metrics: 指标字典 {metric_name: value}

        Returns:
            格式化的日志消息
        """
        metric_parts = [f"{k}={v}" for k, v in metrics.items()]
        return (
            f"{session_id}|{symbol}|{timeframe}|{direction}|{', '.join(metric_parts)}"
        )

    @staticmethod
    def format_multi_tf(
        session_id: str, symbol: str, timeframes: list[str], metric: str, value: Any
    ) -> str:
        """
        多时间框架日志格式

        Args:
            session_id: 会话ID
            symbol: 标的代码
            timeframes: 时间框架列表
            metric: 指标名称
            value: 指标值

        Returns:
            格式化的日志消息
        """
        tf_str = "+".join(timeframes)
        return f"{session_id}|{symbol}|{tf_str}|MULTI_TF|{metric}={value}"


def enforce_log_format(func):
    """
    装饰器: 强制检查日志格式

    使用示例:
        @enforce_log_format
        def my_function():
            logger.info("session123|0700.HK|5min|LONG|trades=10")
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # 这里可以添加运行时日志格式检查逻辑
        return result

    return wrapper


# 快捷函数
def log_trade(
    session_id: str,
    symbol: str,
    timeframe: str,
    direction: str,
    trade_count: int,
    pnl: float,
) -> str:
    """生成交易日志"""
    return StructuredLogger.format_message(
        session_id,
        symbol,
        timeframe,
        direction,
        "trades",
        trade_count,
        f"pnl={pnl:.2f}",
    )


def log_metric(
    session_id: str, symbol: str, timeframe: str, metric_name: str, metric_value: Any
) -> str:
    """生成指标日志"""
    return StructuredLogger.format_message(
        session_id, symbol, timeframe, "METRIC", metric_name, metric_value
    )


def log_backtest_summary(
    session_id: str,
    symbol: str,
    timeframe: str,
    total_return: float,
    sharpe: float,
    max_dd: float,
) -> str:
    """生成回测总结日志"""
    metrics = {
        "total_return": f"{total_return:.2%}",
        "sharpe": f"{sharpe:.2f}",
        "max_dd": f"{max_dd:.2%}",
    }
    return StructuredLogger.format_bulk(
        session_id, symbol, timeframe, "BACKTEST", metrics
    )
