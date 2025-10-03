#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化日志工具 - P3-2监控日志完善
提供结构化日志、性能指标监控、异常告警
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO
    ):
        """
        初始化结构化日志记录器
        
        Args:
            name: 日志器名称
            log_file: 日志文件路径（可选）
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除现有handlers
        self.logger.handlers.clear()
        
        # 添加控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件handler（如果指定）
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        self.metrics: Dict[str, Any] = {}
        self.process = psutil.Process()
    
    def log_structured(
        self,
        level: str,
        message: str,
        **kwargs: Any
    ) -> None:
        """
        记录结构化日志
        
        Args:
            level: 日志级别（info/warning/error/critical）
            message: 日志消息
            **kwargs: 额外的结构化字段
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            **kwargs
        }
        
        # 添加系统指标
        log_entry['system'] = {
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'memory_percent': self.process.memory_percent(),
            'cpu_percent': self.process.cpu_percent(interval=0.1)
        }
        
        # 格式化为JSON字符串
        json_log = json.dumps(log_entry, ensure_ascii=False, default=str)
        
        # 记录到标准日志
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(json_log)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """记录INFO级别结构化日志"""
        self.log_structured('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """记录WARNING级别结构化日志"""
        self.log_structured('warning', message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """记录ERROR级别结构化日志"""
        self.log_structured('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """记录CRITICAL级别结构化日志"""
        self.log_structured('critical', message, **kwargs)
    
    @contextmanager
    def log_performance(
        self,
        operation: str,
        **kwargs: Any
    ):
        """
        性能监控上下文管理器
        
        Args:
            operation: 操作名称
            **kwargs: 额外的上下文信息
            
        Example:
            with logger.log_performance("IC计算", factors_count=217):
                result = calculate_ic(factors, returns)
        """
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        self.info(
            f"开始操作: {operation}",
            operation=operation,
            phase='start',
            **kwargs
        )
        
        exception_occurred = None
        
        try:
            yield
        except Exception as e:
            exception_occurred = e
            raise
        finally:
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            elapsed_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            performance_data = {
                'operation': operation,
                'phase': 'complete' if exception_occurred is None else 'failed',
                'elapsed_seconds': elapsed_time,
                'memory_delta_mb': memory_delta,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                **kwargs
            }
            
            if exception_occurred:
                performance_data['exception'] = str(exception_occurred)
                performance_data['exception_type'] = type(exception_occurred).__name__
                self.error(
                    f"操作失败: {operation}",
                    **performance_data
                )
            else:
                self.info(
                    f"操作完成: {operation}",
                    **performance_data
                )
    
    def log_metric(
        self,
        metric_name: str,
        value: Any,
        unit: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        记录指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            unit: 单位（可选）
            **kwargs: 额外的标签
        """
        metric_data = {
            'metric': metric_name,
            'value': value,
            'unit': unit,
            **kwargs
        }
        
        # 存储到内部指标字典
        self.metrics[metric_name] = {
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        self.info(
            f"指标记录: {metric_name} = {value} {unit or ''}",
            **metric_data
        )
    
    def log_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        **kwargs: Any
    ) -> None:
        """
        记录告警
        
        Args:
            alert_type: 告警类型（memory/performance/data_quality等）
            message: 告警消息
            severity: 严重程度（info/warning/error/critical）
            **kwargs: 额外的告警上下文
        """
        alert_data = {
            'alert_type': alert_type,
            'severity': severity,
            **kwargs
        }
        
        log_method = getattr(self, severity.lower(), self.warning)
        log_method(
            f"[ALERT] {message}",
            **alert_data
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要
        
        Returns:
            指标摘要字典
        """
        return {
            'total_metrics': len(self.metrics),
            'metrics': self.metrics,
            'system_snapshot': {
                'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'cpu_percent': self.process.cpu_percent(interval=0.1)
            }
        }
    
    def export_metrics(self, output_file: Path) -> None:
        """
        导出指标到JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        summary = self.get_metrics_summary()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.info(f"指标已导出到: {output_file}")


# 全局日志器实例
_global_loggers: Dict[str, StructuredLogger] = {}


def get_structured_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> StructuredLogger:
    """
    获取结构化日志器实例（单例模式）
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        
    Returns:
        StructuredLogger实例
    """
    if name not in _global_loggers:
        _global_loggers[name] = StructuredLogger(name, log_file, level)
    return _global_loggers[name]


# 性能监控装饰器

def monitor_performance(operation_name: Optional[str] = None):
    """
    性能监控装饰器
    
    Args:
        operation_name: 操作名称（默认使用函数名）
        
    Example:
        @monitor_performance("因子筛选")
        def screen_factors(symbol, timeframe):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = get_structured_logger(func.__module__)
            
            with logger.log_performance(op_name):
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

