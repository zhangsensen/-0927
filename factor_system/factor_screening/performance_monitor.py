#!/usr/bin/env python3
"""
性能监控模块 - 提供详细的性能分析和热点检测
作者：量化首席工程师
版本：1.0.0
日期：2025-10-07
"""

import time
import logging
import psutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation_name,
            'duration': self.duration,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_delta': self.memory_after - self.memory_before,
            'memory_peak': self.memory_peak,
            'cpu_percent': self.cpu_percent
        }


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, enable_logging: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_logging = enable_logging
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stack: List[str] = []
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.memory_monitoring = True
        self._monitoring_thread = None
        self._stop_monitoring = False
        self._peak_memory = 0.0

        # 启动内存监控线程
        if self.memory_monitoring:
            self._start_memory_monitoring()

    def _start_memory_monitoring(self):
        """启动内存监控线程"""
        def monitor_memory():
            process = psutil.Process()
            while not self._stop_monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self._peak_memory = max(self._peak_memory, memory_mb)
                    time.sleep(0.1)  # 每100ms检查一次
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break

        self._monitoring_thread = threading.Thread(target=monitor_memory, daemon=True)
        self._monitoring_thread.start()

    def _get_system_metrics(self) -> tuple:
        """获取当前系统指标"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            return memory_mb, cpu_percent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0

    @contextmanager
    def time_operation(self, operation_name: str):
        """性能监控上下文管理器"""
        # 记录开始状态
        start_time = time.perf_counter()
        memory_before, cpu_before = self._get_system_metrics()

        if self.enable_logging:
            self.logger.info(f"🔍 开始监控操作: {operation_name}")

        # 添加到操作栈
        self.operation_stack.append(operation_name)

        try:
            yield self
        finally:
            # 记录结束状态
            end_time = time.perf_counter()
            memory_after, cpu_after = self._get_system_metrics()
            duration = end_time - start_time

            # 创建性能指标
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=self._peak_memory,
                cpu_percent=cpu_after
            )

            self.metrics.append(metrics)
            self.operation_times[operation_name].append(duration)

            # 从操作栈中移除
            if operation_name in self.operation_stack:
                self.operation_stack.remove(operation_name)

            if self.enable_logging:
                self.logger.info(
                    f"✅ 操作完成: {operation_name} | "
                    f"耗时: {duration:.3f}s | "
                    f"内存: {memory_before:.1f}→{memory_after:.1f}MB | "
                    f"峰值: {self._peak_memory:.1f}MB"
                )

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """获取操作统计信息"""
        times = list(self.operation_times[operation_name])
        if not times:
            return {}

        return {
            'count': len(times),
            'total_time': sum(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'last_time': times[-1] if times else 0.0
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics:
            return {'total_operations': 0}

        # 按操作分组统计
        operation_stats = {}
        for metrics in self.metrics:
            op_name = metrics.operation_name
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'max_memory': 0.0
                }

            stats = operation_stats[op_name]
            stats['count'] += 1
            stats['total_time'] += metrics.duration
            stats['max_memory'] = max(stats['max_memory'], metrics.memory_peak)
            stats['avg_time'] = stats['total_time'] / stats['count']

        # 找出最耗时的操作
        slowest_operations = sorted(
            operation_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]

        return {
            'total_operations': len(self.metrics),
            'total_time': sum(m.duration for m in self.metrics),
            'peak_memory': max(m.memory_peak for m in self.metrics),
            'operation_stats': operation_stats,
            'slowest_operations': slowest_operations
        }

    def log_performance_bottlenecks(self):
        """记录性能瓶颈"""
        summary = self.get_performance_summary()

        if summary['total_operations'] == 0:
            return

        self.logger.info("🔍 === 性能瓶颈分析 ===")

        # 记录最耗时的操作
        for op_name, stats in summary['slowest_operations']:
            if stats['total_time'] > 0.1:  # 只显示耗时超过0.1秒的操作
                self.logger.warning(
                    f"⚠️ 性能瓶颈: {op_name} | "
                    f"总耗时: {stats['total_time']:.3f}s | "
                    f"调用次数: {stats['count']} | "
                    f"平均耗时: {stats['avg_time']:.3f}s | "
                    f"内存峰值: {stats['max_memory']:.1f}MB"
                )

        # 记录总体统计
        self.logger.info(
            f"📊 总体性能: {summary['total_operations']}个操作 | "
            f"总耗时: {summary['total_time']:.3f}s | "
            f"内存峰值: {summary['peak_memory']:.1f}MB"
        )

    def reset(self):
        """重置监控数据"""
        self.metrics.clear()
        self.operation_stack.clear()
        self.operation_times.clear()
        self._peak_memory = 0.0

        if self.enable_logging:
            self.logger.info("🔄 性能监控数据已重置")

    def __del__(self):
        """清理资源"""
        self._stop_monitoring = True
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1.0)


# 全局性能监控器实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(enable_logging: bool = True) -> PerformanceMonitor:
    """获取全局性能监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(enable_logging=enable_logging)
    return _global_monitor


def reset_global_monitor():
    """重置全局性能监控器"""
    global _global_monitor
    if _global_monitor is not None:
        _global_monitor.reset()


# 便捷装饰器
def monitor_performance(operation_name: Optional[str] = None):
    """性能监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()

            with monitor.time_operation(name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试代码
    monitor = get_performance_monitor()

    with monitor.time_operation("测试操作1"):
        time.sleep(0.1)

    with monitor.time_operation("测试操作2"):
        time.sleep(0.2)

    monitor.log_performance_bottlenecks()
    print(monitor.get_performance_summary())