"""
Factor System - 单因子多时间框架筛选系统

A comprehensive factor screening system for multi-timeframe analysis.
"""

__version__ = "0.1.0"
__author__ = "Quant Team"

try:
    from .data.data_loader import MultiTimeframeDataLoader  # type: ignore
except ImportError:  # pragma: no cover - 在最小依赖环境中容忍缺失
    MultiTimeframeDataLoader = None

__all__ = ["MultiTimeframeDataLoader"]
