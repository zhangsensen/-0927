"""
Factor System - 单因子多时间框架筛选系统

A comprehensive factor screening system for multi-timeframe analysis.
"""

__version__ = "0.1.0"
__author__ = "Quant Team"

from .data.data_loader import MultiTimeframeDataLoader

__all__ = [
    "MultiTimeframeDataLoader"
]