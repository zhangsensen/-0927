"""
Top-200 组合筛选模块

从约 2000 个回测组合中，按「方案 C：强调风格与因子多样化的均衡型」筛选出 Top-200。
"""

from .core import select_top200, DEFAULT_CONFIG
from .analyzer import analyze_single_combo

__all__ = [
    'select_top200',
    'DEFAULT_CONFIG',
    'analyze_single_combo',
]

__version__ = '1.0.0'
