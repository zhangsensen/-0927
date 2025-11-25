"""
Core modules for ETF rotation system.

使用显式导入，不使用 from core import *
"""

from .direct_factor_wfo_optimizer import DirectFactorWFOOptimizer
from .ic_calculator_numba import ICCalculatorNumba

__all__ = ["ICCalculatorNumba", "DirectFactorWFOOptimizer"]

"""当前保留模块：
- precise_factor_library_v2: 精选因子库
- cross_section_processor: 横截面标准化
- data_loader: 数据加载与缓存
- direct_factor_wfo_optimizer: 直接因子级WFO
"""
