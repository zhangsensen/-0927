"""
Core模块
包含基础设施：数据加载、因子计算、横截面处理、IC计算
"""

from .data_loader import DataLoader
from .precise_factor_library_v2 import PreciseFactorLibrary
from .cross_section_processor import CrossSectionProcessor
from .ic_calculator_numba import ICCalculatorNumba

__all__ = ["ICCalculatorNumba", "DataLoader", "PreciseFactorLibrary", "CrossSectionProcessor"]

from .ic_calculator_numba import ICCalculatorNumba

__all__ = ["ICCalculatorNumba", "DirectFactorWFOOptimizer"]

"""当前保留模块：
- precise_factor_library_v2: 精选因子库
- cross_section_processor: 横截面标准化
- data_loader: 数据加载与缓存
- direct_factor_wfo_optimizer: 直接因子级WFO
"""
