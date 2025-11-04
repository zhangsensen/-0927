"""
Core modules for ETF rotation system.

使用显式导入，不使用 from core import *
"""

from .ic_calculator_numba import ICCalculatorNumba

__all__ = ["ICCalculatorNumba"]

"""
核心模块（扁平化）：
- data_validator: 数据质量验证
- precise_factor_library: 26 个全量因子
- precise_factor_library_v2: 10 个精选因子
- cross_section_processor: 横截面标准化
- ic_calculator: IC 计算
- factor_selector: 因子选择器
- walk_forward_optimizer: WFO 框架
- constrained_walk_forward_optimizer: 约束 WFO
"""
