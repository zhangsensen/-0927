"""
VectorBT技术指标因子模块
将enhanced_factor_calculator中的154个指标注册为FactorEngine可用因子
"""

from .moving_averages import *
from .momentum import *
from .volatility import *
from .volume import *
from .overlap import *

# 自动导出所有已导入的因子类
__all__ = []