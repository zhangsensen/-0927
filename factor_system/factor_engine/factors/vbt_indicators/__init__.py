"""
VectorBT技术指标因子模块
将enhanced_factor_calculator中的154个指标注册为FactorEngine可用因子
"""

from .momentum import *
from .moving_averages import *
from .overlap import *
from .volatility import *
from .volume import *

# 自动导出所有已导入的因子类
__all__ = []
