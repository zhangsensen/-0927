"""
Core modules for ETF rotation system.

使用显式导入，不使用 from core import *

核心模块：
- combo_wfo_optimizer: 组合级 Walk-Forward 优化（真正的滚动 WFO）
- precise_factor_library_v2: 精选因子库 (18 因子)
- cross_section_processor: 横截面标准化
- data_loader: 数据加载与缓存
- ic_calculator_numba: IC 计算（Numba 加速）
- market_timing: 择时模块

⚠️ 注意：本项目只使用真正的滚动 WFO 实现，拒绝任何简化版本。
"""

from .combo_wfo_optimizer import ComboWFOOptimizer
from .ic_calculator_numba import ICCalculatorNumba

__all__ = ["ICCalculatorNumba", "ComboWFOOptimizer"]
