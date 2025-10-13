# -*- coding: utf-8 -*-
"""
Factor System - 专业级因子计算与筛选系统

🔧 Linus式重构：消灭非法导入，建立清晰模块边界

核心模块：
- factor_engine: 统一因子计算引擎
- factor_generation: 批量因子生成
- factor_screening: 专业因子筛选
- utils: 工具函数（路径管理、异常处理等）
"""

__version__ = "0.2.0"
__author__ = "Quant Chief Engineer"

# 🔧 清理非法导入 - 移除不存在的 data.data_loader
# 保持 __init__.py 简洁，避免循环依赖

__all__ = []
