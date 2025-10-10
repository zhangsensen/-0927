"""
统一因子引擎 - 研究与回测共享的因子计算核心

提供:
- 统一的因子注册与管理
- 按需计算与缓存策略
- 标准化的数据接口
- 版本化的元数据追溯

推荐使用方式:
    from factor_system.factor_engine import api
    
    # 计算因子
    factors = api.calculate_factors(
        factor_ids=["RSI", "STOCH"],
        symbols=["0700.HK"],
        timeframe="15min",
        start_date=datetime(2025, 9, 1),
        end_date=datetime(2025, 9, 30),
    )
"""

__version__ = "0.2.0"

# 核心类（供高级用户直接使用）
from factor_system.factor_engine.core.engine import FactorEngine
from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.providers.base import DataProvider

# 统一API入口（推荐使用）
from factor_system.factor_engine import api

__all__ = [
    # 核心类
    "FactorEngine",
    "FactorRegistry",
    "DataProvider",
    # 统一API模块
    "api",
]

