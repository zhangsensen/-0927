"""
因子引擎模块 - 重构后的因子导入和定义
消除lint错误，提高代码质量和可维护性
"""

# 标准库导入
import logging

# ETF轮动专用长周期因子
from .etf_momentum import (
    DRAWDOWN_63D,
    MOM_ACCEL,
    VOLATILITY_120D,
    Momentum63,
    Momentum126,
    Momentum252,
)

# 导入分类的因子列表
from .factor_lists import (
    ALL_CORE_FACTORS,
    ALL_FACTORS,
    CORE_FACTORS,
    CORE_MONEY_FLOW_FACTORS,
    ENHANCED_MONEY_FLOW_FACTORS,
)

# 资金流因子 - 核心和增强
from .money_flow.core import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from .money_flow.enhanced import (
    Flow_Reversal_Ratio,
    Flow_Tier_Ratio_Delta,
    Institutional_Absorption,
    Northbound_NetInflow_Rate,
)

# 导入生成的因子（选择性导入，避免import *）
from .overlap_generated import *  # noqa: F403,F401
from .statistic_generated import *  # noqa: F403,F401

# 核心手动创建的因子
from .technical import (
    MACD,
    RSI,
    STOCH,
    MACDHistogram,
    MACDSignal,
)
from .technical_generated import *  # noqa: F403,F401

# 导入VectorBT指标（新增154个技术指标）
from .vbt_indicators import *  # noqa: F403,F401
from .volume_generated import *  # noqa: F403,F401

# 本地导入 - 按模块组织，避免import *


# 设置日志
logger = logging.getLogger(__name__)

# 核心因子类 - 手动创建的因子
CORE_FACTOR_CLASSES = [
    # 技术指标核心因子
    RSI,
    MACD,
    MACDSignal,
    MACDHistogram,
    STOCH,
    # 资金流核心因子
    MainNetInflow_Rate,
    LargeOrder_Ratio,
    SuperLargeOrder_Ratio,
    OrderConcentration,
    MoneyFlow_Hierarchy,
    MoneyFlow_Consensus,
    MainFlow_Momentum,
    Flow_Price_Divergence,
    # 资金流增强因子
    Institutional_Absorption,
    Flow_Tier_Ratio_Delta,
    Flow_Reversal_Ratio,
    Northbound_NetInflow_Rate,
    # ETF轮动长周期因子
    Momentum63,
    Momentum126,
    Momentum252,
    VOLATILITY_120D,
    MOM_ACCEL,
    DRAWDOWN_63D,
]


# 从因子列表模块获取完整的生成因子列表
# 这里我们动态获取所有已导入的生成因子，避免手动维护大型列表
def _get_generated_factors():
    """动态获取所有生成因子类"""
    factors = []

    # 从各个生成模块获取因子（已通过import *导入）
    import sys

    current_module = sys.modules[__name__]

    # 获取所有非私有属性
    for name in dir(current_module):
        obj = getattr(current_module, name)
        # 如果是类且不是核心因子，添加到生成因子列表
        if (
            isinstance(obj, type)
            and hasattr(obj, "calculate")
            and name not in [c.__name__ for c in CORE_FACTOR_CLASSES]
            and not name.startswith("_")
        ):
            factors.append(obj)

    return factors


# 动态生成因子列表
GENERATED_FACTORS = CORE_FACTOR_CLASSES + _get_generated_factors()


# 动态因子类映射 - 使用因子注册表
def _build_factor_class_map():
    """动态构建因子类映射"""
    factor_map = {}

    # 为所有核心因子创建映射
    for factor_class in CORE_FACTOR_CLASSES:
        factor_map[factor_class.__name__] = factor_class

    # 为生成因子创建映射
    for factor_class in _get_generated_factors():
        factor_map[factor_class.__name__] = factor_class

    return factor_map


# 动态因子映射
FACTOR_CLASS_MAP = _build_factor_class_map()


# 兼容性函数 - 保持向后兼容
def get_factor_class(factor_id: str):
    """获取因子类（兼容性函数）

    Args:
        factor_id: 因子ID

    Returns:
        因子类或None
    """
    return FACTOR_CLASS_MAP.get(factor_id)


def list_all_factors():
    """列出所有因子（兼容性函数）

    Returns:
        因子ID列表
    """
    return list(FACTOR_CLASS_MAP.keys())


# 导出的公共接口
__all__ = [
    # 核心因子类
    "CORE_FACTOR_CLASSES",
    "GENERATED_FACTORS",
    "FACTOR_CLASS_MAP",
    # 兼容性函数
    "get_factor_class",
    "list_all_factors",
    # 因子列表
    "CORE_FACTORS",
    "CORE_MONEY_FLOW_FACTORS",
    "ENHANCED_MONEY_FLOW_FACTORS",
    "ALL_CORE_FACTORS",
    "ALL_FACTORS",
]

# 日志记录
logger.info(
    f"因子模块加载完成: {len(GENERATED_FACTORS)} 个因子, {len(FACTOR_CLASS_MAP)} 个映射"
)
