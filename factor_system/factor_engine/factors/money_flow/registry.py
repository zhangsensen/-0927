"""
资金流因子注册表

用于将资金流因子集成到FactorEngine中
"""

import logging
from pathlib import Path

from factor_system.factor_engine.core.registry import FactorRegistry
from factor_system.factor_engine.factors.money_flow.core import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from factor_system.factor_engine.factors.money_flow.enhanced import (
    Flow_Reversal_Ratio,
    Flow_Tier_Ratio_Delta,
    Institutional_Absorption,
)

logger = logging.getLogger(__name__)


def register_money_flow_factors(registry: FactorRegistry):
    """
    注册资金流因子到因子引擎

    Args:
        registry: 因子注册表实例
    """
    money_flow_factors = [
        # 核心因子
        MainNetInflow_Rate,
        LargeOrder_Ratio,
        SuperLargeOrder_Ratio,
        OrderConcentration,
        MoneyFlow_Hierarchy,
        MoneyFlow_Consensus,
        MainFlow_Momentum,
        Flow_Price_Divergence,
        # 增强因子
        Institutional_Absorption,
        Flow_Tier_Ratio_Delta,
        Flow_Reversal_Ratio,
    ]

    registered_count = 0

    for factor_class in money_flow_factors:
        try:
            # 注册因子类
            registry.register(factor_class)

            # 添加数据源标记到元数据
            factor_id = factor_class.factor_id
            metadata = registry.get_metadata(factor_id)
            if metadata:
                metadata["data_source"] = "money_flow"
                metadata["requires_price"] = factor_id in ["Flow_Price_Divergence"]

            registered_count += 1
            logger.info(f"✅ 注册资金流因子: {factor_id} v{factor_class.version}")

        except Exception as e:
            logger.error(f"❌ 注册资金流因子失败 {factor_class.factor_id}: {e}")

    logger.info(f"资金流因子注册完成: {registered_count}/{len(money_flow_factors)} 个")


def get_money_flow_factor_sets() -> dict:
    """
    获取资金流因子集定义

    Returns:
        因子集字典
    """
    return {
        "money_flow_core": {
            "name": "资金流核心因子集",
            "description": "基于主力资金流动的核心技术因子",
            "factors": [
                "MainNetInflow_Rate",
                "LargeOrder_Ratio",
                "SuperLargeOrder_Ratio",
                "OrderConcentration",
                "MoneyFlow_Hierarchy",
                "MoneyFlow_Consensus",
                "MainFlow_Momentum",
                "Flow_Price_Divergence",
            ],
        },
        "money_flow_enhanced": {
            "name": "资金流增强因子集",
            "description": "结合资金流和价格行为的增强信号",
            "factors": [
                "Institutional_Absorption",
                "Flow_Tier_Ratio_Delta",
                "Flow_Reversal_Ratio",
            ],
        },
        "money_flow_all": {
            "name": "资金流完整因子集",
            "description": "全部资金流因子集合",
            "factors": [
                "MainNetInflow_Rate",
                "LargeOrder_Ratio",
                "SuperLargeOrder_Ratio",
                "OrderConcentration",
                "MoneyFlow_Hierarchy",
                "MoneyFlow_Consensus",
                "MainFlow_Momentum",
                "Flow_Price_Divergence",
                "Institutional_Absorption",
                "Flow_Tier_Ratio_Delta",
                "Flow_Reversal_Ratio",
            ],
        },
    }
