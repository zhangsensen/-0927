"""
核心因子定义 - 手动创建的高质量因子
"""

# 手动创建的核心技术指标因子
CORE_FACTORS = [
    "RSI",
    "MACD",
    "MACDSignal",
    "MACDHistogram",
    "STOCH",
]

# 核心资金流因子
CORE_MONEY_FLOW_FACTORS = [
    "MainNetInflow_Rate",
    "LargeOrder_Ratio",
    "SuperLargeOrder_Ratio",
    "OrderConcentration",
    "MoneyFlow_Hierarchy",
    "MoneyFlow_Consensus",
    "MainFlow_Momentum",
    "Flow_Price_Divergence",
]

# 增强资金流因子
ENHANCED_MONEY_FLOW_FACTORS = [
    "Institutional_Absorption",
    "Flow_Tier_Ratio_Delta",
    "Flow_Reversal_Ratio",
    "Northbound_NetInflow_Rate",
]

# 所有核心因子
ALL_CORE_FACTORS = (
    CORE_FACTORS +
    CORE_MONEY_FLOW_FACTORS +
    ENHANCED_MONEY_FLOW_FACTORS
)