"""
资金流因子模块

包含：
- core: 8个核心横截面因子
- enhanced: 4个择时/风控因子
- intraday: 3个日内因子
- constraints: 跳空/午休/尾盘约束
"""

from .core import (
    Flow_Price_Divergence,
    LargeOrder_Ratio,
    MainFlow_Momentum,
    MainNetInflow_Rate,
    MoneyFlow_Consensus,
    MoneyFlow_Hierarchy,
    OrderConcentration,
    SuperLargeOrder_Ratio,
)
from .enhanced import (
    Flow_Reversal_Ratio,
    Flow_Tier_Ratio_Delta,
    Institutional_Absorption,
    Northbound_NetInflow_Rate,
)
from .intraday import (
    IntradayMomentum,
    IntradayPriceBreakout,
    IntradayVolumeSurge,
)

__all__ = [
    # Core factors
    "MainNetInflow_Rate",
    "LargeOrder_Ratio",
    "SuperLargeOrder_Ratio",
    "OrderConcentration",
    "MoneyFlow_Hierarchy",
    "MoneyFlow_Consensus",
    "MainFlow_Momentum",
    "Flow_Price_Divergence",
    # Enhanced factors
    "Institutional_Absorption",
    "Flow_Tier_Ratio_Delta",
    "Flow_Reversal_Ratio",
    "Northbound_NetInflow_Rate",
    # Intraday factors
    "IntradayVolumeSurge",
    "IntradayPriceBreakout",
    "IntradayMomentum",
]
