"""
A股ETF交易成本模型（简化）

注意：
- ETF 通常免征印花税（与股票不同），因此默认设置为 0。
- 佣金与滑点按经验值建模，可按券商/实际流动性调整。

计算公式（简化）：
总成本 = [印花税(仅卖出时) + 佣金 + 滑点] * 交易金额
"""


class AShareETFTradingCost:
    """A股ETF交易成本模型（ETF 默认无印花税）"""

    def __init__(
        self,
        stamp_tax: float = 0.0,  # ETF默认 0
        commission: float = 0.0003,  # 佣金 0.03%
        slippage: float = 0.0005,
    ):  # 滑点 0.05%
        self.stamp_tax = float(stamp_tax)
        self.commission = float(commission)
        self.slippage = float(slippage)

    def calculate_cost(self, trade_value: float, is_sell: bool = False) -> float:
        """
        计算单笔交易成本

        Args:
            trade_value: 交易金额
            is_sell: 是否卖出（影响印花税征收）

        Returns:
            总交易成本
        """
        cost = 0.0

        # 印花税（ETF通常免征；若设置>0，仅在卖出时计提）
        if is_sell and self.stamp_tax > 0:
            cost += self.stamp_tax * trade_value

        # 佣金（双向）
        cost += self.commission * trade_value

        # 滑点（双向）
        cost += self.slippage * trade_value

        return cost
