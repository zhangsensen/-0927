"""
资金流核心因子 - 8个主干因子

向量化实现，禁止循环
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor


class MainNetInflow_Rate(BaseFactor):
    """
    主力净流入率 = 主力净流入 / 成交额（5日均）

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    即：T日使用的main_net和turnover_amount实际是T-1日的数据

    向量化实现，禁止循环
    """

    factor_id = "MainNetInflow_Rate"
    version = "v2.0"  # T+1时序安全版本
    category = "money_flow"
    description = "主力净流入率（5日均，T+1时序安全）"

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        向量化计算

        Args:
            data: 包含 main_net, turnover_amount 的DataFrame
        """
        if "main_net" not in data.columns or "turnover_amount" not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化：rolling mean
        main_net_ma = data["main_net"].rolling(window=self.window, min_periods=1).mean()
        turnover_ma = (
            data["turnover_amount"].rolling(window=self.window, min_periods=1).mean()
        )

        # 避免除零
        result = main_net_ma / np.maximum(turnover_ma, 1e-6)
        return result.rename(self.factor_id)


class LargeOrder_Ratio(BaseFactor):
    """
    大单占比 = (大单买入+大单卖出) / 成交额（10日均）

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "LargeOrder_Ratio"
    version = "v2.0"
    category = "money_flow"
    description = "大单占比（10日均，T+1时序安全）"

    def __init__(self, window: int = 10):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ["buy_large_amount", "sell_large_amount", "turnover_amount"]
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化
        large_total = (
            (data["buy_large_amount"] + data["sell_large_amount"])
            .rolling(self.window)
            .mean()
        )
        turnover_ma = data["turnover_amount"].rolling(self.window).mean()

        result = large_total / np.maximum(turnover_ma, 1e-6)
        return result.rename(self.factor_id)


class SuperLargeOrder_Ratio(BaseFactor):
    """
    超大单占比 = (超大单买入+超大单卖出) / 成交额（20日均）

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "SuperLargeOrder_Ratio"
    version = "v2.0"
    category = "money_flow"
    description = "超大单占比（20日均，T+1时序安全）"

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_cols = [
            "buy_super_large_amount",
            "sell_super_large_amount",
            "turnover_amount",
        ]
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        super_large_total = (
            (data["buy_super_large_amount"] + data["sell_super_large_amount"])
            .rolling(self.window)
            .mean()
        )
        turnover_ma = data["turnover_amount"].rolling(self.window).mean()

        result = super_large_total / np.maximum(turnover_ma, 1e-6)
        return result.rename(self.factor_id)


class OrderConcentration(BaseFactor):
    """
    资金集中度 = ((大单+超大单)净额) / 总净额

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "OrderConcentration"
    version = "v2.0"
    category = "money_flow"
    description = "资金集中度（T+1时序安全）"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_cols = [
            "buy_large_amount",
            "buy_super_large_amount",
            "sell_large_amount",
            "sell_super_large_amount",
        ]
        # 尝试获取 total_net，若缺失则使用 net_mf_amount 作为代理
        has_all_basic = all(col in data.columns for col in required_cols)
        has_total = ("total_net" in data.columns) or ("net_mf_amount" in data.columns)
        if not (has_all_basic and has_total):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化
        large_super_net = (
            data["buy_large_amount"]
            + data["buy_super_large_amount"]
            - data["sell_large_amount"]
            - data["sell_super_large_amount"]
        )
        total_net = (
            data["total_net"] if "total_net" in data.columns else data["net_mf_amount"]
        )

        result = large_super_net / np.maximum(np.abs(total_net), 1e-6)
        return result.rename(self.factor_id)


class MoneyFlow_Hierarchy(BaseFactor):
    """
    资金层级指数 = ((大单+超大单)净额 - 小单净额) / 总净额

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "MoneyFlow_Hierarchy"
    version = "v2.0"
    category = "money_flow"
    description = "资金层级指数（T+1时序安全）"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 允许 total_net 缺失时使用 net_mf_amount 作为代理
        required_cols = ["main_net", "retail_net"]
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)
        if ("total_net" not in data.columns) and ("net_mf_amount" not in data.columns):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        institutional_net = data["main_net"]  # 已计算
        retail_net = data["retail_net"]
        total_net = (
            data["total_net"] if "total_net" in data.columns else data["net_mf_amount"]
        )

        result = (institutional_net - retail_net) / np.maximum(np.abs(total_net), 1e-6)
        return result.rename(self.factor_id)


class MoneyFlow_Consensus(BaseFactor):
    """
    资金共识度 = sign(main_net) == sign(total_net) 的5日均

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "MoneyFlow_Consensus"
    version = "v2.0"
    category = "money_flow"
    description = "资金共识度（5日均，T+1时序安全）"

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # total_net 缺失时允许用 net_mf_amount 代理，保持鲁棒
        if "main_net" not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.factor_id)
        if ("total_net" not in data.columns) and ("net_mf_amount" not in data.columns):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化：sign比较
        main_sign = np.sign(data["main_net"])
        total_base = (
            data["total_net"] if "total_net" in data.columns else data["net_mf_amount"]
        )
        total_sign = np.sign(total_base)
        consensus = (main_sign == total_sign).astype(float)

        # 增加min_periods，减少早期全NA；对齐窗口
        result = consensus.rolling(
            self.window, min_periods=max(1, self.window // 2)
        ).mean()
        return result.rename(self.factor_id)


class MainFlow_Momentum(BaseFactor):
    """
    主力资金动量 = main_net的5-10日变化率

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "MainFlow_Momentum"
    version = "v2.0"
    category = "money_flow"
    description = "主力资金动量（T+1时序安全）"

    def __init__(self, short_window: int = 5, long_window: int = 10):
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(short_window=short_window, long_window=long_window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "main_net" not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化：pct_change
        main_net_short = data["main_net"].rolling(self.short_window).mean()
        main_net_long = data["main_net"].rolling(self.long_window).mean()

        result = (main_net_short - main_net_long) / np.maximum(
            np.abs(main_net_long), 1e-6
        )
        return result.rename(self.factor_id)


class Flow_Price_Divergence(BaseFactor):
    """
    资金价格背离 = -corr(main_net, ret, win=5)

    【时序安全】
    1. 输入数据已由MoneyFlowProvider执行T+1滞后处理
    2. 收益率使用T-1日收盘价计算
    背离时取负号，背离越大因子值越高
    """

    factor_id = "Flow_Price_Divergence"
    version = "v2.0"
    category = "money_flow"
    description = "资金价格背离度（T+1时序安全）"

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 需要价格数据计算收益率
        if "close" not in data.columns or "main_net" not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 【时序安全】使用T-1日收益率，避免未来信息泄露
        ret = data["close"].pct_change().shift(1)

        # 【修复】设置min_periods确保有足够数据才计算相关性
        result = (
            -data["main_net"].rolling(self.window, min_periods=self.window).corr(ret)
        )
        return result.rename(self.factor_id)
