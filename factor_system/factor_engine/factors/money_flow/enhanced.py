"""
资金流增强因子 - 4个择时/风控因子

向量化实现，用于择时和风控
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_system.factor_engine.core.base_factor import BaseFactor


class Institutional_Absorption(BaseFactor):
    """
    机构吸筹信号 = main_net>0 且 realized_vol(3)<realized_vol(10)

    【时序安全】
    1. 输入数据已由MoneyFlowProvider执行T+1滞后处理
    2. 收益率使用T-1日收盘价计算
    返回0/1信号
    """

    factor_id = "Institutional_Absorption"
    version = "v2.0"
    category = "money_flow_enhanced"
    description = "机构吸筹信号（择时用，T+1时序安全）"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 需要价格数据计算波动率
        if "close" not in data.columns or "main_net" not in data.columns:
            return pd.Series(0, index=data.index, name=self.factor_id)

        # 【时序安全】使用T-1日收益率，避免未来信息泄露
        # 在14:30冻结时刻，只能使用历史收盘价计算波动率
        ret = data["close"].pct_change().shift(1)
        vol_short = ret.rolling(3).std()
        vol_long = ret.rolling(10).std()

        # 向量化：条件判断
        signal = ((data["main_net"] > 0) & (vol_short < vol_long)).astype(int)
        return signal.rename(self.factor_id)


class Flow_Tier_Ratio_Delta(BaseFactor):
    """
    资金层级变化率 = (大单+超大单)/(中单+小单) 的变化率

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    """

    factor_id = "Flow_Tier_Ratio_Delta"
    version = "v2.0"
    category = "money_flow_enhanced"
    description = "资金层级变化率（T+1时序安全）"

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        required_cols = [
            "buy_large_amount",
            "buy_super_large_amount",
            "buy_small_amount",
            "buy_medium_amount",
        ]
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.nan, index=data.index, name=self.factor_id)

        # 向量化
        institutional = data["buy_large_amount"] + data["buy_super_large_amount"]
        retail_medium = data["buy_small_amount"] + data["buy_medium_amount"]

        # 分层比率（加入阈值，避免过小分母）
        eps = 1e-6
        tier_ratio = institutional / np.maximum(retail_medium, eps)

        # 安全的百分比变化：当历史值过小或为0时视作NaN，避免inf
        prev = tier_ratio.shift(self.window)
        denom = prev.where(np.abs(prev) >= eps)
        safe_pct = (tier_ratio - prev) / denom

        # 清理inf/-inf
        safe_pct = safe_pct.replace([np.inf, -np.inf], np.nan)
        return safe_pct.rename(self.factor_id)


class Flow_Reversal_Ratio(BaseFactor):
    """
    资金反转信号 = sign(today_main_net) != sign(mean(last_3))

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    返回0/1信号
    """

    factor_id = "Flow_Reversal_Ratio"
    version = "v2.0"
    category = "money_flow_enhanced"
    description = "资金反转信号（风控用，T+1时序安全）"

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        if "main_net" not in data.columns:
            return pd.Series(0, index=data.index, name=self.factor_id)

        # 向量化
        today_sign = np.sign(data["main_net"])
        last3_mean_sign = np.sign(data["main_net"].shift(1).rolling(3).mean())

        signal = (today_sign != last3_mean_sign).astype(int)
        return signal.rename(self.factor_id)


class Northbound_NetInflow_Rate(BaseFactor):
    """
    北向资金净流入率（T+1时序安全版本）

    基于沪港通/深港通资金的净流入情况计算
    由于北向资金数据也是T+1发布，这里使用滞后处理

    【时序安全】输入数据已由MoneyFlowProvider执行T+1滞后处理
    计算北向资金净流入占成交额的比例
    """

    factor_id = "Northbound_NetInflow_Rate"
    version = "v2.0"  # T+1时序安全版本
    category = "money_flow_enhanced"
    description = "北向资金净流入率（T+1时序安全）"

    def __init__(self, window: int = 5):
        self.window = window
        super().__init__(window=window)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        # 检查是否有北向资金相关字段
        northbound_cols = [
            "northbound_net_inflow",  # 北向净流入额
            "northbound_buy_amount",  # 北向买入额
            "northbound_sell_amount",  # 北向卖出额
        ]

        # 如果没有北向资金数据，尝试用主力资金作为代理指标
        if not any(col in data.columns for col in northbound_cols):
            # 代理方法：使用超大单净流入作为北向资金的代理
            # 这是一个合理的假设，因为北向资金通常以大额交易为主
            if (
                "buy_super_large_amount" in data.columns
                and "sell_super_large_amount" in data.columns
            ):
                northbound_net = (
                    data["buy_super_large_amount"] - data["sell_super_large_amount"]
                )
                turnover_amount = data["turnover_amount"]

                # 计算北向资金净流入率（T+1安全）
                northflow_rate_ma = northbound_net.rolling(
                    self.window, min_periods=1
                ).mean()
                turnover_ma = turnover_amount.rolling(self.window, min_periods=1).mean()

                result = northflow_rate_ma / np.maximum(turnover_ma, 1e-6)
                return result.rename(self.factor_id)
            else:
                # 如果没有相关数据，返回0序列
                return pd.Series(0, index=data.index, name=self.factor_id)

        # 如果有真实的北向资金数据
        if "northbound_net_inflow" in data.columns:
            northflow_ma = (
                data["northbound_net_inflow"].rolling(self.window, min_periods=1).mean()
            )
            turnover_ma = (
                data["turnover_amount"].rolling(self.window, min_periods=1).mean()
            )

            result = northflow_ma / np.maximum(turnover_ma, 1e-6)
            return result.rename(self.factor_id)

        # 如果有买卖额数据，计算净流入
        elif (
            "northbound_buy_amount" in data.columns
            and "northbound_sell_amount" in data.columns
        ):
            northflow_net = (
                data["northbound_buy_amount"] - data["northbound_sell_amount"]
            )
            northflow_ma = northflow_net.rolling(self.window, min_periods=1).mean()
            turnover_ma = (
                data["turnover_amount"].rolling(self.window, min_periods=1).mean()
            )

            result = northflow_ma / np.maximum(turnover_ma, 1e-6)
            return result.rename(self.factor_id)

        # 默认返回0序列
        return pd.Series(0, index=data.index, name=self.factor_id)
