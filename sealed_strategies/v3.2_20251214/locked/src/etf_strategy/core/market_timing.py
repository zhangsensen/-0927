import pandas as pd
import numpy as np


class LightTimingModule:
    """
    Light Timing Module as described in 1126.md

    Logic:
    - MA Signal (HS300): Price > MA200 ? 1 : -1
    - Mom Signal (HS300): MOM_20D > 0 ? 1 : -1
    - Gold Signal (Gold ETF): Price > MA200 ? 1 : -1

    Composite = 0.4 * MA + 0.4 * Mom + 0.2 * Gold
    Position = 0.3 if Composite < -0.4 else 1.0
    """

    def __init__(self, extreme_threshold: float = -0.4, extreme_position: float = 0.3):
        self.extreme_threshold = extreme_threshold
        self.extreme_position = extreme_position

    def compute_position_ratios(
        self,
        close_df: pd.DataFrame,
        market_symbol: str = "510300",
        gold_symbol: str = "518880",
    ) -> pd.Series:
        """
        Compute daily position ratios (0.0 to 1.0).

        Args:
            close_df: DataFrame of close prices (Date x Symbol)
            market_symbol: Ticker for Market Index (default 510300 HS300)
            gold_symbol: Ticker for Gold ETF (default 518880)

        Returns:
            pd.Series of position ratios (index=Date)
        """
        # Check if symbols exist
        if market_symbol not in close_df.columns:
            # Fallback: try 510050 if 510300 missing
            if "510050" in close_df.columns:
                market_symbol = "510050"
            else:
                print(
                    f"Warning: Market symbol {market_symbol} not found in prices. Timing disabled."
                )
                return pd.Series(1.0, index=close_df.index)

        market_close = close_df[market_symbol]

        # MA Signal: Price > MA200
        ma200 = market_close.rolling(window=200, min_periods=1).mean()
        # Use 0 for NaN to avoid issues, though min_periods=1 helps
        ma_signal = np.where(market_close > ma200, 1.0, -1.0)
        ma_signal = pd.Series(ma_signal, index=market_close.index)

        # Mom Signal: 20D Return > 0
        mom20 = market_close.pct_change(20)
        mom_signal = np.where(mom20 > 0, 1.0, -1.0)
        # Handle NaNs in mom20 (first 20 days) -> assume Bullish (1.0) to avoid defensive start?
        # Or neutral. Let's assume 1.0 to match "default full position".
        mom_signal[mom20.isna()] = 1.0
        mom_signal = pd.Series(mom_signal, index=market_close.index)

        # Gold Signal: Price > MA200
        if gold_symbol in close_df.columns:
            gold_close = close_df[gold_symbol]
            gold_ma200 = gold_close.rolling(window=200, min_periods=1).mean()
            gold_signal = np.where(gold_close > gold_ma200, 1.0, -1.0)
            gold_signal = pd.Series(gold_signal, index=gold_close.index)
        else:
            # If Gold missing, use Neutral (0.0)
            gold_signal = pd.Series(0.0, index=market_close.index)

        # Composite Score
        composite = 0.4 * ma_signal + 0.4 * mom_signal + 0.2 * gold_signal

        # Position Ratios
        # Default 1.0
        position_ratios = pd.Series(1.0, index=close_df.index)
        # Apply defensive mode
        position_ratios[composite < self.extreme_threshold] = self.extreme_position

        return position_ratios


class DualTimingModule:
    """
    双重择时模块 v3.0 (2025-11-30)

    设计理念：顺势而为，逆势空仓

    层级 1：大盘择时 (Index Timing) —— 战略防御
        - 指标：MA200 (经典牛熊分界线)
        - 逻辑：Index_Price < MA200 → 熊市 → 仓位降至 bear_position
        - 目标：规避系统性崩盘（2015股灾、2018熊市、2022大跌）

    层级 2：个股趋势 (Individual Timing) —— 战术防御
        - 指标：MA20 (短期趋势)
        - 逻辑：Price < MA20 → 剔除该标的，不买入/强制卖出
        - 目标：避免"因子选中但趋势向下"的逆势交易

    优于止损的原因：
        止损是"亏了再跑"（被动），择时是"不好不进"（主动）
        趋势机制本身就是动态止损：价格破位 → 自动卖出
    """

    def __init__(
        self,
        # 层级 1: 大盘择时参数
        index_ma_window: int = 200,
        bear_position: float = 0.1,
        index_symbol: str = "market_avg",  # "market_avg" 或具体代码如 "510300"
        # 层级 2: 个股趋势参数
        individual_ma_window: int = 20,
    ):
        """
        Args:
            index_ma_window: 大盘均线周期 (默认 200，牛熊分界线)
            bear_position: 熊市仓位 (默认 0.1，接近空仓)
            index_symbol: 大盘代理标的 ("market_avg" 使用全ETF等权平均)
            individual_ma_window: 个股趋势均线周期 (默认 20)
        """
        self.index_ma_window = index_ma_window
        self.bear_position = bear_position
        self.index_symbol = index_symbol
        self.individual_ma_window = individual_ma_window

    def compute_index_timing(self, close_df: pd.DataFrame) -> pd.Series:
        """
        计算大盘择时信号 (层级 1)

        Args:
            close_df: DataFrame of close prices (Date x Symbol)

        Returns:
            pd.Series: 仓位系数 (1.0=满仓, bear_position=熊市仓位)
        """
        if self.index_symbol == "market_avg":
            # 使用全市场 ETF 等权平均作为大盘代理
            index_price = close_df.mean(axis=1)
        elif self.index_symbol in close_df.columns:
            index_price = close_df[self.index_symbol]
        else:
            # 回退：尝试常用指数 ETF
            fallback_symbols = ["510300", "510050", "159919", "000300"]
            for symbol in fallback_symbols:
                if symbol in close_df.columns:
                    index_price = close_df[symbol]
                    print(
                        f"⚠️ DualTiming: index_symbol '{self.index_symbol}' not found, using '{symbol}'"
                    )
                    break
            else:
                print(
                    f"⚠️ DualTiming: No valid index symbol found, index timing disabled"
                )
                return pd.Series(1.0, index=close_df.index)

        # 计算 MA
        ma = index_price.rolling(window=self.index_ma_window, min_periods=1).mean()

        # 牛熊判断：Price > MA → 牛市(1.0), Price < MA → 熊市(bear_position)
        is_bull = index_price > ma
        position_ratios = pd.Series(1.0, index=close_df.index)
        position_ratios[~is_bull] = self.bear_position

        return position_ratios

    def compute_individual_trend_matrix(self, close_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算个股趋势状态矩阵 (层级 2)

        Args:
            close_df: DataFrame of close prices (Date x Symbol)

        Returns:
            pd.DataFrame: Boolean matrix (Date x Symbol)
                True = 趋势向上 (Price >= MA20)，可以买入
                False = 趋势向下 (Price < MA20)，不买入/强制卖出
        """
        # 计算每只 ETF 的 MA
        ma = close_df.rolling(window=self.individual_ma_window, min_periods=1).mean()

        # 趋势判断：Price >= MA → True (可买), Price < MA → False (不买)
        trend_ok = close_df >= ma

        return trend_ok

    def compute_all_signals(self, close_df: pd.DataFrame) -> dict:
        """
        一次性计算所有择时信号

        Args:
            close_df: DataFrame of close prices (Date x Symbol)

        Returns:
            dict: {
                'index_timing': pd.Series (仓位系数),
                'individual_trend': pd.DataFrame (趋势状态矩阵),
                'stats': dict (统计信息)
            }
        """
        index_timing = self.compute_index_timing(close_df)
        individual_trend = self.compute_individual_trend_matrix(close_df)

        # 统计信息
        total_days = len(close_df)
        bear_days = (index_timing < 1.0).sum()
        avg_trend_ok_pct = individual_trend.mean().mean() * 100  # 平均趋势良好比例

        stats = {
            "total_days": total_days,
            "bear_days": bear_days,
            "bear_ratio": bear_days / total_days * 100,
            "avg_trend_ok_pct": avg_trend_ok_pct,
        }

        return {
            "index_timing": index_timing,
            "individual_trend": individual_trend,
            "stats": stats,
        }
