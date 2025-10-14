"""
分钟择时最小实现（A股）

设计要点：
- 基数据为1分钟，内部聚合生成5/30/60分钟序列，避免多口径
- 所有信号以bar收盘时间戳打点，执行严格在下一根bar
- 提供VWAP执行的近似撮合（下一根bar的成交额/成交量）
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ExecutionConfig:
    use_vwap: bool = True
    allow_partial_fill: bool = True


class MinuteTimingEngine:
    """分钟择时引擎（最小闭环实现）"""

    def __init__(self, exec_cfg: ExecutionConfig | None = None):
        self.exec_cfg = exec_cfg or ExecutionConfig()

    def _resample(self, df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
        out = df_1m.resample(rule, label="right", closed="right").agg(agg).dropna()
        return out

    def generate_signals(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        生成分钟级择时信号（最小实现）：
        - 主信号（5m）：收盘价 > 5m滚动VWAP
        - 趋势过滤：30m与60m均线向上

        返回：index为bar收盘时间戳，列[signal_5m, trend_filter, final_signal]
        """
        if df_1m.index.tz is None:
            df_1m = df_1m.tz_localize("Asia/Shanghai")

        df_5m = self._resample(df_1m, "5T")
        # 5m近似VWAP：amount/volume
        vwap_5m = (df_5m["amount"] / df_5m["volume"]).replace([pd.NA, float("inf")], pd.NA)
        signal_5m = (df_5m["close"] > vwap_5m).astype(int)

        # 趋势过滤（30/60m）：MA方向一致且price>MA
        df_30m = self._resample(df_1m, "30T")
        df_60m = self._resample(df_1m, "60T")
        ma_30 = df_30m["close"].rolling(5, min_periods=5).mean()
        ma_60 = df_60m["close"].rolling(5, min_periods=5).mean()
        trend_30 = (df_30m["close"] > ma_30) & (ma_30.diff() > 0)
        trend_60 = (df_60m["close"] > ma_60) & (ma_60.diff() > 0)

        # 对齐到5m时间戳（前向填充到近邻5m收盘）
        trend_30_on_5m = trend_30.reindex(df_5m.index, method="ffill").fillna(False)
        trend_60_on_5m = trend_60.reindex(df_5m.index, method="ffill").fillna(False)
        trend_filter = (trend_30_on_5m & trend_60_on_5m).astype(int)

        out = pd.DataFrame({
            "signal_5m": signal_5m,
            "trend_filter": trend_filter,
        }, index=df_5m.index)
        out["final_signal"] = ((out["signal_5m"] == 1) & (out["trend_filter"] == 1)).astype(int)

        # 交易在下一根bar执行（严格避免信息穿越）
        out[["signal_5m", "trend_filter", "final_signal"]] = out[["signal_5m", "trend_filter", "final_signal"]].shift(1)
        return out

    def simulate_execution(self, df_1m: pd.DataFrame, signals_5m: pd.DataFrame) -> pd.DataFrame:
        """
        使用下一根5m的VWAP（或close）进行近似撮合，支持部分成交。

        返回：包含列[exec_price, filled_ratio]，索引为5m收盘时间戳
        """
        df_5m = self._resample(df_1m, "5T")
        # 下一根bar的价格基准
        if self.exec_cfg.use_vwap:
            next_price = (df_5m["amount"] / df_5m["volume"]).shift(-1)
        else:
            next_price = df_5m["close"].shift(-1)

        # 简单的容量/部分成交模型：成交额阈值越大，filled_ratio越高
        turnover = df_5m["amount"].shift(-1)
        cap_low, cap_high = turnover.quantile(0.1), turnover.quantile(0.9)
        filled_ratio = ((turnover - cap_low) / (cap_high - cap_low)).clip(0.0, 1.0)

        out = pd.DataFrame(index=df_5m.index)
        out["exec_price"] = next_price
        out["filled_ratio"] = filled_ratio.where(self.exec_cfg.allow_partial_fill, other=1.0)

        # 仅在有信号时视作下单，其余filled_ratio=0
        active = (signals_5m["final_signal"] == 1)
        out.loc[~active, "filled_ratio"] = 0.0
        return out

