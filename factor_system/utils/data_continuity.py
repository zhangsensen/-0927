"""
数据连续性检查和修复工具

功能：
- 检查数据时间序列连续性
- 修复数据缺失和跳跃
- 验证跳空计算的数据质量
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class DataContinuityChecker:
    """数据连续性检查器"""

    @staticmethod
    def ensure_daily_continuity(
        df: pd.DataFrame, expected_freq: str = "D", fill_method: str = "ffill"
    ) -> pd.DataFrame:
        """
        确保日度数据连续性

        Args:
            df: 输入DataFrame（索引为日期）
            expected_freq: 预期频率 ('D' for daily, 'B' for business day)
            fill_method: 填充方法 ('ffill', 'bfill', 'interpolate', 'none')

        Returns:
            连续性修复后的DataFrame
        """
        if df.empty:
            return df

        # 创建完整日期范围
        full_date_range = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=expected_freq
        )

        # 重新索引确保连续性
        df_reindexed = df.reindex(full_date_range)

        # 填充缺失值
        if fill_method == "ffill":
            df_filled = df_reindexed.fillna(method="ffill")
        elif fill_method == "bfill":
            df_filled = df_reindexed.fillna(method="bfill")
        elif fill_method == "interpolate":
            df_filled = df_reindexed.interpolate(method="linear")
        elif fill_method == "none":
            df_filled = df_reindexed
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")

        return df_filled

    @staticmethod
    def validate_gap_calculation(
        close_series: pd.Series, tolerance_days: int = 5
    ) -> Dict[str, Any]:
        """
        验证跳空计算的数据连续性

        Args:
            close_series: 收盘价序列
            tolerance_days: 容忍的最大间隔天数

        Returns:
            连续性检查结果字典
        """
        if len(close_series) < 2:
            return {
                "is_continuous": True,
                "max_gap_days": 0,
                "gap_count": 0,
                "total_days": len(close_series),
                "continuous_days": len(close_series),
                "warning": "Not enough data for validation",
            }

        # 检查索引连续性
        date_diffs = close_series.index[1:] - close_series.index[:-1]

        # 计算间隔天数
        gap_days = date_diffs.days if hasattr(date_diffs, "days") else date_diffs

        # 统计大间隔
        if isinstance(gap_days, int):
            max_gap = gap_days
            gap_count = 1 if gap_days > tolerance_days else 0
        else:
            max_gap = gap_days.max()
            gap_count = (gap_days > tolerance_days).sum()

        is_continuous = gap_count == 0

        result = {
            "is_continuous": is_continuous,
            "max_gap_days": max_gap,
            "gap_count": gap_count,
            "total_days": len(close_series),
            "continuous_days": len(close_series) - gap_count,
            "continuity_ratio": (len(close_series) - gap_count) / len(close_series),
        }

        # 添加警告信息
        if not is_continuous:
            result["warning"] = (
                f"Found {gap_count} gaps > {tolerance_days} days, "
                f"max gap = {max_gap} days"
            )

        return result

    @staticmethod
    def get_previous_trading_day(
        df: pd.DataFrame, current_date: pd.Timestamp
    ) -> pd.Timestamp:
        """
        获取真正的前一交易日（而非简单shift(1)）

        Args:
            df: 包含日期索引的DataFrame
            current_date: 当前日期

        Returns:
            前一交易日的日期
        """
        # 获取所有早于当前日期的交易日
        previous_dates = df.index[df.index < current_date]

        if len(previous_dates) == 0:
            return None

        # 返回最近的一个交易日
        return previous_dates[-1]

    @staticmethod
    def calculate_safe_gap(df: pd.DataFrame) -> pd.Series:
        """
        安全的跳空计算（考虑数据连续性）

        Args:
            df: 包含open和close列的DataFrame

        Returns:
            跳空百分比序列
        """
        if "open" not in df.columns or "close" not in df.columns:
            raise ValueError("DataFrame must contain 'open' and 'close' columns")

        gaps = []
        dates = []

        for i in range(1, len(df)):
            current_date = df.index[i]
            prev_date = df.index[i - 1]

            # 检查日期间隔
            date_diff = (current_date - prev_date).days

            # 如果间隔过大（>7天），跳过该跳空计算
            if date_diff > 7:
                gaps.append(0.0)  # 或者使用NaN
            else:
                gap_pct = df["open"].iloc[i] / df["close"].iloc[i - 1] - 1
                gaps.append(gap_pct)

            dates.append(current_date)

        return pd.Series(gaps, index=dates, name="gap_pct")

    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """
        检测数据质量问题

        Args:
            df: 待检测的DataFrame

        Returns:
            数据质量报告
        """
        issues = {
            "total_rows": len(df),
            "missing_values": {},
            "duplicate_index": False,
            "negative_values": {},
            "zero_values": {},
            "outliers": {},
        }

        # 检查缺失值
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues["missing_values"][col] = {
                    "count": missing_count,
                    "ratio": missing_count / len(df),
                }

        # 检查重复索引
        issues["duplicate_index"] = df.index.duplicated().any()

        # 检查负值（对于价格和成交量）
        price_volume_cols = ["open", "high", "low", "close", "volume", "turnover"]
        for col in price_volume_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues["negative_values"][col] = negative_count

        # 检查零值
        for col in price_volume_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    issues["zero_values"][col] = {
                        "count": zero_count,
                        "ratio": zero_count / len(df),
                    }

        return issues
