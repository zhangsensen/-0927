"""
分钟级数据提供者

功能：
- 加载单日/多日分钟数据
- 数据预处理和验证
- 支持四段小时K生成
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MinuteDataProvider:
    """分钟级数据提供者"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_minute_data(
        self, symbol: str, date: str, validate: bool = True
    ) -> pd.DataFrame:
        """
        加载单日分钟数据

        Args:
            symbol: 股票代码
            date: 日期字符串 (YYYY-MM-DD)
            validate: 是否验证数据完整性

        Returns:
            单日分钟级DataFrame
        """
        file_path = self.data_dir / f"{symbol}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # 加载完整数据
        df = pd.read_parquet(file_path)

        # 确保datetime列为索引
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")

        # 筛选指定日期
        target_date = pd.to_datetime(date).date()
        mask = df.index.date == target_date
        result = df[mask].copy()

        if result.empty:
            raise ValueError(f"No data found for {symbol} on {date}")

        # 数据验证
        if validate:
            self._validate_minute_data(result, date)

        return result

    def load_date_range(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        加载日期范围内的分钟数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            多日分钟级DataFrame
        """
        file_path = self.data_dir / f"{symbol}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_parquet(file_path)

        # 确保datetime列为索引
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)

        # 筛选日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df.index >= start) & (df.index <= end)

        return df[mask].copy()

    def aggregate_to_daily(self, minute_data: pd.DataFrame) -> pd.DataFrame:
        """
        将分钟数据聚合为日线数据

        Args:
            minute_data: 分钟级DataFrame

        Returns:
            日线级DataFrame
        """
        daily_data = minute_data.groupby(minute_data.index.date).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "turnover": "sum",
            }
        )

        daily_data.index = pd.to_datetime(daily_data.index)
        return daily_data

    def _validate_minute_data(self, data: pd.DataFrame, date: str) -> None:
        """
        验证分钟数据完整性

        检查：
        - 数据时间范围是否在交易时段内
        - 是否有异常的时间跳跃
        """
        if data.empty:
            return

        # 检查时间范围
        min_time = data.index.min().time()
        max_time = data.index.max().time()

        expected_start = pd.Timestamp("09:30").time()
        expected_end = pd.Timestamp("15:00").time()

        if min_time < expected_start or max_time > expected_end:
            print(
                f"Warning: Data time range {min_time}-{max_time} outside trading hours"
            )

        # 检查时间间隔
        time_diffs = data.index[1:] - data.index[:-1]

        # 允许午休时段的大间隔
        large_gaps = time_diffs[time_diffs > pd.Timedelta("2 minutes")]
        if len(large_gaps) > 1:  # 除了午休，不应有其他大间隔
            print(f"Warning: Found {len(large_gaps)} large time gaps in data")

    def get_trading_minutes_count(self, date: str) -> int:
        """
        获取指定日期的交易分钟数

        正常交易日：240分钟 (上午120 + 下午120)
        """
        # 上午：09:30-11:30 = 120分钟
        # 下午：13:00-15:00 = 120分钟
        return 240
