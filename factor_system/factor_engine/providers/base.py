"""数据提供者基类 - 统一数据访问接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd


class DataProvider(ABC):
    """
    统一数据提供接口

    职责:
    - 提供OHLCV价格数据
    - 提供基本面数据（可选）
    - 提供交易日历
    - 标准化数据格式
    """

    @abstractmethod
    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载OHLCV价格数据

        Args:
            symbols: 股票代码列表，如 ["0700.HK", "0005.HK"]
            timeframe: 时间框架，如 "15min", "60min", "daily"
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with MultiIndex(timestamp, symbol) and columns [open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def load_fundamental_data(
        self,
        symbols: List[str],
        fields: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        加载基本面数据

        Args:
            symbols: 股票代码列表
            fields: 字段列表，如 ["pe_ratio", "pb_ratio", "market_cap"]
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）

        Returns:
            DataFrame with MultiIndex(timestamp, symbol) and requested fields
        """
        pass

    @abstractmethod
    def get_trading_calendar(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[datetime]:
        """
        获取交易日历

        Args:
            market: 市场代码，如 "HK", "A", "US"
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据完整性

        Args:
            data: 待验证数据

        Returns:
            是否通过验证
        """
        if data.empty:
            return False

        # 检查必需列
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(data.columns):
            return False

        # 检查是否有NaN
        if data[list(required_columns)].isnull().any().any():
            return False

        return True

