"""CSV数据提供者 - 支持A股CSV数据格式"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class CSVDataProvider(DataProvider):
    """
    CSV数据提供者

    支持A股CSV数据格式:
    - 文件路径: {data_dir}/{symbol}/{symbol}_{timeframe}_YYYY-MM-DD.csv
    - 文件格式: 前2行为标题（跳过），数据列为Date,Close,High,Low,Open,Volume

    Example:
        provider = CSVDataProvider(data_dir='/path/to/a股')
        data = provider.load_price_data(
            symbols=['300450.SZ'],
            timeframe='1d',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2025, 1, 1),
        )
    """

    def __init__(self, data_dir: str):
        """
        初始化CSV数据提供者

        Args:
            data_dir: 数据根目录，包含各股票子目录
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            logger.warning(f"数据目录不存在: {self.data_dir}")

        logger.info(f"初始化CSVDataProvider: {self.data_dir}")

    def load_price_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载A股CSV数据

        Args:
            symbols: 股票代码列表 (e.g. ['300450.SZ', '002074.SZ'])
            timeframe: 时间框架 ('1d', '1h', etc.)
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            MultiIndex DataFrame (timestamp, symbol) with columns: open, high, low, close, volume
        """
        all_data = []

        for symbol in symbols:
            symbol_data = self._load_single_symbol(
                symbol, timeframe, start_date, end_date
            )

            if not symbol_data.empty:
                all_data.append(symbol_data)
            else:
                logger.warning(f"未加载到{symbol}的数据")

        if not all_data:
            logger.warning("所有symbol的数据都为空")
            return pd.DataFrame()

        # 合并所有symbol
        result = pd.concat(all_data)

        logger.info(
            f"加载完成: {len(symbols)}个symbol, "
            f"{len(result)}行数据, "
            f"时间范围 {result.index.get_level_values('timestamp').min()} ~ "
            f"{result.index.get_level_values('timestamp').max()}"
        )

        return result

    def _load_single_symbol(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        加载单个symbol的CSV数据

        Args:
            symbol: 股票代码
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            MultiIndex DataFrame
        """
        # 查找symbol目录
        symbol_dir = self.data_dir / symbol

        if not symbol_dir.exists():
            logger.warning(f"股票目录不存在: {symbol_dir}")
            return pd.DataFrame()

        # 查找匹配时间框架的CSV文件
        pattern = f"{symbol}_{timeframe}_*.csv"
        csv_files = list(symbol_dir.glob(pattern))

        if not csv_files:
            logger.warning(f"未找到{symbol}的{timeframe}数据: {symbol_dir}/{pattern}")
            return pd.DataFrame()

        # 使用最新的文件（按文件名排序，日期在最后）
        latest_file = sorted(csv_files)[-1]
        logger.debug(f"加载文件: {latest_file}")

        try:
            # 加载CSV
            # 注意：A股CSV格式为第1行标题，第2行重复股票代码，第3行开始数据
            # 使用header=0读取第1行作为列名，skiprows=[1]跳过第2行
            df = pd.read_csv(latest_file, header=0, skiprows=[1])

            # 验证列名
            expected_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
            if not all(col in df.columns for col in expected_columns):
                logger.error(
                    f"文件列不匹配: {latest_file}, 实际列: {df.columns.tolist()}"
                )
                return pd.DataFrame()

            # 选择需要的列
            df = df[expected_columns].copy()

            # 转换为标准格式
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.rename(
                columns={
                    "Date": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # 确保数值类型
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # 删除NaN行
            df = df.dropna()

            # 过滤日期范围
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            if df.empty:
                logger.warning(
                    f"{symbol}在指定日期范围内无数据: {start_date} ~ {end_date}"
                )
                return pd.DataFrame()

            # 添加symbol列并设置MultiIndex
            df["symbol"] = symbol
            df = df.set_index(["timestamp", "symbol"])

            # 按时间排序
            df = df.sort_index()

            logger.debug(f"{symbol}加载成功: {len(df)}行数据")

            return df

        except Exception as e:
            logger.error(f"加载{symbol}数据失败: {e}", exc_info=True)
            return pd.DataFrame()

    def get_available_symbols(self) -> List[str]:
        """
        获取可用的股票代码列表

        Returns:
            股票代码列表
        """
        if not self.data_dir.exists():
            return []

        symbols = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                symbols.append(item.name)

        return sorted(symbols)

    def get_available_timeframes(self, symbol: str) -> List[str]:
        """
        获取指定symbol可用的时间框架

        Args:
            symbol: 股票代码

        Returns:
            时间框架列表 (e.g. ['1d', '1h'])
        """
        symbol_dir = self.data_dir / symbol

        if not symbol_dir.exists():
            return []

        timeframes = set()
        for csv_file in symbol_dir.glob(f"{symbol}_*.csv"):
            # 解析文件名: {symbol}_{timeframe}_{date}.csv
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                timeframe = parts[1]
                timeframes.add(timeframe)

        return sorted(timeframes)

    def load_fundamental_data(
        self,
        symbols: List[str],
        fields: List[str],
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        加载基本面数据（CSV数据源不支持）

        Args:
            symbols: 股票代码列表
            fields: 字段列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            空DataFrame（CSV数据源不支持基本面数据）
        """
        logger.warning("CSVDataProvider不支持基本面数据")
        return pd.DataFrame()

    def get_trading_calendar(
        self,
        market: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[datetime]:
        """
        获取交易日历（简化实现：从现有数据推断）

        Args:
            market: 市场代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        # 简化实现：返回日期范围内所有工作日
        # 真实实现应该考虑节假日

        date_range = pd.date_range(
            start=start_date, end=end_date, freq="B"
        )  # B = Business day
        return date_range.to_pydatetime().tolist()
