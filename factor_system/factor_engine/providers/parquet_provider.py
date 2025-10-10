"""Parquet文件数据提供者 - 统一数据接口（修复版）"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from factor_system.factor_engine.providers.base import DataProvider

logger = logging.getLogger(__name__)


class ParquetDataProvider(DataProvider):
    """
    基于PyArrow的Parquet数据提供者（修复版）

    支持从文件名解析symbol和timeframe，自动添加缺失的列
    适配当前数据文件格式：{SYMBOL}{TIMEFRAME}_{START}_{END}.parquet
    """

    def __init__(self, raw_data_dir: Path):
        """
        初始化Parquet数据提供者

        Args:
            raw_data_dir: 原始数据根目录，如 Path("raw")

        Raises:
            ValueError: 如果数据目录不存在或PyArrow不可用
        """
        self.raw_data_dir = raw_data_dir
        self.hk_dir = raw_data_dir / "raw" / "HK"
        self.us_dir = raw_data_dir / "raw" / "US"

        # 强制要求PyArrow
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ValueError("PyArrow是必需的依赖，请安装: pip install pyarrow") from e

        # 强制要求数据目录存在
        if not self.hk_dir.exists():
            raise ValueError(f"HK数据目录不存在: {self.hk_dir}")

        # US数据目录可选
        if hasattr(self, 'us_dir') and not self.us_dir.exists():
            logger.warning(f"US数据目录不存在: {self.us_dir}")

        # 创建文件名到symbol和timeframe的映射
        self._build_file_mapping()

        logger.info(f"ParquetDataProvider初始化成功: {len(self._file_mapping)}个数据文件")

    def _build_file_mapping(self) -> None:
        """构建文件名到symbol和timeframe的映射"""
        self._file_mapping = {}

        # 时间框架映射
        timeframe_mapping = {
            '1min': '1min',
            '2min': '2min',
            '3min': '3min',
            '5min': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '60min',
            '2h': '2h',
            '4h': '4h',
            '1day': 'daily',
            '15min': '15min',
            '30min': '30min',
            '60min': '60min',
        }

        # 扫描所有parquet文件
        for file_path in self.hk_dir.glob("*.parquet"):
            filename = file_path.stem  # 去掉.parquet扩展名

            # 解析文件名：格式如 "0700HK_1day_2025-03-05_2025-09-01"
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol_part = parts[0]  # "0700HK"
                timeframe_part = parts[1]  # "1day"

                # 转换symbol格式：0700HK -> 0700.HK
                if symbol_part.endswith('HK'):
                    symbol = symbol_part[:-2] + '.HK'
                else:
                    symbol = symbol_part  # 保持原样

                # 映射timeframe格式
                timeframe = timeframe_mapping.get(timeframe_part, timeframe_part)

                self._file_mapping[file_path] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'filename': filename
                }

        logger.info(f"文件映射构建完成: {len(self._file_mapping)}个文件")

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
            symbols: 股票代码列表，格式如 ['0700.HK', '0005.HK']
            timeframe: 时间框架，格式如 '15min', 'daily'
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with MultiIndex(timestamp, symbol) and OHLCV columns

        Raises:
            ValueError: 如果数据格式不正确或找不到数据
        """
        # 验证输入格式
        self._validate_inputs(symbols, timeframe)

        # 查找匹配的文件
        matching_files = []
        for file_path, file_info in self._file_mapping.items():
            if file_info['symbol'] in symbols and file_info['timeframe'] == timeframe:
                matching_files.append((file_path, file_info))

        if not matching_files:
            logger.warning(f"没有找到匹配的数据文件: symbols={symbols}, timeframe={timeframe}")
            return pd.DataFrame()

        # 加载并合并数据
        all_data = []
        for file_path, file_info in matching_files:
            try:
                # 读取parquet文件 - 使用日期范围过滤减少内存占用
                start_timestamp = pd.Timestamp(start_date)
                end_timestamp = pd.Timestamp(end_date)

                # 先读取文件获取日期范围
                dataset = ds.dataset(file_path)
                table = dataset.to_table(filter=(
                    (ds.field('timestamp') >= start_timestamp) &
                    (ds.field('timestamp') <= end_timestamp)
                ))
                df = table.to_pandas()

                # 添加symbol和timeframe列
                df['symbol'] = file_info['symbol']
                df['timeframe'] = file_info['timeframe']

                if not df.empty:
                    all_data.append(df)

            except Exception as e:
                logger.error(f"读取文件失败 {file_path}: {e}")
                continue

        if not all_data:
            logger.warning("没有找到符合日期范围的数据")
            return pd.DataFrame()

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)

        # 创建MultiIndex
        combined_df = combined_df.set_index(['timestamp', 'symbol']).sort_index()

        # 验证数据完整性
        self._validate_data_schema(combined_df, symbols, timeframe)

        logger.info(f"数据加载完成: {len(symbols)}个标的, {len(combined_df)}行")
        return combined_df

    def _validate_inputs(self, symbols: List[str], timeframe: str) -> None:
        """验证输入参数格式"""
        valid_timeframes = {'1min', '2min', '3min', '5min', '15min', '30min', '60min', '2h', '4h', 'daily'}

        if not symbols:
            raise ValueError("symbols列表不能为空")

        for symbol in symbols:
            if not (symbol.endswith('.HK') or symbol.endswith('.US')):
                raise ValueError(f"symbol格式错误，应为'0700.HK'或'BABA.US'格式: {symbol}")

        if timeframe not in valid_timeframes:
            raise ValueError(f"不支持的timeframe: {timeframe}，支持的格式: {valid_timeframes}")

    def _validate_data_schema(self, df: pd.DataFrame, symbols: List[str], timeframe: str) -> None:
        """验证数据schema完整性"""
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(f"数据缺少必需列: {missing_columns}")

        # 验证symbol范围
        data_symbols = set(df.index.get_level_values('symbol').unique())
        expected_symbols = set(symbols)
        missing_symbols = expected_symbols - data_symbols

        if missing_symbols:
            logger.warning(f"以下symbol没有数据: {missing_symbols}")

    def load_fundamental_data(
        self,
        symbols: List[str],
        fields: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """加载基本面数据（当前版本不支持）"""
        raise NotImplementedError("ParquetDataProvider不支持基本面数据")

    def get_trading_calendar(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """获取交易日历（从价格数据推断）"""
        try:
            # 加载一个完整的时间段数据来推断交易日历
            sample_symbol = symbols[0] if symbols else "0700.HK"
            sample_timeframe = "daily"  # 使用日线数据获取交易日历

            # 查找匹配的文件
            matching_files = []
            for file_path, file_info in self._file_mapping.items():
                if file_info['symbol'] == sample_symbol and file_info['timeframe'] == sample_timeframe:
                    matching_files.append(file_path)
                    break

            if not matching_files:
                logger.warning("无法找到交易日历数据")
                return pd.DataFrame()

            # 读取数据
            file_path = matching_files[0]
            table = pq.read_table(file_path)
            df = table.to_pandas()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 创建交易日历
            trading_days = df['timestamp'].dt.date.unique()
            trading_days = pd.to_datetime(sorted(trading_days))

            calendar_df = pd.DataFrame({
                'trading_day': trading_days,
                'is_trading_day': True
            }).set_index('trading_day')

            return calendar_df

        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return pd.DataFrame()

    def get_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """获取可用的股票代码列表"""
        symbols = set()

        for file_info in self._file_mapping.values():
            if timeframe is None or file_info['timeframe'] == timeframe:
                symbols.add(file_info['symbol'])

        return sorted(list(symbols))

    def get_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """获取可用的时间框架列表"""
        timeframes = set()

        for file_info in self._file_mapping.values():
            if symbol is None or file_info['symbol'] == symbol:
                timeframes.add(file_info['timeframe'])

        return sorted(list(timeframes))