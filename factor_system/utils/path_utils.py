"""路径处理标准化工具"""

import logging
import re
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


class PathStandardizer:
    """标准化股票代码路径处理"""

    @staticmethod
    def standardize_symbol(symbol: str) -> str:
        """
        标准化股票代码格式
        统一转换为 XXXX.HK 格式

        Args:
            symbol: 输入的股票代码 (可能是 0700, 0700.HK, etc.)

        Returns:
            str: 标准化后的股票代码 (0700.HK)
        """
        if not symbol:
            raise ValueError("股票代码不能为空")

        symbol = str(symbol).strip()

        # 如果已经是.HK格式，直接返回
        if symbol.endswith(".HK"):
            return symbol.upper()

        # 如果已经是.SZ或.SS格式，保持不变
        if symbol.endswith(".SZ") or symbol.endswith(".SS"):
            return symbol.upper()

        # 如果是纯数字，添加.HK后缀
        if re.match(r"^\d+$", symbol):
            return f"{symbol}.HK"

        # 移除现有后缀，添加.HK
        clean_symbol = re.sub(r"\.[A-Z]+$", "", symbol)  # noqa: PD005
        return f"{clean_symbol}.HK"

    @staticmethod
    def get_data_path(
        data_root: Union[str, Path],
        symbol: str,
        timeframe: str,
        date_range: Optional[str] = None,
    ) -> Path:
        """
        获取标准化数据文件路径

        Args:
            data_root: 数据根目录
            symbol: 股票代码
            timeframe: 时间框架
            date_range: 日期范围，默认为None

        Returns:
            Path: 标准化的文件路径
        """
        data_root = Path(data_root)
        standardized_symbol = PathStandardizer.standardize_symbol(symbol)

        # 如果没有指定日期范围，使用默认格式
        if date_range is None:
            date_range = "2025-03-05_2025-09-01"

        # 构建文件名 - 支持多种格式
        possible_extensions = [".parquet", ".csv"]

        for ext in possible_extensions:
            filename = f"{standardized_symbol}_{timeframe}_{date_range}{ext}"
            file_path = data_root / filename

            if file_path.exists():
                return file_path

        # 如果都不存在，返回parquet格式的路径（作为默认）
        filename = f"{standardized_symbol}_{timeframe}_{date_range}.parquet"
        return data_root / filename

    @staticmethod
    def validate_symbol_consistency(symbol: str, file_path: Union[str, Path]) -> bool:
        """
        验证股票代码与文件路径的一致性

        Args:
            symbol: 股票代码
            file_path: 文件路径

        Returns:
            bool: 是否一致
        """
        try:
            file_path = Path(file_path)
            expected_symbol = PathStandardizer.standardize_symbol(symbol)

            # 从文件名中提取股票代码
            file_symbol = file_path.stem.split("_")[0]
            file_symbol = PathStandardizer.standardize_symbol(file_symbol)

            return expected_symbol == file_symbol
        except Exception as e:
            logger.warning(f"路径一致性验证失败: {e}")
            return False

    @staticmethod
    def find_symbol_files(data_root: Union[str, Path], symbol: str) -> list[Path]:
        """
        查找指定股票代码的所有数据文件

        Args:
            data_root: 数据根目录
            symbol: 股票代码

        Returns:
            list[Path]: 找到的文件路径列表
        """
        data_root = Path(data_root)
        standardized_symbol = PathStandardizer.standardize_symbol(symbol)

        # 搜索模式
        patterns = [f"{standardized_symbol}_*.parquet", f"{standardized_symbol}_*.csv"]

        found_files = []
        for pattern in patterns:
            found_files.extend(data_root.glob(pattern))

        return sorted(found_files)

    @staticmethod
    def extract_timeframe_from_path(file_path: Union[str, Path]) -> Optional[str]:
        """
        从文件路径中提取时间框架

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 时间框架，如果无法提取则返回None
        """
        try:
            file_path = Path(file_path)
            parts = file_path.stem.split("_")

            if len(parts) >= 2:
                return parts[1]  # 假设格式为 SYMBOL_TIMEFRAME_DATE

            return None
        except Exception as e:
            logger.warning(f"无法从路径提取时间框架: {e}")
            return None

    @staticmethod
    def normalize_timeframe(timeframe: str) -> str:
        """
        标准化时间框架格式

        Args:
            timeframe: 输入的时间框架

        Returns:
            str: 标准化的时间框架
        """
        timeframe = timeframe.lower().strip()

        # 标准化映射
        timeframe_mapping = {
            "1d": "1day",
            "daily": "1day",
            "day": "1day",
            "1h": "1hour",
            "hourly": "1hour",
            "hour": "1hour",
            "60min": "1hour",
            "30min": "30min",
            "15min": "15min",
            "5min": "5min",
            "1min": "1min",
        }

        return timeframe_mapping.get(timeframe, timeframe)
