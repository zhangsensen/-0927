"""
A股数据加载器

负责从原始parquet文件加载A股分钟级数据，并标准化为因子引擎所需格式。
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AShareDataLoader:
    """A股数据加载器"""

    def __init__(self, data_root: str | None = None):
        """
        初始化数据加载器

        Args:
            data_root: 原始数据根目录
        """
        if data_root is None:
            try:
                from scripts.path_utils import get_paths

                paths = get_paths()
                data_root = Path(paths["raw_root"]) / "SH"
            except Exception:
                data_root = Path("raw/SH")
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"数据目录不存在: {self.data_root}")

    def load_minute_data(self, symbol: str) -> pd.DataFrame:
        """
        加载A股分钟级数据并标准化

        Args:
            symbol: 股票代码（如 600036.SH）

        Returns:
            标准化的OHLCV DataFrame，索引为DatetimeIndex

        Raises:
            FileNotFoundError: 数据文件不存在
            ValueError: 数据格式不符合预期
        """
        file_path = self.data_root / f"{symbol}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.info(f"加载A股数据: {symbol}")

        # 读取parquet文件
        df = pd.read_parquet(file_path)

        # 验证必需列
        required_cols = ["open", "high", "low", "close", "volume", "turnover"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")

        # 处理时间索引（支持timestamp或datetime列名）
        time_col = None
        if "timestamp" in df.columns:
            time_col = "timestamp"
        elif "datetime" in df.columns:
            time_col = "datetime"
        elif df.index.name in ["timestamp", "datetime"]:
            time_col = df.index.name

        if time_col is None:
            raise ValueError("缺少timestamp或datetime列/索引")

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

        # 设置时区为上海时区（如果没有时区信息则本地化，否则转换）
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Shanghai")
        else:
            df.index = df.index.tz_convert("Asia/Shanghai")

        df = df.sort_index()

        # 字段映射：turnover -> amount
        df = df.rename(columns={"turnover": "amount"})

        # 保存元信息到attrs
        if "symbol" in df.columns:
            df.attrs["symbol"] = df["symbol"].iloc[0]
        else:
            df.attrs["symbol"] = symbol

        if "exchange" in df.columns:
            df.attrs["exchange"] = df["exchange"].iloc[0]

        if "sector" in df.columns:
            df.attrs["sector"] = df["sector"].iloc[0]

        # 返回标准OHLCV格式
        output_cols = ["open", "high", "low", "close", "volume", "amount"]
        result = df[output_cols].copy()

        # 数据质量检查
        null_counts = result.isnull().sum()
        if null_counts.any():
            logger.warning(
                f"{symbol} 存在空值: {null_counts[null_counts > 0].to_dict()}"
            )

        logger.info(
            f"✅ {symbol} 数据加载完成: {len(result)} 行, "
            f"时间范围 {result.index[0]} 至 {result.index[-1]}"
        )

        return result

    def validate_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        验证并过滤交易时间（A股：09:30-11:30, 13:00-15:00）

        Args:
            df: 输入DataFrame

        Returns:
            过滤后的DataFrame
        """
        morning_mask = (df.index.time >= pd.Timestamp("09:30").time()) & (
            df.index.time <= pd.Timestamp("11:30").time()
        )
        afternoon_mask = (df.index.time >= pd.Timestamp("13:00").time()) & (
            df.index.time <= pd.Timestamp("15:00").time()
        )

        valid_mask = morning_mask | afternoon_mask
        filtered = df[valid_mask].copy()

        removed_count = len(df) - len(filtered)
        if removed_count > 0:
            logger.info(f"过滤非交易时间数据: {removed_count} 行")

        return filtered


def load_a_share_minute(
    symbol: str, data_root: Optional[str] = None, validate_hours: bool = True
) -> pd.DataFrame:
    """
    便捷函数：加载A股分钟数据

    Args:
        symbol: 股票代码
        data_root: 数据根目录（可选）
        validate_hours: 是否验证交易时间

    Returns:
        标准化的OHLCV DataFrame
    """
    loader = AShareDataLoader(data_root) if data_root else AShareDataLoader()
    df = loader.load_minute_data(symbol)

    if validate_hours:
        df = loader.validate_trading_hours(df)

    return df
