"""价格数据加载器 - P0 优化：路径解耦，错误处理标准化"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from hk_midfreq.config import DEFAULT_PATH_CONFIG, PathConfig

# 设置日志
logger = logging.getLogger(__name__)

_TF_MAP: Dict[str, str] = {
    "daily": "1day",
    "1d": "1day",
    "1day": "1day",
    "60min": "60m",
    "60m": "60m",
    "30min": "30m",
    "30m": "30m",
    "15min": "15m",
    "15m": "15m",
    "5min": "5min",
    "5m": "5min",
}


def _normalize_symbol(symbol: str) -> str:
    """Convert symbol like '0700.HK' to raw filename prefix '0700HK'."""
    return symbol.replace(".", "")


def _normalize_timeframe(tf: str) -> str:
    """标准化时间框架名称"""
    key = tf.strip().lower()
    return _TF_MAP.get(key, key)


class DataLoadError(Exception):
    """价格数据加载异常 - P0 优化：标准化错误处理"""

    pass


@dataclass(frozen=True)
class PriceDataLoader:
    """统一的价格数据加载接口 - Linus 式简洁设计

    P0 优化：
    - 移除硬编码路径，使用 PathConfig
    - 标准化错误处理
    - 保持向后兼容（支持 root 参数）
    """

    path_config: PathConfig = field(default_factory=lambda: DEFAULT_PATH_CONFIG)
    root: Optional[Path] = None  # 向后兼容：可选硬编码路径

    def _get_root_dir(self) -> Path:
        """获取数据根目录 - 优先使用 root，否则使用 path_config"""
        if self.root is not None:
            return self.root
        return self.path_config.hk_raw_dir

    def load_price(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load canonical OHLCV price frame from raw parquet files.

        Requirements:
        - File pattern: <SYMBOL_NO_DOT>_<TF>_*.parquet (e.g., 0700HK_60m_*.parquet)
        - Must contain a 'close' column; otherwise raise DataLoadError.

        P0 优化：
        - 使用 PathConfig 自动发现数据路径
        - 标准化错误处理
        """

        logger.info(f"开始加载价格数据 - 股票: {symbol}, 时间框架: {timeframe}")

        symbol_key = _normalize_symbol(symbol)
        tf_key = _normalize_timeframe(timeframe)

        logger.debug(f"标准化参数 - 股票键: {symbol_key}, 时间框架键: {tf_key}")

        tf_dir = self._get_root_dir()
        if not tf_dir.exists():
            logger.error(f"原始数据目录不存在: {tf_dir}")
            raise DataLoadError(f"Raw data directory not found: {tf_dir}")

        logger.debug(f"搜索数据目录: {tf_dir}")

        pattern = f"{symbol_key}_{tf_key}_*.parquet"
        files = glob.glob(str(tf_dir / pattern))

        logger.debug(f"搜索模式: {pattern}, 找到文件数: {len(files)}")

        if not files:
            logger.error(f"未找到匹配的价格文件 - 模式: {pattern}")
            raise DataLoadError(
                f"No raw price file found for symbol={symbol} "
                f"timeframe={timeframe} pattern={pattern}"
            )

        latest_file = max(files, key=os.path.getmtime)
        logger.info(f"使用最新文件: {latest_file}")

        try:
            price_data = pd.read_parquet(latest_file)
            logger.debug(f"原始数据形状: {price_data.shape}")
            logger.debug(f"原始列名: {list(price_data.columns)}")
        except Exception as e:
            logger.error(f"读取parquet文件失败: {e}")
            raise DataLoadError(f"Failed to read parquet file {latest_file}: {e}")

        # 处理时间索引
        if "datetime" in price_data.columns:
            price_data = price_data.set_index("datetime")
            logger.debug("使用datetime列作为索引")
        elif "timestamp" in price_data.columns:
            price_data = price_data.set_index("timestamp")
            logger.debug("使用timestamp列作为索引")
        else:
            logger.debug("未找到时间列，保持原有索引")

        # 标准化列名
        original_columns = list(price_data.columns)
        price_data.columns = price_data.columns.str.lower()
        logger.debug(f"列名标准化: {original_columns} -> {list(price_data.columns)}")

        # 验证必需列
        if "close" not in price_data.columns:
            logger.error(f"缺少必需的close列，可用列: {list(price_data.columns)}")
            raise DataLoadError(
                f"Raw file {latest_file} lacks 'close' column; "
                f"cannot proceed without true prices."
            )

        # 数据质量检查
        data_start = price_data.index[0] if len(price_data) > 0 else "N/A"
        data_end = price_data.index[-1] if len(price_data) > 0 else "N/A"

        logger.info(f"价格数据加载完成 - 形状: {price_data.shape}")
        logger.info(f"数据时间范围: {data_start} 到 {data_end}")
        logger.debug(f"最终列名: {list(price_data.columns)}")

        return price_data
