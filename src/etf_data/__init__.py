"""
ETF下载管理器包初始化
"""

from .core.config import ETFConfig, ETFDataSource
from .core.data_manager import ETFDataManager
from .core.downloader import ETFDownloadManager
from .core.etf_list import ETFListManager
from .core.models import (
    DownloadResult,
    DownloadStats,
    ETFDownloadType,
    ETFExchange,
    ETFInfo,
    ETFPriority,
)

__version__ = "1.0.0"
__author__ = "量化首席工程师"

__all__ = [
    "ETFDownloadManager",
    "ETFConfig",
    "ETFDataSource",
    "ETFInfo",
    "DownloadResult",
    "DownloadStats",
    "ETFPriority",
    "ETFExchange",
    "ETFDownloadType",
    "ETFListManager",
    "ETFDataManager",
]
