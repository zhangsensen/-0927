"""
ETF下载管理器脚本模块
"""

from .quick_download import main as quick_download_main
from .batch_download import main as batch_download_main
from .download_etf_manager import main as manager_main

__all__ = [
    "quick_download_main",
    "batch_download_main",
    "manager_main"
]