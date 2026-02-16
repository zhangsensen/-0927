"""
ETF数据爬虫模块

提供从各大财经网站爬取ETF相关数据的功能
"""

from .sources.eastmoney_crawler import EastmoneyETFCrawler
from .sources.eastmoney_detail_crawler import EastmoneyDetailCrawler
from .sources.sina_crawler import SinaETFCrawler

__all__ = ["EastmoneyETFCrawler", "EastmoneyDetailCrawler", "SinaETFCrawler"]
