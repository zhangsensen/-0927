"""
爬虫核心基类
提供通用的HTTP请求、重试机制、数据验证功能
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    """爬虫配置"""

    retry_times: int = 3
    retry_delay: float = 2.0
    timeout: int = 30
    rate_limit: float = 1.0  # 请求间隔（秒）
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class BaseCrawler(ABC):
    """
    爬虫基类

    所有具体爬虫都需要继承此类，实现parse和save方法
    """

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = self._create_session()
        self._last_request_time = 0

    def _create_session(self) -> requests.Session:
        """创建带有重试机制的session"""
        session = requests.Session()

        # 配置重试策略
        retry_strategy = Retry(
            total=self.config.retry_times,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置默认headers
        session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "application/json, text/javascript, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

        return session

    def _rate_limit(self):
        """请求频率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.rate_limit:
            time.sleep(self.config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def fetch(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        获取数据（带重试和频率限制）

        Args:
            url: 请求URL
            **kwargs: 传递给requests的参数

        Returns:
            Response对象或None（失败时）
        """
        self._rate_limit()

        try:
            logger.debug(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.config.timeout, **kwargs)
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    @abstractmethod
    def parse(self, response: requests.Response) -> pd.DataFrame:
        """
        解析响应数据

        Args:
            response: HTTP响应

        Returns:
            解析后的DataFrame
        """
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame, output_path: Path):
        """
        保存数据

        Args:
            df: 要保存的数据
            output_path: 输出路径
        """
        pass

    def crawl(self, url: str, output_path: Path, **kwargs) -> bool:
        """
        完整爬取流程

        Args:
            url: 目标URL
            output_path: 输出路径
            **kwargs: 传递给fetch的参数

        Returns:
            是否成功
        """
        response = self.fetch(url, **kwargs)
        if response is None:
            return False

        try:
            df = self.parse(response)
            if df.empty:
                logger.warning(f"Parsed empty data from {url}")
                return False

            self.save(df, output_path)
            logger.info(f"Successfully saved {len(df)} rows to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to process data from {url}: {e}")
            return False
