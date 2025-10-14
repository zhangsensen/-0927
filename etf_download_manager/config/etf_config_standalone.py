#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的ETF配置类
用于替代复杂的导入依赖
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum


class ETFSourceType(Enum):
    """ETF数据源类型"""
    TUSHARE = "tushare"
    YAHOO = "yahoo"
    EASTMONEY = "eastmoney"


class ETFDownloadType(Enum):
    """ETF下载类型"""
    DAILY = "daily"
    MONEYFLOW = "moneyflow"
    MINUTES = "minutes"
    BASIC_INFO = "basic_info"


class ETFConfig:
    """ETF配置类"""

    def __init__(
        self,
        source: ETFSourceType = ETFSourceType.TUSHARE,
        tushare_token: str = "",
        base_dir: str = "raw/ETF",
        create_subdirs: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        years_back: int = 2,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        request_delay: float = 0.2,
        timeout: int = 30,
        batch_size: int = 50,
        parallel: bool = False,
        download_types: Optional[List[ETFDownloadType]] = None,
        save_format: str = "parquet",
        verbose: bool = True
    ):
        self.source = source
        self.tushare_token = tushare_token
        self.base_dir = base_dir
        self.create_subdirs = create_subdirs
        self.start_date = start_date
        self.end_date = end_date
        self.years_back = years_back
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.timeout = timeout
        self.batch_size = batch_size
        self.parallel = parallel
        self.download_types = download_types or [ETFDownloadType.DAILY]
        self.save_format = save_format
        self.verbose = verbose

    @classmethod
    def from_yaml(cls, config_path: str) -> "ETFConfig":
        """从YAML文件加载配置"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 解析数据源
        source_str = config_data.get('source', 'tushare')
        source = ETFSourceType(source_str)

        # 解析下载类型
        download_types_str = config_data.get('download_types', ['daily'])
        download_types = [ETFDownloadType(dt) for dt in download_types_str]

        return cls(
            source=source,
            tushare_token=config_data.get('tushare_token', ''),
            base_dir=config_data.get('base_dir', 'raw/ETF'),
            create_subdirs=config_data.get('create_subdirs', True),
            start_date=config_data.get('start_date'),
            end_date=config_data.get('end_date'),
            years_back=config_data.get('years_back', 2),
            max_retries=config_data.get('max_retries', 3),
            retry_delay=config_data.get('retry_delay', 1.0),
            request_delay=config_data.get('request_delay', 0.2),
            timeout=config_data.get('timeout', 30),
            batch_size=config_data.get('batch_size', 50),
            parallel=config_data.get('parallel', False),
            download_types=download_types,
            save_format=config_data.get('save_format', 'parquet'),
            verbose=config_data.get('verbose', True)
        )

    def save_yaml(self, config_path: str) -> None:
        """保存配置到YAML文件"""
        config_data = {
            'source': self.source.value,
            'tushare_token': self.tushare_token,
            'base_dir': self.base_dir,
            'create_subdirs': self.create_subdirs,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'years_back': self.years_back,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'request_delay': self.request_delay,
            'timeout': self.timeout,
            'batch_size': self.batch_size,
            'parallel': self.parallel,
            'download_types': [dt.value for dt in self.download_types],
            'save_format': self.save_format,
            'verbose': self.verbose
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False,
                     allow_unicode=True, indent=2)

    def get_etf_list(self) -> List[str]:
        """从配置文件获取ETF列表"""
        # 这里应该从配置文件中读取etf_list
        # 由于配置文件结构较复杂，这里返回空列表
        # 实际使用时，应该使用ETFConfigManager来管理ETF列表
        return []