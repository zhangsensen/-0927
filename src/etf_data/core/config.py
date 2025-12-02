#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器配置管理
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ETFExchange, ETFPriority


class ETFDataSource(Enum):
    """ETF数据源枚举"""

    TUSHARE = "tushare"
    YAHOO = "yahoo"
    EASTMONEY = "eastmoney"


class ETFDownloadType(Enum):
    """ETF下载类型枚举"""

    DAILY = "daily"
    MONEYFLOW = "moneyflow"
    MINUTES = "minutes"
    BASIC_INFO = "basic_info"


@dataclass
class ETFConfig:
    """ETF下载配置"""

    # 数据源配置
    source: ETFDataSource = ETFDataSource.TUSHARE
    tushare_token: str = ""

    # 数据目录配置
    base_dir: str = "raw/ETF"
    create_subdirs: bool = True

    # 时间范围配置
    start_date: Optional[str] = None  # 格式: YYYYMMDD
    end_date: Optional[str] = None  # 格式: YYYYMMDD，None表示今天
    years_back: int = 2  # 如果start_date为None，自动计算

    # API请求配置
    max_retries: int = 3
    retry_delay: float = 1.0  # 秒
    request_delay: float = 0.2  # API请求间隔
    timeout: int = 30  # 超时时间

    # 下载配置
    batch_size: int = 50
    parallel: bool = False
    download_types: List[ETFDownloadType] = field(
        default_factory=lambda: [ETFDownloadType.DAILY]
    )

    # ETF筛选配置
    exchanges: List[ETFExchange] = field(
        default_factory=lambda: [ETFExchange.SH, ETFExchange.SZ]
    )
    min_priority: ETFPriority = ETFPriority.OPTIONAL
    include_etfs: List[str] = field(default_factory=list)  # 指定包含的ETF代码
    exclude_etfs: List[str] = field(default_factory=list)  # 排除的ETF代码

    # 输出配置
    save_format: str = "parquet"  # parquet, csv
    create_summary: bool = True
    verbose: bool = True

    def __post_init__(self):
        # 自动计算日期范围
        if not self.start_date:
            end_date = (
                datetime.now()
                if not self.end_date
                else datetime.strptime(self.end_date, "%Y%m%d")
            )
            start_date = end_date - timedelta(days=self.years_back * 365)
            self.start_date = start_date.strftime("%Y%m%d")

        if not self.end_date:
            self.end_date = datetime.now().strftime("%Y%m%d")

        # 从环境变量获取Token
        if not self.tushare_token and self.source == ETFDataSource.TUSHARE:
            self.tushare_token = os.getenv("TUSHARE_TOKEN", "")

    @property
    def data_dir(self) -> Path:
        """数据目录路径"""
        return Path(self.base_dir)

    @property
    def daily_dir(self) -> Path:
        """日线数据目录"""
        return self.data_dir / "daily"

    @property
    def moneyflow_dir(self) -> Path:
        """资金流向数据目录"""
        return self.data_dir / "moneyflow"

    @property
    def minutes_dir(self) -> Path:
        """分钟数据目录"""
        return self.data_dir / "minutes"

    @property
    def basic_dir(self) -> Path:
        """基础信息目录"""
        return self.data_dir / "basic"

    @property
    def summary_dir(self) -> Path:
        """摘要目录"""
        return self.data_dir / "summary"

    def create_directories(self):
        """创建数据目录"""
        if self.create_subdirs:
            self.data_dir.mkdir(parents=True, exist_ok=True)

            if ETFDownloadType.DAILY in self.download_types:
                self.daily_dir.mkdir(exist_ok=True)

            if ETFDownloadType.MONEYFLOW in self.download_types:
                self.moneyflow_dir.mkdir(exist_ok=True)

            if ETFDownloadType.MINUTES in self.download_types:
                self.minutes_dir.mkdir(exist_ok=True)

            if ETFDownloadType.BASIC_INFO in self.download_types:
                self.basic_dir.mkdir(exist_ok=True)

            if self.create_summary:
                self.summary_dir.mkdir(exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ETFConfig":
        """从字典创建配置"""
        # 处理枚举类型
        if "source" in config_dict:
            config_dict["source"] = ETFDataSource(config_dict["source"])

        if "exchanges" in config_dict:
            config_dict["exchanges"] = [
                ETFExchange(ex) for ex in config_dict["exchanges"]
            ]

        if "min_priority" in config_dict:
            config_dict["min_priority"] = ETFPriority(config_dict["min_priority"])

        if "download_types" in config_dict:
            config_dict["download_types"] = [
                ETFDownloadType(dt) for dt in config_dict["download_types"]
            ]

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "ETFConfig":
        """从YAML文件创建配置"""
        import yaml

        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 处理环境变量替换
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj

        config_dict = replace_env_vars(config_dict)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, list) and value and isinstance(value[0], Enum):
                result[key] = [item.value for item in value]
            elif isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def save_yaml(self, yaml_file: str):
        """保存为YAML文件"""
        import yaml

        config_dict = self.to_dict()

        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict, f, default_flow_style=False, allow_unicode=True, indent=2
            )


@dataclass
class ETFListConfig:
    """ETF清单配置"""

    existing_etfs: List[Dict[str, Any]] = field(default_factory=list)
    new_etfs: List[Dict[str, Any]] = field(default_factory=list)
    optional_etfs: List[Dict[str, Any]] = field(default_factory=list)

    def get_all_etfs(self) -> List[Dict[str, Any]]:
        """获取所有ETF"""
        return self.existing_etfs + self.new_etfs + self.optional_etfs

    def get_must_have_etfs(self) -> List[Dict[str, Any]]:
        """获取必须拥有的ETF"""
        all_etfs = self.get_all_etfs()
        return [etf for etf in all_etfs if etf.get("priority") in ["core", "must_have"]]

    def get_high_priority_etfs(self) -> List[Dict[str, Any]]:
        """获取高优先级ETF"""
        all_etfs = self.get_all_etfs()
        return [
            etf
            for etf in all_etfs
            if etf.get("priority") in ["core", "must_have", "high"]
        ]

    @classmethod
    def from_python_file(cls, python_file: str) -> "ETFListConfig":
        """从Python文件加载ETF清单"""
        import importlib.util

        spec = importlib.util.spec_from_file_location("etf_list", python_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return cls(
            existing_etfs=getattr(module, "EXISTING_ETFS", []),
            new_etfs=getattr(module, "NEW_ETFS", []),
            optional_etfs=getattr(module, "OPTIONAL_ETFS", []),
        )
