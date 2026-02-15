#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ETFPriority(Enum):
    """ETF优先级枚举"""

    CORE = "core"
    MUST_HAVE = "must_have"
    HIGH = "high"
    MEDIUM = "medium"
    RECOMMENDED = "recommended"
    HEDGE = "hedge"
    LOW = "low"
    OPTIONAL = "optional"


class ETFExchange(Enum):
    """ETF交易所枚举"""

    SH = "SH"  # 上交所
    SZ = "SZ"  # 深交所


class ETFStatus(Enum):
    """ETF状态枚举"""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


class ETFDownloadType(Enum):
    """ETF下载类型枚举"""

    DAILY = "daily"
    MONEYFLOW = "moneyflow"
    MINUTES = "minutes"
    BASIC = "basic"


@dataclass
class ETFInfo:
    """ETF信息模型"""

    code: str
    name: str
    ts_code: str
    category: str = ""
    subcategory: str = ""
    priority: ETFPriority = ETFPriority.OPTIONAL
    exchange: ETFExchange = ETFExchange.SH
    daily_volume: str = ""
    description: str = ""
    file_exists: bool = False
    download_status: ETFStatus = ETFStatus.PENDING
    note: str = ""

    def __post_init__(self):
        # 自动生成ts_code
        if not self.ts_code and self.code:
            if (
                self.code.startswith("5")
                or self.code.startswith("58")
                or self.code.startswith("51")
            ):
                self.ts_code = f"{self.code}.SH"
            else:
                self.ts_code = f"{self.code}.SZ"

    @property
    def symbol(self) -> str:
        """获取交易代码（不带交易所后缀）"""
        return self.ts_code.split(".")[0]

    @property
    def is_high_priority(self) -> bool:
        """是否为高优先级ETF"""
        return self.priority in [
            ETFPriority.CORE,
            ETFPriority.MUST_HAVE,
            ETFPriority.HIGH,
        ]


@dataclass
class DownloadResult:
    """下载结果模型"""

    etf_info: ETFInfo
    success: bool = False
    daily_records: int = 0
    moneyflow_records: int = 0
    error_message: str = ""
    file_paths: Dict[str, str] = field(default_factory=dict)
    download_time: datetime = field(default_factory=datetime.now)

    @property
    def has_daily_data(self) -> bool:
        """是否有日线数据"""
        return self.daily_records > 0

    @property
    def has_moneyflow_data(self) -> bool:
        """是否有资金流向数据"""
        return self.moneyflow_records > 0

    @property
    def total_records(self) -> int:
        """总记录数"""
        return self.daily_records + self.moneyflow_records


@dataclass
class DownloadStats:
    """下载统计模型"""

    total_etfs: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_daily_records: int = 0
    total_moneyflow_records: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    failed_etfs: List[str] = field(default_factory=list)
    success_etfs: List[str] = field(default_factory=list)
    download_details: List[DownloadResult] = field(default_factory=list)

    @property
    def duration(self) -> Optional[str]:
        """下载耗时"""
        if self.end_time and self.start_time:
            return str(self.end_time - self.start_time)
        return None

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_etfs == 0:
            return 0.0
        return (self.success_count / self.total_etfs) * 100

    def add_result(self, result: DownloadResult):
        """添加下载结果"""
        self.download_details.append(result)

        if result.success:
            self.success_count += 1
            self.success_etfs.append(result.etf_info.ts_code)
            self.total_daily_records += result.daily_records
            self.total_moneyflow_records += result.moneyflow_records
        else:
            self.failed_count += 1
            self.failed_etfs.append(result.etf_info.ts_code)

    def finish(self):
        """标记下载完成"""
        self.end_time = datetime.now()

    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            "total_etfs": self.total_etfs,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "success_rate": f"{self.success_rate:.1f}%",
            "total_records": self.total_daily_records + self.total_moneyflow_records,
            "duration": self.duration,
            "failed_etfs": self.failed_etfs[:10],  # 只显示前10个失败的ETF
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }
