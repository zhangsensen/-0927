#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF数据管理器
"""

import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .models import ETFInfo, DownloadResult, DownloadStats
from .config import ETFConfig

logger = logging.getLogger(__name__)


class ETFDataManager:
    """ETF数据管理器"""

    def __init__(self, config: ETFConfig):
        """
        初始化数据管理器

        Args:
            config: ETF下载配置
        """
        self.config = config
        self.config.create_directories()

    def save_daily_data(self, etf_info: ETFInfo, df: pd.DataFrame) -> Path:
        """
        保存日线数据

        Args:
            etf_info: ETF信息
            df: 日线数据DataFrame

        Returns:
            保存的文件路径
        """
        if df.empty:
            raise ValueError("日线数据为空")

        # 文件命名格式: {SYMBOL}_daily_{START_DATE}_{END_DATE}.parquet
        symbol = etf_info.symbol
        start_date = self.config.start_date
        end_date = self.config.end_date

        if self.config.save_format == "parquet":
            file_path = self.config.daily_dir / f"{symbol}_daily_{start_date}_{end_date}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            file_path = self.config.daily_dir / f"{symbol}_daily_{start_date}_{end_date}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

        logger.info(f"日线数据已保存: {file_path}")
        return file_path

    def save_moneyflow_data(self, etf_info: ETFInfo, df: pd.DataFrame) -> Path:
        """
        保存资金流向数据

        Args:
            etf_info: ETF信息
            df: 资金流向数据DataFrame

        Returns:
            保存的文件路径
        """
        if df.empty:
            raise ValueError("资金流向数据为空")

        # 文件命名格式: {SYMBOL}_moneyflow_{START_DATE}_{END_DATE}.parquet
        symbol = etf_info.symbol
        start_date = self.config.start_date
        end_date = self.config.end_date

        if self.config.save_format == "parquet":
            file_path = self.config.moneyflow_dir / f"{symbol}_moneyflow_{start_date}_{end_date}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            file_path = self.config.moneyflow_dir / f"{symbol}_moneyflow_{start_date}_{end_date}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

        logger.info(f"资金流向数据已保存: {file_path}")
        return file_path

    def save_minutes_data(self, etf_info: ETFInfo, df: pd.DataFrame, trade_date: str, freq: str = "1min") -> Path:
        """
        保存分钟数据

        Args:
            etf_info: ETF信息
            df: 分钟数据DataFrame
            trade_date: 交易日期
            freq: 分钟频度

        Returns:
            保存的文件路径
        """
        if df.empty:
            raise ValueError("分钟数据为空")

        # 创建ETF专属的分钟数据目录
        symbol = etf_info.symbol
        symbol_minutes_dir = self.config.minutes_dir / symbol
        symbol_minutes_dir.mkdir(exist_ok=True)

        # 文件命名格式: {SYMBOL}_{TRADE_DATE}_{FREQ}.parquet
        if self.config.save_format == "parquet":
            file_path = symbol_minutes_dir / f"{symbol}_{trade_date}_{freq}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            file_path = symbol_minutes_dir / f"{symbol}_{trade_date}_{freq}.csv"
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

        logger.info(f"分钟数据已保存: {file_path}")
        return file_path

    def save_basic_info(self, df: pd.DataFrame) -> Path:
        """
        保存ETF基础信息

        Args:
            df: 基础信息DataFrame

        Returns:
            保存的文件路径
        """
        if df.empty:
            raise ValueError("基础信息为空")

        # 保存基础信息
        today_str = datetime.now().strftime('%Y%m%d')

        if self.config.save_format == "parquet":
            # 保存带日期的版本
            dated_file = self.config.basic_dir / f"etf_basic_info_{today_str}.parquet"
            df.to_parquet(dated_file, index=False)

            # 保存最新版本
            latest_file = self.config.basic_dir / "etf_basic_latest.parquet"
            df.to_parquet(latest_file, index=False)

            file_path = latest_file
        else:
            # 保存带日期的版本
            dated_file = self.config.basic_dir / f"etf_basic_info_{today_str}.csv"
            df.to_csv(dated_file, index=False, encoding='utf-8-sig')

            # 保存最新版本
            latest_file = self.config.basic_dir / "etf_basic_latest.csv"
            df.to_csv(latest_file, index=False, encoding='utf-8-sig')

            file_path = latest_file

        logger.info(f"ETF基础信息已保存: {file_path}")
        return file_path

    def save_download_summary(self, stats: DownloadStats) -> Path:
        """
        保存下载摘要

        Args:
            stats: 下载统计信息

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.config.summary_dir / f"download_summary_{timestamp}.json"

        summary_data = stats.get_summary()

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        logger.info(f"下载摘要已保存: {summary_file}")
        return summary_file

    def load_daily_data(self, etf_info: ETFInfo) -> Optional[pd.DataFrame]:
        """
        加载日线数据

        Args:
            etf_info: ETF信息

        Returns:
            日线数据DataFrame，如果文件不存在返回None
        """
        symbol = etf_info.symbol
        start_date = self.config.start_date
        end_date = self.config.end_date

        if self.config.save_format == "parquet":
            file_path = self.config.daily_dir / f"{symbol}_daily_{start_date}_{end_date}.parquet"
            if file_path.exists():
                return pd.read_parquet(file_path)
        else:
            file_path = self.config.daily_dir / f"{symbol}_daily_{start_date}_{end_date}.csv"
            if file_path.exists():
                return pd.read_csv(file_path, encoding='utf-8-sig')

        return None

    def load_moneyflow_data(self, etf_info: ETFInfo) -> Optional[pd.DataFrame]:
        """
        加载资金流向数据

        Args:
            etf_info: ETF信息

        Returns:
            资金流向数据DataFrame，如果文件不存在返回None
        """
        symbol = etf_info.symbol
        start_date = self.config.start_date
        end_date = self.config.end_date

        if self.config.save_format == "parquet":
            file_path = self.config.moneyflow_dir / f"{symbol}_moneyflow_{start_date}_{end_date}.parquet"
            if file_path.exists():
                return pd.read_parquet(file_path)
        else:
            file_path = self.config.moneyflow_dir / f"{symbol}_moneyflow_{start_date}_{end_date}.csv"
            if file_path.exists():
                return pd.read_csv(file_path, encoding='utf-8-sig')

        return None

    def load_basic_info(self, latest: bool = True) -> Optional[pd.DataFrame]:
        """
        加载ETF基础信息

        Args:
            latest: 是否加载最新版本

        Returns:
            基础信息DataFrame，如果文件不存在返回None
        """
        if self.config.save_format == "parquet":
            if latest:
                file_path = self.config.basic_dir / "etf_basic_latest.parquet"
            else:
                # 查找最新的带日期文件
                files = list(self.config.basic_dir.glob("etf_basic_info_*.parquet"))
                if files:
                    file_path = max(files)
                else:
                    file_path = self.config.basic_dir / "etf_basic_latest.parquet"

            if file_path.exists():
                return pd.read_parquet(file_path)
        else:
            if latest:
                file_path = self.config.basic_dir / "etf_basic_latest.csv"
            else:
                # 查找最新的带日期文件
                files = list(self.config.basic_dir.glob("etf_basic_info_*.csv"))
                if files:
                    file_path = max(files)
                else:
                    file_path = self.config.basic_dir / "etf_basic_latest.csv"

            if file_path.exists():
                return pd.read_csv(file_path, encoding='utf-8-sig')

        return None

    def validate_data_integrity(self, etf_info: ETFInfo) -> Dict[str, Union[bool, str, int]]:
        """
        验证数据完整性

        Args:
            etf_info: ETF信息

        Returns:
            验证结果字典
        """
        result = {
            "etf_code": etf_info.ts_code,
            "etf_name": etf_info.name,
            "daily_exists": False,
            "daily_records": 0,
            "daily_valid": False,
            "moneyflow_exists": False,
            "moneyflow_records": 0,
            "moneyflow_valid": False,
            "overall_valid": False
        }

        # 验证日线数据
        daily_df = self.load_daily_data(etf_info)
        if daily_df is not None and not daily_df.empty:
            result["daily_exists"] = True
            result["daily_records"] = len(daily_df)

            # 检查必要的列
            required_columns = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
            if all(col in daily_df.columns for col in required_columns):
                # 检查是否有缺失值
                missing_data = daily_df[required_columns].isnull().sum().sum()
                result["daily_valid"] = missing_data == 0

        # 验证资金流向数据
        moneyflow_df = self.load_moneyflow_data(etf_info)
        if moneyflow_df is not None and not moneyflow_df.empty:
            result["moneyflow_exists"] = True
            result["moneyflow_records"] = len(moneyflow_df)

            # 检查必要的列
            required_columns = ['trade_date', 'net_mf_amount']
            if all(col in moneyflow_df.columns for col in required_columns):
                # 检查是否有缺失值
                missing_data = moneyflow_df[required_columns].isnull().sum().sum()
                result["moneyflow_valid"] = missing_data == 0

        # 整体有效性
        result["overall_valid"] = result["daily_valid"]

        return result

    def get_data_summary(self) -> Dict[str, Union[int, List[str]]]:
        """
        获取数据摘要统计

        Returns:
            数据摘要字典
        """
        summary = {
            "daily_files": 0,
            "moneyflow_files": 0,
            "basic_files": 0,
            "daily_etfs": [],
            "moneyflow_etfs": [],
            "total_size_mb": 0
        }

        # 统计日线数据文件
        daily_files = list(self.config.daily_dir.glob(f"*.{self.config.save_format}"))
        summary["daily_files"] = len(daily_files)
        summary["daily_etfs"] = [f.stem.split('_')[0] for f in daily_files]

        # 统计资金流向数据文件
        moneyflow_files = list(self.config.moneyflow_dir.glob(f"*.{self.config.save_format}"))
        summary["moneyflow_files"] = len(moneyflow_files)
        summary["moneyflow_etfs"] = [f.stem.split('_')[0] for f in moneyflow_files]

        # 统计基础信息文件
        basic_files = list(self.config.basic_dir.glob(f"*.{self.config.save_format}"))
        summary["basic_files"] = len(basic_files)

        # 计算总大小（简化计算）
        all_files = daily_files + moneyflow_files + basic_files
        total_size = sum(f.stat().st_size for f in all_files if f.exists())
        summary["total_size_mb"] = round(total_size / (1024 * 1024), 2)

        return summary