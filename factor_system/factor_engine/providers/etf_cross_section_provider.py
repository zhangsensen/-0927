#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面数据管理器
统一43只ETF数据接口，为横截面分析提供标准化数据
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from factor_system.utils import get_raw_data_dir

logger = logging.getLogger(__name__)


class ETFCrossSectionDataManager:
    """ETF横截面数据管理器"""

    def __init__(self, etf_data_dir: Optional[str] = None):
        """
        初始化ETF横截面数据管理器

        Args:
            etf_data_dir: ETF数据目录，默认使用 raw/ETF/daily
        """
        if etf_data_dir is None:
            self.etf_data_dir = get_raw_data_dir() / "ETF" / "daily"
        else:
            self.etf_data_dir = Path(etf_data_dir)

        self.etf_list: List[str] = []
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self._load_etf_list()

    def _load_etf_list(self):
        """加载所有可用的ETF列表"""
        if not self.etf_data_dir.exists():
            raise FileNotFoundError(f"ETF数据目录不存在: {self.etf_data_dir}")

        # 扫描parquet文件获取ETF列表
        etf_files = list(self.etf_data_dir.glob("*.parquet"))
        self.etf_list = [f.stem.split('_daily_')[0] for f in etf_files]

        logger.info(f"加载ETF列表: {len(self.etf_list)}只ETF")

    def get_etf_list(self) -> List[str]:
        """获取所有ETF代码列表"""
        return self.etf_list.copy()

    def load_single_etf(self, etf_code: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        加载单只ETF数据

        Args:
            etf_code: ETF代码
            use_cache: 是否使用缓存

        Returns:
            ETF数据DataFrame
        """
        if use_cache and etf_code in self.data_cache:
            return self.data_cache[etf_code].copy()

        # 查找ETF数据文件
        etf_files = list(self.etf_data_dir.glob(f"{etf_code}_daily_*.parquet"))
        if not etf_files:
            logger.warning(f"未找到ETF {etf_code} 的数据文件")
            return None

        if len(etf_files) > 1:
            logger.warning(f"ETF {etf_code} 有多个数据文件，使用第一个")

        try:
            df = pd.read_parquet(etf_files[0])

            # 数据预处理
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df = df.sort_values('trade_date').reset_index(drop=True)

            # 基础数据验证
            required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必需列: {missing_cols}")

            if use_cache:
                self.data_cache[etf_code] = df.copy()

            logger.info(f"加载ETF {etf_code}: {len(df)} 条记录 ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
            return df

        except Exception as e:
            logger.error(f"加载ETF {etf_code} 数据失败: {e}")
            return None

    def load_multiple_etfs(self, etf_codes: List[str], use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载多只ETF数据

        Args:
            etf_codes: ETF代码列表
            use_cache: 是否使用缓存

        Returns:
            ETF数据字典 {etf_code: DataFrame}
        """
        result = {}

        for etf_code in etf_codes:
            df = self.load_single_etf(etf_code, use_cache)
            if df is not None:
                result[etf_code] = df

        logger.info(f"加载ETF数据: {len(result)}/{len(etf_codes)} 只成功")
        return result

    def load_all_etfs(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        加载所有ETF数据

        Args:
            use_cache: 是否使用缓存

        Returns:
            所有ETF数据字典
        """
        return self.load_multiple_etfs(self.etf_list, use_cache)

    def get_cross_section_data(self, date: Union[str, datetime]) -> Optional[pd.DataFrame]:
        """
        获取指定日期的横截面数据

        Args:
            date: 查询日期

        Returns:
            横截面数据DataFrame
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        cross_section_data = []

        for etf_code in self.etf_list:
            df = self.load_single_etf(etf_code)
            if df is None:
                continue

            # 查找指定日期或之前最近的数据
            valid_data = df[df['trade_date'] <= date]
            if valid_data.empty:
                continue

            latest_data = valid_data.iloc[-1]

            # 构建横截面数据
            cross_section_row = {
                'etf_code': etf_code,
                'ts_code': latest_data['ts_code'],
                'date': date,
                'close': latest_data['close'],
                'volume': latest_data['vol'],
                'amount': latest_data['amount'],
                'pct_chg': latest_data.get('pct_chg', 0),
                'data_date': latest_data['trade_date']
            }

            cross_section_data.append(cross_section_row)

        if not cross_section_data:
            logger.warning(f"日期 {date} 没有可用的ETF数据")
            return None

        cross_section_df = pd.DataFrame(cross_section_data)
        cross_section_df = cross_section_df.sort_values('etf_code').reset_index(drop=True)

        logger.info(f"横截面数据 {date}: {len(cross_section_df)} 只ETF")
        return cross_section_df

    def get_time_series_data(self,
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime],
                           etf_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取时间序列数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            etf_codes: ETF代码列表，None表示所有ETF

        Returns:
            时间序列数据DataFrame
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if etf_codes is None:
            etf_codes = self.etf_list

        time_series_data = []

        for etf_code in etf_codes:
            df = self.load_single_etf(etf_code)
            if df is None:
                continue

            # 筛选日期范围
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            filtered_df = df[mask].copy()

            if filtered_df.empty:
                continue

            filtered_df['etf_code'] = etf_code
            time_series_data.append(filtered_df)

        if not time_series_data:
            logger.warning(f"日期范围 {start_date} ~ {end_date} 没有可用的ETF数据")
            return pd.DataFrame()

        result_df = pd.concat(time_series_data, ignore_index=True)
        result_df = result_df.sort_values(['etf_code', 'trade_date']).reset_index(drop=True)

        logger.info(f"时间序列数据 {start_date} ~ {end_date}: {len(result_df)} 条记录，{result_df['etf_code'].nunique()} 只ETF")
        return result_df

    def get_etf_universe(self, min_trading_days: int = 252) -> List[str]:
        """
        获取ETF投资域（过滤掉数据不足的ETF）

        Args:
            min_trading_days: 最少交易日数

        Returns:
            符合条件的ETF代码列表
        """
        eligible_etfs = []

        for etf_code in self.etf_list:
            df = self.load_single_etf(etf_code)
            if df is not None and len(df) >= min_trading_days:
                eligible_etfs.append(etf_code)

        logger.info(f"ETF投资域: {len(eligible_etfs)}/{len(self.etf_list)} 只符合条件")
        return eligible_etfs

    def clear_cache(self):
        """清空数据缓存"""
        self.data_cache.clear()
        logger.info("数据缓存已清空")

    def get_data_summary(self) -> Dict:
        """获取数据摘要信息"""
        summary = {
            'total_etfs': len(self.etf_list),
            'cache_size': len(self.data_cache),
            'data_directory': str(self.etf_data_dir),
            'etf_list': self.etf_list
        }

        # 获取数据时间范围
        all_start_dates = []
        all_end_dates = []

        for etf_code in self.etf_list[:5]:  # 只检查前5只，节省时间
            df = self.load_single_etf(etf_code, use_cache=True)
            if df is not None and not df.empty:
                all_start_dates.append(df['trade_date'].min())
                all_end_dates.append(df['trade_date'].max())

        if all_start_dates:
            summary['date_range'] = {
                'start': min(all_start_dates),
                'end': max(all_end_dates)
            }

        return summary


# 便捷函数
def get_etf_cross_section_manager() -> ETFCrossSectionDataManager:
    """获取ETF横截面数据管理器实例"""
    return ETFCrossSectionDataManager()


def load_all_etf_data() -> Dict[str, pd.DataFrame]:
    """加载所有ETF数据的便捷函数"""
    manager = ETFCrossSectionDataManager()
    return manager.load_all_etfs()


def get_etf_cross_section(date: Union[str, datetime]) -> Optional[pd.DataFrame]:
    """获取指定日期横截面数据的便捷函数"""
    manager = ETFCrossSectionDataManager()
    return manager.get_cross_section_data(date)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    manager = ETFCrossSectionDataManager()

    # 测试基础功能
    print("=== ETF横截面数据管理器测试 ===")

    # 数据摘要
    summary = manager.get_data_summary()
    print(f"数据摘要: {summary}")

    # 测试横截面数据
    test_date = "2025-10-14"
    cross_section = manager.get_cross_section_data(test_date)
    if cross_section is not None:
        print(f"横截面数据示例 ({test_date}):")
        print(cross_section.head())

    # 测试时间序列数据
    start_date = "2025-10-01"
    end_date = "2025-10-14"
    time_series = manager.get_time_series_data(start_date, end_date, ['159801.SZ', '510300.SH'])
    if not time_series.empty:
        print(f"时间序列数据示例 ({start_date} ~ {end_date}):")
        print(time_series.head())