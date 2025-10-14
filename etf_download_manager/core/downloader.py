#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF下载管理器核心下载器
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

try:
    import tushare as ts
except ImportError:
    ts = None

import pandas as pd

from .models import ETFInfo, DownloadResult, DownloadStats, ETFStatus, ETFPriority
from .config import ETFConfig, ETFDataSource, ETFDownloadType
from .data_manager import ETFDataManager

logger = logging.getLogger(__name__)


class ETFDownloadManager:
    """ETF下载管理器"""

    def __init__(self, config: ETFConfig):
        """
        初始化下载管理器

        Args:
            config: ETF下载配置
        """
        self.config = config
        self.data_manager = ETFDataManager(config)
        self._init_data_source()

    def _init_data_source(self):
        """初始化数据源"""
        if self.config.source == ETFDataSource.TUSHARE:
            if ts is None:
                raise ImportError("需要安装tushare: pip install tushare")

            if not self.config.tushare_token:
                raise ValueError("Tushare Token未设置，请设置config.tushare_token或环境变量TUSHARE_TOKEN")

            self.pro = ts.pro_api(self.config.tushare_token)
            logger.info("Tushare数据源初始化完成")

        else:
            raise NotImplementedError(f"数据源 {self.config.source} 暂未实现")

    def _retry_request(self, func, *args, max_retries: Optional[int] = None, **kwargs):
        """带重试机制的API请求"""
        max_retries = max_retries or self.config.max_retries

        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
                else:
                    logger.warning(f"API返回空结果，尝试 {attempt + 1}/{max_retries}")

            except Exception as e:
                logger.warning(f"API请求失败，尝试 {attempt + 1}/{max_retries}: {e}")

            if attempt < max_retries - 1:
                time.sleep(self.config.retry_delay)

        logger.error(f"API请求最终失败，已重试{max_retries}次")
        return None

    def download_etf_daily_data(self, etf_info: ETFInfo) -> Optional[pd.DataFrame]:
        """
        下载ETF日线数据

        Args:
            etf_info: ETF信息

        Returns:
            日线数据DataFrame
        """
        logger.info(f"开始下载 {etf_info.ts_code} - {etf_info.name} 的日线数据")

        if self.config.source == ETFDataSource.TUSHARE:
            # 使用通用行情接口
            start_date_str = self.config.start_date.replace('-', '')
            end_date_str = self.config.end_date.replace('-', '')

            # 将日期格式转换为YYYY-MM-DD
            start_date_formatted = f"{start_date_str[:4]}-{start_date_str[4:6]}-{start_date_str[6:8]}"
            end_date_formatted = f"{end_date_str[:4]}-{end_date_str[4:6]}-{end_date_str[6:8]}"

            df = self._retry_request(
                ts.pro_bar,
                ts_code=etf_info.ts_code,
                adj='qfq',  # 前复权
                start_date=start_date_formatted,
                end_date=end_date_formatted,
                asset='FD',  # FD表示基金
                freq='D'     # 日线
            )

            if df is not None and not df.empty:
                # 数据预处理
                df = df.sort_values('trade_date')
                df.reset_index(drop=True, inplace=True)
                logger.info(f"成功下载 {len(df)} 条日线数据")
                return df
            else:
                logger.warning(f"ETF {etf_info.ts_code} 日线数据为空")
                return pd.DataFrame()

        return pd.DataFrame()

    def download_etf_moneyflow_data(self, etf_info: ETFInfo) -> Optional[pd.DataFrame]:
        """
        下载ETF资金流向数据（注意：Tushare标准moneyflow接口不包含ETF数据）

        Args:
            etf_info: ETF信息

        Returns:
            资金流向数据DataFrame（通常是空的）
        """
        logger.info(f"尝试下载 {etf_info.ts_code} - {etf_info.name} 的资金流向数据")

        if self.config.source == ETFDataSource.TUSHARE:
            # 注意：Tushare的moneyflow接口不包含ETF数据
            # 这里尝试调用，但通常会返回空结果
            start_date_str = self.config.start_date
            end_date_str = self.config.end_date

            df = self._retry_request(
                self.pro.moneyflow,
                ts_code=etf_info.ts_code,
                start_date=start_date_str,
                end_date=end_date_str
            )

            if df is not None and not df.empty:
                # 数据预处理
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.sort_values('trade_date')

                # 计算衍生指标
                if 'buy_sm_amount' in df.columns and 'buy_md_amount' in df.columns:
                    total_amount = (df['buy_sm_amount'] + df['buy_md_amount'] +
                                  df.get('buy_lg_amount', 0) + df.get('buy_elg_amount', 0))
                    df['net_inflow_rate'] = df['net_mf_amount'] / total_amount * 100

                logger.info(f"成功下载 {len(df)} 条资金流向数据")
                return df
            else:
                logger.warning(f"ETF {etf_info.ts_code} 资金流向数据不可用（Tushare限制）")
                return pd.DataFrame()

        return pd.DataFrame()

    def download_etf_basic_info(self) -> Optional[pd.DataFrame]:
        """
        下载ETF基础信息

        Returns:
            基础信息DataFrame
        """
        logger.info("开始下载ETF基础信息")

        if self.config.source == ETFDataSource.TUSHARE:
            df = self._retry_request(
                self.pro.etf_basic,
                list_status='L',  # L上市 D退市 P待上市
                fields='ts_code,csname,extname,cname,index_code,index_name,setup_date,list_date,list_status,exchange,mgr_name,custod_name,mgt_fee,etf_type'
            )

            if df is not None and not df.empty:
                logger.info(f"成功获取 {len(df)} 只ETF基础信息")
                return df
            else:
                logger.error("获取ETF基础信息失败")
                return pd.DataFrame()

        return pd.DataFrame()

    def download_single_etf(self, etf_info: ETFInfo) -> DownloadResult:
        """
        下载单只ETF的数据

        Args:
            etf_info: ETF信息

        Returns:
            下载结果
        """
        result = DownloadResult(etf_info=etf_info)

        try:
            etf_info.download_status = ETFStatus.DOWNLOADING

            # 下载日线数据
            if ETFDownloadType.DAILY in self.config.download_types:
                daily_df = self.download_etf_daily_data(etf_info)
                if not daily_df.empty:
                    file_path = self.data_manager.save_daily_data(etf_info, daily_df)
                    result.file_paths['daily'] = str(file_path)
                    result.daily_records = len(daily_df)

            # ETF没有资金流向数据，跳过
            logger.info(f"ETF {etf_info.ts_code} 跳过资金流向数据下载（Tushare不提供ETF资金流向数据）")

            # 判断是否成功
            result.success = (result.daily_records > 0) or (result.moneyflow_records > 0)

            if result.success:
                etf_info.download_status = ETFStatus.COMPLETED
                logger.info(f"ETF {etf_info.ts_code} 下载成功")
            else:
                etf_info.download_status = ETFStatus.FAILED
                result.error_message = "未获取到任何数据"

        except Exception as e:
            etf_info.download_status = ETFStatus.FAILED
            result.success = False
            result.error_message = str(e)
            logger.error(f"ETF {etf_info.ts_code} 下载失败: {e}")

        # 添加请求延迟
        time.sleep(self.config.request_delay)

        return result

    def download_multiple_etfs(self, etf_list: List[ETFInfo]) -> DownloadStats:
        """
        批量下载ETF数据

        Args:
            etf_list: ETF列表

        Returns:
            下载统计信息
        """
        stats = DownloadStats(total_etfs=len(etf_list))

        logger.info(f"开始批量下载 {len(etf_list)} 只ETF的数据")

        # 按优先级排序
        priority_order = {
            ETFPriority.CORE: 0,
            ETFPriority.MUST_HAVE: 1,
            ETFPriority.HIGH: 2,
            ETFPriority.MEDIUM: 3,
            ETFPriority.RECOMMENDED: 4,
            ETFPriority.HEDGE: 5,
            ETFPriority.LOW: 6,
            ETFPriority.OPTIONAL: 7
        }
        etf_list.sort(key=lambda x: priority_order.get(x.priority, 999))

        for i, etf_info in enumerate(etf_list, 1):
            logger.info(f"处理第 {i}/{len(etf_list)} 只ETF: {etf_info.ts_code} - {etf_info.name}")

            result = self.download_single_etf(etf_info)
            stats.add_result(result)

            # 显示进度
            progress = i / len(etf_list) * 100
            logger.info(f"进度: {progress:.1f}% ({i}/{len(etf_list)})")

            # 分批处理
            if self.config.batch_size > 0 and i % self.config.batch_size == 0:
                logger.info(f"已完成 {self.config.batch_size} 只ETF的处理")

        stats.finish()
        logger.info(f"批量下载完成，成功: {stats.success_count}, 失败: {stats.failed_count}")

        # 保存下载摘要
        if self.config.create_summary:
            self.data_manager.save_download_summary(stats)

        return stats

    def update_etf_data(self, etf_info: ETFInfo, days_back: int = 30) -> DownloadResult:
        """
        更新单只ETF的最近数据

        Args:
            etf_info: ETF信息
            days_back: 更新最近多少天的数据

        Returns:
            下载结果
        """
        # 临时修改配置的日期范围
        original_start_date = self.config.start_date
        original_end_date = self.config.end_date

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        self.config.start_date = start_date.strftime('%Y%m%d')
        self.config.end_date = end_date.strftime('%Y%m%d')

        try:
            result = self.download_single_etf(etf_info)
            logger.info(f"ETF {etf_info.ts_code} 更新完成")
            return result

        finally:
            # 恢复原始配置
            self.config.start_date = original_start_date
            self.config.end_date = original_end_date

    def validate_downloaded_data(self, etf_list: List[ETFInfo]) -> Dict[str, Dict]:
        """
        验证已下载的数据

        Args:
            etf_list: ETF列表

        Returns:
            验证结果字典
        """
        validation_results = {}

        for etf_info in etf_list:
            validation_result = self.data_manager.validate_data_integrity(etf_info)
            validation_results[etf_info.ts_code] = validation_result

        return validation_results

    def get_download_progress(self, etf_list: List[ETFInfo]) -> Dict[str, Union[int, float, List[str]]]:
        """
        获取下载进度

        Args:
            etf_list: ETF列表

        Returns:
            进度信息字典
        """
        completed_count = 0
        failed_count = 0
        pending_count = 0
        downloading_count = 0

        completed_etfs = []
        failed_etfs = []
        pending_etfs = []

        for etf_info in etf_list:
            if etf_info.download_status == ETFStatus.COMPLETED:
                completed_count += 1
                completed_etfs.append(etf_info.ts_code)
            elif etf_info.download_status == ETFStatus.FAILED:
                failed_count += 1
                failed_etfs.append(etf_info.ts_code)
            elif etf_info.download_status == ETFStatus.DOWNLOADING:
                downloading_count += 1
            else:
                pending_count += 1
                pending_etfs.append(etf_info.ts_code)

        total_count = len(etf_list)
        success_rate = (completed_count / total_count * 100) if total_count > 0 else 0

        return {
            "total_count": total_count,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "pending_count": pending_count,
            "downloading_count": downloading_count,
            "success_rate": round(success_rate, 1),
            "completed_etfs": completed_etfs,
            "failed_etfs": failed_etfs,
            "pending_etfs": pending_etfs
        }