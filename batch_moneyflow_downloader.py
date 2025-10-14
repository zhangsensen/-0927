#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模股票资金流数据批量下载脚本
支持5600+只股票的资金流数据下载，具有断点续传和进度监控功能
"""

import os
import time
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import traceback
import signal
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/zhangshenshen/深度量化0927/moneyflow_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchMoneyFlowDownloader:
    def __init__(self, token=None):
        """
        初始化大规模下载器

        Parameters:
        token (str): Tushare Pro API token
        """
        if token:
            ts.set_token(token)
        else:
            # 尝试从环境变量获取
            token = os.getenv('TUSHARE_TOKEN')
            if token:
                ts.set_token(token)
            else:
                # 使用默认token
                token = "4a24bcfff16f7593632e6c46976a83e6a26f8f565daa156cb9ea9c1f"
                logger.info("使用默认Tushare token")
                ts.set_token(token)

        self.pro = ts.pro_api()
        self.source_dir = "/Volumes/Share_Data/data/output/1d"
        self.output_dir = "/Volumes/Share_Data/data/output/money_flow"
        self.progress_file = "/Users/zhangshenshen/深度量化0927/moneyflow_progress.json"
        self.error_file = "/Users/zhangshenshen/深度量化0927/moneyflow_errors.json"

        # 下载参数
        self.start_date = "20240823"
        self.end_date = "20251013"
        self.delay = 1.5  # 请求间隔1.5秒（更保守）
        self.max_retries = 3
        self.batch_size = 100  # 每批处理的股票数量

        # 状态跟踪
        self.processed_stocks = set()
        self.failed_stocks = set()
        self.start_time = None

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号，保存进度"""
        logger.info(f"接收到中断信号 {signum}，正在保存进度...")
        self.save_progress()
        sys.exit(0)

    def get_stock_list(self):
        """
        获取源目录下的所有股票代码
        """
        stock_files = []
        for file in os.listdir(self.source_dir):
            if file.endswith('.parquet'):
                stock_code = file.replace('.parquet', '')
                stock_files.append(stock_code)

        logger.info(f"发现 {len(stock_files)} 只股票文件")
        return sorted(stock_files)

    def load_progress(self):
        """
        加载之前的下载进度
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    self.processed_stocks = set(progress.get('processed_stocks', []))
                    self.failed_stocks = set(progress.get('failed_stocks', []))
                logger.info(f"加载进度: 已处理 {len(self.processed_stocks)} 只股票, 失败 {len(self.failed_stocks)} 只")
            except Exception as e:
                logger.warning(f"加载进度文件失败: {e}")

    def save_progress(self):
        """
        保存下载进度
        """
        try:
            progress = {
                'processed_stocks': list(self.processed_stocks),
                'failed_stocks': list(self.failed_stocks),
                'timestamp': datetime.now().isoformat(),
                'total_processed': len(self.processed_stocks),
                'total_failed': len(self.failed_stocks)
            }
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")

    def log_error(self, stock_code, error_msg):
        """
        记录错误信息
        """
        error_info = {
            'stock_code': stock_code,
            'error': error_msg,
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc()
        }

        errors = []
        if os.path.exists(self.error_file):
            try:
                with open(self.error_file, 'r') as f:
                    errors = json.load(f)
            except Exception:
                errors = []

        errors.append(error_info)

        try:
            with open(self.error_file, 'w') as f:
                json.dump(errors, f, indent=2)
        except Exception as e:
            logger.error(f"保存错误日志失败: {e}")

    def is_already_downloaded(self, stock_code):
        """
        检查股票是否已经下载过且数据完整
        """
        output_file = os.path.join(self.output_dir, f"{stock_code}_moneyflow.parquet")
        if not os.path.exists(output_file):
            return False

        try:
            df = pd.read_parquet(output_file)
            if df.empty:
                return False

            # 检查日期范围是否符合要求
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            min_date = df['trade_date'].min().strftime('%Y%m%d')
            max_date = df['trade_date'].max().strftime('%Y%m%d')

            # 检查日期范围是否覆盖我们的需求
            if min_date <= self.start_date and max_date >= self.end_date:
                return True
            else:
                logger.info(f"{stock_code} 数据日期范围不符: {min_date} ~ {max_date}, 需要重新下载")
                return False

        except Exception as e:
            logger.warning(f"检查 {stock_code} 已下载数据时出错: {e}")
            return False

    def download_moneyflow(self, stock_code, max_retries=3):
        """
        下载单只股票的资金流向数据

        Parameters:
        stock_code (str): 股票代码
        max_retries (int): 最大重试次数

        Returns:
        pd.DataFrame or None: 资金流向数据
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"下载 {stock_code} 资金流向数据 (尝试 {attempt+1}/{max_retries})")

                # 调用资金流向接口
                df = self.pro.moneyflow(
                    ts_code=stock_code,
                    start_date=self.start_date,
                    end_date=self.end_date
                )

                if df is not None and not df.empty:
                    logger.debug(f"成功下载 {stock_code} 资金流向数据: {len(df)} 条记录")
                    return df
                else:
                    logger.warning(f"{stock_code} 没有资金流向数据")
                    return None

            except Exception as e:
                error_msg = f"下载 {stock_code} 资金流向数据失败 (尝试 {attempt+1}): {e}"
                logger.error(error_msg)

                if attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = (2 ** attempt) * 2
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        error_msg = f"下载 {stock_code} 资金流向数据最终失败"
        logger.error(error_msg)
        self.log_error(stock_code, error_msg)
        return None

    def save_moneyflow(self, stock_code, df):
        """
        保存资金流向数据

        Parameters:
        stock_code (str): 股票代码
        df (pd.DataFrame): 资金流向数据
        """
        if df is None or df.empty:
            logger.warning(f"{stock_code} 没有数据可保存")
            return False

        output_file = os.path.join(self.output_dir, f"{stock_code}_moneyflow.parquet")

        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 保存为parquet格式
            df.to_parquet(output_file, index=False)
            logger.debug(f"保存 {stock_code} 资金流向数据到: {output_file}")
            return True

        except Exception as e:
            error_msg = f"保存 {stock_code} 资金流向数据失败: {e}"
            logger.error(error_msg)
            self.log_error(stock_code, error_msg)
            return False

    def print_statistics(self):
        """
        打印下载统计信息
        """
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            rate = len(self.processed_stocks) / elapsed.total_seconds() * 60 if elapsed.total_seconds() > 0 else 0

            logger.info("=== 下载统计 ===")
            logger.info(f"总耗时: {elapsed}")
            logger.info(f"已处理: {len(self.processed_stocks)} 只")
            logger.info(f"失败: {len(self.failed_stocks)} 只")
            logger.info(f"处理速度: {rate:.1f} 只/分钟")

            if len(self.processed_stocks) > 0:
                success_rate = (len(self.processed_stocks) - len(self.failed_stocks)) / len(self.processed_stocks) * 100
                logger.info(f"成功率: {success_rate:.1f}%")

    def download_batch(self, stock_list, batch_start=0):
        """
        批量下载股票的资金流向数据

        Parameters:
        stock_list (list): 股票代码列表
        batch_start (int): 批次起始位置
        """
        logger.info(f"开始下载第 {batch_start//self.batch_size + 1} 批次股票 (共 {len(stock_list)} 只)")

        for i, stock_code in enumerate(stock_list, batch_start + 1):
            # 检查是否已经处理过
            if stock_code in self.processed_stocks:
                logger.debug(f"跳过已处理的股票: {stock_code}")
                continue

            try:
                # 检查是否已经下载过且数据完整
                if self.is_already_downloaded(stock_code):
                    logger.info(f"跳过已下载的股票: {stock_code}")
                    self.processed_stocks.add(stock_code)
                    self.save_progress()
                    continue

                logger.info(f"处理进度: {i}/{len(stock_list)} - {stock_code}")

                # 下载数据
                df = self.download_moneyflow(stock_code, self.max_retries)

                # 保存数据
                success = self.save_moneyflow(stock_code, df)

                if success:
                    self.processed_stocks.add(stock_code)
                else:
                    self.failed_stocks.add(stock_code)

                # 每处理10只股票保存一次进度
                if i % 10 == 0:
                    self.save_progress()
                    self.print_statistics()

                # 请求间隔
                if self.delay > 0:
                    time.sleep(self.delay)

            except KeyboardInterrupt:
                logger.info("用户中断下载")
                break
            except Exception as e:
                error_msg = f"处理 {stock_code} 时发生错误: {e}"
                logger.error(error_msg)
                self.log_error(stock_code, error_msg)
                self.failed_stocks.add(stock_code)
                continue

        # 保存最终进度
        self.save_progress()
        self.print_statistics()

    def download_all(self):
        """
        下载所有股票的资金流向数据（分批处理）
        """
        # 加载之前的进度
        self.load_progress()

        # 获取股票列表
        stock_list = self.get_stock_list()

        # 过滤已处理的股票
        remaining_stocks = [stock for stock in stock_list if stock not in self.processed_stocks]

        logger.info(f"总股票数: {len(stock_list)}")
        logger.info(f"已处理: {len(self.processed_stocks)}")
        logger.info(f"待处理: {len(remaining_stocks)}")

        if not remaining_stocks:
            logger.info("所有股票已处理完成！")
            return

        self.start_time = datetime.now()

        # 分批处理
        for batch_start in range(0, len(remaining_stocks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(remaining_stocks))
            batch_stocks = remaining_stocks[batch_start:batch_end]

            logger.info(f"处理批次 {batch_start//self.batch_size + 1}: 股票 {batch_start+1}-{batch_end}")

            try:
                self.download_batch(batch_stocks, batch_start)

                # 批次间稍作休息
                if batch_end < len(remaining_stocks):
                    logger.info(f"批次完成，休息 30 秒...")
                    time.sleep(30)

            except Exception as e:
                logger.error(f"批次 {batch_start//self.batch_size + 1} 处理失败: {e}")
                continue

        logger.info("=== 大规模下载完成 ===")
        self.print_statistics()


def main():
    """
    主函数
    """
    logger.info("=== 大规模股票资金流数据下载开始 ===")
    logger.info(f"时间范围: 2024-08-23 ~ 2025-10-13")
    logger.info(f"存储目录: /Volumes/Share_Data/data/output/money_flow")

    try:
        # 初始化下载器
        downloader = BatchMoneyFlowDownloader()

        # 开始批量下载
        downloader.download_all()

        logger.info("=== 下载任务完成 ===")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()