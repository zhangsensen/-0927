#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH股票资金流数据下载脚本
基于Tushare Pro接口下载个股资金流向数据
"""

import os
import time
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SHMoneyFlowDownloader:
    def __init__(self, token=None):
        """
        初始化下载器

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
        self.sh_dir = "/Users/zhangshenshen/深度量化0927/raw/SH"
        self.output_dir = "/Users/zhangshenshen/深度量化0927/raw/SH/money_flow"

    def get_stock_list(self):
        """
        获取SH目录下的所有股票代码
        """
        stock_files = []
        for file in os.listdir(self.sh_dir):
            if file.endswith('.parquet'):
                stock_code = file.replace('.parquet', '')
                stock_files.append(stock_code)

        logger.info(f"发现 {len(stock_files)} 只股票文件")
        return sorted(stock_files)

    def get_date_range(self, stock_code):
        """
        设置固定的时间范围：2024-08-23 到 2025-08-22

        Parameters:
        stock_code (str): 股票代码

        Returns:
        tuple: (start_date, end_date)
        """
        # 设置固定的时间范围
        start_date = "20240823"
        end_date = "20250822"
        logger.info(f"{stock_code} 数据范围: {start_date} ~ {end_date}")

        return start_date, end_date

    def download_moneyflow(self, stock_code, start_date, end_date, max_retries=3):
        """
        下载单只股票的资金流向数据

        Parameters:
        stock_code (str): 股票代码
        start_date (str): 开始日期 YYYYMMDD
        end_date (str): 结束日期 YYYYMMDD
        max_retries (int): 最大重试次数

        Returns:
        pd.DataFrame or None: 资金流向数据
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"下载 {stock_code} 资金流向数据 (尝试 {attempt+1}/{max_retries})")

                # 调用资金流向接口
                df = self.pro.moneyflow(
                    ts_code=stock_code,
                    start_date=start_date,
                    end_date=end_date
                )

                if df is not None and not df.empty:
                    logger.info(f"成功下载 {stock_code} 资金流向数据: {len(df)} 条记录")
                    return df
                else:
                    logger.warning(f"{stock_code} 没有资金流向数据")
                    return None

            except Exception as e:
                logger.error(f"下载 {stock_code} 资金流向数据失败 (尝试 {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试

        logger.error(f"下载 {stock_code} 资金流向数据最终失败")
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
            return

        output_file = os.path.join(self.output_dir, f"{stock_code}_moneyflow.parquet")

        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)

            # 保存为parquet格式
            df.to_parquet(output_file, index=False)
            logger.info(f"保存 {stock_code} 资金流向数据到: {output_file}")

        except Exception as e:
            logger.error(f"保存 {stock_code} 资金流向数据失败: {e}")

    def download_all(self, stock_list=None, delay=1.0):
        """
        批量下载所有股票的资金流向数据

        Parameters:
        stock_list (list): 股票代码列表，如果为None则获取所有股票
        delay (float): 请求间隔时间（秒）
        """
        if stock_list is None:
            stock_list = self.get_stock_list()

        logger.info(f"开始下载 {len(stock_list)} 只股票的资金流向数据")

        success_count = 0
        failed_count = 0

        for i, stock_code in enumerate(stock_list, 1):
            logger.info(f"处理进度: {i}/{len(stock_list)} - {stock_code}")

            try:
                # 获取日期范围
                start_date, end_date = self.get_date_range(stock_code)
                logger.info(f"{stock_code} 数据范围: {start_date} ~ {end_date}")

                # 下载数据
                df = self.download_moneyflow(stock_code, start_date, end_date)

                # 保存数据
                self.save_moneyflow(stock_code, df)

                if df is not None:
                    success_count += 1
                else:
                    failed_count += 1

                # 请求间隔
                if delay > 0:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"处理 {stock_code} 时发生错误: {e}")
                failed_count += 1
                continue

        logger.info(f"下载完成! 成功: {success_count}, 失败: {failed_count}")
        return success_count, failed_count


def main():
    """
    主函数
    """
    logger.info("=== SH股票资金流数据下载开始 ===")

    try:
        # 初始化下载器
        downloader = SHMoneyFlowDownloader()

        # 获取股票列表
        stock_list = downloader.get_stock_list()

        # 可以指定特定股票进行测试
        # stock_list = ['600519.SH', '600036.SH', '688200.SH']  # 测试用

        # 批量下载
        success, failed = downloader.download_all(
            stock_list=stock_list,
            delay=1.2  # 请求间隔1.2秒，避免频率限制
        )

        logger.info(f"=== 下载完成 ===")
        logger.info(f"成功下载: {success} 只股票")
        logger.info(f"下载失败: {failed} 只股票")

    except Exception as e:
        logger.error(f"程序执行失败: {e}")


if __name__ == "__main__":
    main()