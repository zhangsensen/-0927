#!/usr/bin/env python3
"""
简单的股票数据下载工具
Linus风格：简单就是美
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(ticker_symbol, output_dir):
    """下载股票数据，简单直接"""
    print(f"下载 {ticker_symbol} 数据...")

    # 创建目录
    stock_dir = os.path.join(output_dir, ticker_symbol)
    os.makedirs(stock_dir, exist_ok=True)

    # 计算时间范围 - 近一年数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # 下载日线数据
    print(f"下载日线数据...")
    daily_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    if not daily_data.empty:
        daily_data = daily_data.reset_index()
        daily_data['Date'] = daily_data['Date'].dt.strftime('%Y-%m-%d')
        daily_file = os.path.join(stock_dir, f"{ticker_symbol}_1d_{end_date.strftime('%Y-%m-%d')}.csv")
        daily_data.to_csv(daily_file, index=False)
        print(f"日线数据已保存: {daily_file}")
        print(f"日线数据条数: {len(daily_data)}")
    else:
        print(f"日线数据下载失败")

    # 下载小时线数据
    print(f"下载小时线数据...")
    try:
        hourly_data = yf.download(ticker_symbol, start=start_date, end=end_date, interval='1h')
        if not hourly_data.empty:
            hourly_data = hourly_data.reset_index()
            # 确保列名正确
            if 'Datetime' in hourly_data.columns:
                hourly_data = hourly_data.rename(columns={'Datetime': 'Date'})
            hourly_data['Date'] = pd.to_datetime(hourly_data['Date']).dt.strftime('%Y-%m-%d %H:%M')

            # 简单去重
            hourly_data = hourly_data.drop_duplicates()

            hourly_file = os.path.join(stock_dir, f"{ticker_symbol}_1h_{end_date.strftime('%Y-%m-%d')}.csv")
            hourly_data.to_csv(hourly_file, index=False)
            print(f"小时线数据已保存: {hourly_file}")
            print(f"小时线数据条数: {len(hourly_data)}")
        else:
            print(f"小时线数据下载失败")
    except Exception as e:
        print(f"小时线数据下载出错: {e}")

    print(f"{ticker_symbol} 数据下载完成\n")

def main():
    """主函数"""
    # 配置
    output_dir = "/Users/zhangshenshen/深度量化0927/a股/存储概念"

    # 需要下载的股票列表
    required_stocks = [
        "002371.SZ",  # 北方华创
        "300408.SZ",  # 三环集团
    ]

    print("开始下载推荐股票数据...")
    print(f"将下载 {len(required_stocks)} 只股票的数据到: {output_dir}")

    # 下载所有股票数据
    for ticker in required_stocks:
        try:
            download_stock_data(ticker, output_dir)
        except Exception as e:
            print(f"下载 {ticker} 失败: {e}")

    print("所有数据下载完成！")

if __name__ == "__main__":
    main()