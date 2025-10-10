#!/usr/bin/env python3
"""
下载603920股票数据
近1年小时线和日线数据
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def download_603920_data():
    """下载603920股票数据"""
    ticker_symbol = "603920.SS"  # 上海证券交易所
    output_dir = "/Users/zhangshenshen/深度量化0927/a股"

    print(f"开始下载 {ticker_symbol} 数据...")
    print(f"输出目录: {output_dir}")

    # 创建目录
    stock_dir = os.path.join(output_dir, ticker_symbol)
    os.makedirs(stock_dir, exist_ok=True)

    # 计算时间范围 - 近一年数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(
        f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}"
    )

    # 下载日线数据
    print(f"\n下载日线数据...")
    try:
        daily_data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if not daily_data.empty:
            daily_data = daily_data.reset_index()
            daily_data["Date"] = daily_data["Date"].dt.strftime("%Y-%m-%d")
            daily_file = os.path.join(
                stock_dir, f"{ticker_symbol}_1d_{end_date.strftime('%Y-%m-%d')}.csv"
            )
            daily_data.to_csv(daily_file, index=False)
            print(f"✅ 日线数据已保存: {daily_file}")
            print(f"   数据条数: {len(daily_data)}")
            print(
                f"   数据范围: {daily_data['Date'].min()} 到 {daily_data['Date'].max()}"
            )
        else:
            print(f"❌ 日线数据下载失败")
    except Exception as e:
        print(f"❌ 日线数据下载出错: {e}")

    # 下载小时线数据
    print(f"\n下载小时线数据...")
    try:
        hourly_data = yf.download(
            ticker_symbol, start=start_date, end=end_date, interval="1h"
        )
        if not hourly_data.empty:
            hourly_data = hourly_data.reset_index()
            # 确保列名正确
            if "Datetime" in hourly_data.columns:
                hourly_data = hourly_data.rename(columns={"Datetime": "Date"})
            hourly_data["Date"] = pd.to_datetime(hourly_data["Date"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

            # 简单去重
            hourly_data = hourly_data.drop_duplicates()

            hourly_file = os.path.join(
                stock_dir, f"{ticker_symbol}_1h_{end_date.strftime('%Y-%m-%d')}.csv"
            )
            hourly_data.to_csv(hourly_file, index=False)
            print(f"✅ 小时线数据已保存: {hourly_file}")
            print(f"   数据条数: {len(hourly_data)}")
            print(
                f"   数据范围: {hourly_data['Date'].min()} 到 {hourly_data['Date'].max()}"
            )
        else:
            print(f"❌ 小时线数据下载失败")
    except Exception as e:
        print(f"❌ 小时线数据下载出错: {e}")

    print(f"\n{ticker_symbol} 数据下载完成！")


if __name__ == "__main__":
    download_603920_data()
