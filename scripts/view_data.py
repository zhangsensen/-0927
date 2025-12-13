#!/usr/bin/env python3
"""查看ETF数据"""
import sys
from pathlib import Path
import pandas as pd

data_dir = Path("./data/etf_daily")

if len(sys.argv) > 1:
    symbol = sys.argv[1]
    parquet_file = data_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        print(f"❌ 文件不存在: {parquet_file}")
        exit(1)
    
    df = pd.read_parquet(parquet_file)
    print(f"\n{symbol} - 总计 {len(df)} 条记录")
    print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print("\n最近5天数据:")
    print(df.tail(5).to_string(index=False))
else:
    print("\n已下载的ETF列表:")
    print("=" * 60)
    
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print("❌ 暂无数据")
    else:
        for f in files:
            df = pd.read_parquet(f)
            symbol = f.stem
            print(f"{symbol:8s} - {len(df):4d} 条  ({df['trade_date'].min()} ~ {df['trade_date'].max()})")
    
    print("\n使用方法: python view_data.py 510300")
