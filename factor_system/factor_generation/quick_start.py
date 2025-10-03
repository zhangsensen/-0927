#!/usr/bin/env python3
"""
快速启动脚本 - 多时间框架VectorBT检测器
简化启动流程，一键执行多时间框架因子分析
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    factor_system_dir = Path(__file__).parent
    os.chdir(factor_system_dir)

    if len(sys.argv) < 2:
        print("用法: python quick_start.py <股票代码>")
        print("示例: python quick_start.py 0700.HK")
        print("\n可用股票:")
        data_dir = Path("/Users/zhangshenshen/深度量化0927/raw/HK")
        stocks = []
        for file in data_dir.glob("*.parquet"):
            stock_name = file.stem.split("_")[0]
            if stock_name not in stocks:
                stocks.append(stock_name)
        for i in range(0, len(stocks), 5):
            print(f"  {', '.join(stocks[i:i+5])}")
        return

    stock_code = sys.argv[1]
    cmd = [sys.executable, "multi_tf_vbt_detector.py", stock_code]

    print(f"🚀 分析股票: {stock_code}")
    print(f"📍 执行目录: {factor_system_dir}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ 分析完成!")
        print(result.stdout)
    else:
        print("❌ 分析失败!")
        print(result.stderr)


if __name__ == "__main__":
    main()
