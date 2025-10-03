#!/usr/bin/env python3
"""
存储概念股票批量分析脚本
顺序处理所有存储概念股票并生成分析报告
"""

import os
import subprocess
import sys
from datetime import datetime

# 存储概念股票列表
STORAGE_STOCKS = [
    "000021.SZ",  # 长城开发
    "001309.SZ",  # 德明利
    "002049.SZ",  # 紫光国微
    "002156.SZ",  # 通富微电
    "300223.SZ",  # 北京君正
    "300661.SZ",  # 圣邦股份
    "300782.SZ",  # 卓胜微
    "301308.SZ",  # 江波龙
    "603986.SS",  # 兆易创新
    "688008.SS",  # 澜起科技
    "688123.SS",  # 聚辰股份
    "688200.SS",  # 华峰测控
    "688516.SS",  # 奥普特
    "688525.SS",  # 佰维存储
    "688766.SS",  # 普冉股份
    "688981.SS",  # 中芯国际
]


def analyze_stock(stock_code):
    """分析单只股票"""
    print(f"\n{'='*60}")
    print(f"正在分析: {stock_code}")
    print(f"{'='*60}")

    # 构建命令
    cmd = [
        sys.executable,
        "/Users/zhangshenshen/深度量化0927/a股/sz_technical_analysis.py",
        stock_code,
        "--data-dir",
        "/Users/zhangshenshen/深度量化0927/a股/存储概念",
        "--output-dir",
        "/Users/zhangshenshen/深度量化0927/存储概念分析报告",
    ]

    # 创建输出目录
    os.makedirs("/Users/zhangshenshen/深度量化0927/存储概念分析报告", exist_ok=True)

    try:
        # 运行分析
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ {stock_code} 分析完成")
            print(result.stdout)
        else:
            print(f"❌ {stock_code} 分析失败")
            print(f"错误: {result.stderr}")

    except subprocess.TimeoutExpired:
        print(f"⏰ {stock_code} 分析超时")
    except Exception as e:
        print(f"❌ {stock_code} 分析异常: {e}")


def main():
    """主函数"""
    print("存储概念股票批量分析开始")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共需分析 {len(STORAGE_STOCKS)} 只股票")
    print("=" * 60)

    # 顺序分析每只股票
    for i, stock_code in enumerate(STORAGE_STOCKS, 1):
        print(f"\n进度: {i}/{len(STORAGE_STOCKS)}")
        analyze_stock(stock_code)

    print(f"\n{'='*60}")
    print("批量分析完成！")
    print(f"分析报告保存在: /Users/zhangshenshen/深度量化0927/存储概念分析报告/")
    print("=" * 60)


if __name__ == "__main__":
    main()
