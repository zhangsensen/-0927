#!/usr/bin/env python3
"""
批量重采样所有HK 1分钟数据到15m/30m/60m
Linus风格：极简实现，直接解决问题
"""

import os
import sys
from pathlib import Path

import pandas as pd

# 添加路径以导入HKResampler
sys.path.append("/Users/zhangshenshen/深度量化0927/data-resampling")
from resampling.hk_resampler import HKResampler


def batch_resample_all_1m():
    """批量处理所有1分钟数据"""

    # 查找所有1分钟文件
    hk_raw_dir = Path("/Users/zhangshenshen/深度量化0927/raw/HK")
    files_1m = list(hk_raw_dir.glob("*1m*.parquet"))

    print(f"发现 {len(files_1m)} 个1分钟文件待处理")

    # 输出目录
    output_dir = Path("/Users/zhangshenshen/深度量化0927/raw/HK/resampled")
    output_dir.mkdir(exist_ok=True)

    # 要生成的时间框架 (Linus风格修复：使用1h而不是60m)
    timeframes = ["15m", "30m", "1h"]

    success_count = 0
    error_count = 0

    for file_path in files_1m:
        try:
            print(f"处理: {file_path.name}")

            # 读取数据
            data = pd.read_parquet(file_path)

            # Linus风格关键修复：确保DatetimeIndex
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
                data = data.set_index("timestamp")

            # 初始化重采样器 (Linus风格：无参数构造函数)
            resampler = HKResampler()
            resampler.data = data  # 直接设置处理好的数据

            original_rows = len(data)

            # 对每个时间框架进行重采样
            for tf in timeframes:
                try:
                    # Linus风格修复：传入正确的参数
                    resampled_data = resampler.resample(data, tf)

                    # 构建输出文件名 (与原始文件保持一致的日期范围格式)
                    stock_code = file_path.stem.split("_")[0]
                    # 从原始文件名提取日期范围 (去掉原始时间周期)
                    date_range = "_".join(
                        file_path.stem.split("_")[2:]
                    )  # 取 "2025-03-06_2025-09-02"
                    output_file = output_dir / f"{stock_code}_{tf}_{date_range}.parquet"

                    # 保存
                    resampled_data.to_parquet(output_file)
                    compression_ratio = original_rows / len(resampled_data)

                    print(
                        f"  {tf}: {len(resampled_data)} 行 (压缩比 {compression_ratio:.1f}:1)"
                    )

                except Exception as e:
                    print(f"  {tf} 失败: {e}")
                    continue

            success_count += 1

        except Exception as e:
            print(f"❌ {file_path.name} 失败: {e}")
            error_count += 1
            continue

    print(f"\n批量处理完成:")
    print(f"✅ 成功: {success_count} 个文件")
    print(f"❌ 失败: {error_count} 个文件")
    print(f"📁 输出目录: {output_dir}")


if __name__ == "__main__":
    batch_resample_all_1m()
