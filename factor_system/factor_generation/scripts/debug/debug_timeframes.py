#!/usr/bin/env python3
"""调试时间框架重复问题"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_generation.batch_factor_processor import BatchFactorProcessor

processor = BatchFactorProcessor()
stocks = processor.discover_stocks("/Users/zhangshenshen/深度量化0927/raw")

stock_0700 = stocks.get("0700.HK")
if stock_0700:
    print(f"0700.HK 发现的时间框架:")
    for tf in sorted(stock_0700.timeframes):
        print(f"  {tf}")

    print(f"\n文件路径映射:")
    for tf, path in sorted(stock_0700.file_paths.items()):
        print(f"  {tf:10s} -> {Path(path).name}")

    # 模拟 ensure_all_timeframes
    if processor.resampler:
        required_timeframes = processor.config.get("timeframes", {}).get("enabled", [])
        print(f"\n配置要求的时间框架:")
        for tf in required_timeframes:
            print(f"  {tf}")

        complete_file_paths = processor.resampler.ensure_all_timeframes(
            "0700.HK", stock_0700.file_paths, required_timeframes, Path("./temp_test")
        )

        print(f"\n补全后的时间框架:")
        for tf in sorted(complete_file_paths.keys()):
            print(f"  {tf}")
