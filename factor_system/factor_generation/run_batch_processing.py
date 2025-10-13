#!/usr/bin/env python3
"""
批量因子处理快速启动脚本
一键处理 raw/ 目录下所有股票的因子计算
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from batch_factor_processor import BatchFactorProcessor
from config import setup_logging


def main():
    """主函数 - 一键批量处理"""

    print("🚀 批量因子处理系统启动")
    print("=" * 50)

    # 设置日志
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(timestamp)
    print(f"📝 日志文件: {log_file}")

    # 原始数据目录（仍可通过配置覆盖）
    # 🔧 Linus式修复：使用 ProjectPaths 统一路径管理
    from factor_system.utils import get_raw_data_dir
    raw_dir = str(get_raw_data_dir())

    try:
        # 初始化处理器
        print("\n🔧 初始化批量处理器...")
        processor = BatchFactorProcessor()
        print(f"⚙️  配置文件: {processor.config.config_file}")
        print(f"📂 原始数据: {raw_dir}")

        # 发现股票
        print("\n🔍 扫描股票数据...")
        stocks = processor.discover_stocks(raw_dir)

        if not stocks:
            print("❌ 未发现任何股票数据")
            return

        print(f"✅ 发现 {len(stocks)} 只股票")

        # 显示概览
        hk_count = sum(1 for s in stocks.values() if s.market == "HK")
        us_count = sum(1 for s in stocks.values() if s.market == "US")

        print(f"   📊 HK市场: {hk_count} 只")
        print(f"   📊 US市场: {us_count} 只")

        # 确认处理
        print(f"\n🎯 准备处理 {len(stocks)} 只股票的因子计算")
        print(f"   并行进程: {processor.max_workers}")
        print(f"   内存限制: {processor.memory_limit_gb}GB")

        response = input("\n是否开始处理? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            print("❌ 用户取消处理")
            return

        # 开始批量处理
        print("\n🚀 开始批量处理...")
        start_time = time.time()

        stats = processor.process_batch(stocks)

        # 生成报告
        print("\n📊 生成处理报告...")
        report = processor.generate_report()

        # 显示结果
        print("\n" + "=" * 50)
        print("🎉 批量处理完成!")
        print("=" * 50)
        print(report)

        # 性能总结
        total_time = time.time() - start_time
        if stats.processed_stocks > 0:
            avg_time = total_time / stats.processed_stocks
            print(f"\n⚡ 性能指标:")
            print(f"   平均处理时间: {avg_time:.2f}秒/股票")
            print(f"   因子生成速度: {stats.total_factors/total_time:.0f}个/秒")

        print(f"\n📁 输出目录: {processor.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理异常: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
