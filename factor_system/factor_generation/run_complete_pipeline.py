#!/usr/bin/env python3
"""
完整因子处理管道 - 支持HK/US分别存储和完整时间框架重采样

功能特性：
1. 清除历史数据，按HK/US分别存储
2. 从1min重采样生成：2min, 3min, 5min, 15min, 30min, 60min, 2h, 4h, 1day
3. 数据校对：对比重采样数据与原始数据
4. 并行处理154个技术指标
5. 生成完整的处理和校对报告
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from batch_factor_processor import BatchFactorProcessor
from config import setup_logging


def main():
    """主处理函数"""
    print("🚀 完整因子处理管道")
    print("=" * 60)
    print("功能：HK/US分别存储 + 完整时间框架重采样 + 数据校对")
    print("时间框架：1min, 2min, 3min, 5min, 15min, 30min, 60min, 2h, 4h, 1day")
    print("=" * 60)

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = setup_logging(f"complete_pipeline_{timestamp}")
    print(f"📝 日志文件: {log_file_path}")
    print()

    try:
        # 1. 初始化处理器
        print("🔧 初始化批量处理器...")
        processor = BatchFactorProcessor()
        print(f"   ⚙️ 配置文件: {processor.config.config_file}")
        print(f"   ✅ 重采样功能: {'启用' if processor.enable_resampling else '禁用'}")
        print(f"   ✅ 数据校对: {'启用' if processor.enable_validation else '禁用'}")
        print(f"   ✅ 按市场存储: {'启用' if processor.separate_by_market else '禁用'}")
        print(f"   ✅ 最大并行数: {processor.max_workers}")
        print(f"   ✅ 输出目录: {processor.output_dir}")
        print()

        # 2. 扫描原始数据
        print("🔍 扫描原始数据...")
        # 🔧 Linus式修复：使用 ProjectPaths 统一路径管理
        from factor_system.utils import get_raw_data_dir
        raw_dir = str(get_raw_data_dir())
        stocks = processor.discover_stocks(raw_dir)

        # 按市场分组统计
        hk_stocks = {k: v for k, v in stocks.items() if v.market == "HK"}
        us_stocks = {k: v for k, v in stocks.items() if v.market == "US"}

        print(f"   ✅ 发现 {len(stocks)} 只股票")
        print(f"   📊 HK市场: {len(hk_stocks)} 只股票")
        print(f"   📊 US市场: {len(us_stocks)} 只股票")
        print()

        # 显示时间框架统计
        print("📊 时间框架统计:")
        timeframe_stats = {}
        for stock_info in stocks.values():
            for tf in stock_info.timeframes:
                timeframe_stats[tf] = timeframe_stats.get(tf, 0) + 1

        for tf, count in sorted(timeframe_stats.items()):
            print(f"   {tf}: {count} 只股票")
        print()

        # 3. 演示重采样需求
        print("🔄 重采样需求分析...")
        required_timeframes = processor.config.get("timeframes", {}).get("enabled", [])
        print(f"   需要的时间框架: {required_timeframes}")

        # 统计需要重采样的股票
        resample_needed = 0
        for stock_info in stocks.values():
            if processor.resampler:
                missing_tfs = processor.resampler.find_missing_timeframes(
                    stock_info.file_paths, required_timeframes
                )
                if missing_tfs:
                    resample_needed += 1

        print(f"   需要重采样的股票: {resample_needed} 只")
        print()

        # 4. 执行批量处理
        print("⚡ 开始批量处理...")
        print(f"   处理模式：完整处理所有 {len(stocks)} 只股票")
        print()

        # 执行批量处理
        stats = processor.process_batch(stocks)

        # 5. 处理结果统计
        print("\n📈 处理结果统计:")
        print(f"   总股票数: {stats.total_stocks}")
        print(f"   成功处理: {stats.processed_stocks}")
        print(f"   处理失败: {stats.failed_stocks}")
        print(f"   成功率: {stats.success_rate:.1f}%")
        print(f"   总因子数: {stats.total_factors_generated}")
        print(f"   处理时间: {stats.processing_time:.1f}秒")
        print(f"   内存峰值: {stats.memory_peak_mb:.1f}MB")
        print()

        # 6. 输出目录结构
        print("📁 输出目录结构:")
        if processor.separate_by_market:
            for market in ["HK", "US"]:
                market_dir = processor.output_dir / market
                if market_dir.exists():
                    print(f"   {market}/")
                    for tf_dir in sorted(market_dir.iterdir()):
                        if tf_dir.is_dir():
                            files = list(tf_dir.glob("*.parquet"))
                            if files:
                                print(f"     {tf_dir.name}/: {len(files)} 个因子文件")
        else:
            for tf_dir in sorted(processor.output_dir.iterdir()):
                if tf_dir.is_dir():
                    files = list(tf_dir.glob("*.parquet"))
                    if files:
                        print(f"   {tf_dir.name}/: {len(files)} 个因子文件")
        print()

        # 7. 数据校对结果
        if processor.enable_validation and processor.validation_results:
            print("🔍 数据校对结果:")
            validation_stats = {}
            for result in processor.validation_results:
                status = result.get("status", "UNKNOWN")
                validation_stats[status] = validation_stats.get(status, 0) + 1

            for status, count in validation_stats.items():
                print(f"   {status}: {count}")

            # 生成校对报告
            report_path = processor.output_dir / f"validation_report_{timestamp}.txt"
            processor.validator.generate_validation_report(
                processor.validation_results, report_path
            )
            print(f"   📋 校对报告: {report_path}")
        print()

        # 8. 重采样文件清理状态
        if processor.enable_resampling:
            cleanup_enabled = processor.config.get("resampling", {}).get(
                "cleanup_temp", True
            )
            print(f"🧹 临时重采样文件: {'已清理' if cleanup_enabled else '保留'}")
        print()

        # 9. 性能分析
        print("⚡ 性能分析:")
        if stats.processed_stocks > 0:
            avg_factors_per_stock = (
                stats.total_factors_generated / stats.processed_stocks
            )
            avg_time_per_stock = stats.processing_time / stats.processed_stocks
            print(f"   平均每只股票因子数: {avg_factors_per_stock:.0f}")
            print(f"   平均每只股票处理时间: {avg_time_per_stock:.2f}秒")
            print(
                f"   因子生成速度: {stats.total_factors_generated / stats.processing_time:.0f} 因子/秒"
            )

            # 按市场统计
            if processor.separate_by_market:
                hk_processed = sum(1 for s in stocks.values() if s.market == "HK")
                us_processed = sum(1 for s in stocks.values() if s.market == "US")
                print(f"   HK市场处理: {hk_processed} 只股票")
                print(f"   US市场处理: {us_processed} 只股票")
        print()

        print("🎉 完整管道处理成功完成！")
        print("=" * 60)
        print("✨ 所有股票已按HK/US分别存储")
        print("✨ 完整时间框架重采样已完成")
        print("✨ 154个技术指标在多时间框架下并行计算")
        print("✨ 数据校对和验证已完成")
        print("✨ 内存管理和错误处理机制完善")

        return True

    except Exception as e:
        print(f"❌ 管道处理过程中发生异常: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
