#!/usr/bin/env python3
"""
单股票因子生成脚本 - 用于快速测试和单标的处理
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from factor_system.factor_generation.batch_factor_processor import BatchFactorProcessor
from factor_system.factor_generation.config import setup_logging


def main():
    """主函数 - 处理单只股票"""

    # 从命令行获取股票代码，默认为 0700.HK
    symbol = sys.argv[1] if len(sys.argv) > 1 else "0700.HK"

    print("🚀 单股票因子生成")
    print("=" * 60)
    print(f"目标股票: {symbol}")
    print("=" * 60)

    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = setup_logging(
        f"single_stock_{symbol.replace('.', '_')}_{timestamp}"
    )
    print(f"📝 日志文件: {log_file_path}")
    print()

    try:
        # 1. 初始化处理器
        print("🔧 初始化批量处理器...")
        processor = BatchFactorProcessor()
        print(f"   ⚙️ 配置文件: {processor.config.config_file}")
        print(f"   ✅ 重采样功能: {'启用' if processor.enable_resampling else '禁用'}")
        print(f"   ✅ 输出目录: {processor.output_dir}")
        print()

        # 2. 扫描原始数据
        print("🔍 扫描原始数据...")
        # 🔧 Linus式修复：使用 ProjectPaths 统一路径管理
        from factor_system.utils import get_raw_data_dir

        raw_dir = str(get_raw_data_dir())
        stocks = processor.discover_stocks(raw_dir)

        # 3. 查找目标股票
        if symbol not in stocks:
            print(f"❌ 未找到股票 {symbol}")
            print(f"可用股票: {list(stocks.keys())[:10]}")
            return False

        stock_info = stocks[symbol]
        print(f"✅ 找到股票: {symbol}")
        print(f"   市场: {stock_info.market}")
        print(f"   现有时间框架: {stock_info.timeframes}")
        print()

        # 4. 处理单只股票
        print(f"⚡ 开始处理 {symbol}...")
        print()

        # 直接调用 process_single_stock（主进程执行，避免多进程问题）
        symbol_result, success, error_msg, factor_count = (
            processor.process_single_stock(stock_info)
        )

        # 5. 显示结果
        print("\n📈 处理结果:")
        if success:
            print(f"   ✅ 成功处理 {symbol_result}")
            print(f"   📊 生成因子数: {factor_count}")
            print()

            # 显示输出文件
            print("📁 输出文件:")
            market_dir = processor.output_dir / stock_info.market
            if market_dir.exists():
                for tf_dir in sorted(market_dir.iterdir()):
                    if tf_dir.is_dir():
                        files = list(tf_dir.glob(f"{symbol}_*.parquet"))
                        if files:
                            print(f"   {tf_dir.name}/: {len(files)} 个文件")
                            for f in files[:3]:  # 显示前3个
                                print(f"     - {f.name}")
        else:
            print(f"   ❌ 处理失败: {error_msg}")
            return False

        print()
        print("🎉 单股票处理完成！")
        return True

    except Exception as e:
        print(f"❌ 处理过程中发生异常: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
