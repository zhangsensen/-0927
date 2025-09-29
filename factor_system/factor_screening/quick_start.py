#!/usr/bin/env python3
"""
快速启动脚本
作者：量化首席工程师
版本：1.0.0
日期：2025-09-30

一键启动常用的筛选任务
"""

import logging
from pathlib import Path
from config_manager import ConfigManager
from batch_screener import BatchScreener

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def quick_single_screening(symbol: str = "0700.HK", timeframe: str = "60min"):
    """快速单股票筛选"""
    logger.info(f"🚀 快速启动: {symbol} {timeframe} 筛选")
    
    config_manager = ConfigManager()
    batch_screener = BatchScreener(config_manager)
    
    # 创建快速配置
    batch_config = config_manager.create_batch_config(
        task_name=f"quick_{symbol}_{timeframe}",
        symbols=[symbol],
        timeframes=[timeframe],
        preset="quick"
    )
    
    # 运行筛选
    batch_result = batch_screener.run_batch(batch_config)
    
    # 保存结果
    saved_files = batch_screener.save_results(batch_result)
    
    logger.info(f"✅ 筛选完成! 结果保存在: {saved_files.get('summary')}")
    return batch_result


def quick_multi_timeframe(symbol: str = "0700.HK"):
    """快速多时间框架筛选"""
    timeframes = ["15min", "30min", "60min"]
    logger.info(f"🚀 快速启动: {symbol} 多时间框架筛选 {timeframes}")
    
    config_manager = ConfigManager()
    batch_screener = BatchScreener(config_manager)
    
    # 创建多时间框架配置
    batch_config = config_manager.create_batch_config(
        task_name=f"multi_tf_{symbol}",
        symbols=[symbol],
        timeframes=timeframes,
        preset="multi_timeframe"
    )
    
    # 运行筛选
    batch_result = batch_screener.run_batch(batch_config)
    
    # 保存结果
    saved_files = batch_screener.save_results(batch_result)
    
    logger.info(f"✅ 多时间框架筛选完成! 结果保存在: {saved_files.get('summary')}")
    return batch_result


def quick_multi_stocks():
    """快速多股票筛选"""
    symbols = ["0700.HK", "0005.HK", "0941.HK"]
    timeframe = "60min"
    logger.info(f"🚀 快速启动: 多股票筛选 {symbols} {timeframe}")
    
    config_manager = ConfigManager()
    batch_screener = BatchScreener(config_manager)
    
    # 创建多股票配置
    batch_config = config_manager.create_batch_config(
        task_name="multi_stocks",
        symbols=symbols,
        timeframes=[timeframe],
        preset="default"
    )
    
    # 运行筛选
    batch_result = batch_screener.run_batch(batch_config)
    
    # 保存结果
    saved_files = batch_screener.save_results(batch_result)
    
    logger.info(f"✅ 多股票筛选完成! 结果保存在: {saved_files.get('summary')}")
    return batch_result


def demo_all_presets():
    """演示所有预设配置"""
    logger.info("🚀 演示所有预设配置")
    
    config_manager = ConfigManager()
    
    print("\n=== 可用预设配置 ===")
    for name, desc in config_manager.list_presets().items():
        print(f"{name:15} - {desc}")
        
        # 显示配置详情
        config = config_manager.get_preset(name)
        print(f"  IC周期: {config.ic_horizons}")
        print(f"  最小样本: {config.min_sample_size}")
        print(f"  并行数: {config.max_workers}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("""
快速启动选项:

1. python quick_start.py single [股票代码] [时间框架]
   例: python quick_start.py single 0700.HK 60min

2. python quick_start.py multi_tf [股票代码]
   例: python quick_start.py multi_tf 0700.HK

3. python quick_start.py multi_stocks
   例: python quick_start.py multi_stocks

4. python quick_start.py demo
   例: python quick_start.py demo
        """)
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "single":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "0700.HK"
            timeframe = sys.argv[3] if len(sys.argv) > 3 else "60min"
            quick_single_screening(symbol, timeframe)
            
        elif command == "multi_tf":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "0700.HK"
            quick_multi_timeframe(symbol)
            
        elif command == "multi_stocks":
            quick_multi_stocks()
            
        elif command == "demo":
            demo_all_presets()
            
        else:
            print(f"未知命令: {command}")
            
    except Exception as e:
        logger.error(f"执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
