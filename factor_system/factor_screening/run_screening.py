#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子筛选系统 - 统一启动入口 (v3.1.0 公平评分集成版)
量化首席工程师 | 简洁、直接、务实

版本：3.1.0
状态：生产就绪
核心：公平评分算法，时间框架因子公平竞争

使用方法：
    # 单股单时间框架
    python run_screening.py --symbol 0700.HK --timeframe 5min
    
    # 单股多时间框架
    python run_screening.py --symbol 0700.HK --timeframes 5min 15min 60min
    
    # 批量筛选（高性能并行）
    python run_screening.py --batch --market HK --limit 10
    
    # 全市场筛选
    python run_screening.py --batch --market HK
    python run_screening.py --batch --market US
    python run_screening.py --batch --all-markets
"""

import argparse
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_single_screening(symbol: str, timeframe: str):
    """单股单时间框架筛选"""
    from config_manager import ScreeningConfig
    from data_loader_patch import patch_data_loader
    from professional_factor_screener import ProfessionalFactorScreener

    logger.info(f"🎯 单股筛选: {symbol} {timeframe}")

    # 🔧 修复硬编码路径 - 使用项目根目录相对路径
    try:
        project_root = Path(__file__).parent.parent
        data_root = project_root / "factor_output"
        raw_data_root = project_root / ".." / "raw"
    except Exception:
        data_root = Path("../factor_output")
        raw_data_root = Path("../raw")

    config = ScreeningConfig(
        data_root=str(data_root),
        raw_data_root=str(raw_data_root),
        output_root="./screening_results",
        enable_legacy_format=False,
    )

    screener = ProfessionalFactorScreener(config=config)
    patch_data_loader(screener)

    results = screener.screen_factors_comprehensive(symbol=symbol, timeframe=timeframe)

    logger.info(f"✅ 完成！发现 {len(results)} 个优质因子")
    return results


def run_multi_timeframe_screening(symbol: str, timeframes: list):
    """单股多时间框架筛选"""
    from config_manager import ScreeningConfig
    from data_loader_patch import patch_data_loader
    from professional_factor_screener import ProfessionalFactorScreener

    logger.info(f"🎯 多时间框架筛选: {symbol}")
    logger.info(f"⏰ 时间框架: {timeframes}")

    # 🔧 修复硬编码路径 - 使用项目根目录相对路径
    try:
        project_root = Path(__file__).parent.parent
        data_root = project_root / "factor_output"
        raw_data_root = project_root / ".." / "raw"
    except Exception:
        data_root = Path("../factor_output")
        raw_data_root = Path("../raw")

    config = ScreeningConfig(
        data_root=str(data_root),
        raw_data_root=str(raw_data_root),
        output_root="./screening_results",
        enable_legacy_format=False,
    )

    screener = ProfessionalFactorScreener(config=config)
    patch_data_loader(screener)

    results = screener.screen_multiple_timeframes(symbol, timeframes)

    total_factors = sum(len(r) for r in results.values() if isinstance(r, dict))
    logger.info(f"✅ 完成！共 {len(results)} 个时间框架，{total_factors} 个优质因子")
    return results


def run_batch_screening(market: str = None, limit: int = None, workers: int = 8):
    """批量高性能筛选"""
    import multiprocessing as mp

    from batch_screen_all_stocks_parallel import batch_screen_market_parallel

    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)

    if market:
        logger.info(f"🚀 批量筛选: {market}市场")
        if limit:
            logger.info(f"⚠️  限制前 {limit} 只股票（测试模式）")

        # 导入必要的函数
        from batch_screen_all_stocks_parallel import discover_stocks

        stocks = discover_stocks(market)

        if limit:
            stocks = stocks[:limit]

        # 临时修改：创建限制版本的任务
        import logging
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from datetime import datetime
        from pathlib import Path

        # 🔧 修复硬编码路径 - 使用项目根目录相对路径
        try:
            # 尝试获取项目根目录
            project_root = Path(__file__).parent.parent
            factor_output_root = project_root / "factor_output"
            raw_data_root = project_root / ".." / "raw"
        except Exception:
            # 回退到相对路径
            factor_output_root = Path("../factor_output")
            raw_data_root = Path("../raw")

        FACTOR_OUTPUT_ROOT = factor_output_root
        RAW_DATA_ROOT = raw_data_root
        ALL_TIMEFRAMES = [
            "1min",
            "2min",
            "3min",
            "5min",
            "15min",
            "30min",
            "60min",
            "2h",
            "4h",
            "1day",
        ]
        MAX_WORKERS = workers

        from batch_screen_all_stocks_parallel import screen_single_stock_worker

        start_time = datetime.now()
        logger.info(
            f"🚀 开始批量筛选 - {market}市场 (并发度: {MAX_WORKERS}进程, 股票: {len(stocks)}只)"
        )
        if limit:
            logger.info(f"⚠️ 测试模式: 前 {limit} 只股票")
        logger.info(
            f"📊 时间框架: {ALL_TIMEFRAMES}, 总任务: {len(stocks) * len(ALL_TIMEFRAMES)}"
        )

        # 准备任务
        tasks = [(symbol, market) for symbol in stocks]

        # 并行执行
        results = {}
        success_count = 0
        failed_stocks = []
        completed = 0

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(screen_single_stock_worker, task): task[0]
                for task in tasks
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1

                try:
                    symbol_result, succ_count, failed_tfs, error_msg = future.result()

                    if succ_count > 0:
                        results[symbol_result] = (succ_count, failed_tfs)
                        success_count += 1
                        status = f"✅ {succ_count}/{len(ALL_TIMEFRAMES)}"
                    else:
                        failed_stocks.append((symbol_result, error_msg))
                        status = (
                            f"❌ 失败: {error_msg[:50] if error_msg else 'Unknown'}"
                        )

                    # 进度显示
                    progress = completed / len(stocks) * 100
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta = (
                        elapsed / completed * (len(stocks) - completed)
                        if completed > 0
                        else 0
                    )

                    print(
                        f"[{completed:>3}/{len(stocks)}] {progress:>5.1f}% | {symbol:>12} | {status} | ETA: {eta/60:.1f}分钟"
                    )

                except Exception as e:
                    failed_stocks.append((symbol, str(e)))
                    print(
                        f"[{completed:>3}/{len(stocks)}] ❌ {symbol:>12} | 异常: {str(e)[:50]}"
                    )

        # 统计汇总
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n{'#'*100}")
        print(f"🎉 {market}市场批量筛选完成!")
        print(f"{'#'*100}")
        print(f"📊 总股票数: {len(stocks)}")
        print(f"✅ 成功筛选: {success_count} ({success_count/len(stocks)*100:.1f}%)")
        print(
            f"❌ 失败筛选: {len(failed_stocks)} ({len(failed_stocks)/len(stocks)*100:.1f}%)"
        )
        print(f"⏱️  总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"⚡ 平均每股: {duration/len(stocks):.1f}秒")
        print(f"🚀 吞吐量: {len(stocks)/duration*60:.1f} 股票/分钟")
        print(f"{'#'*100}")

        if failed_stocks:
            print(f"\n❌ 失败股票列表 (前10个):")
            for stock, error in failed_stocks[:10]:
                print(f"  - {stock}: {error}")
            if len(failed_stocks) > 10:
                print(f"  ... 还有 {len(failed_stocks) - 10} 个失败股票")

        return results
    else:
        logger.error(
            "批量筛选必须指定市场：--market HK 或 --market US 或 --all-markets"
        )
        return None


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="因子筛选系统 - 统一启动入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单股单时间框架
  python run_screening.py --symbol 0700.HK --timeframe 5min
  
  # 单股多时间框架
  python run_screening.py --symbol 0700.HK --timeframes 5min 15min 60min
  
  # 批量筛选（测试模式，前10只）
  python run_screening.py --batch --market HK --limit 10
  
  # 批量筛选全市场
  python run_screening.py --batch --market HK
  python run_screening.py --batch --market US
  
  # 批量筛选所有市场
  python run_screening.py --batch --all-markets
        """,
    )

    # 模式选择
    parser.add_argument("--batch", action="store_true", help="批量筛选模式")
    parser.add_argument("--all-markets", action="store_true", help="筛选所有市场")

    # 单股筛选参数
    parser.add_argument("--symbol", type=str, help="股票代码 (如: 0700.HK, AAPL.US)")
    parser.add_argument("--timeframe", type=str, help="单个时间框架 (如: 5min, 60min)")
    parser.add_argument(
        "--timeframes", nargs="+", help="多个时间框架 (如: 5min 15min 60min)"
    )

    # 批量筛选参数
    parser.add_argument(
        "--market", type=str, choices=["HK", "US"], help="市场代码 (HK或US)"
    )
    parser.add_argument("--limit", type=int, help="限制筛选的股票数量（用于测试）")
    parser.add_argument(
        "--workers", type=int, default=8, help="并行工作进程数（默认8）"
    )

    args = parser.parse_args()

    # 参数验证
    if args.batch:
        # 批量模式
        if args.all_markets:
            logger.info("🚀 开始筛选所有市场")
            results_hk = run_batch_screening("HK", args.limit, args.workers)
            results_us = run_batch_screening("US", args.limit, args.workers)
            logger.info("✅ 所有市场筛选完成！")
        elif args.market:
            run_batch_screening(args.market, args.limit, args.workers)
        else:
            parser.error("批量模式需要指定 --market 或 --all-markets")
    else:
        # 单股模式
        if not args.symbol:
            parser.error("单股模式需要指定 --symbol")

        if args.timeframes:
            # 多时间框架
            run_multi_timeframe_screening(args.symbol, args.timeframes)
        elif args.timeframe:
            # 单时间框架
            run_single_screening(args.symbol, args.timeframe)
        else:
            # 默认使用5min
            logger.warning("未指定时间框架，使用默认值: 5min")
            run_single_screening(args.symbol, "5min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}", exc_info=True)
        sys.exit(1)
