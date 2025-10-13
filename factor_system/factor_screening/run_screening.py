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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - 避免运行期循环导入
    from config_manager import ScreeningConfig

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _build_screening_config(*, session_dir: Optional[str] = None) -> "ScreeningConfig":
    """创建筛选配置，可选指定输出目录"""
    from config_manager import ScreeningConfig

    # 🔧 修复硬编码路径 - 使用项目根目录相对路径
    try:
        project_root = Path(__file__).parent.parent
        data_root = project_root / "factor_output"
        raw_data_root = project_root / ".." / "raw"
    except Exception:
        data_root = Path("../factor_output")
        raw_data_root = Path("../raw")

    base_kwargs = dict(
        data_root=str(data_root),
        raw_data_root=str(raw_data_root),
        output_root=str(session_dir) if session_dir else "./screening_results",
        enable_legacy_format=False,
    )

    return ScreeningConfig(**base_kwargs)


def run_single_screening(symbol: str, timeframe: str):
    """单股单时间框架筛选"""
    from data_loader_patch import patch_data_loader
    from professional_factor_screener import ProfessionalFactorScreener

    logger.info(f"🎯 单股筛选: {symbol} {timeframe}")

    config = _build_screening_config()

    screener = ProfessionalFactorScreener(config=config)
    patch_data_loader(screener)

    results = screener.screen_factors_comprehensive(symbol=symbol, timeframe=timeframe)

    logger.info(f"✅ 完成！发现 {len(results)} 个优质因子")
    return results


def _screen_single_timeframe_worker(args):
    """多进程工作函数"""
    symbol, timeframe, session_dir = args

    try:
        from data_loader_patch import patch_data_loader
        from professional_factor_screener import ProfessionalFactorScreener

        config = _build_screening_config(session_dir=session_dir)
        screener = ProfessionalFactorScreener(config=config)
        patch_data_loader(screener)

        result = screener.screen_factors_comprehensive(symbol=symbol, timeframe=timeframe)
        factor_count = len(result) if isinstance(result, dict) else 0
        return timeframe, result, factor_count, None
    except Exception as exc:  # pragma: no cover - 防御性日志
        return timeframe, None, 0, str(exc)


def run_multi_timeframe_screening(symbol: str, timeframes: list):
    """单股多时间框架筛选"""
    from data_loader_patch import patch_data_loader
    from professional_factor_screener import ProfessionalFactorScreener

    logger.info(f"🎯 多时间框架筛选: {symbol}")
    logger.info(f"⏰ 时间框架: {timeframes}")

    config = _build_screening_config()

    screener = ProfessionalFactorScreener(config=config)
    patch_data_loader(screener)

    results = screener.screen_multiple_timeframes(symbol, timeframes)

    total_factors = sum(len(r) for r in results.values() if isinstance(r, dict))
    logger.info(f"✅ 完成！共 {len(results)} 个时间框架，{total_factors} 个优质因子")
    return results


def run_multi_timeframe_screening_parallel(
    symbol: str,
    timeframes: list,
    *,
    max_workers: int = 4,
):
    """单股多时间框架并行筛选"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from datetime import datetime

    logger.info(f"🚀 并行多时间框架筛选: {symbol}")
    logger.info(f"⏰ 时间框架: {timeframes}")
    logger.info(f"⚡ 并行度={max_workers}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("screening_results") / f"{symbol}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_dir_str = str(session_dir.resolve())
    logger.info(f"📁 会话输出目录: {session_dir_str}")

    start_time = time.time()

    results = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _screen_single_timeframe_worker,
                (symbol, timeframe, session_dir_str),
            ): timeframe
            for timeframe in timeframes
        }

        for future in as_completed(futures):
            completed += 1

            tf, result, factor_count, error = future.result()
            if error is None and result is not None:
                results[tf] = result
                logger.info(f"✅ {tf} 完成 - {factor_count} 因子")
            else:
                logger.error(f"❌ {tf} 失败: {error}")

            progress = completed / len(timeframes) * 100
            logger.info(f"📈 进度 {completed}/{len(timeframes)} ({progress:.1f}%)")

    duration = time.time() - start_time
    total_factors = sum(len(r) for r in results.values() if isinstance(r, dict))

    logger.info("✅ 并行筛选完成！")
    logger.info(f"📈 完成时间框架: {len(results)}/{len(timeframes)}")
    logger.info(f"🎯 总因子数: {total_factors}")
    logger.info(f"⚡ 总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")

    return results


def run_batch_screening(market: str = None, limit: int = None, workers: int = 8):
    """批量高性能筛选"""
    import multiprocessing as mp

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
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from datetime import datetime

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

        print("\n" + "#" * 100)
        print(f"🎉 {market}市场批量筛选完成!")
        print("#" * 100)
        print(f"📊 总股票数: {len(stocks)}")
        print(f"✅ 成功筛选: {success_count} ({success_count/len(stocks)*100:.1f}%)")
        print(
            f"❌ 失败筛选: {len(failed_stocks)} ({len(failed_stocks)/len(stocks)*100:.1f}%)"
        )
        print(f"⏱️  总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
        print(f"⚡ 平均每股: {duration/len(stocks):.1f}秒")
        print(f"🚀 吞吐量: {len(stocks)/duration*60:.1f} 股票/分钟")
        print("#" * 100)

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
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用多进程并行处理多个时间框架",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="并行筛选进程数（仅 --parallel 时生效）",
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
            run_batch_screening("HK", args.limit, args.workers)
            run_batch_screening("US", args.limit, args.workers)
            logger.info("✅ 所有市场筛选完成！")
        elif args.market:
            run_batch_screening(args.market, args.limit, args.workers)
        else:
            parser.error("批量模式需要指定 --market 或 --all-markets")
    else:
        # 单股模式
        if not args.symbol:
            parser.error("单股模式需要指定 --symbol")

        if args.parallel and not args.timeframes:
            parser.error("并行模式需要指定 --timeframes")

        if args.timeframes:
            # 多时间框架
            if args.parallel:
                run_multi_timeframe_screening_parallel(
                    args.symbol,
                    args.timeframes,
                    max_workers=args.parallel_workers,
                )
            else:
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
