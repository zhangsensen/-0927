#!/usr/bin/env python3
"""
高性能并行因子筛选 - 适配M4芯片24GB内存
- 并发度: 8个进程 (M4 10核心，留2核给系统)
- HK市场: 108个股票 × 10个时间框架
- US市场: 68个股票 × 10个时间框架
- 筛选标准: 统计显著 OR Tier 1/2
"""

import logging
from pathlib import Path
from professional_factor_screener import ProfessionalFactorScreener
from config_manager import ScreeningConfig
from data_loader_patch import patch_data_loader
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 降低日志级别以提升性能
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔧 修复硬编码路径 - 智能路径解析
try:
    # 尝试获取项目根目录
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    factor_output_root = project_root / "factor_output"
    raw_data_root = project_root / ".." / "raw"

    # 验证目录存在性
    if not factor_output_root.exists():
        factor_output_root = Path("../factor_output")
    if not raw_data_root.exists():
        raw_data_root = Path("../raw")

except Exception:
    # 回退到相对路径
    factor_output_root = Path("../factor_output")
    raw_data_root = Path("../raw")

FACTOR_OUTPUT_ROOT = factor_output_root
RAW_DATA_ROOT = raw_data_root
ALL_TIMEFRAMES = ['1min', '2min', '3min', '5min', '15min', '30min', '60min', '2h', '4h', '1day']
MAX_WORKERS = 8  # M4芯片优化并发度


def discover_stocks(market: str) -> list[str]:
    """🔧 修复版：发现指定市场的所有股票，使用统一的market_utils"""
    try:
        from utils.market_utils import discover_stocks as utils_discover_stocks

        # 使用统一的股票发现函数
        stocks_dict = utils_discover_stocks(FACTOR_OUTPUT_ROOT, market)

        if market in stocks_dict:
            stocks = stocks_dict[market]
            logger.info(f"🔍 发现 {market} 市场股票: {len(stocks)} 只")
            return stocks
        else:
            logger.warning(f"未找到 {market} 市场股票")
            return []

    except ImportError:
        # 回退到简单的文件扫描（如果market_utils不可用）
        logger.warning("market_utils不可用，使用回退方案")

        # 🔧 修复：支持扁平目录结构
        market_dir = FACTOR_OUTPUT_ROOT / market
        if not market_dir.exists():
            logger.warning(f"市场目录不存在: {market_dir}")
            return []

        stocks = set()
        # 扫描所有因子文件
        for factor_file in market_dir.glob("*_factors_*.parquet"):
            try:
                # 解析股票代码：0005HK_1min_factors_20251008_224251.parquet -> 0005HK
                filename_parts = factor_file.stem.split('_')
                if len(filename_parts) >= 2:
                    symbol = filename_parts[0]
                    if symbol.endswith(market):
                        stocks.add(symbol)
            except Exception:
                continue

        result = sorted(list(stocks))
        logger.info(f"🔍 回退方案发现 {market} 市场股票: {len(result)} 只")
        return result


def screen_single_stock_worker(args):
    """工作进程：筛选单个股票
    
    Args:
        args: (symbol, market) 元组
    
    Returns:
        (symbol, success_count, failed_timeframes, error_msg)
    """
    symbol, market = args
    
    try:
        # 每个进程创建独立的筛选器实例
        config = ScreeningConfig(
            data_root=str(FACTOR_OUTPUT_ROOT),
            raw_data_root=str(RAW_DATA_ROOT),
            output_dir='screening_results',
            enable_legacy_format=False
        )
        
        screener = ProfessionalFactorScreener(config=config)
        patch_data_loader(screener)
        
        # 执行筛选
        result = screener.screen_multiple_timeframes(symbol, ALL_TIMEFRAMES)
        
        if result:
            success_count = sum(
                1 for tf_result in result.values() 
                if isinstance(tf_result, dict) and tf_result
            )
            failed_timeframes = [
                tf for tf, tf_result in result.items()
                if not (isinstance(tf_result, dict) and tf_result)
            ]
            return (symbol, success_count, failed_timeframes, None)
        else:
            return (symbol, 0, ALL_TIMEFRAMES, "筛选返回空结果")
            
    except Exception as e:
        return (symbol, 0, ALL_TIMEFRAMES, str(e))


def batch_screen_market_parallel(market: str):
    """高性能并行筛选指定市场的所有股票
    
    Args:
        market: 'HK' 或 'US'
    """
    start_time = datetime.now()
    print(f"\n{'#'*100}")
    print(f"🚀 开始高性能并行筛选 - {market}市场")
    print(f"⚡ 并发度: {MAX_WORKERS} 进程")
    print(f"{'#'*100}\n")
    
    # 1. 发现所有股票
    stocks = discover_stocks(market)
    if not stocks:
        print(f"❌ {market}市场未找到股票")
        return
    
    print(f"📊 {market}市场股票总数: {len(stocks)}")
    print(f"⏰ 时间框架: {ALL_TIMEFRAMES}")
    print(f"📈 总任务数: {len(stocks)} × {len(ALL_TIMEFRAMES)} = {len(stocks) * len(ALL_TIMEFRAMES)}\n")
    
    # 2. 准备任务参数
    tasks = [(symbol, market) for symbol in stocks]
    
    # 3. 并行执行
    results = {}
    success_count = 0
    failed_stocks = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_symbol = {
            executor.submit(screen_single_stock_worker, task): task[0]
            for task in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            
            try:
                symbol, succ_count, failed_tfs, error_msg = future.result()
                
                if succ_count > 0:
                    results[symbol] = (succ_count, failed_tfs)
                    success_count += 1
                    status = f"✅ {succ_count}/{len(ALL_TIMEFRAMES)}"
                else:
                    failed_stocks.append((symbol, error_msg))
                    status = f"❌ 失败: {error_msg[:50]}"
                
                # 进度显示
                progress = completed / len(stocks) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                eta = elapsed / completed * (len(stocks) - completed) if completed > 0 else 0
                
                print(f"[{completed:>3}/{len(stocks)}] {progress:>5.1f}% | {symbol:>12} | {status} | ETA: {eta/60:.1f}分钟")
                
            except Exception as e:
                failed_stocks.append((symbol, str(e)))
                print(f"[{completed:>3}/{len(stocks)}] ❌ {symbol:>12} | 异常: {str(e)[:50]}")
    
    # 4. 统计汇总
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'#'*100}")
    print(f"🎉 {market}市场批量筛选完成!")
    print(f"{'#'*100}")
    print(f"📊 总股票数: {len(stocks)}")
    print(f"✅ 成功筛选: {success_count} ({success_count/len(stocks)*100:.1f}%)")
    print(f"❌ 失败筛选: {len(failed_stocks)} ({len(failed_stocks)/len(stocks)*100:.1f}%)")
    print(f"⏱️  总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")
    print(f"⚡ 平均每股耗时: {duration/len(stocks):.1f}秒")
    print(f"🚀 吞吐量: {len(stocks)/duration*60:.1f} 股票/分钟")
    
    # 5. 详细统计
    if results:
        print(f"\n📈 成功股票详细统计:")
        full_success = sum(1 for _, (sc, _) in results.items() if sc == len(ALL_TIMEFRAMES))
        partial_success = success_count - full_success
        print(f"  - 全部时间框架成功: {full_success} ({full_success/len(stocks)*100:.1f}%)")
        print(f"  - 部分时间框架成功: {partial_success} ({partial_success/len(stocks)*100:.1f}%)")
    
    if failed_stocks:
        print(f"\n❌ 失败股票列表 (前10个):")
        for stock, error in failed_stocks[:10]:
            print(f"  - {stock}: {error}")
        if len(failed_stocks) > 10:
            print(f"  ... 还有 {len(failed_stocks) - 10} 个失败股票")
    
    return {
        'market': market,
        'total': len(stocks),
        'success': success_count,
        'failed': len(failed_stocks),
        'duration': duration,
        'results': results,
        'failed_stocks': failed_stocks
    }


def main():
    """主函数"""
    overall_start = datetime.now()
    
    print("="*100)
    print("🚀 高性能并行因子筛选系统")
    print("="*100)
    print(f"💻 硬件配置: M4芯片 + 24GB内存")
    print(f"⚡ 并发度: {MAX_WORKERS} 进程")
    print(f"📁 因子数据: {FACTOR_OUTPUT_ROOT}")
    print(f"📁 原始数据: {RAW_DATA_ROOT}")
    print(f"⏰ 时间框架: {ALL_TIMEFRAMES}")
    print(f"🔧 筛选标准: 统计显著 OR Tier 1/2")
    print("="*100 + "\n")
    
    # 统计数据
    hk_stocks = len(discover_stocks("HK"))
    us_stocks = len(discover_stocks("US"))
    total_stocks = hk_stocks + us_stocks
    total_tasks = total_stocks * len(ALL_TIMEFRAMES)
    
    print(f"📊 预览:")
    print(f"  - HK市场: {hk_stocks} 个股票")
    print(f"  - US市场: {us_stocks} 个股票")
    print(f"  - 总计: {total_stocks} 个股票")
    print(f"  - 总任务数: {total_tasks}")
    print(f"  - 预计耗时: {total_stocks * 2 / MAX_WORKERS / 60:.1f} 分钟 (假设每股2分钟)\n")
    
    # HK市场
    hk_result = batch_screen_market_parallel("HK")
    
    # US市场
    us_result = batch_screen_market_parallel("US")
    
    # 总体统计
    overall_end = datetime.now()
    overall_duration = (overall_end - overall_start).total_seconds()
    
    print(f"\n{'='*100}")
    print(f"🎉 全部市场筛选完成!")
    print(f"{'='*100}")
    print(f"⏱️  总耗时: {overall_duration:.1f}秒 ({overall_duration/60:.1f}分钟 / {overall_duration/3600:.1f}小时)")
    print(f"📊 HK市场: {hk_result['success']}/{hk_result['total']} 成功 ({hk_result['success']/hk_result['total']*100:.1f}%)")
    print(f"📊 US市场: {us_result['success']}/{us_result['total']} 成功 ({us_result['success']/us_result['total']*100:.1f}%)")
    print(f"🚀 总吞吐量: {total_stocks/overall_duration*60:.1f} 股票/分钟")
    print(f"⚡ 平均每股耗时: {overall_duration/total_stocks:.1f}秒")
    print("="*100)


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()

