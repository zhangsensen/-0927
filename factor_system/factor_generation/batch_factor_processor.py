#!/usr/bin/env python3
"""
批量因子处理器 - 企业级多股票因子计算系统
支持 raw/ 目录下所有股票的并行因子计算与存储

设计原则：
1. 高性能：并行处理 + 内存优化
2. 容错性：单股票失败不影响整体
3. 可监控：详细进度和性能指标
4. 可扩展：支持新市场和时间框架
"""

import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import psutil
from tqdm import tqdm

# 🔧 修复：确保子进程能找到 factor_system 模块
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from factor_system.factor_generation.config import get_config, setup_logging
except ImportError:
    # 相对导入
    from config import get_config, setup_logging
try:
    from factor_system.factor_generation.data_validator import DataValidator
    from factor_system.factor_generation.enhanced_factor_calculator import (
        EnhancedFactorCalculator,
        IndicatorConfig,
    )
    from factor_system.factor_generation.integrated_resampler import IntegratedResampler
except ImportError:
    # 相对导入
    from data_validator import DataValidator
    from enhanced_factor_calculator import EnhancedFactorCalculator, IndicatorConfig
    from integrated_resampler import IntegratedResampler

logger = logging.getLogger(__name__)


@dataclass
class StockInfo:
    """股票信息"""

    symbol: str
    market: str  # HK, US
    timeframes: List[str]
    file_paths: Dict[str, str]  # timeframe -> file_path


@dataclass
class ProcessingStats:
    """处理统计信息"""

    total_stocks: int = 0
    processed_stocks: int = 0
    failed_stocks: int = 0
    total_factors: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0

    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_stocks == 0:
            return 0.0
        return (self.processed_stocks / self.total_stocks) * 100

    @property
    def total_factors_generated(self) -> int:
        """兼容性属性，返回总因子数"""
        return self.total_factors


class BatchFactorProcessor:
    """批量因子处理器"""

    def __init__(self, config_path: Optional[str] = None):
        """初始化处理器"""
        self.config = get_config(config_path)
        if self.config.config_data is None:
            self.config.load_config()

        self.stats = ProcessingStats()
        self.failed_stocks: List[Tuple[str, str]] = []  # (symbol, error)

        # 性能配置
        perf_config = self.config.get("performance", {})
        self.max_workers = min(
            mp.cpu_count(), perf_config.get("max_workers", mp.cpu_count())
        )
        self.memory_limit_gb = perf_config.get("memory_limit_gb", 8)

        # 输出配置
        output_config = self.config.get("output", {})
        self.output_dir = Path(output_config.get("directory", "./factor_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.separate_by_market = output_config.get("separate_by_market", True)
        self.enable_validation = output_config.get("enable_validation", True)

        # 批处理时间戳（用于文件命名）
        self.batch_timestamp = None

        # 重采样配置
        self.enable_resampling = self.config.get("resampling", {}).get("enable", True)
        self.temp_resample_dir = Path(
            self.config.get("resampling", {}).get("temp_dir", "./temp_resampled")
        )
        self.resampler = IntegratedResampler() if self.enable_resampling else None

        # 数据校对配置
        self.validator = DataValidator() if self.enable_validation else None
        self.validation_results = []

        logger.info(f"批量处理器初始化完成")
        logger.info(f"最大并行数: {self.max_workers}")
        logger.info(f"内存限制: {self.memory_limit_gb}GB")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"重采样功能: {'启用' if self.enable_resampling else '禁用'}")

    def discover_stocks(self, raw_dir: str) -> Dict[str, StockInfo]:
        """发现所有股票及其数据文件"""
        raw_path = Path(raw_dir)
        stocks = {}

        logger.info(f"扫描原始数据目录: {raw_path}")

        # 扫描所有市场
        for market_dir in raw_path.iterdir():
            if not market_dir.is_dir():
                continue

            market = market_dir.name
            logger.info(f"扫描市场: {market}")

            # 按股票分组文件
            stock_files = {}
            for file_path in market_dir.glob("*.parquet"):
                # 解析文件名：0700HK_5min_2025-03-05_2025-09-01.parquet
                parts = file_path.stem.split("_")
                if len(parts) < 2:
                    continue

                symbol = parts[0]
                original_timeframe = parts[1]

                # 🔧 标准化时间框架标签：15m->15min, 30m->30min, 60m->60min
                if self.resampler:
                    timeframe = self.resampler.normalize_timeframe_label(
                        original_timeframe
                    )
                    if original_timeframe != timeframe:
                        logger.debug(
                            f"标准化时间框架: {original_timeframe} -> {timeframe}"
                        )
                else:
                    timeframe = original_timeframe

                if symbol not in stock_files:
                    stock_files[symbol] = {}
                # 🔧 关键：如果同一个标准化时间框架有多个文件，优先使用原始文件
                if timeframe not in stock_files[symbol]:
                    stock_files[symbol][timeframe] = str(file_path)

            # 创建股票信息
            for symbol, files in stock_files.items():
                # 🔧 标准化股票符号：0700HK -> 0700.HK
                if symbol.endswith(market):
                    clean_symbol = symbol[: -len(market)]
                    standardized_symbol = f"{clean_symbol}.{market}"
                else:
                    standardized_symbol = f"{symbol}.{market}"

                stocks[standardized_symbol] = StockInfo(
                    symbol=standardized_symbol,
                    market=market,
                    timeframes=list(files.keys()),
                    file_paths=files,
                )

        logger.info(f"发现 {len(stocks)} 只股票")
        for symbol, info in list(stocks.items())[:5]:  # 显示前5只
            logger.info(f"  {symbol}: {len(info.timeframes)} 个时间框架")

        return stocks

    def validate_stock_data(self, stock_info: StockInfo) -> bool:
        """验证股票数据完整性"""
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}

        for timeframe, file_path in stock_info.file_paths.items():
            try:
                # 快速检查：读取文件并检查前几行
                df = pd.read_parquet(file_path)
                if df.empty:
                    logger.warning(f"{stock_info.symbol} {timeframe}: 数据为空")
                    return False

                # 只检查前几行的列
                sample_df = df.head(5)
                if not required_columns.issubset(set(sample_df.columns)):
                    logger.warning(f"{stock_info.symbol} {timeframe}: 缺少必要列")
                    return False

            except Exception as e:
                logger.warning(f"{stock_info.symbol} {timeframe}: 数据读取失败 - {e}")
                return False

        return True

    def process_single_stock(self, stock_info: StockInfo) -> Tuple[str, bool, str, int]:
        """处理单只股票的所有时间框架

        Returns:
            (symbol, success, error_msg, factor_count)
        """
        symbol = stock_info.symbol

        try:
            # 内存监控
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            total_factors = 0

            # 检查是否需要重采样生成缺失时间框架
            complete_file_paths = stock_info.file_paths.copy()

            if self.enable_resampling and self.resampler:
                # 获取配置中需要的时间框架
                required_timeframes = self.config.get("timeframes", {}).get(
                    "enabled",
                    ["1min", "2min", "3min", "5min", "15min", "30min", "60min", "1day"],
                )

                # 确保所有时间框架数据存在
                complete_file_paths = self.resampler.ensure_all_timeframes(
                    symbol,
                    stock_info.file_paths,
                    required_timeframes,
                    self.temp_resample_dir,
                )

                logger.debug(
                    f"{symbol}: 完成时间框架检查，共 {len(complete_file_paths)} 个时间框架"
                )

            # 初始化因子计算器 - 复用实例避免重复初始化
            from pathlib import Path

            from factor_system.factor_engine.batch_calculator import (
                BatchFactorCalculator,
            )

            # 🔧 Linus式修复：使用 ProjectPaths 统一路径管理
            from factor_system.utils import get_project_root

            project_root = get_project_root()
            calculator = BatchFactorCalculator(
                raw_data_dir=project_root,
                enable_cache=True,
            )

            # 为每个时间框架计算因子
            for timeframe, file_path in complete_file_paths.items():

                # 读取数据
                df = pd.read_parquet(file_path)

                # 数据预处理
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)

                is_resampled = timeframe not in stock_info.file_paths

                factors_df = pd.DataFrame()

                if not is_resampled:
                    # 优先使用FactorEngine（具备缓存能力）
                    factors_df = calculator.calculate_all_factors(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=df.index.min(),
                        end_date=df.index.max(),
                    )

                    if isinstance(factors_df.index, pd.MultiIndex):
                        factors_df = factors_df.xs(symbol, level="symbol")

                    if not factors_df.empty:
                        factors_df = factors_df.reindex(df.index)

                if factors_df is None or factors_df.empty:
                    # 🔧 对补充时间框架或缓存未命中的场景，直接消费DataFrame
                    factors_df = calculator.calculate_factors_from_df(df, timeframe)

                if factors_df is not None and not factors_df.empty:
                    # 🔧 关键修改：将价格数据合并到因子数据中
                    price_columns = ["open", "high", "low", "close", "volume"]
                    available_price_columns = [
                        col for col in price_columns if col in df.columns
                    ]

                    if available_price_columns:
                        # 将价格数据添加到因子数据中
                        price_data = df[available_price_columns]

                        # 合并因子数据和价格数据
                        combined_df = pd.concat([price_data, factors_df], axis=1)

                        logger.debug(
                            f"{symbol} {timeframe}: 合并数据 - 价格列: {len(available_price_columns)}, 因子列: {len(factors_df.columns)}"
                        )
                    else:
                        combined_df = factors_df
                        logger.warning(
                            f"{symbol} {timeframe}: 未找到价格数据列，仅保存因子数据"
                        )

                    # 保存合并后的数据 - 按市场分别存储
                    timestamp_suffix = (
                        f"_{self.batch_timestamp}" if self.batch_timestamp else ""
                    )
                    if self.separate_by_market:
                        market_dir = self.output_dir / stock_info.market
                        output_path = (
                            market_dir
                            / timeframe
                            / f"{symbol}_{timeframe}_factors{timestamp_suffix}.parquet"
                        )
                    else:
                        output_path = (
                            self.output_dir
                            / timeframe
                            / f"{symbol}_{timeframe}_factors{timestamp_suffix}.parquet"
                        )

                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # 保存合并后的数据（包含价格+因子）
                    combined_df.to_parquet(
                        output_path, compression="snappy", index=True
                    )

                    total_factors += len(factors_df.columns)
                    logger.debug(
                        f"{symbol} {timeframe}: {len(factors_df.columns)} 个因子 + {len(available_price_columns)} 个价格列"
                    )

                # 内存检查
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > self.memory_limit_gb * 1024:
                    logger.warning(f"{symbol}: 内存使用过高 {current_memory:.1f}MB")

            end_memory = process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            logger.info(
                f"✅ {symbol}: {total_factors} 个因子, 内存: {memory_used:.1f}MB"
            )
            return symbol, True, "", total_factors

        except Exception as e:
            error_msg = f"{symbol} 处理失败: {str(e)}"
            logger.error(error_msg)
            return symbol, False, error_msg, 0

    def process_batch(
        self, stocks: Dict[str, StockInfo], batch_size: Optional[int] = None
    ) -> ProcessingStats:
        """批量处理所有股票"""

        if batch_size is None:
            batch_size = len(stocks)

        # 验证数据
        logger.info("验证股票数据完整性...")
        valid_stocks = {}
        for symbol, stock_info in stocks.items():
            if self.validate_stock_data(stock_info):
                valid_stocks[symbol] = stock_info
            else:
                self.failed_stocks.append((symbol, "数据验证失败"))

        logger.info(f"有效股票: {len(valid_stocks)}/{len(stocks)}")

        # 更新统计
        self.stats.total_stocks = len(valid_stocks)
        start_time = time.time()

        # 并行处理
        logger.info(f"开始并行处理 {len(valid_stocks)} 只股票...")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_symbol = {
                executor.submit(self.process_single_stock, stock_info): symbol
                for symbol, stock_info in valid_stocks.items()
            }

            # 处理结果
            with tqdm(total=len(valid_stocks), desc="处理股票") as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]

                    try:
                        symbol_result, success, error_msg, factor_count = (
                            future.result()
                        )

                        if success:
                            self.stats.processed_stocks += 1
                            self.stats.total_factors += factor_count
                        else:
                            self.stats.failed_stocks += 1
                            self.failed_stocks.append((symbol_result, error_msg))

                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "processed": self.stats.processed_stocks,
                                "failed": self.stats.failed_stocks,
                                "factors": self.stats.total_factors,
                            }
                        )

                    except Exception as e:
                        self.stats.failed_stocks += 1
                        self.failed_stocks.append((symbol, f"执行异常: {str(e)}"))
                        pbar.update(1)

        # 完成统计
        self.stats.processing_time = time.time() - start_time
        self.stats.memory_peak_mb = psutil.Process().memory_info().rss / 1024 / 1024

        if self.failed_stocks:
            preview = ", ".join(f"{sym}: {err}" for sym, err in self.failed_stocks[:5])
            logger.error(
                f"处理失败 {len(self.failed_stocks)} 只股票 (前5条): {preview}"
            )

            # 输出完整失败列表供排查
            failed_summary = "\n".join(
                f"{sym}: {err}" for sym, err in self.failed_stocks
            )
            logger.error(f"处理失败股票明细:\n{failed_summary}")

        # 清理临时重采样文件
        if self.enable_resampling and self.config.get("resampling", {}).get(
            "cleanup_temp", True
        ):
            self.cleanup_temp_files()

        return self.stats

    def cleanup_temp_files(self):
        """清理临时重采样文件"""
        try:
            if self.temp_resample_dir.exists():
                import shutil

                shutil.rmtree(self.temp_resample_dir)
                logger.info(f"已清理临时重采样目录: {self.temp_resample_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

    def generate_report(self) -> str:
        """生成处理报告"""
        report = []
        report.append("=" * 60)
        report.append("批量因子处理报告")
        report.append("=" * 60)
        report.append(f"总股票数: {self.stats.total_stocks}")
        report.append(f"成功处理: {self.stats.processed_stocks}")
        report.append(f"处理失败: {self.stats.failed_stocks}")
        report.append(
            f"成功率: {self.stats.processed_stocks/self.stats.total_stocks*100:.1f}%"
        )
        report.append(f"总因子数: {self.stats.total_factors}")
        report.append(f"处理时间: {self.stats.processing_time:.1f}秒")
        report.append(f"内存峰值: {self.stats.memory_peak_mb:.1f}MB")

        if self.failed_stocks:
            report.append("\n失败股票:")
            for symbol, error in self.failed_stocks[:10]:  # 显示前10个
                report.append(f"  {symbol}: {error}")
            if len(self.failed_stocks) > 10:
                report.append(f"  ... 还有 {len(self.failed_stocks)-10} 个失败")

        report.append("=" * 60)

        report_text = "\n".join(report)

        # 保存报告
        report_path = self.output_dir / "batch_processing_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"处理报告已保存: {report_path}")
        return report_text


def main():
    """主函数"""
    import argparse

    from factor_system.utils import get_raw_data_dir

    parser = argparse.ArgumentParser(description="批量因子处理器")
    parser.add_argument(
        "--raw-dir",
        default=str(get_raw_data_dir()),
        help="原始数据目录",
    )
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--workers", type=int, help="并行工作进程数")
    parser.add_argument("--memory-limit", type=float, help="内存限制(GB)")

    args = parser.parse_args()

    # 设置日志
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(timestamp)

    # 初始化处理器
    processor = BatchFactorProcessor(args.config)

    # 传递时间戳到处理器用于文件命名
    processor.batch_timestamp = timestamp

    # 覆盖配置
    if args.workers:
        processor.max_workers = args.workers
    if args.memory_limit:
        processor.memory_limit_gb = args.memory_limit

    try:
        # 发现股票
        stocks = processor.discover_stocks(args.raw_dir)

        if not stocks:
            logger.error("未发现任何股票数据")
            return

        # 批量处理
        stats = processor.process_batch(stocks)

        # 生成报告
        report = processor.generate_report()
        print(report)

        logger.info("批量处理完成")

    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理异常: {e}")
        raise


if __name__ == "__main__":
    main()
