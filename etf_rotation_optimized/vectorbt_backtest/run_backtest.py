#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯向量化回测 - 主运行脚本
暴力测试所有因子组合，不使用WFO
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from core.backtest_engine import ParallelBacktestEngine
from core.config_loader import BacktestConfig


def setup_logger(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"backtest_{timestamp}.log"

    logger = logging.getLogger("VectorBacktest")
    logger.setLevel(getattr(logging, log_level.upper()))

    # 文件处理器
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper()))

    # 格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="纯向量化回测引擎")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backtest_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="并行进程数（默认CPU核心数-1）"
    )
    parser.add_argument(
        "--top-k", type=int, default=100, help="保存Top-K策略的权益曲线"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置路径
    project_root = Path(__file__).parent
    config_path = project_root / args.config

    # 设置日志
    logger = setup_logger(project_root / "logs", args.log_level)
    logger.info("=" * 80)
    logger.info("纯向量化回测引擎启动")
    logger.info("=" * 80)

    try:
        # 加载配置
        logger.info(f"加载配置: {config_path}")
        config = BacktestConfig.from_yaml(config_path)
        logger.info(f"配置加载成功")
        logger.info(f"  - 因子目录: {config.factor_dir}")
        logger.info(f"  - OHLCV目录: {config.ohlcv_dir}")
        logger.info(f"  - 初始资金: {config.init_cash:,.0f}")
        logger.info(f"  - 手续费: {config.fees:.4%}")
        logger.info(f"  - Top-N列表: {config.top_n_list}")
        logger.info(f"  - 调仓频率: {config.rebalance_freq_list}")
        logger.info(f"  - 因子组合大小: {config.combination_sizes}")

        # 创建回测引擎
        logger.info("初始化回测引擎...")
        engine = ParallelBacktestEngine(config, logger)

        # 枚举策略
        logger.info("枚举策略组合...")
        strategies = engine.enumerate_strategies()
        logger.info(f"共 {len(strategies)} 个策略待测试")

        # 执行并行回测
        logger.info("开始并行回测...")
        results_df = engine.run_parallel_backtest(strategies, n_workers=args.workers)

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "results" / f"backtest_{timestamp}"
        logger.info(f"保存结果至: {output_dir}")
        engine.save_results(results_df, output_dir, top_k=args.top_k)

        # 打印Top-10摘要
        logger.info("\n" + "=" * 80)
        logger.info("Top-10 策略:")
        logger.info("=" * 80)
        top10 = results_df.head(10)
        for idx, row in top10.iterrows():
            logger.info(
                f"#{idx+1} | 因子: {row['factors']} | 权重: {row['weights']} | "
                f"Top-N: {row['top_n']} | 调仓: {row['rebalance_freq']}天 | "
                f"Sharpe: {row['sharpe_ratio']:.4f} | 年化: {row['annual_return']:.2%} | "
                f"最大回撤: {row['max_drawdown']:.2%}"
            )

        logger.info("=" * 80)
        logger.info("回测完成！")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"回测失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
