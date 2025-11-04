#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超高性能回测 - 主运行脚本
全因子梯度权重组合，无多进程开销
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from core.config_loader import BacktestConfig
from core.ultra_fast_engine import UltraFastBacktestEngine


def setup_logger(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"ultra_backtest_{timestamp}.log"

    logger = logging.getLogger("UltraBacktest")
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
    parser = argparse.ArgumentParser(description="超高性能向量化回测引擎")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/backtest_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--n-weights", type=int, default=10000, help="权重组合数（默认1万）"
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
    logger.info("超高性能向量化回测引擎启动")
    logger.info("=" * 80)

    try:
        # 加载配置
        logger.info(f"加载配置: {config_path}")
        config = BacktestConfig.from_yaml(config_path)

        # 创建引擎
        logger.info("初始化引擎...")
        engine = UltraFastBacktestEngine(config, logger)

        # 运行回测
        import time

        start_time = time.time()

        logger.info(f"开始回测（目标权重组合数: {args.n_weights}）...")
        results_df = engine.run_backtest(n_weight_combinations=args.n_weights)

        elapsed_time = time.time() - start_time
        n_strategies = len(results_df)
        speed = n_strategies / elapsed_time if elapsed_time > 0 else 0

        logger.info(f"回测完成！耗时: {elapsed_time:.2f}秒")
        logger.info(f"速度: {speed:.1f} 策略/秒")

        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "results" / f"ultra_backtest_{timestamp}"
        logger.info(f"保存结果至: {output_dir}")
        engine.save_results(results_df, output_dir, top_k=args.top_k)

        # 打印Top-10
        logger.info("\n" + "=" * 80)
        logger.info("Top-10 策略:")
        logger.info("=" * 80)
        top10 = results_df.head(10)
        for idx, row in top10.iterrows():
            # 显示前5个因子
            factors_str = ", ".join(
                [
                    f"{f}({w:.2f})"
                    for f, w in zip(row["factors"][:5], row["weights"][:5])
                ]
            )
            if len(row["factors"]) > 5:
                factors_str += f" +{len(row['factors'])-5}更多"

            logger.info(
                f"#{idx+1} | 因子: {factors_str} | "
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
