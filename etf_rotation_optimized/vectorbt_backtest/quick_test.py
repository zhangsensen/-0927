#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 使用真实数据测试优化效果
"""

import logging
import sys
import time
from pathlib import Path

# 确保在正确的目录
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import BacktestConfig
from core.ultra_fast_engine import UltraFastBacktestEngine

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # 加载配置
    config_path = Path(__file__).parent / "configs/backtest_config.yaml"
    logger.info(f"加载配置: {config_path}")
    config = BacktestConfig.from_yaml(config_path)

    # 创建引擎
    logger.info("初始化引擎...")
    engine = UltraFastBacktestEngine(config, logger)

    # 小规模测试
    n_weights = 500
    logger.info(f"开始小规模回测: {n_weights} 权重组合...")

    start = time.time()
    results_df = engine.run_backtest(n_weight_combinations=n_weights)
    elapsed = time.time() - start

    n_strategies = len(results_df)
    speed = n_strategies / elapsed if elapsed > 0 else 0

    logger.info(f"\n{'='*80}")
    logger.info(f"回测完成!")
    logger.info(f"{'='*80}")
    logger.info(f"总策略数: {n_strategies}")
    logger.info(f"总耗时: {elapsed:.2f} 秒")
    logger.info(f"平均速度: {speed:.1f} 策略/秒")
    logger.info(f"{'='*80}")

    # 打印Top-5
    logger.info("\nTop-5 策略:")
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        factors_str = ", ".join(
            [f"{f}({w:.2f})" for f, w in zip(row["factors"][:3], row["weights"][:3])]
        )
        logger.info(
            f"#{idx+1} | {factors_str} | "
            f"Top-N: {row['top_n']} | 调仓: {row['rebalance_freq']}天 | "
            f"Sharpe: {row['sharpe_ratio']:.4f} | 年化: {row['annual_return']:.2%}"
        )

    # 保存结果
    output_dir = Path(__file__).parent / "results" / f"quick_test_{int(time.time())}"
    logger.info(f"\n保存结果至: {output_dir}")
    engine.save_results(results_df, output_dir, top_k=20)

    logger.info("\n测试完成!")


if __name__ == "__main__":
    main()
