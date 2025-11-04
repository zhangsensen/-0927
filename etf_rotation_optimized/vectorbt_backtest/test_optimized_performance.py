#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化后性能测试脚本
"""

import logging
import time
from pathlib import Path

import numpy as np
from core.config_loader import BacktestConfig
from core.ultra_fast_engine import UltraFastBacktestEngine


def test_optimized_performance():
    """测试优化后的性能"""

    # 设置日志
    logger = logging.getLogger("Test")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 加载配置
    config_path = Path(__file__).parent / "configs/backtest_config.yaml"
    config = BacktestConfig.from_yaml(config_path)

    logger.info("=" * 80)
    logger.info("优化后性能测试开始")
    logger.info("=" * 80)

    try:
        # 初始化引擎（包含因子有效性检验）
        logger.info("初始化优化引擎...")
        engine = UltraFastBacktestEngine(config, logger)

        # 测试参数
        n_weights_list = [1000, 5000, 10000]
        test_results = {}

        for n_weights in n_weights_list:
            logger.info("\n测试权重组合数: {}".format(n_weights))

            # 测试内存使用
            import os

            import psutil

            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB

            # 运行回测
            start_time = time.time()
            results_df = engine.run_backtest(n_weight_combinations=n_weights)
            elapsed_time = time.time() - start_time

            # 内存使用
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - start_memory

            # 性能指标
            n_strategies = len(results_df)
            speed = n_strategies / elapsed_time if elapsed_time > 0 else 0

            # 结果统计
            best_sharpe = results_df["sharpe_ratio"].max()
            best_return = results_df["annual_return"].max()
            best_drawdown = results_df["max_drawdown"].max()

            test_results[n_weights] = {
                "time": elapsed_time,
                "memory": memory_used,
                "speed": speed,
                "strategies": n_strategies,
                "best_sharpe": best_sharpe,
                "best_return": best_return,
                "best_drawdown": best_drawdown,
            }

            logger.info(
                "结果: {:.2f}s | {:.1f}MB | {:.1f}策略/秒".format(
                    elapsed_time, memory_used, speed
                )
            )
            logger.info(
                "性能: Sharpe={:.4f} | 年化={:.2%} | 回撤={:.2%}".format(
                    best_sharpe, best_return, best_drawdown
                )
            )

        # 输出总结
        logger.info("\n" + "=" * 80)
        logger.info("优化后性能测试总结")
        logger.info("=" * 80)

        for n_weights, result in test_results.items():
            logger.info("权重组合 {}:".format(n_weights))
            logger.info("  耗时: {:.2f}s".format(result["time"]))
            logger.info("  内存: {:.1f}MB".format(result["memory"]))
            logger.info("  速度: {:.1f} 策略/秒".format(result["speed"]))
            logger.info("  Sharpe: {:.4f}".format(result["best_sharpe"]))
            logger.info("  年化: {:.2%}".format(result["best_return"]))
            logger.info("  最大回撤: {:.2%}".format(result["best_drawdown"]))

        # 与优化前对比
        logger.info("\n与优化前对比:")
        logger.info("原性能: ~1150策略/秒, 850MB内存, Sharpe~0.484")
        logger.info("预期提升: 速度+100%, 内存-50%, Sharpe+30%")

    except Exception as e:
        logger.error("测试失败: {}".format(e), exc_info=True)
        raise


if __name__ == "__main__":
    test_optimized_performance()
