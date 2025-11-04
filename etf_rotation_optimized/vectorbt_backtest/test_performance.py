#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本 - 验证完全向量化优化效果
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from core.vectorized_engine import UltraFastVectorEngine
from core.weight_generator import SmartWeightGenerator


def generate_mock_data(n_dates=1000, n_etfs=50, n_factors=20):
    """生成模拟数据"""
    print(f"生成模拟数据: {n_dates}天 × {n_etfs}ETF × {n_factors}因子")

    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    etfs = [f"ETF_{i:03d}" for i in range(n_etfs)]

    # 价格数据
    np.random.seed(42)
    close_data = 100 * np.exp(
        np.cumsum(np.random.randn(n_dates, n_etfs) * 0.01, axis=0)
    )
    close_df = pd.DataFrame(close_data, index=dates, columns=etfs)

    # 因子数据
    factors_dict = {}
    for i in range(n_factors):
        factor_data = np.random.randn(n_dates, n_etfs)
        factors_dict[f"factor_{i:02d}"] = pd.DataFrame(
            factor_data, index=dates, columns=etfs
        )

    return factors_dict, close_df


def test_weight_generation_speed():
    """测试权重生成速度"""
    print("\n" + "=" * 80)
    print("测试1: 权重生成速度")
    print("=" * 80)

    n_factors = 18
    gen = SmartWeightGenerator(n_factors)

    # 测试各种方法
    methods = [
        (
            "Dirichlet (稀疏)",
            lambda n: gen.generate_dirichlet_weights(n, sparsity_bias=0.5),
        ),
        ("Sobol (低差异)", lambda n: gen.generate_sobol_weights(n)),
        (
            "稀疏Dirichlet",
            lambda n: gen.generate_sparse_dirichlet_weights(
                n, min_active_factors=1, max_active_factors=9
            ),
        ),
        ("L1高斯", lambda n: gen.generate_l1_projected_gaussian_weights(n)),
        ("混合策略", lambda n: gen.generate_mixed_strategy_weights(n)),
    ]

    for n_combos in [1000, 5000, 10000]:
        print(f"\n生成 {n_combos} 个权重组合:")
        for method_name, method_func in methods:
            start = time.time()
            weights = method_func(n_combos)
            elapsed = time.time() - start
            speed = n_combos / elapsed if elapsed > 0 else 0
            print(
                f"  {method_name:20s}: {elapsed:6.3f}秒, {speed:8.0f} 组合/秒, shape={weights.shape}"
            )


def test_signal_generation_speed():
    """测试信号生成速度"""
    print("\n" + "=" * 80)
    print("测试2: 信号生成速度（单策略）")
    print("=" * 80)

    factors_dict, close_df = generate_mock_data(n_dates=1400, n_etfs=43, n_factors=18)
    engine = UltraFastVectorEngine(factors_dict, close_df)

    # 测试不同参数
    test_cases = [
        (10, 5, "Top-10, 5天调仓"),
        (20, 10, "Top-20, 10天调仓"),
        (30, 20, "Top-30, 20天调仓"),
    ]

    weights = np.random.dirichlet(np.ones(18), size=1)[0]

    for top_n, rebalance_freq, desc in test_cases:
        start = time.time()
        for _ in range(100):  # 重复100次测试
            equity, metrics = engine.backtest_single_strategy(
                weights, top_n, rebalance_freq
            )
        elapsed = time.time() - start
        speed = 100 / elapsed if elapsed > 0 else 0
        print(f"  {desc:25s}: {elapsed:6.3f}秒 (100次), {speed:7.1f} 策略/秒")


def test_batch_backtest_speed():
    """测试批量回测速度"""
    print("\n" + "=" * 80)
    print("测试3: 批量回测速度（完全向量化）")
    print("=" * 80)

    factors_dict, close_df = generate_mock_data(n_dates=1400, n_etfs=43, n_factors=18)
    engine = UltraFastVectorEngine(factors_dict, close_df)

    gen = SmartWeightGenerator(18)

    # 测试不同规模
    test_cases = [
        (100, "100 权重组合"),
        (500, "500 权重组合"),
        (1000, "1000 权重组合"),
        (2000, "2000 权重组合"),
    ]

    for n_weights, desc in test_cases:
        # 生成权重
        weight_matrix = gen.generate_mixed_strategy_weights(n_weights)

        # 批量回测
        top_n_list = [10, 20, 30]
        rebalance_freq_list = [5, 10, 20]
        n_strategies = n_weights * len(top_n_list) * len(rebalance_freq_list)

        start = time.time()
        results = engine.batch_backtest(weight_matrix, top_n_list, rebalance_freq_list)
        elapsed = time.time() - start
        speed = n_strategies / elapsed if elapsed > 0 else 0

        print(
            f"  {desc:25s}: {n_strategies:5d} 策略, {elapsed:6.2f}秒, {speed:7.1f} 策略/秒"
        )


def test_memory_efficiency():
    """测试内存效率"""
    print("\n" + "=" * 80)
    print("测试4: 内存效率分析")
    print("=" * 80)

    import os

    import psutil

    process = psutil.Process(os.getpid())

    # 初始内存
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初始内存: {mem_before:.1f} MB")

    # 加载数据
    factors_dict, close_df = generate_mock_data(n_dates=1400, n_etfs=43, n_factors=18)
    mem_after_data = process.memory_info().rss / 1024 / 1024
    print(
        f"加载数据后: {mem_after_data:.1f} MB (增加 {mem_after_data - mem_before:.1f} MB)"
    )

    # 创建引擎
    engine = UltraFastVectorEngine(factors_dict, close_df)
    mem_after_engine = process.memory_info().rss / 1024 / 1024
    print(
        f"创建引擎后: {mem_after_engine:.1f} MB (增加 {mem_after_engine - mem_after_data:.1f} MB)"
    )

    # 生成权重
    gen = SmartWeightGenerator(18)
    weight_matrix = gen.generate_mixed_strategy_weights(5000)
    mem_after_weights = process.memory_info().rss / 1024 / 1024
    print(
        f"生成5000权重后: {mem_after_weights:.1f} MB (增加 {mem_after_weights - mem_after_engine:.1f} MB)"
    )

    # 批量回测
    results = engine.batch_backtest(weight_matrix[:1000], [10, 20], [5, 10])
    mem_after_backtest = process.memory_info().rss / 1024 / 1024
    print(
        f"回测4000策略后: {mem_after_backtest:.1f} MB (增加 {mem_after_backtest - mem_after_weights:.1f} MB)"
    )

    print(f"\n总内存占用: {mem_after_backtest:.1f} MB")


def main():
    print("\n" + "=" * 80)
    print("完全向量化回测引擎 - 性能测试")
    print("=" * 80)

    # 运行所有测试
    test_weight_generation_speed()
    test_signal_generation_speed()
    test_batch_backtest_speed()
    test_memory_efficiency()

    print("\n" + "=" * 80)
    print("性能测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
