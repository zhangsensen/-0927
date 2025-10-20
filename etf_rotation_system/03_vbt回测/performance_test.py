#!/usr/bin/env python3
"""
性能测试脚本 - 验证向量化优化效果
测试权重组合生成的性能差异
"""

import time
import numpy as np
import pandas as pd
import itertools
import logging
from typing import List, Tuple

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('performance_test')

def generate_combinations_original(factors: List[str], weight_grid: List[float],
                                   weight_sum_range: Tuple[float, float],
                                   max_combinations: int) -> List[Tuple[float, ...]]:
    """原始方法 - 非向量化"""
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))

    # 非向量化求和
    weight_sums = np.array([sum(w) for w in weight_combos])
    valid_mask = (weight_sums >= weight_sum_range[0]) & (weight_sums <= weight_sum_range[1])
    valid_combos = [weight_combos[i] for i in range(len(weight_combos)) if valid_mask[i]]

    if len(valid_combos) > max_combinations:
        valid_combos = valid_combos[:max_combinations]

    return valid_combos

def generate_combinations_optimized(factors: List[str], weight_grid: List[float],
                                     weight_sum_range: Tuple[float, float],
                                     max_combinations: int) -> List[Tuple[float, ...]]:
    """优化方法 - 向量化"""
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))

    # 向量化计算
    weight_array = np.array(weight_combos)
    weight_sums = np.sum(weight_array, axis=1)
    valid_mask = (weight_sums >= weight_sum_range[0]) & (weight_sums <= weight_sum_range[1])

    # 优化：先过滤，再限制组合数
    if len(weight_combos) > max_combinations:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > max_combinations:
            valid_indices = valid_indices[:max_combinations]
        valid_combos = [weight_combos[i] for i in valid_indices]
    else:
        valid_combos = [weight_combos[i] for i in range(len(weight_combos)) if valid_mask[i]]

    return valid_combos

def benchmark_performance():
    """性能基准测试"""
    logger = setup_logging()

    # 测试参数
    factors = ['PRICE_POSITION_60D', 'MOM_ACCEL', 'VOLATILITY_120D',
              'VOL_VOLATILITY_20', 'VOLUME_PRICE_TREND', 'RSI_6',
              'INTRADAY_POSITION', 'INTRA_DAY_RANGE']
    weight_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    weight_sum_range = (0.8, 1.2)
    test_cases = [1000, 2000, 5000, 10000]

    logger.info("=== 性能测试开始 ===")
    logger.info(f"因子数: {len(factors)}")
    logger.info(f"权重网格: {weight_grid}")
    logger.info(f"权重和范围: {weight_sum_range}")

    results = []

    for max_combos in test_cases:
        logger.info(f"\n测试最大组合数: {max_combos:,}")

        # 测试原始方法
        logger.info("测试原始方法...")
        start_time = time.time()
        original_result = generate_combinations_original(
            factors, weight_grid, weight_sum_range, max_combos
        )
        original_time = time.time() - start_time

        # 测试优化方法
        logger.info("测试优化方法...")
        start_time = time.time()
        optimized_result = generate_combinations_optimized(
            factors, weight_grid, weight_sum_range, max_combos
        )
        optimized_time = time.time() - start_time

        # 验证结果一致性
        results_match = len(original_result) == len(optimized_result)
        if results_match:
            logger.info("✅ 结果一致性验证通过")
        else:
            logger.error(f"❌ 结果不一致: 原始={len(original_result)}, 优化={len(optimized_result)}")

        # 性能统计
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        improvement = ((original_time - optimized_time) / original_time) * 100

        result = {
            'max_combos': max_combos,
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'improvement_pct': improvement,
            'results_count': len(optimized_result),
            'results_match': results_match
        }
        results.append(result)

        logger.info(f"原始方法耗时: {original_time:.3f}秒")
        logger.info(f"优化方法耗时: {optimized_time:.3f}秒")
        logger.info(f"性能提升: {speedup:.2f}x ({improvement:.1f}%)")
        logger.info(f"有效组合数: {len(optimized_result):,}")

    # 输出汇总
    logger.info("\n=== 性能测试汇总 ===")
    logger.info(f"{'最大组合数':<10} {'原始耗时':<10} {'优化耗时':<10} {'加速比':<8} {'提升%':<8} {'有效组合':<10}")
    logger.info("-" * 70)

    for result in results:
        logger.info(f"{result['max_combos']:<10,} "
                   f"{result['original_time']:<10.3f} "
                   f"{result['optimized_time']:<10.3f} "
                   f"{result['speedup']:<8.2f} "
                   f"{result['improvement_pct']:<8.1f} "
                   f"{result['results_count']:<10,}")

    # 计算平均性能提升
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_improvement = np.mean([r['improvement_pct'] for r in results])

    logger.info(f"\n平均性能提升: {avg_speedup:.2f}x ({avg_improvement:.1f}%)")

    if avg_speedup > 1.5:
        logger.info("🎉 优化效果显著！向量化改进有效提升了性能。")
    elif avg_speedup > 1.1:
        logger.info("✅ 优化效果良好，性能有所提升。")
    else:
        logger.info("⚠️ 优化效果有限，可能需要进一步优化。")

    return results

def test_memory_usage():
    """测试内存使用情况"""
    logger = setup_logging()

    logger.info("\n=== 内存使用测试 ===")

    import psutil
    import os

    process = psutil.Process(os.getpid())

    # 基线内存
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"基线内存使用: {baseline_memory:.1f} MB")

    # 生成大型权重组合
    factors = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
    weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 测试理论组合数
    theoretical_combos = len(weight_grid) ** len(factors)
    logger.info(f"理论组合数: {theoretical_combos:,}")

    # 生成组合并测试内存
    start_time = time.time()
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))
    generation_time = time.time() - start_time

    combo_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = combo_memory - baseline_memory

    logger.info(f"组合生成耗时: {generation_time:.3f}秒")
    logger.info(f"组合生成后内存: {combo_memory:.1f} MB")
    logger.info(f"内存增加: {memory_increase:.1f} MB")
    logger.info(f"每个组合内存: {memory_increase * 1024 / len(weight_combos):.2f} KB")

    # 清理内存
    del weight_combos
    import gc
    gc.collect()

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"清理后内存: {final_memory:.1f} MB")

if __name__ == "__main__":
    print("ETF轮动系统 - 性能测试")
    print("=" * 50)

    # 运行性能基准测试
    benchmark_performance()

    # 运行内存使用测试
    test_memory_usage()

    print("\n性能测试完成！")