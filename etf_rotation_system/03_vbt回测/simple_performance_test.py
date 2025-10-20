#!/usr/bin/env python3
"""
简化性能测试 - 测试向量化优化的真实效果
"""

import itertools
import logging
import time

import numpy as np


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("simple_test")


def test_weight_generation():
    """测试权重生成的性能"""
    logger = setup_logging()

    # 测试参数
    factors = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    weight_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    weight_sum_range = (0.8, 1.2)
    max_combinations = 5000

    logger.info(f"测试参数:")
    logger.info(f"  因子数: {len(factors)}")
    logger.info(f"  权重网格: {weight_grid}")
    logger.info(f"  权重和范围: {weight_sum_range}")
    logger.info(f"  最大组合数: {max_combinations}")

    # 生成所有权重组合
    logger.info("生成权重组合...")
    start_time = time.time()
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))
    generation_time = time.time() - start_time

    logger.info(f"权重组合生成耗时: {generation_time:.3f}秒")
    logger.info(f"理论组合数: {len(weight_combos):,}")

    # 测试原始方法：列表推导式求和
    logger.info("测试原始方法...")
    start_time = time.time()
    weight_sums_old = [sum(w) for w in weight_combos]
    old_time = time.time() - start_time

    # 测试优化方法：numpy向量化求和
    logger.info("测试优化方法...")
    start_time = time.time()
    weight_array = np.array(weight_combos)
    weight_sums_new = np.sum(weight_array, axis=1)
    new_time = time.time() - start_time

    # 验证结果一致性
    results_match = np.allclose(weight_sums_old, weight_sums_new)
    logger.info(f"结果一致性: {'✅ 通过' if results_match else '❌ 失败'}")

    # 性能对比
    speedup = old_time / new_time if new_time > 0 else float("inf")
    improvement = ((old_time - new_time) / old_time) * 100

    logger.info(f"原始方法耗时: {old_time:.3f}秒")
    logger.info(f"优化方法耗时: {new_time:.3f}秒")
    logger.info(f"性能提升: {speedup:.2f}x ({improvement:.1f}%)")

    # 测试过滤性能
    logger.info("测试过滤性能...")

    # 原始过滤方法
    start_time = time.time()
    valid_mask_old = [
        (s >= weight_sum_range[0]) & (s <= weight_sum_range[1]) for s in weight_sums_old
    ]
    valid_indices_old = [i for i, valid in enumerate(valid_mask_old) if valid]
    if len(valid_indices_old) > max_combinations:
        valid_indices_old = valid_indices_old[:max_combinations]
    old_filter_time = time.time() - start_time

    # 优化过滤方法
    start_time = time.time()
    valid_mask_new = (weight_sums_new >= weight_sum_range[0]) & (
        weight_sums_new <= weight_sum_range[1]
    )
    valid_indices_new = np.where(valid_mask_new)[0]
    if len(valid_indices_new) > max_combinations:
        valid_indices_new = valid_indices_new[:max_combinations]
    new_filter_time = time.time() - start_time

    # 验证过滤结果一致性
    filter_match = len(valid_indices_old) == len(valid_indices_new)
    logger.info(f"过滤结果一致性: {'✅ 通过' if filter_match else '❌ 失败'}")
    logger.info(f"有效组合数: {len(valid_indices_new):,}")

    # 过滤性能对比
    filter_speedup = (
        old_filter_time / new_filter_time if new_filter_time > 0 else float("inf")
    )
    filter_improvement = ((old_filter_time - new_filter_time) / old_filter_time) * 100

    logger.info(f"原始过滤耗时: {old_filter_time:.3f}秒")
    logger.info(f"优化过滤耗时: {new_filter_time:.3f}秒")
    logger.info(f"过滤性能提升: {filter_speedup:.2f}x ({filter_improvement:.1f}%)")

    # 总体性能
    total_old = generation_time + old_time + old_filter_time
    total_new = generation_time + new_time + new_filter_time
    total_speedup = total_old / total_new if total_new > 0 else float("inf")
    total_improvement = ((total_old - total_new) / total_old) * 100

    logger.info(f"\n=== 总体性能对比 ===")
    logger.info(f"原始总耗时: {total_old:.3f}秒")
    logger.info(f"优化总耗时: {total_new:.3f}秒")
    logger.info(f"总体性能提升: {total_speedup:.2f}x ({total_improvement:.1f}%)")

    if total_speedup > 1.5:
        logger.info("🎉 向量化优化效果显著！")
    elif total_speedup > 1.1:
        logger.info("✅ 向量化优化有效果")
    else:
        logger.info("⚠️ 优化效果有限")

    return {
        "generation_time": generation_time,
        "old_calc_time": old_time,
        "new_calc_time": new_time,
        "old_filter_time": old_filter_time,
        "new_filter_time": new_filter_time,
        "total_speedup": total_speedup,
        "valid_combinations": len(valid_indices_new),
    }


if __name__ == "__main__":
    print("简化性能测试 - 权重组合向量化优化")
    print("=" * 50)
    test_weight_generation()
    print("\n测试完成！")
