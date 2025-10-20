"""
向量化优化版本的权重网格搜索
主要优化点：
1. 向量化权重组合生成
2. 向量化权重和计算
3. 向量化有效组合过滤
4. 批量处理优化
"""

import numpy as np
import pandas as pd
import itertools
import logging
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

def generate_weight_combinations_vectorized(
    factors: List[str],
    weight_grid: List[float],
    weight_sum_range: Tuple[float, float],
    max_combinations: int = 1000
) -> List[Tuple[float, ...]]:
    """
    向量化生成权重组合

    Args:
        factors: 因子列表
        weight_grid: 权重网格点
        weight_sum_range: 权重和范围
        max_combinations: 最大组合数

    Returns:
        有效的权重组合列表
    """
    logger = logging.getLogger('backtest_engine')

    # 生成所有可能的权重组合
    weight_combos = list(itertools.product(weight_grid, repeat=len(factors)))
    logger.info(f"生成权重组合: {len(weight_combos):,} 个理论组合")

    # 向量化计算权重和
    weight_array = np.array(weight_combos)
    weight_sums = np.sum(weight_array, axis=1)

    # 向量化过滤有效组合
    valid_mask = (weight_sums >= weight_sum_range[0]) & (weight_sums <= weight_sum_range[1])
    valid_combos = weight_combos[:max_combinations] if max_combinations < len(weight_combos) else weight_combos

    # 更高效的方法：使用numpy的布尔索引
    if len(weight_combos) > max_combinations:
        # 先过滤，再取前N个
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > max_combinations:
            valid_indices = valid_indices[:max_combinations]
        valid_combos = [weight_combos[i] for i in valid_indices]
    else:
        # 直接过滤
        valid_combos = [weight_combos[i] for i in range(len(weight_combos)) if valid_mask[i]]

    logger.info(f"过滤后有效组合: {len(valid_combos):,} 个")
    return valid_combos

def grid_search_weights_vectorized(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    factors: List[str],
    top_n_list: List[int],
    weight_grid: List[float],
    weight_sum_range: Tuple[float, float],
    max_combinations: int,
    rebalance_freq: int = 20,
    enable_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    向量化权重网格搜索

    主要优化：
    1. 向量化权重组合生成
    2. 批量Top-N测试
    3. 内存优化
    """
    from .backtest_engine_configurable import calculate_composite_score, backtest_topn_rotation

    logger = logging.getLogger('backtest_engine')

    # 向量化生成权重组合
    logger.info("开始向量化权重组合生成...")
    valid_combos = generate_weight_combinations_vectorized(
        factors=factors,
        weight_grid=weight_grid,
        weight_sum_range=weight_sum_range,
        max_combinations=max_combinations
    )

    logger.info(f"权重搜索配置: {len(valid_combos)} 组合 × {len(top_n_list)} Top-N")
    logger.info(f"权重网格: {weight_grid}")
    logger.info(f"权重和范围: {weight_sum_range}")
    print(f"开始向量化网格搜索: {len(valid_combos)} 组合 × {len(top_n_list)} Top-N")

    # 预计算所有得分矩阵
    score_cache = {}
    results = []
    tested_count = 0

    logger.info("开始向量化权重组合测试...")

    # 优化主循环：批量处理
    for i, weights in enumerate(tqdm(valid_combos, desc="权重组合")):
        weight_dict = dict(zip(factors, weights))
        weights_key = tuple(weights)

        # 缓存得分矩阵计算
        if enable_cache and weights_key not in score_cache:
            score_cache[weights_key] = calculate_composite_score(panel, factors, weight_dict)

        scores = score_cache[weights_key] if enable_cache else calculate_composite_score(panel, factors, weight_dict)

        # 批量测试所有Top-N值
        for top_n in top_n_list:
            try:
                result = backtest_topn_rotation(
                    prices=prices,
                    scores=scores,
                    top_n=top_n,
                    rebalance_freq=rebalance_freq
                )

                results.append({
                    'weights': str(weight_dict),
                    'top_n': top_n,
                    'total_return': result['total_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'final_value': result['final_value'],
                    'turnover': result['turnover']
                })

                tested_count += 1

                # 每100个组合记录一次进度
                if tested_count % 100 == 0:
                    logger.info(f"已测试 {tested_count} 个策略组合...")

            except Exception as e:
                logger.error(f"权重组合 {weights} Top-N {top_n} 回测失败: {e}")
                continue

    logger.info(f"开始处理搜索结果，共收集到 {len(results)} 个策略结果")

    # 按夏普比率排序
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    logger.info(f"结果已按 sharpe_ratio 排序")
    if results:
        best_weights = eval(results[0]['weights'])
        best_top_n = results[0]['top_n']
        best_sharpe = results[0]['sharpe_ratio']
        logger.info(f"最优策略: {best_weights}, top_n={best_top_n}, sharpe={best_sharpe:.3f}")

    logger.info(f"向量化权重网格搜索完成: 共测试 {len(results)} 个策略组合")
    return results

def optimize_memory_usage():
    """
    内存使用优化
    """
    import gc

    # 强制垃圾回收
    gc.collect()

    # 设置numpy内存限制
    # 注意：这需要在程序启动时设置
    # os.environ['OMP_NUM_THREADS'] = '2'
    # os.environ['VECLIB_MAXIMUM_THREADS'] = '2'