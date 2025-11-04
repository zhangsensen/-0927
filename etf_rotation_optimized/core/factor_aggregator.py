"""
因子汇总器 | Factor Aggregator

从WFO结果中提取稳定的因子组合
- 统计因子出现频率
- 计算因子平均性能
- 输出最终推荐因子

作者: Linus Fix
日期: 2025-10-28
"""

import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorAggregator:
    """因子汇总器 - 从WFO结果提取稳定因子"""

    @staticmethod
    def aggregate_from_wfo(
        wfo_results: pd.DataFrame, top_n: int = 10, min_frequency: float = 0.3
    ) -> Dict:
        """
        从WFO结果中汇总因子

        Args:
            wfo_results: WFO结果DataFrame
            top_n: 返回前N个因子
            min_frequency: 最小出现频率（0-1）

        Returns:
            {
                'top_factors': List[str],  # 推荐因子
                'factor_stats': DataFrame,  # 因子统计
                'top_combinations': List[Tuple],  # 推荐组合
            }
        """
        # 1. 统计因子出现频率
        factor_counter = Counter()
        factor_ic_sum = {}
        factor_ic_count = {}

        for idx, row in wfo_results.iterrows():
            top10_combos = eval(row["top10_combos"])  # 字符串转列表
            oos_ic = row["oos_ensemble_ic"]

            for combo in top10_combos:
                for factor in combo:
                    factor_counter[factor] += 1

                    if factor not in factor_ic_sum:
                        factor_ic_sum[factor] = 0
                        factor_ic_count[factor] = 0

                    factor_ic_sum[factor] += oos_ic
                    factor_ic_count[factor] += 1

        # 2. 计算因子统计
        total_windows = len(wfo_results)
        factor_stats = []

        for factor, count in factor_counter.items():
            frequency = count / (total_windows * 10)  # 每窗口10个组合
            avg_ic = factor_ic_sum[factor] / factor_ic_count[factor]

            factor_stats.append(
                {
                    "factor": factor,
                    "frequency": frequency,
                    "avg_oos_ic": avg_ic,
                    "appearances": count,
                }
            )

        factor_stats_df = pd.DataFrame(factor_stats).sort_values(
            "frequency", ascending=False
        )

        # 3. 筛选稳定因子
        stable_factors = factor_stats_df[factor_stats_df["frequency"] >= min_frequency]

        if len(stable_factors) == 0:
            logger.warning(f"没有因子频率 >= {min_frequency}，降低阈值到0.1")
            stable_factors = factor_stats_df[factor_stats_df["frequency"] >= 0.1]

        top_factors = stable_factors.head(top_n)["factor"].tolist()

        # 4. 提取最优组合
        top_combinations = FactorAggregator._extract_top_combinations(
            wfo_results, top_k=5
        )

        logger.info(f"✅ 因子汇总完成:")
        logger.info(f"   - 推荐因子: {len(top_factors)} 个")
        logger.info(f"   - 最小频率: {min_frequency:.1%}")
        logger.info(f"   - 推荐组合: {len(top_combinations)} 个")

        return {
            "top_factors": top_factors,
            "factor_stats": factor_stats_df,
            "top_combinations": top_combinations,
        }

    @staticmethod
    def _extract_top_combinations(
        wfo_results: pd.DataFrame, top_k: int = 5
    ) -> List[Tuple[str, ...]]:
        """提取表现最好的因子组合"""
        combo_performance = {}

        for idx, row in wfo_results.iterrows():
            top10_combos = eval(row["top10_combos"])
            oos_ic = row["oos_ensemble_ic"]

            for combo in top10_combos:
                if combo not in combo_performance:
                    combo_performance[combo] = []
                combo_performance[combo].append(oos_ic)

        # 计算每个组合的平均OOS IC
        combo_avg_ic = {combo: np.mean(ics) for combo, ics in combo_performance.items()}

        # 排序并返回Top K
        sorted_combos = sorted(combo_avg_ic.items(), key=lambda x: x[1], reverse=True)

        return [combo for combo, ic in sorted_combos[:top_k]]
