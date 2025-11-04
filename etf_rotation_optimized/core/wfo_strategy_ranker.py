"""
WFO 策略排序器 (轻量版)

当前 DirectFactorWFOOptimizer 输出的是各窗口的“选中因子+权重”，
并非显式的多策略枚举。此排序器基于窗口结果聚合：
- 统计因子被选中频率
- 统计因子平均权重
并据此给出 Top-5 因子“策略位型”的建议排名。

输出: top5_strategies.csv (factor, select_freq, avg_weight, score)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class WfoStrategyRanker:
    def rank_and_save(self, results_list, out_csv_path) -> pd.DataFrame:
        freq: Dict[str, int] = {}
        weight_sum: Dict[str, float] = {}

        for r in results_list:
            for f in r.selected_factors:
                freq[f] = freq.get(f, 0) + 1
                weight_sum[f] = weight_sum.get(f, 0.0) + float(
                    r.factor_weights.get(f, 0.0)
                )

        if not freq:
            df = pd.DataFrame(columns=["factor", "select_freq", "avg_weight", "score"])
            df.to_csv(out_csv_path, index=False)
            return df

        total_windows = max(1, len(results_list))
        rows = []
        for f, c in freq.items():
            avg_w = weight_sum.get(f, 0.0) / c
            score = 0.7 * (c / total_windows) + 0.3 * avg_w  # 简单综合分
            rows.append(
                {
                    "factor": f,
                    "select_freq": c,
                    "avg_weight": avg_w,
                    "score": score,
                }
            )

        df = pd.DataFrame(rows).sort_values(
            ["score", "select_freq", "avg_weight"], ascending=False
        )
        df_top5 = df.head(5).reset_index(drop=True)
        df_top5.to_csv(out_csv_path, index=False)
        return df_top5


"""
WFO策略排序器 - Top-N最佳策略选择

功能:
1. 统计高频因子组合
2. 枚举因子组合并评估
3. 按综合得分排序
4. 选出Top-N最佳策略

作者: Linus Mode  
日期: 2025-11-03
"""

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WFOStrategyRanker:
    """
    WFO策略排序器

    核心功能:
    - 从WFO结果中提取高频因子
    - 枚举因子组合
    - 计算综合得分
    - 选出Top-N策略
    """

    def __init__(
        self,
        min_frequency: float = 0.3,
        max_factors_per_combo: int = 5,
        score_weights: Dict[str, float] = None,
    ):
        """
        初始化排序器

        Args:
            min_frequency: 最小出现频率（0-1）
            max_factors_per_combo: 每个组合最多包含的因子数
            score_weights: 评分权重
        """
        self.min_frequency = min_frequency
        self.max_factors_per_combo = max_factors_per_combo

        # 默认评分权重
        self.score_weights = score_weights or {
            "annual_return": 0.3,
            "sharpe_ratio": 0.25,
            "calmar_ratio": 0.2,
            "ic": 0.15,
            "stability": 0.1,
        }

    def extract_frequent_factors(self, wfo_results: List[Dict]) -> Dict[str, float]:
        """
        提取高频因子

        Args:
            wfo_results: WFO结果列表

        Returns:
            factor_freq: 因子频率字典 {factor_name: frequency}
        """
        factor_counts = {}
        total_windows = len(wfo_results)

        for result in wfo_results:
            factors = result.get("selected_factors", [])
            for factor in factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1

        # 计算频率
        factor_freq = {
            factor: count / total_windows for factor, count in factor_counts.items()
        }

        # 过滤低频因子
        frequent_factors = {
            factor: freq
            for factor, freq in factor_freq.items()
            if freq >= self.min_frequency
        }

        # 按频率排序
        frequent_factors = dict(
            sorted(frequent_factors.items(), key=lambda x: x[1], reverse=True)
        )

        logger.info(f"\n高频因子 (频率>={self.min_frequency:.0%}):")
        for factor, freq in frequent_factors.items():
            logger.info(f"  {factor:30s}: {freq:.1%}")

        return frequent_factors

    def enumerate_strategies(
        self,
        frequent_factors: Dict[str, float],
        min_factors: int = 3,
        max_factors: int = 5,
    ) -> List[Tuple[str, ...]]:
        """
        枚举因子组合

        Args:
            frequent_factors: 高频因子字典
            min_factors: 最少因子数
            max_factors: 最多因子数

        Returns:
            strategies: 策略列表（因子组合）
        """
        factor_list = list(frequent_factors.keys())
        strategies = []

        for n in range(min_factors, max_factors + 1):
            combos = list(combinations(factor_list, n))
            strategies.extend(combos)

        logger.info(f"\n枚举策略数: {len(strategies)}")
        logger.info(f"  因子数范围: {min_factors}-{max_factors}")
        logger.info(f"  候选因子数: {len(factor_list)}")

        return strategies

    def evaluate_strategy(
        self,
        strategy_factors: Tuple[str, ...],
        wfo_results: List[Dict],
        kpi_df: pd.DataFrame,
    ) -> Dict:
        """
        评估单个策略

        Args:
            strategy_factors: 策略因子组合
            wfo_results: WFO结果列表
            kpi_df: KPI DataFrame

        Returns:
            evaluation: 评估结果
        """
        # 找出使用这些因子的窗口
        matching_windows = []
        for result in wfo_results:
            selected = set(result.get("selected_factors", []))
            strategy_set = set(strategy_factors)

            # 策略因子是选中因子的子集
            if strategy_set.issubset(selected):
                matching_windows.append(result["window_index"])

        if not matching_windows:
            return None

        # 提取这些窗口的KPI
        window_kpi = kpi_df[kpi_df["window_index"].isin(matching_windows)]

        if len(window_kpi) == 0:
            return None

        # 计算平均KPI
        avg_return = window_kpi["annual_return"].mean()
        avg_sharpe = window_kpi["sharpe_ratio"].mean()
        avg_calmar = window_kpi["calmar_ratio"].mean()
        avg_ic = np.mean([wfo_results[i]["oos_ensemble_ic"] for i in matching_windows])

        # 计算稳定性（标准差的倒数）
        std_return = window_kpi["annual_return"].std()
        stability = 1 / (std_return + 0.01)  # 避免除零

        # 计算综合得分
        score = (
            self.score_weights["annual_return"] * avg_return
            + self.score_weights["sharpe_ratio"] * avg_sharpe
            + self.score_weights["calmar_ratio"] * avg_calmar
            + self.score_weights["ic"] * avg_ic
            + self.score_weights["stability"] * stability
        )

        evaluation = {
            "factors": strategy_factors,
            "n_factors": len(strategy_factors),
            "n_windows": len(matching_windows),
            "coverage": len(matching_windows) / len(wfo_results),
            "avg_annual_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_calmar": avg_calmar,
            "avg_ic": avg_ic,
            "stability": stability,
            "std_return": std_return,
            "score": score,
        }

        return evaluation

    def rank_strategies(
        self, wfo_results: List[Dict], kpi_df: pd.DataFrame, top_n: int = 5
    ) -> pd.DataFrame:
        """
        排序策略并选出Top-N

        Args:
            wfo_results: WFO结果列表
            kpi_df: KPI DataFrame
            top_n: 选出前N个策略

        Returns:
            top_strategies_df: Top-N策略DataFrame
        """
        logger.info("\n" + "=" * 70)
        logger.info("策略排序")
        logger.info("=" * 70)

        # 1. 提取高频因子
        frequent_factors = self.extract_frequent_factors(wfo_results)

        if len(frequent_factors) < 3:
            logger.warning("高频因子不足3个，无法枚举策略")
            return pd.DataFrame()

        # 2. 枚举策略
        strategies = self.enumerate_strategies(
            frequent_factors, min_factors=3, max_factors=min(5, len(frequent_factors))
        )

        # 3. 评估每个策略
        evaluations = []
        for strategy in strategies:
            eval_result = self.evaluate_strategy(strategy, wfo_results, kpi_df)
            if eval_result and eval_result["coverage"] >= 0.2:  # 至少覆盖20%窗口
                evaluations.append(eval_result)

        if not evaluations:
            logger.warning("没有找到有效策略")
            return pd.DataFrame()

        # 4. 转换为DataFrame并排序
        strategies_df = pd.DataFrame(evaluations)
        strategies_df = strategies_df.sort_values("score", ascending=False)

        # 5. 选出Top-N
        top_strategies_df = strategies_df.head(top_n).copy()
        top_strategies_df["rank"] = range(1, len(top_strategies_df) + 1)

        # 6. 输出Top-N
        logger.info(f"\nTop-{top_n} 策略:")
        logger.info("=" * 70)
        for _, row in top_strategies_df.iterrows():
            logger.info(f"\nRank {row['rank']}:")
            logger.info(f"  因子: {', '.join(row['factors'])}")
            logger.info(
                f"  覆盖窗口: {row['n_windows']}/{len(wfo_results)} ({row['coverage']:.1%})"
            )
            logger.info(f"  年化收益: {row['avg_annual_return']:.2%}")
            logger.info(f"  Sharpe: {row['avg_sharpe']:.2f}")
            logger.info(f"  Calmar: {row['avg_calmar']:.2f}")
            logger.info(f"  平均IC: {row['avg_ic']:.4f}")
            logger.info(f"  稳定性: {row['stability']:.2f}")
            logger.info(f"  综合得分: {row['score']:.4f}")
        logger.info("=" * 70)

        return top_strategies_df

    def save_top_strategies(self, top_strategies_df: pd.DataFrame, output_path: str):
        """
        保存Top-N策略

        Args:
            top_strategies_df: Top-N策略DataFrame
            output_path: 输出路径
        """
        # 转换factors为字符串（便于保存）
        df_to_save = top_strategies_df.copy()
        df_to_save["factors"] = df_to_save["factors"].apply(lambda x: "|".join(x))

        df_to_save.to_csv(output_path, index=False)
        logger.info(f"\nTop策略已保存: {output_path}")
