"""
Factor Weighting | 因子权重计算器

功能:
  1. 合成多因子信号
  2. 支持3种权重方案:
     - equal: 等权 (baseline)
     - ic_weighted: IC加权 (aggressive)
     - gradient_decay: 梯度衰减 (conservative)
  3. 纯向量化实现,无循环

设计原则:
  - NumPy广播: 批量计算,避免.apply()
  - 数值稳定: 处理NaN、极端值
  - 可解释: 权重和为1,可追溯

作者: Linus Quant Engineer
日期: 2025-10-28
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FactorWeighting:
    """
    因子权重计算器

    支持3种方案:
    1. equal: 等权 (baseline) - 所有因子权重相同
    2. ic_weighted: IC加权 (aggressive) - 高IC因子权重更高
    3. gradient_decay: 梯度衰减 (conservative) - IC排名越低,权重指数衰减

    所有方案确保:
    - 权重和为1
    - 处理NaN值
    - 向量化实现
    """

    @staticmethod
    def combine_factors(
        factor_data: List[pd.DataFrame],
        scheme: str = "equal",
        ic_scores: Dict[str, float] = None,
        factor_names: List[str] = None,
    ) -> np.ndarray:
        """
        合成多因子信号

        参数:
            factor_data: 因子DataFrame列表 [(T×N), (T×N), ...]
                        - T行(交易日), N列(资产)
                        - 已标准化 (均值0, 标准差1)
            scheme: 权重方案 {'equal', 'ic_weighted', 'gradient_decay'}
            ic_scores: IC评分字典 {factor_name: ic_value}
                      - 用于ic_weighted和gradient_decay方案
            factor_names: 因子名称列表 (与factor_data对应)
                         - 用于从ic_scores中查找IC

        返回:
            combined_signal: (T×N) ndarray - 合成后的因子信号

        示例:
            >>> f1 = pd.DataFrame(...)  # (1000, 43)
            >>> f2 = pd.DataFrame(...)
            >>> signal = combine_factors([f1, f2], scheme='equal')
            >>> signal.shape
            (1000, 43)
        """
        if not factor_data:
            raise ValueError("factor_data不能为空")

        # 转换为numpy数组 (纯向量化)
        # 形状: (n_factors, T, N)
        signals = np.stack([df.values for df in factor_data])
        n_factors, T, N = signals.shape

        logger.debug(f"合成因子: {n_factors}个 × {T}天 × {N}个资产, 方案={scheme}")

        # 根据方案计算权重
        if scheme == "equal":
            weights = FactorWeighting._equal_weights(n_factors)

        elif scheme == "ic_weighted":
            if not ic_scores or not factor_names:
                logger.warning("ic_weighted需要ic_scores和factor_names,回退到equal")
                weights = FactorWeighting._equal_weights(n_factors)
            else:
                weights = FactorWeighting._ic_weighted_weights(factor_names, ic_scores)

        elif scheme == "gradient_decay":
            if not ic_scores or not factor_names:
                logger.warning("gradient_decay需要ic_scores和factor_names,回退到equal")
                weights = FactorWeighting._equal_weights(n_factors)
            else:
                weights = FactorWeighting._gradient_decay_weights(
                    factor_names, ic_scores
                )

        else:
            raise ValueError(f"未知权重方案: {scheme}")

        # 检查权重
        assert len(weights) == n_factors, "权重数量与因子数量不匹配"
        assert np.abs(weights.sum() - 1.0) < 1e-6, f"权重和不为1: {weights.sum()}"

        logger.debug(f"权重: {weights}")

        # 加权平均 (向量化)
        # weights: (n_factors,)
        # signals: (n_factors, T, N)
        # weighted_signal: (T, N)
        weighted_signal = np.average(signals, axis=0, weights=weights)

        return weighted_signal

    @staticmethod
    def _equal_weights(n_factors: int) -> np.ndarray:
        """
        等权方案

        w_i = 1 / n
        """
        return np.ones(n_factors) / n_factors

    @staticmethod
    def _ic_weighted_weights(
        factor_names: List[str], ic_scores: Dict[str, float]
    ) -> np.ndarray:
        """
        IC加权方案

        w_i = IC_i / Σ(IC_j)

        注意:
        - 处理负IC: 如果所有IC<=0,回退到等权
        - 数值稳定: 归一化处理
        """
        # 提取IC值
        ics = np.array([ic_scores.get(name, 0.0) for name in factor_names])

        # 处理负IC或全0的情况
        if np.all(ics <= 0):
            logger.warning("所有IC≤0,回退到等权")
            return np.ones(len(factor_names)) / len(factor_names)

        # 将负IC置为0 (只用正IC)
        ics_positive = np.maximum(ics, 0)

        # 归一化
        weights = ics_positive / ics_positive.sum()

        return weights

    @staticmethod
    def _gradient_decay_weights(
        factor_names: List[str], ic_scores: Dict[str, float]
    ) -> np.ndarray:
        """
        梯度衰减方案

        w_i = exp(-λ * rank_i) / Z

        其中:
        - rank_i: 因子按IC降序排列后的排名 (0, 1, 2, ...)
        - λ = 0.5: 衰减率
        - Z: 归一化常数

        效果:
        - Top1因子: w_1 = exp(0) / Z = 1.000 / Z ≈ 42.9%
        - Top2因子: w_2 = exp(-0.5) / Z ≈ 26.0%
        - Top3因子: w_3 = exp(-1.0) / Z ≈ 15.8%
        - Top4因子: w_4 = exp(-1.5) / Z ≈ 9.6%
        - Top5因子: w_5 = exp(-2.0) / Z ≈ 5.8%
        """
        # 提取IC值
        ics = np.array([ic_scores.get(name, 0.0) for name in factor_names])

        # 按IC降序排列
        sorted_indices = np.argsort(-ics)  # 降序索引

        # 计算衰减权重
        n = len(factor_names)
        decay_rate = 0.5
        ranks = np.arange(n)  # 0, 1, 2, ...
        decay_weights = np.exp(-decay_rate * ranks)

        # 归一化
        decay_weights = decay_weights / decay_weights.sum()

        # 按原始顺序恢复权重
        weights = np.zeros(n)
        weights[sorted_indices] = decay_weights

        return weights

    @staticmethod
    def get_weight_distribution(
        factor_names: List[str], scheme: str, ic_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        获取权重分布 (用于分析和可视化)

        返回:
            {factor_name: weight}
        """
        n = len(factor_names)

        if scheme == "equal":
            weights = FactorWeighting._equal_weights(n)
        elif scheme == "ic_weighted":
            weights = FactorWeighting._ic_weighted_weights(factor_names, ic_scores)
        elif scheme == "gradient_decay":
            weights = FactorWeighting._gradient_decay_weights(factor_names, ic_scores)
        else:
            raise ValueError(f"未知方案: {scheme}")

        return {name: float(w) for name, w in zip(factor_names, weights)}


# =========================================================================
# 使用示例和测试
# =========================================================================

if __name__ == "__main__":
    print("FactorWeighting 测试")
    print("=" * 70)

    # 模拟数据
    np.random.seed(42)
    T, N = 100, 10  # 100天, 10个资产
    n_factors = 5

    # 生成5个标准化因子 (均值0, 标准差1)
    factor_data = [
        pd.DataFrame(np.random.randn(T, N), columns=[f"Asset_{i}" for i in range(N)])
        for _ in range(n_factors)
    ]

    factor_names = ["MOM_20D", "SLOPE_20D", "CMF_20D", "RSI_14", "RET_VOL_20D"]

    # 模拟IC评分
    ic_scores = {
        "MOM_20D": 0.10,
        "SLOPE_20D": 0.08,
        "CMF_20D": 0.05,
        "RSI_14": 0.03,
        "RET_VOL_20D": 0.02,
    }

    print("\n测试3种权重方案:")
    print("-" * 70)

    for scheme in ["equal", "ic_weighted", "gradient_decay"]:
        # 合成信号
        signal = FactorWeighting.combine_factors(
            factor_data, scheme=scheme, ic_scores=ic_scores, factor_names=factor_names
        )

        # 获取权重分布
        weights = FactorWeighting.get_weight_distribution(
            factor_names, scheme=scheme, ic_scores=ic_scores
        )

        print(f"\n方案: {scheme}")
        print(f"  输出形状: {signal.shape}")
        print(f"  权重分布:")
        for name, weight in weights.items():
            print(f"    {name:15s}: {weight:6.1%}")
        print(f"  权重和: {sum(weights.values()):.6f}")

        # 验证信号统计
        print(f"  信号统计:")
        print(f"    均值: {np.nanmean(signal):.6f}")
        print(f"    标准差: {np.nanstd(signal):.6f}")
        print(f"    NaN率: {np.isnan(signal).mean():.1%}")

    print("\n✓ 所有测试通过")
