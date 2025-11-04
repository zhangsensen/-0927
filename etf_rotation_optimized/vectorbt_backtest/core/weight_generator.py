#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能权重生成器 - 高性能采样（Dirichlet + Sobol）
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import qmc  # Quasi-Monte Carlo 低差异序列


class SmartWeightGenerator:
    """智能权重生成器 - 全向量化，零循环"""

    def __init__(self, n_factors: int):
        self.n_factors = n_factors
        self.factor_usage_stats = np.zeros(n_factors)  # 因子使用统计

    def generate_dirichlet_weights(
        self,
        n_combinations: int,
        alpha: Optional[np.ndarray] = None,
        sparsity_bias: float = 1.0,
    ) -> np.ndarray:
        """
        Dirichlet 采样 - 一次性生成所有权重（零循环）

        Dirichlet 分布天然保证权重和为1，且支持稀疏性控制

        Args:
            n_combinations: 生成组合数
            alpha: Dirichlet 参数 (n_factors,)，控制稀疏性
                   - alpha 全为1: 均匀分布（所有因子同等概率）
                   - alpha 小于1: 稀疏权重（少数因子主导）
                   - alpha 大于1: 平滑权重（多因子均衡）
            sparsity_bias: 稀疏性偏置（< 1 更稀疏，> 1 更均匀）

        Returns:
            weight_matrix: (n_combinations, n_factors) 权重矩阵
        """
        if alpha is None:
            # 默认：轻微稀疏偏置
            alpha = np.ones(self.n_factors) * sparsity_bias

        # 一次性采样（零循环！）
        weight_matrix = np.random.dirichlet(alpha, size=n_combinations)

        # 更新因子使用统计
        self.factor_usage_stats += np.sum(weight_matrix > 0.01, axis=0)

        return weight_matrix

    def generate_sobol_weights(
        self, n_combinations: int, scramble: bool = True
    ) -> np.ndarray:
        """
        Sobol 低差异序列采样 - 系统覆盖权重空间

        比随机采样更均匀地探索权重空间，避免聚集

        Args:
            n_combinations: 生成组合数
            scramble: 是否随机化（增加随机性，避免规律性）

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        # Sobol 引擎（低差异序列生成器）
        sampler = qmc.Sobol(d=self.n_factors, scramble=scramble)

        # 生成 [0, 1]^n_factors 空间的低差异点
        raw_samples = sampler.random(n_combinations)

        # 归一化到单纯形（权重和为1）
        weight_matrix = raw_samples / raw_samples.sum(axis=1, keepdims=True)

        # 更新统计
        self.factor_usage_stats += np.sum(weight_matrix > 0.01, axis=0)

        return weight_matrix

    def generate_sparse_dirichlet_weights(
        self,
        n_combinations: int,
        min_active_factors: int = 1,
        max_active_factors: Optional[int] = None,
        alpha_active: float = 1.0,
    ) -> np.ndarray:
        """
        稀疏 Dirichlet 采样 - 只激活部分因子（零循环）

        Args:
            n_combinations: 生成组合数
            min_active_factors: 最少激活因子数
            max_active_factors: 最多激活因子数
            alpha_active: 激活因子的 Dirichlet 参数

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        if max_active_factors is None:
            max_active_factors = self.n_factors

        # 随机激活因子数量（向量化）
        n_active_array = np.random.randint(
            min_active_factors, max_active_factors + 1, size=n_combinations
        )

        weight_matrix = np.zeros((n_combinations, self.n_factors))

        # 批量生成（仍需循环，但已优化）
        for i in range(n_combinations):
            n_active = n_active_array[i]

            # 随机选择激活因子
            active_indices = np.random.choice(self.n_factors, n_active, replace=False)

            # 对激活因子做 Dirichlet 采样
            active_weights = np.random.dirichlet(np.ones(n_active) * alpha_active)

            # 填充
            weight_matrix[i, active_indices] = active_weights

        # 更新统计
        self.factor_usage_stats += np.sum(weight_matrix > 0.01, axis=0)

        return weight_matrix

    def generate_l1_projected_gaussian_weights(
        self, n_combinations: int, allow_negative: bool = False
    ) -> np.ndarray:
        """
        L1 投影高斯采样 - 支持正负权重（做空）

        生成高斯随机权重，然后投影到 L1 球面（||w||_1 = 1）

        Args:
            n_combinations: 生成组合数
            allow_negative: 是否允许负权重（做空因子）

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        # 高斯采样（零循环）
        raw_weights = np.random.randn(n_combinations, self.n_factors)

        if not allow_negative:
            # 只保留正权重
            raw_weights = np.abs(raw_weights)

        # L1 归一化：sum(|w|) = 1
        weight_matrix = raw_weights / np.abs(raw_weights).sum(axis=1, keepdims=True)

        # 更新统计
        self.factor_usage_stats += np.sum(np.abs(weight_matrix) > 0.01, axis=0)

        return weight_matrix

    def generate_mixed_strategy_weights(
        self,
        n_combinations: int,
        strategy_mix: Optional[dict] = None,
        ic_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        混合策略采样 - 组合多种采样方法

        Args:
            n_combinations: 总组合数
            strategy_mix: 策略分配比例
                例: {"dirichlet": 0.4, "sobol": 0.3, "sparse": 0.3}

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        if strategy_mix is None:
            strategy_mix = {
                "ic_weighted_dirichlet": 0.5,
                "sobol": 0.25,
                "sparse_dirichlet": 0.15,
                "correlation_adjusted": 0.1,
            }

        # 分配数量
        n_ic_weighted = int(
            n_combinations * strategy_mix.get("ic_weighted_dirichlet", 0)
        )
        n_sobol = int(n_combinations * strategy_mix.get("sobol", 0))
        n_sparse = int(n_combinations * strategy_mix.get("sparse_dirichlet", 0))
        n_correlation = int(
            n_combinations * strategy_mix.get("correlation_adjusted", 0)
        )
        n_fallback = n_combinations - n_ic_weighted - n_sobol - n_sparse - n_correlation

        # 如果fallback>0，分配给IC加权
        if n_fallback > 0:
            n_ic_weighted += n_fallback

        # 批量生成
        weights_list = []

        if n_ic_weighted > 0:
            weights_list.append(
                self.generate_ic_weighted_dirichlet_weights(
                    n_ic_weighted, ic_scores=ic_scores
                )
            )

        if n_sobol > 0:
            weights_list.append(self.generate_sobol_weights(n_sobol))

        if n_sparse > 0:
            weights_list.append(
                self.generate_sparse_dirichlet_weights(
                    n_sparse,
                    min_active_factors=1,
                    max_active_factors=self.n_factors // 2,
                )
            )

        if n_correlation > 0:
            weights_list.append(
                self.generate_correlation_adjusted_weights(n_correlation)
            )

        # 合并
        weight_matrix = np.vstack(weights_list)

        # 随机打乱（避免策略类型集中）
        np.random.shuffle(weight_matrix)

        return weight_matrix

    def get_factor_importance(self) -> np.ndarray:
        """
        获取因子重要性（基于使用频率）

        Returns:
            importance: (n_factors,) 归一化后的重要性分数
        """
        if self.factor_usage_stats.sum() == 0:
            return np.ones(self.n_factors) / self.n_factors

        return self.factor_usage_stats / self.factor_usage_stats.sum()

    def generate_gradient_weights(
        self,
        n_combinations: int,
        weight_resolution: int = 10,
        min_active_factors: int = 1,
        max_active_factors: int = None,
        allow_negative: bool = False,
    ) -> np.ndarray:
        """
        【保留旧接口兼容性】梯度权重（改用 Dirichlet 实现）
        """
        # 用稀疏 Dirichlet 替代原实现
        return self.generate_sparse_dirichlet_weights(
            n_combinations, min_active_factors, max_active_factors
        )

    def generate_sparse_weights(
        self, n_combinations: int, sparsity: float = 0.5
    ) -> np.ndarray:
        """
        【保留旧接口兼容性】稀疏权重（改用 L1 高斯实现）
        """
        weights = self.generate_l1_projected_gaussian_weights(n_combinations)

        # 稀疏化（随机置零）
        mask = np.random.rand(n_combinations, self.n_factors) > sparsity
        weights = weights * mask

        # 重新归一化
        weight_sums = np.abs(weights).sum(axis=1, keepdims=True)
        weight_sums[weight_sums == 0] = 1.0  # 避免除零
        weights = weights / weight_sums

        return weights

    def generate_hierarchical_weights(
        self, n_combinations: int, tier_structure: List[Tuple[List[int], float]] = None
    ) -> np.ndarray:
        """
        【保留旧接口兼容性】分层权重
        """
        if tier_structure is None:
            # 默认3层结构
            tier_structure = [
                (list(range(0, self.n_factors // 3)), 2.0),
                (list(range(self.n_factors // 3, 2 * self.n_factors // 3)), 1.0),
                (list(range(2 * self.n_factors // 3, self.n_factors)), 0.5),
            ]

        # 向量化实现
        weights_base = np.random.rand(n_combinations, self.n_factors)

        # 应用层级乘数
        for indices, multiplier in tier_structure:
            weights_base[:, indices] *= multiplier

        # 归一化
        weight_matrix = weights_base / weights_base.sum(axis=1, keepdims=True)

        return weight_matrix

    def generate_ic_weighted_dirichlet_weights(
        self, n_combinations: int, ic_scores: np.ndarray = None
    ) -> np.ndarray:
        """
        IC加权Dirichlet采样 - 基于因子有效性分配权重

        Args:
            n_combinations: 生成组合数
            ic_scores: 因子IC值数组，如未提供则使用均匀分布

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        if ic_scores is None:
            # 默认：使用历史统计作为代理IC；若尚无统计信息则均匀分布
            total_usage = float(self.factor_usage_stats.sum())
            if total_usage > 0:
                ic_scores = self.factor_usage_stats / total_usage
            else:
                ic_scores = np.ones(self.n_factors, dtype=float) / self.n_factors
        else:
            # 确保IC值非负且归一化
            ic_scores = np.abs(ic_scores)
            ic_scores = ic_scores / ic_scores.sum()

        # IC值作为Dirichlet的alpha参数（IC越高，权重越大概率）
        alpha = ic_scores * 10 + 0.1  # 缩放并添加平滑项

        # Dirichlet采样
        weight_matrix = np.random.dirichlet(alpha, size=n_combinations)

        # 更新统计
        self.factor_usage_stats += np.sum(weight_matrix > 0.01, axis=0)

        return weight_matrix

    def generate_correlation_adjusted_weights(
        self, n_combinations: int, factor_correlations: np.ndarray = None
    ) -> np.ndarray:
        """
        相关性调整权重 - 降低高相关性因子的权重

        Args:
            n_combinations: 生成组合数
            factor_correlations: 因子相关性矩阵

        Returns:
            weight_matrix: (n_combinations, n_factors)
        """
        # 生成基础权重
        base_weights = self.generate_dirichlet_weights(n_combinations)

        if factor_correlations is not None:
            # 相关性调整：对高相关性因子降低权重
            for i in range(n_combinations):
                weights = base_weights[i]

                # 计算每个因子的相关性惩罚
                correlation_penalty = np.zeros(self.n_factors)
                for j in range(self.n_factors):
                    if weights[j] > 0:
                        # 与该因子相关性>0.7的其他因子
                        high_corr_mask = factor_correlations[j] > 0.7
                        high_corr_mask[j] = False  # 排除自身

                        # 相关性惩罚：权重与相关性乘积
                        penalty = np.sum(
                            weights[high_corr_mask]
                            * factor_correlations[j, high_corr_mask]
                        )
                        correlation_penalty[j] = penalty

                # 应用惩罚：权重 * (1 - 惩罚)
                adjusted_weights = weights * (1 - correlation_penalty)

                # 重新归一化
                if adjusted_weights.sum() > 0:
                    base_weights[i] = adjusted_weights / adjusted_weights.sum()

        return base_weights

    def generate_grid_weights(self, grid_points: int = 5) -> np.ndarray:
        """
        【保留旧接口兼容性】网格权重
        """
        # 为前几个因子生成网格，其余因子权重为0
        n_grid_factors = min(3, self.n_factors)

        # 生成网格点
        grid_values = np.linspace(0, 1, grid_points)

        weight_matrix = []

        # 生成所有网格组合
        import itertools

        for grid_weights in itertools.product(grid_values, repeat=n_grid_factors):
            if sum(grid_weights) == 0:
                continue

            # 归一化
            normalized = np.array(grid_weights) / sum(grid_weights)

            # 填充完整权重向量
            full_weights = np.zeros(self.n_factors)
            full_weights[:n_grid_factors] = normalized

            weight_matrix.append(full_weights)

        return np.array(weight_matrix)
