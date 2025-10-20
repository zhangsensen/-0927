#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优化的权重组合生成器
通过智能采样和数学优化，减少无效组合，提升搜索效率
"""

import itertools
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class SearchStrategy(Enum):
    """搜索策略枚举"""
    GRID = "grid"                    # 网格搜索
    SMART = "smart"                  # 智能采样
    HIERARCHICAL = "hierarchical"    # 分层搜索
    EVOLUTIONARY = "evolutionary"    # 进化算法


@dataclass
class WeightGenerationConfig:
    """权重生成配置"""
    strategy: SearchStrategy = SearchStrategy.SMART
    weight_grid: List[float] = None
    weight_sum_range: Tuple[float, float] = (0.7, 1.3)
    max_combinations: int = 5000
    diversity_threshold: float = 0.1
    convergence_threshold: float = 0.01
    max_iterations: int = 100


class OptimizedWeightGenerator:
    """优化的权重组合生成器"""

    def __init__(self, config: WeightGenerationConfig = None):
        """
        初始化权重生成器

        Args:
            config: 权重生成配置
        """
        self.config = config or WeightGenerationConfig()
        self.logger = logging.getLogger(__name__)

        if self.config.weight_grid is None:
            # 默认权重网格，更密集的分布
            self.config.weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.logger.info(f"初始化权重生成器: 策略={self.config.strategy.value}")

    def generate_grid_weights(self, factors: List[str]) -> List[Tuple[float, ...]]:
        """网格搜索权重生成"""
        self.logger.info("使用网格搜索生成权重组合...")

        # 生成所有可能的权重组合
        weight_combos = list(itertools.product(self.config.weight_grid, repeat=len(factors)))
        self.logger.info(f"理论组合数: {len(weight_combos):,}")

        # 向量化过滤有效组合
        weight_array = np.array(weight_combos)
        weight_sums = np.sum(weight_array, axis=1)

        # 更严格的过滤条件
        valid_mask = (
            (weight_sums >= self.config.weight_sum_range[0]) &
            (weight_sums <= self.config.weight_sum_range[1]) &
            (weight_sums > 0.1)  # 权重和不能太小
        )
        valid_indices = np.where(valid_mask)[0]

        # 进一步优化：过滤掉权重过于分散的组合
        filtered_indices = []
        for idx in valid_indices:
            weights = weight_combos[idx]
            # 计算权重分散度
            non_zero_weights = [w for w in weights if w > 0.01]
            if len(non_zero_weights) >= 2:  # 至少两个有效权重
                max_weight = max(non_zero_weights)
                min_weight = min(non_zero_weights)
                if max_weight - min_weight <= 0.8:  # 权重差异不过大
                    filtered_indices.append(idx)

        # 限制组合数
        if len(filtered_indices) > self.config.max_combinations:
            # 智能采样：优先选择权重分布更均匀的组合
            filtered_indices = self._smart_sampling(weight_combos, filtered_indices)

        valid_combos = [weight_combos[i] for i in filtered_indices]
        self.logger.info(f"有效组合数: {len(valid_combos):,}")

        return valid_combos

    def generate_smart_weights(self, factors: List[str]) -> List[Tuple[float, ...]]:
        """智能采样权重生成"""
        self.logger.info("使用智能采样生成权重组合...")

        n_factors = len(factors)
        valid_combos = []

        # 1. 生成均匀分布的组合
        uniform_combos = self._generate_uniform_combinations(n_factors)
        valid_combos.extend(uniform_combos)

        # 2. 生成集中分布的组合（少数因子主导）
        concentrated_combos = self._generate_concentrated_combinations(n_factors)
        valid_combos.extend(concentrated_combos)

        # 3. 生成梯度分布的组合
        gradient_combos = self._generate_gradient_combinations(n_factors)
        valid_combos.extend(gradient_combos)

        # 4. 基于经验的优质组合
        empirical_combos = self._generate_empirical_combinations(n_factors)
        valid_combos.extend(empirical_combos)

        # 去重和限制数量
        unique_combos = list(set(valid_combos))

        if len(unique_combos) > self.config.max_combinations:
            # 按多样性排序
            unique_combos = self._rank_by_diversity(unique_combos)
            unique_combos = unique_combos[:self.config.max_combinations]

        self.logger.info(f"智能采样生成: {len(unique_combos):,}个组合")
        return unique_combos

    def generate_hierarchical_weights(self, factors: List[str]) -> List[Tuple[float, ...]]:
        """分层搜索权重生成"""
        self.logger.info("使用分层搜索生成权重组合...")

        n_factors = len(factors)
        valid_combos = []

        # 第一层：粗粒度搜索
        coarse_grid = [0.0, 0.5, 1.0]
        coarse_combos = self._search_layer(factors, coarse_grid, "coarse")
        valid_combos.extend(coarse_combos)

        # 第二层：在粗粒度最优解周围细粒度搜索
        if coarse_combos:
            best_coarse = max(coarse_combos, key=lambda x: self._evaluate_weights(x))
            refined_combos = self._refine_around_best(factors, best_coarse)
            valid_combos.extend(refined_combos)

        # 第三层：随机探索
        random_combos = self._generate_random_combinations(n_factors, 100)
        valid_combos.extend(random_combos)

        # 限制数量
        if len(valid_combos) > self.config.max_combinations:
            valid_combos = valid_combos[:self.config.max_combinations]

        self.logger.info(f"分层搜索生成: {len(valid_combos):,}个组合")
        return valid_combos

    def generate_evolutionary_weights(self, factors: List[str]) -> List[Tuple[float, ...]]:
        """进化算法权重生成"""
        self.logger.info("使用进化算法生成权重组合...")

        n_factors = len(factors)
        population_size = min(100, self.config.max_combinations // 10)

        # 1. 初始种群
        population = self._initialize_population(n_factors, population_size)

        best_combos = []
        convergence_history = []

        for generation in range(self.config.max_iterations):
            # 2. 评估适应度
            fitness_scores = [self._evaluate_weights(ind) for ind in population]

            # 3. 选择
            selected = self._selection(population, fitness_scores)

            # 4. 交叉
            offspring = self._crossover(selected, n_factors)

            # 5. 变异
            mutated = self._mutation(offspring)

            # 6. 新种群
            population = selected + offspring + mutated
            population = population[:population_size * 2]  # 限制种群大小

            # 记录最优个体
            best_idx = np.argmax(fitness_scores)
            best_individual = population[best_idx]
            best_combos.append(best_individual)

            # 检查收敛
            avg_fitness = np.mean(fitness_scores)
            convergence_history.append(avg_fitness)

            if len(convergence_history) > 10:
                recent_improvement = abs(convergence_history[-1] - convergence_history[-10])
                if recent_improvement < self.config.convergence_threshold:
                    self.logger.info(f"进化算法在第{generation}代收敛")
                    break

        # 去重和限制数量
        unique_combos = list(set(best_combos))
        if len(unique_combos) > self.config.max_combinations:
            unique_combos = unique_combos[:self.config.max_combinations]

        self.logger.info(f"进化算法生成: {len(unique_combos):,}个组合")
        return unique_combos

    def generate_weights(self, factors: List[str]) -> List[Tuple[float, ...]]:
        """统一的权重生成接口"""
        if self.config.strategy == SearchStrategy.GRID:
            return self.generate_grid_weights(factors)
        elif self.config.strategy == SearchStrategy.SMART:
            return self.generate_smart_weights(factors)
        elif self.config.strategy == SearchStrategy.HIERARCHICAL:
            return self.generate_hierarchical_weights(factors)
        elif self.config.strategy == SearchStrategy.EVOLUTIONARY:
            return self.generate_evolutionary_weights(factors)
        else:
            raise ValueError(f"未支持的搜索策略: {self.config.strategy}")

    def _generate_uniform_combinations(self, n_factors: int) -> List[Tuple[float, ...]]:
        """生成均匀分布的权重组合"""
        combos = []

        # 所有因子等权重
        equal_weight = 1.0 / n_factors
        combos.append(tuple([equal_weight] * n_factors))

        # 部分因子等权重
        for k in range(1, n_factors + 1):
            weight = 1.0 / k
            combo = [weight if i < k else 0.0 for i in range(n_factors)]
            combos.append(tuple(combo))

        return combos

    def _generate_concentrated_combinations(self, n_factors: int) -> List[Tuple[float, ...]]:
        """生成集中分布的权重组合（少数因子主导）"""
        combos = []

        # 单个因子主导
        for i in range(n_factors):
            combo = [0.0] * n_factors
            combo[i] = 1.0
            combos.append(tuple(combo))

        # 两个因子主导
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                combo = [0.0] * n_factors
                combo[i] = 0.6
                combo[j] = 0.4
                combos.append(tuple(combo))
                combo = [0.0] * n_factors
                combo[i] = 0.4
                combo[j] = 0.6
                combos.append(tuple(combo))

        return combos

    def _generate_gradient_combinations(self, n_factors: int) -> List[Tuple[float, ...]]:
        """生成梯度分布的权重组合"""
        combos = []

        # 线性递减
        for start_weight in [0.8, 0.6, 0.4]:
            weights = []
            remaining = 1.0
            for i in range(n_factors):
                if i == n_factors - 1:
                    weights.append(remaining)
                else:
                    w = min(start_weight * (0.7 ** i), remaining)
                    weights.append(w)
                    remaining -= w
            combos.append(tuple(weights))

        # 线性递增
        for start_weight in [0.2, 0.1, 0.05]:
            weights = []
            remaining = 1.0
            for i in range(n_factors):
                if i == n_factors - 1:
                    weights.append(remaining)
                else:
                    w = min(start_weight * (1.5 ** i), remaining)
                    weights.append(w)
                    remaining -= w
            combos.append(tuple(weights))

        return combos

    def _generate_empirical_combinations(self, n_factors: int) -> List[Tuple[float, ...]]:
        """生成基于经验的优质组合"""
        combos = []

        # 经典的 60/40 组合
        if n_factors >= 2:
            combo = [0.0] * n_factors
            combo[0] = 0.6
            combo[1] = 0.4
            combos.append(tuple(combo))

        # 三因子组合
        if n_factors >= 3:
            combos.extend([
                (0.5, 0.3, 0.2) + (0.0,) * (n_factors - 3),
                (0.4, 0.3, 0.3) + (0.0,) * (n_factors - 3),
                (0.6, 0.2, 0.2) + (0.0,) * (n_factors - 3),
            ])

        # 多因子分散组合
        if n_factors >= 5:
            base_weight = 0.2
            combo = tuple([base_weight] * 5 + (0.0,) * (n_factors - 5))
            combos.append(combo)

        return combos

    def _generate_random_combinations(self, n_factors: int, count: int) -> List[Tuple[float, ...]]:
        """生成随机权重组合"""
        combos = []
        np.random.seed(42)  # 确保可重现

        for _ in range(count):
            # Dirichlet分布生成随机权重
            weights = np.random.dirichlet(np.ones(n_factors))

            # 模拟离散化到网格
            discrete_weights = []
            for w in weights:
                # 找到最接近的网格点
                closest = min(self.config.weight_grid, key=lambda x: abs(x - w))
                discrete_weights.append(closest)

            # 归一化到权重和为1
            total = sum(discrete_weights)
            if total > 0:
                discrete_weights = [w / total for w in discrete_weights]
                combos.append(tuple(discrete_weights))

        return combos

    def _smart_sampling(self, weight_combos: List[Tuple], indices: List[int]) -> List[int]:
        """智能采样：选择多样性最高的组合"""
        if len(indices) <= self.config.max_combinations:
            return indices

        selected = []
        remaining = indices.copy()

        # 选择第一个（通常是权重和最接近1的组合）
        selected.append(remaining.pop(0))

        # 贪心算法选择最不相似的组合
        while len(selected) < self.config.max_combinations and remaining:
            best_idx = None
            best_distance = -1

            for idx in remaining:
                candidate = weight_combos[idx]
                # 计算与已选组合的最小距离
                min_distance = min(self._calculate_distance(candidate, weight_combos[s])
                                 for s in selected)

                if min_distance > best_distance:
                    best_distance = min_distance
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break

        return selected

    def _calculate_distance(self, weights1: Tuple, weights2: Tuple) -> float:
        """计算权重组合之间的距离"""
        return sum(abs(w1 - w2) for w1, w2 in zip(weights1, weights2))

    def _rank_by_diversity(self, combos: List[Tuple]) -> List[Tuple]:
        """按多样性排序权重组合"""
        if not combos:
            return combos

        # 计算每个组合的多样性得分
        diversity_scores = []
        for i, combo in enumerate(combos):
            # 计算与其他组合的平均距离
            distances = [self._calculate_distance(combo, other)
                        for j, other in enumerate(combos) if i != j]
            diversity = np.mean(distances) if distances else 0
            diversity_scores.append(diversity)

        # 按多样性排序
        sorted_indices = np.argsort(diversity_scores)[::-1]  # 降序
        return [combos[i] for i in sorted_indices]

    def _evaluate_weights(self, weights: Tuple) -> float:
        """评估权重组合的适应度（简化版本）"""
        # 这里使用简化的启发式评估
        # 实际应用中应该基于历史回测结果

        # 1. 权重和接近1
        weight_sum = sum(weights)
        sum_score = 1.0 - abs(weight_sum - 1.0)

        # 2. 权重分布不过度集中
        non_zero_weights = [w for w in weights if w > 0.01]
        if non_zero_weights:
            max_weight = max(non_zero_weights)
            distribution_score = 1.0 - max_weight
        else:
            distribution_score = 0.0

        # 3. 有效因子数量
        factor_count_score = len(non_zero_weights) / len(weights)

        # 综合得分
        total_score = 0.4 * sum_score + 0.4 * distribution_score + 0.2 * factor_count_score
        return total_score

    def _initialize_population(self, n_factors: int, size: int) -> List[Tuple]:
        """初始化进化算法种群"""
        population = []

        # 混合不同策略生成初始种群
        population.extend(self._generate_uniform_combinations(n_factors))
        population.extend(self._generate_concentrated_combinations(n_factors))
        population.extend(self._generate_random_combinations(n_factors, size // 2))

        # 补充到指定数量
        while len(population) < size:
            random_combo = tuple(np.random.dirichlet(np.ones(n_factors)))
            population.append(random_combo)

        return population[:size]

    def _selection(self, population: List[Tuple], fitness_scores: List[float]) -> List[Tuple]:
        """选择操作：锦标赛选择"""
        tournament_size = 3
        selected = []

        for _ in range(len(population) // 2):
            # 随机选择tournament_size个个体
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]

            # 选择适应度最高的
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])

        return selected

    def _crossover(self, parents: List[Tuple], n_factors: int) -> List[Tuple]:
        """交叉操作"""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]

            # 单点交叉
            cross_point = np.random.randint(1, n_factors)
            child1 = parent1[:cross_point] + parent2[cross_point:]
            child2 = parent2[:cross_point] + parent1[cross_point:]

            # 归一化
            child1 = self._normalize_weights(child1)
            child2 = self._normalize_weights(child2)

            offspring.extend([child1, child2])

        return offspring

    def _mutation(self, individuals: List[Tuple]) -> List[Tuple]:
        """变异操作"""
        mutated = []

        for individual in individuals:
            if np.random.random() < 0.1:  # 10%变异概率
                # 随机选择一个位置进行变异
                idx = np.random.randint(len(individual))
                # 在网格值中选择新的权重
                new_weight = np.random.choice(self.config.weight_grid)

                mutated_list = list(individual)
                mutated_list[idx] = new_weight

                # 归一化
                normalized = self._normalize_weights(mutated_list)
                mutated.append(normalized)
            else:
                mutated.append(individual)

        return mutated

    def _normalize_weights(self, weights: List[float]) -> Tuple[float, ...]:
        """归一化权重使得和为1"""
        total = sum(weights)
        if total > 0:
            normalized = [w / total for w in weights]
        else:
            # 如果全为0，则均匀分布
            normalized = [1.0 / len(weights)] * len(weights)
        return tuple(normalized)

    def _search_layer(self, factors: List[str], grid: List[float], layer_name: str) -> List[Tuple]:
        """分层搜索的一层"""
        combos = list(itertools.product(grid, repeat=len(factors)))

        # 过滤有效组合
        valid_combos = []
        for combo in combos:
            weight_sum = sum(combo)
            if self.config.weight_sum_range[0] <= weight_sum <= self.config.weight_sum_range[1]:
                valid_combos.append(combo)

        self.logger.info(f"{layer_name}层搜索: {len(valid_combos)}个组合")
        return valid_combos[:self.config.max_combinations // 3]

    def _refine_around_best(self, factors: List[str], best_combo: Tuple) -> List[Tuple]:
        """在最优解周围进行精细搜索"""
        refined = []

        # 在最优解周围的小范围内搜索
        for i, weight in enumerate(best_combo):
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                new_weight = max(0.0, min(1.0, weight + delta))
                # 找到最近的网格点
                closest = min(self.config.weight_grid, key=lambda x: abs(x - new_weight))

                refined_combo = list(best_combo)
                refined_combo[i] = closest

                # 归一化
                normalized = self._normalize_weights(refined_combo)
                refined.append(normalized)

        return list(set(refined))  # 去重


def test_weight_generator():
    """测试权重生成器"""
    print("测试优化权重生成器")
    print("=" * 50)

    factors = ['RSI', 'MACD', 'STOCH', 'MA']

    # 测试不同策略
    strategies = [
        SearchStrategy.GRID,
        SearchStrategy.SMART,
        SearchStrategy.HIERARCHICAL,
        SearchStrategy.EVOLUTIONARY
    ]

    for strategy in strategies:
        print(f"\n🧪 测试策略: {strategy.value}")

        config = WeightGenerationConfig(
            strategy=strategy,
            max_combinations=100
        )

        generator = OptimizedWeightGenerator(config)
        weights = generator.generate_weights(factors)

        print(f"  生成组合数: {len(weights)}")
        if weights:
            print(f"  示例组合: {weights[0]}")
            avg_sum = np.mean([sum(w) for w in weights])
            print(f"  平均权重和: {avg_sum:.3f}")


if __name__ == "__main__":
    test_weight_generator()