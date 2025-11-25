#!/usr/bin/env python3
"""
ETF轮动策略优化器
基于筛选结果进行精细化权重优化
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """优化配置"""

    # 搜索参数
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # 并行参数
    n_workers: int = 8
    chunk_size: int = 20

    # 约束参数
    weight_sum_tolerance: float = 0.01
    min_weight: float = 0.001
    max_weight: float = 0.7
    min_factors: int = 2
    max_factors: int = 5

    # 目标函数权重
    sharpe_weight: float = 0.6
    return_weight: float = 0.3
    drawdown_weight: float = 0.1

    # 高级选项
    enable_adaptive_search: bool = True
    enable_local_search: bool = True
    local_search_radius: float = 0.05

    # 可复现性
    random_seed: Optional[int] = 42


class StrategyOptimizer:
    """策略优化器"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        初始化优化器

        Args:
            config: 优化配置
        """
        self.config = config or OptimizationConfig()
        self.screening_results = None
        self.factor_universe = []
        self.optimization_history = []
        self.best_solution = None

        # 设置随机种子以确保可复现性
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            logger.info(f"随机种子已设置: {self.config.random_seed}")

    def load_screening_results(self, screening_path: str) -> bool:
        """加载筛选结果"""
        try:
            with open(screening_path, "r", encoding="utf-8") as f:
                self.screening_results = json.load(f)

            # 提取因子集合
            screened_strategies = self.screening_results.get("screened_strategies", [])
            if screened_strategies:
                all_factors = set()
                for strategy in screened_strategies:
                    all_factors.update(strategy["weights"].keys())
                self.factor_universe = sorted(list(all_factors))

            logger.info(f"成功加载筛选结果，因子集合: {self.factor_universe}")
            return True

        except Exception as e:
            logger.error(f"加载筛选结果失败: {e}")
            return False

    def objective_function(self, weights: np.ndarray, top_n: int = 3) -> float:
        """目标函数 - 综合评分"""
        # 转换为权重字典
        weight_dict = {}
        for i, factor in enumerate(self.factor_universe):
            if i < len(weights) and weights[i] > self.config.min_weight:
                weight_dict[factor] = weights[i]

        # 评估策略表现
        performance = self._evaluate_weights(weight_dict, top_n)

        # 综合评分
        score = (
            self.config.sharpe_weight * performance["sharpe_ratio"]
            + self.config.return_weight * (performance["total_return"] / 100)
            + self.config.drawdown_weight * (performance["max_drawdown"] / -100)
        )

        return score

    def _evaluate_weights(self, weights: Dict, top_n: int) -> Dict:
        """评估权重表现"""
        # 基于筛选结果模拟性能
        if not self.screening_results:
            # 默认性能
            return {
                "sharpe_ratio": 0.45,
                "total_return": 45.0,
                "max_drawdown": -35.0,
                "volatility": 15.0,
            }

        # 找到相似的策略作为参考
        screened_strategies = self.screening_results.get("screened_strategies", [])
        similar_strategies = self._find_similar_strategies(weights, screened_strategies)

        if similar_strategies:
            # 基于相似策略估算性能
            return self._estimate_performance_from_similar(weights, similar_strategies)
        else:
            # 基于权重特性估算性能
            return self._estimate_performance_from_weights(weights)

    def _find_similar_strategies(
        self, target_weights: Dict, strategies: List[Dict], top_k: int = 5
    ) -> List[Dict]:
        """找到相似的策略"""
        similarities = []

        for strategy in strategies:
            strategy_weights = strategy.get("weights", {})
            similarity = self._calculate_weight_similarity(
                target_weights, strategy_weights
            )
            similarities.append((similarity, strategy))

        # 按相似度排序，返回最相似的几个
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [strategy for _, strategy in similarities[:top_k]]

    def _calculate_weight_similarity(self, weights1: Dict, weights2: Dict) -> float:
        """计算权重相似度"""
        all_factors = set(weights1.keys()) | set(weights2.keys())

        if not all_factors:
            return 0.0

        # 计算余弦相似度
        vec1 = [weights1.get(f, 0) for f in all_factors]
        vec2 = [weights2.get(f, 0) for f in all_factors]

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _estimate_performance_from_similar(
        self, target_weights: Dict, similar_strategies: List[Dict]
    ) -> Dict:
        """基于相似策略估算性能"""
        if not similar_strategies:
            return self._estimate_performance_from_weights(target_weights)

        # 加权平均相似策略的性能
        total_similarity = 0
        weighted_performance = {
            "sharpe_ratio": 0,
            "total_return": 0,
            "max_drawdown": 0,
            "volatility": 0,
        }

        for strategy in similar_strategies:
            strategy_weights = strategy.get("weights", {})
            similarity = self._calculate_weight_similarity(
                target_weights, strategy_weights
            )

            if similarity > 0:
                weight = similarity
                weighted_performance["sharpe_ratio"] += (
                    weight * strategy["sharpe_ratio"]
                )
                weighted_performance["total_return"] += (
                    weight * strategy["total_return"]
                )
                weighted_performance["max_drawdown"] += (
                    weight * strategy["max_drawdown"]
                )
                weighted_performance["volatility"] += weight * strategy.get(
                    "volatility", 15.0
                )
                total_similarity += weight

        if total_similarity > 0:
            for key in weighted_performance:
                weighted_performance[key] /= total_similarity

        return weighted_performance

    def _estimate_performance_from_weights(self, weights: Dict) -> Dict:
        """基于权重特性估算性能"""
        # 基础性能
        base_performance = {
            "sharpe_ratio": 0.45,
            "total_return": 45.0,
            "max_drawdown": -35.0,
            "volatility": 15.0,
        }

        # 根据权重分布调整性能
        total_weight = sum(weights.values())
        effective_factors = sum(1 for w in weights.values() if w > 0.01)

        # 权重集中度影响
        if effective_factors > 0:
            weight_concentration = max(weights.values()) if weights else 0
            concentration_bonus = 0.02 if weight_concentration > 0.4 else -0.01
            base_performance["sharpe_ratio"] += concentration_bonus

        # 因子数量影响
        if 2 <= effective_factors <= 4:
            base_performance["sharpe_ratio"] += 0.01
        elif effective_factors > 4:
            base_performance["sharpe_ratio"] -= 0.01

        # 添加随机性
        base_performance["sharpe_ratio"] += np.random.normal(0, 0.02)
        base_performance["total_return"] += np.random.normal(0, 5)
        base_performance["max_drawdown"] += np.random.normal(0, 3)

        # 确保合理范围
        base_performance["sharpe_ratio"] = max(
            0.2, min(0.8, base_performance["sharpe_ratio"])
        )
        base_performance["total_return"] = max(
            10, min(120, base_performance["total_return"])
        )
        base_performance["max_drawdown"] = max(
            -70, min(-15, base_performance["max_drawdown"])
        )

        return base_performance

    def generate_initial_population(self, size: int) -> np.ndarray:
        """生成初始种群"""
        population = []

        # 基于筛选结果生成初始解
        screened_strategies = (
            self.screening_results.get("screened_strategies", [])
            if self.screening_results
            else []
        )

        if screened_strategies:
            # 使用筛选结果作为种子
            for i in range(min(size, len(screened_strategies))):
                strategy = screened_strategies[i]
                weights_dict = strategy["weights"]
                weights_array = [
                    weights_dict.get(factor, 0) for factor in self.factor_universe
                ]
                population.append(np.array(weights_array))

        # 补充随机解
        while len(population) < size:
            weights = self._generate_random_weights()
            population.append(weights)

        return np.array(population)

    def _generate_random_weights(self) -> np.ndarray:
        """生成随机权重"""
        # 防御性检查：确保有足够的因子
        if len(self.factor_universe) < self.config.min_factors:
            logger.warning(
                f"因子数量({len(self.factor_universe)})少于最小要求({self.config.min_factors})，"
                f"使用所有可用因子"
            )
            n_factors = len(self.factor_universe)
        else:
            # 随机选择因子数量
            upper_bound = min(
                self.config.max_factors + 1, len(self.factor_universe) + 1
            )
            lower_bound = min(self.config.min_factors, len(self.factor_universe))

            # 确保上界大于下界
            if upper_bound <= lower_bound:
                n_factors = lower_bound
            else:
                n_factors = np.random.randint(lower_bound, upper_bound)

        weights = np.zeros(len(self.factor_universe))

        # 防止选择数量超过可用因子数
        n_factors = min(n_factors, len(self.factor_universe))

        if n_factors == 0:
            logger.error("无法生成权重：没有可用因子")
            return weights

        # 随机选择因子
        selected_indices = np.random.choice(
            len(self.factor_universe), n_factors, replace=False
        )

        # 生成权重
        raw_weights = np.random.exponential(1, n_factors)
        raw_weights = np.clip(
            raw_weights, self.config.min_weight, self.config.max_weight
        )

        # 标准化
        total_weight = np.sum(raw_weights)
        if total_weight > 0:
            raw_weights = raw_weights / total_weight

        # 填充权重数组
        for i, idx in enumerate(selected_indices):
            weights[idx] = raw_weights[i]

        return weights

    def tournament_selection(
        self, population: np.ndarray, fitness: np.ndarray, tournament_size: int = 3
    ) -> np.ndarray:
        """锦标赛选择"""
        selected = []
        n_selected = len(population)

        for _ in range(n_selected):
            # 随机选择锦标赛参与者
            participants = np.random.choice(
                len(population), tournament_size, replace=False
            )
            participant_fitness = fitness[participants]

            # 选择最佳个体
            winner_idx = participants[np.argmax(participant_fitness)]
            selected.append(population[winner_idx].copy())

        return np.array(selected)

    def crossover(
        self, parent1: np.ndarray, parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """交叉操作"""
        if np.random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        # 单点交叉
        crossover_point = np.random.randint(1, len(parent1))

        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

        return self._repair_weights(child1), self._repair_weights(child2)

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """变异操作"""
        mutated = individual.copy()

        if np.random.random() < self.config.mutation_rate:
            # 随机选择一个基因进行变异
            gene_idx = np.random.randint(len(mutated))

            # 高斯变异
            mutation = np.random.normal(0, 0.1)
            mutated[gene_idx] += mutation

            # 确保权重非负
            mutated = np.maximum(mutated, 0)

        return self._repair_weights(mutated)

    def _repair_weights(self, weights: np.ndarray) -> np.ndarray:
        """修复权重以满足约束"""
        # 移除过小的权重
        weights[weights < self.config.min_weight] = 0

        # 限制最大权重
        weights = np.minimum(weights, self.config.max_weight)

        # 标准化权重和
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # 如果全为0，重新生成
            weights = self._generate_random_weights()

        return weights

    def local_search(self, solution: np.ndarray, top_n: int = 3) -> np.ndarray:
        """局部搜索"""
        if not self.config.enable_local_search:
            return solution

        best_solution = solution.copy()
        best_fitness = self.objective_function(best_solution, top_n)

        # 在解的邻域内搜索
        for _ in range(10):  # 10次局部搜索尝试
            neighbor = best_solution.copy()

            # 随机扰动
            perturbation = np.random.normal(
                0, self.config.local_search_radius, len(neighbor)
            )
            neighbor = neighbor + perturbation
            neighbor = self._repair_weights(neighbor)

            # 评估邻域解
            neighbor_fitness = self.objective_function(neighbor, top_n)

            if neighbor_fitness > best_fitness:
                best_solution = neighbor
                best_fitness = neighbor_fitness

        return best_solution

    def genetic_algorithm(
        self, max_iterations: int, top_n: int = 3
    ) -> Tuple[np.ndarray, float]:
        """遗传算法优化"""
        logger.info(f"开始遗传算法优化 (最大迭代: {max_iterations}, Top-N: {top_n})")

        # 初始化种群
        population = self.generate_initial_population(self.config.population_size)

        # 计算适应度
        fitness = np.array([self.objective_function(ind, top_n) for ind in population])

        # 记录最佳解
        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        self.optimization_history = [
            {
                "iteration": 0,
                "best_fitness": best_fitness,
                "avg_fitness": np.mean(fitness),
                "best_solution": best_solution.tolist(),
            }
        ]

        # 进化循环
        for iteration in range(1, max_iterations + 1):
            # 选择
            selected = self.tournament_selection(population, fitness)

            # 交叉和变异
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # 局部搜索增强
            if self.config.enable_local_search and iteration % 10 == 0:
                for i in range(min(5, len(new_population))):
                    new_population[i] = self.local_search(new_population[i], top_n)

            population = np.array(new_population[: self.config.population_size])

            # 计算新适应度
            fitness = np.array(
                [self.objective_function(ind, top_n) for ind in population]
            )

            # 更新最佳解
            current_best_idx = np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]

            if current_best_fitness > best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = current_best_fitness

            # 记录历史
            self.optimization_history.append(
                {
                    "iteration": iteration,
                    "best_fitness": best_fitness,
                    "avg_fitness": np.mean(fitness),
                    "best_solution": best_solution.tolist(),
                }
            )

            # 检查收敛
            if iteration > 10:
                recent_improvement = (
                    self.optimization_history[-1]["best_fitness"]
                    - self.optimization_history[-10]["best_fitness"]
                )
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"在第{iteration}代收敛")
                    break

            # 报告进度
            if iteration % 50 == 0:
                logger.info(
                    f"第{iteration}代 - 最佳适应度: {best_fitness:.4f}, 平均适应度: {np.mean(fitness):.4f}"
                )

        self.best_solution = best_solution
        logger.info(f"遗传算法完成，最佳适应度: {best_fitness:.4f}")

        return best_solution, best_fitness

    def run_optimization(self, optimization_targets: List[int] = [3, 5, 8]) -> Dict:
        """运行完整优化流程"""
        logger.info("开始策略优化...")

        if not self.screening_results:
            return {"error": "未加载筛选结果"}

        try:
            optimization_results = {}

            for top_n in optimization_targets:
                logger.info(f"优化 Top-{top_n} 策略...")

                # 运行遗传算法
                best_weights, best_fitness = self.genetic_algorithm(
                    self.config.max_iterations, top_n
                )

                # 转换为权重字典
                weight_dict = {}
                for i, factor in enumerate(self.factor_universe):
                    if (
                        i < len(best_weights)
                        and best_weights[i] > self.config.min_weight
                    ):
                        weight_dict[factor] = best_weights[i]

                # 评估最终性能
                final_performance = self._evaluate_weights(weight_dict, top_n)

                optimization_results[f"top_{top_n}"] = {
                    "weights": weight_dict,
                    "fitness_score": best_fitness,
                    "performance": final_performance,
                    "optimization_history": self.optimization_history.copy(),
                }

                # 重置历史记录
                self.optimization_history = []

            # 找到整体最佳解
            best_overall = None
            best_score = -float("inf")

            for target, result in optimization_results.items():
                if result["fitness_score"] > best_score:
                    best_score = result["fitness_score"]
                    best_overall = target

            # 持久化优化结果到实例属性
            self.optimization_results = optimization_results

            return {
                "success": True,
                "optimization_results": optimization_results,
                "best_target": best_overall,
                "best_score": best_score,
                "factor_universe": self.factor_universe,
                "config_used": {
                    "max_iterations": self.config.max_iterations,
                    "population_size": self.config.population_size,
                    "n_workers": self.config.n_workers,
                },
            }

        except Exception as e:
            logger.error(f"优化过程中发生错误: {e}")
            return {"error": str(e)}

    def save_optimization_results(self, output_path: str) -> bool:
        """保存优化结果"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                "optimization_config": {
                    "max_iterations": self.config.max_iterations,
                    "population_size": self.config.population_size,
                    "mutation_rate": self.config.mutation_rate,
                    "crossover_rate": self.config.crossover_rate,
                    "sharpe_weight": self.config.sharpe_weight,
                    "return_weight": self.config.return_weight,
                    "drawdown_weight": self.config.drawdown_weight,
                },
                "factor_universe": self.factor_universe,
                "timestamp": datetime.now().isoformat(),
            }

            # 如果有优化结果，添加到保存数据中
            if hasattr(self, "optimization_results"):
                save_data["optimization_results"] = self.optimization_results

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)

            logger.info(f"优化结果已保存至: {output_file}")
            return True

        except Exception as e:
            logger.error(f"保存优化结果失败: {e}")
            return False

    def print_optimization_summary(self, results: Dict):
        """打印优化摘要"""
        if not results.get("success"):
            logger.error("优化失败，无法显示摘要")
            return

        print("\n" + "=" * 70)
        print("🚀 策略优化结果摘要")
        print("=" * 70)

        optimization_results = results["optimization_results"]
        best_target = results["best_target"]

        print(f"🎯 最佳目标: {best_target}")
        print(f"🏆 最佳评分: {results['best_score']:.4f}")
        print(f"🔧 优化配置: {results['config_used']}")

        print(f"\n📊 各Top-N优化结果:")
        for target, result in optimization_results.items():
            performance = result["performance"]
            print(f"  {target}:")
            print(f"    适应度评分: {result['fitness_score']:.4f}")
            print(f"    夏普比率: {performance['sharpe_ratio']:.3f}")
            print(f"    总收益: {performance['total_return']:.2f}%")
            print(f"    最大回撤: {performance['max_drawdown']:.2f}%")

        if best_target:
            best_result = optimization_results[best_target]
            print(f"\n🎯 最优策略配置 ({best_target}):")
            weights = best_result["weights"]
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

            for factor, weight in sorted_weights:
                if weight > 0.01:
                    print(f"  {factor:20s}: {weight:.3f}")

        print("=" * 70)


def main():
    """主函数 - 示例用法"""
    # 创建优化配置
    config = OptimizationConfig(
        max_iterations=500,
        population_size=30,
        n_workers=6,
        sharpe_weight=0.7,
        return_weight=0.2,
        drawdown_weight=0.1,
    )

    # 创建优化器
    optimizer = StrategyOptimizer(config)

    # 加载筛选结果
    screening_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/04_精细策略/output/screening_results.json"

    if optimizer.load_screening_results(screening_path):
        # 运行优化
        results = optimizer.run_optimization([3, 5, 8])

        if results.get("success"):
            # 保存结果
            output_path = "/Users/zhangshenshen/深度量化0927/etf_rotation_system/04_精细策略/output/optimization_results.json"
            optimizer.save_optimization_results(output_path)

            # 打印摘要
            optimizer.print_optimization_summary(results)
        else:
            logger.error(f"优化失败: {results.get('error')}")
    else:
        logger.error("无法加载筛选结果")


if __name__ == "__main__":
    main()
