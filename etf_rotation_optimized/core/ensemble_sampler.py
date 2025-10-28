"""
Ensemble Sampler | 智能因子组合采样器

功能:
  1. 从18个因子中采样N个大小为K的组合
  2. 三层分层采样: 家族配额(50%) + IC加权(30%) + 随机探索(20%)
  3. 约束验证: 家族配额、互斥对、组合大小
  4. 去重确保唯一性

设计原则:
  - 纯向量化: 无.apply(), 最小化循环
  - 确定性: 固定随机种子可复现
  - 高效: O(N)采样,避免O(N^2)检查

作者: Linus Quant Engineer
日期: 2025-10-28
"""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleSampler:
    """
    智能因子组合采样器

    采样空间: 18个因子,选5个 → C(18,5)=8568种组合
    约束后空间: ~3000组合 (家族配额+互斥规则)
    采样目标: 1000组合

    三层采样:
    - Layer 1 (50%): 家族配额采样 - 保证多样性
    - Layer 2 (30%): IC加权采样 - 利用历史信息
    - Layer 3 (20%): 随机探索 - 发现新模式
    """

    def __init__(self, constraints_config: Dict, random_seed: int = 42):
        """
        初始化采样器

        参数:
            constraints_config: 约束配置字典,来自FACTOR_SELECTION_CONSTRAINTS.yaml
            random_seed: 随机种子 (保证可复现)
        """
        self.constraints = constraints_config
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # 解析约束配置
        self.family_quotas = constraints_config.get("family_quotas", {})
        self.mutual_exclusions = self._parse_mutual_exclusions(
            constraints_config.get("mutually_exclusive_pairs", [])
        )

        # 构建因子到家族的映射
        self.factor_to_family = self._build_factor_family_mapping()

        logger.info(f"✓ EnsembleSampler初始化完成 (seed={random_seed})")
        logger.info(f"  - 家族数: {len(self.family_quotas)}")
        logger.info(f"  - 互斥对数: {len(self.mutual_exclusions)}")

    def _build_factor_family_mapping(self) -> Dict[str, str]:
        """构建因子→家族的映射"""
        mapping = {}
        for family_name, config in self.family_quotas.items():
            candidates = config.get("candidates", [])
            for factor in candidates:
                mapping[factor] = family_name
        return mapping

    def _parse_mutual_exclusions(self, pairs: List[Dict]) -> Set[Tuple[str, str]]:
        """解析互斥对,转为集合便于快速查找"""
        exclusions = set()
        for pair_config in pairs:
            pair = pair_config.get("pair", [])
            if len(pair) == 2:
                # 双向互斥: (A, B) 和 (B, A)
                exclusions.add(tuple(sorted(pair)))
        return exclusions

    def sample_combinations(
        self,
        n_samples: int,
        factor_pool: List[str],
        ic_scores: Dict[str, float] = None,
        combo_size: int = 5,
    ) -> List[Tuple[str, ...]]:
        """
        生成N个因子组合

        参数:
            n_samples: 采样数量 (如1000)
            factor_pool: 候选因子列表 (18个)
            ic_scores: 历史IC评分 {factor_name: ic_value} (可选)
            combo_size: 每个组合的因子数 (默认5)

        返回:
            List[Tuple]: 采样的组合列表,每个是(f1, f2, f3, f4, f5)的元组
        """
        logger.info(f"开始采样: 目标{n_samples}个组合 (size={combo_size})")

        samples = []

        # Layer 1: 家族配额采样 (50%)
        n_layer1 = int(n_samples * 0.5)
        layer1_samples = self._sample_by_family_quota(
            n_samples=n_layer1, factor_pool=factor_pool, combo_size=combo_size
        )
        samples.extend(layer1_samples)
        logger.info(f"  Layer 1 (家族配额): {len(layer1_samples)} 个")

        # Layer 2: IC加权采样 (30%)
        n_layer2 = int(n_samples * 0.3)
        if ic_scores:
            layer2_samples = self._sample_by_ic_weights(
                n_samples=n_layer2,
                factor_pool=factor_pool,
                ic_scores=ic_scores,
                combo_size=combo_size,
            )
            samples.extend(layer2_samples)
            logger.info(f"  Layer 2 (IC加权): {len(layer2_samples)} 个")
        else:
            logger.warning("  Layer 2 跳过: 无IC评分")

        # Layer 3: 随机探索 (20%)
        n_layer3 = int(n_samples * 0.2)
        layer3_samples = self._sample_random(
            n_samples=n_layer3, factor_pool=factor_pool, combo_size=combo_size
        )
        samples.extend(layer3_samples)
        logger.info(f"  Layer 3 (随机探索): {len(layer3_samples)} 个")

        # 去重并验证约束
        unique_samples = self._deduplicate_and_validate(samples, combo_size)
        logger.info(f"去重后: {len(unique_samples)} 个有效组合")

        # 如果不足目标数量,补充随机采样
        if len(unique_samples) < n_samples:
            shortage = n_samples - len(unique_samples)
            logger.warning(f"样本不足,补充随机采样: {shortage} 个")
            補充 = self._sample_random(shortage, factor_pool, combo_size)
            unique_samples.extend(補充)
            unique_samples = self._deduplicate_and_validate(unique_samples, combo_size)

        # 截断到目标数量
        final_samples = unique_samples[:n_samples]
        logger.info(f"✓ 采样完成: {len(final_samples)} 个组合")

        return final_samples

    def _sample_by_family_quota(
        self, n_samples: int, factor_pool: List[str], combo_size: int
    ) -> List[Tuple[str, ...]]:
        """
        按家族配额采样

        逻辑:
        1. 统计每个家族的因子数和max_count
        2. 按家族重要性(因子数×max_count)分配采样配额
        3. 从每个家族内随机采样因子
        4. 验证家族配额约束
        """
        samples = []

        # 按家族分组因子
        family_factors = defaultdict(list)
        for factor in factor_pool:
            family = self.factor_to_family.get(factor)
            if family:
                family_factors[family].append(factor)

        # 计算每个家族的采样配额 (按因子数×max_count加权)
        family_weights = {}
        for family, factors in family_factors.items():
            max_count = self.family_quotas[family].get("max_count", 2)
            family_weights[family] = len(factors) * max_count

        total_weight = sum(family_weights.values())
        family_quotas = {
            family: max(1, int(n_samples * weight / total_weight))
            for family, weight in family_weights.items()
        }

        # 从每个家族采样
        for family, quota in family_quotas.items():
            family_pool = family_factors[family]
            max_count = self.family_quotas[family].get("max_count", 2)

            for _ in range(quota):
                # 从该家族随机选择因子 (不超过max_count)
                n_from_family = min(random.randint(1, max_count), len(family_pool))

                selected_from_family = random.sample(family_pool, n_from_family)

                # 从其他家族补足到combo_size
                other_factors = [
                    f for f in factor_pool if f not in selected_from_family
                ]
                n_from_others = combo_size - n_from_family

                if n_from_others > 0 and len(other_factors) >= n_from_others:
                    selected_others = random.sample(other_factors, n_from_others)
                    combo = tuple(sorted(selected_from_family + selected_others))
                    samples.append(combo)

        return samples

    def _sample_by_ic_weights(
        self,
        n_samples: int,
        factor_pool: List[str],
        ic_scores: Dict[str, float],
        combo_size: int,
    ) -> List[Tuple[str, ...]]:
        """
        按IC加权采样

        逻辑:
        1. 高IC因子出现概率更高
        2. 权重 = softmax(IC_scores) - 防止极端权重
        3. 多项式采样 (允许重复,最后去重)
        """
        samples = []

        # 过滤出在factor_pool中的因子的IC
        valid_ics = {f: ic_scores[f] for f in factor_pool if f in ic_scores}

        if not valid_ics:
            return []

        # Softmax归一化 (避免负IC导致的问题)
        factors = list(valid_ics.keys())
        ics = np.array([valid_ics[f] for f in factors])

        # 将IC转换为正数 (IC可能为负)
        ics_shifted = ics - ics.min() + 0.01  # 偏移确保正数

        # Softmax
        exp_ics = np.exp(ics_shifted / ics_shifted.std())  # 温度缩放
        weights = exp_ics / exp_ics.sum()

        # 多项式采样
        for _ in range(n_samples):
            # 按权重采样combo_size个因子 (不放回)
            selected_indices = np.random.choice(
                len(factors), size=combo_size, replace=False, p=weights
            )
            combo = tuple(sorted([factors[i] for i in selected_indices]))
            samples.append(combo)

        return samples

    def _sample_random(
        self, n_samples: int, factor_pool: List[str], combo_size: int
    ) -> List[Tuple[str, ...]]:
        """
        随机探索采样

        逻辑:
        纯随机采样,无任何偏好
        """
        samples = []

        for _ in range(n_samples):
            if len(factor_pool) >= combo_size:
                combo = tuple(sorted(random.sample(factor_pool, combo_size)))
                samples.append(combo)

        return samples

    def _deduplicate_and_validate(
        self, samples: List[Tuple[str, ...]], combo_size: int
    ) -> List[Tuple[str, ...]]:
        """
        去重并验证约束

        逻辑:
        1. 集合去重
        2. 验证每个组合是否满足约束
        3. 返回有效组合列表
        """
        # 去重
        unique_samples = list(set(samples))

        # 验证约束
        valid_samples = [
            combo
            for combo in unique_samples
            if self._validate_constraints(combo, combo_size)
        ]

        return valid_samples

    def _validate_constraints(self, combo: Tuple[str, ...], combo_size: int) -> bool:
        """
        验证组合是否满足约束

        检查:
        1. 组合大小正确
        2. 家族配额: 每个家族不超过max_count
        3. 互斥对: 不能同时包含互斥因子
        """
        # 检查1: 组合大小
        if len(combo) != combo_size:
            return False

        # 检查2: 家族配额
        family_counts = defaultdict(int)
        for factor in combo:
            family = self.factor_to_family.get(factor)
            if family:
                family_counts[family] += 1

        for family, count in family_counts.items():
            max_count = self.family_quotas[family].get("max_count", 999)
            if count > max_count:
                return False

        # 检查3: 互斥对
        combo_set = set(combo)
        for f1, f2 in self.mutual_exclusions:
            if f1 in combo_set and f2 in combo_set:
                return False

        return True

    def get_sampling_statistics(self, samples: List[Tuple[str, ...]]) -> Dict:
        """
        计算采样统计信息 (用于验证)

        返回:
        - 每个家族的覆盖率
        - 每个因子的出现频率
        - 约束满足率
        """
        if not samples:
            return {}

        # 家族覆盖率
        family_coverage = defaultdict(int)
        for combo in samples:
            families_in_combo = set()
            for factor in combo:
                family = self.factor_to_family.get(factor)
                if family:
                    families_in_combo.add(family)
            for family in families_in_combo:
                family_coverage[family] += 1

        family_coverage_rate = {
            family: count / len(samples) for family, count in family_coverage.items()
        }

        # 因子出现频率
        factor_frequency = defaultdict(int)
        for combo in samples:
            for factor in combo:
                factor_frequency[factor] += 1

        factor_frequency_rate = {
            factor: count / len(samples) for factor, count in factor_frequency.items()
        }

        # 约束满足率
        valid_count = sum(
            1 for combo in samples if self._validate_constraints(combo, len(samples[0]))
        )
        constraint_compliance_rate = valid_count / len(samples)

        return {
            "family_coverage": family_coverage_rate,
            "factor_frequency": factor_frequency_rate,
            "constraint_compliance_rate": constraint_compliance_rate,
            "total_samples": len(samples),
            "unique_samples": len(set(samples)),
        }


# =========================================================================
# 使用示例
# =========================================================================

if __name__ == "__main__":
    print("EnsembleSampler 测试")
    print("=" * 70)

    # 模拟约束配置
    mock_constraints = {
        "family_quotas": {
            "momentum_trend": {"max_count": 2, "candidates": ["MOM_20D", "SLOPE_20D"]},
            "volatility_risk": {
                "max_count": 2,
                "candidates": ["RET_VOL_20D", "MAX_DD_60D"],
            },
        },
        "mutually_exclusive_pairs": [{"pair": ["MOM_20D", "SLOPE_20D"]}],
    }

    # 创建采样器
    sampler = EnsembleSampler(mock_constraints, random_seed=42)

    # 模拟因子池
    factor_pool = [
        "MOM_20D",
        "SLOPE_20D",
        "RET_VOL_20D",
        "MAX_DD_60D",
        "RSI_14",
        "CMF_20D",
        "SHARPE_RATIO_20D",
    ]

    # 模拟IC评分
    ic_scores = {f: np.random.randn() * 0.05 for f in factor_pool}

    # 采样
    samples = sampler.sample_combinations(
        n_samples=100, factor_pool=factor_pool, ic_scores=ic_scores, combo_size=5
    )

    print(f"\n✓ 采样完成: {len(samples)} 个组合")
    print(f"  - 前5个组合:")
    for i, combo in enumerate(samples[:5], 1):
        print(f"    {i}. {combo}")

    # 统计分析
    stats = sampler.get_sampling_statistics(samples)
    print(f"\n统计信息:")
    print(f"  - 约束满足率: {stats['constraint_compliance_rate']:.1%}")
    print(f"  - 唯一组合数: {stats['unique_samples']}")
    print(f"  - 家族覆盖率: {stats['family_coverage']}")
