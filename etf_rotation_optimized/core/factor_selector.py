"""
因子选择器 | Factor Selector with Constraints

功能:
  1. 加载约束配置 (FACTOR_SELECTION_CONSTRAINTS.yaml)
  2. 验证约束配置的一致性
  3. 应用多种约束类型:
     - 家族配额 (Family Quota)
     - 互斥对 (Mutual Exclusivity)
     - 相关性去冗余 (Correlation Deduplication)
     - 最小IC约束 (Minimum IC)
  4. 从候选因子中筛选满足约束的最优子集
  5. 生成详细的约束应用报告

工作流:
  候选因子 (按IC排序)
    ↓
  应用最小IC约束 → 过滤低IC因子
    ↓
  应用相关性去冗余 → 去除高相关因子
    ↓
  应用互斥对约束 → 解决冲突因子对
    ↓
  应用家族配额约束 → 控制因子类型多样性
    ↓
  应用必选因子约束 → 确保关键因子被选中
    ↓
  最终选择 + 约束报告

作者: Step 5 Factor Selector
日期: 2025-10-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class ConstraintViolation:
    """约束违反记录"""

    constraint_type: str
    reason: str
    affected_factors: List[str]
    severity: str = "info"  # info, warning, error
    action_taken: str = ""

    def __repr__(self):
        return f"{self.constraint_type}: {self.reason} ({self.affected_factors})"


@dataclass
class SelectionReport:
    """因子选择报告"""

    candidate_factors: List[str]
    applied_constraints: List[str]
    violations: List[ConstraintViolation]
    final_selection: List[str]
    selection_scores: Dict[str, float]
    constraint_impacts: Dict[str, List[str]]

    def __repr__(self):
        return f"""
因子选择报告
├─ 候选因子数: {len(self.candidate_factors)}
├─ 最终选择数: {len(self.final_selection)}
├─ 应用约束数: {len(self.applied_constraints)}
├─ 约束违反数: {len(self.violations)}
├─ 最终选择: {self.final_selection}
└─ 选择评分: {self.selection_scores}
"""


class FactorSelector:
    """
    因子选择器

    属性:
        constraints: 约束配置字典
        verbose: 是否输出详细信息
        report: 最后一次选择的报告
    """

    def __init__(self, constraints_file: str = None, verbose: bool = True):
        """
        初始化因子选择器

        参数:
            constraints_file: 约束配置文件路径
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.constraints = {}
        self.report = None
        self.factor_family = {}  # 因子到家族的映射
        self._build_factor_family()

        if constraints_file:
            self.load_constraints(constraints_file)

    def _build_factor_family(self):
        """构建因子到家族的映射"""
        # 默认的因子家族分类
        default_families = {
            "momentum": ["MOM_20D", "SLOPE_20D"],
            "volatility": ["RET_VOL_20D", "VOL_RATIO_20D", "VOL_RATIO_60D"],
            "risk_adjusted": ["MAX_DD_60D"],
            "price_features": ["PRICE_POSITION_20D", "PRICE_POSITION_120D"],
            "correlation": ["PV_CORR_20D"],
            "technical": ["RSI_14"],
        }

        for family, factors in default_families.items():
            for factor in factors:
                self.factor_family[factor] = family

    def load_constraints(self, constraints_file: str):
        """
        从YAML文件加载约束配置

        参数:
            constraints_file: 配置文件路径
        """
        try:
            with open(constraints_file, "r", encoding="utf-8") as f:
                self.constraints = yaml.safe_load(f)

            if self.verbose:
                print(f"✓ 加载约束配置: {constraints_file}")
                print(
                    f"  - 家族配额: {len(self.constraints.get('family_quota', {}))} 个"
                )
                print(
                    f"  - 互斥对: {len(self.constraints.get('mutual_exclusivity', []))} 对"
                )
                print(
                    f"  - 相关性去重: {self.constraints.get('correlation_deduplication', {}).get('threshold', 'N/A')}"
                )

        except FileNotFoundError:
            print(f"⚠️ 约束配置文件未找到: {constraints_file}")

    def select_factors(
        self,
        ic_scores: Dict[str, float],
        factor_correlations: Optional[Dict[Tuple[str, str], float]] = None,
        target_count: int = None,
    ) -> Tuple[List[str], SelectionReport]:
        """
        选择满足约束的因子子集

        参数:
            ic_scores: 因子IC分数 {factor: ic_value}
            factor_correlations: 因子间相关系数 {(factor1, factor2): correlation}
            target_count: 目标选择数量 (若为None则尽量多选)

        返回:
            (selected_factors, report)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"因子选择")
            print(f"{'='*70}")
            print(f"候选因子数: {len(ic_scores)}")
            print(f"目标选择数: {target_count if target_count else '不限'}")

        # 1. 按IC排序候选因子
        sorted_candidates = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_names = [f for f, _ in sorted_candidates]

        violations = []
        applied_constraints = []
        constraint_impacts = {}

        # 2. 应用最小IC约束
        min_ic = self.constraints.get("minimum_ic", {}).get("global_minimum", 0.0)
        if min_ic > 0:
            before = len(candidate_names)
            candidate_names = [f for f in candidate_names if ic_scores[f] > min_ic]
            after = len(candidate_names)

            if before > after:
                applied_constraints.append(f"minimum_ic (threshold={min_ic})")
                constraint_impacts["minimum_ic"] = [
                    f for f in ic_scores if ic_scores[f] <= min_ic
                ]
                violations.append(
                    ConstraintViolation(
                        constraint_type="minimum_ic",
                        reason=f"IC ≤ {min_ic}",
                        affected_factors=constraint_impacts["minimum_ic"],
                        severity="info",
                        action_taken=f"排除 {before - after} 个因子",
                    )
                )

        # 3. 应用相关性去冗余
        selected = candidate_names.copy()
        if factor_correlations and "correlation_deduplication" in self.constraints:
            dedup_config = self.constraints["correlation_deduplication"]
            threshold = dedup_config.get("threshold", 0.8)
            strategy = dedup_config.get("strategy", "keep_higher_ic")

            removed = self._apply_correlation_deduplication(
                selected, ic_scores, factor_correlations, threshold, strategy
            )

            if removed:
                selected = [f for f in selected if f not in removed]
                applied_constraints.append(
                    f"correlation_deduplication (threshold={threshold})"
                )
                constraint_impacts["correlation_deduplication"] = removed
                violations.append(
                    ConstraintViolation(
                        constraint_type="correlation_deduplication",
                        reason=f"相关系数 > {threshold}",
                        affected_factors=removed,
                        severity="info",
                        action_taken=f"排除 {len(removed)} 个高相关因子",
                    )
                )

        # 4. 应用互斥对约束
        if "mutual_exclusivity" in self.constraints:
            removed = self._apply_mutual_exclusivity(selected, ic_scores)
            if removed:
                selected = [f for f in selected if f not in removed]
                applied_constraints.append("mutual_exclusivity")
                constraint_impacts["mutual_exclusivity"] = removed
                violations.append(
                    ConstraintViolation(
                        constraint_type="mutual_exclusivity",
                        reason="因子存在互斥关系",
                        affected_factors=removed,
                        severity="info",
                        action_taken=f"排除 {len(removed)} 个冲突因子",
                    )
                )

        # 5. 应用家族配额
        if "family_quota" in self.constraints:
            removed = self._apply_family_quota(selected, ic_scores)
            if removed:
                selected = [f for f in selected if f not in removed]
                applied_constraints.append("family_quota")
                constraint_impacts["family_quota"] = removed
                violations.append(
                    ConstraintViolation(
                        constraint_type="family_quota",
                        reason="超过家族配额限制",
                        affected_factors=removed,
                        severity="info",
                        action_taken=f"排除 {len(removed)} 个低优先级因子",
                    )
                )

        # 6. 应用必选因子约束
        required = self.constraints.get("required_factors", [])
        selected = self._apply_required_factors(selected, required)
        if required:
            applied_constraints.append("required_factors")

        # 7. 控制选择数量
        if target_count and len(selected) > target_count:
            # 按IC降序截断
            selected = sorted(selected, key=lambda f: ic_scores[f], reverse=True)[
                :target_count
            ]

        # 生成报告
        selection_scores = {f: ic_scores[f] for f in selected}

        report = SelectionReport(
            candidate_factors=candidate_names,
            applied_constraints=applied_constraints,
            violations=violations,
            final_selection=selected,
            selection_scores=selection_scores,
            constraint_impacts=constraint_impacts,
        )

        self.report = report

        if self.verbose:
            self._print_report(report)

        return selected, report

    def _apply_correlation_deduplication(
        self,
        candidates: List[str],
        ic_scores: Dict[str, float],
        correlations: Dict[Tuple[str, str], float],
        threshold: float,
        strategy: str,
    ) -> Set[str]:
        """
        应用相关性去冗余

        参数:
            candidates: 候选因子列表
            ic_scores: IC分数
            correlations: 相关系数字典
            threshold: 相关性阈值
            strategy: 去冗余策略

        返回:
            被移除的因子集合
        """
        removed = set()

        for i, f1 in enumerate(candidates):
            if f1 in removed:
                continue

            for f2 in candidates[i + 1 :]:
                if f2 in removed:
                    continue

                # 查找相关系数
                key = tuple(sorted([f1, f2]))
                corr = correlations.get(key, 0)

                if abs(corr) > threshold:
                    # 根据策略决定移除哪个
                    if strategy == "keep_higher_ic":
                        to_remove = f2 if ic_scores[f1] > ic_scores[f2] else f1
                    elif strategy == "keep_longer_period":
                        # 选择周期更长的 (假设名字中包含周期信息)
                        to_remove = f2 if "60" in f2 or "120" in f2 else f1
                    else:  # keep_first
                        to_remove = f2

                    removed.add(to_remove)

        return removed

    def _apply_mutual_exclusivity(
        self, candidates: List[str], ic_scores: Dict[str, float]
    ) -> Set[str]:
        """
        应用互斥对约束

        参数:
            candidates: 候选因子列表
            ic_scores: IC分数

        返回:
            被移除的因子集合
        """
        removed = set()
        mutex_pairs = self.constraints.get("mutual_exclusivity", [])

        for pair_config in mutex_pairs:
            pair = pair_config.get("pair", [])
            if len(pair) != 2:
                continue

            f1, f2 = pair

            # 检查两个因子是否都在候选中
            if f1 in candidates and f2 in candidates:
                # 保留IC更高的
                if ic_scores[f1] > ic_scores[f2]:
                    removed.add(f2)
                else:
                    removed.add(f1)

        return removed

    def _apply_family_quota(
        self, candidates: List[str], ic_scores: Dict[str, float]
    ) -> Set[str]:
        """
        应用家族配额约束

        参数:
            candidates: 候选因子列表
            ic_scores: IC分数

        返回:
            被移除的因子集合
        """
        removed = set()
        family_quota = self.constraints.get("family_quota", {})

        # 按家族分组
        family_factors = {}
        for family_name, family_config in family_quota.items():
            max_count = family_config.get("max_count", 999)
            factors = family_config.get("factors", [])

            # 找出该家族中的候选因子
            selected_in_family = [
                f for f in candidates if f in factors and f not in removed
            ]

            if len(selected_in_family) > max_count:
                # 按IC降序选择，移除低IC的
                sorted_by_ic = sorted(
                    selected_in_family, key=lambda f: ic_scores[f], reverse=True
                )
                to_remove = sorted_by_ic[max_count:]
                removed.update(to_remove)

        return removed

    def _apply_required_factors(
        self, candidates: List[str], required: List[str]
    ) -> List[str]:
        """
        应用必选因子约束

        参数:
            candidates: 候选因子列表
            required: 必选因子列表

        返回:
            更新后的因子列表
        """
        # 确保必选因子被包含
        selected = candidates.copy()
        for factor in required:
            if factor not in selected:
                selected.append(factor)

        return selected

    def _print_report(self, report: SelectionReport):
        """打印选择报告"""
        print(f"\n【约束应用】")
        print(f"  候选因子: {len(report.candidate_factors)}")
        print(
            f"  应用约束: {', '.join(report.applied_constraints) if report.applied_constraints else '无'}"
        )

        if report.violations:
            print(f"\n  约束违反情况:")
            for v in report.violations:
                print(f"    • {v.constraint_type}: {v.reason}")
                print(f"      → {v.action_taken}")

        print(f"\n【最终选择】")
        print(f"  选择数量: {len(report.final_selection)}")
        print(f"  选择的因子:")
        for factor, score in sorted(
            report.selection_scores.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    • {factor:20s}: IC = {score:.4f}")

        print(f"\n{'='*70}\n")


def create_default_selector() -> FactorSelector:
    """
    创建默认的因子选择器

    返回:
        FactorSelector 实例，预装默认约束
    """
    selector = FactorSelector(verbose=True)

    # 设置默认约束
    selector.constraints = {
        "family_quota": {
            "momentum": {"factors": ["MOM_20D", "SLOPE_20D"], "max_count": 1},
            "volatility": {
                "factors": ["RET_VOL_20D", "VOL_RATIO_20D", "VOL_RATIO_60D"],
                "max_count": 2,
            },
            "risk_adjusted": {"factors": ["MAX_DD_60D"], "max_count": 1},
            "price_features": {
                "factors": ["PRICE_POSITION_20D", "PRICE_POSITION_120D"],
                "max_count": 1,
            },
            "correlation": {"factors": ["PV_CORR_20D"], "max_count": 1},
            "technical": {"factors": ["RSI_14"], "max_count": 1},
        },
        "mutual_exclusivity": [
            {
                "pair": ["PRICE_POSITION_20D", "PRICE_POSITION_120D"],
                "reason": "周期重叠",
            },
            {"pair": ["MOM_20D", "SLOPE_20D"], "reason": "动量重叠"},
        ],
        "correlation_deduplication": {"threshold": 0.8, "strategy": "keep_higher_ic"},
        "minimum_ic": {"global_minimum": 0.02},
        "required_factors": [],
    }

    return selector


if __name__ == "__main__":
    # 示例用法
    print("因子选择器示例")

    selector = create_default_selector()

    # 模拟IC分数
    ic_scores = {
        "MOM_20D": 0.05,
        "SLOPE_20D": 0.03,
        "RET_VOL_20D": 0.04,
        "MAX_DD_60D": 0.02,
        "VOL_RATIO_20D": 0.06,
        "VOL_RATIO_60D": 0.055,
        "PRICE_POSITION_20D": 0.03,
        "PRICE_POSITION_120D": 0.025,
        "PV_CORR_20D": 0.04,
        "RSI_14": 0.015,
    }

    # 模拟因子相关系数
    factor_correlations = {
        ("MOM_20D", "SLOPE_20D"): 0.85,
        ("VOL_RATIO_20D", "VOL_RATIO_60D"): 0.75,
        ("PRICE_POSITION_20D", "PRICE_POSITION_120D"): 0.82,
    }

    # 执行选择
    selected, report = selector.select_factors(
        ic_scores, factor_correlations, target_count=5
    )

    print(f"\n最终选中: {selected}")
