"""
因子分桶映射 (基于截面 Rank 相关性矩阵, 2026-02-11 验证)

设计原则:
  - 同桶内因子高度相关 (>0.4), 选1-2个即可饱和该维度
  - 跨桶因子低相关 (<0.3), 组合跨桶选才能获取正交信息
  - 分桶依据: PCA + 截面 rank 相关性矩阵 (results/alpha_dimension_analysis/)

用途:
  - WFO 组合生成时加跨桶约束, 提高搜索效率
  - 因子选择可解释性: 明确组合覆盖了哪些信息维度
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple


# ─── 5 维度分桶 ───────────────────────────────────────────────────
# 基于 17 因子截面 Rank 相关性矩阵 (Kaiser 有效维度=5)

FACTOR_BUCKETS: Dict[str, List[str]] = {
    # Bucket A: 趋势/动量 (PC1 核心, 内部相关 0.47-0.87)
    # 这些因子几乎测量同一件事: "哪只ETF在涨"
    "TREND_MOMENTUM": [
        "MOM_20D",              # 20日收益率
        "SHARPE_RATIO_20D",     # 风险调整收益 (与MOM r=0.87)
        "SLOPE_20D",            # 价格斜率 (与SHARPE r=0.73)
        "VORTEX_14D",           # 方向运动 (与SHARPE r=0.77)
        "BREAKOUT_20D",         # 突破信号 (与PP_20D r=0.74)
        "PRICE_POSITION_20D",   # 短期价格位置 (与VORTEX r=0.75)
    ],

    # Bucket B: 持续位置/长期趋势 (内部相关 0.77)
    # 与 Bucket A 有中等相关 (0.43-0.58), 但时间尺度更长
    "SUSTAINED_POSITION": [
        "PRICE_POSITION_120D",  # 120日价格位置
        "CALMAR_RATIO_60D",     # 60日风险调整 (与PP120 r=0.77)
    ],

    # Bucket C: 量能确认 (内部相关 0.43)
    # 与 Bucket A 的相关低于桶内 (OBV avg 0.27, UDVOL avg 0.50)
    "VOLUME_CONFIRMATION": [
        "OBV_SLOPE_10D",            # OBV 斜率 — 较正交, 与趋势桶 avg=0.27
        "UP_DOWN_VOL_RATIO_20D",    # 上涨量/下跌量 — 与趋势桶 avg=0.50 (边界)
    ],

    # Bucket D: 微观结构/流动性 (内部相关 0.02-0.05, 各自独立)
    # 与其他桶大部分 <0.15, 捕获不同信息维度
    "MICROSTRUCTURE": [
        "PV_CORR_20D",         # 价量相关 — 与MOM r=0.33 (最高跨桶)
        "AMIHUD_ILLIQUIDITY",  # 流动性 — 与MAX_DD r=0.30, 其余 <0.10
        "GK_VOL_RATIO_20D",   # GK波动率 — 与所有因子 |r|<0.08, 近乎纯正交
    ],

    # Bucket E: 趋势强度/风险/市场关联 (混合, 但都是"条件因子")
    # ADX 与所有因子 |r|<0.13, 是最正交的单因子
    "TREND_STRENGTH_RISK": [
        "ADX_14D",                  # 趋势强度 (方向无关) — 近乎纯正交
        "CORRELATION_TO_MARKET_20D",# 市场 beta — 与趋势桶负相关 -0.06~-0.19
        "MAX_DD_60D",               # 最大回撤 — 与趋势桶负相关 -0.21~-0.52
        "VOL_RATIO_20D",            # 波动率变化 — 弱相关 0.02-0.20
    ],
}

# 反向映射: 因子 → 桶名
FACTOR_TO_BUCKET: Dict[str, str] = {}
for bucket_name, factors in FACTOR_BUCKETS.items():
    for f in factors:
        FACTOR_TO_BUCKET[f] = bucket_name

# ─── S1 覆盖分析 ─────────────────────────────────────────────────
# S1 = SLOPE(A) + SHARPE(A) + OBV(C) + ADX(E)
# 覆盖: A + C + E = 3/5 桶
# 缺失: B (持续位置), D (微观结构)
# → S2 加了 PP_120D 补了 B, 但 D 仍空白

# 冠军 = AMIHUD(D) + PP_20D(A) + PV_CORR(D) + SLOPE(A)
# 覆盖: A + D = 2/5 桶
# 缺失: B, C, E → 缺量能确认和趋势强度, 执行框架下不稳定


def get_bucket_coverage(factor_names: List[str]) -> Dict[str, List[str]]:
    """返回组合覆盖的桶及每桶选了哪些因子."""
    coverage: Dict[str, List[str]] = {}
    for f in factor_names:
        bucket = FACTOR_TO_BUCKET.get(f)
        if bucket:
            coverage.setdefault(bucket, []).append(f)
    return coverage


def check_cross_bucket_constraint(
    factor_names: List[str],
    min_buckets: int = 3,
    max_per_bucket: int = 2,
) -> Tuple[bool, str]:
    """检查组合是否满足跨桶约束.

    Args:
        factor_names: 因子名列表
        min_buckets: 最少覆盖桶数 (默认3)
        max_per_bucket: 每桶最多选几个 (默认2)

    Returns:
        (通过, 原因)
    """
    coverage = get_bucket_coverage(factor_names)
    n_buckets = len(coverage)

    if n_buckets < min_buckets:
        return False, f"仅覆盖 {n_buckets}/{min_buckets} 个桶: {list(coverage.keys())}"

    for bucket, factors in coverage.items():
        if len(factors) > max_per_bucket:
            return False, f"桶 {bucket} 选了 {len(factors)} 个因子 (上限 {max_per_bucket}): {factors}"

    return True, f"覆盖 {n_buckets} 个桶, 分布: {dict_summary(coverage)}"


def dict_summary(coverage: Dict[str, List[str]]) -> str:
    return ", ".join(f"{k}={len(v)}" for k, v in sorted(coverage.items()))


def generate_cross_bucket_combos(
    factor_names: List[str],
    combo_size: int,
    min_buckets: int = 3,
    max_per_bucket: int = 2,
) -> List[Tuple[str, ...]]:
    """生成满足跨桶约束的因子组合.

    比暴力枚举+过滤高效: 先从每桶选子集, 再跨桶笛卡尔积.

    Args:
        factor_names: 可用因子列表
        combo_size: 组合大小 (e.g. 4)
        min_buckets: 最少覆盖桶数
        max_per_bucket: 每桶最多选几个

    Returns:
        满足约束的因子名组合列表
    """
    from itertools import combinations, product

    # 按桶分组可用因子
    available_by_bucket: Dict[str, List[str]] = {}
    for f in factor_names:
        bucket = FACTOR_TO_BUCKET.get(f)
        if bucket:
            available_by_bucket.setdefault(bucket, []).append(f)

    bucket_names = sorted(available_by_bucket.keys())
    n_buckets_available = len(bucket_names)

    if n_buckets_available < min_buckets:
        return []  # 不够桶

    # 枚举桶的选择方案: 选哪些桶, 每桶选几个
    # 用整数分配: combo_size 个因子分到 k 个桶, 每桶 1~max_per_bucket 个
    results: Set[FrozenSet[str]] = set()

    for k in range(min_buckets, min(n_buckets_available, combo_size) + 1):
        for bucket_subset in combinations(bucket_names, k):
            # 枚举分配: combo_size 分到 k 个桶
            for allocation in _partitions(combo_size, k, max_per_bucket):
                # 从每桶选 allocation[i] 个因子
                per_bucket_choices = []
                valid = True
                for i, bname in enumerate(bucket_subset):
                    avail = available_by_bucket[bname]
                    n_pick = allocation[i]
                    if n_pick > len(avail):
                        valid = False
                        break
                    per_bucket_choices.append(
                        list(combinations(avail, n_pick))
                    )
                if not valid:
                    continue

                # 笛卡尔积
                for combo_parts in product(*per_bucket_choices):
                    combo = []
                    for part in combo_parts:
                        combo.extend(part)
                    results.add(frozenset(combo))

    return [tuple(sorted(c)) for c in sorted(results)]


def _partitions(total: int, k: int, max_each: int) -> List[Tuple[int, ...]]:
    """将 total 分成 k 份, 每份 1~max_each."""
    if k == 1:
        if 1 <= total <= max_each:
            return [(total,)]
        return []
    result = []
    for first in range(1, min(total - k + 1, max_each) + 1):
        for rest in _partitions(total - first, k - 1, max_each):
            result.append((first,) + rest)
    return result
