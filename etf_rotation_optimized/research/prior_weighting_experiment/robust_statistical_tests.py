#!/usr/bin/env python3
"""
稳健统计检验：Wilcoxon + Block Bootstrap + Permutation
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_test(ic_series: pd.Series, prior_series: pd.Series) -> Dict:
    """Wilcoxon符号秩检验（对非正态/厚尾更稳）"""
    diff = prior_series - ic_series
    stat, p_value = stats.wilcoxon(diff, alternative="greater")

    return {
        "test": "Wilcoxon",
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.10,
    }


def block_bootstrap_test(
    ic_series: pd.Series,
    prior_series: pd.Series,
    n_bootstrap: int = 10000,
    block_size: int = 6,
) -> Dict:
    """Block Bootstrap（处理序列相关）"""
    diff = (prior_series - ic_series).values
    n = len(diff)
    n_blocks = n // block_size

    # 生成bootstrap样本
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # 重采样块
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        bootstrap_sample = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = start + block_size
            bootstrap_sample.extend(diff[start:end])

        bootstrap_means.append(np.mean(bootstrap_sample))

    bootstrap_means = np.array(bootstrap_means)

    # 计算p值（单侧检验：先验>IC）
    p_value = (bootstrap_means <= 0).mean()

    # 95%置信区间
    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    return {
        "test": "Block Bootstrap",
        "mean_diff": diff.mean(),
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "significant": p_value < 0.10,
    }


def permutation_test(
    ic_series: pd.Series,
    prior_series: pd.Series,
    n_permutations: int = 10000,
) -> Dict:
    """置换检验（经验分布）"""
    diff = (prior_series - ic_series).values
    observed_mean = diff.mean()

    # 生成置换分布
    permutation_means = []
    for _ in range(n_permutations):
        # 随机翻转符号
        signs = np.random.choice([-1, 1], size=len(diff))
        permutation_means.append((diff * signs).mean())

    permutation_means = np.array(permutation_means)

    # 计算p值（单侧检验）
    p_value = (permutation_means >= observed_mean).mean()

    return {
        "test": "Permutation",
        "observed_mean": observed_mean,
        "p_value": p_value,
        "significant": p_value < 0.10,
    }


def run_all_tests(csv_path: str = "results/wfo/prior_weighted_validation.csv"):
    """运行所有稳健统计检验"""
    df = pd.read_csv(csv_path)
    ic_series = df["ic_weighted_ic"]
    prior_series = df["prior_weighted_ic"]

    print("=" * 80)
    print("稳健统计检验")
    print("=" * 80)
    print()

    # 1. Wilcoxon
    wilcoxon = wilcoxon_test(ic_series, prior_series)
    print("## 1. Wilcoxon符号秩检验（非参数）")
    print(f"   统计量: {wilcoxon['statistic']:.2f}")
    print(f"   p值:     {wilcoxon['p_value']:.4f}")
    print(f"   显著性: {'✅ 是 (p<0.10)' if wilcoxon['significant'] else '❌ 否'}")
    print()

    # 2. Block Bootstrap
    bootstrap = block_bootstrap_test(ic_series, prior_series)
    print("## 2. Block Bootstrap（处理序列相关）")
    print(f"   均值差: {bootstrap['mean_diff']:.4f}")
    print(f"   p值:     {bootstrap['p_value']:.4f}")
    print(f"   95% CI: [{bootstrap['ci_95'][0]:.4f}, {bootstrap['ci_95'][1]:.4f}]")
    print(f"   显著性: {'✅ 是 (p<0.10)' if bootstrap['significant'] else '❌ 否'}")
    print()

    # 3. Permutation
    permutation = permutation_test(ic_series, prior_series)
    print("## 3. 置换检验（经验分布）")
    print(f"   观测均值: {permutation['observed_mean']:.4f}")
    print(f"   p值:       {permutation['p_value']:.4f}")
    print(f"   显著性:   {'✅ 是 (p<0.10)' if permutation['significant'] else '❌ 否'}")
    print()

    # 综合判断
    print("=" * 80)
    print("综合判断")
    print("=" * 80)

    n_significant = sum(
        [
            wilcoxon["significant"],
            bootstrap["significant"],
            permutation["significant"],
        ]
    )

    print(f"显著性检验通过: {n_significant}/3")

    if n_significant >= 2:
        print("✅ **通过**: 至少2个检验显著，先验加权有统计证据")
    elif n_significant == 1:
        print("⚠️  **边缘**: 仅1个检验显著，证据不足但有潜力")
    else:
        print("❌ **未通过**: 无显著性证据，不建议继续投入")

    print()

    return {
        "wilcoxon": wilcoxon,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "n_significant": n_significant,
    }


if __name__ == "__main__":
    results = run_all_tests()
