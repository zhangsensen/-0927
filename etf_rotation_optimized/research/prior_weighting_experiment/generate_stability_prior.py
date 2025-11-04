#!/usr/bin/env python3
"""
生成纯稳定性先验（去掉强度，只用IC标准差倒数）
假设：稳定的因子在regime变化时更可靠
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cross_section_processor import CrossSectionProcessor
from core.data_manager import DataManager
from core.factor_calculator import FactorCalculator

# 配置
PRIOR_START = "2020-01-01"
PRIOR_END = "2022-12-31"
IS_WINDOW_DAYS = 120
OOS_WINDOW_DAYS = 40
MIN_IC = 0.01
OUTPUT_PATH = Path("configs/stability_prior_contributions.yaml")


def calculate_stability_scores(cross_section_df: pd.DataFrame) -> pd.DataFrame:
    """计算因子稳定性分数（纯防御性）"""
    factor_cols = [c for c in cross_section_df.columns if c.startswith("factor_")]

    stability_scores = []

    for factor in factor_cols:
        factor_data = cross_section_df[factor].dropna()

        if len(factor_data) < 10:
            continue

        # 计算IC序列的稳定性
        ic_std = factor_data.std()
        ic_mean = factor_data.mean()

        # 稳定性 = 1 / CV（变异系数）
        if abs(ic_mean) > 1e-6:
            cv = ic_std / abs(ic_mean)
            stability = 1.0 / (1.0 + cv)  # 归一化到[0,1]
        else:
            stability = 0.0

        # IC胜率（正IC比例）
        win_rate = (factor_data > 0).mean()

        # 综合稳定性分数 = 0.7*稳定性 + 0.3*胜率
        combined_score = 0.7 * stability + 0.3 * win_rate

        stability_scores.append(
            {
                "factor": factor,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "cv": cv if abs(ic_mean) > 1e-6 else np.inf,
                "stability": stability,
                "win_rate": win_rate,
                "combined_score": combined_score,
            }
        )

    return pd.DataFrame(stability_scores)


def main():
    """生成纯稳定性先验"""
    print("=" * 80)
    print("生成纯稳定性先验")
    print("=" * 80)
    print()

    # 1. 加载数据
    print("加载数据...")
    dm = DataManager()
    df = dm.load_data()

    # 2. 计算因子
    print("计算因子...")
    fc = FactorCalculator()
    df = fc.calculate_all_factors(df)

    # 3. 生成横截面
    print("生成横截面...")
    processor = CrossSectionProcessor(
        is_window_days=IS_WINDOW_DAYS,
        oos_window_days=OOS_WINDOW_DAYS,
        min_ic=MIN_IC,
    )

    prior_df = df[(df.index >= PRIOR_START) & (df.index <= PRIOR_END)]
    cross_section = processor.create_cross_section(prior_df)

    # 4. 计算稳定性分数
    print("计算稳定性分数...")
    stability_df = calculate_stability_scores(cross_section)

    # 5. 过滤和排序
    stability_df = stability_df[stability_df["combined_score"] > 0.3]
    stability_df = stability_df.sort_values("combined_score", ascending=False)

    print(f"\n稳定因子数量: {len(stability_df)}")
    print("\nTop 10 稳定因子:")
    print(stability_df.head(10).to_string(index=False))

    # 6. 保存为YAML
    prior_dict = {}
    for _, row in stability_df.iterrows():
        factor = row["factor"]
        prior_dict[factor] = {
            "score": float(row["combined_score"]),
            "ic_mean": float(row["ic_mean"]),
            "ic_std": float(row["ic_std"]),
            "stability": float(row["stability"]),
            "win_rate": float(row["win_rate"]),
        }

    output_data = {
        "prior_type": "stability",
        "prior_period": f"{PRIOR_START} to {PRIOR_END}",
        "description": "纯稳定性先验（防御性）：基于IC变异系数和胜率",
        "factors": prior_dict,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ 稳定性先验已保存: {OUTPUT_PATH}")
    print()

    # 7. 对比原先验
    original_prior_path = Path("configs/prior_contributions.yaml")
    if original_prior_path.exists():
        with open(original_prior_path) as f:
            original = yaml.safe_load(f)

        print("## 对比原先验")
        print("-" * 80)

        original_factors = set(original.get("factors", {}).keys())
        stability_factors = set(prior_dict.keys())

        common = original_factors & stability_factors
        only_original = original_factors - stability_factors
        only_stability = stability_factors - original_factors

        print(f"共同因子: {len(common)}")
        print(f"仅原先验: {len(only_original)}")
        print(f"仅稳定性: {len(only_stability)}")
        print()

        if only_stability:
            print("新增稳定因子（Top 5）:")
            new_factors = stability_df[
                stability_df["factor"].isin(only_stability)
            ].head(5)
            print(
                new_factors[["factor", "combined_score", "win_rate"]].to_string(
                    index=False
                )
            )


if __name__ == "__main__":
    main()
