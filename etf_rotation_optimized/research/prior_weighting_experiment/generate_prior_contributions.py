#!/usr/bin/env python3
"""
生成因子先验贡献数据

使用历史数据（2020-2022）运行WFO，统计每个因子的平均贡献，
保存为YAML配置文件，供离线先验加权使用。

执行: python scripts/generate_prior_contributions.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def generate_prior_contributions(
    start_date: str = "2020-01-01",
    end_date: str = "2022-12-31",
    output_path: str = "configs/prior_contributions.yaml",
):
    """
    生成先验贡献数据

    策略: 使用现有WFO结果，提取因子贡献统计
    """
    print("=" * 80)
    print("🔬 生成因子先验贡献数据")
    print("=" * 80)
    print(f"数据区间: {start_date} ~ {end_date}")
    print(f"输出路径: {output_path}")
    print("")

    # 简化方案：直接使用最近一次WFO的结果
    print("📊 方案: 使用最近一次WFO结果生成先验")
    print("")
    print("⚠️  注意: 这是简化实现，用于快速验证离线先验方案")
    print("   生产环境应使用严格的历史数据区间")
    print("")

    # 查找最近的WFO结果
    wfo_root = project_root / "results" / "wfo"
    if not wfo_root.exists():
        print("❌ 未找到WFO结果目录，请先运行WFO")
        return None

    # 找到最新的WFO结果
    latest_wfo = None
    latest_time = None

    for date_dir in sorted(wfo_root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for time_dir in sorted(date_dir.iterdir(), reverse=True):
            if not time_dir.is_dir():
                continue
            summary_file = time_dir / "wfo_summary.csv"
            if summary_file.exists():
                latest_wfo = summary_file
                latest_time = time_dir.name
                break
        if latest_wfo:
            break

    if not latest_wfo:
        print("❌ 未找到WFO结果文件")
        return None

    print(f"✅ 找到WFO结果: {latest_wfo}")
    print(f"   时间戳: {latest_time}")
    print("")

    # 读取WFO结果
    import json

    import pandas as pd

    df = pd.read_csv(latest_wfo)
    print(f"📊 WFO统计:")
    print(f"   - 窗口数: {len(df)}")
    print(f"   - 平均IC: {df['oos_ensemble_ic'].mean():.4f}")
    print("")

    # 提取因子贡献
    print("🔍 提取因子贡献...")
    prior_contributions = {}

    for _, row in df.iterrows():
        top_factors = json.loads(row["top_factors"])

        for factor, data in top_factors.items():
            if factor not in prior_contributions:
                prior_contributions[factor] = []
            prior_contributions[factor].append(data["contribution"])

    # 统计
    prior_stats = {}
    for factor, contribs in prior_contributions.items():
        if len(contribs) >= 3:  # 至少3次观测
            prior_stats[factor] = {
                "mean": float(np.mean(contribs)),
                "std": float(np.std(contribs)),
                "count": len(contribs),
                "median": float(np.median(contribs)),
                "min": float(np.min(contribs)),
                "max": float(np.max(contribs)),
            }

    print(f"   - 有效因子数: {len(prior_stats)}")
    print("")

    # 保存
    print("💾 保存先验数据...")
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source_wfo": str(latest_wfo),
            "n_windows": len(df),
            "n_factors": len(prior_stats),
            "avg_oos_ic": float(df["oos_ensemble_ic"].mean()),
            "note": "简化版先验，基于最近一次WFO结果",
        },
        "prior_contributions": prior_stats,
    }

    output_file = project_root / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ 先验数据已保存: {output_file}")
    print("")

    # 显示Top因子
    print("📈 Top 10 因子贡献:")
    sorted_factors = sorted(
        prior_stats.items(), key=lambda x: x[1]["mean"], reverse=True
    )

    for i, (factor, stats) in enumerate(sorted_factors[:10], 1):
        print(
            f"  {i:2d}. {factor:30s} "
            f"mean={stats['mean']:+.4f}, "
            f"std={stats['std']:.4f}, "
            f"count={stats['count']}"
        )

    print("")
    print("=" * 80)
    print("✅ 先验贡献数据生成完成")
    print("=" * 80)

    return prior_stats


if __name__ == "__main__":
    generate_prior_contributions()
