#!/usr/bin/env python3
"""生成相关性热图 - 最近3个月数据

输出：
1. 相关性矩阵（CSV）
2. 高相关对列表（ρ>0.9）
3. 热图可视化（PNG）
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_correlation_heatmap():
    """生成相关性热图"""

    logger.info("=" * 80)
    logger.info("相关性热图生成 - 最近3个月")
    logger.info("=" * 80)

    # 加载面板
    panel_file = Path(
        "factor_output/etf_rotation_production/panel_FULL_20200102_20251014.parquet"
    )
    panel = pd.read_parquet(panel_file)

    # 获取最近3个月数据
    dates = panel.index.get_level_values("date").unique()
    recent_dates = dates[-60:]  # 约3个月（60个交易日）

    logger.info(f"\n数据范围:")
    logger.info(f"  起始日期: {recent_dates[0]}")
    logger.info(f"  结束日期: {recent_dates[-1]}")
    logger.info(f"  交易日数: {len(recent_dates)}")

    # 筛选最近3个月数据
    panel_recent = panel[panel.index.get_level_values("date").isin(recent_dates)]

    logger.info(f"\n面板统计:")
    logger.info(f"  因子数: {panel_recent.shape[1]}")
    logger.info(f"  样本数: {panel_recent.shape[0]}")

    # 计算相关性矩阵
    logger.info(f"\n计算相关性矩阵...")
    corr_matrix = panel_recent.corr()

    # 保存相关性矩阵
    output_dir = Path("factor_output/etf_rotation_production")
    corr_file = output_dir / "correlation_matrix_3m.csv"
    corr_matrix.to_csv(corr_file)
    logger.info(f"✅ 相关性矩阵已保存: {corr_file}")

    # 找出高相关对（ρ>0.9）
    logger.info(f"\n高相关对（ρ>0.9）:")
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.9:
                factor1 = corr_matrix.columns[i]
                factor2 = corr_matrix.columns[j]
                high_corr_pairs.append(
                    {"factor1": factor1, "factor2": factor2, "correlation": corr_val}
                )

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values(
        "correlation", ascending=False
    )

    logger.info(f"  发现{len(high_corr_df)}对高相关因子")

    if len(high_corr_df) > 0:
        logger.info(f"\n  Top 20:")
        for idx, row in high_corr_df.head(20).iterrows():
            logger.info(
                f"    {row['factor1']} ↔ {row['factor2']}: ρ={row['correlation']:.6f}"
            )

        # 保存高相关对
        high_corr_file = output_dir / "high_correlation_pairs_3m.csv"
        high_corr_df.to_csv(high_corr_file, index=False)
        logger.info(f"\n✅ 高相关对已保存: {high_corr_file}")

    # 生成热图（采样显示，避免过大）
    logger.info(f"\n生成热图...")

    # 如果因子太多，采样显示
    if len(corr_matrix) > 50:
        # 选择覆盖率最高的50个因子
        summary = pd.read_csv(output_dir / "factor_summary_20200102_20251014.csv")
        top_factors = summary.nlargest(50, "coverage")["factor_id"].tolist()
        corr_matrix_plot = corr_matrix.loc[top_factors, top_factors]
        logger.info(f"  采样显示Top 50因子（按覆盖率）")
    else:
        corr_matrix_plot = corr_matrix

    # 绘制热图
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        corr_matrix_plot,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Factor Correlation Heatmap (Recent 3 Months)", fontsize=16, pad=20)
    plt.tight_layout()

    heatmap_file = output_dir / "correlation_heatmap_3m.png"
    plt.savefig(heatmap_file, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ 热图已保存: {heatmap_file}")

    # 统计相关性分布
    logger.info(f"\n相关性分布:")
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

    logger.info(f"  |ρ| > 0.9: {(np.abs(corr_values) > 0.9).sum()} 对")
    logger.info(f"  |ρ| > 0.8: {(np.abs(corr_values) > 0.8).sum()} 对")
    logger.info(f"  |ρ| > 0.7: {(np.abs(corr_values) > 0.7).sum()} 对")
    logger.info(f"  |ρ| > 0.5: {(np.abs(corr_values) > 0.5).sum()} 对")

    logger.info(f"\n{'=' * 80}")
    logger.info("✅ 相关性分析完成")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = generate_correlation_heatmap()
    sys.exit(0 if success else 1)
