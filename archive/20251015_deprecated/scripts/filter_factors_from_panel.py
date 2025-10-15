#!/usr/bin/env python3
"""从全量面板筛选高质量因子

用法：
1. 研究模式：coverage≥30%，快速做IC/IR分析
2. 生产模式：coverage≥80%，zero_variance=False，leak_suspect=False
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FactorFilter:
    """因子筛选器"""

    def __init__(
        self,
        panel_file: str,
        summary_file: str,
        output_dir: str = "factor_output/etf_rotation",
    ):
        self.panel_file = Path(panel_file)
        self.summary_file = Path(summary_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载数据
        logger.info(f"加载面板: {panel_file}")
        self.panel = pd.read_parquet(panel_file)
        logger.info(f"面板形状: {self.panel.shape}")

        logger.info(f"加载因子概要: {summary_file}")
        self.summary = pd.read_csv(summary_file)
        logger.info(f"因子概要: {len(self.summary)} 个因子")

    def filter_factors(
        self,
        min_coverage: float = 0.8,
        allow_zero_variance: bool = False,
        allow_leak_suspect: bool = False,
        deduplicate: bool = True,
    ) -> list:
        """筛选因子"""
        logger.info("\n" + "=" * 60)
        logger.info("因子筛选")
        logger.info("=" * 60)
        logger.info(f"最小覆盖率: {min_coverage:.0%}")
        logger.info(f"允许零方差: {allow_zero_variance}")
        logger.info(f"允许泄露嫌疑: {allow_leak_suspect}")
        logger.info(f"去重: {deduplicate}")

        # 初始候选
        candidates = self.summary.copy()
        logger.info(f"\n初始候选: {len(candidates)} 个因子")

        # 1. 覆盖率过滤
        candidates = candidates[candidates["coverage"] >= min_coverage]
        logger.info(f"覆盖率≥{min_coverage:.0%}: {len(candidates)} 个因子")

        # 2. 零方差过滤
        if not allow_zero_variance:
            candidates = candidates[~candidates["zero_variance"]]
            logger.info(f"非零方差: {len(candidates)} 个因子")

        # 3. 泄露嫌疑过滤
        if not allow_leak_suspect and "leak_suspect" in candidates.columns:
            candidates = candidates[~candidates["leak_suspect"].fillna(False)]
            logger.info(f"非泄露嫌疑: {len(candidates)} 个因子")

        # 4. 去重（保留每组第一个）
        if deduplicate and "identical_group_id" in candidates.columns:
            # 有group_id的，每组只保留第一个
            grouped = candidates[candidates["identical_group_id"].notna()]
            unique_groups = grouped.drop_duplicates(
                subset=["identical_group_id"], keep="first"
            )

            # 无group_id的，全部保留
            ungrouped = candidates[candidates["identical_group_id"].isna()]

            candidates = pd.concat([unique_groups, ungrouped])
            logger.info(f"去重后: {len(candidates)} 个因子")

        # 5. 按覆盖率排序
        candidates = candidates.sort_values("coverage", ascending=False)

        selected_factors = candidates["factor_id"].tolist()
        logger.info(f"\n✅ 最终筛选: {len(selected_factors)} 个因子")

        return selected_factors

    def save_selected_panel(self, selected_factors: list, suffix: str = ""):
        """保存筛选后的面板"""
        logger.info("\n" + "=" * 60)
        logger.info("保存筛选后的面板")
        logger.info("=" * 60)

        # 提取选中的因子列
        selected_panel = self.panel[selected_factors]

        # 保存
        output_file = self.output_dir / f"panel_filtered{suffix}.parquet"
        selected_panel.to_parquet(output_file)
        logger.info(f"✅ 筛选面板已保存: {output_file}")
        logger.info(f"   形状: {selected_panel.shape}")

        # 保存因子清单
        factors_file = self.output_dir / f"factors_selected{suffix}.yaml"
        with open(factors_file, "w") as f:
            yaml.dump({"factors": selected_factors}, f)
        logger.info(f"✅ 因子清单已保存: {factors_file}")

        return selected_panel

    def analyze_factors(self, selected_factors: list):
        """分析筛选后的因子"""
        logger.info("\n" + "=" * 60)
        logger.info("因子分析")
        logger.info("=" * 60)

        selected_panel = self.panel[selected_factors]

        # 覆盖率分布
        coverage = selected_panel.notna().mean()
        logger.info(f"\n覆盖率分布:")
        logger.info(coverage.describe())

        # 相关性矩阵
        corr_matrix = selected_panel.corr()

        # 高相关对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.7:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr)
                    )

        if high_corr_pairs:
            logger.info(f"\n高相关对（|ρ|>0.7）: {len(high_corr_pairs)} 对")
            for f1, f2, corr in sorted(
                high_corr_pairs, key=lambda x: x[2], reverse=True
            )[:10]:
                logger.info(f"  {f1} <-> {f2}: {corr:.3f}")
        else:
            logger.info("\n✅ 无高相关对（|ρ|>0.7）")

        # 保存相关性矩阵
        corr_file = self.output_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)
        logger.info(f"\n✅ 相关性矩阵已保存: {corr_file}")


def main():
    parser = argparse.ArgumentParser(description="从全量面板筛选高质量因子")
    parser.add_argument("--panel-file", required=True, help="全量面板文件路径")
    parser.add_argument("--summary-file", required=True, help="因子概要文件路径")
    parser.add_argument(
        "--mode",
        choices=["research", "production"],
        default="production",
        help="筛选模式（research: 宽松, production: 严格）",
    )
    parser.add_argument("--min-coverage", type=float, help="最小覆盖率（覆盖默认值）")
    parser.add_argument(
        "--output-dir", default="factor_output/etf_rotation", help="输出目录"
    )

    args = parser.parse_args()

    # 根据模式设置参数
    if args.mode == "research":
        min_coverage = args.min_coverage or 0.3
        allow_zero_variance = True
        allow_leak_suspect = True
        suffix = "_research"
        logger.info("研究模式：宽松筛选，快速IC/IR分析")
    else:
        min_coverage = args.min_coverage or 0.8
        allow_zero_variance = False
        allow_leak_suspect = False
        suffix = "_production"
        logger.info("生产模式：严格筛选，高质量因子")

    # 创建筛选器
    filter_obj = FactorFilter(
        panel_file=args.panel_file,
        summary_file=args.summary_file,
        output_dir=args.output_dir,
    )

    # 筛选因子
    selected_factors = filter_obj.filter_factors(
        min_coverage=min_coverage,
        allow_zero_variance=allow_zero_variance,
        allow_leak_suspect=allow_leak_suspect,
        deduplicate=True,
    )

    # 保存筛选后的面板
    filter_obj.save_selected_panel(selected_factors, suffix=suffix)

    # 分析因子
    filter_obj.analyze_factors(selected_factors)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 因子筛选完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
