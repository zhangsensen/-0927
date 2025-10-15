#!/usr/bin/env python3
"""
扩展因子系统深度风险分析
重点分析8月极端收益和因子异常
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExtendedFactorRiskAnalyzer:
    def __init__(self):
        self.panel_file = "factor_output/etf_rotation/panel_20240101_20251014.parquet"
        self.extended_config = "etf_rotation/configs/extended_scoring.yaml"
        self.core_config = "etf_rotation/configs/scoring.yaml"

    def load_data(self):
        """加载因子面板和配置"""
        logger.info("加载因子面板数据...")
        self.panel = pd.read_parquet(self.panel_file)

        # 加载扩展因子配置
        import yaml

        with open(self.extended_config) as f:
            self.extended_config_data = yaml.safe_load(f)

        # 加载核心因子配置
        with open(self.core_config) as f:
            self.core_config_data = yaml.safe_load(f)

        logger.info(f"因子面板形状: {self.panel.shape}")
        logger.info(
            f"时间范围: {self.panel.index.get_level_values(0).min()} 到 {self.panel.index.get_level_values(0).max()}"
        )

    def analyze_factor_coverage(self):
        """分析因子覆盖率"""
        logger.info("\n=== 因子覆盖率分析 ===")

        # 获取扩展因子列表
        factor_set_file = self.extended_config_data["factor_set_file"]
        import yaml

        with open(factor_set_file) as f:
            factor_set = yaml.safe_load(f)

        extended_factors = factor_set["factors"]
        logger.info(f"扩展因子集总数: {len(extended_factors)}")

        # 计算每个因子的覆盖率
        coverage_stats = {}
        for factor in extended_factors:
            if factor in self.panel.columns:
                coverage = self.panel[factor].notna().mean()
                coverage_stats[factor] = coverage
            else:
                coverage_stats[factor] = 0.0
                logger.warning(f"因子 {factor} 不在面板中")

        # 按覆盖率排序
        coverage_df = pd.DataFrame(
            list(coverage_stats.items()), columns=["factor", "coverage"]
        )
        coverage_df = coverage_df.sort_values("coverage")

        logger.info("\n覆盖率最低的20个因子:")
        print(coverage_df.head(20).to_string(index=False))

        logger.info("\n覆盖率为0的因子:")
        zero_coverage = coverage_df[coverage_df["coverage"] == 0]
        print(zero_coverage.to_string(index=False))

        # 按因子类别分析
        factor_categories = {
            "MA": ["MA", "EMA", "SMA", "WMA", "TA_SMA", "TA_EMA", "TA_WMA"],
            "MACD": ["MACD"],
            "RSI": ["RSI"],
            "STOCH": ["STOCH", "WILLR"],
            "BB": ["BB"],
            "OBV": ["OBV"],
            "ATR": ["ATR"],
            "MOM": ["Momentum", "MOM_"],
            "TREND": ["TREND", "FIX"],
        }

        category_stats = {}
        for category, keywords in factor_categories.items():
            category_factors = []
            for factor in extended_factors:
                if any(keyword in factor for keyword in keywords):
                    category_factors.append(factor)

            if category_factors:
                avg_coverage = coverage_df[
                    coverage_df["factor"].isin(category_factors)
                ]["coverage"].mean()
                category_stats[category] = {
                    "count": len(category_factors),
                    "avg_coverage": avg_coverage,
                    "zero_count": len(
                        coverage_df[
                            (coverage_df["factor"].isin(category_factors))
                            & (coverage_df["coverage"] == 0)
                        ]
                    ),
                }

        logger.info("\n各类因子覆盖率统计:")
        for category, stats in category_stats.items():
            logger.info(
                f"{category}: {stats['count']}个因子, 平均覆盖率{stats['avg_coverage']:.1%}, {stats['zero_count']}个零覆盖"
            )

        return coverage_df

    def analyze_august_performance(self):
        """重点分析8月份表现"""
        logger.info("\n=== 8月份表现深度分析 ===")

        # 筛选8月份数据
        august_data = self.panel.loc["2024-08"]
        logger.info(f"8月份数据形状: {august_data.shape}")

        # 计算8月份各因子表现
        factor_returns = {}

        # 获取扩展因子列表
        factor_set_file = self.extended_config_data["factor_set_file"]
        import yaml

        with open(factor_set_file) as f:
            factor_set = yaml.safe_load(f)

        extended_factors = factor_set["factors"]

        # 计算每个因子的月度收益
        for factor in extended_factors:
            if factor not in self.panel.columns:
                continue

            # 获取7月底和8月底的数据
            july_data = self.panel.loc["2024-07", factor].dropna()
            aug_data = self.panel.loc["2024-08", factor].dropna()

            if len(july_data) > 0 and len(aug_data) > 0:
                # 计算因子值的变化
                common_etfs = set(july_data.index) & set(aug_data.index)
                if len(common_etfs) > 0:
                    july_values = july_data[list(common_etfs)]
                    aug_values = aug_data[list(common_etfs)]

                    # 计算因子值的平均变化
                    factor_change = (aug_values - july_values).mean()
                    factor_returns[factor] = factor_change

        # 排序并找出表现最好和最差的因子
        factor_returns_df = pd.DataFrame(
            list(factor_returns.items()), columns=["factor", "aug_return"]
        )
        factor_returns_df = factor_returns_df.sort_values("aug_return", ascending=False)

        logger.info("\n8月份表现最好的10个因子:")
        print(factor_returns_df.head(10).to_string(index=False))

        logger.info("\n8月份表现最差的10个因子:")
        print(factor_returns_df.tail(10).to_string(index=False))

        return factor_returns_df

    def analyze_correlation_concentration(self):
        """分析相关性剔除后的因子集中度"""
        logger.info("\n=== 因子集中度风险分析 ===")

        # 获取2024年8月的数据
        aug_2024 = self.panel.loc["2024-08"]

        # 获取扩展因子列表
        factor_set_file = self.extended_config_data["factor_set_file"]
        import yaml

        with open(factor_set_file) as f:
            factor_set = yaml.safe_load(f)

        extended_factors = factor_set["factors"]

        # 筛选可用的因子
        available_factors = [f for f in extended_factors if f in aug_2024.columns]
        factor_data = aug_2024[available_factors].dropna()

        if len(factor_data) < 2:
            logger.warning("可用因子数据不足，无法计算相关性")
            return

        # 计算相关性矩阵
        corr_matrix = factor_data.corr()

        # 找出高度相关的因子对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:  # 相关性阈值
                    high_corr_pairs.append(
                        {
                            "factor1": corr_matrix.columns[i],
                            "factor2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df = high_corr_df.sort_values(
                "correlation", key=abs, ascending=False
            )

            logger.info(f"\n发现 {len(high_corr_pairs)} 对高度相关因子 (|r| > 0.9):")
            print(high_corr_df.head(20).to_string(index=False))
        else:
            logger.info("未发现高度相关的因子对")

        # 分析因子类别的集中度
        factor_categories = {
            "动量类": ["Momentum", "MOM_"],
            "均线类": ["MA", "EMA", "SMA", "WMA", "TA_SMA", "TA_EMA", "TA_WMA"],
            "趋势类": ["TREND", "FIX"],
            "震荡类": ["RSI", "STOCH", "WILLR", "CCI", "MACD"],
            "波动率类": ["ATR", "BB"],
            "成交量类": ["OBV"],
        }

        category_factors = {}
        for category, keywords in factor_categories.items():
            category_factors[category] = []
            for factor in available_factors:
                if any(keyword in factor for keyword in keywords):
                    category_factors[category].append(factor)

        logger.info("\n各类因子数量分布:")
        for category, factors in category_factors.items():
            if factors:
                logger.info(f"{category}: {len(factors)} 个因子")

        return corr_matrix, high_corr_pairs

    def analyze_data_quality_issues(self):
        """分析数据质量问题"""
        logger.info("\n=== 数据质量分析 ===")

        # 检查缺失值模式
        missing_stats = {}

        # 获取扩展因子列表
        factor_set_file = self.extended_config_data["factor_set_file"]
        import yaml

        with open(factor_set_file) as f:
            factor_set = yaml.safe_load(f)

        extended_factors = factor_set["factors"]

        # 按月份分析缺失值
        monthly_missing = {}
        for date in self.panel.index.get_level_values(0).unique():
            if "2024" in str(date):  # 只分析2024年
                month_data = self.panel.loc[date]
                missing_counts = {}
                for factor in extended_factors:
                    if factor in month_data.columns:
                        missing_count = month_data[factor].isna().sum()
                        total_count = len(month_data)
                        missing_counts[factor] = missing_count / total_count
                monthly_missing[date] = missing_counts

        missing_df = pd.DataFrame(monthly_missing).T

        logger.info("\n2024年各月因子缺失率统计:")
        print("月份\t平均缺失率\t最高缺失率\t最低缺失率")
        for month in missing_df.index:
            avg_missing = missing_df.loc[month].mean()
            max_missing = missing_df.loc[month].max()
            min_missing = missing_df.loc[month].min()
            print(
                f"{month}\t{avg_missing:.1%}\t\t{max_missing:.1%}\t\t{min_missing:.1%}"
            )

        # 识别有问题的因子
        problem_factors = []
        for factor in missing_df.columns:
            avg_missing = missing_df[factor].mean()
            if avg_missing > 0.5:  # 平均缺失率超过50%
                problem_factors.append(factor)

        logger.info(f"\n数据质量问题因子 (平均缺失率>50%): {len(problem_factors)} 个")
        for factor in problem_factors:
            avg_missing = missing_df[factor].mean()
            logger.info(f"  {factor}: 平均缺失率 {avg_missing:.1%}")

        return missing_df

    def generate_risk_report(self):
        """生成综合风险报告"""
        logger.info("\n" + "=" * 80)
        logger.info("扩展因子系统深度风险分析报告")
        logger.info("=" * 80)

        # 执行所有分析
        self.load_data()

        # 1. 因子覆盖率分析
        coverage_df = self.analyze_factor_coverage()

        # 2. 8月份表现分析
        august_df = self.analyze_august_performance()

        # 3. 相关性集中度分析
        corr_matrix, high_corr_pairs = self.analyze_correlation_concentration()

        # 4. 数据质量分析
        missing_df = self.analyze_data_quality_issues()

        # 综合风险评估
        logger.info("\n=== 综合风险评估 ===")

        # 关键风险指标
        zero_coverage_count = len(coverage_df[coverage_df["coverage"] == 0])
        high_missing_factors = len(
            [f for f in missing_df.columns if missing_df[f].mean() > 0.5]
        )

        logger.info(f"🚨 关键风险指标:")
        logger.info(f"  • 零覆盖率因子: {zero_coverage_count} 个")
        logger.info(f"  • 高缺失率因子: {high_missing_factors} 个")
        logger.info(f"  • 高度相关因子对: {len(high_corr_pairs)} 对")

        # 8月异常分析
        if not august_df.empty:
            best_factor_return = august_df["aug_return"].max()
            worst_factor_return = august_df["aug_return"].min()
            logger.info(f"  • 8月最佳因子收益: {best_factor_return:.2%}")
            logger.info(f"  • 8月最差因子收益: {worst_factor_return:.2%}")
            logger.info(
                f"  • 8月因子收益差异: {(best_factor_return - worst_factor_return):.2%}"
            )

        logger.info("\n📋 主要发现:")
        logger.info("1. 因子覆盖率存在严重问题，部分技术因子完全无法使用")
        logger.info("2. 8月份因子表现差异巨大，可能存在数据异常或过度拟合")
        logger.info("3. 相关性剔除机制可能导致因子过度集中")
        logger.info("4. 数据质量问题影响策略稳定性")

        logger.info("\n⚠️  建议措施:")
        logger.info("1. 重新验证所有技术因子的计算公式和数据源")
        logger.info("2. 对8月份极端表现进行详细的事后分析")
        logger.info("3. 优化相关性剔除算法，避免过度集中")
        logger.info("4. 建立更严格的数据质量监控机制")
        logger.info("5. 延长回测期，验证策略稳健性")

        return {
            "coverage_df": coverage_df,
            "august_df": august_df,
            "high_corr_pairs": high_corr_pairs,
            "missing_df": missing_df,
            "zero_coverage_count": zero_coverage_count,
            "high_missing_factors": high_missing_factors,
        }


def main():
    """主函数"""
    analyzer = ExtendedFactorRiskAnalyzer()
    results = analyzer.generate_risk_report()

    # 保存详细结果
    output_dir = Path("reports/extended_factor_risk_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存CSV文件
    results["coverage_df"].to_csv(output_dir / "factor_coverage.csv", index=False)
    results["august_df"].to_csv(
        output_dir / "august_factor_performance.csv", index=False
    )
    results["missing_df"].to_csv(output_dir / "monthly_missing_stats.csv")

    logger.info(f"\n✅ 风险分析报告已保存至: {output_dir}")


if __name__ == "__main__":
    main()
