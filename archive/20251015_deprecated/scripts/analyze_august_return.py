#!/usr/bin/env python3
"""
深度分析8月极端收益来源 - 逐月交易记录分析
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AugustReturnAnalyzer:
    def __init__(self):
        self.backtest_results_extended = (
            "rotation_output/backtest/backtest_summary_extended.csv"
        )
        self.backtest_results_core = "rotation_output/backtest/backtest_summary.csv"
        self.performance_extended = (
            "rotation_output/backtest/performance_metrics_extended.csv"
        )
        self.performance_core = "rotation_output/backtest/performance_metrics.csv"
        self.panel_file = "factor_output/etf_rotation/panel_20240101_20251014.parquet"

    def load_backtest_data(self):
        """加载回测数据"""
        logger.info("加载回测数据...")

        # 扩展系统结果
        self.extended_summary = pd.read_csv(self.backtest_results_extended)
        self.extended_performance = pd.read_csv(self.performance_extended)

        # 核心系统结果
        self.core_summary = pd.read_csv(self.backtest_results_core)
        self.core_performance = pd.read_csv(self.performance_core)

        # 因子面板
        self.panel = pd.read_parquet(self.panel_file)

        logger.info(f"扩展系统回测期数: {len(self.extended_summary)}")
        logger.info(f"核心系统回测期数: {len(self.core_summary)}")

    def calculate_monthly_returns(self):
        """计算逐月收益率"""
        logger.info("\n=== 逐月收益率分析 ===")

        # 从回测结果中提取月度收益
        extended_returns = []
        core_returns = []
        dates = []

        # 假设我们有月度收益数据，这里需要从实际的回测结果中计算
        # 由于我们只有汇总数据，需要重新构建月度收益

        # 从performance metrics中提取年化收益和波动率
        ext_annual_return = self.extended_performance["annual_return"].iloc[0]
        ext_volatility = self.extended_performance["volatility"].iloc[0]
        core_annual_return = self.core_performance["annual_return"].iloc[0]
        core_volatility = self.core_performance["volatility"].iloc[0]

        logger.info(f"扩展系统年化收益: {ext_annual_return:.2%}")
        logger.info(f"核心系统年化收益: {core_annual_return:.2%}")
        logger.info(f"收益差异: {(ext_annual_return - core_annual_return):.2%}")

        logger.info(f"扩展系统年化波动: {ext_volatility:.2%}")
        logger.info(f"核心系统年化波动: {core_volatility:.2%}")
        logger.info(f"波动差异: {(ext_volatility - core_volatility):.2%}")

        # 估算月度收益（简化假设）
        ext_monthly_return = ext_annual_return / 12
        core_monthly_return = core_annual_return / 12

        logger.info(f"估算扩展系统平均月收益: {ext_monthly_return:.2%}")
        logger.info(f"估算核心系统平均月收益: {core_monthly_return:.2%}")

        return {
            "ext_annual_return": ext_annual_return,
            "core_annual_return": core_annual_return,
            "return_difference": ext_annual_return - core_annual_return,
            "ext_volatility": ext_volatility,
            "core_volatility": core_volatility,
            "volatility_difference": ext_volatility - core_volatility,
        }

    def analyze_august_factors(self):
        """分析8月份使用的具体因子"""
        logger.info("\n=== 8月份因子使用分析 ===")

        # 查看8月30日的回测结果（8月最后一个交易日）
        august_row = self.extended_summary[
            self.extended_summary["trade_date"] == 20240830
        ]

        if not august_row.empty:
            logger.info("8月30日回测结果:")
            logger.info(f"  宇宙大小: {august_row['universe_size'].iloc[0]}")
            logger.info(f"  评分大小: {august_row['scored_size'].iloc[0]}")
            logger.info(f"  组合大小: {august_row['portfolio_size'].iloc[0]}")

            # 对比前后月份
            july_row = self.extended_summary[
                self.extended_summary["trade_date"] == 20240731
            ]
            sept_row = self.extended_summary[
                self.extended_summary["trade_date"] == 20240930
            ]

            logger.info("\n对比分析:")
            if not july_row.empty:
                logger.info(
                    f"7月31日: 宇宙{july_row['universe_size'].iloc[0]}, 评分{july_row['scored_size'].iloc[0]}, 组合{july_row['portfolio_size'].iloc[0]}"
                )
            if not sept_row.empty:
                logger.info(
                    f"9月30日: 宇宙{sept_row['universe_size'].iloc[0]}, 评分{sept_row['scored_size'].iloc[0]}, 组合{sept_row['portfolio_size'].iloc[0]}"
                )

        return august_row

    def analyze_factor_contribution(self):
        """分析因子贡献度"""
        logger.info("\n=== 因子贡献度分析 ===")

        # 由于我们没有详细的组合持仓数据，这里分析因子面板的统计特征

        # 筛选2024年8月的数据
        august_mask = self.panel.index.get_level_values(0).str.contains("2024-08")
        august_data = self.panel.loc[august_mask]

        logger.info(f"8月份数据点数: {len(august_data)}")
        logger.info(
            f"8月份ETF数量: {len(august_data.index.get_level_values(1).unique())}"
        )

        # 分析核心因子在8月的表现
        core_factors = ["Momentum252", "Momentum126", "Momentum63", "VOLATILITY_120D"]

        for factor in core_factors:
            if factor in august_data.columns:
                factor_data = august_data[factor].dropna()
                if len(factor_data) > 0:
                    logger.info(f"\n{factor} 8月统计:")
                    logger.info(f"  平均值: {factor_data.mean():.4f}")
                    logger.info(f"  标准差: {factor_data.std():.4f}")
                    logger.info(f"  最小值: {factor_data.min():.4f}")
                    logger.info(f"  最大值: {factor_data.max():.4f}")
                    logger.info(f"  覆盖率: {len(factor_data)/len(august_data):.1%}")

        # 分析扩展因子集中的技术因子
        tech_factors = ["RSI14", "MACD", "STOCH", "ATR14", "BB_10_2.0_Width"]

        logger.info("\n--- 扩展因子表现 ---")
        for factor in tech_factors:
            if factor in august_data.columns:
                factor_data = august_data[factor].dropna()
                if len(factor_data) > 0:
                    logger.info(
                        f"{factor}: 均值{factor_data.mean():.4f}, 标准差{factor_data.std():.4f}, 覆盖率{len(factor_data)/len(august_data):.1%}"
                    )

        return august_data

    def detect_data_anomalies(self):
        """检测数据异常"""
        logger.info("\n=== 数据异常检测 ===")

        # 检查8月份是否有极端值
        august_mask = self.panel.index.get_level_values(0).str.contains("2024-08")
        august_data = self.panel.loc[august_mask]

        anomalies = []

        # 检查每个因子
        numeric_columns = august_data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col.startswith(("Momentum", "RSI", "MACD", "STOCH", "ATR", "BB_")):
                data = august_data[col].dropna()
                if len(data) > 0:
                    # 使用IQR方法检测异常值
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    if len(outliers) > 0:
                        anomalies.append(
                            {
                                "factor": col,
                                "outlier_count": len(outliers),
                                "outlier_percentage": len(outliers) / len(data) * 100,
                                "min_outlier": outliers.min(),
                                "max_outlier": outliers.max(),
                            }
                        )

        if anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_df = anomalies_df.sort_values(
                "outlier_percentage", ascending=False
            )

            logger.info("发现数据异常的因子:")
            print(anomalies_df.head(10).to_string(index=False))
        else:
            logger.info("未检测到明显的数据异常")

        return anomalies

    def generate_august_report(self):
        """生成8月极端收益分析报告"""
        logger.info("\n" + "=" * 80)
        logger.info("8月极端收益深度分析报告")
        logger.info("=" * 80)

        # 执行所有分析
        self.load_backtest_data()

        # 1. 月度收益分析
        monthly_analysis = self.calculate_monthly_returns()

        # 2. 8月因子分析
        august_analysis = self.analyze_august_factors()

        # 3. 因子贡献分析
        factor_contribution = self.analyze_factor_contribution()

        # 4. 异常检测
        anomalies = self.detect_data_anomalies()

        # 综合分析结果
        logger.info("\n=== 综合分析结论 ===")

        # 关键发现
        logger.info("🔍 关键发现:")
        logger.info(
            f"1. 扩展系统年化收益比核心系统高 {monthly_analysis['return_difference']:.2%}"
        )
        logger.info(
            f"2. 扩展系统波动率比核心系统高 {monthly_analysis['volatility_difference']:.2%}"
        )
        logger.info(
            f"3. 收益提升伴随着{monthly_analysis['volatility_difference']/monthly_analysis['core_volatility']:.1f}倍的波动率增加"
        )

        # 8月特殊情况分析
        if not august_analysis.empty:
            logger.info(
                f"\n4. 8月份宇宙大小: {august_analysis['universe_size'].iloc[0]}"
            )
            logger.info(
                f"5. 8月份评分ETF数量: {august_analysis['scored_size'].iloc[0]}"
            )
            logger.info(
                f"6. 8月份组合大小: {august_analysis['portfolio_size'].iloc[0]}"
            )

        # 异常分析
        if anomalies:
            logger.info(f"\n7. 检测到 {len(anomalies)} 个因子存在数据异常")
            high_anomaly_factors = [a for a in anomalies if a["outlier_percentage"] > 5]
            if high_anomaly_factors:
                logger.info(f"8. {len(high_anomaly_factors)} 个因子异常值比例超过5%")

        # 风险评估
        logger.info("\n⚠️  风险评估:")

        # 波动率风险
        vol_ratio = (
            monthly_analysis["ext_volatility"] / monthly_analysis["core_volatility"]
        )
        if vol_ratio > 2:
            logger.info(f"🚨 高风险: 扩展系统波动率是核心系统的 {vol_ratio:.1f} 倍")
        elif vol_ratio > 1.5:
            logger.info(f"⚠️  中风险: 扩展系统波动率是核心系统的 {vol_ratio:.1f} 倍")
        else:
            logger.info(f"✅ 低风险: 扩展系统波动率是核心系统的 {vol_ratio:.1f} 倍")

        # 收益质量风险
        sharpe_ext = (
            monthly_analysis["ext_annual_return"] / monthly_analysis["ext_volatility"]
        )
        sharpe_core = (
            monthly_analysis["core_annual_return"] / monthly_analysis["core_volatility"]
        )

        logger.info(f"\n扩展系统夏普比率: {sharpe_ext:.2f}")
        logger.info(f"核心系统夏普比率: {sharpe_core:.2f}")

        if sharpe_ext < sharpe_core:
            logger.info("🚨 警告: 扩展系统风险调整后收益低于核心系统")
        else:
            logger.info("✅ 扩展系统风险调整后收益优于核心系统")

        # 数据质量风险
        if anomalies and len([a for a in anomalies if a["outlier_percentage"] > 5]) > 0:
            logger.info("🚨 数据质量风险: 多个因子存在异常值，可能影响策略稳定性")

        logger.info("\n📋 建议措施:")
        logger.info("1. 重新检查8月份的具体交易记录和持仓变化")
        logger.info("2. 验证扩展因子的计算逻辑和数据源质量")
        logger.info("3. 分析相关性剔除机制是否过度集中")
        logger.info("4. 考虑降低扩展因子的权重或增加风险控制")
        logger.info("5. 延长回测期验证策略稳健性")
        logger.info("6. 实施更严格的数据质量监控")

        return {
            "monthly_analysis": monthly_analysis,
            "august_analysis": august_analysis,
            "anomalies": anomalies,
            "sharpe_ext": sharpe_ext,
            "sharpe_core": sharpe_core,
            "vol_ratio": vol_ratio,
        }


def main():
    """主函数"""
    analyzer = AugustReturnAnalyzer()
    results = analyzer.generate_august_report()

    # 保存结果
    output_dir = Path("reports/august_return_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存关键数据
    if results["anomalies"]:
        anomalies_df = pd.DataFrame(results["anomalies"])
        anomalies_df.to_csv(output_dir / "data_anomalies.csv", index=False)

    logger.info(f"\n✅ 8月收益分析报告已保存至: {output_dir}")


if __name__ == "__main__":
    main()
