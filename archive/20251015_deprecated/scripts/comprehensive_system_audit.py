#!/usr/bin/env python3
"""ETF轮动系统全面审计脚本"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ETFSystemAuditor:
    """ETF轮动系统全面审计器"""

    def __init__(self):
        self.audit_results = {}
        self.risks = []

    def audit_data_coverage(self, panel_file: str):
        """审计数据覆盖率"""
        logger.info("=" * 60)
        logger.info("1. 数据覆盖率审计")
        logger.info("=" * 60)

        panel = pd.read_parquet(panel_file)
        dates = panel.index.get_level_values(0).unique()

        coverage_stats = {}
        for factor in panel.columns:
            non_null_count = panel[factor].notna().sum()
            total_count = len(panel)
            coverage = non_null_count / total_count
            coverage_stats[factor] = coverage

            # 检查覆盖率趋势
            monthly_coverage = []
            for i in range(0, len(dates), 30):  # 每30天一个周期
                period_dates = dates[i : i + 30]
                period_panel = panel.loc[period_dates, :]
                period_coverage = period_panel[factor].notna().sum() / len(period_panel)
                monthly_coverage.append(period_coverage)

            coverage_trend = "stable"
            if len(monthly_coverage) > 1:
                if monthly_coverage[-1] > monthly_coverage[0] + 0.1:
                    coverage_trend = "improving"
                elif monthly_coverage[-1] < monthly_coverage[0] - 0.1:
                    coverage_trend = "declining"

            logger.info(f"{factor}: {coverage:.1%} (趋势: {coverage_trend})")

            # 检查风险
            if coverage < 0.5:
                self.risks.append(f"因子 {factor} 覆盖率过低: {coverage:.1%}")
            elif coverage_trend == "declining":
                self.risks.append(f"因子 {factor} 覆盖率呈下降趋势")

        self.audit_results["data_coverage"] = coverage_stats

    def audit_factor_consistency(self, panel_file: str):
        """审计因子一致性"""
        logger.info("=" * 60)
        logger.info("2. 因子一致性审计")
        logger.info("=" * 60)

        panel = pd.read_parquet(panel_file)

        # 检查动量因子之间的关系
        if all(
            col in panel.columns for col in ["Momentum63", "Momentum126", "Momentum252"]
        ):
            # 计算相关性
            correlations = panel[["Momentum63", "Momentum126", "Momentum252"]].corr()
            logger.info("动量因子相关性:")
            logger.info(correlations)

            # 检查异常相关性
            if abs(correlations.loc["Momentum63", "Momentum252"]) > 0.95:
                self.risks.append("Momentum63与Momentum252相关性过高，可能存在计算错误")

            # 检查动量递减关系
            momentum_pairs = [
                ("Momentum63", "Momentum126"),
                ("Momentum126", "Momentum252"),
            ]
            for short, long in momentum_pairs:
                if short in panel.columns and long in panel.columns:
                    valid_data = panel[[short, long]].dropna()
                    if not valid_data.empty:
                        # 长期动量绝对值应该小于等于短期动量
                        violation_ratio = (
                            abs(valid_data[short]) < abs(valid_data[long])
                        ).sum() / len(valid_data)
                        if violation_ratio > 0.3:
                            self.risks.append(f"动量递减关系异常: {short} vs {long}")

        # 检查波动率因子
        if "VOLATILITY_120D" in panel.columns:
            vol_data = panel["VOLATILITY_120D"].dropna()
            if not vol_data.empty:
                # 波动率应该为正
                negative_vol_ratio = (vol_data < 0).sum() / len(vol_data)
                if negative_vol_ratio > 0.01:
                    self.risks.append(f"波动率因子存在负值: {negative_vol_ratio:.1%}")

                # 检查极端值
                extreme_vol_ratio = (vol_data > 1.0).sum() / len(vol_data)
                if extreme_vol_ratio > 0.05:
                    self.risks.append(f"波动率因子存在极端值: {extreme_vol_ratio:.1%}")

    def audit_portfolio_concentration(self, backtest_file: str):
        """审计组合集中度风险"""
        logger.info("=" * 60)
        logger.info("3. 组合集中度审计")
        logger.info("=" * 60)

        if not Path(backtest_file).exists():
            logger.warning(f"回测文件不存在: {backtest_file}")
            return

        # 读取组合权重数据
        portfolio_files = list(Path("rotation_output/backtest").glob("weights_*.csv"))
        if not portfolio_files:
            logger.warning("未找到组合权重文件")
            return

        concentration_stats = []
        for weight_file in sorted(portfolio_files):
            weights_df = pd.read_csv(weight_file, index_col=0)
            weights = weights_df["weight"]

            # 计算集中度指标
            hhi = (weights**2).sum()  # Herfindahl-Hirschman Index
            max_weight = weights.max()
            top3_weight = weights.nlargest(3).sum()

            concentration_stats.append(
                {
                    "date": weight_file.stem.split("_")[1],
                    "hhi": hhi,
                    "max_weight": max_weight,
                    "top3_weight": top3_weight,
                    "n_holdings": len(weights),
                }
            )

        concentration_df = pd.DataFrame(concentration_stats)

        logger.info("组合集中度统计:")
        logger.info(f"平均持仓数量: {concentration_df['n_holdings'].mean():.1f}")
        logger.info(f"平均最大权重: {concentration_df['max_weight'].mean():.1%}")
        logger.info(f"平均HHI: {concentration_df['hhi'].mean():.3f}")

        # 检查集中度风险
        if concentration_df["max_weight"].max() > 0.25:
            self.risks.append(
                f"存在单票权重过高的月份: {concentration_df['max_weight'].max():.1%}"
            )

        if concentration_df["n_holdings"].min() < 5:
            self.risks.append(
                f"存在持仓数量过少的月份: {concentration_df['n_holdings'].min()}"
            )

    def audit_turnover_liquidity(self):
        """审计换手率和流动性风险"""
        logger.info("=" * 60)
        logger.info("4. 换手率和流动性审计")
        logger.info("=" * 60)

        # 检查ETF流动性配置
        with open("etf_rotation/configs/scoring.yaml") as f:
            config = yaml.safe_load(f)

        min_amount = config["liquidity"]["min_amount_20d"]
        logger.info(f"最低成交额要求: {min_amount/1e8:.1f}亿元")

        # 检查是否合理
        if min_amount < 10000000:  # 1000万
            self.risks.append("流动性阈值过低，可能存在流动性风险")
        elif min_amount > 100000000:  # 1亿
            self.risks.append("流动性阈值过高，可能过度限制投资范围")

        # 检查换手率
        portfolio_files = list(Path("rotation_output/backtest").glob("weights_*.csv"))
        if len(portfolio_files) < 2:
            logger.warning("组合文件不足，无法计算换手率")
            return

        turnovers = []
        portfolio_files = sorted(portfolio_files)

        for i in range(1, len(portfolio_files)):
            prev_weights = pd.read_csv(portfolio_files[i - 1], index_col=0)["weight"]
            curr_weights = pd.read_csv(portfolio_files[i], index_col=0)["weight"]

            # 计算换手率
            all_etfs = set(prev_weights.index) | set(curr_weights.index)
            prev_aligned = prev_weights.reindex(all_etfs, fill_value=0)
            curr_aligned = curr_weights.reindex(all_etfs, fill_value=0)

            turnover = 0.5 * abs(curr_aligned - prev_aligned).sum()
            turnovers.append(turnover)

        avg_turnover = np.mean(turnovers)
        logger.info(f"平均月度换手率: {avg_turnover:.1%}")

        # 检查换手率风险
        if avg_turnover > 0.8:
            self.risks.append(f"换手率过高: {avg_turnover:.1%}")
        elif avg_turnover < 0.1:
            self.risks.append(f"换手率过低: {avg_turnover:.1%}")

    def audit_market_regime_adaptation(self):
        """审计市场环境适应性"""
        logger.info("=" * 60)
        logger.info("5. 市场环境适应性审计")
        logger.info("=" * 60)

        # 读取回测结果
        performance_file = Path("rotation_output/backtest/performance_metrics.csv")
        if not performance_file.exists():
            logger.warning("未找到绩效指标文件")
            return

        perf_df = pd.read_csv(performance_file)

        # 分析市场环境
        # 2024年市场特点：9月24日政策驱动的大反弹
        # 需要检查策略在不同市场环境下的表现

        logger.info("市场环境分析:")
        logger.info("- 2024年1-9月: 震荡下行")
        logger.info("- 2024年9-12月: 政策驱动反弹")

        # 检查回撤控制
        max_dd = perf_df.loc[0, "max_drawdown"]
        if max_dd > -0.15:
            logger.warning(f"最大回撤较大: {max_dd:.1%}")
            self.risks.append(f"风险控制能力待提升，最大回撤: {max_dd:.1%}")

        # 检查收益稳定性
        annual_return = perf_df.loc[0, "annual_return"]
        volatility = perf_df.loc[0, "volatility"]
        sharpe = perf_df.loc[0, "sharpe"]

        if sharpe < 1.0:
            self.risks.append(f"夏普比率偏低: {sharpe:.2f}")

        if volatility > 0.15:
            self.risks.append(f"波动率偏高: {volatility:.1%}")

    def audit_implementation_risks(self):
        """审计实施风险"""
        logger.info("=" * 60)
        logger.info("6. 实施风险审计")
        logger.info("=" * 60)

        # 检查数据依赖
        data_files = {
            "ETF价格数据": Path("raw/ETF/daily"),
            "因子面板": Path("factor_output/etf_rotation"),
            "配置文件": Path("etf_rotation/configs"),
        }

        for name, path in data_files.items():
            if not path.exists():
                self.risks.append(f"关键数据缺失: {name} - {path}")
            elif path.is_dir():
                file_count = len(list(path.glob("*")))
                if file_count == 0:
                    self.risks.append(f"目录为空: {name} - {path}")
                else:
                    logger.info(f"{name}: {file_count} 个文件")

        # 检查脚本完整性
        required_scripts = [
            "scripts/produce_etf_panel.py",
            "scripts/etf_monthly_rotation.py",
            "scripts/backtest_12months.py",
            "scripts/verify_no_lookahead.py",
        ]

        for script in required_scripts:
            if not Path(script).exists():
                self.risks.append(f"核心脚本缺失: {script}")

        # 检查模块依赖
        required_modules = [
            "etf_rotation/universe_manager.py",
            "etf_rotation/scorer.py",
            "etf_rotation/portfolio.py",
        ]

        for module in required_modules:
            if not Path(module).exists():
                self.risks.append(f"核心模块缺失: {module}")

    def generate_audit_report(self):
        """生成审计报告"""
        logger.info("=" * 60)
        logger.info("7. 审计总结")
        logger.info("=" * 60)

        if not self.risks:
            logger.info("✅ 未发现重大风险")
            risk_level = "低"
        else:
            logger.info(f"⚠️  发现 {len(self.risks)} 项潜在风险:")
            for i, risk in enumerate(self.risks, 1):
                logger.info(f"  {i}. {risk}")

            if len(self.risks) <= 3:
                risk_level = "中等"
            else:
                risk_level = "高"

        # 生成报告
        report = {
            "audit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_level": risk_level,
            "total_risks": len(self.risks),
            "risks": self.risks,
            "audit_results": self.audit_results,
        }

        # 保存报告
        report_file = Path("rotation_output/audit_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"审计报告已保存: {report_file}")

        # 给出建议
        logger.info("=" * 60)
        logger.info("改进建议:")
        logger.info("=" * 60)

        if risk_level == "低":
            logger.info("✅ 系统运行良好，建议定期审计")
        else:
            logger.info("1. 优先处理高风险项目")
            logger.info("2. 加强数据质量监控")
            logger.info("3. 完善风险控制机制")
            logger.info("4. 增加异常检测逻辑")
            logger.info("5. 定期进行系统审计")

    def run_full_audit(self, panel_file: str = None):
        """运行全面审计"""
        logger.info("开始ETF轮动系统全面审计")

        if panel_file is None:
            panel_file = "factor_output/etf_rotation/panel_20200101_20251014.parquet"

        try:
            self.audit_data_coverage(panel_file)
            self.audit_factor_consistency(panel_file)
            self.audit_portfolio_concentration(
                "rotation_output/backtest/backtest_summary.csv"
            )
            self.audit_turnover_liquidity()
            self.audit_market_regime_adaptation()
            self.audit_implementation_risks()
            self.generate_audit_report()

        except Exception as e:
            logger.error(f"审计过程出错: {e}")
            self.risks.append(f"审计执行失败: {str(e)}")


def main():
    auditor = ETFSystemAuditor()
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
