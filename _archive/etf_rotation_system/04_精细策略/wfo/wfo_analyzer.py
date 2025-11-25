#!/usr/bin/env python3
"""
WFO分析器
分析IS/OOS表现，检测过拟合
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class WFOAnalyzer:
    """WFO分析器 - IS/OOS对比与过拟合检测"""

    def __init__(self):
        """初始化WFO分析器"""
        logger.info("WFO分析器初始化完成")

    def analyze_overfitting(self, periods: List) -> Dict:
        """
        分析所有周期的过拟合情况

        Args:
            periods: WFO周期列表

        Returns:
            过拟合分析结果
        """
        logger.info("=" * 80)
        logger.info("开始过拟合分析")
        logger.info("=" * 80)

        if not periods:
            logger.warning("没有周期数据可供分析")
            return {}

        # 提取IS和OOS性能
        is_sharpes = []
        oos_sharpes = []
        period_ids = []

        for period in periods:
            if period.is_results and period.oos_results:
                is_sharpe = period.is_results.get("best_sharpe", 0)
                oos_sharpe = period.oos_results.get("avg_sharpe", 0)

                is_sharpes.append(is_sharpe)
                oos_sharpes.append(oos_sharpe)
                period_ids.append(period.period_id)

        if not is_sharpes:
            logger.warning("没有有效的IS/OOS数据")
            return {}

        # 计算关键指标
        analysis = {
            "periods_analyzed": len(is_sharpes),
            "metrics": self._calculate_metrics(is_sharpes, oos_sharpes),
            "period_details": self._generate_period_details(
                period_ids, is_sharpes, oos_sharpes
            ),
            "recommendations": self._generate_recommendations(is_sharpes, oos_sharpes),
        }

        # 打印分析结果
        self._print_analysis(analysis)

        return analysis

    def _calculate_metrics(
        self, is_sharpes: List[float], oos_sharpes: List[float]
    ) -> Dict:
        """计算过拟合指标"""
        is_sharpes = np.array(is_sharpes)
        oos_sharpes = np.array(oos_sharpes)

        # 1. 过拟合比 (Overfitting Ratio)
        overfit_ratios = is_sharpes / np.maximum(oos_sharpes, 0.01)
        avg_overfit_ratio = np.mean(overfit_ratios)

        # 2. 性能衰减 (Performance Decay)
        decay_rates = (is_sharpes - oos_sharpes) / np.maximum(is_sharpes, 0.01)
        avg_decay_rate = np.mean(decay_rates)

        # 3. 相关性 (IS-OOS Correlation)
        if len(is_sharpes) > 1:
            correlation = np.corrcoef(is_sharpes, oos_sharpes)[0, 1]
        else:
            correlation = 0.0

        # 4. OOS胜率 (OOS Win Rate)
        oos_win_rate = np.sum(oos_sharpes > 0.3) / len(oos_sharpes)

        # 5. 稳定性 (Stability)
        is_cv = np.std(is_sharpes) / np.maximum(np.mean(is_sharpes), 0.01)  # 变异系数
        oos_cv = np.std(oos_sharpes) / np.maximum(np.mean(oos_sharpes), 0.01)

        # 6. 最差周期 (Worst Period)
        worst_oos_sharpe = np.min(oos_sharpes)
        worst_period_idx = np.argmin(oos_sharpes)

        return {
            "is_sharpe_mean": float(np.mean(is_sharpes)),
            "is_sharpe_std": float(np.std(is_sharpes)),
            "oos_sharpe_mean": float(np.mean(oos_sharpes)),
            "oos_sharpe_std": float(np.std(oos_sharpes)),
            "avg_overfit_ratio": float(avg_overfit_ratio),
            "max_overfit_ratio": float(np.max(overfit_ratios)),
            "min_overfit_ratio": float(np.min(overfit_ratios)),
            "avg_decay_rate": float(avg_decay_rate),
            "is_oos_correlation": float(correlation),
            "oos_win_rate": float(oos_win_rate),
            "is_stability_cv": float(is_cv),
            "oos_stability_cv": float(oos_cv),
            "worst_oos_sharpe": float(worst_oos_sharpe),
            "worst_period_index": int(worst_period_idx) + 1,
        }

    def _generate_period_details(
        self, period_ids: List[int], is_sharpes: List[float], oos_sharpes: List[float]
    ) -> List[Dict]:
        """生成每个周期的详细信息"""
        details = []

        for pid, is_s, oos_s in zip(period_ids, is_sharpes, oos_sharpes):
            overfit_ratio = is_s / max(oos_s, 0.01)
            decay_rate = (is_s - oos_s) / max(is_s, 0.01)

            details.append(
                {
                    "period_id": pid,
                    "is_sharpe": round(is_s, 4),
                    "oos_sharpe": round(oos_s, 4),
                    "overfit_ratio": round(overfit_ratio, 3),
                    "decay_rate": round(decay_rate * 100, 2),  # 百分比
                    "pass_threshold": oos_s >= 0.3,
                }
            )

        return details

    def _generate_recommendations(
        self, is_sharpes: List[float], oos_sharpes: List[float]
    ) -> Dict:
        """生成部署建议"""
        metrics = self._calculate_metrics(is_sharpes, oos_sharpes)

        # 评估等级
        grade = self._assess_grade(metrics)

        # 风险评估
        risks = self._assess_risks(metrics)

        # 行动建议
        actions = self._suggest_actions(metrics, grade)

        return {
            "overall_grade": grade,
            "risks": risks,
            "suggested_actions": actions,
            "deployment_ready": grade in ["A", "B"],
        }

    def _assess_grade(self, metrics: Dict) -> str:
        """评估等级"""
        score = 100

        # 过拟合比
        if metrics["avg_overfit_ratio"] > 1.5:
            score -= 30
        elif metrics["avg_overfit_ratio"] > 1.2:
            score -= 15

        # 性能衰减
        if metrics["avg_decay_rate"] > 0.25:
            score -= 25
        elif metrics["avg_decay_rate"] > 0.15:
            score -= 10

        # OOS胜率
        if metrics["oos_win_rate"] < 0.5:
            score -= 20
        elif metrics["oos_win_rate"] < 0.7:
            score -= 10

        # OOS平均表现
        if metrics["oos_sharpe_mean"] < 0.3:
            score -= 25
        elif metrics["oos_sharpe_mean"] < 0.4:
            score -= 10

        # IS-OOS相关性
        if metrics["is_oos_correlation"] < 0.3:
            score -= 10

        # 评级
        if score >= 90:
            return "A"  # 优秀
        elif score >= 75:
            return "B"  # 良好
        elif score >= 60:
            return "C"  # 及格
        elif score >= 45:
            return "D"  # 需改进
        else:
            return "F"  # 不及格

    def _assess_risks(self, metrics: Dict) -> List[str]:
        """评估风险"""
        risks = []

        if metrics["avg_overfit_ratio"] > 1.5:
            risks.append("严重过拟合 - IS/OOS性能差距过大")
        elif metrics["avg_overfit_ratio"] > 1.2:
            risks.append("轻度过拟合 - IS/OOS性能有差距")

        if metrics["avg_decay_rate"] > 0.25:
            risks.append("高性能衰减 - OOS表现显著低于IS")

        if metrics["oos_win_rate"] < 0.5:
            risks.append("低OOS胜率 - 多数周期表现不佳")

        if metrics["worst_oos_sharpe"] < 0:
            risks.append(f"存在负Sharpe周期 (Period {metrics['worst_period_index']})")

        if metrics["is_oos_correlation"] < 0.3:
            risks.append("IS/OOS低相关性 - 参数不稳定")

        if metrics["oos_stability_cv"] > 0.5:
            risks.append("OOS波动大 - 不同周期表现差异显著")

        if not risks:
            risks.append("未发现明显风险")

        return risks

    def _suggest_actions(self, metrics: Dict, grade: str) -> List[str]:
        """建议行动"""
        actions = []

        if grade in ["A", "B"]:
            actions.append("✅ 可以考虑实盘部署")
            actions.append("建议: 小资金先行验证，密切监控")
        elif grade == "C":
            actions.append("⚠️  谨慎部署，需要进一步优化")
            actions.append("建议: 缩小参数空间，增强稳健性")
        else:
            actions.append("❌ 不建议部署")
            actions.append("建议: 重新设计因子或策略逻辑")

        # 具体优化建议
        if metrics["avg_overfit_ratio"] > 1.3:
            actions.append("减少训练窗口复杂度，避免过拟合")

        if metrics["oos_win_rate"] < 0.7:
            actions.append("提高IS期筛选标准，确保更稳定的策略进入OOS")

        if metrics["is_oos_correlation"] < 0.5:
            actions.append("增加参数约束，提高策略一致性")

        return actions

    def _print_analysis(self, analysis: Dict):
        """打印分析结果"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 过拟合分析结果")
        logger.info("=" * 80)

        metrics = analysis["metrics"]

        logger.info("\n【核心指标】")
        logger.info(
            f"  IS平均Sharpe: {metrics['is_sharpe_mean']:.3f} ± {metrics['is_sharpe_std']:.3f}"
        )
        logger.info(
            f"  OOS平均Sharpe: {metrics['oos_sharpe_mean']:.3f} ± {metrics['oos_sharpe_std']:.3f}"
        )
        logger.info(
            f"  平均过拟合比: {metrics['avg_overfit_ratio']:.3f} (范围: {metrics['min_overfit_ratio']:.2f} ~ {metrics['max_overfit_ratio']:.2f})"
        )
        logger.info(f"  平均性能衰减: {metrics['avg_decay_rate']*100:.1f}%")
        logger.info(f"  IS-OOS相关性: {metrics['is_oos_correlation']:.3f}")
        logger.info(f"  OOS胜率: {metrics['oos_win_rate']*100:.1f}%")

        logger.info("\n【稳定性】")
        logger.info(f"  IS变异系数: {metrics['is_stability_cv']:.3f}")
        logger.info(f"  OOS变异系数: {metrics['oos_stability_cv']:.3f}")
        logger.info(
            f"  最差OOS Sharpe: {metrics['worst_oos_sharpe']:.3f} (Period {metrics['worst_period_index']})"
        )

        recommendations = analysis["recommendations"]

        logger.info("\n【综合评级】")
        logger.info(f"  等级: {recommendations['overall_grade']}")
        logger.info(
            f"  可部署: {'是' if recommendations['deployment_ready'] else '否'}"
        )

        logger.info("\n【风险评估】")
        for risk in recommendations["risks"]:
            logger.info(f"  • {risk}")

        logger.info("\n【行动建议】")
        for action in recommendations["suggested_actions"]:
            logger.info(f"  • {action}")

        logger.info("\n" + "=" * 80)

    def save_analysis(self, analysis: Dict, output_path: Path):
        """保存分析结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        logger.info(f"分析结果已保存: {output_path}")

    def generate_report(self, analysis: Dict, output_path: Path):
        """生成Markdown报告"""
        metrics = analysis["metrics"]
        recommendations = analysis["recommendations"]
        period_details = analysis["period_details"]

        report = []
        report.append("# WFO (Walk-Forward Optimization) 分析报告\n")
        report.append(f"**生成时间**: {np.datetime64('now').astype(str)}\n")
        report.append(f"**分析周期数**: {analysis['periods_analyzed']}\n")
        report.append("\n---\n")

        report.append("## 📊 核心指标\n")
        report.append("| 指标 | IS | OOS | 差异 |\n")
        report.append("|------|-----|-----|------|\n")
        report.append(
            f"| 平均Sharpe | {metrics['is_sharpe_mean']:.3f} | {metrics['oos_sharpe_mean']:.3f} | {metrics['avg_decay_rate']*100:.1f}% |\n"
        )
        report.append(
            f"| 标准差 | {metrics['is_sharpe_std']:.3f} | {metrics['oos_sharpe_std']:.3f} | - |\n"
        )
        report.append(
            f"| 变异系数 | {metrics['is_stability_cv']:.3f} | {metrics['oos_stability_cv']:.3f} | - |\n"
        )
        report.append("\n")

        report.append("## 🔍 过拟合检测\n")
        report.append(f"- **平均过拟合比**: {metrics['avg_overfit_ratio']:.3f}\n")
        report.append(
            f"- **过拟合比范围**: [{metrics['min_overfit_ratio']:.2f}, {metrics['max_overfit_ratio']:.2f}]\n"
        )
        report.append(f"- **IS-OOS相关性**: {metrics['is_oos_correlation']:.3f}\n")
        report.append(f"- **OOS胜率**: {metrics['oos_win_rate']*100:.1f}%\n")
        report.append("\n")

        report.append("## 📈 周期明细\n")
        report.append(
            "| Period | IS Sharpe | OOS Sharpe | 过拟合比 | 衰减率 | 通过 |\n"
        )
        report.append(
            "|--------|-----------|------------|----------|--------|------|\n"
        )
        for detail in period_details:
            pass_icon = "✅" if detail["pass_threshold"] else "❌"
            report.append(
                f"| {detail['period_id']} | {detail['is_sharpe']:.3f} | "
                f"{detail['oos_sharpe']:.3f} | {detail['overfit_ratio']:.2f} | "
                f"{detail['decay_rate']:.1f}% | {pass_icon} |\n"
            )
        report.append("\n")

        report.append("## 🎯 综合评级\n")
        report.append(f"**等级**: {recommendations['overall_grade']}\n\n")
        report.append(
            f"**可部署**: {'✅ 是' if recommendations['deployment_ready'] else '❌ 否'}\n\n"
        )

        report.append("### 风险评估\n")
        for risk in recommendations["risks"]:
            report.append(f"- {risk}\n")
        report.append("\n")

        report.append("### 行动建议\n")
        for action in recommendations["suggested_actions"]:
            report.append(f"- {action}\n")
        report.append("\n")

        report.append("---\n")
        report.append("*报告由WFO分析器自动生成*\n")

        # 保存
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(report)

        logger.info(f"Markdown报告已生成: {output_path}")
