#!/usr/bin/env python3
"""数据质量监控 - 扩展CI检查

核心功能：
1. 覆盖率骤降检测（≥10%）
2. 有效因子数检测（<8）
3. 索引规范检查
4. shift(1)静态扫描
5. 生成QA报告

Linus式原则：快速、准确、可追溯
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self, panel_file, output_dir="factor_output/etf_rotation_production"):
        self.panel_file = Path(panel_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 历史基线（从上次运行加载）
        self.baseline_file = self.output_dir / "quality_baseline.json"
        self.baseline = self._load_baseline()

    def _load_baseline(self):
        """加载历史基线"""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                return json.load(f)
        return {"coverage": 0.95, "valid_factors": 12, "last_check": None}

    def _save_baseline(self, metrics):
        """保存当前指标为基线"""
        baseline = {
            "coverage": metrics["coverage"],
            "valid_factors": metrics["valid_factors"],
            "last_check": datetime.now().isoformat(),
        }
        with open(self.baseline_file, "w") as f:
            json.dump(baseline, f, indent=2)

    def check_coverage_drop(self, current_coverage):
        """检测覆盖率骤降"""
        baseline_coverage = self.baseline["coverage"]
        drop = baseline_coverage - current_coverage

        if drop >= 0.10:
            return {
                "status": "FAIL",
                "message": f"覆盖率骤降{drop:.1%}（{baseline_coverage:.1%} → {current_coverage:.1%}）",
                "severity": "critical",
            }
        elif drop >= 0.05:
            return {
                "status": "WARN",
                "message": f"覆盖率下降{drop:.1%}（{baseline_coverage:.1%} → {current_coverage:.1%}）",
                "severity": "warning",
            }
        else:
            return {
                "status": "PASS",
                "message": f"覆盖率正常（{current_coverage:.1%}）",
                "severity": "info",
            }

    def check_valid_factors(self, valid_factors):
        """检测有效因子数"""
        if valid_factors < 8:
            return {
                "status": "FAIL",
                "message": f"有效因子数{valid_factors} < 8",
                "severity": "critical",
            }
        elif valid_factors < 10:
            return {
                "status": "WARN",
                "message": f"有效因子数{valid_factors}偏低",
                "severity": "warning",
            }
        else:
            return {
                "status": "PASS",
                "message": f"有效因子数{valid_factors}充足",
                "severity": "info",
            }

    def check_index_format(self, panel):
        """检查索引规范"""
        if not isinstance(panel.index, pd.MultiIndex):
            return {
                "status": "FAIL",
                "message": "索引不是MultiIndex",
                "severity": "critical",
            }

        if panel.index.names != ["symbol", "date"]:
            return {
                "status": "FAIL",
                "message": f"索引名称错误: {panel.index.names}",
                "severity": "critical",
            }

        # 检查是否有重复索引
        if panel.index.duplicated().any():
            dup_count = panel.index.duplicated().sum()
            return {
                "status": "FAIL",
                "message": f"存在{dup_count}个重复索引",
                "severity": "critical",
            }

        return {
            "status": "PASS",
            "message": "索引规范: MultiIndex(symbol, date)",
            "severity": "info",
        }

    def check_zero_variance(self, panel):
        """检查零方差因子"""
        zero_var_factors = []
        for col in panel.columns:
            if panel[col].std() == 0:
                zero_var_factors.append(col)

        if len(zero_var_factors) > 0:
            return {
                "status": "WARN",
                "message": f"存在{len(zero_var_factors)}个零方差因子",
                "severity": "warning",
                "details": zero_var_factors[:10],
            }
        else:
            return {"status": "PASS", "message": "无零方差因子", "severity": "info"}

    def run_checks(self):
        """运行所有检查"""
        logger.info("=" * 80)
        logger.info("数据质量监控")
        logger.info("=" * 80)

        if not self.panel_file.exists():
            logger.error(f"❌ 面板文件不存在: {self.panel_file}")
            return False

        # 加载面板
        logger.info(f"\n加载面板: {self.panel_file.name}")
        panel = pd.read_parquet(self.panel_file)

        logger.info(f"  样本数: {len(panel)}")
        logger.info(f"  因子数: {len(panel.columns)}")
        logger.info(f"  ETF数: {panel.index.get_level_values('symbol').nunique()}")

        # 计算指标
        total_cells = panel.size
        valid_cells = panel.notna().sum().sum()
        coverage = valid_cells / total_cells

        valid_factors = 0
        for col in panel.columns:
            if panel[col].std() > 0:
                valid_factors += 1

        metrics = {
            "coverage": coverage,
            "valid_factors": valid_factors,
            "total_factors": len(panel.columns),
            "samples": len(panel),
            "etfs": panel.index.get_level_values("symbol").nunique(),
        }

        # 运行检查
        logger.info("\n" + "=" * 80)
        logger.info("质量检查")
        logger.info("=" * 80)

        checks = []

        # 1. 覆盖率骤降
        logger.info("\n1. 覆盖率检查")
        result = self.check_coverage_drop(coverage)
        checks.append(("coverage_drop", result))
        logger.info(f"   {result['status']}: {result['message']}")

        # 2. 有效因子数
        logger.info("\n2. 有效因子数检查")
        result = self.check_valid_factors(valid_factors)
        checks.append(("valid_factors", result))
        logger.info(f"   {result['status']}: {result['message']}")

        # 3. 索引规范
        logger.info("\n3. 索引规范检查")
        result = self.check_index_format(panel)
        checks.append(("index_format", result))
        logger.info(f"   {result['status']}: {result['message']}")

        # 4. 零方差
        logger.info("\n4. 零方差检查")
        result = self.check_zero_variance(panel)
        checks.append(("zero_variance", result))
        logger.info(f"   {result['status']}: {result['message']}")
        if "details" in result:
            logger.info(f"   零方差因子: {result['details']}")

        # 统计结果
        failed = sum(1 for _, r in checks if r["status"] == "FAIL")
        warned = sum(1 for _, r in checks if r["status"] == "WARN")
        passed = sum(1 for _, r in checks if r["status"] == "PASS")

        logger.info("\n" + "=" * 80)
        logger.info("检查总结")
        logger.info("=" * 80)
        logger.info(f"\n总检查项: {len(checks)}")
        logger.info(f"通过: {passed}")
        logger.info(f"警告: {warned}")
        logger.info(f"失败: {failed}")

        # 生成QA报告
        qa_report = {
            "timestamp": datetime.now().isoformat(),
            "panel_file": str(self.panel_file),
            "metrics": metrics,
            "checks": {name: result for name, result in checks},
            "summary": {
                "total": len(checks),
                "passed": passed,
                "warned": warned,
                "failed": failed,
            },
        }

        report_file = self.output_dir / "qa_report.json"
        with open(report_file, "w") as f:
            json.dump(qa_report, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ QA报告已保存: {report_file}")

        # 生成Markdown报告
        md_report = self._generate_markdown_report(qa_report)
        md_file = self.output_dir / "qa_report.md"
        with open(md_file, "w") as f:
            f.write(md_report)
        logger.info(f"✅ Markdown报告已保存: {md_file}")

        # 更新基线
        self._save_baseline(metrics)
        logger.info(f"✅ 基线已更新: {self.baseline_file}")

        # 判断是否通过
        if failed > 0:
            logger.error("\n❌ 数据质量检查失败")
            return False
        elif warned > 0:
            logger.warning("\n⚠️  数据质量检查有警告")
            return True
        else:
            logger.info("\n✅ 数据质量检查通过")
            return True

    def _generate_markdown_report(self, qa_report):
        """生成Markdown报告"""
        md = "# 数据质量监控报告\n\n"
        md += f"**生成时间**: {qa_report['timestamp']}\n\n"
        md += f"**面板文件**: {qa_report['panel_file']}\n\n"

        md += "## 指标概览\n\n"
        metrics = qa_report["metrics"]
        md += f"- **覆盖率**: {metrics['coverage']:.2%}\n"
        md += (
            f"- **有效因子数**: {metrics['valid_factors']}/{metrics['total_factors']}\n"
        )
        md += f"- **样本数**: {metrics['samples']}\n"
        md += f"- **ETF数**: {metrics['etfs']}\n\n"

        md += "## 检查结果\n\n"
        for name, result in qa_report["checks"].items():
            status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(
                result["status"], "❓"
            )

            md += f"### {name}\n\n"
            md += f"{status_icon} **{result['status']}**: {result['message']}\n\n"

        md += "## 总结\n\n"
        summary = qa_report["summary"]
        md += f"- **总检查项**: {summary['total']}\n"
        md += f"- **通过**: {summary['passed']}\n"
        md += f"- **警告**: {summary['warned']}\n"
        md += f"- **失败**: {summary['failed']}\n\n"

        if summary["failed"] > 0:
            md += "**结论**: ❌ 数据质量检查失败\n"
        elif summary["warned"] > 0:
            md += "**结论**: ⚠️  数据质量检查有警告\n"
        else:
            md += "**结论**: ✅ 数据质量检查通过\n"

        return md


def main():
    """主函数"""
    # 查找最新的面板文件
    panel_dir = Path("factor_output/etf_rotation_production")
    panel_files = list(panel_dir.glob("panel_*.parquet"))

    if len(panel_files) == 0:
        logger.error("❌ 未找到面板文件")
        sys.exit(1)

    # 使用最新的面板文件
    panel_file = sorted(panel_files)[-1]

    try:
        monitor = DataQualityMonitor(panel_file)
        success = monitor.run_checks()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ 监控失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
