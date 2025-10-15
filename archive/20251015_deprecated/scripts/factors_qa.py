#!/usr/bin/env python3
"""因子质量保证（QA）- 覆盖率/零方差/系列差异/静态扫描

Day 3任务：生成whitelist.yaml和qa_report.md
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FactorQA:
    """因子质量保证检查器"""

    def __init__(self, panel_file: str, output_dir: str = "factor_output/etf_rotation"):
        self.panel_file = panel_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载面板
        logger.info(f"加载因子面板: {panel_file}")
        self.panel = pd.read_parquet(panel_file)
        logger.info(f"面板形状: {self.panel.shape}")

        # QA结果
        self.qa_results = {}

    def check_coverage(self, threshold: float = 0.8) -> dict:
        """检查因子覆盖率（横截面非NaN占比）"""
        logger.info("\n" + "=" * 60)
        logger.info("1. 因子覆盖率检查（阈值≥80%）")
        logger.info("=" * 60)

        coverage = {}
        for col in self.panel.columns:
            cov = self.panel[col].notna().mean()
            coverage[col] = cov

            status = "✅" if cov >= threshold else "❌"
            logger.info(f"{status} {col}: {cov:.1%}")

        passed = [k for k, v in coverage.items() if v >= threshold]
        failed = [k for k, v in coverage.items() if v < threshold]

        logger.info(f"\n通过: {len(passed)}/{len(coverage)} 个因子")
        if failed:
            logger.warning(f"未通过: {', '.join(failed)}")

        return {
            "coverage": coverage,
            "passed": passed,
            "failed": failed,
            "threshold": threshold,
        }

    def check_zero_variance(self) -> dict:
        """检查零方差因子（整列NaN或零方差）"""
        logger.info("\n" + "=" * 60)
        logger.info("2. 零方差检查")
        logger.info("=" * 60)

        zero_var_factors = []
        all_nan_factors = []

        for col in self.panel.columns:
            # 检查全NaN
            if self.panel[col].isna().all():
                all_nan_factors.append(col)
                logger.warning(f"❌ {col}: 全部NaN")
                continue

            # 检查零方差
            var = self.panel[col].var()
            if pd.isna(var) or var == 0:
                zero_var_factors.append(col)
                logger.warning(f"❌ {col}: 零方差")
            else:
                logger.info(f"✅ {col}: 方差={var:.6f}")

        return {
            "zero_variance": zero_var_factors,
            "all_nan": all_nan_factors,
            "passed": len(self.panel.columns)
            - len(zero_var_factors)
            - len(all_nan_factors),
        }

    def check_factor_family_diff(self) -> dict:
        """检查同家族不同参数因子的差异性"""
        logger.info("\n" + "=" * 60)
        logger.info("3. 因子家族差异性检查")
        logger.info("=" * 60)

        # 定义因子家族
        families = {
            "Momentum": ["Momentum63", "Momentum126", "Momentum252"],
        }

        diff_results = {}
        for family, factors in families.items():
            available = [f for f in factors if f in self.panel.columns]
            if len(available) < 2:
                logger.warning(f"⚠️  {family}: 可用因子不足2个，跳过")
                continue

            logger.info(f"\n{family}家族: {', '.join(available)}")

            # 计算两两差异
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    f1, f2 = available[i], available[j]

                    # 提取共同非NaN样本
                    mask = self.panel[f1].notna() & self.panel[f2].notna()
                    if mask.sum() < 10:
                        logger.warning(f"  {f1} vs {f2}: 共同样本不足10个")
                        continue

                    diff = self.panel.loc[mask, f1] - self.panel.loc[mask, f2]
                    var_diff = diff.var()
                    corr = self.panel.loc[mask, f1].corr(self.panel.loc[mask, f2])

                    # 差异性判断：方差>0.001 或 相关性<0.99
                    is_different = var_diff > 0.001 or corr < 0.99
                    status = "✅" if is_different else "❌"

                    logger.info(
                        f"  {status} {f1} vs {f2}: var(diff)={var_diff:.6f}, corr={corr:.4f}"
                    )

                    diff_results[f"{f1}_vs_{f2}"] = {
                        "var_diff": var_diff,
                        "corr": corr,
                        "is_different": is_different,
                    }

        return diff_results

    def static_scan_lookahead(self) -> dict:
        """静态扫描源码中的未来函数"""
        logger.info("\n" + "=" * 60)
        logger.info("4. 未来函数静态扫描")
        logger.info("=" * 60)

        factor_file = Path("factor_system/factor_engine/factors/etf_momentum.py")
        if not factor_file.exists():
            logger.warning(f"因子文件不存在: {factor_file}")
            return {}

        with open(factor_file) as f:
            content = f.read()

        # 扫描pct_change()和rolling()调用
        issues = []

        # 检查pct_change()前是否有shift(1)
        pct_change_pattern = r"(\w+)\.pct_change\("
        for match in re.finditer(pct_change_pattern, content):
            var_name = match.group(1)
            # 向前查找是否有shift(1)
            before_text = content[: match.start()]
            if f"{var_name} = " in before_text:
                # 提取变量定义
                var_def_match = re.search(rf"{var_name}\s*=\s*(.+)", before_text)
                if var_def_match:
                    var_def = var_def_match.group(1)
                    if (
                        ".shift(1)" not in var_def
                        and ".shift(1)" not in before_text[-200:]
                    ):
                        issues.append(
                            {
                                "type": "pct_change_without_shift",
                                "variable": var_name,
                                "line": content[: match.start()].count("\n") + 1,
                            }
                        )

        # 检查rolling()前是否有shift(1)
        rolling_pattern = r"(\w+)\.rolling\("
        for match in re.finditer(rolling_pattern, content):
            var_name = match.group(1)
            before_text = content[: match.start()]
            if f"{var_name} = " in before_text:
                var_def_match = re.search(rf"{var_name}\s*=\s*(.+)", before_text)
                if var_def_match:
                    var_def = var_def_match.group(1)
                    if (
                        ".shift(1)" not in var_def
                        and ".shift(1)" not in before_text[-200:]
                    ):
                        issues.append(
                            {
                                "type": "rolling_without_shift",
                                "variable": var_name,
                                "line": content[: match.start()].count("\n") + 1,
                            }
                        )

        if issues:
            logger.warning(f"❌ 发现 {len(issues)} 个潜在未来函数问题:")
            for issue in issues:
                logger.warning(
                    f"  Line {issue['line']}: {issue['type']} on {issue['variable']}"
                )
        else:
            logger.info("✅ 未发现明显的未来函数问题")

        return {"issues": issues, "passed": len(issues) == 0}

    def generate_whitelist(self, coverage_threshold: float = 0.8) -> dict:
        """生成因子白名单"""
        logger.info("\n" + "=" * 60)
        logger.info("5. 生成因子白名单")
        logger.info("=" * 60)

        # 综合QA结果
        coverage_qa = self.qa_results.get("coverage", {})
        zero_var_qa = self.qa_results.get("zero_variance", {})

        # 白名单：覆盖率≥阈值 且 非零方差 且 非全NaN
        whitelist = []
        blacklist = []

        for col in self.panel.columns:
            # 检查覆盖率
            cov = coverage_qa.get("coverage", {}).get(col, 0)
            if cov < coverage_threshold:
                blacklist.append(
                    {
                        "factor": col,
                        "reason": f"coverage={cov:.1%}<{coverage_threshold:.0%}",
                    }
                )
                continue

            # 检查零方差
            if col in zero_var_qa.get("zero_variance", []):
                blacklist.append({"factor": col, "reason": "zero_variance"})
                continue

            if col in zero_var_qa.get("all_nan", []):
                blacklist.append({"factor": col, "reason": "all_nan"})
                continue

            # 通过所有检查
            whitelist.append(
                {
                    "factor_id": col,
                    "coverage": float(cov),
                    "category": self._infer_category(col),
                    "min_history": self._infer_min_history(col),
                }
            )

        logger.info(f"✅ 白名单: {len(whitelist)} 个因子")
        logger.info(f"❌ 黑名单: {len(blacklist)} 个因子")

        # 保存白名单
        whitelist_file = self.output_dir / "whitelist.yaml"
        with open(whitelist_file, "w") as f:
            yaml.dump(
                {"factors": whitelist, "blacklist": blacklist}, f, allow_unicode=True
            )

        logger.info(f"白名单已保存: {whitelist_file}")

        return {"whitelist": whitelist, "blacklist": blacklist}

    def _infer_category(self, factor_id: str) -> str:
        """推断因子类别"""
        if "Momentum" in factor_id or "MOM_" in factor_id:
            return "momentum"
        elif "VOLATILITY" in factor_id or "ATR" in factor_id:
            return "volatility"
        elif "DRAWDOWN" in factor_id:
            return "risk"
        elif "ADX" in factor_id:
            return "trend"
        else:
            return "unknown"

    def _infer_min_history(self, factor_id: str) -> int:
        """推断最小历史数据要求"""
        # 从因子名称中提取数字
        numbers = re.findall(r"\d+", factor_id)
        if numbers:
            return int(numbers[0]) + 1
        return 100  # 默认值

    def generate_qa_report(self) -> str:
        """生成QA报告"""
        logger.info("\n" + "=" * 60)
        logger.info("6. 生成QA报告")
        logger.info("=" * 60)

        report = []
        report.append("# 因子质量保证报告\n")
        report.append(f"**生成时间**: {pd.Timestamp.now()}\n")
        report.append(f"**面板文件**: {self.panel_file}\n")
        report.append(f"**面板形状**: {self.panel.shape}\n")
        report.append("\n---\n")

        # 1. 覆盖率
        coverage_qa = self.qa_results.get("coverage", {})
        report.append("\n## 1. 因子覆盖率\n")
        report.append(f"**阈值**: ≥{coverage_qa.get('threshold', 0.8):.0%}\n")
        report.append(
            f"**通过**: {len(coverage_qa.get('passed', []))}/{len(coverage_qa.get('coverage', {}))}\n\n"
        )
        report.append("| 因子 | 覆盖率 | 状态 |\n")
        report.append("|------|--------|------|\n")
        for factor, cov in sorted(
            coverage_qa.get("coverage", {}).items(), key=lambda x: x[1], reverse=True
        ):
            status = "✅" if cov >= coverage_qa.get("threshold", 0.8) else "❌"
            report.append(f"| {factor} | {cov:.1%} | {status} |\n")

        # 2. 零方差
        zero_var_qa = self.qa_results.get("zero_variance", {})
        report.append("\n## 2. 零方差检查\n")
        report.append(
            f"**通过**: {zero_var_qa.get('passed', 0)}/{len(self.panel.columns)}\n"
        )
        if zero_var_qa.get("zero_variance"):
            report.append(
                f"**零方差因子**: {', '.join(zero_var_qa['zero_variance'])}\n"
            )
        if zero_var_qa.get("all_nan"):
            report.append(f"**全NaN因子**: {', '.join(zero_var_qa['all_nan'])}\n")

        # 3. 因子家族差异
        diff_qa = self.qa_results.get("factor_family_diff", {})
        report.append("\n## 3. 因子家族差异性\n")
        for pair, result in diff_qa.items():
            status = "✅" if result["is_different"] else "❌"
            report.append(
                f"- {status} {pair}: var(diff)={result['var_diff']:.6f}, corr={result['corr']:.4f}\n"
            )

        # 4. 未来函数扫描
        lookahead_qa = self.qa_results.get("static_scan", {})
        report.append("\n## 4. 未来函数静态扫描\n")
        if lookahead_qa.get("passed"):
            report.append("✅ 未发现明显的未来函数问题\n")
        else:
            report.append(f"❌ 发现 {len(lookahead_qa.get('issues', []))} 个潜在问题\n")
            for issue in lookahead_qa.get("issues", []):
                report.append(
                    f"- Line {issue['line']}: {issue['type']} on {issue['variable']}\n"
                )

        # 5. 白名单
        whitelist_qa = self.qa_results.get("whitelist", {})
        report.append("\n## 5. 因子白名单\n")
        report.append(f"**白名单**: {len(whitelist_qa.get('whitelist', []))} 个因子\n")
        report.append(f"**黑名单**: {len(whitelist_qa.get('blacklist', []))} 个因子\n")

        report_text = "".join(report)

        # 保存报告
        report_file = self.output_dir / "qa_report.md"
        with open(report_file, "w") as f:
            f.write(report_text)

        logger.info(f"QA报告已保存: {report_file}")

        return report_text

    def run_all_checks(self):
        """运行所有QA检查"""
        logger.info("=" * 60)
        logger.info("开始因子质量保证检查")
        logger.info("=" * 60)

        # 1. 覆盖率检查
        self.qa_results["coverage"] = self.check_coverage()

        # 2. 零方差检查
        self.qa_results["zero_variance"] = self.check_zero_variance()

        # 3. 因子家族差异检查
        self.qa_results["factor_family_diff"] = self.check_factor_family_diff()

        # 4. 未来函数静态扫描
        self.qa_results["static_scan"] = self.static_scan_lookahead()

        # 5. 生成白名单
        self.qa_results["whitelist"] = self.generate_whitelist()

        # 6. 生成QA报告
        self.generate_qa_report()

        logger.info("\n" + "=" * 60)
        logger.info("✅ 因子质量保证检查完成")
        logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="因子质量保证检查")
    parser.add_argument(
        "--panel-file",
        default="factor_output/etf_rotation/panel_20200101_20251014.parquet",
        help="因子面板文件路径",
    )
    parser.add_argument(
        "--output-dir", default="factor_output/etf_rotation", help="输出目录"
    )
    parser.add_argument(
        "--coverage-threshold", type=float, default=0.8, help="覆盖率阈值（默认0.8）"
    )

    args = parser.parse_args()

    qa = FactorQA(args.panel_file, args.output_dir)
    qa.run_all_checks()


if __name__ == "__main__":
    main()
