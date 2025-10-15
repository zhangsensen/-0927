#!/usr/bin/env python3
"""
量化交易系统代码质量检查脚本
使用 pyscn 进行深度代码质量分析
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class QuantQualityChecker:
    """量化交易系统代码质量检查器"""

    def __init__(self, target_path: str = "factor_system/"):
        self.target_path = target_path
        self.project_root = Path.cwd()
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

    def run_quality_check(self, format_type: str = "text") -> Dict[str, Any]:
        """运行代码质量检查"""
        print(f"🔍 开始分析 {self.target_path} 的代码质量...")

        cmd = ["pyscn", "analyze", self.target_path, f"--{format_type}"]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                if format_type == "json":
                    data = json.loads(result.stdout)
                    return self._analyze_json_results(data)
                else:
                    return self._analyze_text_results(result.stdout)
            else:
                print(f"❌ 代码质量检查失败:")
                print(result.stderr)
                return {"success": False, "errors": result.stderr}

        except Exception as e:
            print(f"❌ 执行质量检查时出错: {e}")
            return {"success": False, "errors": str(e)}

    def _analyze_json_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """分析 JSON 格式的结果"""
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "issues": {
                "complexity": [],
                "dead_code": [],
                "clones": [],
                "architecture": [],
            },
        }

        # 复杂度分析
        if "complexity" in data:
            complexity_data = data["complexity"]
            results["summary"]["complexity"] = {
                "total_functions": len(complexity_data.get("functions", [])),
                "high_risk": len(
                    [
                        f
                        for f in complexity_data.get("functions", [])
                        if f.get("complexity", 0) > 10
                    ]
                ),
                "medium_risk": len(
                    [
                        f
                        for f in complexity_data.get("functions", [])
                        if 5 < f.get("complexity", 0) <= 10
                    ]
                ),
            }
            results["issues"]["complexity"] = [
                f
                for f in complexity_data.get("functions", [])
                if f.get("complexity", 0) > 5
            ]

        # 代码克隆分析
        if "clones" in data:
            clones_data = data["clones"]
            results["summary"]["clones"] = {
                "total_clones": len(clones_data.get("clones", [])),
                "high_similarity": len(
                    [
                        c
                        for c in clones_data.get("clones", [])
                        if c.get("similarity", 0) > 0.9
                    ]
                ),
            }
            results["issues"]["clones"] = clones_data.get("clones", [])

        # 死代码分析
        if "dead_code" in data:
            dead_code_data = data["dead_code"]
            results["summary"]["dead_code"] = {
                "total_issues": len(dead_code_data.get("issues", [])),
                "critical": len(
                    [
                        i
                        for i in dead_code_data.get("issues", [])
                        if i.get("severity") == "critical"
                    ]
                ),
            }
            results["issues"]["dead_code"] = dead_code_data.get("issues", [])

        return results

    def _analyze_text_results(self, output: str) -> Dict[str, Any]:
        """分析文本格式的结果"""
        lines = output.split("\n")
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "raw_output": output,
        }

        # 简单解析文本输出
        complexity_issues = 0
        clone_issues = 0
        dead_code_issues = 0

        for line in lines:
            if "too complex" in line.lower():
                complexity_issues += 1
            elif "clone of" in line.lower():
                clone_issues += 1
            elif "dead code" in line.lower() or "unreachable" in line.lower():
                dead_code_issues += 1

        results["summary"] = {
            "complexity_issues": complexity_issues,
            "clone_issues": clone_issues,
            "dead_code_issues": dead_code_issues,
            "total_issues": complexity_issues + clone_issues + dead_code_issues,
        }

        return results

    def generate_quality_report(
        self, results: Dict[str, Any], output_file: Optional[str] = None
    ) -> str:
        """生成质量报告"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.reports_dir / f"quality_report_{timestamp}.md"

        report_content = self._create_markdown_report(results)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"📊 质量报告已生成: {output_file}")
        return str(output_file)

    def _create_markdown_report(self, results: Dict[str, Any]) -> str:
        """创建 Markdown 格式的报告"""
        report = []
        report.append("# 量化交易系统代码质量报告\n")
        report.append(f"**生成时间**: {results.get('timestamp', 'N/A')}\n")

        if not results.get("success", False):
            report.append("## ❌ 质量检查失败\n")
            report.append(f"**错误信息**: {results.get('errors', 'Unknown error')}\n")
            return "\n".join(report)

        # 摘要信息
        summary = results.get("summary", {})
        report.append("## 📊 质量摘要\n")

        if "complexity" in summary:
            comp = summary["complexity"]
            report.append(f"- **复杂度分析**: {comp.get('total_functions', 0)} 个函数")
            report.append(f"  - 高风险函数: {comp.get('high_risk', 0)}")
            report.append(f"  - 中风险函数: {comp.get('medium_risk', 0)}")

        if "clones" in summary:
            clones = summary["clones"]
            report.append(f"- **代码克隆**: {clones.get('total_clones', 0)} 个克隆")
            report.append(f"  - 高相似度克隆: {clones.get('high_similarity', 0)}")

        if "dead_code" in summary:
            dead = summary["dead_code"]
            report.append(f"- **死代码**: {dead.get('total_issues', 0)} 个问题")
            report.append(f"  - 关键问题: {dead.get('critical', 0)}")

        report.append("")

        # 详细问题
        issues = results.get("issues", {})

        # 复杂度问题
        if issues.get("complexity"):
            report.append("## 🚨 复杂度问题\n")
            for issue in issues["complexity"]:
                report.append(
                    f"### {issue.get('name', 'Unknown')} (复杂度: {issue.get('complexity', 'N/A')})"
                )
                report.append(
                    f"- **位置**: {issue.get('file', 'N/A')}:{issue.get('line', 'N/A')}"
                )
                report.append("")

        # 代码克隆
        if issues.get("clones"):
            report.append("## 🔄 代码克隆\n")
            for i, clone in enumerate(issues["clones"][:10], 1):  # 只显示前10个
                similarity = clone.get("similarity", 0)
                report.append(f"### 克隆 #{i} (相似度: {similarity:.1%})")
                fragments = clone.get("fragments", [])
                if len(fragments) >= 2:
                    report.append(
                        f"- **位置1**: {fragments[0].get('file', 'N/A')}:{fragments[0].get('start_line', 'N/A')}"
                    )
                    report.append(
                        f"- **位置2**: {fragments[1].get('file', 'N/A')}:{fragments[1].get('start_line', 'N/A')}"
                    )
                report.append("")

        # 死代码
        if issues.get("dead_code"):
            report.append("## 💀 死代码问题\n")
            for issue in issues["dead_code"]:
                report.append(f"### {issue.get('severity', 'Unknown').title()} 级别")
                report.append(
                    f"- **位置**: {issue.get('file', 'N/A')}:{issue.get('line', 'N/A')}"
                )
                report.append(f"- **描述**: {issue.get('message', 'N/A')}")
                report.append("")

        report.append("---\n")
        report.append("*报告由 pyscn 量化交易质量检查器生成*\n")

        return "\n".join(report)

    def check_critical_files(self) -> Dict[str, Any]:
        """检查关键文件的代码质量"""
        critical_paths = [
            "factor_system/factor_engine/core/engine.py",
            "factor_system/factor_engine/api.py",
            "factor_system/factor_generation/enhanced_factor_calculator.py",
            "factor_system/factor_screening/professional_factor_screener.py",
        ]

        results = {"critical_files": {}, "summary": {"total_files": 0, "passed": 0}}

        for path in critical_paths:
            if Path(path).exists():
                results["summary"]["total_files"] += 1
                file_result = self._run_pyscn_check(path)
                results["critical_files"][path] = file_result

                if file_result.get("success", False):
                    results["summary"]["passed"] += 1

        return results

    def _run_pyscn_check(self, path: str) -> Dict[str, Any]:
        """运行 pyscn 检查单个文件"""
        cmd = ["pyscn", "check", path]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
            }

        except Exception as e:
            return {"success": False, "errors": str(e)}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="量化交易系统代码质量检查")
    parser.add_argument(
        "--target", default="factor_system/", help="分析目标路径 (默认: factor_system/)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式 (默认: text)",
    )
    parser.add_argument("--output", help="报告输出文件路径")
    parser.add_argument("--critical-only", action="store_true", help="只检查关键文件")
    parser.add_argument("--no-report", action="store_true", help="不生成报告文件")

    args = parser.parse_args()

    checker = QuantQualityChecker(args.target)

    if args.critical_only:
        print("🎯 检查关键文件质量...")
        results = checker.check_critical_files()

        print(f"\n📊 关键文件检查结果:")
        print(f"总文件数: {results['summary']['total_files']}")
        print(f"通过检查: {results['summary']['passed']}")

        for file_path, file_result in results["critical_files"].items():
            status = "✅" if file_result.get("success", False) else "❌"
            print(f"{status} {file_path}")
    else:
        # 运行完整质量检查
        results = checker.run_quality_check(args.format)

        if results.get("success", False):
            print("✅ 代码质量检查完成")

            # 显示摘要
            summary = results.get("summary", {})
            if "total_issues" in summary:
                print(f"📊 发现问题总数: {summary['total_issues']}")
                print(f"  - 复杂度问题: {summary.get('complexity_issues', 0)}")
                print(f"  - 代码克隆: {summary.get('clone_issues', 0)}")
                print(f"  - 死代码问题: {summary.get('dead_code_issues', 0)}")
        else:
            print("❌ 代码质量检查失败")
            print(results.get("errors", "Unknown error"))
            sys.exit(1)

        # 生成报告
        if not args.no_report:
            report_file = checker.generate_quality_report(results, args.output)
            print(f"📄 详细报告: {report_file}")


if __name__ == "__main__":
    main()
