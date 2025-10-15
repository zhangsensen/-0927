#!/usr/bin/env python3
"""泄露静态扫描增强 - 扩展未来函数检测

核心功能：
1. 扫描pct_change未shift
2. 扫描rolling.apply未来函数
3. 扫描diff/cumsum等时序操作
4. 生成扫描报告

Linus式原则：长效安全网，减少漏报
"""

import ast
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class LeakageScanner:
    """泄露扫描器"""

    def __init__(self):
        self.patterns = {
            "pct_change": self._check_pct_change,
            "diff": self._check_diff,
            "rolling": self._check_rolling,
            "expanding": self._check_expanding,
            "cumsum": self._check_cumsum,
            "cumprod": self._check_cumprod,
        }
        self.violations = []

    def _check_pct_change(self, node, source_lines):
        """检查pct_change是否有shift"""
        # 查找.pct_change()调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "pct_change":
                    # 检查是否有.shift(1)
                    line_no = node.lineno
                    line = (
                        source_lines[line_no - 1]
                        if line_no <= len(source_lines)
                        else ""
                    )

                    # 简单检查：是否包含.shift(
                    if ".shift(" not in line and ".shift(1)" not in line:
                        self.violations.append(
                            {
                                "type": "pct_change_no_shift",
                                "line": line_no,
                                "code": line.strip(),
                                "severity": "high",
                            }
                        )

    def _check_diff(self, node, source_lines):
        """检查diff是否有shift"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "diff":
                    line_no = node.lineno
                    line = (
                        source_lines[line_no - 1]
                        if line_no <= len(source_lines)
                        else ""
                    )

                    if ".shift(" not in line:
                        self.violations.append(
                            {
                                "type": "diff_no_shift",
                                "line": line_no,
                                "code": line.strip(),
                                "severity": "medium",
                            }
                        )

    def _check_rolling(self, node, source_lines):
        """检查rolling.apply是否使用未来数据"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # 检查rolling().apply()模式
                if node.func.attr == "apply":
                    # 检查是否是rolling的apply
                    if isinstance(node.func.value, ast.Call):
                        if isinstance(node.func.value.func, ast.Attribute):
                            if node.func.value.func.attr == "rolling":
                                line_no = node.lineno
                                line = (
                                    source_lines[line_no - 1]
                                    if line_no <= len(source_lines)
                                    else ""
                                )

                                # 警告：rolling.apply可能使用未来数据
                                self.violations.append(
                                    {
                                        "type": "rolling_apply_warning",
                                        "line": line_no,
                                        "code": line.strip(),
                                        "severity": "medium",
                                        "message": "rolling.apply需确保lambda/func不使用未来数据",
                                    }
                                )

    def _check_expanding(self, node, source_lines):
        """检查expanding操作"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ["expanding"]:
                    line_no = node.lineno
                    line = (
                        source_lines[line_no - 1]
                        if line_no <= len(source_lines)
                        else ""
                    )

                    # expanding通常安全，但需检查
                    if ".shift(" not in line:
                        self.violations.append(
                            {
                                "type": "expanding_no_shift",
                                "line": line_no,
                                "code": line.strip(),
                                "severity": "low",
                            }
                        )

    def _check_cumsum(self, node, source_lines):
        """检查cumsum是否有shift"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "cumsum":
                    line_no = node.lineno
                    line = (
                        source_lines[line_no - 1]
                        if line_no <= len(source_lines)
                        else ""
                    )

                    if ".shift(" not in line:
                        self.violations.append(
                            {
                                "type": "cumsum_no_shift",
                                "line": line_no,
                                "code": line.strip(),
                                "severity": "medium",
                            }
                        )

    def _check_cumprod(self, node, source_lines):
        """检查cumprod是否有shift"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "cumprod":
                    line_no = node.lineno
                    line = (
                        source_lines[line_no - 1]
                        if line_no <= len(source_lines)
                        else ""
                    )

                    if ".shift(" not in line:
                        self.violations.append(
                            {
                                "type": "cumprod_no_shift",
                                "line": line_no,
                                "code": line.strip(),
                                "severity": "medium",
                            }
                        )

    def scan_file(self, file_path):
        """扫描单个文件"""
        try:
            with open(file_path) as f:
                source = f.read()
                source_lines = source.splitlines()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                for pattern_name, check_func in self.patterns.items():
                    check_func(node, source_lines)

            return True
        except Exception as e:
            logger.warning(f"扫描{file_path}失败: {e}")
            return False

    def scan_directory(self, directory, pattern="*.py"):
        """扫描目录"""
        logger.info("=" * 80)
        logger.info("泄露静态扫描 - 增强版")
        logger.info("=" * 80)

        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"❌ 目录不存在: {directory}")
            return False

        # 查找所有Python文件
        py_files = list(dir_path.rglob(pattern))
        logger.info(f"\n找到{len(py_files)}个Python文件")

        # 扫描每个文件
        for py_file in py_files:
            logger.info(f"\n扫描: {py_file.relative_to(dir_path)}")
            self.scan_file(py_file)

        # 生成报告
        self._generate_report()

        return len(self.violations) == 0

    def _generate_report(self):
        """生成扫描报告"""
        logger.info("\n" + "=" * 80)
        logger.info("扫描报告")
        logger.info("=" * 80)

        if len(self.violations) == 0:
            logger.info("\n✅ 未发现泄露风险")
            return

        # 按严重性分组
        by_severity = {"high": [], "medium": [], "low": []}
        for v in self.violations:
            by_severity[v["severity"]].append(v)

        logger.info(f"\n总违规数: {len(self.violations)}")
        logger.info(f"  高风险: {len(by_severity['high'])}")
        logger.info(f"  中风险: {len(by_severity['medium'])}")
        logger.info(f"  低风险: {len(by_severity['low'])}")

        # 显示高风险
        if by_severity["high"]:
            logger.info("\n" + "=" * 80)
            logger.info("高风险违规")
            logger.info("=" * 80)
            for v in by_severity["high"][:10]:
                logger.info(f"\n{v['type']} (行{v['line']}):")
                logger.info(f"  {v['code']}")
                if "message" in v:
                    logger.info(f"  说明: {v['message']}")

        # 显示中风险（部分）
        if by_severity["medium"]:
            logger.info("\n" + "=" * 80)
            logger.info("中风险违规（前10条）")
            logger.info("=" * 80)
            for v in by_severity["medium"][:10]:
                logger.info(f"\n{v['type']} (行{v['line']}):")
                logger.info(f"  {v['code']}")
                if "message" in v:
                    logger.info(f"  说明: {v['message']}")

        logger.info("\n" + "=" * 80)
        logger.info("建议")
        logger.info("=" * 80)
        logger.info("\n1. 高风险违规需立即修复")
        logger.info("2. 中风险违规需人工审查")
        logger.info("3. 低风险违规可延后处理")
        logger.info("4. 适配器级T+1已安全，此扫描为长效安全网")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="泄露静态扫描")
    parser.add_argument("--dir", default="factor_system", help="扫描目录")
    parser.add_argument("--pattern", default="*.py", help="文件模式")

    args = parser.parse_args()

    try:
        scanner = LeakageScanner()
        success = scanner.scan_directory(args.dir, args.pattern)

        if success:
            logger.info("\n✅ 扫描完成，无泄露风险")
            sys.exit(0)
        else:
            logger.warning("\n⚠️  扫描完成，发现潜在风险")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ 扫描失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
