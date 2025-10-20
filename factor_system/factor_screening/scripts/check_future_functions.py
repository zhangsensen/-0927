#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
未来函数静态检查脚本
作者：量化首席工程师
版本：1.0.0
日期：2025-10-02

功能：
- 扫描Python文件中的未来函数使用
- 检查shift(-n), future_, lead_等关键词
- 提供详细的位置报告
"""

import ast
import re
import sys
from pathlib import Path
from typing import List


class FutureFunctionChecker(ast.NodeVisitor):
    """AST访问器，检测未来函数使用"""

    def __init__(self):
        self.issues = []
        self.current_file = None

    def visit_Call(self, node):
        """检查函数调用"""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.attr, str):
                attr = node.func.attr.lower()
                if attr in ["shift", "lead"]:
                    # 检查参数
                    if (
                        len(node.args) > 0
                        and isinstance(node.args[0], ast.UnaryOp)
                        and isinstance(node.args[0].op, ast.USub)
                        and isinstance(node.args[0].operand, ast.Constant)
                    ):

                        if (
                            isinstance(node.args[0].operand.value, int)
                            and node.args[0].operand.value < 0
                        ):
                            self.issues.append(
                                {
                                    "file": self.current_file,
                                    "line": node.lineno,
                                    "type": "negative_shift",
                                    "code": "function_call",
                                    "message": "发现未来函数: shift({})".format(
                                        node.args[0].operand.value
                                    ),
                                }
                            )

        elif isinstance(node.func, ast.Name):
            if isinstance(node.func.id, str) and node.func.id.lower() in [
                "future",
                "lead",
            ]:
                self.issues.append(
                    {
                        "file": self.current_file,
                        "line": node.lineno,
                        "type": "future_variable",
                        "code": ast.get_source_segment(node),
                        "message": "发现未来变量: {}".format(node.func.id),
                    }
                )

        self.generic_visit(node)


def check_file_for_future_functions(file_path: Path) -> List[dict]:
    """检查单个文件的未来函数使用"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 正则表达式快速检查
        patterns = [
            r"\.shift\(-\d+\)",  # .shift(-n)
            r"future_\w+",  # future_变量
            r"lead_\w+",  # lead_变量
        ]

        issues = []
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    {
                        "file": file_path,
                        "line": line_num,
                        "type": "regex_match",
                        "code": match.group(),
                        "message": "发现可疑模式: {}".format(match.group()),
                    }
                )

        # AST深度检查
        try:
            tree = ast.parse(content)
            checker = FutureFunctionChecker()
            checker.current_file = file_path
            checker.visit(tree)
            issues.extend(checker.issues)
        except SyntaxError as e:
            print(f"⚠️  语法错误，跳过文件 {file_path}: {e}")

        return issues

    except Exception as e:
        print(f"❌ 检查文件失败 {file_path}: {e}")
        return []


def main():
    """主函数"""
    print("🔍 开始扫描未来函数使用...")

    # 扫描当前目录下的Python文件
    current_dir = Path(__file__).parent
    python_files = list(current_dir.rglob("*.py"))

    all_issues = []

    for file_path in python_files:
        if file_path.name == "check_future_functions.py":
            continue  # 跳过检查脚本本身

        issues = check_file_for_future_functions(file_path)
        all_issues.extend(issues)

    # 输出报告
    if all_issues:
        print(f"\n❌ 发现 {len(all_issues)} 个未来函数问题:")
        print("=" * 60)

        # 按文件分组
        issues_by_file = {}
        for issue in all_issues:
            file_path = issue["file"]
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)

        for file_path, issues in sorted(issues_by_file.items()):
            print(f"\n📁 文件: {file_path}")
            for issue in sorted(issues, key=lambda x: x["line"]):
                print(f"  🚨 第{issue['line']}行: {issue['message']}")
                print(f"     代码: {issue['code']}")

        print("\n📊 统计:")
        print(f"  - 总问题数: {len(all_issues)}")
        print(f"  - 涉及文件: {len(issues_by_file)}")

        return 1
    else:
        print("✅ 未发现未来函数使用")
        return 0


if __name__ == "__main__":
    sys.exit(main())
