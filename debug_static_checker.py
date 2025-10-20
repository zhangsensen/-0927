#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试静态检查器
"""

import ast
import os
import tempfile

from factor_system.future_function_guard.static_checker import (
    StaticCheckConfig,
    StaticChecker,
)


def debug_ast_parsing():
    """调试AST解析"""
    print("=== 调试AST解析 ===")

    test_code = """
import pandas as pd

def test_negative_shift():
    data = pd.Series([1, 2, 3, 4, 5])
    result = data.shift(-1)  # 未来函数！
    return result

def test_future_variable():
    future_price = get_future_price()
    return future_price
"""

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        # 解析AST
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()

        print("原始代码:")
        print(content)
        print("\n" + "=" * 60)

        tree = ast.parse(content)

        # 打印AST结构
        print("AST结构:")
        print(ast.dump(tree, indent=2))

        print("\n" + "=" * 60)

        # 使用静态检查器
        config = StaticCheckConfig(enabled=True)
        checker = StaticChecker(config)

        result = checker.check_file(temp_file)

        print(f"检查结果:")
        print(f"状态: {result['status']}")
        print(f"问题数: {result['issue_count']}")

        if result["issues"]:
            print("\n问题详情:")
            for issue in result["issues"]:
                print(
                    f"  第{issue['line']}行: {issue['message']} ({issue['issue_type']})"
                )
        else:
            print("未发现问题")

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    debug_ast_parsing()
