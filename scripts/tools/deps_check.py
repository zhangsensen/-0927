#!/usr/bin/env python3
"""Python 依赖检查工具"""

import importlib
import sys


def check_dependencies(modules=None):
    """检查 Python 依赖

    Args:
        modules: 模块列表，默认检查核心依赖

    Returns:
        bool: 全部依赖可用返回 True
    """
    if modules is None:
        modules = ["pandas", "pyarrow", "yaml"]

    missing = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as e:
            print(f"Missing or broken module: {module}, {e}", file=sys.stderr)
            missing.append(module)

    if missing:
        print(f'❌ 缺失依赖: {", ".join(missing)}', file=sys.stderr)
        return False

    print("✅ 依赖检查通过")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python 依赖检查")
    parser.add_argument(
        "--modules", nargs="+", help="要检查的模块列表（默认: pandas, pyarrow, yaml）"
    )
    args = parser.parse_args()

    success = check_dependencies(args.modules)
    sys.exit(0 if success else 1)
