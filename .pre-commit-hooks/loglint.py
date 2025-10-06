#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-commit hook: 检查日志格式是否符合规范

强制规范:
所有 logger.info 必须包含 {session_id}|{symbol}|{tf}|{direction}|{metric}={value}
"""
import re
import sys
from pathlib import Path


def check_log_format(file_path: Path) -> list[str]:
    """
    检查文件中的日志格式

    Returns:
        违规行列表
    """
    violations = []
    log_pattern = re.compile(r'logger\.(info|debug|warning|error)\s*\(\s*["\']')
    required_pattern = re.compile(r"^.*\|.*\|.*\|.*\|.*=.*$")

    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            # 检测logger调用
            if log_pattern.search(line):
                # 提取日志消息（简化版，假设消息在单行内）
                match = re.search(r'logger\.\w+\(\s*["\'](.+?)["\']', line)
                if match:
                    log_message = match.group(1)
                    # 检查是否符合格式: xxx|xxx|xxx|xxx|xxx=xxx
                    if not required_pattern.match(log_message):
                        # 排除特殊情况：f-string, 变量, StructuredLogger
                        if (
                            "f" not in line[: line.index("logger")]
                            and "StructuredLogger" not in line
                            and ".format_message" not in line
                        ):
                            violations.append(
                                f"{file_path}:{line_no}: "
                                f"日志格式不符合规范: {line.strip()}"
                            )

    return violations


def main() -> int:
    """主函数"""
    violations = []

    # 获取所有待提交的Python文件
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if path.suffix == ".py" and path.exists():
            file_violations = check_log_format(path)
            violations.extend(file_violations)

    if violations:
        print("❌ 日志格式检查失败:")
        print("=" * 80)
        for violation in violations:
            print(violation)
        print("=" * 80)
        print(
            "提示: 所有 logger.info 必须包含 "
            "{session_id}|{symbol}|{tf}|{direction}|{metric}={value}"
        )
        print("使用 StructuredLogger.format_message() 来生成标准日志")
        return 1

    print("✅ 日志格式检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
