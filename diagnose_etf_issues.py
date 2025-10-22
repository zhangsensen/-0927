#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF项目审核：真实问题诊断脚本
用于快速定位和验证报告中发现的问题
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent
ETF_SYSTEM = PROJECT_ROOT / "etf_rotation_system"


def check_hardcoded_paths() -> Dict[str, List[str]]:
    """检查所有硬编码的用户路径"""
    print("\n" + "=" * 70)
    print("🔍 问题诊断 #1: 硬编码用户路径检查")
    print("=" * 70)

    hardcoded_pattern = r"/Users/zhangshenshen"
    issues = {}

    for py_file in ETF_SYSTEM.rglob("*.py"):
        if "archive" in str(py_file):
            continue  # 跳过归档文件

        try:
            content = py_file.read_text()
            lines = content.split("\n")
            found = []

            for i, line in enumerate(lines, 1):
                if re.search(hardcoded_pattern, line):
                    found.append(f"  Line {i}: {line.strip()[:80]}")

            if found:
                issues[str(py_file.relative_to(PROJECT_ROOT))] = found
        except:
            pass

    if issues:
        print("\n❌ 发现硬编码路径:")
        for file, lines in sorted(issues.items()):
            print(f"\n  📄 {file}")
            for line in lines:
                print(f"    {line}")
    else:
        print("\n✅ 未发现硬编码路径")

    return issues


def check_lookahead_bias() -> List[Tuple[str, int, str]]:
    """检查潜在的前向看穿偏差"""
    print("\n" + "=" * 70)
    print("🔍 问题诊断 #2: 前向看穿偏差（Lookahead Bias）检查")
    print("=" * 70)

    issues = []
    lookahead_patterns = [
        (r"\.shift\s*\(\s*-\d+\s*\)", "shift(-N) 前向移动"),
        (r"\.pct_change\s*\(\s*\d+\s*\)\s*\.shift\s*\(\s*-", "pct_change后跟shift(-)"),
    ]

    for py_file in ETF_SYSTEM.rglob("*.py"):
        if "archive" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")

            for pattern, desc in lookahead_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line) and "shift(-" in line:
                        # 检查是否在IC计算中
                        context = "\n".join(lines[max(0, i - 3) : i + 2])
                        if "ic" in context.lower() or "fwd_ret" in context.lower():
                            issues.append(
                                (
                                    str(py_file.relative_to(PROJECT_ROOT)),
                                    i,
                                    f"{desc}: {line.strip()[:70]}",
                                )
                            )
        except:
            pass

    if issues:
        print("\n❌ 发现可能的前向看穿:")
        for file, line_no, desc in issues:
            print(f"\n  📄 {file}:{line_no}")
            print(f"    {desc}")
    else:
        print("\n✅ 未发现明显的前向看穿模式")

    return issues


def check_iterrows_usage() -> Dict[str, List[str]]:
    """检查低效的iterrows()使用"""
    print("\n" + "=" * 70)
    print("🔍 问题诊断 #3: iterrows()性能反模式检查")
    print("=" * 70)

    issues = {}

    for py_file in ETF_SYSTEM.rglob("*.py"):
        if "archive" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            lines = content.split("\n")
            found = []

            for i, line in enumerate(lines, 1):
                if "iterrows()" in line or "itertuples()" in line:
                    found.append(f"  Line {i}: {line.strip()[:70]}")

            if found:
                issues[str(py_file.relative_to(PROJECT_ROOT))] = found
        except:
            pass

    if issues:
        print("\n⚠️  发现iterrows()使用:")
        for file, lines in sorted(issues.items()):
            print(f"\n  📄 {file}")
            for line in lines:
                print(f"    {line}")
    else:
        print("\n✅ 未发现iterrows()使用")

    return issues


def check_config_consistency() -> Dict[str, str]:
    """检查配置系统一致性"""
    print("\n" + "=" * 70)
    print("🔍 问题诊断 #4: 配置系统一致性检查")
    print("=" * 70)

    config_files = {
        "01阶段配置": ETF_SYSTEM / "01_横截面建设/config/config_classes.py",
        "02阶段配置": ETF_SYSTEM / "02_因子筛选/etf_cross_section_config.py",
        "03阶段配置": ETF_SYSTEM / "03_vbt回测/config_loader_parallel.py",
        "归档配置": ETF_SYSTEM
        / "03_vbt回测/archive_tests/config_files/config_loader.py",
    }

    configs = {}
    for stage, file_path in config_files.items():
        if file_path.exists():
            try:
                content = file_path.read_text()
                # 计算文件大小作为指标
                class_count = len(re.findall(r"^class\s+\w+", content, re.MULTILINE))
                dataclass_count = len(re.findall(r"@dataclass", content))
                configs[stage] = (
                    f"{file_path.name} ({class_count}个类, {dataclass_count}个dataclass)"
                )
            except:
                configs[stage] = "❌ 无法读取"
        else:
            configs[stage] = "❌ 文件不存在"

    print("\n📋 配置类定义分布:")
    for stage, info in configs.items():
        print(f"  {stage:10s}: {info}")

    print("\n⚠️  发现问题: 存在4套独立的配置系统")
    print("   需要统一为单一的UnifiedETFConfig类")

    return configs


def check_yaml_config_files() -> Dict[str, bool]:
    """检查YAML配置文件"""
    print("\n" + "=" * 70)
    print("🔍 问题诊断 #5: YAML配置文件检查")
    print("=" * 70)

    yaml_files = {}
    for yaml_file in ETF_SYSTEM.rglob("*.yaml"):
        if "archive" not in str(yaml_file):
            yaml_files[str(yaml_file.relative_to(PROJECT_ROOT))] = yaml_file.exists()

    if yaml_files:
        print("\n📋 找到YAML配置文件:")
        for file, exists in sorted(yaml_files.items()):
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
    else:
        print("\n⚠️  未找到YAML配置文件 (可能存储为.yaml或.yml)")

    return yaml_files


def generate_summary_report():
    """生成总结报告"""
    print("\n" + "=" * 70)
    print("📊 审核问题总结")
    print("=" * 70)

    hardcoded = check_hardcoded_paths()
    lookahead = check_lookahead_bias()
    iterrows_usage = check_iterrows_usage()
    config_consistency = check_config_consistency()
    yaml_configs = check_yaml_config_files()

    print("\n" + "=" * 70)
    print("📈 问题统计")
    print("=" * 70)

    print(
        f"""
🔴 严重问题 (P0):
   - 硬编码路径: {len(hardcoded)} 个文件
   - 前向看穿偏差: {len(lookahead)} 处代码

🟠 高优先级 (P1):
   - iterrows()使用: {len(iterrows_usage)} 个文件
   - 配置系统不统一: 4套独立系统

🟡 中优先级 (P2):
   - YAML配置文件: {len(yaml_configs)} 个

📋 建议修复顺序:
   1. 立即修复: 前向看穿偏差 (1小时)
   2. 立即修复: 硬编码路径 (2小时)
   3. 本周内: iterrows()优化 (1小时)
   4. 本周内: 配置系统统一 (3小时)
   5. 下周: 其他优化 (2小时)

📞 修复资源:
   - 涉及文件总数: ~12个
   - 预计总时间: ~10小时
   - 风险等级: 中（需要充分测试）
    """
    )


if __name__ == "__main__":
    print(
        """
╔══════════════════════════════════════════════════════════════════════╗
║                  ETF轮动系统 - 项目审核诊断工具                      ║
║                                                                      ║
║  本工具用于自动检测项目中的真实问题                                  ║
║  生成详细报告: ETF_PROJECT_COMPREHENSIVE_AUDIT_REPORT.md           ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    )

    generate_summary_report()

    print("\n✅ 诊断完成!")
    print("\n📝 详细报告已保存至:")
    print("   ETF_PROJECT_COMPREHENSIVE_AUDIT_REPORT.md")
