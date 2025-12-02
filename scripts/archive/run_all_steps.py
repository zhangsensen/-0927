#!/usr/bin/env python3
"""
完整流程执行脚本 - 自动运行3个步骤

功能：
- 自动按顺序执行3个步骤
- 传递数据目录路径
- 汇总最终结果

步骤：
1. Step 1: 横截面建设
2. Step 2: 因子筛选
3. Step 3: WFO优化

用法：
    python scripts/run_all_steps.py
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent

from scripts.step1_cross_section import main as run_step1
from scripts.step2_factor_selection import main as run_step2
from scripts.step3_run_wfo import main as run_step3


def main():
    print("=" * 80)
    print("开始执行完整3步流程")
    print("=" * 80)
    print("")

    # Step 1: 横截面建设
    print("🚀 Step 1/3: 横截面建设...")
    print("")
    cross_section_dir = run_step1()
    print("")
    print(f"✅ Step 1 完成，输出目录: {cross_section_dir}")
    print("")
    print("-" * 80)

    # Step 2: 因子筛选
    print("🚀 Step 2/3: 因子筛选...")
    print("")
    selection_dir = run_step2(cross_section_dir=cross_section_dir)
    print("")
    print(f"✅ Step 2 完成，输出目录: {selection_dir}")
    print("")
    print("-" * 80)

    # Step 3: WFO优化
    print("🚀 Step 3/3: WFO优化...")
    print("")
    wfo_dir = run_step3(selection_dir=selection_dir)
    print("")
    print(f"✅ Step 3 完成，输出目录: {wfo_dir}")
    print("")
    print("-" * 80)

    # 汇总
    print("")
    print("=" * 80)
    print("🎉 完整流程执行成功！")
    print("=" * 80)
    print("")
    print("输出目录汇总:")
    print(f"  - 横截面: {cross_section_dir}")
    print(f"  - 因子筛选: {selection_dir}")
    print(f"  - WFO结果: {wfo_dir}")
    print("")
    print("详细日志:")
    print(f"  - {cross_section_dir}/step1_cross_section.log")
    print(f"  - {selection_dir}/step2_factor_selection.log")
    print(f"  - {wfo_dir}/step3_wfo.log")
    print("")


if __name__ == "__main__":
    main()
