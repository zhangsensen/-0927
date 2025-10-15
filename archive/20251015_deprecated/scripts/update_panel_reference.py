#!/usr/bin/env python3
"""更新脚本中的面板文件引用"""

import os
import re
from pathlib import Path


def update_script_references():
    """更新脚本中的面板文件引用"""

    # 需要更新的脚本列表
    scripts_to_update = [
        "scripts/etf_monthly_rotation.py",
        "scripts/backtest_12months.py",
        "scripts/verify_no_lookahead.py",
    ]

    # 旧面板文件 -> 新面板文件
    panel_mappings = {
        "panel_20200101_20251014.parquet": "panel_corrected_20240101_20251014.parquet",
        "panel_20240101_20241014.parquet": "panel_corrected_20240101_20251014.parquet",
    }

    for script_file in scripts_to_update:
        if not Path(script_file).exists():
            print(f"❌ 脚本文件不存在: {script_file}")
            continue

        print(f"更新脚本: {script_file}")

        # 读取脚本内容
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 更新面板文件引用
        updated_content = content
        for old_panel, new_panel in panel_mappings.items():
            updated_content = updated_content.replace(old_panel, new_panel)

        # 保存更新后的脚本
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(updated_content)

        print(f"✅ 已更新: {script_file}")


def main():
    """主函数"""
    print("=== 更新脚本中的面板文件引用 ===")
    update_script_references()
    print("\n=== 更新完成 ===")
    print("现在所有脚本都使用修正后的面板文件")
    print("面板文件: panel_corrected_20240101_20251014.parquet")
    print("包含因子: Momentum20, Momentum15, Momentum10, ATR14, TRENDLB5")
    print("对应白名单: Momentum252, Momentum126, Momentum63, ATR14, TA_ADX_14")


if __name__ == "__main__":
    main()
