#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行主系统专业筛选器"""
import sys

sys.path.insert(0, "..")

from pathlib import Path

from factor_system.factor_screening.professional_factor_screener import (
    ProfessionalFactorScreener,
)

if __name__ == "__main__":
    # 配置
    panel_path = "etf_cross_section_results/panel_20251018_025457.parquet"
    price_dir = "/Users/zhangshenshen/深度量化0927/raw/ETF/daily"
    output_dir = "etf_cross_section_results/screening_professional"

    print("=" * 80)
    print("主系统专业筛选器 - ProfessionalFactorScreener")
    print("=" * 80)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 运行筛选
    screener = ProfessionalFactorScreener()

    # 注意：需要适配主系统的输入格式
    # 主系统期望的是标准的因子面板格式
    print(f"面板: {panel_path}")
    print(f"价格: {price_dir}")
    print(f"输出: {output_dir}")
    print("\n开始筛选...")

    try:
        # 主系统的run方法需要查看其参数
        screener.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()
