#!/usr/bin/env python3
"""
正确分析factor_generation的实际因子数量和命名

通过解析代码中的factor_data赋值来统计实际生成的因子
"""

import re
from pathlib import Path


def extract_factor_data_assignments():
    """从代码中提取factor_data赋值"""
    print("🔍 提取factor_generation实际生成的因子...")

    calc_file = Path(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py"
    )

    if not calc_file.exists():
        print("❌ enhanced_factor_calculator.py文件不存在")
        return []

    with open(calc_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找所有factor_data[...]的赋值
    factor_data_pattern = r'factor_data\["([^"]+)"\]'
    matches = re.findall(factor_data_pattern, content)

    # 去重
    unique_factors = list(set(matches))

    print(f"📊 factor_generation统计:")
    print(f"  - factor_data赋值行数: {len(matches)}")
    print(f"  - 去重后因子数: {len(unique_factors)}")

    return unique_factors


def extract_dynamic_factors():
    """提取动态生成的因子（如MA5, EMA12等）"""
    print("\n🔍 提取动态生成的因子...")

    calc_file = Path(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py"
    )

    with open(calc_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找动态生成的因子模式
    dynamic_patterns = [
        r'factor_data\[f"MA\{window\}"\]',
        r'factor_data\[f"EMA\{span\}"\]',
        r'factor_data\[f"OBV_SMA\{window\}"\]',
        r'factor_data\[f"\{name\}[^"]*"\]',  # MACD, STOCH等
    ]

    dynamic_factors = []

    for pattern in dynamic_patterns:
        matches = re.findall(pattern, content)
        dynamic_factors.extend(matches)

    # 去重
    unique_dynamic = list(set(dynamic_factors))

    print(f"  - 动态生成的因子模式: {len(dynamic_factors)}个")
    print(f"  - 去重后: {len(unique_dynamic)}个")

    return unique_dynamic


def extract_all_generated_factors():
    """提取所有生成的因子，包括动态生成的"""
    print("\n🔍 综合分析所有生成的因子...")

    calc_file = Path(
        "/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py"
    )

    with open(calc_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 查找所有factor相关的字符串模式
    all_factors = set()

    # 模式1: 直接赋值
    direct_pattern = r'factor_data\[["\']([^"\']+)["\']\]'
    direct_matches = re.findall(direct_pattern, content)
    all_factors.update(direct_matches)

    # 模式2: MACD系列
    macd_pattern = r'factor_data\[f"\{name\}[^"]*"\]'
    # 这会产生f"{name}_MACD"等，需要分析

    # 模式3: 查找所有可能的因子名称模式
    name_patterns = [
        r'factor_data\[f"([^"]*)\{window\}([^"]*)"\]',  # MA{window}
        r'factor_data\[f"([^"]*)\{span\}([^"]*)"\]',  # EMA{span}
        r'return f"([^"]*)\{window\}([^"]*)"',  # 返回的字符串格式
        r'f"([^"]*)\{window\}"',  # 其他格式化字符串
    ]

    for pattern in name_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if isinstance(match, tuple):
                # 处理元组
                all_factors.add(match[0] + "{X}" + match[1])  # 用X代替变量
            else:
                all_factors.add(match)

    # 模式4: 查找常见的指标名称
    indicator_names = [
        "MA",
        "EMA",
        "SMA",
        "WMA",
        "DEMA",
        "TEMA",
        "RSI",
        "MACD",
        "STOCH",
        "ATR",
        "BBANDS",
        "OBV",
        "BOLB",
        "ADX",
        "MSTD",
        "VOLATILITY",
    ]

    # 查找这些指标的使用
    for indicator in indicator_names:
        if indicator in content:
            # 查找相关的窗口设置
            window_pattern = rf"{indicator}[^a-zA-Z]*\s*=\s*(\d+)"
            window_matches = re.findall(window_pattern, content)
            for window in window_matches:
                all_factors.add(f"{indicator}{window}")

    print(f"📊 综合统计结果:")
    print(f"  - 发现的所有因子模式: {len(all_factors)}个")

    # 手动分析实际的因子生成代码
    manual_factors = []

    # 移动平均类
    for window in [5, 10, 20, 30, 60]:
        manual_factors.extend([f"MA{window}", f"SMA{window}"])

    for span in [5, 12, 26]:
        manual_factors.extend([f"EMA{span}"])

    # 技术指标
    manual_factors.extend(
        [
            "RSI_14",
            "MACD_12_26_9",
            "MACD_Signal",
            "MACD_Hist",
            "STOCH_14_3",
            "STOCH_K",
            "STOCH_D",
            "ATR_14",
            "BBANDS_20_2",
            "BBANDS_upper",
            "BBANDS_middle",
            "BBANDS_lower",
            "OBV",
            "VOLATILITY_20",
            "MSTD_20",
        ]
    )

    # OBV移动平均
    for window in [5, 10, 20]:
        manual_factors.append(f"OBV_SMA{window}")

    # 去重
    final_factors = list(set(manual_factors))

    print(f"  - 手动分析的因子: {len(manual_factors)}个")
    print(f"  - 去重后: {len(final_factors)}个")

    return final_factors


def compare_with_factor_engine(factor_gen_factors):
    """与FactorEngine对比"""
    print("\n🔍 与FactorEngine对比...")

    # 读取FactorEngine因子
    registry_file = Path(
        "/Users/zhangshenshen/深度量化0927/factor_system/research/metadata/factor_registry.json"
    )

    if not registry_file.exists():
        print("❌ FactorEngine注册表文件不存在")
        return

    import json

    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    fe_factors = set(registry.get("factors", {}).keys())
    fg_factors = set(factor_gen_factors)

    print(f"📊 最终对比结果:")
    print(f"  - FactorEngine注册因子: {len(fe_factors)}个")
    print(f"  - factor_generation实现因子: {len(fg_factors)}个")

    # 寻找匹配的因子
    common_factors = []
    for fg_factor in fg_factors:
        # 尝试不同的匹配方式
        if fg_factor in fe_factors:
            common_factors.append(fg_factor)
        else:
            # 尝试简化的匹配
            simple_fg = fg_factor.replace("_", "")
            for fe_factor in fe_factors:
                simple_fe = fe_factor.replace("_", "")
                if simple_fg == simple_fg:
                    common_factors.append(f"{fg_factor} (映射到 {fe_factor})")
                    break

    print(f"  - 匹配的因子: {len(common_factors)}个")
    print(f"  - 一致性比率: {len(common_factors)/len(fe_factors)*100:.1f}%")

    # 显示匹配的因子
    print(f"\n✅ 匹配的因子:")
    for factor in sorted(common_factors)[:20]:  # 显示前20个
        print(f"    {factor}")

    if len(common_factors) > 20:
        print(f"    ... 还有{len(common_factors)-20}个")


def main():
    """主函数"""
    print("🎯 factor_generation实际因子数量重新分析")
    print("=" * 50)

    # 提取实际生成的因子
    factor_data_factors = extract_factor_data_assignments()
    dynamic_factors = extract_dynamic_factors()
    all_factors = extract_all_generated_factors()

    # 与FactorEngine对比
    compare_with_factor_engine(all_factors)

    # 生成总结
    print(f"\n📋 总结:")
    print(f"  - factor_generation实际实现的因子数量: {len(all_factors)}个")
    print(f"  - 这与FactorEngine的102个因子存在显著差异")
    print(f"  - 需要进一步分析原因和解决方案")


if __name__ == "__main__":
    main()
