#!/usr/bin/env python3
"""
FactorEngine与factor_generation一致性对比分析

对比内容：
1. 因子数量对比
2. 因子ID命名对比
3. 计算方式对比
4. 参数设置对比
"""

import json
import pandas as pd
from pathlib import Path

def analyze_factor_engine_registry():
    """分析FactorEngine注册表"""
    print("🔍 分析FactorEngine因子注册表...")

    registry_file = Path("/Users/zhangshenshen/深度量化0927/factor_system/research/metadata/factor_registry.json")

    if not registry_file.exists():
        print("❌ FactorEngine注册表文件不存在")
        return {}

    with open(registry_file, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    factors = registry.get('factors', {})
    metadata = registry.get('metadata', {})

    print(f"📊 FactorEngine统计:")
    print(f"  - 版本: {metadata.get('version', 'unknown')}")
    print(f"  - 总因子数: {metadata.get('total_factors', 0)}")
    print(f"  - 实际因子数: {len(factors)}")

    # 按类别统计
    categories = {}
    for factor_id, factor_info in factors.items():
        category = factor_info.get('category', 'unknown')
        if category not in categories:
            categories[category] = []
        categories[category].append(factor_id)

    print(f"  - 按类别分布:")
    for category, factor_ids in categories.items():
        print(f"    {category}: {len(factor_ids)}个")

    return {
        'total_count': len(factors),
        'metadata_count': metadata.get('total_factors', 0),
        'categories': categories,
        'factors': factors
    }

def analyze_factor_generation():
    """分析factor_generation系统"""
    print("\n🔍 分析factor_generation系统...")

    # 读取enhanced_factor_calculator.py来分析实际支持的因子
    calc_file = Path("/Users/zhangshenshen/深度量化0927/factor_system/factor_generation/enhanced_factor_calculator.py")

    if not calc_file.exists():
        print("❌ enhanced_factor_calculator.py文件不存在")
        return {}

    with open(calc_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分析VectorBT指标
    vbt_indicators = []
    if 'available_indicators = [' in content:
        start = content.find('available_indicators = [')
        end = content.find(']', start)
        if start != -1 and end != -1:
            indicators_section = content[start:end+1]
            # 提取指标名称
            lines = indicators_section.split('\n')
            for line in lines:
                if '"' in line and 'vbt' not in line:
                    indicator = line.strip().strip('"').strip(',')
                    if indicator and indicator != 'available_indicators':
                        vbt_indicators.append(indicator)

    print(f"📊 factor_generation统计:")
    print(f"  - VectorBT核心指标: {len(vbt_indicators)}个")
    print(f"  - VectorBT指标列表: {vbt_indicators}")

    # 分析TA-Lib指标
    talib_indicators = []
    if 'common_talib = [' in content:
        start = content.find('common_talib = [')
        end = content.find(']', start)
        if start != -1 and end != -1:
            talib_section = content[start:end+1]
            lines = talib_section.split('\n')
            for line in lines:
                if '"' in line and 'common_talib' not in line:
                    indicator = line.strip().strip('"').strip(',')
                    if indicator and indicator != 'common_talib':
                        talib_indicators.append(f"TA_{indicator}")

    print(f"  - TA-Lib指标: {len(talib_indicators)}个")

    # 分析实际实现的因子
    factor_calculations = []

    # 查找factor_calculations.append的调用
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'factor_calculations.append(' in line:
            # 提取因子名称
            start = line.find('factor_calculations.append((')
            if start != -1:
                end = line.find(')', start)
                if end != -1:
                    factor_calc = line[start+len('factor_calculations.append(('):end].strip()
                    if ',' in factor_calc:
                        factor_name = factor_calc.split(',')[0].strip('"')
                        if factor_name and factor_name not in factor_calculations:
                            factor_calculations.append(factor_name)

    print(f"  - 实际实现的因子: {len(factor_calculations)}个")

    return {
        'vbt_indicators': vbt_indicators,
        'talib_indicators': talib_indicators,
        'implemented_factors': factor_calculations,
        'total_count': len(factor_calculations)
    }

def compare_factor_naming(factor_engine_data, factor_gen_data):
    """对比因子命名"""
    print("\n🔍 对比因子命名...")

    fe_factors = set(factor_engine_data.get('factors', {}).keys())
    fg_factors = set(factor_gen_data.get('implemented_factors', []))

    print(f"FactorEngine因子: {len(fe_factors)}个")
    print(f"factor_generation因子: {len(fg_factors)}个")

    # 寻找共同的因子
    common_factors = fe_factors.intersection(fg_factors)
    print(f"  - 共同因子: {len(common_factors)}个")

    # 只在FactorEngine中的因子
    fe_only = fe_factors - fg_factors
    print(f"  - 仅FactorEngine有: {len(fe_only)}个")
    if len(fe_only) <= 20:
        print(f"    示例: {list(fe_only)[:10]}")

    # 只在factor_generation中的因子
    fg_only = fg_factors - fe_factors
    print(f"  - 仅factor_generation有: {len(fg_only)}个")
    if len(fg_only) <= 20:
        print(f"    示例: {list(fg_only)[:10]}")

    return {
        'common_count': len(common_factors),
        'common_factors': common_factors,
        'fe_only_count': len(fe_only),
        'fe_only_factors': fe_only,
        'fg_only_count': len(fg_only),
        'fg_only_factors': fg_only
    }

def analyze_calculation_methods():
    """分析计算方式差异"""
    print("\n🔍 分析计算方式差异...")

    # 读取VectorBT适配器
    adapter_file = Path("/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/core/vectorbt_adapter.py")

    if not adapter_file.exists():
        print("❌ VectorBT适配器文件不存在")
        return {}

    with open(adapter_file, 'r', encoding='utf-8') as f:
        adapter_content = f.read()

    # 统计VectorBT适配器中实现的函数
    adapter_functions = []
    lines = adapter_content.split('\n')
    for line in lines:
        if 'def ' in line and '(' in line and 'calc_' in line:
            func_name = line.strip().split('(')[0].replace('def ', '')
            if func_name:
                adapter_functions.append(func_name)

    print(f"VectorBT适配器实现的函数: {len(adapter_functions)}个")

    return {
        'adapter_functions': adapter_functions,
        'adapter_count': len(adapter_functions)
    }

def generate_comparison_report():
    """生成对比报告"""
    print("=" * 60)
    print("🎯 FactorEngine vs factor_generation 一致性分析报告")
    print("=" * 60)

    # 分析两个系统
    fe_data = analyze_factor_engine_registry()
    fg_data = analyze_factor_generation()
    naming_comparison = compare_factor_naming(fe_data, fg_data)
    calc_methods = analyze_calculation_methods()

    # 生成报告
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'factor_engine': {
            'total_factors': fe_data.get('total_count', 0),
            'metadata_factors': fe_data.get('metadata_count', 0),
            'categories': fe_data.get('categories', {})
        },
        'factor_generation': {
            'total_factors': fg_data.get('total_count', 0),
            'vbt_indicators': fg_data.get('vbt_indicators', []),
            'talib_indicators': fg_data.get('talib_indicators', []),
            'implemented_factors': fg_data.get('implemented_factors', [])
        },
        'comparison': naming_comparison,
        'calculation_methods': calc_methods,
        'consistency_analysis': {
            'consistent_factors': naming_comparison.get('common_count', 0),
            'factor_engine_exclusive': naming_comparison.get('fe_only_count', 0),
            'factor_gen_exclusive': naming_comparison.get('fg_only_count', 0),
            'consistency_ratio': naming_comparison.get('common_count', 0) / max(fe_data.get('total_count', 1), 1) if fe_data.get('total_count', 0) > 0 else 0
        }
    }

    # 关键发现
    print(f"\n🚨 关键发现:")
    print(f"  1. FactorEngine注册因子数: {fe_data.get('total_count', 0)}")
    print(f"  2. factor_generation实现因子数: {fg_data.get('total_count', 0)}")
    print(f"  3. 一致性比率: {report['consistency_analysis']['consistency_ratio']:.2%}")

    if report['consistency_analysis']['consistency_ratio'] < 0.8:
        print(f"  ⚠️  一致性低于80%，存在重大差异！")
    elif report['consistency_analysis']['consistency_ratio'] < 0.9:
        print(f"  ⚠️  一致性低于90%，存在一些差异")
    else:
        print(f"  ✅ 一致性良好 (>90%)")

    print(f"  4. 仅FactorEngine有的因子: {naming_comparison.get('fe_only_count', 0)}个")
    print(f"  5. 仅factor_generation有的因子: {naming_comparison.get('fg_only_count', 0)}个")

    return report

def main():
    """主函数"""
    try:
        report = generate_comparison_report()

        # 保存报告
        report_file = Path("/Users/zhangshenshen/深度量化0927/factor_engine_consistency_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n📄 详细报告已保存到: {report_file}")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()