#!/usr/bin/env python3
"""
最终系统验证脚本
确认所有因子都正常工作
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from factor_system.utils import get_project_root
from factor_system.factor_engine.providers.money_flow_provider import MoneyFlowProvider
from factor_system.factor_engine.factors.money_flow.core import (
    MainNetInflow_Rate, LargeOrder_Ratio, SuperLargeOrder_Ratio,
    OrderConcentration, MoneyFlow_Hierarchy, MoneyFlow_Consensus,
    MainFlow_Momentum, Flow_Price_Divergence
)
from factor_system.factor_engine.factors.money_flow.enhanced import (
    Institutional_Absorption, Flow_Tier_Ratio_Delta,
    Flow_Reversal_Ratio, Northbound_NetInflow_Rate
)

def load_test_data():
    """加载测试数据"""
    print("=== 📊 加载测试数据 ===")

    # 加载资金流数据
    mf_dir = get_project_root() / "raw" / "SH" / "money_flow"
    mf_provider = MoneyFlowProvider(data_dir=str(mf_dir))

    try:
        data = mf_provider.load_money_flow("600036.SH", "2024-01-01", "2024-12-31")
        print(f"✅ 资金流数据加载成功: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ 资金流数据加载失败: {e}")
        return None

def test_core_factors(data):
    """测试核心资金流因子"""
    print("\n=== 🎯 测试核心资金流因子 ===")

    core_factors = {
        "MainNetInflow_Rate": MainNetInflow_Rate(window=5),
        "LargeOrder_Ratio": LargeOrder_Ratio(window=10),
        "SuperLargeOrder_Ratio": SuperLargeOrder_Ratio(window=20),
        "OrderConcentration": OrderConcentration(),
        "MoneyFlow_Hierarchy": MoneyFlow_Hierarchy(),
        "MoneyFlow_Consensus": MoneyFlow_Consensus(window=5),
        "MainFlow_Momentum": MainFlow_Momentum(short_window=5, long_window=10),
        "Flow_Price_Divergence": Flow_Price_Divergence(window=5)  # 修复后的因子
    }

    results = {}

    for name, factor in core_factors.items():
        try:
            result = factor.calculate(data)

            valid_count = result.notna().sum()
            total_count = len(result)
            valid_ratio = valid_count / total_count * 100

            results[name] = {
                'shape': result.shape,
                'valid_count': valid_count,
                'total_count': total_count,
                'valid_ratio': valid_ratio,
                'mean': result.mean() if valid_count > 0 else np.nan,
                'std': result.std() if valid_count > 0 else np.nan
            }

            print(f"✅ {name}: {valid_ratio:.1f}% 有效 ({valid_count}/{total_count})")

        except Exception as e:
            print(f"❌ {name}: 计算失败 - {e}")
            results[name] = {'error': str(e)}

    return results

def test_enhanced_factors(data):
    """测试增强资金流因子"""
    print("\n=== 🚀 测试增强资金流因子 ===")

    enhanced_factors = {
        "Institutional_Absorption": Institutional_Absorption(),
        "Flow_Tier_Ratio_Delta": Flow_Tier_Ratio_Delta(window=5),
        "Flow_Reversal_Ratio": Flow_Reversal_Ratio(),
        "Northbound_NetInflow_Rate": Northbound_NetInflow_Rate(window=5)
    }

    results = {}

    for name, factor in enhanced_factors.items():
        try:
            result = factor.calculate(data)

            valid_count = result.notna().sum()
            total_count = len(result)
            valid_ratio = valid_count / total_count * 100

            results[name] = {
                'shape': result.shape,
                'valid_count': valid_count,
                'total_count': total_count,
                'valid_ratio': valid_ratio,
                'mean': result.mean() if valid_count > 0 else np.nan,
                'std': result.std() if valid_count > 0 else np.nan
            }

            print(f"✅ {name}: {valid_ratio:.1f}% 有效 ({valid_count}/{total_count})")

        except Exception as e:
            print(f"❌ {name}: 计算失败 - {e}")
            results[name] = {'error': str(e)}

    return results

def analyze_results(core_results, enhanced_results):
    """分析测试结果"""
    print("\n=== 📈 测试结果分析 ===")

    all_results = {**core_results, **enhanced_results}

    total_factors = len(all_results)
    successful_factors = len([r for r in all_results.values() if 'error' not in r])
    failed_factors = total_factors - successful_factors

    print(f"总因子数: {total_factors}")
    print(f"成功因子数: {successful_factors}")
    print(f"失败因子数: {failed_factors}")

    if successful_factors > 0:
        valid_ratios = [r['valid_ratio'] for r in all_results.values() if 'error' not in r]
        avg_valid_ratio = np.mean(valid_ratios)
        print(f"平均有效率: {avg_valid_ratio:.1f}%")

    print(f"\n详细结果:")
    for name, result in all_results.items():
        if 'error' not in result:
            print(f"✅ {name}: {result['valid_ratio']:.1f}% 有效")
        else:
            print(f"❌ {name}: 失败")

    # 特别检查Flow_Price_Divergence修复情况
    if 'Flow_Price_Divergence' in all_results:
        fpa_result = all_results['Flow_Price_Divergence']
        if 'error' not in fpa_result and fpa_result['valid_ratio'] > 90:
            print(f"\n🎉 Flow_Price_Divergence修复成功！")
            print(f"   有效率: {fpa_result['valid_ratio']:.1f}%")
            print(f"   均值: {fpa_result['mean']:.4f}")
            print(f"   标准差: {fpa_result['std']:.4f}")
        elif 'error' in fpa_result:
            print(f"\n❌ Flow_Price_Divergence仍然失败: {fpa_result['error']}")
        else:
            print(f"\n⚠️ Flow_Price_Divergence修复不完整: {fpa_result['valid_ratio']:.1f}%")

def main():
    """主函数"""
    print("🚀 最终系统验证开始")
    print("=" * 60)

    # 1. 加载测试数据
    data = load_test_data()

    if data is None:
        print("❌ 数据加载失败，无法进行验证")
        return

    # 2. 测试核心因子
    core_results = test_core_factors(data)

    # 3. 测试增强因子
    enhanced_results = test_enhanced_factors(data)

    # 4. 分析结果
    analyze_results(core_results, enhanced_results)

    print("\n" + "=" * 60)
    print("🎯 最终系统验证完成")

    # 5. 总结
    print("\n📋 验证总结:")
    all_results = {**core_results, **enhanced_results}
    successful = len([r for r in all_results.values() if 'error' not in r])
    total = len(all_results)

    if successful == total:
        print("🎉 所有因子都正常工作！系统修复成功！")
    elif successful >= total * 0.9:
        print(f"✅ 系统基本正常：{successful}/{total} 个因子正常工作")
    else:
        print(f"⚠️ 系统需要进一步修复：{successful}/{total} 个因子正常工作")

if __name__ == "__main__":
    main()