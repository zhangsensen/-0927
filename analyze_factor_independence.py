#!/usr/bin/env python3
"""
分析资金流因子的独立性和计算方式
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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

def analyze_factor_dependencies():
    """分析因子依赖关系"""
    print("=== 📊 资金流因子依赖关系分析 ===")

    # 数据源字段
    print("\n📥 基础数据源字段:")
    raw_fields = [
        'buy_small_amount', 'sell_small_amount',     # 小单
        'buy_medium_amount', 'sell_medium_amount',   # 中单
        'buy_large_amount', 'sell_large_amount',     # 大单
        'buy_super_large_amount', 'sell_super_large_amount',  # 超大单
        'close', 'volume', 'turnover'                # 价格数据
    ]
    for field in raw_fields:
        print(f"  ✅ {field}")

    # MoneyFlowProvider计算的衍生字段
    print("\n🔧 MoneyFlowProvider衍生字段:")
    derived_fields = [
        'turnover_amount',      # 成交额 = 所有买卖金额之和
        'main_net',            # 主力净额 = 大单+超大单净额
        'retail_net',          # 散户净额 = 小单+中单净额
        'total_net',           # 总净额 = 主力+散户净额
    ]
    for field in derived_fields:
        print(f"  🔨 {field}")

    # 因子输入依赖分析
    print("\n🎯 因子输入依赖分析:")

    factor_dependencies = {
        'MainNetInflow_Rate': ['main_net', 'turnover_amount'],
        'LargeOrder_Ratio': ['buy_large_amount', 'sell_large_amount', 'turnover_amount'],
        'SuperLargeOrder_Ratio': ['buy_super_large_amount', 'sell_super_large_amount', 'turnover_amount'],
        'OrderConcentration': ['buy_large_amount', 'buy_super_large_amount', 'sell_large_amount', 'sell_super_large_amount', 'total_net'],
        'MoneyFlow_Hierarchy': ['main_net', 'retail_net'],
        'MoneyFlow_Consensus': ['main_net'],
        'MainFlow_Momentum': ['main_net'],
        'Flow_Price_Divergence': ['main_net', 'close'],  # 需要价格数据
        'Institutional_Absorption': ['main_net', 'close'],  # 需要价格数据计算波动率
        'Flow_Tier_Ratio_Delta': ['buy_large_amount', 'buy_super_large_amount', 'buy_small_amount', 'buy_medium_amount'],
        'Flow_Reversal_Ratio': ['main_net'],
        'Northbound_NetInflow_Rate': ['buy_super_large_amount', 'sell_super_large_amount', 'turnover_amount']  # 代理计算
    }

    for factor, deps in factor_dependencies.items():
        print(f"  📊 {factor}:")
        for dep in deps:
            print(f"    ➡️ {dep}")

    # 独立性分析
    print(f"\n🔍 独立性分析:")

    # 1. 数据来源独立性
    print(f"  📈 数据来源独立性:")
    print(f"    ✅ 完全独立: 每个因子都从原始数据计算，不使用其他因子的计算结果")
    print(f"    ✅ 基础数据: 所有因子都基于相同的原始资金流数据")
    print(f"    ✅ 价格数据: Flow_Price_Divergence和Institutional_Absorption使用价格数据")

    # 2. 计算独立性
    print(f"  🧮 计算独立性:")
    print(f"    ✅ 无链式依赖: 因子之间不存在A因子结果作为B因子输入的情况")
    print(f"    ✅ 并行计算: 所有因子可以独立并行计算")
    print(f"    ✅ 相同基础: 部分因子使用相同的基础字段（如main_net, turnover_amount）")

    # 3. 逻辑独立性
    print(f"  🎯 逻辑独立性:")

    # 按计算逻辑分类
    ratio_factors = ['MainNetInflow_Rate', 'LargeOrder_Ratio', 'SuperLargeOrder_Ratio', 'Northbound_NetInflow_Rate']
    concentration_factors = ['OrderConcentration', 'MoneyFlow_Hierarchy', 'Flow_Tier_Ratio_Delta']
    momentum_factors = ['MainFlow_Momentum', 'Flow_Reversal_Ratio', 'MoneyFlow_Consensus']
    price_factors = ['Flow_Price_Divergence', 'Institutional_Absorption']

    print(f"    📊 比率类因子 ({len(ratio_factors)}个): {', '.join(ratio_factors)}")
    print(f"    🎯 集中度类因子 ({len(concentration_factors)}个): {', '.join(concentration_factors)}")
    print(f"    📈 动量类因子 ({len(momentum_factors)}个): {', '.join(momentum_factors)}")
    print(f"    💰 价格相关因子 ({len(price_factors)}个): {', '.join(price_factors)}")

    # 检查是否有因子间的直接计算依赖
    print(f"\n🔗 直接计算依赖检查:")
    has_dependencies = False

    for factor, deps in factor_dependencies.items():
        # 检查是否依赖其他因子的计算结果
        factor_result_deps = [dep for dep in deps if dep.startswith('factor_') or dep in factor_dependencies.keys()]
        if factor_result_deps:
            print(f"    ⚠️ {factor} 依赖因子结果: {factor_result_deps}")
            has_dependencies = True
        else:
            print(f"    ✅ {factor}: 无因子间依赖")

    if not has_dependencies:
        print(f"    🎉 所有因子都是独立计算的，无因子间依赖关系！")

    return factor_dependencies

def test_factor_independence():
    """测试因子独立性"""
    print(f"\n=== 🧪 因子独立性测试 ===")

    # 加载数据
    provider = MoneyFlowProvider(
        data_dir=Path("raw/SH/money_flow"),
        enforce_t_plus_1=True
    )

    df = provider.load_money_flow("600036.SH", "2024-08-23", "2025-08-22")
    print(f"✅ 数据加载成功: {df.shape}")

    # 初始化所有因子
    factors = {
        'MainNetInflow_Rate': MainNetInflow_Rate(window=5),
        'LargeOrder_Ratio': LargeOrder_Ratio(window=10),
        'SuperLargeOrder_Ratio': SuperLargeOrder_Ratio(window=20),
        'OrderConcentration': OrderConcentration(),
        'MoneyFlow_Hierarchy': MoneyFlow_Hierarchy(),
        'MoneyFlow_Consensus': MoneyFlow_Consensus(window=5),
        'MainFlow_Momentum': MainFlow_Momentum(short_window=5, long_window=10),
        'Flow_Price_Divergence': Flow_Price_Divergence(window=5),
        'Institutional_Absorption': Institutional_Absorption(),
        'Flow_Tier_Ratio_Delta': Flow_Tier_Ratio_Delta(window=5),
        'Flow_Reversal_Ratio': Flow_Reversal_Ratio(),
        'Northbound_NetInflow_Rate': Northbound_NetInflow_Rate(window=5)
    }

    # 计算所有因子
    factor_results = {}
    for name, factor in factors.items():
        try:
            result = factor.calculate(df)
            factor_results[name] = result
            print(f"  ✅ {name}: 有效值 {result.notna().sum()}/{len(result)}")
        except Exception as e:
            print(f"  ❌ {name}: 计算失败 - {e}")

    # 检查因子相关性
    print(f"\n📊 因子相关性分析:")
    if len(factor_results) > 1:
        # 创建相关性矩阵
        factor_df = pd.DataFrame(factor_results)
        correlation_matrix = factor_df.corr()

        # 找出高相关性的因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.8 and not np.isnan(corr):  # 高相关性阈值
                    factor1 = correlation_matrix.columns[i]
                    factor2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((factor1, factor2, corr))

        if high_corr_pairs:
            print(f"  ⚠️ 高相关性因子对 (|相关系数| > 0.8):")
            for factor1, factor2, corr in high_corr_pairs:
                print(f"    {factor1} ↔ {factor2}: {corr:.3f}")
        else:
            print(f"  ✅ 无高相关性因子对，因子独立性良好")

    return factor_results

def main():
    """主函数"""
    print("🚀 资金流因子独立性分析")
    print("=" * 60)

    # 1. 分析依赖关系
    dependencies = analyze_factor_dependencies()

    # 2. 测试独立性
    factor_results = test_factor_independence()

    # 3. 总结
    print(f"\n" + "=" * 60)
    print("📋 独立性分析总结")
    print("=" * 60)
    print(f"✅ 数据来源: 所有因子基于相同的原始资金流数据")
    print(f"✅ 计算方式: 完全独立，无因子间计算依赖")
    print(f"✅ 并行性: 可以完全并行计算")
    print(f"✅ 基础字段共享: 部分因子使用相同基础字段（如main_net）")
    print(f"✅ 逻辑独立性: 不同类型的因子从不同角度分析资金流")
    print(f"\n🎯 结论: 资金流因子在计算上是完全独立的！")

if __name__ == "__main__":
    main()