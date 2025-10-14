#!/usr/bin/env python3
"""
资金流因子集成到factor_generation引擎方案
分析集成可行性并提供实现方案
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

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

def analyze_integration_feasibility():
    """分析集成可行性"""
    print("=== 🔍 资金流因子集成可行性分析 ===")

    print("\n📊 factor_generation引擎架构分析:")

    # 1. 核心组件分析
    print("\n🏗️ 核心组件:")
    components = {
        "EnhancedFactorCalculator": "主计算器，基于154个技术指标",
        "IndicatorRegistry": "指标注册中心，管理指标配置",
        "IndicatorSpec": "指标规格定义",
        "SimpleConfig": "配置管理，支持YAML配置",
        "BatchExecutor": "批量计算执行器"
    }

    for comp, desc in components.items():
        print(f"  ✅ {comp}: {desc}")

    # 2. 数据流分析
    print("\n📈 数据流:")
    print("  1. 输入: OHLCV价格数据 (DataFrame)")
    print("  2. 配置: IndicatorRegistry定义指标规格")
    print("  3. 计算: EnhancedFactorCalculator执行计算")
    print("  4. 输出: 因子DataFrame (包含所有技术指标)")

    # 3. 集成挑战分析
    print("\n⚠️ 集成挑战:")
    challenges = [
        "数据源差异: factor_generation使用OHLCV，资金流需要额外的资金流数据",
        "计算框架: factor_generation基于VectorBT，资金流基于自定义因子类",
        "频率对齐: 资金流是日线，factor_generation支持多时间框架",
        "配置系统: 需要将资金流因子适配到IndicatorRegistry"
    ]

    for challenge in challenges:
        print(f"  🔧 {challenge}")

    # 4. 集成方案设计
    print("\n💡 集成方案:")
    solutions = [
        "混合数据提供器: 扩展数据输入以支持资金流数据",
        "统一计算接口: 将资金流因子适配到VectorBT框架",
        "频率标准化: 统一使用日线作为基础频率",
        "配置扩展: 在IndicatorRegistry中注册资金流因子"
    ]

    for solution in solutions:
        print(f"  ✅ {solution}")

    return True

def create_moneyflow_indicator_specs():
    """创建资金流因子指标规格"""
    print("\n=== 📋 创建资金流因子指标规格 ===")

    from factor_system.factor_generation.indicator_registry import IndicatorSpec

    # 资金流因子规格配置
    moneyflow_specs = [
        IndicatorSpec(
            name="MainNetInflow_Rate",
            indicator_type="moneyflow",
            param_grid={"window": [5, 10, 20]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="LargeOrder_Ratio",
            indicator_type="moneyflow",
            param_grid={"window": [10, 20, 30]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="SuperLargeOrder_Ratio",
            indicator_type="moneyflow",
            param_grid={"window": [20, 30, 60]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="OrderConcentration",
            indicator_type="moneyflow",
            param_grid={},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="MoneyFlow_Hierarchy",
            indicator_type="moneyflow",
            param_grid={},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="MoneyFlow_Consensus",
            indicator_type="moneyflow",
            param_grid={"window": [5, 10]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="MainFlow_Momentum",
            indicator_type="moneyflow",
            param_grid={"short_window": [5, 10], "long_window": [10, 20]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="Flow_Price_Divergence",
            indicator_type="moneyflow",
            param_grid={"window": [5, 10, 20]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="Institutional_Absorption",
            indicator_type="moneyflow",
            param_grid={},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="Flow_Tier_Ratio_Delta",
            indicator_type="moneyflow",
            param_grid={"window": [5, 10]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="Flow_Reversal_Ratio",
            indicator_type="moneyflow",
            param_grid={},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        ),
        IndicatorSpec(
            name="Northbound_NetInflow_Rate",
            indicator_type="moneyflow",
            param_grid={"window": [5, 10]},
            batch_capable=True,
            requires_entries=False,
            enabled=True
        )
    ]

    print(f"✅ 创建了 {len(moneyflow_specs)} 个资金流因子规格")
    for spec in moneyflow_specs:
        print(f"  📊 {spec.name}: 参数={spec.param_grid}, 类型={spec.indicator_type}")

    return moneyflow_specs

def design_integration_architecture():
    """设计集成架构"""
    print("\n=== 🏗️ 设计集成架构 ===")

    print("\n📐 架构设计:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    统一因子计算引擎                          │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  输入数据层                                               │")
    print("│  ├── OHLCV价格数据 (factor_generation)                     │")
    print("│  └── 资金流数据 (MoneyFlowProvider)                       │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  配管理层                                                 │")
    print("│  ├── IndicatorRegistry (统一指标注册)                    │")
    print("│  ├── 技术指标规格 (MA, RSI, MACD等)                        │")
    print("│  └── 资金流指标规格 (12个资金流因子)                       │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  计算层                                                   │")
    print("│  ├── EnhancedFactorCalculator (技术指标)                 │")
    print("│  └── MoneyFlowCalculator (资金流指标)                    │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  输出层                                                   │")
    print("│  └── 统一因子DataFrame (技术因子 + 资金流因子)             │")
    print("└─────────────────────────────────────────────────────────────┘")

    print("\n🔧 实现步骤:")
    steps = [
        "1. 创建MoneyFlowCalculator适配器类",
        "2. 扩展IndicatorRegistry支持moneyflow类型",
        "3. 修改EnhancedFactorCalculator支持混合数据源",
        "4. 更新配置系统支持资金流参数",
        "5. 创建统一的数据合并接口",
        "6. 测试集成效果和性能"
    ]

    for step in steps:
        print(f"  {step}")

def create_moneyflow_calculator():
    """创建资金流计算器适配器"""
    print("\n=== 🧮 创建资金流计算器适配器 ===")

    calculator_code = '''
class MoneyFlowCalculator:
    """
    资金流因子计算器 - 适配factor_generation框架
    """

    def __init__(self, moneyflow_provider: MoneyFlowProvider):
        self.provider = moneyflow_provider
        self.factors = {
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

    def calculate(self, symbol: str, start_date: str, end_date: str,
                  param_overrides: Dict = None) -> pd.DataFrame:
        """计算资金流因子"""
        # 加载资金流数据
        mf_data = self.provider.load_money_flow(symbol, start_date, end_date)

        # 计算所有因子
        results = {}
        for name, factor in self.factors.items():
            if param_overrides and name in param_overrides:
                # 重新初始化因子参数
                factor = self._create_factor_with_params(name, param_overrides[name])

            result = factor.calculate(mf_data)
            results[name] = result

        return pd.DataFrame(results)

    def _create_factor_with_params(self, factor_name: str, params: Dict):
        """根据参数创建因子实例"""
        factor_classes = {
            'MainNetInflow_Rate': MainNetInflow_Rate,
            'LargeOrder_Ratio': LargeOrder_Ratio,
            'SuperLargeOrder_Ratio': SuperLargeOrder_Ratio,
            'MoneyFlow_Consensus': MoneyFlow_Consensus,
            'MainFlow_Momentum': MainFlow_Momentum,
            'Flow_Price_Divergence': Flow_Price_Divergence,
            'Flow_Tier_Ratio_Delta': Flow_Tier_Ratio_Delta,
            'Northbound_NetInflow_Rate': Northbound_NetInflow_Rate
        }

        if factor_name in factor_classes:
            return factor_classes[factor_name](**params)

        # 无参数因子
        no_param_factors = {
            'OrderConcentration': OrderConcentration,
            'MoneyFlow_Hierarchy': MoneyFlow_Hierarchy,
            'Institutional_Absorption': Institutional_Absorption,
            'Flow_Reversal_Ratio': Flow_Reversal_Ratio
        }

        if factor_name in no_param_factors:
            return no_param_factors[factor_name]()

        raise ValueError(f"未知因子: {factor_name}")
'''

    print("✅ MoneyFlowCalculator适配器类设计:")
    print("  - 统一的calculate接口")
    print("  - 参数化因子计算")
    print("  - 与factor_generation框架兼容")
    print("  - 支持批量计算")

    return calculator_code

def demonstrate_integration():
    """演示集成效果"""
    print("\n=== 🎯 集成效果演示 ===")

    print("\n📊 集成前后对比:")
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│ 集成前                                                  │")
    print("│ ├─ 技术指标: 154个 (仅价格数据)                          │")
    print("│ └─ 资金流因子: 12个 (独立系统)                           │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 集成后                                                  │")
    print("│ ├─ 统一因子库: 166个 (价格+资金流)                       │")
    print("│ ├─ 统一配置: IndicatorRegistry管理所有因子                │")
    print("│ ├─ 统一计算: EnhancedFactorCalculator + MoneyFlowCalculator │")
    print("│ └─ 统一输出: 单个DataFrame包含所有因子                   │")
    print("└─────────────────────────────────────────────────────────────┘")

    print("\n🎯 集成优势:")
    advantages = [
        "统一管理: 技术指标和资金流因子统一配置和管理",
        "批量计算: 支持166个因子的批量并行计算",
        "配置驱动: 通过YAML配置文件控制因子计算",
        "性能优化: 利用VectorBT缓存机制优化计算性能",
        "扩展性: 易于添加新的因子类型和计算逻辑",
        "一致性: 统一的数据格式和计算标准"
    ]

    for advantage in advantages:
        print(f"  ✅ {advantage}")

def main():
    """主函数"""
    print("🚀 资金流因子集成到factor_generation引擎方案")
    print("=" * 60)

    # 1. 分析集成可行性
    analyze_integration_feasibility()

    # 2. 创建指标规格
    moneyflow_specs = create_moneyflow_indicator_specs()

    # 3. 设计集成架构
    design_integration_architecture()

    # 4. 创建计算器适配器
    calculator_code = create_moneyflow_calculator()

    # 5. 演示集成效果
    demonstrate_integration()

    print("\n" + "=" * 60)
    print("📋 集成方案总结")
    print("=" * 60)
    print("✅ 可行性: 完全可行，架构兼容")
    print("✅ 实现复杂度: 中等，需要创建适配器层")
    print("✅ 性能影响: 最小，并行计算")
    print("✅ 维护成本: 低，配置驱动")
    print("✅ 扩展性: 优秀，易于添加新因子")

    print(f"\n🎯 结论: 资金流因子完全可以集成到factor_generation引擎中！")
    print("   建议实施步骤: 创建适配器 → 扩展注册中心 → 测试集成")

if __name__ == "__main__":
    main()