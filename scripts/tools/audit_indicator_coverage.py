#!/usr/bin/env python3
"""
指标覆盖率审计脚本
对比VectorBT可用指标与当前引擎实际执行的指标
"""

from scripts.path_utils import get_paths

paths = get_paths()

from datetime import datetime

import pandas as pd
import vectorbt as vbt

from factor_system.factor_generation.enhanced_factor_calculator import (
    EnhancedFactorCalculator,
    IndicatorConfig,
)


def get_vbt_available_indicators():
    """获取VectorBT所有可用指标"""
    all_attrs = [name for name in dir(vbt) if name.isupper()]
    return set(all_attrs)


def get_calculator_indicators():
    """获取计算器中实际使用的指标"""
    # 创建测试配置
    config = IndicatorConfig(
        enable_ma=True,
        enable_ema=True,
        enable_macd=True,
        enable_rsi=True,
        enable_bbands=True,
        enable_stoch=True,
        enable_atr=True,
        enable_obv=True,
        enable_mstd=True,
        enable_manual_indicators=True,
        enable_all_periods=True,
        memory_efficient=False,
        market="A_SHARES",
    )

    calculator = EnhancedFactorCalculator(config)
    available = calculator._check_available_indicators()

    # 提取VBT指标（排除TA_前缀）
    vbt_indicators = set([ind for ind in available if not ind.startswith("TA_")])

    return vbt_indicators


def audit_coverage():
    """审计指标覆盖率"""
    print("=" * 70)
    print("VectorBT 指标覆盖率审计")
    print("=" * 70)
    print()

    # 获取指标集合
    vbt_available = get_vbt_available_indicators()
    calculator_used = get_calculator_indicators()

    # 计算差异
    missing = vbt_available - calculator_used
    covered = vbt_available & calculator_used

    # 统计
    coverage_rate = len(covered) / len(vbt_available) * 100 if vbt_available else 0

    print(f"📊 统计摘要")
    print(f"  - VectorBT可用指标: {len(vbt_available)} 个")
    print(f"  - 当前已使用指标: {len(covered)} 个")
    print(f"  - 未使用指标: {len(missing)} 个")
    print(f"  - 覆盖率: {coverage_rate:.1f}%")
    print()

    print(f"✅ 已覆盖指标 ({len(covered)} 个):")
    for ind in sorted(covered):
        print(f"  - {ind}")
    print()

    print(f"❌ 未覆盖指标 ({len(missing)} 个):")
    for ind in sorted(missing):
        print(f"  - {ind}")
    print()

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "vbt_available": len(vbt_available),
        "calculator_used": len(covered),
        "missing": len(missing),
        "coverage_rate": coverage_rate,
        "covered_indicators": sorted(covered),
        "missing_indicators": sorted(missing),
    }

    # 保存到CSV
    df = pd.DataFrame(
        {
            "indicator": sorted(vbt_available),
            "covered": [ind in covered for ind in sorted(vbt_available)],
        }
    )

    output_path = str(paths["output_root"] / "indicator_coverage_report.csv")
    df.to_csv(output_path, index=False)
    print(f"📄 报告已保存: {output_path}")

    return report


if __name__ == "__main__":
    audit_coverage()
