#!/usr/bin/env python3
"""
因子选择器确定性测试

测试修复后的 factor_selector 是否能在多次运行中产生一致的结果
"""

import sys
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.factor_selector import FactorSelector


def test_determinism():
    """测试因子选择的确定性"""
    print("=" * 80)
    print("【因子选择器确定性测试】")
    print("=" * 80)

    # 准备测试数据：故意让一些因子IC值相同
    ic_scores = {
        "MOM_20D": 0.0500,
        "CMF_20D": 0.0500,  # 与 MOM_20D 相同
        "SLOPE_20D": 0.0500,  # 与 MOM_20D 相同
        "RSI_14": 0.0499,
        "PRICE_POSITION_20D": 0.0501,
        "CORRELATION_TO_MARKET_20D": 0.0480,
        "SHARPE_RATIO_20D": 0.0500,  # 与 MOM_20D 相同
        "RET_VOL_20D": 0.0460,
    }

    # 不加载约束，只测试基本排序
    selector = FactorSelector(verbose=False)

    print(f"\n测试数据: {len(ic_scores)} 个因子")
    print("其中有4个因子IC=0.0500 (相同值)")

    # 运行100次
    print(f"\n运行100次选择...")
    results = []
    for i in range(100):
        selected, report = selector.select_factors(ic_scores=ic_scores, target_count=5)
        results.append(tuple(selected))  # 转为tuple便于比较

        if i == 0:
            print(f"\n第1次结果: {selected}")

    # 检查一致性
    unique_results = set(results)

    print(f"\n" + "=" * 80)
    if len(unique_results) == 1:
        print("✅ 测试通过！所有100次运行结果完全一致")
        print(f"   结果: {list(results[0])}")
        return True
    else:
        print("❌ 测试失败！发现不一致的结果")
        print(f"   发现 {len(unique_results)} 种不同的结果:")
        for i, result in enumerate(unique_results):
            count = results.count(result)
            print(f"   变种{i+1} (出现{count}次): {list(result)}")
        return False


def test_with_constraints():
    """测试带约束的因子选择确定性"""
    print("\n" + "=" * 80)
    print("【带约束的确定性测试】")
    print("=" * 80)

    # 加载实际的约束配置
    constraints_file = PROJECT_ROOT / "configs" / "FACTOR_SELECTION_CONSTRAINTS.yaml"
    if not constraints_file.exists():
        print("⚠️  约束配置文件不存在，跳过此测试")
        return True

    selector = FactorSelector(constraints_file=str(constraints_file), verbose=False)

    # 准备测试数据
    ic_scores = {
        "MOM_20D": 0.055,
        "CMF_20D": 0.055,
        "SLOPE_20D": 0.050,
        "RSI_14": 0.045,
        "PRICE_POSITION_20D": 0.060,
        "PRICE_POSITION_120D": 0.058,
        "CORRELATION_TO_MARKET_20D": 0.065,
        "SHARPE_RATIO_20D": 0.052,
        "CALMAR_RATIO_60D": 0.048,
        "RET_VOL_20D": 0.042,
        "VOL_RATIO_60D": 0.040,
        "RELATIVE_STRENGTH_VS_MARKET_20D": 0.055,
    }

    # 相关性矩阵（MOM_20D 和 SLOPE_20D 高相关）
    correlations = {
        ("MOM_20D", "SLOPE_20D"): 0.92,
        ("PRICE_POSITION_20D", "PRICE_POSITION_120D"): 0.85,
    }

    print(f"\n运行50次选择（含约束）...")
    results = []
    for i in range(50):
        selected, report = selector.select_factors(
            ic_scores=ic_scores, factor_correlations=correlations, target_count=5
        )
        results.append(tuple(selected))

        if i == 0:
            print(f"\n第1次结果: {selected}")
            print(f"应用的约束: {report.applied_constraints}")

    # 检查一致性
    unique_results = set(results)

    print(f"\n" + "=" * 80)
    if len(unique_results) == 1:
        print("✅ 测试通过！所有50次运行结果完全一致")
        print(f"   结果: {list(results[0])}")
        return True
    else:
        print("❌ 测试失败！发现不一致的结果")
        print(f"   发现 {len(unique_results)} 种不同的结果:")
        for i, result in enumerate(unique_results):
            count = results.count(result)
            print(f"   变种{i+1} (出现{count}次): {list(result)}")
        return False


if __name__ == "__main__":
    print("\n")
    print("🔍 测试因子选择器的确定性...")
    print("修复目标: 即使因子IC值相同，多次运行也应产生完全一致的结果")
    print("\n")

    # 运行测试
    test1_pass = test_determinism()
    test2_pass = test_with_constraints()

    # 总结
    print("\n" + "=" * 80)
    print("【测试总结】")
    print("=" * 80)
    print(f"基础排序测试: {'✅ 通过' if test1_pass else '❌ 失败'}")
    print(f"带约束测试: {'✅ 通过' if test2_pass else '❌ 失败'}")

    if test1_pass and test2_pass:
        print("\n🎉 所有测试通过！因子选择器现在是确定性的。")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，需要进一步修复。")
        sys.exit(1)
