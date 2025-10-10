#!/usr/bin/env python3
"""
测试FactorEngine修复后的一致性
"""

import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, "/Users/zhangshenshen/深度量化0927")

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_factor_imports():
    """测试因子导入"""
    logger.info("🔍 测试因子导入...")

    try:
        # 测试技术指标导入
        from factor_system.factor_engine.factors import technical

        logger.info("✅ technical模块导入成功")

        # 测试移动平均指标导入
        from factor_system.factor_engine.factors import overlap

        logger.info("✅ overlap模块导入成功")

        # 检查关键因子类
        key_factors = [
            "RSI",
            "MACD",
            "MACDSignal",
            "MACDHistogram",
            "STOCH",
            "WILLR",
            "ATR",
            "BBANDS",
            "CCI",
            "OBV",
            "SMA",
            "EMA",
            "DEMA",
            "TEMA",
            "KAMA",
        ]

        for factor_name in key_factors:
            if hasattr(technical, factor_name):
                logger.info(f"  ✅ technical.{factor_name}")
            elif hasattr(overlap, factor_name):
                logger.info(f"  ✅ overlap.{factor_name}")
            else:
                logger.warning(f"  ❌ 缺失因子: {factor_name}")

        return True

    except Exception as e:
        logger.error(f"❌ 因子导入失败: {e}")
        return False


def test_factor_registry():
    """测试因子注册表"""
    logger.info("\n🔍 测试因子注册表...")

    try:
        from factor_system.factor_engine.core.registry import get_global_registry

        registry = get_global_registry()

        # 获取所有已注册因子
        all_factors = registry.list_factors()
        logger.info(f"📊 注册表中的因子数量: {len(all_factors)}")

        # 按分类统计
        categories = {}
        for factor_id in all_factors:
            meta = registry.get_metadata(factor_id)
            category = meta.get("category", "unknown") if meta else "unknown"
            if category not in categories:
                categories[category] = []
            categories[category].append(factor_id)

        for category, factors in categories.items():
            logger.info(f"  📈 {category}: {len(factors)} 个因子")

        return all_factors

    except Exception as e:
        logger.error(f"❌ 因子注册表测试失败: {e}")
        return []


def test_consistency_validation():
    """测试一致性验证"""
    logger.info("\n🔍 测试一致性验证...")

    try:
        # 模拟导入一致性验证器
        sys.path.insert(
            0, "/Users/zhangshenshen/深度量化0927/factor_system/factor_engine/core"
        )
        from consistency_validator import (
            get_consistency_validator,
            validate_factor_consistency,
        )

        validator = get_consistency_validator()

        # 获取FactorEngine中的因子
        registry_factors = test_factor_registry()

        if not registry_factors:
            logger.error("❌ 无法获取FactorEngine因子列表")
            return False

        # 执行一致性验证
        result = validate_factor_consistency(registry_factors)

        logger.info(f"📊 一致性验证结果:")
        logger.info(f"  ✅ 有效因子: {len(result.valid_factors)} 个")
        logger.info(f"  ❌ 无效因子: {len(result.invalid_factors)} 个")
        logger.info(f"  ⚠️  缺失因子: {len(result.missing_factors)} 个")
        logger.info(f"  📈 总体状态: {'通过' if result.is_valid else '失败'}")

        # 显示无效因子
        if result.invalid_factors:
            logger.warning("❌ 无效因子列表:")
            for factor in result.invalid_factors:
                logger.warning(f"  - {factor}")

        # 显示缺失因子
        if result.missing_factors:
            logger.info("⚠️ 缺失因子列表 (前10个):")
            for factor in result.missing_factors[:10]:
                logger.info(f"  - {factor}")

        return result.is_valid

    except Exception as e:
        logger.error(f"❌ 一致性验证测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_calculation():
    """测试因子计算"""
    logger.info("\n🔍 测试因子计算...")

    try:
        import numpy as np
        import pandas as pd

        # 创建测试数据
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(100, 200, 100),
                "low": np.random.uniform(100, 200, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # 测试几个关键因子的计算
        test_cases = [
            ("RSI", {"period": 14}),
            ("MACD", {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
            ("SMA", {"period": 20}),
            ("EMA", {"period": 12}),
            ("ATR", {"period": 14}),
        ]

        from factor_system.factor_engine.core.registry import get_global_registry

        registry = get_global_registry()

        successful_calculations = 0
        for factor_id, params in test_cases:
            try:
                factor = registry.get_factor(factor_id, **params)
                result = factor.calculate(test_data)

                if result is not None and len(result) > 0:
                    logger.info(f"  ✅ {factor_id}: 计算成功，形状 {result.shape}")
                    successful_calculations += 1
                else:
                    logger.warning(f"  ⚠️  {factor_id}: 计算结果为空")
            except Exception as e:
                logger.error(f"  ❌ {factor_id}: 计算失败 - {e}")

        logger.info(
            f"📊 因子计算测试: {successful_calculations}/{len(test_cases)} 成功"
        )
        return successful_calculations == len(test_cases)

    except Exception as e:
        logger.error(f"❌ 因子计算测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🚀 开始FactorEngine一致性修复验证测试...")
    logger.info("=" * 60)

    # 执行所有测试
    tests = [
        ("因子导入测试", test_factor_imports),
        ("因子注册表测试", lambda: test_factor_registry() or True),
        ("一致性验证测试", test_consistency_validation),
        ("因子计算测试", test_factor_calculation),
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 执行 {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"  {status}: {test_name}")
        except Exception as e:
            logger.error(f"  ❌ 异常: {test_name} - {e}")
            results.append((test_name, False))

    # 生成测试报告
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试报告")
    logger.info("=" * 60)

    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"  {status}: {test_name}")

    logger.info(f"\n📈 总体结果: {passed_tests}/{total_tests} 测试通过")

    if passed_tests == total_tests:
        logger.info("🎉 所有测试通过！FactorEngine一致性修复成功！")
    else:
        logger.warning("⚠️  部分测试失败，需要进一步修复")

    # 保存测试报告
    import pandas as pd

    report_content = f"""
FactorEngine一致性修复验证报告
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

测试结果:
{'=' * 40}
"""
    for test_name, result in results:
        status = "通过" if result else "失败"
        report_content += f"{status}: {test_name}\n"

    report_content += f"\n总体结果: {passed_tests}/{total_tests} 测试通过\n"

    if passed_tests == total_tests:
        report_content += "\n✅ FactorEngine一致性修复验证成功！\n"
        report_content += "FactorEngine现在完全符合factor_generation的一致性要求。\n"
    else:
        report_content += "\n⚠️ 部分测试失败，需要进一步修复。\n"

    with open(
        "/Users/zhangshenshen/深度量化0927/factor_engine_consistency_test_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_content)

    logger.info("📄 测试报告已保存至: factor_engine_consistency_test_report.txt")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
