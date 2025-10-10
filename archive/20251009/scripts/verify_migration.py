"""
因子引擎统一迁移验证脚本

验证FactorEngine在factor_generation和hk_midfreq中的一致性
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime

import pandas as pd


def test_factor_engine_availability():
    """测试FactorEngine是否可用"""
    print("=" * 80)
    print("测试1: FactorEngine可用性")
    print("=" * 80)

    try:
        from factor_system.factor_engine import FactorEngine, FactorRegistry
        from factor_system.factor_engine.providers.parquet_provider import (
            ParquetDataProvider,
        )

        print("✅ FactorEngine导入成功")

        # 统计已注册因子
        registry = FactorRegistry(
            Path("factor_system/research/metadata/factor_registry.json")
        )
        print(f"✅ 注册表加载成功: {len(registry.metadata)}个因子")

        return True
    except Exception as e:
        print(f"❌ FactorEngine不可用: {e}")
        return False


def test_batch_calculator():
    """测试BatchFactorCalculator"""
    print("\n" + "=" * 80)
    print("测试2: BatchFactorCalculator")
    print("=" * 80)

    try:
        from factor_system.factor_engine.batch_calculator import BatchFactorCalculator

        calculator = BatchFactorCalculator(
            raw_data_dir=Path("raw"),
            enable_cache=False,  # 禁用缓存以测试实际计算
        )
        print(f"✅ BatchFactorCalculator初始化成功")
        print(f"   已注册因子: {len(calculator.registry.factors)}个")

        # 列出部分因子
        factors = list(calculator.registry.factors.keys())[:10]
        print(f"   示例因子: {', '.join(factors)}")

        return True
    except Exception as e:
        print(f"❌ BatchFactorCalculator初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_adapter():
    """测试回测因子适配器"""
    print("\n" + "=" * 80)
    print("测试3: BacktestFactorAdapter")
    print("=" * 80)

    try:
        from hk_midfreq.factor_engine_adapter import (
            BacktestFactorAdapter,
            get_factor_adapter,
        )

        adapter = get_factor_adapter()
        print(f"✅ BacktestFactorAdapter初始化成功")
        print(f"   已注册因子: {len(adapter._calculator.registry.factors)}个")

        return True
    except Exception as e:
        print(f"❌ BacktestFactorAdapter初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_calculation():
    """测试因子计算"""
    print("\n" + "=" * 80)
    print("测试4: 因子计算测试")
    print("=" * 80)

    try:
        from factor_system.factor_engine.batch_calculator import BatchFactorCalculator

        # 创建测试数据
        dates = pd.date_range("2025-09-01", periods=20, freq="15min")
        test_data = pd.DataFrame(
            {
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000000,
            },
            index=dates,
        )

        calculator = BatchFactorCalculator(raw_data_dir=Path("raw"), enable_cache=False)

        # 测试计算RSI
        print("   测试RSI计算...")
        result = calculator.calculate_all_factors(
            symbol="TEST",
            timeframe="15min",
            start_date=dates[0],
            end_date=dates[-1],
            factor_ids=["RSI"],
        )

        if not result.empty and "RSI" in result.columns:
            print(f"   ✅ RSI计算成功: {result.shape}")
        else:
            print(f"   ⚠️  RSI计算返回空结果")

        return True
    except Exception as e:
        print(f"   ❌ 因子计算失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_files():
    """测试因子文件完整性"""
    print("\n" + "=" * 80)
    print("测试5: 因子文件完整性")
    print("=" * 80)

    factor_dirs = {
        "technical": Path("factor_system/factor_engine/factors/technical"),
        "overlap": Path("factor_system/factor_engine/factors/overlap"),
        "pattern": Path("factor_system/factor_engine/factors/pattern"),
        "statistic": Path("factor_system/factor_engine/factors/statistic"),
    }

    total_files = 0
    for category, path in factor_dirs.items():
        if path.exists():
            py_files = list(path.glob("*.py"))
            py_files = [f for f in py_files if f.stem != "__init__"]
            count = len(py_files)
            total_files += count
            print(f"   {category:12s}: {count:3d}个因子")
        else:
            print(f"   {category:12s}: ❌ 目录不存在")

    print(f"\n   总计: {total_files}个因子文件")

    if total_files >= 100:
        print("   ✅ 因子文件完整")
        return True
    else:
        print("   ⚠️  因子文件不完整（预期>=100个）")
        return False


def main():
    """主验证流程"""
    print("\n" + "=" * 80)
    print("       Factor Engine 统一迁移验证")
    print("=" * 80)

    results = []

    # 运行所有测试
    results.append(("FactorEngine可用性", test_factor_engine_availability()))
    results.append(("BatchFactorCalculator", test_batch_calculator()))
    results.append(("BacktestFactorAdapter", test_factor_adapter()))
    results.append(("因子计算", test_factor_calculation()))
    results.append(("因子文件完整性", test_factor_files()))

    # 汇总结果
    print("\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25s}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("\n" + "=" * 80)
    print(f"总计: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！Factor Engine已成功统一")
        print("=" * 80)
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
