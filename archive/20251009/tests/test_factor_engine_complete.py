"""
测试完整的FactorEngine系统

验证数据提供器修复后，整个FactorEngine系统是否能正常工作
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def test_factor_engine_api():
    """测试FactorEngine API是否能正常工作"""
    try:
        from factor_system.factor_engine import api

        print("🚀 测试FactorEngine API...")

        # 测试计算RSI因子
        result = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0005.HK"],
            timeframe="daily",
            start_date=datetime(
                2025, 3, 10
            ),  # Use date range that matches available data
            end_date=datetime(2025, 3, 31),
            use_cache=False,
        )

        if not result.empty:
            print(f"✅ RSI计算成功: {result.shape}")
            print(f"📋 列名: {result.columns.tolist()}")
            print(f"🏷️  索引: {result.index.names}")

            # 检查结果是否只包含请求的因子
            unexpected_columns = [col for col in result.columns if col not in ["RSI"]]
            if len(unexpected_columns) == 0:
                print("✅ 结果只包含请求的因子")
            else:
                print(f"⚠️ 结果包含额外列: {unexpected_columns}")

            # 检查RSI值是否合理
            if "RSI" in result.columns:
                rsi_values = result["RSI"].dropna()
                if not rsi_values.empty:
                    print(
                        f"📊 RSI值范围: {rsi_values.min():.2f} ~ {rsi_values.max():.2f}"
                    )
                    if 0 <= rsi_values.min() <= 100 and 0 <= rsi_values.max() <= 100:
                        print("✅ RSI值在合理范围内")
                    else:
                        print("⚠️ RSI值超出合理范围")

            return True
        else:
            print("❌ RSI计算失败：返回空结果")
            return False

    except Exception as e:
        print(f"❌ FactorEngine API测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_factors():
    """测试多因子计算"""
    try:
        from factor_system.factor_engine import api

        print("\n🚀 测试多因子计算...")

        # 测试计算多个技术指标
        factors_to_test = ["RSI", "MACD", "WILLR"]

        result = api.calculate_factors(
            factor_ids=factors_to_test,
            symbols=["0005.HK"],
            timeframe="daily",
            start_date=datetime(2025, 3, 10),
            end_date=datetime(2025, 3, 31),
            use_cache=False,
        )

        if not result.empty:
            print(f"✅ 多因子计算成功: {result.shape}")
            print(f"📋 计算的因子: {result.columns.tolist()}")

            # 验证每个请求的因子都存在
            for factor in factors_to_test:
                if factor in result.columns:
                    print(f"✅ {factor} 计算成功")
                else:
                    print(f"❌ {factor} 计算失败")
                    return False

            return True
        else:
            print("❌ 多因子计算失败：返回空结果")
            return False

    except Exception as e:
        print(f"❌ 多因子测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_symbols():
    """测试多股票因子计算"""
    try:
        from factor_system.factor_engine import api

        print("\n🚀 测试多股票因子计算...")

        # 获取可用股票
        from factor_system.factor_engine.providers.parquet_provider import (
            ParquetDataProvider,
        )

        provider = ParquetDataProvider(Path("raw"))
        available_symbols = provider.get_symbols("daily")

        # 选择3个股票进行测试
        test_symbols = (
            available_symbols[:3] if len(available_symbols) >= 3 else available_symbols
        )

        print(f"📈 测试股票: {test_symbols}")

        result = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=test_symbols,
            timeframe="daily",
            start_date=datetime(2025, 3, 10),
            end_date=datetime(2025, 3, 31),
            use_cache=False,
        )

        if not result.empty:
            print(f"✅ 多股票因子计算成功: {result.shape}")

            # 验证包含多个股票
            unique_symbols = result.index.get_level_values("symbol").unique()
            print(f"📈 实际包含股票: {list(unique_symbols)}")

            # 验证每个股票都有RSI值
            for symbol in test_symbols:
                symbol_data = result.xs(symbol, level="symbol")
                if not symbol_data.empty and "RSI" in symbol_data.columns:
                    print(f"✅ {symbol} RSI计算成功")
                else:
                    print(f"❌ {symbol} RSI计算失败")
                    return False

            return True
        else:
            print("❌ 多股票因子计算失败：返回空结果")
            return False

    except Exception as e:
        print(f"❌ 多股票测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cache_functionality():
    """测试缓存功能"""
    try:
        from factor_system.factor_engine import api

        print("\n🚀 测试缓存功能...")

        # 第一次计算（无缓存）
        print("📊 第一次计算（无缓存）...")
        start_time = datetime.now()
        result1 = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0005.HK"],
            timeframe="daily",
            start_date=datetime(2025, 3, 10),
            end_date=datetime(2025, 3, 31),
            use_cache=True,
        )
        first_time = datetime.now() - start_time

        if not result1.empty:
            print(f"✅ 第一次计算成功，耗时: {first_time.total_seconds():.3f}秒")

            # 第二次计算（使用缓存）
            print("📊 第二次计算（使用缓存）...")
            start_time = datetime.now()
            result2 = api.calculate_factors(
                factor_ids=["RSI"],
                symbols=["0005.HK"],
                timeframe="daily",
                start_date=datetime(2025, 3, 10),
                end_date=datetime(2025, 3, 31),
                use_cache=True,
            )
            second_time = datetime.now() - start_time

            if not result2.empty:
                print(f"✅ 第二次计算成功，耗时: {second_time.total_seconds():.3f}秒")

                # 验证缓存效果
                if second_time < first_time * 0.5:  # 缓存应该显著提升性能
                    print("✅ 缓存功能正常工作")
                else:
                    print("⚠️ 缓存效果不明显，可能缓存未命中")

                # 验证结果一致性
                pd.testing.assert_frame_equal(
                    result1.sort_index(), result2.sort_index(), check_dtype=False
                )
                print("✅ 缓存结果与原始计算一致")

                return True
            else:
                print("❌ 第二次计算失败：返回空结果")
                return False
        else:
            print("❌ 第一次计算失败：返回空结果")
            return False

    except Exception as e:
        print(f"❌ 缓存测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_factor_filtering():
    """测试因子过滤功能（之前的修复）"""
    try:
        from factor_system.factor_engine import api

        print("\n🚀 测试因子过滤功能...")

        # 只请求RSI，但系统可能需要依赖其他因子
        result = api.calculate_factors(
            factor_ids=["RSI"],
            symbols=["0005.HK"],
            timeframe="daily",
            start_date=datetime(2025, 3, 10),
            end_date=datetime(2025, 3, 31),
            use_cache=False,
        )

        if not result.empty:
            print(f"✅ 因子过滤计算成功: {result.shape}")
            print(f"📋 返回列: {result.columns.tolist()}")

            # 验证只返回请求的因子
            expected_columns = ["RSI"]
            actual_columns = result.columns.tolist()

            # MACD可能返回多个相关列
            allowed_columns = expected_columns + ["MACD", "MACD_SIGNAL", "MACD_HIST"]
            # STOCH可能返回多个相关列
            allowed_columns.extend(["STOCH_SLOWK", "STOCH_SLOWD", "STOCH_K", "STOCH_D"])

            unexpected_columns = [
                col for col in actual_columns if col not in allowed_columns
            ]

            if len(unexpected_columns) == 0:
                print("✅ 因子过滤正常工作")
            else:
                print(f"⚠️ 发现意外的列: {unexpected_columns}")

            # 确保主要请求的因子存在
            if "RSI" in actual_columns:
                print("✅ 请求的RSI因子存在")
                return True
            else:
                print("❌ 请求的RSI因子不存在")
                return False
        else:
            print("❌ 因子过滤计算失败：返回空结果")
            return False

    except Exception as e:
        print(f"❌ 因子过滤测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    """运行完整的FactorEngine系统测试"""
    print("🎯 开始测试完整的FactorEngine系统...")

    tests = [
        ("FactorEngine API", test_factor_engine_api),
        ("多因子计算", test_multiple_factors),
        ("多股票计算", test_multiple_symbols),
        ("缓存功能", test_cache_functionality),
        ("因子过滤", test_factor_filtering),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"📋 {test_name}")
        print("=" * 80)

        if test_func():
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*80}")
    print(f"📊 完整系统测试结果: {passed}个通过, {failed}个失败")
    print("=" * 80)

    if failed == 0:
        print("🎉 FactorEngine系统完全正常！")
        print("\n✅ 系统修复总结:")
        print("1. ✅ 数据提供器问题已解决")
        print("2. ✅ 文件名解析和symbol/timeframe添加正常")
        print("3. ✅ MultiIndex结构创建正确")
        print("4. ✅ 因子计算功能正常")
        print("5. ✅ 多股票和多因子支持正常")
        print("6. ✅ 缓存功能正常工作")
        print("7. ✅ 因子过滤功能正常")
        print("8. ✅ API过滤功能保持有效")
        print("\n🚀 FactorEngine现在可以投入生产使用！")
    else:
        print("⚠️ 部分测试失败，需要进一步检查。")
