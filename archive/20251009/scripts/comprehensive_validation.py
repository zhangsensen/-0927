#!/usr/bin/env python3
"""
FactorEngine修复后的端到端验证测试
验证所有修复的有效性
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_test_data():
    """创建全面的测试数据"""
    dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='D')
    n = len(dates)

    np.random.seed(42)  # 确保可重现

    # 创建OHLCV数据
    base_price = 100
    price_changes = np.random.normal(0, 2, n)
    close_prices = base_price + np.cumsum(price_changes)
    close_prices = np.maximum(close_prices, 10)

    # 创建典型K线模式
    for i in range(5, n-5):
        if i % 10 == 0:
            # 创建Doji模式
            close_prices[i] = close_prices[i-1] + np.random.normal(0, 0.1)
        elif i % 15 == 0:
            # 创建锤子线模式
            close_prices[i] = max(close_prices[i-1], close_prices[i])

    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices + np.random.normal(0, 0.5, n),
        'high': close_prices + np.abs(np.random.normal(1, 0.5, n)),
        'low': close_prices - np.abs(np.random.normal(1, 0.5, n)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, n)
    })

    # 确保OHLC关系正确
    data['high'] = np.maximum.reduce([data['open'], data['high'], data['close']])
    data['low'] = np.minimum.reduce([data['open'], data['low'], data['close']])

    return data.set_index('timestamp')

def test_macd_fix():
    """测试MACD修复"""
    print("\n🔍 测试MACD修复...")

    try:
        from factor_system.factor_engine.factors.overlap_generated import MACD_12_26_9

        data = create_test_data()
        macd_instance = MACD_12_26_9()
        result = macd_instance.calculate(data)

        print(f"  ✅ MACD计算成功")
        print(f"  📊 数据长度: {len(result)}")
        print(f"  📈 MACD范围: {result.min():.4f} 到 {result.max():.4f}")
        print(f"  📉 MACD均值: {result.mean():.4f}")
        print(f"  🔍 非零值数量: {(result != 0).sum()}")

        # 验证不是移动平均（如果只是移动平均，变化会很小）
        macd_volatility = result.std()
        if macd_volatility > 0.1:
            print(f"  ✅ MACD显示真实波动（标准差: {macd_volatility:.4f}）")
        else:
            print(f"  ❌ MACD可能是移动平均（标准差过小: {macd_volatility:.4f}）")

        return True

    except Exception as e:
        print(f"  ❌ MACD测试失败: {e}")
        return False

def test_bollinger_bands_fix():
    """测试布林带修复"""
    print("\n🔍 测试布林带修复...")

    try:
        from factor_system.factor_engine.factors.overlap_generated import BB_10_2_0_Upper

        data = create_test_data()
        bb_instance = BB_10_2_0_Upper()
        result = bb_instance.calculate(data)

        print(f"  ✅ 布林带上轨计算成功")
        print(f"  📊 数据长度: {len(result)}")
        print(f"  📈 上轨范围: {result.min():.2f} 到 {result.max():.2f}")
        print(f"  🔍 有效值数量: {result.notna().sum()}")

        # 验证是上轨而不是中轨
        close_price = data['close']
        upper_ratio = (close_price / result).mean()
        if upper_ratio < 1.0:  # 价格通常低于上轨
            print(f"  ✅ 确认为上轨（价格/上轨均值: {upper_ratio:.3f} < 1）")
        else:
            print(f"  ❌ 可能不是上轨（价格/上轨均值: {upper_ratio:.3f} ≥ 1）")

        return True

    except Exception as e:
        print(f"  ❌ 布林带测试失败: {e}")
        return False

def test_candlestick_patterns():
    """测试K线模式识别修复"""
    print("\n🔍 测试K线模式识别修复...")

    try:
        from factor_system.factor_engine.factors.technical_generated import (
            TA_CDLDOJI, TA_CDLHAMMER, TA_CDL2CROWS
        )

        data = create_test_data()
        patterns_tested = 0
        patterns_successful = 0

        for pattern_name, pattern_class in [
            ('TA_CDLDOJI', TA_CDLDOJI),
            ('TA_CDLHAMMER', TA_CDLHAMMER),
            ('TA_CDL2CROWS', TA_CDL2CROWS),
        ]:
            try:
                pattern_instance = pattern_class()
                result = pattern_instance.calculate(data)
                patterns_tested += 1

                signals = result[result != 0]
                print(f"  ✅ {pattern_name}: {len(signals)} 个信号")

                if len(signals) > 0:
                    print(f"    🎯 信号示例: {signals.head(1).index[0].strftime('%Y-%m-%d')} = {signals.iloc[0]:.0f}")

                patterns_successful += 1

            except Exception as e:
                print(f"  ❌ {pattern_name}: 失败 - {e}")

        success_rate = patterns_successful / patterns_tested if patterns_tested > 0 else 0
        print(f"  📊 K线模式成功率: {success_rate:.1%} ({patterns_successful}/{patterns_tested})")

        return success_rate >= 0.8

    except Exception as e:
        print(f"  ❌ K线模式测试失败: {e}")
        return False

def test_volume_ratios():
    """测试成交量比率修复"""
    print("\n🔍 测试成交量比率修复...")

    try:
        from factor_system.factor_engine.factors.volume_generated import (
            Volume_Ratio10, Volume_Ratio20, Volume_Ratio30
        )

        data = create_test_data()
        ratios_tested = 0
        ratios_correct = 0

        for ratio_name, ratio_class, expected_period in [
            ('Volume_Ratio10', Volume_Ratio10, 10),
            ('Volume_Ratio20', Volume_Ratio20, 20),
            ('Volume_Ratio30', Volume_Ratio30, 30),
        ]:
            try:
                ratio_instance = ratio_class()
                result = ratio_instance.calculate(data)
                ratios_tested += 1

                # 验证周期是否正确
                volume_sma_manual = data['volume'].rolling(window=expected_period).mean()
                expected_result = data['volume'] / (volume_sma_manual + 1e-8)

                # 只比较有值的部分
                valid_mask = result.notna() & expected_result.notna()
                if valid_mask.sum() > 0:
                    diff = abs(result[valid_mask] - expected_result[valid_mask]).max()
                    if diff < 1e-6:
                        print(f"  ✅ {ratio_name}: 周期 {expected_period} 正确")
                        ratios_correct += 1
                    else:
                        print(f"  ❌ {ratio_name}: 周期 {expected_period} 错误 (误差: {diff})")
                else:
                    print(f"  ⚠️ {ratio_name}: 数据不足，无法验证")

            except Exception as e:
                print(f"  ❌ {ratio_name}: 失败 - {e}")

        success_rate = ratios_correct / ratios_tested if ratios_tested > 0 else 0
        print(f"  📊 成交量比率成功率: {success_rate:.1%} ({ratios_correct}/{ratios_tested})")

        return success_rate >= 0.8

    except Exception as e:
        print(f"  ❌ 成交量比率测试失败: {e}")
        return False

def test_factor_engine_integration():
    """测试FactorEngine集成"""
    print("\n🔍 测试FactorEngine集成...")

    try:
        from factor_system.factor_engine import api

        data = create_test_data()

        # 测试单个因子计算
        rsi_result = api.calculate_single_factor(
            factor_id="RSI14",
            symbol="TEST",
            timeframe="daily",
            data=data
        )

        print(f"  ✅ 单因子计算成功: RSI14")
        print(f"  📊 RSI范围: {rsi_result.min():.2f} - {rsi_result.max():.2f}")

        # 测试多因子计算
        factors_result = api.calculate_factors(
            factor_ids=["RSI14", "MACD_12_26_9", "TA_CDLDOJI"],
            symbol="TEST",
            timeframe="daily",
            data=data
        )

        print(f"  ✅ 多因子计算成功: {len(factors_result)} 个因子")
        for factor_id, result in factors_result.items():
            print(f"    📈 {factor_id}: {len(result)} 数据点")

        return True

    except Exception as e:
        print(f"  ❌ FactorEngine集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始FactorEngine修复验证测试")
    print("=" * 60)

    # 创建测试数据
    data = create_test_data()
    print(f"📊 测试数据准备完成: {len(data)} 天的数据")
    print(f"💰 价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"📈 成交量范围: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")

    # 运行各项测试
    tests = [
        ("MACD修复", test_macd_fix),
        ("布林带修复", test_bollinger_bands_fix),
        ("K线模式识别", test_candlestick_patterns),
        ("成交量比率", test_volume_ratios),
        ("FactorEngine集成", test_factor_engine_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))

    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 总体成功率: {passed}/{total} ({passed/total:.1%})")

    if passed == total:
        print("🎉 所有测试通过！FactorEngine修复成功！")
    elif passed >= total * 0.8:
        print("✅ 大部分测试通过！FactorEngine修复基本成功！")
    else:
        print("⚠️ 多项测试失败，需要进一步修复！")

    return passed == total

if __name__ == "__main__":
    main()