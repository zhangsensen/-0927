#!/usr/bin/env python3
"""
测试成交量比率修复效果
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def test_volume_ratios():
    """测试成交量比率"""

    # 创建测试数据
    dates = pd.date_range(start="2025-01-01", end="2025-01-20", freq="D")
    n = len(dates)

    # 创建OHLCV数据
    np.random.seed(42)
    base_volume = 1000000
    volume_changes = np.random.normal(1.0, 0.3, n)
    volumes = base_volume * np.cumprod(volume_changes)

    # 确保成交量为正
    volumes = np.maximum(volumes, 100000)

    # 创建价格数据
    base_price = 100
    price_changes = np.random.normal(0, 1, n)
    close_prices = base_price + np.cumsum(price_changes)
    close_prices = np.maximum(close_prices, 10)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "open": close_prices + np.random.normal(0, 0.5, n),
            "high": close_prices + np.abs(np.random.normal(1, 0.5, n)),
            "low": close_prices - np.abs(np.random.normal(1, 0.5, n)),
            "close": close_prices,
            "volume": volumes.astype(int),
        }
    )

    # 确保OHLC关系正确
    data["high"] = np.maximum.reduce([data["open"], data["high"], data["close"]])
    data["low"] = np.minimum.reduce([data["open"], data["low"], data["close"]])

    # 设置时间戳为索引
    data = data.set_index("timestamp")

    print("测试数据创建完成，包含OHLCV数据")
    print(f"成交量范围: {data['volume'].min():,.0f} - {data['volume'].max():,.0f}")

    # 测试FactorEngine的成交量比率
    try:
        from factor_system.factor_engine.factors.volume_generated import (
            Volume_Ratio10,
            Volume_Ratio15,
            Volume_Ratio20,
            Volume_Ratio25,
            Volume_Ratio30,
        )

        # 测试不同周期的成交量比率
        ratios_to_test = [
            ("Volume_Ratio10", Volume_Ratio10, 10),
            ("Volume_Ratio15", Volume_Ratio15, 15),
            ("Volume_Ratio20", Volume_Ratio20, 20),
            ("Volume_Ratio25", Volume_Ratio25, 25),
            ("Volume_Ratio30", Volume_Ratio30, 30),
        ]

        print("\n开始测试成交量比率...")

        for ratio_name, ratio_class, expected_period in ratios_to_test:
            try:
                ratio_instance = ratio_class()
                result = ratio_instance.calculate(data)

                print(f"\n{ratio_name} (预期周期: {expected_period}) 测试结果:")
                print(f"  数据类型: {type(result)}")
                print(f"  数据长度: {len(result)}")
                print(f"  有效值数量: {result.notna().sum()}")
                print(f"  最大值: {result.max():.3f}")
                print(f"  最小值: {result.min():.3f}")
                print(f"  平均值: {result.mean():.3f}")
                print(f"  标准差: {result.std():.3f}")
                print(f"  前5个值: {result.head().round(3).tolist()}")

                # 验证周期是否正确
                # 计算理论上的移动平均值，验证周期
                volume_sma_manual = (
                    data["volume"].rolling(window=expected_period).mean()
                )
                expected_result = data["volume"] / (volume_sma_manual + 1e-8)

                # 比较结果
                diff = abs(result - expected_result).max()
                print(f"  🎯 周期验证误差: {diff:.10f} (应该接近0)")

                if diff < 1e-6:
                    print(f"  ✅ {ratio_name} 使用了正确的周期 {expected_period}")
                else:
                    print(f"  ❌ {ratio_name} 可能使用了错误的周期")

            except Exception as e:
                print(f"❌ {ratio_name} 测试失败: {e}")

    except ImportError as e:
        print(f"❌ 导入成交量比率失败: {e}")
        return False

    print("\n✅ 成交量比率测试完成！")
    return True


if __name__ == "__main__":
    test_volume_ratios()
