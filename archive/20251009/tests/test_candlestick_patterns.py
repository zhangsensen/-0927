#!/usr/bin/env python3
"""
测试K线模式识别修复
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_candlestick_patterns():
    """测试K线模式识别"""

    # 创建测试数据
    dates = pd.date_range(start='2025-01-01', end='2025-01-20', freq='D')
    n = len(dates)

    # 创建OHLCV数据，包含一些典型的K线模式
    np.random.seed(42)  # 确保可重现

    # 基础价格
    base_price = 100
    price_changes = np.random.normal(0, 2, n)
    close_prices = base_price + np.cumsum(price_changes)

    # 确保价格为正
    close_prices = np.maximum(close_prices, 10)

    # 创建典型的K线模式数据
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

    # 设置时间戳为索引
    data = data.set_index('timestamp')

    print("测试数据创建完成，包含OHLCV数据")
    print(data.head())

    # 测试FactorEngine的K线模式识别
    try:
        from factor_system.factor_engine.factors.technical_generated import TA_CDL2CROWS, TA_CDLHAMMER, TA_CDLDOJI

        # 测试几个K线模式
        patterns_to_test = [
            ('TA_CDL2CROWS', TA_CDL2CROWS),
            ('TA_CDLHAMMER', TA_CDLHAMMER),
            ('TA_CDLDOJI', TA_CDLDOJI)
        ]

        print("\n开始测试K线模式识别...")

        for pattern_name, pattern_class in patterns_to_test:
            try:
                pattern_instance = pattern_class()
                result = pattern_instance.calculate(data)

                print(f"\n{pattern_name} 测试结果:")
                print(f"  数据类型: {type(result)}")
                print(f"  数据长度: {len(result)}")
                print(f"  非零值数量: {(result != 0).sum()}")
                print(f"  最大值: {result.max()}")
                print(f"  最小值: {result.min()}")
                print(f"  平均值: {result.mean()}")
                print(f"  前5个值: {result.head().tolist()}")

                # 显示有信号的位置
                signals = result[result != 0]
                if len(signals) > 0:
                    print(f"  🎯 发现 {len(signals)} 个信号:")
                    for date, value in signals.items():
                        signal_type = "看涨" if value > 0 else "看跌"
                        print(f"    {date.strftime('%Y-%m-%d')}: {signal_type} ({value})")
                else:
                    print(f"  ⚪ 未检测到该模式")

            except Exception as e:
                print(f"❌ {pattern_name} 测试失败: {e}")

    except ImportError as e:
        print(f"❌ 导入K线模式失败: {e}")
        return False

    print("\n✅ K线模式识别测试完成！")
    return True

if __name__ == "__main__":
    test_candlestick_patterns()