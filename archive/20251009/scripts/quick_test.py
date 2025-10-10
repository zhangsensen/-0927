#!/usr/bin/env python3
"""
快速测试修复效果
"""
import pandas as pd
import numpy as np

def quick_test():
    """快速测试"""
    print("🚀 快速测试FactorEngine修复效果")

    # 创建简单测试数据
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100.5, 101.5, 102.5, 103.5, 104.5],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })

    print(f"📊 测试数据: {len(data)} 行")

    # 测试K线模式识别
    try:
        from factor_system.factor_engine.factors.technical_generated import TA_CDLDOJI
        doji = TA_CDLDOJI()
        result = doji.calculate(data)
        print(f"✅ TA_CDLDOJI: 成功，{len(result)} 个结果")
    except Exception as e:
        print(f"❌ TA_CDLDOJI: 失败 - {e}")

    # 测试成交量比率
    try:
        from factor_system.factor_engine.factors.volume_generated import Volume_Ratio10
        vol_ratio = Volume_Ratio10()
        result = vol_ratio.calculate(data)
        print(f"✅ Volume_Ratio10: 成功，{len(result)} 个结果")
    except Exception as e:
        print(f"❌ Volume_Ratio10: 失败 - {e}")

    # 测试MACD
    try:
        from factor_system.factor_engine.factors.overlap_generated import MACD_12_26_9
        macd = MACD_12_26_9()
        result = macd.calculate(data)
        print(f"✅ MACD_12_26_9: 成功，{len(result)} 个结果")
    except Exception as e:
        print(f"❌ MACD_12_26_9: 失败 - {e}")

    # 测试布林带
    try:
        from factor_system.factor_engine.factors.overlap_generated import BB_10_2_0_Upper
        bb = BB_10_2_0_Upper()
        result = bb.calculate(data)
        print(f"✅ BB_10_2_0_Upper: 成功，{len(result)} 个结果")
    except Exception as e:
        print(f"❌ BB_10_2_0_Upper: 失败 - {e}")

    print("🎯 快速测试完成！")

if __name__ == "__main__":
    quick_test()