#!/usr/bin/env python3
"""
实际项目性能分析示例
分析因子系统中的真实函数
"""

from line_profiler import profile
import sys
import os

# 添加项目路径
sys.path.insert(0, '/Users/zhangshenshen/深度量化0927/factor_system/factor_screening')

@profile
def test_real_function():
    """测试实际项目中的函数"""
    try:
        # 模拟真实调用
        import numpy as np
        import pandas as pd

        # 模拟因子计算
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        data = np.random.randn(1000, 20)
        df = pd.DataFrame(data, index=dates, columns=[f'factor_{i}' for i in range(20)])

        # 模拟计算密集型操作
        result = df.rolling(window=20).mean()
        result = result.dropna()

        print(f"✅ 模拟函数完成，输出形状: {result.shape}")
        return result

    except Exception as e:
        print(f"❌ 错误: {e}")
        return None

if __name__ == "__main__":
    test_real_function()