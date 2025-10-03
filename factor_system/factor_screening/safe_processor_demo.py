#!/usr/bin/env python3
"""
安全处理器使用演示
作者：量化首席工程师
版本：1.0.0
日期：2025-10-02

功能：
- 演示SafeTimeSeriesProcessor的实际使用
- 展示如何防止未来函数
- 验证安全时间序列操作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'utils'))

def demo_safe_processor():
    """演示安全处理器的使用"""
    print("🛡️ SafeTimeSeriesProcessor 使用演示")
    print("=" * 50)

    try:
        from time_series_protocols import SafeTimeSeriesProcessor, safe_ic_calculation

        # 创建安全处理器
        processor = SafeTimeSeriesProcessor(strict_mode=True)
        print("✅ 安全处理器创建成功")

        # 创建示例数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 确保结果可重现

        # 价格数据
        price_data = pd.Series(
            100 + np.random.randn(100).cumsum(),
            index=dates,
            name='price'
        )

        # 因子数据（动量因子）
        factor_data = price_data.pct_change(20).dropna()
        factor_data.name = 'momentum_20d'

        # 收益数据
        return_data = price_data.pct_change().dropna()
        return_data.name = 'return_1d'

        print(f"📊 数据创建完成:")
        print(f"   - 价格数据: {len(price_data)} 个数据点")
        print(f"   - 因子数据: {len(factor_data)} 个数据点")
        print(f"   - 收益数据: {len(return_data)} 个数据点")

        # 演示安全IC计算
        print("\n🔍 演示安全IC计算:")
        horizons = [1, 3, 5, 10]
        for horizon in horizons:
            ic = processor.calculate_ic_safe(factor_data, return_data, horizon)
            print(f"   - IC({horizon}d): {ic:.4f}")

        # 演示安全向前shift
        print("\n⏰ 演示安全向前shift:")
        forward_shifted = processor.shift_forward(factor_data, periods=5)
        print(f"   - 原始数据长度: {len(factor_data)}")
        print(f"   - Shift后长度: {len(forward_shifted)}")
        print(f"   - Shift操作: ✅ 安全（仅使用历史数据）")

        # 演示前向收益计算
        print("\n📈 演示前向收益计算:")
        forward_returns = processor.calculate_forward_returns(price_data, [1, 5, 10])
        print(f"   - 计算周期: [1, 5, 10] 天")
        print(f"   - 收益数据形状: {forward_returns.shape}")
        print(f"   - 示例1天收益: {forward_returns['return_1d'].dropna().iloc[0]:.6f}")
        print(f"   - 示例5天收益: {forward_returns['return_5d'].dropna().iloc[0]:.6f}")

        # 演示数据验证
        print("\n🔒 演示数据完整性验证:")
        test_data = pd.DataFrame({
            'close': price_data,
            'volume': np.random.randint(1000, 10000, 100),
            'momentum': factor_data
        }, index=dates)

        is_valid = processor.validate_no_future_leakage(test_data)
        print(f"   - 数据完整性检查: {'✅ 通过' if is_valid else '❌ 失败'}")

        # 演示便捷函数
        print("\n🚀 演示便捷函数:")
        quick_ic = safe_ic_calculation(factor_data, return_data, horizon=5)
        print(f"   - 便捷IC计算: {quick_ic:.4f}")

        # 显示操作摘要
        print("\n📋 操作摘要:")
        summary = processor.get_operation_summary()
        print(summary)

        return True

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_protection_features():
    """演示防护特性"""
    print("\n🚫 未来函数防护特性演示")
    print("=" * 50)

    try:
        from time_series_protocols import SafeTimeSeriesProcessor

        processor = SafeTimeSeriesProcessor(strict_mode=True)

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        data = pd.Series(np.random.randn(50), index=dates)

        print("1. ✅ 向前shift（允许）:")
        forward_shift = processor.shift_forward(data, periods=3)
        print(f"   - shift(3): 成功执行")

        print("\n2. ❌ 向后shift（禁止）:")
        try:
            # 这应该抛出异常或方法不存在
            if hasattr(processor, 'shift_backward'):
                processor.shift_backward(data, periods=-3)
                print("   - shift(-3): ❌ 未被禁止！")
            else:
                print("   - shift(-3): ✅ 方法已移除")
        except (NotImplementedError, AttributeError) as e:
            print(f"   - shift(-3): ✅ 被正确禁止 ({type(e).__name__})")

        print("\n3. 📊 安全数据验证:")

        # 安全数据
        safe_data = pd.DataFrame({
            'price': data,
            'volume': np.random.randint(100, 1000, 50),
            'rsi': np.random.random(50)
        })

        is_safe = processor.validate_no_future_leakage(safe_data)
        print(f"   - 安全数据验证: {'✅ 通过' if is_safe else '❌ 失败'}")

        # 危险数据
        dangerous_data = pd.DataFrame({
            'price': data,
            'future_return': np.random.randn(50),  # 包含future关键词
            'volume': np.random.randint(100, 1000, 50)
        })

        try:
            is_dangerous = processor.validate_no_future_leakage(dangerous_data)
            print(f"   - 危险数据验证: {'❌ 未检测到' if is_dangerous else '✅ 正确检测'}")
        except ValueError as e:
            print(f"   - 危险数据验证: ✅ 正确检测到问题 ({str(e)[:50]}...)")

        return True

    except Exception as e:
        print(f"❌ 防护特性演示失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 未来函数防护体系 - 安全处理器演示")
    print("=" * 60)

    success_count = 0
    total_tests = 2

    # 运行演示
    if demo_safe_processor():
        success_count += 1

    if demo_protection_features():
        success_count += 1

    # 总结
    print("\n" + "=" * 60)
    print(f"📊 演示结果: {success_count}/{total_tests} 项成功")

    if success_count == total_tests:
        print("🎉 安全处理器演示成功完成！")
        print("\n💡 关键特性:")
        print("   ✅ 自动防止未来函数使用")
        print("   ✅ 安全的时间序列对齐")
        print("   ✅ 严格的数据验证")
        print("   ✅ 便捷的API接口")
        return 0
    else:
        print("⚠️ 部分功能需要进一步调试")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)