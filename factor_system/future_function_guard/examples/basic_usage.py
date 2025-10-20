#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FutureFunctionGuard 基础使用示例
演示基本的装饰器、函数调用和上下文管理器用法
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入FutureFunctionGuard
from factor_system.future_function_guard import (
    future_safe,
    safe_research,
    safe_production,
    create_guard,
    quick_check,
    validate_factors
)

# ==================== 示例数据准备 ====================

def create_sample_data():
    """创建示例数据"""
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    # 生成模拟价格数据
    price_data = pd.Series(100.0, index=dates)
    returns = np.random.normal(0.001, 0.02, len(dates))
    price_data = price_data * (1 + returns).cumprod()

    return price_data

# ==================== 装饰器示例 ====================

@future_safe()
def calculate_simple_moving_average(data, window=20):
    """
    计算简单移动平均线 - 基础装饰器示例
    """
    return data.rolling(window).mean()

@safe_research()
def calculate_rsi(data, periods=14):
    """
    计算RSI指标 - 研究环境装饰器示例
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@safe_production()
def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    计算布林带 - 生产环境装饰器示例
    """
    sma = data.rolling(window).mean()
    std = data.rolling(window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return pd.DataFrame({
        'sma': sma,
        'upper': upper_band,
        'lower': lower_band,
        'bandwidth': (upper_band - lower_band) / sma
    })

# ==================== 函数调用示例 ====================

def demonstrate_function_calls():
    """演示函数调用模式的防护"""
    print("=== 函数调用模式演示 ===")

    # 创建示例数据
    price_data = create_sample_data()
    print(f"创建了 {len(price_data)} 天的价格数据")

    # 1. 使用装饰器保护的函数
    print("\n1. 装饰器保护的函数:")
    ma_20 = calculate_simple_moving_average(price_data, 20)
    print(f"✅ 20日移动平均线计算完成，有效数据点: {ma_20.notna().sum()}")

    rsi_14 = calculate_rsi(price_data, 14)
    print(f"✅ RSI(14)计算完成，有效数据点: {rsi_14.notna().sum()}")

    bb_20 = calculate_bollinger_bands(price_data, 20)
    print(f"✅ 布林带计算完成，有效数据点: {bb_20['sma'].notna().sum()}")

    # 2. 使用便捷函数验证因子
    print("\n2. 便捷验证函数:")
    factor_panel = pd.DataFrame({
        'MA_20': ma_20,
        'RSI_14': rsi_14,
        'BB_Width': bb_20['bandwidth']
    })

    # 移除NaN值以便验证
    clean_panel = factor_panel.dropna()
    print(f"清理后的因子面板形状: {clean_panel.shape}")

    # 验证因子数据
    validation_result = validate_factors(
        clean_panel,
        factor_ids=['MA_20', 'RSI_14', 'BB_Width'],
        timeframe="daily"
    )

    print(f"✅ 因子验证状态: {validation_result['is_valid']}")
    if validation_result['warnings']:
        print(f"⚠️  警告: {validation_result['warnings']}")

# ==================== 上下文管理器示例 ====================

def demonstrate_context_manager():
    """演示上下文管理器的使用"""
    print("\n=== 上下文管理器演示 ===")

    # 创建防护组件
    guard = create_guard(mode="research")
    price_data = create_sample_data()

    # 使用上下文管理器保护代码块
    print("\n使用保护上下文:")

    with guard.protect(mode="strict"):
        print("进入严格保护模式")

        # 在保护上下文中进行计算
        # 所有时序操作都会被验证
        shifted_returns = price_data.pct_change().shift(1)
        print(f"✅ 收益率shift(1)计算完成")

        # 尝试负数shift（会被阻止或警告）
        try:
            negative_shift = price_data.pct_change().shift(-1)
            print("⚠️  负数shift被允许（在warn_only模式下）")
        except Exception as e:
            print(f"🚫 负数shift被阻止: {e}")

        print("退出保护上下文")

# ==================== 综合检查示例 ====================

def demonstrate_comprehensive_check():
    """演示综合安全检查"""
    print("\n=== 综合安全检查演示 ===")

    # 创建生产环境防护组件
    guard = create_guard(mode="production")

    # 创建测试数据
    price_data = create_sample_data()
    factor_panel = pd.DataFrame({
        'MA_10': price_data.rolling(10).mean(),
        'MA_30': price_data.rolling(30).mean(),
        'Volatility': price_data.pct_change().rolling(20).std(),
        'Momentum': price_data.pct_change(20)
    })

    print(f"创建了 {factor_panel.shape[1]} 个因子，{factor_panel.shape[0]} 个时间点")

    # 执行综合检查
    print("\n执行综合安全检查...")
    result = guard.comprehensive_security_check(
        # 这里可以添加代码文件路径进行静态检查
        # code_targets=["./examples/"],
        data_targets={
            "factor_panel": factor_panel.dropna(),
            "price_data": price_data
        }
    )

    print(f"✅ 综合检查完成")
    print(f"整体状态: {result['overall_status']}")
    print(f"检查耗时: {result['total_time']:.3f}秒")
    print(f"组件检查: {', '.join(result['check_components'])}")

    if result.get('report'):
        print("\n检查报告摘要:")
        print(result['report'][:500] + "..." if len(result['report']) > 500 else result['report'])

# ==================== 错误处理示例 ====================

def demonstrate_error_handling():
    """演示错误处理和报警"""
    print("\n=== 错误处理演示 ===")

    guard = create_guard(mode="research")

    # 创建有问题的数据进行测试
    dates = pd.date_range(start="2023-01-01", periods=50, freq="D")

    # 1. 数据不足的因子
    short_data = pd.Series(np.random.randn(10), index=dates[:10])
    print("\n1. 测试数据不足的情况:")

    try:
        result = guard.validate_factor_calculation(
            short_data,
            factor_id="Short_Factor",
            timeframe="daily"
        )
        print(f"验证结果: {result['is_valid']}")
        print(f"消息: {result['message']}")
    except Exception as e:
        print(f"捕获异常: {e}")

    # 2. 方差过低的因子
    constant_data = pd.Series(1.0, index=dates[:100])
    print("\n2. 测试方差过低的因子:")

    try:
        result = guard.check_factor_health(constant_data, "Constant_Factor")
        print(f"质量评分: {result['quality_score']:.1f}")
        print(f"警告: {len(result['warnings'])} 个")
    except Exception as e:
        print(f"捕获异常: {e}")

    # 3. 查看报警信息
    print(f"\n3. 当前报警数量: {len(guard.health_monitor.alerts)}")
    if guard.health_monitor.alerts:
        print("最近报警:")
        for alert in guard.health_monitor.alerts[-3:]:
            print(f"  - {alert.severity}: {alert.message}")

# ==================== 配置自定义示例 ====================

def demonstrate_custom_config():
    """演示自定义配置"""
    print("\n=== 自定义配置演示 ===")

    from factor_system.future_function_guard import (
        GuardConfig, RuntimeValidationConfig, StrictMode
    )

    # 创建自定义配置
    custom_config = GuardConfig(
        mode="custom",
        strict_mode=StrictMode.WARN_ONLY,
        runtime_validation=RuntimeValidationConfig(
            correlation_threshold=0.98,  # 更宽松的相关性阈值
            coverage_threshold=0.85,       # 更低的覆盖率要求
            time_series_safety=True,
            statistical_checks=True
        )
    )

    # 使用自定义配置创建防护组件
    custom_guard = create_guard()
    custom_guard.update_config(custom_config)

    print(f"✅ 自定义配置已应用")
    print(f"模式: {custom_guard.config.mode}")
    print(f"严格模式: {custom_guard.config.runtime_validation.strict_mode.value}")
    print(f"相关性阈值: {custom_guard.config.runtime_validation.correlation_threshold}")

    # 测试自定义配置的效果
    price_data = create_sample_data()
    factor_data = price_data.rolling(20).mean()

    result = custom_guard.validate_factor_calculation(
        factor_data,
        factor_id="Custom_Test_Factor",
        timeframe="daily"
    )

    print(f"✅ 自定义配置测试: {result['is_valid']}")

# ==================== 性能监控示例 ====================

def demonstrate_performance_monitoring():
    """演示性能监控功能"""
    print("\n=== 性能监控演示 ===")

    guard = create_guard(mode="research")

    # 执行多次操作以生成统计数据
    print("执行多次操作...")

    price_data = create_sample_data()

    # 执行静态检查（如果有的话）
    # guard.check_code_for_future_functions(["./examples/"])

    # 执行多次验证
    for i in range(5):
        factor_data = price_data.rolling(20).mean()
        guard.validate_factor_calculation(factor_data, f"Test_Factor_{i}")
        guard.check_factor_health(factor_data, f"Test_Factor_{i}")

    # 获取统计信息
    stats = guard.get_statistics()

    print(f"✅ 性能统计:")
    print(f"  - 静态检查次数: {stats['static_checks']}")
    print(f"  - 运行时验证次数: {stats['runtime_validations']}")
    print(f"  - 健康检查次数: {stats['health_checks']}")
    print(f"  - 检测问题总数: {stats['issues_detected']}")
    print(f"  - 生成报警总数: {stats['alerts_generated']}")
    print(f"  - 运行时间: {stats['uptime_human']}")

    # 缓存信息
    cache_info = stats['cache_info']
    print(f"\n缓存信息:")
    print(f"  - 静态检查缓存: {cache_info['static_check']['file_count']} 个文件")
    print(f"  - 健康监控缓存: {cache_info['health_monitor']['file_count']} 个文件")

# ==================== 主函数 ====================

def main():
    """主函数 - 运行所有示例"""
    print("🚀 FutureFunctionGuard 基础使用示例")
    print("=" * 60)

    try:
        # 运行各种示例
        demonstrate_function_calls()
        demonstrate_context_manager()
        demonstrate_comprehensive_check()
        demonstrate_error_handling()
        demonstrate_custom_config()
        demonstrate_performance_monitoring()

        print("\n" + "=" * 60)
        print("✅ 所有示例运行完成！")
        print("\n💡 提示:")
        print("- 装饰器模式最适合保护单个函数")
        print("- 上下文管理器适合保护代码块")
        print("- 便捷函数适合快速验证和检查")
        print("- 生产环境建议使用综合安全检查")

    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()