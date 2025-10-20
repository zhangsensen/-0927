#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FutureFunctionGuard 高级使用示例
演示高级功能包括批量处理、自定义验证器、集成使用等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入FutureFunctionGuard高级功能
from factor_system.future_function_guard import (
    FutureFunctionGuard,
    GuardConfig,
    RuntimeValidationConfig,
    StaticChecker,
    HealthMonitor,
    batch_safe,
    safe_shift,
    monitor_factor_health,
    validate_time_series
)

# ==================== 批量处理示例 ====================

@batch_safe(batch_size=100, validate_batch=True, aggregate_results=True)
def calculate_factors_batch(symbols_list):
    """
    批量计算因子的高级示例
    自动分批处理和验证每个批次的结果
    """
    results = {}

    for symbol in symbols_list:
        try:
            # 模拟数据获取和因子计算
            np.random.seed(hash(symbol) % 2**32)
            dates = pd.date_range(start="2020-01-01", periods=500, freq="D")

            # 模拟价格数据
            price_data = pd.Series(100.0, index=dates)
            returns = np.random.normal(0.001, 0.02, len(dates))
            price_data = price_data * (1 + returns).cumprod()

            # 计算多个因子
            factors = pd.DataFrame({
                'MA_10': price_data.rolling(10).mean(),
                'MA_30': price_data.rolling(30).mean(),
                'RSI_14': calculate_rsi_simple(price_data, 14),
                'Volatility_20': price_data.pct_change().rolling(20).std(),
                'Momentum_20': price_data.pct_change(20)
            })

            results[symbol] = factors

        except Exception as e:
            print(f"计算 {symbol} 失败: {e}")
            results[symbol] = None

    return results

def calculate_rsi_simple(data, periods=14):
    """简单RSI计算"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def demonstrate_batch_processing():
    """演示批量处理功能"""
    print("=== 批量处理演示 ===")

    # 创建大量标的列表
    symbols = [f"STOCK_{i:04d}" for i in range(1, 501)]  # 500个标的
    print(f"创建了 {len(symbols)} 个标的进行批量处理")

    # 执行批量计算
    start_time = time.time()

    try:
        results = calculate_factors_batch(symbols)

        processing_time = time.time() - start_time
        successful_count = sum(1 for r in results.values() if r is not None)

        print(f"✅ 批量处理完成:")
        print(f"  - 处理时间: {processing_time:.2f}秒")
        print(f"  - 成功处理: {successful_count}/{len(symbols)}")
        print(f"  - 平均处理速度: {len(symbols)/processing_time:.1f} 标的/秒")

        # 展示部分结果
        if successful_count > 0:
            sample_symbol = next(k for k, v in results.items() if v is not None)
            sample_data = results[sample_symbol]
            print(f"\n样本结果 ({sample_symbol}):")
            print(f"  - 数据形状: {sample_data.shape}")
            print(f"  - 因子列: {list(sample_data.columns)}")
            print(f"  - 有效数据点: {sample_data.dropna().shape[0]}")

    except Exception as e:
        print(f"❌ 批量处理失败: {e}")

# ==================== 自定义验证器示例 ====================

class CustomFactorValidator:
    """自定义因子验证器"""

    def __init__(self, min_ic=0.02, max_turnover=0.4):
        self.min_ic = min_ic
        self.max_turnover = max_turnover

    def validate_ic_performance(self, factor_data, return_data, horizon=1):
        """验证IC表现"""
        try:
            # 对齐数据
            common_index = factor_data.notna() & return_data.notna()
            if common_index.sum() < 30:
                return {"valid": False, "reason": "有效数据不足"}

            aligned_factor = factor_data[common_index]
            aligned_return = return_data[common_index].shift(horizon).dropna()
            aligned_factor = aligned_factor.loc[aligned_return.index]

            # 计算IC
            ic = aligned_factor.corr(aligned_return)
            ic_mean = ic.mean() if len(ic) > 1 else ic
            ic_std = ic.std() if len(ic) > 1 else 0
            ir = ic_mean / ic_std if ic_std != 0 else 0

            result = {
                "valid": True,
                "ic_mean": float(ic_mean),
                "ic_std": float(ic_std),
                "ir": float(ir),
                "ic_count": len(aligned_factor)
            }

            # 检查IC是否满足要求
            if abs(ic_mean) < self.min_ic:
                result["valid"] = False
                result["reason"] = f"IC均值 {ic_mean:.4f} 低于阈值 {self.min_ic}"

            return result

        except Exception as e:
            return {"valid": False, "reason": f"IC计算失败: {e}"}

    def validate_turnover(self, factor_data, threshold=0.1):
        """验证换手率"""
        try:
            # 计算因子分位数
            ranks = factor_data.rank(pct=True)

            # 计算换手率（相邻时期排名变化）
            turnover = ranks.diff().abs().mean()

            result = {
                "valid": True,
                "turnover": float(turnover),
                "threshold": threshold
            }

            if turnover > self.max_turnover:
                result["valid"] = False
                result["reason"] = f"换手率 {turnover:.3f} 超过阈值 {self.max_turnover}"

            return result

        except Exception as e:
            return {"valid": False, "reason": f"换手率计算失败: {e}"}

def demonstrate_custom_validation():
    """演示自定义验证功能"""
    print("\n=== 自定义验证演示 ===")

    # 创建自定义验证器
    validator = CustomFactorValidator(min_ic=0.03, max_turnover=0.3)

    # 创建测试数据
    dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
    np.random.seed(42)

    # 模拟因子数据
    factor_data = pd.Series(np.random.randn(252), index=dates)

    # 模拟收益率数据（与因子有一定相关性）
    returns = pd.Series(
        0.001 + 0.02 * factor_data + np.random.randn(252) * 0.01,
        index=dates
    )

    print(f"创建了 {len(factor_data)} 天的测试数据")

    # 执行自定义验证
    print("\n执行自定义验证:")

    # IC验证
    ic_result = validator.validate_ic_performance(factor_data, returns)
    print(f"✅ IC验证:")
    print(f"  - 验证状态: {'通过' if ic_result['valid'] else '失败'}")
    if ic_result['valid']:
        print(f"  - IC均值: {ic_result['ic_mean']:.4f}")
        print(f"  - IR: {ic_result['ir']:.4f}")
    else:
        print(f"  - 失败原因: {ic_result['reason']}")

    # 换手率验证
    turnover_result = validator.validate_turnover(factor_data)
    print(f"\n✅ 换手率验证:")
    print(f"  - 验证状态: {'通过' if turnover_result['valid'] else '失败'}")
    if turnover_result['valid']:
        print(f"  - 换手率: {turnover_result['turnover']:.3f}")
    else:
        print(f"  - 失败原因: {turnover_result['reason']}")

# ==================== 多线程安全验证示例 ====================

def parallel_factor_validation(symbols, guard):
    """并行验证多个因子"""
    def validate_single_symbol(symbol):
        try:
            # 模拟数据获取
            np.random.seed(hash(symbol) % 2**32)
            dates = pd.date_range(start="2022-01-01", periods=100, freq="D")

            price_data = pd.Series(100.0, index=dates)
            returns = np.random.normal(0.001, 0.02, len(dates))
            price_data = price_data * (1 + returns).cumprod()

            # 计算因子
            factor_data = price_data.rolling(10).mean()

            # 验证因子
            result = guard.validate_factor_calculation(
                factor_data,
                factor_id=f"{symbol}_MA10",
                timeframe="daily",
                reference_data=pd.DataFrame({'price': price_data})
            )

            return {
                'symbol': symbol,
                'success': True,
                'result': result,
                'quality_score': result.get('validation_time', 0)
            }

        except Exception as e:
            return {
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'quality_score': 0
            }

    # 使用线程池并行验证
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(validate_single_symbol, symbol): symbol for symbol in symbols}
        results = []

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # 实时进度显示
            progress = len(results) / len(symbols)
            status = "✅" if result['success'] else "❌"
            print(f"\r进度: {progress:.1%} | {status} {result['symbol']}", end="", flush=True)

    print()  # 换行

    return results

def demonstrate_parallel_validation():
    """演示并行验证功能"""
    print("\n=== 并行验证演示 ===")

    # 创建防护组件
    guard = FutureFunctionGuard(GuardConfig.preset("research"))

    # 创建测试标的列表
    symbols = [f"PARALLEL_{i:03d}" for i in range(1, 51)]
    print(f"准备并行验证 {len(symbols)} 个标的...")

    # 执行并行验证
    start_time = time.time()
    results = parallel_factor_validation(symbols, guard)
    validation_time = time.time() - start_time

    # 统计结果
    successful = sum(1 for r in results if r['success'])
    avg_quality = np.mean([r['quality_score'] for r in results if r['success']])

    print(f"\n✅ 并行验证完成:")
    print(f"  - 验证时间: {validation_time:.2f}秒")
    print(f"  - 成功率: {successful}/{len(symbols)} ({successful/len(symbols):.1%})")
    print(f"  - 平均质量评分: {avg_quality:.4f}")
    print(f"  - 验证速度: {len(symbols)/validation_time:.1f} 标的/秒")

# ==================== 时间序列高级验证示例 ====================

@validate_time_series(
    require_datetime_index=True,
    check_monotonic=True,
    check_duplicates=True,
    min_length=50
)
@safe_shift(max_periods=30, allow_negative=False)
@monitor_factor_health(strict_mode=True)
def advanced_time_series_processor(data, operation_type="momentum"):
    """
    高级时间序列处理函数
    集成了多种验证和监控功能
    """
    if operation_type == "momentum":
        # 动量因子
        result = data.pct_change(20)
    elif operation_type == "reversal":
        # 反转因子
        result = -data.pct_change(5)
    elif operation_type == "volatility":
        # 波动率因子
        result = data.pct_change().rolling(20).std()
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")

    return result

def demonstrate_advanced_time_series():
    """演示高级时间序列处理"""
    print("\n=== 高级时间序列处理演示 ===")

    # 创建测试数据
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    np.random.seed(42)

    # 创建有问题的数据进行测试
    price_data = pd.Series(100.0, index=dates)
    returns = np.random.normal(0.001, 0.02, len(dates))
    price_data = price_data * (1 + returns).cumprod()

    print(f"创建了 {len(price_data)} 天的测试数据")

    # 测试不同的操作类型
    operations = ["momentum", "reversal", "volatility"]

    for op in operations:
        print(f"\n处理操作: {op}")
        try:
            start_time = time.time()
            result = advanced_time_series_processor(price_data, op)
            processing_time = time.time() - start_time

            print(f"✅ {op} 操作成功:")
            print(f"  - 处理时间: {processing_time:.3f}秒")
            print(f"  - 结果形状: {result.shape}")
            print(f"  - 有效数据点: {result.notna().sum()}")

        except Exception as e:
            print(f"❌ {op} 操作失败: {e}")

# ==================== 集成监控示例 ====================

def setup_integrated_monitoring():
    """设置集成监控系统"""
    print("\n=== 集成监控设置演示 ===")

    # 创建生产级配置
    config = GuardConfig.preset("production")

    # 启用实时监控
    config.health_monitor.real_time_alerts = True
    config.health_monitor.monitoring_level = "comprehensive"
    config.health_monitor.export_reports = True

    # 创建防护组件
    guard = FutureFunctionGuard(config)

    print("✅ 集成监控系统已配置:")
    print(f"  - 模式: {config.mode}")
    print(f"  - 实时报警: {config.health_monitor.real_time_alerts}")
    print(f"  - 监控级别: {config.health_monitor.monitoring_level}")
    print(f"  - 报告导出: {config.health_monitor.export_reports}")

    return guard

def simulate_integrated_monitoring(guard):
    """模拟集成监控场景"""
    print("\n模拟集成监控场景:")

    # 模拟多个时间点的因子计算和监控
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")

    for i, date in enumerate(dates):
        print(f"\n--- 日期: {date.strftime('%Y-%m-%d')} ---")

        # 模拟当日因子数据
        np.random.seed(int(date.timestamp()))
        factor_data = pd.Series(np.random.randn(100))

        # 计算因子健康（会触发报警）
        health_result = guard.check_factor_health(
            factor_data,
            f"Daily_Factor_{i}",
            strict_mode=False
        )

        print(f"质量评分: {health_result['quality_score']:.1f}")

        # 检查新产生的报警
        current_alert_count = len(guard.health_monitor.alerts)
        if i > 0:
            new_alerts = current_alert_count - (5 + i * 2)  # 估算的基准报警数
            if new_alerts > 0:
                print(f"🚨 新增报警: {new_alerts} 个")

    # 生成监控报告
    print(f"\n生成监控报告...")
    report = guard.generate_comprehensive_report()
    print(report[:300] + "..." if len(report) > 300 else report)

# ==================== 性能优化示例 ====================

def demonstrate_performance_optimization():
    """演示性能优化技巧"""
    print("\n=== 性能优化演示 ===")

    # 创建优化配置
    config = GuardConfig.preset("research")

    # 启用缓存
    config.cache.enabled = True
    config.cache.max_cache_size_mb = 200

    # 调整验证策略
    config.runtime_validation.statistical_checks = False  # 跳过统计检查以提升性能
    config.health_monitor.monitoring_level = "basic"  # 基础监控

    guard = FutureFunctionGuard(config)

    print("✅ 性能优化配置已应用:")
    print(f"  - 缓存启用: {config.cache.enabled}")
    print(f"  - 统计检查: {config.runtime_validation.statistical_checks}")
    print(f"  - 监控级别: {config.health_monitor.monitoring_level}")

    # 性能测试
    symbols = [f"PERF_TEST_{i:03d}" for i in range(1, 101)]

    print(f"\n开始性能测试 ({len(symbols)} 个标的)...")
    start_time = time.time()

    # 批量处理
    results = calculate_factors_batch(symbols[:20])  # 限制数量以避免过长时间

    processing_time = time.time() - start_time

    print(f"✅ 性能测试完成:")
    print(f"  - 处理时间: {processing_time:.2f}秒")
    print(f"  - 处理速度: {len(symbols[:20])/processing_time:.1f} 标的/秒")

    # 缓存效果测试
    print(f"\n缓存效果测试:")
    cache_info = guard.get_cache_info()['cache_info']
    print(f"  - 静态检查缓存文件: {cache_info['static_check']['file_count']}")
    print(f"  - 健康监控缓存文件: {cache_info['health_monitor']['file_count']}")

# ==================== 主函数 ====================

def main():
    """主函数 - 运行所有高级示例"""
    print("🚀 FutureFunctionGuard 高级使用示例")
    print("=" * 60)

    try:
        # 运行高级示例
        demonstrate_batch_processing()
        demonstrate_custom_validation()
        demonstrate_parallel_validation()
        demonstrate_advanced_time_series()

        # 集成监控演示
        guard = setup_integrated_monitoring()
        simulate_integrated_monitoring(guard)

        # 性能优化演示
        demonstrate_performance_optimization()

        print("\n" + "=" * 60)
        print("✅ 所有高级示例运行完成！")
        print("\n💡 高级功能总结:")
        print("- 批量处理: 自动分批和验证")
        print("- 自定义验证: 扩展验证逻辑")
        print("- 并行处理: 多线程安全验证")
        print("- 集成监控: 全面的监控和报警")
        print("- 性能优化: 缓存和配置优化")

    except Exception as e:
        print(f"\n❌ 高级示例运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()