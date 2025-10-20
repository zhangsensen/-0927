#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并行回测引擎性能测试脚本
对比串行和并行版本的性能差异，验证加速效果
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from parallel_backtest_engine import ParallelBacktestEngine


def generate_test_data():
    """生成测试数据"""
    print("生成测试数据...")

    # 创建测试目录
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # 生成模拟面板数据
    n_dates = 252  # 一年的交易日
    n_symbols = 20  # 20个ETF
    n_factors = 10  # 10个因子

    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    symbols = [f"ETF{i:03d}" for i in range(n_symbols)]
    factors = [f"FACTOR_{i}" for i in range(n_factors)]

    # 创建 MultiIndex
    multi_index = pd.MultiIndex.from_product(
        [symbols, dates], names=['symbol', 'date']
    )

    # 生成随机因子数据
    np.random.seed(42)
    factor_data = np.random.randn(len(multi_index), n_factors)
    panel_df = pd.DataFrame(factor_data, index=multi_index, columns=factors)

    # 保存面板数据
    panel_file = test_dir / "test_panel.parquet"
    panel_df.to_parquet(panel_file)

    # 生成模拟价格数据
    price_data = []
    for symbol in symbols:
        # 随机游走价格
        initial_price = 100 + np.random.randn() * 10
        returns = np.random.randn(n_dates) * 0.02  # 2%日波动率
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        for i, (date, price) in enumerate(zip(dates, prices)):
            price_data.append({
                'date': date,
                'symbol': symbol,
                'close': price,
                'trade_date': date.strftime('%Y%m%d')
            })

    price_df = pd.DataFrame(price_data)

    # 保存每个ETF的价格数据
    for symbol in symbols:
        symbol_data = price_df[price_df['symbol'] == symbol]
        price_file = test_dir / f"{symbol}_daily_20230101_20231231.parquet"
        symbol_data[['date', 'close', 'trade_date']].to_parquet(price_file)

    # 生成模拟筛选结果
    screening_data = pd.DataFrame({
        'factor': factors,
        'ic_mean': np.random.randn(n_factors) * 0.05,
        'ic_std': np.abs(np.random.randn(n_factors) * 0.1),
        'sharpe_ratio': np.random.randn(n_factors) * 2,
    })
    screening_df = screening_data.sort_values('sharpe_ratio', ascending=False)

    # 保存筛选结果
    screening_file = test_dir / "test_screening.csv"
    screening_df.to_csv(screening_file, index=False)

    print(f"测试数据生成完成:")
    print(f"  面板数据: {panel_file}")
    print(f"  价格数据: {test_dir}/ETF*_daily_*.parquet")
    print(f"  筛选结果: {screening_file}")

    return str(panel_file), str(test_dir), str(screening_file)


def test_serial_vs_parallel():
    """测试串行vs并行性能"""
    print("\n" + "=" * 80)
    print("串行 vs 并行性能对比测试")
    print("=" * 80)

    # 生成测试数据
    panel_path, price_dir, screening_path = generate_test_data()

    # 测试配置
    test_configs = [
        {"name": "小规模测试", "max_combos": 100, "top_k": 3},
        {"name": "中等规模测试", "max_combos": 500, "top_k": 5},
        {"name": "大规模测试", "max_combos": 1000, "top_k": 8},
    ]

    results = []

    for config in test_configs:
        print(f"\n🧪 {config['name']}:")
        print(f"  权重组合: {config['max_combos']}")
        print(f"  因子数量: {config['top_k']}")

        # 串行测试
        print("\n🔄 串行执行测试...")
        start_time = time.time()

        # 这里应该调用原有的串行版本
        # 为了测试，我们用单进程并行引擎模拟串行
        serial_engine = ParallelBacktestEngine(n_workers=1, chunk_size=1)
        serial_results, serial_config = serial_engine.run_parallel_backtest(
            panel_path=panel_path,
            price_dir=price_dir,
            screening_csv=screening_path,
            output_dir="test_results/serial",
            top_k=config['top_k'],
            top_n_list=[3, 5],
            max_combinations=config['max_combos'],
        )

        serial_time = time.time() - start_time
        serial_speed = len(serial_results) / serial_time

        print(f"  串行耗时: {serial_time:.2f}秒")
        print(f"  串行速度: {serial_speed:.1f}策略/秒")

        # 并行测试
        print("\n🚀 并行执行测试...")
        start_time = time.time()

        # 多进程并行
        import multiprocessing as mp
        n_workers = max(1, mp.cpu_count() - 1)
        parallel_engine = ParallelBacktestEngine(
            n_workers=n_workers,
            chunk_size=20
        )
        parallel_results, parallel_config = parallel_engine.run_parallel_backtest(
            panel_path=panel_path,
            price_dir=price_dir,
            screening_csv=screening_path,
            output_dir="test_results/parallel",
            top_k=config['top_k'],
            top_n_list=[3, 5],
            max_combinations=config['max_combos'],
        )

        parallel_time = time.time() - start_time
        parallel_speed = len(parallel_results) / parallel_time

        print(f"  并行耗时: {parallel_time:.2f}秒")
        print(f"  并行速度: {parallel_speed:.1f}策略/秒")

        # 计算加速比
        speedup = serial_time / parallel_time if parallel_time > 0 else float('inf')
        efficiency = speedup / n_workers * 100 if n_workers > 0 else 0

        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {efficiency:.1f}%")
        print(f"  工作进程: {n_workers}")

        # 记录结果
        results.append({
            'test_name': config['name'],
            'max_combos': config['max_combos'],
            'top_k': config['top_k'],
            'n_workers': n_workers,
            'serial_time': serial_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'serial_speed': serial_speed,
            'parallel_speed': parallel_speed,
            'strategies_tested': len(serial_results),
        })

    return results


def analyze_performance_results(results):
    """分析性能测试结果"""
    print("\n" + "=" * 80)
    print("性能测试结果分析")
    print("=" * 80)

    # 创建结果DataFrame
    df = pd.DataFrame(results)

    print("\n📊 详细结果表:")
    print(df[['test_name', 'max_combos', 'serial_time', 'parallel_time', 'speedup', 'efficiency']].round(2))

    # 分析加速比
    avg_speedup = df['speedup'].mean()
    avg_efficiency = df['efficiency'].mean()

    print(f"\n📈 性能总结:")
    print(f"  平均加速比: {avg_speedup:.2f}x")
    print(f"  平均并行效率: {avg_efficiency:.1f}%")
    print(f"  最优加速比: {df['speedup'].max():.2f}x ({df.loc[df['speedup'].idxmax(), 'test_name']})")

    # 预估实际任务性能
    print(f"\n🔮 实际任务性能预估:")
    actual_combos = 5000
    actual_top_n = 4  # [3,5,8,10]
    actual_strategies = actual_combos * actual_top_n

    for _, row in df.iterrows():
        # 基于测试结果预估
        serial_time_per_strategy = row['serial_time'] / row['strategies_tested']
        parallel_time_per_strategy = row['parallel_time'] / row['strategies_tested']

        estimated_serial = actual_strategies * serial_time_per_strategy
        estimated_parallel = actual_strategies * parallel_time_per_strategy

        print(f"  {row['test_name']} 预估:")
        print(f"    串行: {estimated_serial:.1f}秒 ({estimated_serial/60:.1f}分钟)")
        print(f"    并行: {estimated_parallel:.1f}秒")
        print(f"    加速比: {estimated_serial/estimated_parallel:.2f}x")

    # 保存结果
    results_file = Path("test_results/performance_comparison.json")
    results_file.parent.mkdir(exist_ok=True)

    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'avg_speedup': avg_speedup,
            'avg_efficiency': avg_efficiency,
            'max_speedup': df['speedup'].max(),
        },
        'detailed_results': df.to_dict('records'),
        'hardware_info': {
            'cpu_count': mp.cpu_count(),
            'workers_used': int(df.iloc[0]['n_workers']),
        }
    }

    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细结果保存至: {results_file}")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)

    panel_path, price_dir, screening_path = generate_test_data()

    # 测试单进程
    print("\n🧪 单进程测试...")
    single_engine = ParallelBacktestEngine(n_workers=1)
    start_time = time.time()
    try:
        results, config = single_engine.run_parallel_backtest(
            panel_path, price_dir, screening_path,
            output_dir="test_results/single",
            max_combinations=50
        )
        single_time = time.time() - start_time
        print(f"✅ 单进程测试成功，耗时: {single_time:.2f}秒")
    except Exception as e:
        print(f"❌ 单进程测试失败: {e}")

    # 测试大量小任务
    print("\n🧪 大量小任务测试...")
    chunk_engine = ParallelBacktestEngine(n_workers=4, chunk_size=5)
    start_time = time.time()
    try:
        results, config = chunk_engine.run_parallel_backtest(
            panel_path, price_dir, screening_path,
            output_dir="test_results/chunk",
            max_combinations=200
        )
        chunk_time = time.time() - start_time
        print(f"✅ 大量小任务测试成功，耗时: {chunk_time:.2f}秒")
    except Exception as e:
        print(f"❌ 大量小任务测试失败: {e}")

    # 测试内存使用
    print("\n🧪 内存使用测试...")
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    memory_engine = ParallelBacktestEngine(n_workers=2)
    try:
        results, config = memory_engine.run_parallel_backtest(
            panel_path, price_dir, screening_path,
            output_dir="test_results/memory",
            max_combinations=100
        )
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        print(f"✅ 内存使用测试成功")
        print(f"  内存使用: {memory_before:.1f}MB → {memory_after:.1f}MB (+{memory_used:.1f}MB)")
        print(f"  每策略内存: {memory_used/len(results)*1024:.2f}KB")

    except Exception as e:
        print(f"❌ 内存使用测试失败: {e}")


def main():
    """主函数"""
    print("并行回测引擎性能测试")
    print("=" * 80)

    import multiprocessing as mp
    print(f"硬件信息: {mp.cpu_count()}个CPU核心")

    try:
        # 主要性能对比测试
        results = test_serial_vs_parallel()

        # 分析结果
        analyze_performance_results(results)

        # 边界情况测试
        test_edge_cases()

        print("\n" + "=" * 80)
        print("🎯 测试完成总结")
        print("=" * 80)
        print("✅ 性能对比测试完成")
        print("✅ 边界情况测试完成")
        print("✅ 结果已保存到 test_results/ 目录")
        print("\n💡 建议:")
        print("1. 对于生产环境，建议使用 6-8 个工作进程")
        print("2. chunk_size 建议 10-20，平衡任务分配和内存使用")
        print("3. 大规模回测时注意监控内存使用")
        print("4. 可根据数据规模调整并行参数")

    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()