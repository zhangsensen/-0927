#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ETF轮动系统性能基准测试
对比不同优化策略的性能表现，提供最优配置建议
"""

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from parallel_backtest_engine import ParallelBacktestEngine
from optimized_weight_generator import OptimizedWeightGenerator, WeightGenerationConfig, SearchStrategy


def create_benchmark_suite():
    """创建基准测试套件"""
    print("创建基准测试套件...")

    # 测试配置
    test_configs = [
        {
            "name": "小规模测试",
            "factors": 3,
            "max_combinations": 100,
            "top_n_list": [3, 5],
            "description": "3个因子，100个权重组合"
        },
        {
            "name": "中等规模测试",
            "factors": 5,
            "max_combinations": 500,
            "top_n_list": [3, 5, 8],
            "description": "5个因子，500个权重组合"
        },
        {
            "name": "大规模测试",
            "factors": 8,
            "max_combinations": 1000,
            "top_n_list": [3, 5, 8, 10],
            "description": "8个因子，1000个权重组合"
        },
        {
            "name": "超大规模测试",
            "factors": 10,
            "max_combinations": 2000,
            "top_n_list": [3, 5, 8, 10],
            "description": "10个因子，2000个权重组合"
        }
    ]

    # 优化策略配置
    strategy_configs = [
        {
            "name": "串行网格搜索",
            "n_workers": 1,
            "weight_strategy": SearchStrategy.GRID,
            "chunk_size": 1
        },
        {
            "name": "并行网格搜索",
            "n_workers": 4,
            "weight_strategy": SearchStrategy.GRID,
            "chunk_size": 20
        },
        {
            "name": "并行智能采样",
            "n_workers": 4,
            "weight_strategy": SearchStrategy.SMART,
            "chunk_size": 20
        },
        {
            "name": "并行分层搜索",
            "n_workers": 4,
            "weight_strategy": SearchStrategy.HIERARCHICAL,
            "chunk_size": 20
        },
        {
            "name": "高并发并行",
            "n_workers": 8,
            "weight_strategy": SearchStrategy.SMART,
            "chunk_size": 50
        }
    ]

    return test_configs, strategy_configs


def generate_benchmark_data(n_dates: int = 126, n_symbols: int = 15) -> Tuple[str, str, str]:
    """生成基准测试数据"""
    print(f"生成基准测试数据: {n_dates}天 × {n_symbols}个ETF")

    # 创建测试目录
    test_dir = Path("benchmark_data")
    test_dir.mkdir(exist_ok=True)

    # 生成模拟面板数据
    factors = [f"FACTOR_{i}" for i in range(10)]  # 生成10个因子
    dates = pd.date_range('2023-01-01', periods=n_dates, freq='D')
    symbols = [f"ETF{i:03d}" for i in range(n_symbols)]

    # 创建 MultiIndex
    multi_index = pd.MultiIndex.from_product(
        [symbols, dates], names=['symbol', 'date']
    )

    # 生成有意义的因子数据（加入一些趋势和相关性）
    np.random.seed(42)
    n_samples = len(multi_index)

    # 基础随机数据
    factor_data = np.random.randn(n_samples, len(factors)) * 0.1

    # 添加时间趋势
    time_trend = np.linspace(0, 0.5, n_dates)
    for i in range(len(factors)):
        factor_data[:, i] += np.repeat(time_trend, n_symbols)

    # 添加因子间相关性
    correlation_matrix = np.array([
        [1.0, 0.3, -0.2, 0.1, 0.0, 0.2, -0.1, 0.3, 0.1, -0.2],
        [0.3, 1.0, 0.1, -0.3, 0.2, 0.0, 0.3, -0.1, 0.2, 0.1],
        [-0.2, 0.1, 1.0, 0.2, -0.1, 0.3, 0.0, -0.2, 0.3, 0.1],
        [0.1, -0.3, 0.2, 1.0, 0.3, -0.1, 0.2, 0.0, -0.1, 0.3],
        [0.0, 0.2, -0.1, 0.3, 1.0, 0.1, -0.2, 0.3, 0.0, -0.1],
        [0.2, 0.0, 0.3, -0.1, 0.1, 1.0, 0.2, -0.3, 0.1, 0.2],
        [-0.1, 0.3, 0.0, 0.2, -0.2, 0.2, 1.0, 0.1, -0.3, 0.0],
        [0.3, -0.1, -0.2, 0.0, 0.3, -0.3, 0.1, 1.0, 0.2, -0.1],
        [0.1, 0.2, 0.3, -0.1, 0.0, 0.1, -0.3, 0.2, 1.0, 0.3],
        [-0.2, 0.1, 0.1, 0.3, -0.1, 0.2, 0.0, -0.1, 0.3, 1.0]
    ])

    # 应用相关性
    factor_data = factor_data @ correlation_matrix.T

    panel_df = pd.DataFrame(factor_data, index=multi_index, columns=factors)

    # 保存面板数据
    panel_file = test_dir / "benchmark_panel.parquet"
    panel_df.to_parquet(panel_file)

    # 生成模拟价格数据（更真实的价格动态）
    price_data = []
    base_prices = np.random.uniform(50, 200, n_symbols)  # 基础价格

    for symbol_idx, symbol in enumerate(symbols):
        base_price = base_prices[symbol_idx]
        prices = [base_price]

        # 生成有趋势和波动的收益率
        for i in range(1, n_dates):
            # 基础收益率
            base_return = np.random.randn() * 0.015  # 1.5%日波动率

            # 添加趋势
            trend = 0.0002 * (i - n_dates/2) / n_dates  # 轻微趋势

            # 均值回归
            mean_reversion = -0.01 * (prices[-1] - base_price) / base_price

            total_return = base_return + trend + mean_reversion
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 1.0))  # 价格不能为负

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
        price_file = test_dir / f"{symbol}_daily_20230101_20230601.parquet"
        symbol_data[['date', 'close', 'trade_date']].to_parquet(price_file)

    # 生成模拟筛选结果（有意义的因子排序）
    factor_performance = []
    np.random.seed(42)

    for i, factor in enumerate(factors):
        # 模拟因子表现，前几个因子表现更好
        base_ic = 0.05 - i * 0.005  # IC递减
        base_sharpe = 1.5 - i * 0.1   # 夏普比率递减

        ic_mean = base_ic + np.random.randn() * 0.01
        ic_std = abs(0.15 + np.random.randn() * 0.05)
        sharpe_ratio = base_sharpe + np.random.randn() * 0.3

        factor_performance.append({
            'factor': factor,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'sharpe_ratio': sharpe_ratio,
            'ir': ic_mean / ic_std if ic_std > 0 else 0
        })

    screening_df = pd.DataFrame(factor_performance)
    screening_df = screening_df.sort_values('sharpe_ratio', ascending=False)

    # 保存筛选结果
    screening_file = test_dir / "benchmark_screening.csv"
    screening_df.to_csv(screening_file, index=False)

    print(f"基准测试数据生成完成:")
    print(f"  面板数据: {panel_file}")
    print(f"  价格数据: {test_dir}/ETF*_daily_*.parquet")
    print(f"  筛选结果: {screening_file}")

    return str(panel_file), str(test_dir), str(screening_file)


def run_single_benchmark(
    test_config: Dict,
    strategy_config: Dict,
    panel_path: str,
    price_dir: str,
    screening_path: str
) -> Dict[str, Any]:
    """运行单个基准测试"""
    test_name = f"{test_config['name']} - {strategy_config['name']}"
    print(f"\n🧪 运行测试: {test_name}")

    try:
        # 创建权重生成器配置
        weight_config = WeightGenerationConfig(
            strategy=strategy_config['weight_strategy'],
            max_combinations=test_config['max_combinations']
        )

        # 创建并行引擎
        engine = ParallelBacktestEngine(
            n_workers=strategy_config['n_workers'],
            chunk_size=strategy_config['chunk_size']
        )

        # 记录开始时间
        start_time = time.time()
        memory_start = get_memory_usage()

        # 运行回测
        results, config = engine.run_parallel_backtest(
            panel_path=panel_path,
            price_dir=price_dir,
            screening_csv=screening_path,
            output_dir=f"benchmark_results/{test_config['name']}/{strategy_config['name']}",
            top_k=test_config['factors'],
            top_n_list=test_config['top_n_list'],
            max_combinations=test_config['max_combinations']
        )

        # 记录结束时间
        end_time = time.time()
        memory_end = get_memory_usage()

        # 计算性能指标
        execution_time = end_time - start_time
        memory_used = memory_end - memory_start
        strategies_tested = len(results)
        speed = strategies_tested / execution_time

        # 获取最优策略性能
        best_strategy = results.iloc[0] if len(results) > 0 else None

        benchmark_result = {
            'test_name': test_name,
            'test_config': test_config,
            'strategy_config': strategy_config,
            'execution_time': execution_time,
            'memory_used_mb': memory_used,
            'strategies_tested': strategies_tested,
            'speed_per_second': speed,
            'best_sharpe_ratio': best_strategy['sharpe_ratio'] if best_strategy is not None else None,
            'best_return': best_strategy['total_return'] if best_strategy is not None else None,
            'success': True,
            'error': None
        }

        print(f"✅ 测试完成: {execution_time:.2f}秒, {speed:.1f}策略/秒")
        return benchmark_result

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return {
            'test_name': test_name,
            'test_config': test_config,
            'strategy_config': strategy_config,
            'execution_time': None,
            'memory_used_mb': None,
            'strategies_tested': 0,
            'speed_per_second': 0,
            'best_sharpe_ratio': None,
            'best_return': None,
            'success': False,
            'error': str(e)
        }


def get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def run_full_benchmark_suite():
    """运行完整的基准测试套件"""
    print("ETF轮动系统性能基准测试")
    print("=" * 80)

    # 生成测试数据
    panel_path, price_dir, screening_path = generate_benchmark_data()

    # 获取测试配置
    test_configs, strategy_configs = create_benchmark_suite()

    # 运行基准测试
    all_results = []

    for test_config in test_configs:
        for strategy_config in strategy_configs:
            result = run_single_benchmark(
                test_config, strategy_config,
                panel_path, price_dir, screening_path
            )
            all_results.append(result)

    return all_results


def analyze_benchmark_results(results: List[Dict]):
    """分析基准测试结果"""
    print("\n" + "=" * 80)
    print("基准测试结果分析")
    print("=" * 80)

    # 转换为DataFrame便于分析
    df = pd.DataFrame(results)

    # 基本统计
    total_tests = len(results)
    successful_tests = len(results[results['success']])
    print(f"总测试数: {total_tests}")
    print(f"成功测试数: {successful_tests}")
    print(f"成功率: {successful_tests/total_tests*100:.1f}%")

    if successful_tests == 0:
        print("❌ 没有成功的测试，无法进行分析")
        return

    # 按测试规模分析
    print(f"\n📊 按测试规模分析:")
    scale_analysis = df[df['success']].groupby('test_config.apply(lambda x: x["factors"])').agg({
        'execution_time': ['mean', 'std'],
        'speed_per_second': ['mean', 'std'],
        'memory_used_mb': ['mean']
    }).round(2)
    print(scale_analysis)

    # 按策略分析
    print(f"\n📊 按优化策略分析:")
    strategy_analysis = df[df['success']].groupby('strategy_config.apply(lambda x: x["name"])').agg({
        'execution_time': ['mean', 'std'],
        'speed_per_second': ['mean', 'std'],
        'best_sharpe_ratio': ['mean'],
        'success': 'count'
    }).round(2)
    print(strategy_analysis)

    # 性能对比
    print(f"\n🚀 性能对比:")
    successful_df = df[df['success']]

    # 找出最快的配置
    fastest = successful_df.loc[successful_df['speed_per_second'].idxmax()]
    print(f"最快配置: {fastest['test_name']}")
    print(f"  速度: {fastest['speed_per_second']:.1f}策略/秒")
    print(f"  时间: {fastest['execution_time']:.2f}秒")

    # 找出最优策略质量的配置
    best_quality = successful_df.loc[successful_df['best_sharpe_ratio'].idxmax()]
    print(f"最优策略质量: {best_quality['test_name']}")
    print(f"  最佳夏普比率: {best_quality['best_sharpe_ratio']:.3f}")
    print(f"  最佳收益率: {best_quality['best_return']:.2f}%")

    # 计算并行加速效果
    print(f"\n⚡ 并行加速效果:")
    serial_results = successful_df[successful_df['strategy_config.apply(lambda x: x["n_workers"])'] == 1]
    parallel_results = successful_df[successful_df['strategy_config.apply(lambda x: x["n_workers"])'] > 1]

    if len(serial_results) > 0 and len(parallel_results) > 0:
        # 找到相同的测试配置进行对比
        for _, serial_row in serial_results.iterrows():
            test_factors = serial_row['test_config']['factors']
            matching_parallel = parallel_results[
                parallel_results['test_config.apply(lambda x: x["factors"])'] == test_factors
            ]

            if len(matching_parallel) > 0:
                best_parallel = matching_parallel.loc[matching_parallel['speed_per_second'].idxmax()]
                speedup = best_parallel['speed_per_second'] / serial_row['speed_per_second']
                efficiency = speedup / best_parallel['strategy_config']['n_workers'] * 100

                print(f"  {test_factors}因子测试:")
                print(f"    串行: {serial_row['speed_per_second']:.1f}策略/秒")
                print(f"    并行: {best_parallel['speed_per_second']:.1f}策略/秒")
                print(f"    加速比: {speedup:.2f}x")
                print(f"    并行效率: {efficiency:.1f}%")

    return df


def create_performance_visualizations(df: pd.DataFrame):
    """创建性能可视化图表"""
    print("\n📈 生成性能可视化图表...")

    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 过滤成功的结果
    successful_df = df[df['success']].copy()

    if len(successful_df) == 0:
        print("❌ 没有成功的数据用于可视化")
        return

    # 创建图表目录
    viz_dir = Path("benchmark_results/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. 执行时间对比图
    plt.figure(figsize=(12, 8))

    # 提取因子数量和执行时间
    factor_counts = [r['test_config']['factors'] for r in successful_df['test_config']]
    strategy_names = [r['name'] for r in successful_df['strategy_config']]

    # 创建分组柱状图
    unique_factors = sorted(set(factor_counts))
    unique_strategies = list(set(strategy_names))

    x = np.arange(len(unique_factors))
    width = 0.8 / len(unique_strategies)

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, strategy in enumerate(unique_strategies):
        strategy_times = []
        for factors in unique_factors:
            mask = (successful_df['test_config.apply(lambda x: x["factors"])'] == factors) & \
                   (successful_df['strategy_config.apply(lambda x: x["name"])'] == strategy)
            strategy_data = successful_df[mask]
            if len(strategy_data) > 0:
                strategy_times.append(strategy_data['execution_time'].mean())
            else:
                strategy_times.append(0)

        ax.bar(x + i * width, strategy_times, width, label=strategy)

    ax.set_xlabel('因子数量')
    ax.set_ylabel('执行时间 (秒)')
    ax.set_title('不同策略的执行时间对比')
    ax.set_xticks(x + width * (len(unique_strategies) - 1) / 2)
    ax.set_xticklabels([f'{f}因子' for f in unique_factors])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 处理速度对比图
    plt.figure(figsize=(12, 8))

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, strategy in enumerate(unique_strategies):
        strategy_speeds = []
        for factors in unique_factors:
            mask = (successful_df['test_config.apply(lambda x: x["factors"])'] == factors) & \
                   (successful_df['strategy_config.apply(lambda x: x["name"])'] == strategy)
            strategy_data = successful_df[mask]
            if len(strategy_data) > 0:
                strategy_speeds.append(strategy_data['speed_per_second'].mean())
            else:
                strategy_speeds.append(0)

        ax.bar(x + i * width, strategy_speeds, width, label=strategy)

    ax.set_xlabel('因子数量')
    ax.set_ylabel('处理速度 (策略/秒)')
    ax.set_title('不同策略的处理速度对比')
    ax.set_xticks(x + width * (len(unique_strategies) - 1) / 2)
    ax.set_xticklabels([f'{f}因子' for f in unique_factors])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'processing_speed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 内存使用对比图
    plt.figure(figsize=(12, 8))

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, strategy in enumerate(unique_strategies):
        strategy_memory = []
        for factors in unique_factors:
            mask = (successful_df['test_config.apply(lambda x: x["factors"])'] == factors) & \
                   (successful_df['strategy_config.apply(lambda x: x["name"])'] == strategy)
            strategy_data = successful_df[mask]
            if len(strategy_data) > 0:
                strategy_memory.append(strategy_data['memory_used_mb'].mean())
            else:
                strategy_memory.append(0)

        ax.bar(x + i * width, strategy_memory, width, label=strategy)

    ax.set_xlabel('因子数量')
    ax.set_ylabel('内存使用 (MB)')
    ax.set_title('不同策略的内存使用对比')
    ax.set_xticks(x + width * (len(unique_strategies) - 1) / 2)
    ax.set_xticklabels([f'{f}因子' for f in unique_factors])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 散点图：速度 vs 质量 trade-off
    plt.figure(figsize=(12, 8))

    speeds = successful_df['speed_per_second']
    sharpe_ratios = successful_df['best_sharpe_ratio']
    colors = [r['strategy_config']['n_workers'] for r in successful_df['strategy_config']]

    scatter = plt.scatter(speeds, sharpe_ratios, c=colors, s=100, alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, label='工作进程数')

    plt.xlabel('处理速度 (策略/秒)')
    plt.ylabel('最优夏普比率')
    plt.title('速度与策略质量的权衡')
    plt.grid(True, alpha=0.3)

    # 添加标注
    for i, row in successful_df.iterrows():
        plt.annotate(row['test_name'].split(' - ')[1],
                    (row['speed_per_second'], row['best_sharpe_ratio']),
                    fontsize=8, alpha=0.7)

    plt.tight_layout()
    plt.savefig(viz_dir / 'speed_vs_quality_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 可视化图表已保存至: {viz_dir}")


def generate_benchmark_report(df: pd.DataFrame):
    """生成基准测试报告"""
    print("\n📝 生成基准测试报告...")

    successful_df = df[df['success']]

    if len(successful_df) == 0:
        print("❌ 没有成功的测试数据")
        return

    # 计算关键指标
    fastest_config = successful_df.loc[successful_df['speed_per_second'].idxmax()]
    best_quality_config = successful_df.loc[successful_df['best_sharpe_ratio'].idxmax()]

    # 计算平均性能
    avg_speed = successful_df['speed_per_second'].mean()
    avg_time = successful_df['execution_time'].mean()
    avg_memory = successful_df['memory_used_mb'].mean()

    # 计算并行加速比
    serial_configs = successful_df[successful_df['strategy_config.apply(lambda x: x["n_workers"])'] == 1]
    parallel_configs = successful_df[successful_df['strategy_config.apply(lambda x: x["n_workers"])'] > 1]

    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(df),
            "successful_tests": len(successful_df),
            "success_rate": len(successful_df) / len(df) * 100,
            "average_speed": avg_speed,
            "average_time": avg_time,
            "average_memory_mb": avg_memory
        },
        "best_performance": {
            "fastest": {
                "name": fastest_config['test_name'],
                "speed": fastest_config['speed_per_second'],
                "time": fastest_config['execution_time'],
                "memory_mb": fastest_config['memory_used_mb']
            },
            "best_quality": {
                "name": best_quality_config['test_name'],
                "sharpe_ratio": best_quality_config['best_sharpe_ratio'],
                "return": best_quality_config['best_return'],
                "speed": best_quality_config['speed_per_second']
            }
        },
        "parallel_analysis": {
            "serial_avg_speed": serial_configs['speed_per_second'].mean() if len(serial_configs) > 0 else 0,
            "parallel_avg_speed": parallel_configs['speed_per_second'].mean() if len(parallel_configs) > 0 else 0,
            "speedup": (parallel_configs['speed_per_second'].mean() / serial_configs['speed_per_second'].mean()
                      if len(serial_configs) > 0 and len(parallel_configs) > 0 else 0)
        },
        "recommendations": generate_recommendations(successful_df),
        "detailed_results": df.to_dict('records')
    }

    # 保存报告
    report_file = Path("benchmark_results/performance_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # 生成Markdown报告
    markdown_report = generate_markdown_report(report)
    markdown_file = Path("benchmark_results/PERFORMANCE_REPORT.md")

    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_report)

    print(f"✅ 报告已保存:")
    print(f"  JSON格式: {report_file}")
    print(f"  Markdown格式: {markdown_file}")

    return report


def generate_recommendations(df: pd.DataFrame) -> List[str]:
    """生成性能优化建议"""
    recommendations = []

    # 分析最优配置
    fastest = df.loc[df['speed_per_second'].idxmax()]
    best_quality = df.loc[df['best_sharpe_ratio'].idxmax()]

    # 基于结果生成建议
    recommendations.append("🚀 性能优化建议:")

    # 并行化建议
    parallel_configs = df[df['strategy_config.apply(lambda x: x["n_workers"])'] > 1]
    if len(parallel_configs) > 0:
        avg_parallel_speed = parallel_configs['speed_per_second'].mean()
        avg_speed = df['speed_per_second'].mean()
        if avg_parallel_speed > avg_speed * 2:
            recommendations.append("  • 强烈推荐使用并行计算，平均加速比超过2倍")

    # 策略选择建议
    strategy_performance = df.groupby('strategy_config.apply(lambda x: x["name"])')['speed_per_second'].mean()
    best_strategy = strategy_performance.idxmax()
    recommendations.append(f"  • 推荐使用 {best_strategy} 策略，平均速度最快")

    # 工作进程数建议
    worker_performance = df.groupby('strategy_config.apply(lambda x: x["n_workers"])')['speed_per_second'].mean()
    if len(worker_performance) > 1:
        best_workers = worker_performance.idxmax()
        recommendations.append(f"  • 推荐使用 {best_workers} 个工作进程")

    # 内存使用建议
    high_memory = df[df['memory_used_mb'] > 1000]
    if len(high_memory) > 0:
        recommendations.append("  • 注意内存使用，建议监控大规模回测的内存消耗")

    # 质量vs速度权衡建议
    if fastest['test_name'] != best_quality['test_name']:
        recommendations.append("  • 存在速度与质量的权衡，根据需求选择合适的配置")
        recommendations.append(f"    - 追求速度: {fastest['test_name']}")
        recommendations.append(f"    - 追求质量: {best_quality['test_name']}")

    return recommendations


def generate_markdown_report(report: Dict) -> str:
    """生成Markdown格式的报告"""
    md = []

    # 标题
    md.append("# ETF轮动系统性能基准测试报告")
    md.append("")
    md.append(f"**生成时间**: {report['timestamp']}")
    md.append("")

    # 执行摘要
    md.append("## 📊 执行摘要")
    md.append("")
    summary = report['summary']
    md.append(f"- **总测试数**: {summary['total_tests']}")
    md.append(f"- **成功率**: {summary['success_rate']:.1f}%")
    md.append(f"- **平均处理速度**: {summary['average_speed']:.1f} 策略/秒")
    md.append(f"- **平均执行时间**: {summary['average_time']:.2f} 秒")
    md.append(f"- **平均内存使用**: {summary['average_memory_mb']:.1f} MB")
    md.append("")

    # 最佳性能配置
    md.append("## 🏆 最佳性能配置")
    md.append("")

    fastest = report['best_performance']['fastest']
    md.append("### ⚡ 最快配置")
    md.append(f"- **配置**: {fastest['name']}")
    md.append(f"- **速度**: {fastest['speed']:.1f} 策略/秒")
    md.append(f"- **执行时间**: {fastest['time']:.2f} 秒")
    md.append(f"- **内存使用**: {fastest['memory_mb']:.1f} MB")
    md.append("")

    best_quality = report['best_performance']['best_quality']
    md.append("### 🎯 最优策略质量")
    md.append(f"- **配置**: {best_quality['name']}")
    md.append(f"- **夏普比率**: {best_quality['sharpe_ratio']:.3f}")
    md.append(f"- **收益率**: {best_quality['return']:.2f}%")
    md.append(f"- **速度**: {best_quality['speed']:.1f} 策略/秒")
    md.append("")

    # 并行分析
    md.append("## ⚡ 并行计算效果")
    md.append("")
    parallel = report['parallel_analysis']
    if parallel['speedup'] > 0:
        md.append(f"- **串行平均速度**: {parallel['serial_avg_speed']:.1f} 策略/秒")
        md.append(f"- **并行平均速度**: {parallel['parallel_avg_speed']:.1f} 策略/秒")
        md.append(f"- **平均加速比**: {parallel['speedup']:.2f}x")
    else:
        md.append("- 无法计算并行加速比（缺少对比数据）")
    md.append("")

    # 优化建议
    md.append("## 💡 性能优化建议")
    md.append("")
    for rec in report['recommendations']:
        md.append(rec)
    md.append("")

    # 结论
    md.append("## 📈 结论")
    md.append("")
    md.append("通过基准测试，我们验证了并行计算和智能权重生成策略的显著效果。")
    md.append("建议在生产环境中使用推荐的配置以获得最佳性能。")
    md.append("")

    return "\n".join(md)


def main():
    """主函数"""
    print("ETF轮动系统性能基准测试")
    print("=" * 80)

    try:
        # 运行基准测试
        results = run_full_benchmark_suite()

        # 分析结果
        df = analyze_benchmark_results(results)

        # 创建可视化
        create_performance_visualizations(df)

        # 生成报告
        report = generate_benchmark_report(df)

        print("\n" + "=" * 80)
        print("🎯 基准测试完成总结")
        print("=" * 80)

        if len(results) > 0 and any(r['success'] for r in results):
            summary = report['summary']
            fastest = report['best_performance']['fastest']
            best_quality = report['best_performance']['best_quality']

            print(f"✅ 成功完成 {summary['successful_tests']}/{summary['total_tests']} 个测试")
            print(f"📈 平均处理速度: {summary['average_speed']:.1f} 策略/秒")
            print(f"⚡ 最快配置: {fastest['name']} ({fastest['speed']:.1f} 策略/秒)")
            print(f"🎯 最优质量: {best_quality['name']} (夏普 {best_quality['sharpe_ratio']:.3f})")
            print(f"📊 详细报告: benchmark_results/PERFORMANCE_REPORT.md")

            # 显示主要建议
            print(f"\n💡 主要建议:")
            for rec in report['recommendations'][1:4]:  # 显示前3个建议
                print(f"  {rec}")
        else:
            print("❌ 所有测试都失败了，请检查配置和环境")

    except Exception as e:
        print(f"\n❌ 基准测试过程中出现错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()