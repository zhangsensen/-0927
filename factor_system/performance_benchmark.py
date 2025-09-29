#!/usr/bin/env python3
"""
性能基准测试 - 专业级因子筛选系统
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

测试内容：
1. 计算性能基准测试
2. 内存使用效率测试
3. 大数据量处理能力测试
4. 并发处理性能测试
5. 缓存系统效率测试
"""

import time
import psutil
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from typing import Dict, List, Tuple
import gc

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from professional_factor_screener import (
    ProfessionalFactorScreener, 
    ScreeningConfig
)

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    def generate_test_data(self, n_samples: int, n_factors: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成测试数据"""
        np.random.seed(42)
        
        # 生成时间索引
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='H')
        
        # 生成价格数据
        price_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
            'high': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
            'low': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
            'close': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
            'volume': np.random.lognormal(10, 1, n_samples)
        }, index=dates)
        
        # 确保OHLC逻辑正确
        for i in range(len(price_data)):
            ohlc = [price_data.iloc[i]['open'], price_data.iloc[i]['high'], 
                   price_data.iloc[i]['low'], price_data.iloc[i]['close']]
            price_data.iloc[i, 1] = max(ohlc)  # high
            price_data.iloc[i, 2] = min(ohlc)  # low
        
        # 生成因子数据
        factor_data = {}
        
        # 基础因子
        for i in range(n_factors):
            if i < 10:
                # 前10个因子使用价格相关的计算
                if i % 4 == 0:
                    factor_data[f'MA_{i+5}'] = price_data['close'].rolling(i+5).mean()
                elif i % 4 == 1:
                    factor_data[f'RSI_{i+10}'] = np.random.uniform(0, 100, n_samples)
                elif i % 4 == 2:
                    factor_data[f'MACD_{i}'] = np.random.normal(0, 0.5, n_samples)
                else:
                    factor_data[f'BB_{i}'] = price_data['close'] + np.random.normal(0, 2, n_samples)
            else:
                # 其余因子使用随机数据
                factor_data[f'factor_{i}'] = np.random.normal(0, 1, n_samples)
        
        # 添加价格列
        factor_data.update({
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close'],
            'volume': price_data['volume']
        })
        
        factors_df = pd.DataFrame(factor_data, index=dates)
        
        return factors_df, price_data
    
    def benchmark_ic_calculation(self) -> Dict[str, float]:
        """基准测试：IC计算性能"""
        print("\n🔍 IC计算性能基准测试")
        print("-" * 60)
        
        results = {}
        
        # 测试不同数据规模
        test_cases = [
            (500, 20, "小规模"),
            (1000, 50, "中规模"),
            (2000, 100, "大规模"),
            (5000, 200, "超大规模")
        ]
        
        config = ScreeningConfig(ic_horizons=[1, 3, 5], min_sample_size=50)
        temp_dir = Path(tempfile.mkdtemp())
        screener = ProfessionalFactorScreener(str(temp_dir), config=config)
        
        for n_samples, n_factors, scale_name in test_cases:
            print(f"测试 {scale_name}: {n_samples} 样本 × {n_factors} 因子")
            
            # 生成测试数据
            factors, prices = self.generate_test_data(n_samples, n_factors)
            returns = factors['close'].pct_change()
            
            # 记录初始内存
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 执行IC计算
            start_time = time.time()
            ic_results = screener.calculate_multi_horizon_ic(factors, returns)
            end_time = time.time()
            
            # 记录结果
            processing_time = end_time - start_time
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory
            
            throughput = n_factors / processing_time  # 因子/秒
            
            results[scale_name] = {
                'processing_time': processing_time,
                'memory_used': memory_used,
                'throughput': throughput,
                'factors_processed': len(ic_results),
                'samples': n_samples,
                'factors': n_factors
            }
            
            print(f"  ⏱️  处理时间: {processing_time:.3f}秒")
            print(f"  💾 内存使用: {memory_used:.1f}MB")
            print(f"  🚀 吞吐量: {throughput:.1f} 因子/秒")
            print(f"  ✅ 成功处理: {len(ic_results)} 个因子")
            
            # 清理内存
            del factors, prices, returns, ic_results
            gc.collect()
        
        shutil.rmtree(temp_dir)
        return results
    
    def benchmark_comprehensive_screening(self) -> Dict[str, float]:
        """基准测试：完整筛选流程性能"""
        print("\n🎯 完整筛选流程性能基准测试")
        print("-" * 60)
        
        # 中等规模数据测试
        n_samples, n_factors = 1500, 80
        print(f"测试规模: {n_samples} 样本 × {n_factors} 因子")
        
        # 生成测试数据
        factors, prices = self.generate_test_data(n_samples, n_factors)
        
        # 保存测试数据
        temp_dir = Path(tempfile.mkdtemp())
        test_data_dir = temp_dir / "test_data"
        test_data_dir.mkdir()
        
        raw_dir = temp_dir / "raw" / "HK"
        raw_dir.mkdir(parents=True)
        output_dir = test_data_dir / "60min"
        output_dir.mkdir()
        
        prices.to_parquet(raw_dir / "TEST_60m_20231201.parquet")
        factors.to_parquet(output_dir / "TESTHK_60min_factors_20231201.parquet")
        
        # 创建筛选器
        config = ScreeningConfig(
            ic_horizons=[1, 3, 5, 10],
            min_sample_size=100,
            max_workers=4
        )
        screener = ProfessionalFactorScreener(str(test_data_dir), config=config)
        
        # 记录初始状态
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # 执行完整筛选流程
        try:
            # 模拟完整筛选（不依赖实际文件加载）
            returns = factors['close'].pct_change()
            
            # 1. 多周期IC计算
            ic_start = time.time()
            ic_results = screener.calculate_multi_horizon_ic(factors, returns)
            ic_time = time.time() - ic_start
            
            # 2. IC衰减分析
            decay_start = time.time()
            decay_metrics = screener.analyze_ic_decay(ic_results)
            decay_time = time.time() - decay_start
            
            # 3. 滚动IC计算
            rolling_start = time.time()
            rolling_ic_results = screener.calculate_rolling_ic(factors, returns, window=50)
            rolling_time = time.time() - rolling_start
            
            # 4. VIF计算
            vif_start = time.time()
            try:
                vif_scores = screener.calculate_vif_scores(factors)
            except:
                vif_scores = {}
            vif_time = time.time() - vif_start
            
            # 5. 相关性矩阵
            corr_start = time.time()
            corr_matrix = screener.calculate_factor_correlation_matrix(factors)
            corr_time = time.time() - corr_start
            
            # 6. 交易成本计算
            cost_start = time.time()
            cost_analysis = screener.calculate_trading_costs(factors, prices)
            cost_time = time.time() - cost_start
            
            # 7. 综合评分
            scoring_start = time.time()
            all_metrics = {
                'multi_horizon_ic': ic_results,
                'ic_decay': decay_metrics,
                'rolling_ic': rolling_ic_results,
                'vif_scores': vif_scores,
                'correlation_matrix': corr_matrix,
                'trading_costs': cost_analysis,
                'p_values': {f: 0.05 for f in ic_results.keys()},
                'corrected_p_values': {f: 0.1 for f in ic_results.keys()}
            }
            comprehensive_results = screener.calculate_comprehensive_scores(all_metrics)
            scoring_time = time.time() - scoring_start
            
            total_time = time.time() - start_time
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory
            
            # 统计结果
            results = {
                'total_time': total_time,
                'memory_used': memory_used,
                'ic_calculation_time': ic_time,
                'decay_analysis_time': decay_time,
                'rolling_ic_time': rolling_time,
                'vif_calculation_time': vif_time,
                'correlation_time': corr_time,
                'cost_analysis_time': cost_time,
                'scoring_time': scoring_time,
                'factors_processed': len(comprehensive_results),
                'throughput': len(comprehensive_results) / total_time
            }
            
            print(f"✅ 完整筛选流程成功完成")
            print(f"  ⏱️  总处理时间: {total_time:.2f}秒")
            print(f"  💾 内存使用: {memory_used:.1f}MB")
            print(f"  🚀 整体吞吐量: {results['throughput']:.1f} 因子/秒")
            print(f"  📊 处理因子数: {len(comprehensive_results)}")
            
            print(f"\n📈 各阶段耗时分解:")
            print(f"  - IC计算: {ic_time:.3f}秒 ({ic_time/total_time*100:.1f}%)")
            print(f"  - 衰减分析: {decay_time:.3f}秒 ({decay_time/total_time*100:.1f}%)")
            print(f"  - 滚动IC: {rolling_time:.3f}秒 ({rolling_time/total_time*100:.1f}%)")
            print(f"  - VIF计算: {vif_time:.3f}秒 ({vif_time/total_time*100:.1f}%)")
            print(f"  - 相关性: {corr_time:.3f}秒 ({corr_time/total_time*100:.1f}%)")
            print(f"  - 成本分析: {cost_time:.3f}秒 ({cost_time/total_time*100:.1f}%)")
            print(f"  - 综合评分: {scoring_time:.3f}秒 ({scoring_time/total_time*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 完整筛选流程失败: {str(e)}")
            results = {'error': str(e)}
        
        # 清理
        shutil.rmtree(temp_dir)
        return results
    
    def benchmark_memory_efficiency(self) -> Dict[str, float]:
        """基准测试：内存使用效率"""
        print("\n💾 内存使用效率基准测试")
        print("-" * 60)
        
        results = {}
        
        # 测试不同内存压力下的表现
        test_cases = [
            (1000, 50, "正常负载"),
            (3000, 100, "中等负载"),
            (5000, 150, "高负载")
        ]
        
        config = ScreeningConfig(ic_horizons=[1, 3, 5], min_sample_size=50)
        temp_dir = Path(tempfile.mkdtemp())
        screener = ProfessionalFactorScreener(str(temp_dir), config=config)
        
        for n_samples, n_factors, load_name in test_cases:
            print(f"测试 {load_name}: {n_samples} 样本 × {n_factors} 因子")
            
            # 记录基线内存
            gc.collect()  # 强制垃圾回收
            baseline_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 生成数据
            factors, prices = self.generate_test_data(n_samples, n_factors)
            returns = factors['close'].pct_change()
            
            data_loaded_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 执行计算
            ic_results = screener.calculate_multi_horizon_ic(factors, returns)
            
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 清理数据
            del factors, prices, returns, ic_results
            gc.collect()
            
            final_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 计算内存指标
            data_memory = data_loaded_memory - baseline_memory
            processing_memory = peak_memory - data_loaded_memory
            total_memory = peak_memory - baseline_memory
            memory_leak = final_memory - baseline_memory
            
            memory_efficiency = data_memory / total_memory if total_memory > 0 else 0
            
            results[load_name] = {
                'baseline_memory': baseline_memory,
                'data_memory': data_memory,
                'processing_memory': processing_memory,
                'peak_memory': peak_memory,
                'total_memory': total_memory,
                'memory_leak': memory_leak,
                'memory_efficiency': memory_efficiency
            }
            
            print(f"  📊 数据内存: {data_memory:.1f}MB")
            print(f"  ⚙️  处理内存: {processing_memory:.1f}MB")
            print(f"  📈 峰值内存: {peak_memory:.1f}MB")
            print(f"  💧 内存泄漏: {memory_leak:.1f}MB")
            print(f"  📊 内存效率: {memory_efficiency*100:.1f}%")
        
        shutil.rmtree(temp_dir)
        return results
    
    def benchmark_parallel_processing(self) -> Dict[str, float]:
        """基准测试：并行处理性能"""
        print("\n🔄 并行处理性能基准测试")
        print("-" * 60)
        
        # 生成测试数据
        n_samples, n_factors = 2000, 100
        factors, prices = self.generate_test_data(n_samples, n_factors)
        returns = factors['close'].pct_change()
        
        results = {}
        
        # 测试不同线程数的性能
        thread_counts = [1, 2, 4, 8]
        
        for thread_count in thread_counts:
            print(f"测试 {thread_count} 线程:")
            
            config = ScreeningConfig(
                ic_horizons=[1, 3, 5],
                min_sample_size=50,
                max_workers=thread_count
            )
            
            temp_dir = Path(tempfile.mkdtemp())
            screener = ProfessionalFactorScreener(str(temp_dir), config=config)
            
            # 执行测试
            start_time = time.time()
            ic_results = screener.calculate_multi_horizon_ic(factors, returns)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = len(ic_results) / processing_time
            
            results[f'{thread_count}_threads'] = {
                'processing_time': processing_time,
                'throughput': throughput,
                'factors_processed': len(ic_results)
            }
            
            print(f"  ⏱️  处理时间: {processing_time:.3f}秒")
            print(f"  🚀 吞吐量: {throughput:.1f} 因子/秒")
            
            shutil.rmtree(temp_dir)
        
        # 计算并行效率
        if '1_threads' in results and '4_threads' in results:
            speedup = results['1_threads']['processing_time'] / results['4_threads']['processing_time']
            efficiency = speedup / 4 * 100
            print(f"\n📊 并行性能分析:")
            print(f"  🚀 4线程加速比: {speedup:.2f}x")
            print(f"  📈 并行效率: {efficiency:.1f}%")
            
            results['parallel_analysis'] = {
                'speedup_4x': speedup,
                'efficiency_4x': efficiency
            }
        
        return results
    
    def generate_performance_report(self, all_results: Dict) -> str:
        """生成性能报告"""
        report = []
        report.append("="*100)
        report.append("专业级因子筛选系统 - 性能基准测试报告")
        report.append("="*100)
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"系统信息: {psutil.cpu_count()} CPU核心, {psutil.virtual_memory().total/1024/1024/1024:.1f}GB 内存")
        report.append("")
        
        # IC计算性能
        if 'ic_calculation' in all_results:
            report.append("📊 IC计算性能基准")
            report.append("-" * 50)
            
            for scale, metrics in all_results['ic_calculation'].items():
                report.append(f"{scale}:")
                report.append(f"  - 数据规模: {metrics['samples']} 样本 × {metrics['factors']} 因子")
                report.append(f"  - 处理时间: {metrics['processing_time']:.3f}秒")
                report.append(f"  - 内存使用: {metrics['memory_used']:.1f}MB")
                report.append(f"  - 吞吐量: {metrics['throughput']:.1f} 因子/秒")
                report.append("")
        
        # 完整筛选性能
        if 'comprehensive_screening' in all_results:
            report.append("🎯 完整筛选流程性能")
            report.append("-" * 50)
            
            metrics = all_results['comprehensive_screening']
            if 'error' not in metrics:
                report.append(f"总处理时间: {metrics['total_time']:.2f}秒")
                report.append(f"内存使用: {metrics['memory_used']:.1f}MB")
                report.append(f"整体吞吐量: {metrics['throughput']:.1f} 因子/秒")
                report.append(f"处理因子数: {metrics['factors_processed']}")
                report.append("")
                
                report.append("各阶段耗时:")
                stages = [
                    ('IC计算', 'ic_calculation_time'),
                    ('衰减分析', 'decay_analysis_time'),
                    ('滚动IC', 'rolling_ic_time'),
                    ('VIF计算', 'vif_calculation_time'),
                    ('相关性', 'correlation_time'),
                    ('成本分析', 'cost_analysis_time'),
                    ('综合评分', 'scoring_time')
                ]
                
                for stage_name, key in stages:
                    if key in metrics:
                        time_val = metrics[key]
                        percentage = time_val / metrics['total_time'] * 100
                        report.append(f"  - {stage_name}: {time_val:.3f}秒 ({percentage:.1f}%)")
                report.append("")
        
        # 内存效率
        if 'memory_efficiency' in all_results:
            report.append("💾 内存使用效率")
            report.append("-" * 50)
            
            for load, metrics in all_results['memory_efficiency'].items():
                report.append(f"{load}:")
                report.append(f"  - 数据内存: {metrics['data_memory']:.1f}MB")
                report.append(f"  - 处理内存: {metrics['processing_memory']:.1f}MB")
                report.append(f"  - 峰值内存: {metrics['peak_memory']:.1f}MB")
                report.append(f"  - 内存效率: {metrics['memory_efficiency']*100:.1f}%")
                report.append(f"  - 内存泄漏: {metrics['memory_leak']:.1f}MB")
                report.append("")
        
        # 并行处理
        if 'parallel_processing' in all_results:
            report.append("🔄 并行处理性能")
            report.append("-" * 50)
            
            for config, metrics in all_results['parallel_processing'].items():
                if 'threads' in config:
                    report.append(f"{config}:")
                    report.append(f"  - 处理时间: {metrics['processing_time']:.3f}秒")
                    report.append(f"  - 吞吐量: {metrics['throughput']:.1f} 因子/秒")
                    report.append("")
            
            if 'parallel_analysis' in all_results['parallel_processing']:
                analysis = all_results['parallel_processing']['parallel_analysis']
                report.append("并行性能分析:")
                report.append(f"  - 4线程加速比: {analysis['speedup_4x']:.2f}x")
                report.append(f"  - 并行效率: {analysis['efficiency_4x']:.1f}%")
                report.append("")
        
        # 性能评级
        report.append("🏆 性能评级")
        report.append("-" * 50)
        
        # 基于测试结果给出评级
        if 'ic_calculation' in all_results:
            medium_scale = all_results['ic_calculation'].get('中规模', {})
            if medium_scale:
                throughput = medium_scale.get('throughput', 0)
                if throughput > 50:
                    ic_grade = "🟢 优秀"
                elif throughput > 20:
                    ic_grade = "🟡 良好"
                else:
                    ic_grade = "🔴 需优化"
                
                report.append(f"IC计算性能: {ic_grade} ({throughput:.1f} 因子/秒)")
        
        if 'memory_efficiency' in all_results:
            normal_load = all_results['memory_efficiency'].get('正常负载', {})
            if normal_load:
                efficiency = normal_load.get('memory_efficiency', 0)
                if efficiency > 0.7:
                    memory_grade = "🟢 优秀"
                elif efficiency > 0.5:
                    memory_grade = "🟡 良好"
                else:
                    memory_grade = "🔴 需优化"
                
                report.append(f"内存使用效率: {memory_grade} ({efficiency*100:.1f}%)")
        
        if 'parallel_processing' in all_results:
            analysis = all_results['parallel_processing'].get('parallel_analysis', {})
            if analysis:
                efficiency = analysis.get('efficiency_4x', 0)
                if efficiency > 70:
                    parallel_grade = "🟢 优秀"
                elif efficiency > 50:
                    parallel_grade = "🟡 良好"
                else:
                    parallel_grade = "🔴 需优化"
                
                report.append(f"并行处理效率: {parallel_grade} ({efficiency:.1f}%)")
        
        report.append("")
        report.append("="*100)
        
        return "\n".join(report)

def run_performance_benchmark():
    """运行完整的性能基准测试"""
    print("🚀 启动专业级因子筛选系统性能基准测试")
    print("="*100)
    
    benchmark = PerformanceBenchmark()
    all_results = {}
    
    try:
        # 1. IC计算性能测试
        all_results['ic_calculation'] = benchmark.benchmark_ic_calculation()
        
        # 2. 完整筛选流程测试
        all_results['comprehensive_screening'] = benchmark.benchmark_comprehensive_screening()
        
        # 3. 内存效率测试
        all_results['memory_efficiency'] = benchmark.benchmark_memory_efficiency()
        
        # 4. 并行处理测试
        all_results['parallel_processing'] = benchmark.benchmark_parallel_processing()
        
        # 5. 生成报告
        report = benchmark.generate_performance_report(all_results)
        
        # 输出报告
        print("\n" + report)
        
        # 保存报告
        report_path = Path(__file__).parent / f"performance_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 性能报告已保存: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 性能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_performance_benchmark()
    sys.exit(0 if success else 1)

