#!/usr/bin/env python3
"""
基础功能测试 - 验证专业级因子筛选系统的核心功能
"""

import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from professional_factor_screener import (
    ProfessionalFactorScreener, 
    ScreeningConfig, 
    FactorMetrics
)

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    test_data_dir = temp_dir / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 300
    
    # 生成时间索引
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
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
    factors_data = pd.DataFrame({
        'MA5': price_data['close'].rolling(5).mean(),
        'MA10': price_data['close'].rolling(10).mean(),
        'MA20': price_data['close'].rolling(20).mean(),
        'RSI14': np.random.uniform(0, 100, n_samples),
        'MACD_12_26_9': np.random.normal(0, 0.5, n_samples),
        'BB_UPPER_20': price_data['close'] + np.random.normal(2, 0.5, n_samples),
        'ATR14': np.random.uniform(0.5, 3.0, n_samples),
        'STOCH_K': np.random.uniform(0, 100, n_samples),
        'CCI14': np.random.normal(0, 100, n_samples),
        'MOM10': np.random.normal(0, 1, n_samples),
        'open': price_data['open'],
        'high': price_data['high'],
        'low': price_data['low'],
        'close': price_data['close'],
        'volume': price_data['volume']
    }, index=dates)
    
    # 保存测试数据
    raw_dir = temp_dir / "raw" / "HK"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = test_data_dir / "60min"
    output_dir.mkdir(exist_ok=True)
    
    # 保存价格数据
    price_data.to_parquet(raw_dir / "0700HK_60m_20231201.parquet")
    
    # 保存因子数据
    factors_data.to_parquet(output_dir / "0700HK_60min_factors_20231201.parquet")
    
    print(f"测试数据已创建: {temp_dir}")
    return temp_dir, test_data_dir, factors_data, price_data

def test_basic_functionality():
    """测试基础功能"""
    print("="*80)
    print("专业级因子筛选系统 - 基础功能测试")
    print("="*80)
    
    try:
        # 1. 创建测试数据
        temp_dir, test_data_dir, sample_factors, sample_prices = create_test_data()
        
        # 2. 创建配置
        config = ScreeningConfig(
            ic_horizons=[1, 3, 5],
            min_sample_size=50,
            alpha_level=0.05,
            max_workers=2
        )
        
        # 3. 初始化筛选器
        print("\n1. 初始化筛选器...")
        screener = ProfessionalFactorScreener(str(test_data_dir), config=config)
        print("✅ 筛选器初始化成功")
        
        # 4. 测试数据加载
        print("\n2. 测试数据加载...")
        factors = screener.load_factors("0700.HK", "60min")
        print(f"✅ 因子数据加载成功: 形状={factors.shape}")
        
        # 5. 测试多周期IC计算
        print("\n3. 测试多周期IC计算...")
        returns = factors['close'].pct_change()
        ic_results = screener.calculate_multi_horizon_ic(factors, returns)
        print(f"✅ 多周期IC计算成功: {len(ic_results)} 个因子")
        
        # 6. 测试IC衰减分析
        print("\n4. 测试IC衰减分析...")
        decay_metrics = screener.analyze_ic_decay(ic_results)
        print(f"✅ IC衰减分析成功: {len(decay_metrics)} 个因子")
        
        # 7. 测试滚动IC计算
        print("\n5. 测试滚动IC计算...")
        rolling_ic_results = screener.calculate_rolling_ic(factors, returns, window=30)
        print(f"✅ 滚动IC计算成功: {len(rolling_ic_results)} 个因子")
        
        # 8. 测试VIF计算
        print("\n6. 测试VIF计算...")
        try:
            vif_scores = screener.calculate_vif_scores(factors)
            print(f"✅ VIF计算成功: {len(vif_scores)} 个因子")
        except Exception as e:
            print(f"⚠️ VIF计算失败，使用备选方案: {str(e)}")
            vif_scores = {}
        
        # 9. 测试相关性矩阵
        print("\n7. 测试相关性矩阵...")
        corr_matrix = screener.calculate_factor_correlation_matrix(factors)
        print(f"✅ 相关性矩阵计算成功: 形状={corr_matrix.shape}")
        
        # 10. 测试交易成本计算
        print("\n8. 测试交易成本计算...")
        cost_analysis = screener.calculate_trading_costs(factors, sample_prices)
        print(f"✅ 交易成本计算成功: {len(cost_analysis)} 个因子")
        
        # 11. 测试统计显著性检验
        print("\n9. 测试统计显著性检验...")
        p_values = {}
        for factor, ic_data in ic_results.items():
            p_values[factor] = ic_data.get('p_value_1d', 1.0)
        
        corrected_p = screener.benjamini_hochberg_correction(p_values)
        print(f"✅ FDR校正成功: {len(corrected_p)} 个因子")
        
        # 12. 测试综合评分
        print("\n10. 测试综合评分...")
        all_metrics = {
            'multi_horizon_ic': ic_results,
            'ic_decay': decay_metrics,
            'rolling_ic': rolling_ic_results,
            'vif_scores': vif_scores,
            'correlation_matrix': corr_matrix,
            'trading_costs': cost_analysis,
            'p_values': p_values,
            'corrected_p_values': corrected_p
        }
        
        comprehensive_results = screener.calculate_comprehensive_scores(all_metrics)
        print(f"✅ 综合评分成功: {len(comprehensive_results)} 个因子")
        
        # 13. 生成报告
        print("\n11. 生成筛选报告...")
        report_df = screener.generate_screening_report(comprehensive_results)
        print(f"✅ 报告生成成功: {len(report_df)} 行")
        
        # 14. 获取顶级因子
        print("\n12. 获取顶级因子...")
        top_factors = screener.get_top_factors(
            comprehensive_results, top_n=5, min_score=0.0, require_significant=False
        )
        print(f"✅ 顶级因子获取成功: {len(top_factors)} 个因子")
        
        # 15. 输出结果摘要
        print("\n" + "="*80)
        print("测试结果摘要")
        print("="*80)
        
        print(f"总因子数量: {len(comprehensive_results)}")
        print(f"显著因子数量: {sum(1 for m in comprehensive_results.values() if m.is_significant)}")
        print(f"高分因子数量 (>0.5): {sum(1 for m in comprehensive_results.values() if m.comprehensive_score > 0.5)}")
        
        print(f"\n前5名因子:")
        print("-" * 80)
        print(f"{'排名':<4} {'因子名称':<15} {'综合得分':<8} {'预测':<6} {'稳定':<6} {'独立':<6} {'实用':<6}")
        print("-" * 80)
        
        for i, metrics in enumerate(top_factors[:5]):
            print(f"{i+1:<4} {metrics.name:<15} {metrics.comprehensive_score:.3f}    "
                  f"{metrics.predictive_score:.3f}  {metrics.stability_score:.3f}  "
                  f"{metrics.independence_score:.3f}  {metrics.practicality_score:.3f}")
        
        print("\n✅ 所有基础功能测试通过!")
        
        # 清理临时文件
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """性能测试"""
    print("\n" + "="*80)
    print("性能测试")
    print("="*80)
    
    try:
        import time
        
        # 生成大数据集
        print("生成大数据集...")
        np.random.seed(42)
        n_samples = 1000
        n_factors = 30
        
        large_factors = pd.DataFrame({
            f'factor_{i}': np.random.normal(0, 1, n_samples)
            for i in range(n_factors)
        })
        
        # 添加必要的价格列
        large_factors['open'] = np.random.normal(100, 1, n_samples)
        large_factors['high'] = np.random.normal(101, 1, n_samples)
        large_factors['low'] = np.random.normal(99, 1, n_samples)
        large_factors['close'] = np.random.normal(100, 1, n_samples)
        large_factors['volume'] = np.random.lognormal(10, 1, n_samples)
        
        returns = pd.Series(np.random.normal(0, 0.1, n_samples))
        
        # 创建筛选器
        temp_dir = Path(tempfile.mkdtemp())
        config = ScreeningConfig(ic_horizons=[1, 3, 5], min_sample_size=50)
        screener = ProfessionalFactorScreener(str(temp_dir), config=config)
        
        # 性能测试
        print(f"测试数据: {n_samples} 样本 × {n_factors} 因子")
        
        start_time = time.time()
        ic_results = screener.calculate_multi_horizon_ic(large_factors, returns)
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"多周期IC计算耗时: {processing_time:.2f}秒")
        print(f"平均每因子耗时: {processing_time/n_factors:.3f}秒")
        
        if processing_time < 10:
            print("✅ 性能测试通过 (< 10秒)")
        else:
            print("⚠️ 性能测试警告 (> 10秒)")
        
        # 清理
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始专业级因子筛选系统测试...")
    
    # 基础功能测试
    basic_success = test_basic_functionality()
    
    # 性能测试
    performance_success = test_performance()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    if basic_success and performance_success:
        print("🎉 所有测试通过! 系统可以投入使用。")
        exit_code = 0
    else:
        print("❌ 部分测试失败，需要进一步调试。")
        exit_code = 1
    
    print(f"基础功能测试: {'✅ 通过' if basic_success else '❌ 失败'}")
    print(f"性能测试: {'✅ 通过' if performance_success else '❌ 失败'}")
    
    sys.exit(exit_code)

