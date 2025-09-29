#!/usr/bin/env python3
"""
专业级因子筛选系统测试套件
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

测试覆盖：
1. 单元测试：各个功能模块的独立测试
2. 集成测试：完整流程测试
3. 性能测试：大数据量和内存使用测试
4. 边界测试：异常情况和边界条件测试
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time
import psutil
import os
import sys
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from professional_factor_screener import (
    ProfessionalFactorScreener, 
    ScreeningConfig, 
    FactorMetrics
)

class TestProfessionalFactorScreener(unittest.TestCase):
    """专业级因子筛选器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_data_dir = cls.temp_dir / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)
        
        # 创建测试配置
        cls.test_config = ScreeningConfig(
            ic_horizons=[1, 3, 5],
            min_sample_size=50,
            alpha_level=0.05,
            max_workers=2
        )
        
        # 生成测试数据
        cls._generate_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """测试类清理"""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _generate_test_data(cls):
        """生成测试数据"""
        np.random.seed(42)
        n_samples = 500
        
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
            'BB_LOWER_20': price_data['close'] - np.random.normal(2, 0.5, n_samples),
            'ATR14': np.random.uniform(0.5, 3.0, n_samples),
            'STOCH_K': np.random.uniform(0, 100, n_samples),
            'STOCH_D': np.random.uniform(0, 100, n_samples),
            'CCI14': np.random.normal(0, 100, n_samples),
            'WILLR14': np.random.uniform(-100, 0, n_samples),
            'MOM10': np.random.normal(0, 1, n_samples),
            'ROC10': np.random.normal(0, 0.1, n_samples),
            'TRIX': np.random.normal(0, 0.01, n_samples),
            'open': price_data['open'],
            'high': price_data['high'],
            'low': price_data['low'],
            'close': price_data['close'],
            'volume': price_data['volume']
        }, index=dates)
        
        # 保存测试数据
        raw_dir = cls.temp_dir / "raw" / "HK"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        output_dir = cls.test_data_dir / "60min"
        output_dir.mkdir(exist_ok=True)
        
        # 保存价格数据
        price_data.to_parquet(raw_dir / "0700HK_60m_20231201.parquet")
        
        # 保存因子数据
        factors_data.to_parquet(output_dir / "0700HK_60min_factors_20231201.parquet")
        
        cls.sample_factors = factors_data
        cls.sample_prices = price_data

class TestDataLoading(TestProfessionalFactorScreener):
    """数据加载测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
    
    def test_load_factors_success(self):
        """测试因子数据加载成功"""
        factors = self.screener.load_factors("0700.HK", "60min")
        
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertGreater(len(factors), 0)
        self.assertGreater(len(factors.columns), 5)
        self.assertIsInstance(factors.index, pd.DatetimeIndex)
    
    def test_load_price_data_success(self):
        """测试价格数据加载成功"""
        # 设置原始数据路径
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value = self.temp_dir / "raw" / "HK"
            
            prices = self.screener.load_price_data("0700.HK")
            
            self.assertIsInstance(prices, pd.DataFrame)
            self.assertGreater(len(prices), 0)
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                self.assertIn(col, prices.columns)
    
    def test_load_nonexistent_data(self):
        """测试加载不存在的数据"""
        with self.assertRaises(FileNotFoundError):
            self.screener.load_factors("NONEXISTENT", "60min")

class TestPredictivePowerAnalysis(TestProfessionalFactorScreener):
    """预测能力分析测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        # 准备测试数据
        self.factors = self.sample_factors.copy()
        self.returns = self.sample_factors['close'].pct_change()
    
    def test_multi_horizon_ic_calculation(self):
        """测试多周期IC计算"""
        ic_results = self.screener.calculate_multi_horizon_ic(
            self.factors, self.returns
        )
        
        self.assertIsInstance(ic_results, dict)
        self.assertGreater(len(ic_results), 0)
        
        # 检查结果结构
        for factor, metrics in ic_results.items():
            self.assertIsInstance(metrics, dict)
            
            # 检查必要的键
            for horizon in self.test_config.ic_horizons:
                ic_key = f'ic_{horizon}d'
                p_key = f'p_value_{horizon}d'
                size_key = f'sample_size_{horizon}d'
                
                if ic_key in metrics:
                    self.assertIsInstance(metrics[ic_key], (int, float))
                    self.assertGreaterEqual(metrics[ic_key], -1)
                    self.assertLessEqual(metrics[ic_key], 1)
                    
                    self.assertIn(p_key, metrics)
                    self.assertIn(size_key, metrics)
    
    def test_ic_decay_analysis(self):
        """测试IC衰减分析"""
        ic_results = self.screener.calculate_multi_horizon_ic(
            self.factors, self.returns
        )
        decay_metrics = self.screener.analyze_ic_decay(ic_results)
        
        self.assertIsInstance(decay_metrics, dict)
        
        for factor, metrics in decay_metrics.items():
            self.assertIn('decay_rate', metrics)
            self.assertIn('ic_stability', metrics)
            self.assertIn('max_ic', metrics)
            self.assertIn('ic_longevity', metrics)
            
            # 检查数值范围
            self.assertGreaterEqual(metrics['ic_stability'], 0)
            self.assertLessEqual(metrics['ic_stability'], 1)
            self.assertIsInstance(metrics['ic_longevity'], int)

class TestStabilityAnalysis(TestProfessionalFactorScreener):
    """稳定性分析测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        self.factors = self.sample_factors.copy()
        self.returns = self.sample_factors['close'].pct_change()
    
    def test_rolling_ic_calculation(self):
        """测试滚动IC计算"""
        rolling_ic_results = self.screener.calculate_rolling_ic(
            self.factors, self.returns, window=30
        )
        
        self.assertIsInstance(rolling_ic_results, dict)
        
        for factor, metrics in rolling_ic_results.items():
            self.assertIn('rolling_ic_mean', metrics)
            self.assertIn('rolling_ic_std', metrics)
            self.assertIn('rolling_ic_stability', metrics)
            self.assertIn('ic_consistency', metrics)
            
            # 检查稳定性指标范围
            self.assertGreaterEqual(metrics['rolling_ic_stability'], 0)
            self.assertLessEqual(metrics['rolling_ic_stability'], 1)
            self.assertGreaterEqual(metrics['ic_consistency'], 0)
            self.assertLessEqual(metrics['ic_consistency'], 1)
    
    def test_cross_sectional_stability(self):
        """测试截面稳定性计算"""
        stability_results = self.screener.calculate_cross_sectional_stability(
            self.factors
        )
        
        self.assertIsInstance(stability_results, dict)
        
        for factor, metrics in stability_results.items():
            self.assertIn('cross_section_stability', metrics)
            self.assertIn('cross_section_cv', metrics)
            
            self.assertGreaterEqual(metrics['cross_section_stability'], 0)
            self.assertLessEqual(metrics['cross_section_stability'], 1)

class TestIndependenceAnalysis(TestProfessionalFactorScreener):
    """独立性分析测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        # 创建具有已知相关性的因子数据
        np.random.seed(42)
        n_samples = 200
        
        base_factor = np.random.normal(0, 1, n_samples)
        self.factors = pd.DataFrame({
            'factor_1': base_factor,
            'factor_2': base_factor + np.random.normal(0, 0.1, n_samples),  # 高相关
            'factor_3': np.random.normal(0, 1, n_samples),  # 低相关
            'factor_4': base_factor * 0.5 + np.random.normal(0, 0.5, n_samples),  # 中等相关
            'open': np.random.normal(100, 1, n_samples),
            'high': np.random.normal(101, 1, n_samples),
            'low': np.random.normal(99, 1, n_samples),
            'close': np.random.normal(100, 1, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        self.returns = pd.Series(np.random.normal(0, 0.1, n_samples))
    
    def test_vif_calculation(self):
        """测试VIF计算"""
        vif_scores = self.screener.calculate_vif_scores(self.factors)
        
        self.assertIsInstance(vif_scores, dict)
        self.assertGreater(len(vif_scores), 0)
        
        for factor, vif in vif_scores.items():
            self.assertIsInstance(vif, (int, float))
            self.assertGreater(vif, 0)
    
    def test_correlation_matrix_calculation(self):
        """测试相关性矩阵计算"""
        corr_matrix = self.screener.calculate_factor_correlation_matrix(
            self.factors
        )
        
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        
        if not corr_matrix.empty:
            # 检查矩阵是对称的
            np.testing.assert_array_almost_equal(
                corr_matrix.values, corr_matrix.T.values, decimal=10
            )
            
            # 检查对角线为1
            np.testing.assert_array_almost_equal(
                np.diag(corr_matrix), 1.0, decimal=10
            )
    
    def test_information_increment_calculation(self):
        """测试信息增量计算"""
        base_factors = ['factor_1']
        info_increment = self.screener.calculate_information_increment(
            self.factors, self.returns, base_factors
        )
        
        self.assertIsInstance(info_increment, dict)
        
        for factor, increment in info_increment.items():
            self.assertIsInstance(increment, (int, float))

class TestPracticalityAnalysis(TestProfessionalFactorScreener):
    """实用性分析测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        self.factors = self.sample_factors.copy()
        self.prices = self.sample_prices.copy()
    
    def test_trading_costs_calculation(self):
        """测试交易成本计算"""
        cost_analysis = self.screener.calculate_trading_costs(
            self.factors, self.prices
        )
        
        self.assertIsInstance(cost_analysis, dict)
        
        for factor, metrics in cost_analysis.items():
            self.assertIn('turnover_rate', metrics)
            self.assertIn('total_cost', metrics)
            self.assertIn('cost_efficiency', metrics)
            
            # 检查数值合理性
            self.assertGreaterEqual(metrics['turnover_rate'], 0)
            self.assertGreaterEqual(metrics['total_cost'], 0)
            self.assertGreaterEqual(metrics['cost_efficiency'], 0)
            self.assertLessEqual(metrics['cost_efficiency'], 1)
    
    def test_liquidity_requirements_calculation(self):
        """测试流动性需求计算"""
        liquidity_analysis = self.screener.calculate_liquidity_requirements(
            self.factors, self.prices['volume']
        )
        
        self.assertIsInstance(liquidity_analysis, dict)
        
        for factor, metrics in liquidity_analysis.items():
            self.assertIn('liquidity_demand', metrics)
            self.assertIn('liquidity_score', metrics)
            self.assertIn('capacity_score', metrics)
            
            self.assertGreaterEqual(metrics['liquidity_score'], 0)
            self.assertLessEqual(metrics['liquidity_score'], 1)

class TestAdaptabilityAnalysis(TestProfessionalFactorScreener):
    """适应性分析测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        self.factors = self.sample_factors.copy()
        self.returns = self.sample_factors['close'].pct_change()
    
    def test_reversal_effects_detection(self):
        """测试反转效应检测"""
        reversal_effects = self.screener.detect_reversal_effects(
            self.factors, self.returns
        )
        
        self.assertIsInstance(reversal_effects, dict)
        
        for factor, metrics in reversal_effects.items():
            self.assertIn('reversal_effect', metrics)
            self.assertIn('reversal_strength', metrics)
            self.assertIn('reversal_consistency', metrics)
            
            self.assertIsInstance(metrics['reversal_effect'], (int, float))
            self.assertGreaterEqual(metrics['reversal_strength'], 0)
    
    def test_momentum_persistence_analysis(self):
        """测试动量持续性分析"""
        momentum_analysis = self.screener.analyze_momentum_persistence(
            self.factors, self.returns
        )
        
        self.assertIsInstance(momentum_analysis, dict)
        
        for factor, metrics in momentum_analysis.items():
            self.assertIn('momentum_persistence', metrics)
            self.assertIn('momentum_consistency', metrics)
            
            self.assertGreaterEqual(metrics['momentum_persistence'], -1)
            self.assertLessEqual(metrics['momentum_persistence'], 1)
            self.assertGreaterEqual(metrics['momentum_consistency'], 0)
            self.assertLessEqual(metrics['momentum_consistency'], 1)

class TestStatisticalTesting(TestProfessionalFactorScreener):
    """统计检验测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        # 创建测试p值
        self.p_values = {
            'factor_1': 0.001,
            'factor_2': 0.01,
            'factor_3': 0.05,
            'factor_4': 0.1,
            'factor_5': 0.5
        }
    
    def test_benjamini_hochberg_correction(self):
        """测试Benjamini-Hochberg校正"""
        corrected_p = self.screener.benjamini_hochberg_correction(
            self.p_values, alpha=0.05
        )
        
        self.assertIsInstance(corrected_p, dict)
        self.assertEqual(len(corrected_p), len(self.p_values))
        
        # 校正后的p值应该不小于原始p值
        for factor in self.p_values:
            self.assertGreaterEqual(corrected_p[factor], self.p_values[factor])
            self.assertLessEqual(corrected_p[factor], 1.0)
    
    def test_bonferroni_correction(self):
        """测试Bonferroni校正"""
        corrected_p = self.screener.bonferroni_correction(self.p_values)
        
        self.assertIsInstance(corrected_p, dict)
        self.assertEqual(len(corrected_p), len(self.p_values))
        
        # Bonferroni校正应该更保守
        for factor in self.p_values:
            expected = min(self.p_values[factor] * len(self.p_values), 1.0)
            self.assertAlmostEqual(corrected_p[factor], expected)

class TestComprehensiveScoring(TestProfessionalFactorScreener):
    """综合评分测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
        
        # 创建模拟的所有指标数据
        self.all_metrics = {
            'multi_horizon_ic': {
                'test_factor': {
                    'ic_1d': 0.05,
                    'ic_3d': 0.04,
                    'ic_5d': 0.03
                }
            },
            'rolling_ic': {
                'test_factor': {
                    'rolling_ic_stability': 0.8,
                    'ic_consistency': 0.7
                }
            },
            'vif_scores': {
                'test_factor': 2.5
            },
            'p_values': {
                'test_factor': 0.01
            },
            'corrected_p_values': {
                'test_factor': 0.02
            }
        }
    
    def test_comprehensive_scoring(self):
        """测试综合评分计算"""
        results = self.screener.calculate_comprehensive_scores(
            self.all_metrics
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('test_factor', results)
        
        metrics = results['test_factor']
        self.assertIsInstance(metrics, FactorMetrics)
        
        # 检查评分范围
        self.assertGreaterEqual(metrics.comprehensive_score, 0)
        self.assertLessEqual(metrics.comprehensive_score, 1)
        
        # 检查各维度评分
        self.assertGreaterEqual(metrics.predictive_score, 0)
        self.assertGreaterEqual(metrics.stability_score, 0)
        self.assertGreaterEqual(metrics.independence_score, 0)

class TestIntegrationWorkflow(TestProfessionalFactorScreener):
    """集成测试 - 完整工作流程"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
    
    def test_complete_screening_workflow(self):
        """测试完整筛选流程"""
        try:
            # 执行完整筛选
            results = self.screener.screen_factors_comprehensive(
                "0700.HK", "60min"
            )
            
            # 验证结果
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # 验证结果结构
            for factor_name, metrics in results.items():
                self.assertIsInstance(metrics, FactorMetrics)
                self.assertEqual(metrics.name, factor_name)
                
                # 验证评分范围
                self.assertGreaterEqual(metrics.comprehensive_score, 0)
                self.assertLessEqual(metrics.comprehensive_score, 1)
            
            # 生成报告 - 指定临时输出路径避免Path._flavour问题
            temp_report_path = self.temp_dir / "test_report.csv"
            report_df = self.screener.generate_screening_report(results, output_path=str(temp_report_path))
            self.assertIsInstance(report_df, pd.DataFrame)
            self.assertGreater(len(report_df), 0)
            
            # 验证报告文件是否生成
            self.assertTrue(temp_report_path.exists())
            
            # 获取顶级因子
            top_factors = self.screener.get_top_factors(
                results, top_n=5, min_score=0.0, require_significant=False
            )
            self.assertIsInstance(top_factors, list)
            self.assertLessEqual(len(top_factors), 5)
            
        except Exception as e:
            self.fail(f"完整筛选流程失败: {str(e)}")

class TestPerformance(TestProfessionalFactorScreener):
    """性能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        # 生成大数据集
        np.random.seed(42)
        n_samples = 2000
        n_factors = 50
        
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
        
        # 测试性能
        start_time = time.time()
        
        ic_results = self.screener.calculate_multi_horizon_ic(
            large_factors, returns
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证处理时间合理 (应该在10秒内)
        self.assertLess(processing_time, 10)
        
        # 验证结果完整性
        self.assertEqual(len(ic_results), n_factors)
    
    def test_memory_usage(self):
        """测试内存使用"""
        process = psutil.Process()
        
        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 处理数据
        factors = self.sample_factors.copy()
        returns = self.sample_factors['close'].pct_change()
        
        self.screener.calculate_multi_horizon_ic(factors, returns)
        
        # 记录峰值内存
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 验证内存增长合理 (应该小于100MB)
        memory_increase = peak_memory - initial_memory
        self.assertLess(memory_increase, 100)

class TestEdgeCases(TestProfessionalFactorScreener):
    """边界条件测试"""
    
    def setUp(self):
        """测试初始化"""
        self.screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=self.test_config
        )
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        empty_factors = pd.DataFrame()
        empty_returns = pd.Series(dtype=float)
        
        # 应该返回空结果而不是崩溃
        ic_results = self.screener.calculate_multi_horizon_ic(
            empty_factors, empty_returns
        )
        self.assertEqual(len(ic_results), 0)
    
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        factors_with_na = self.sample_factors.copy()
        
        # 随机设置一些缺失值 - 修复pandas 2.3.2+兼容性
        factors_with_na.iloc[10:21, factors_with_na.columns.get_loc('MA5')] = np.nan
        factors_with_na.iloc[30:41, factors_with_na.columns.get_loc('RSI14')] = np.nan
        
        returns = self.sample_factors['close'].pct_change()
        
        # 应该能正常处理缺失值
        ic_results = self.screener.calculate_multi_horizon_ic(
            factors_with_na, returns
        )
        
        self.assertIsInstance(ic_results, dict)
    
    def test_small_sample_handling(self):
        """测试小样本处理"""
        small_factors = self.sample_factors.iloc[:30].copy()  # 只有30个样本
        small_returns = self.sample_factors['close'].iloc[:30].pct_change()
        
        # 应该能处理小样本或返回空结果
        ic_results = self.screener.calculate_multi_horizon_ic(
            small_factors, small_returns
        )
        
        self.assertIsInstance(ic_results, dict)
    
    def test_extreme_values_handling(self):
        """测试极值处理"""
        factors_with_extremes = self.sample_factors.copy()
        
        # 添加极值
        factors_with_extremes.loc[0, 'MA5'] = 1e6
        factors_with_extremes.loc[1, 'MA5'] = -1e6
        factors_with_extremes.loc[2, 'RSI14'] = 1e6
        
        returns = self.sample_factors['close'].pct_change()
        
        # 应该能处理极值
        ic_results = self.screener.calculate_multi_horizon_ic(
            factors_with_extremes, returns
        )
        
        self.assertIsInstance(ic_results, dict)

class TestConfigurationSystem(TestProfessionalFactorScreener):
    """配置系统测试"""
    
    def test_default_configuration(self):
        """测试默认配置"""
        default_config = ScreeningConfig()
        
        # 验证默认值
        self.assertEqual(default_config.ic_horizons, [1, 3, 5, 10, 20])
        self.assertEqual(default_config.min_sample_size, 100)
        self.assertEqual(default_config.alpha_level, 0.05)
        self.assertEqual(default_config.fdr_method, "benjamini_hochberg")
    
    def test_custom_configuration(self):
        """测试自定义配置"""
        custom_config = ScreeningConfig(
            ic_horizons=[1, 5, 10],
            min_sample_size=50,
            alpha_level=0.01,
            fdr_method="bonferroni"
        )
        
        screener = ProfessionalFactorScreener(
            str(self.test_data_dir), 
            config=custom_config
        )
        
        self.assertEqual(screener.config.ic_horizons, [1, 5, 10])
        self.assertEqual(screener.config.min_sample_size, 50)
        self.assertEqual(screener.config.alpha_level, 0.01)
        self.assertEqual(screener.config.fdr_method, "bonferroni")

def run_all_tests():
    """运行所有测试"""
    # 创建测试套件
    test_classes = [
        TestDataLoading,
        TestPredictivePowerAnalysis,
        TestStabilityAnalysis,
        TestIndependenceAnalysis,
        TestPracticalityAnalysis,
        TestAdaptabilityAnalysis,
        TestStatisticalTesting,
        TestComprehensiveScoring,
        TestIntegrationWorkflow,
        TestPerformance,
        TestEdgeCases,
        TestConfigurationSystem
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出测试结果统计
    print(f"\n{'='*80}")
    print(f"测试结果统计:")
    print(f"{'='*80}")
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"  - {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
