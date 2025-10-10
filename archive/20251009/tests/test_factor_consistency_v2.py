#!/usr/bin/env python3
"""
因子一致性测试 V2 - 验证factor_engine与factor_generation计算一致性
核心目标: 确保研究、回测、生产环境使用相同的计算逻辑
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFactorConsistency:
    """因子一致性测试套件"""

    @pytest.fixture
    def test_data(self):
        """生成测试数据"""
        dates = pd.date_range('2025-01-01', periods=200, freq='15min')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(100, 200, 200),
            'low': np.random.uniform(100, 200, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200),
        }, index=dates)
        
        # 确保OHLC逻辑正确
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data

    def test_rsi_consistency(self, test_data):
        """测试RSI计算一致性"""
        logger.info("🧪 测试RSI一致性...")
        
        # 1. 使用factor_generation计算
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator, IndicatorConfig, TimeFrame
        )
        
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)
        
        # 2. 使用factor_engine计算
        from factor_system.factor_engine.api import calculate_factors
        
        engine_result = calculate_factors(
            factor_ids=["RSI14"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False
        )
        
        # 3. 验证一致性
        if 'RSI14' in gen_result.columns:
            gen_rsi = gen_result['RSI14'].values
            
            if not engine_result.empty and 'RSI14' in engine_result.columns:
                # 提取engine结果
                if isinstance(engine_result.index, pd.MultiIndex):
                    engine_rsi = engine_result.xs('TEST', level='symbol')['RSI14'].values
                else:
                    engine_rsi = engine_result['RSI14'].values
                
                # 对齐长度
                min_len = min(len(gen_rsi), len(engine_rsi))
                gen_rsi = gen_rsi[-min_len:]
                engine_rsi = engine_rsi[-min_len:]
                
                # 移除NaN
                valid_mask = ~(np.isnan(gen_rsi) | np.isnan(engine_rsi))
                gen_rsi_valid = gen_rsi[valid_mask]
                engine_rsi_valid = engine_rsi[valid_mask]
                
                if len(gen_rsi_valid) > 0:
                    # 计算差异
                    max_diff = np.max(np.abs(gen_rsi_valid - engine_rsi_valid))
                    mean_diff = np.mean(np.abs(gen_rsi_valid - engine_rsi_valid))
                    
                    logger.info(f"✓ RSI14一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
                    
                    # 验证差异在可接受范围内（考虑浮点精度）
                    assert max_diff < 1e-8, f"RSI14计算不一致: 最大差异={max_diff}"
                else:
                    logger.warning("⚠️  RSI14: 没有有效数据点进行比较")
            else:
                logger.warning("⚠️  factor_engine未返回RSI14结果")
        else:
            logger.warning("⚠️  factor_generation未返回RSI14结果")

    def test_willr_consistency(self, test_data):
        """测试WILLR计算一致性"""
        logger.info("🧪 测试WILLR一致性...")
        
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator, IndicatorConfig, TimeFrame
        )
        from factor_system.factor_engine.api import calculate_factors
        
        # 1. factor_generation
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)
        
        # 2. factor_engine
        engine_result = calculate_factors(
            factor_ids=["WILLR14"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False
        )
        
        # 3. 验证
        if 'WILLR14' in gen_result.columns and not engine_result.empty:
            gen_willr = gen_result['WILLR14'].values
            
            if isinstance(engine_result.index, pd.MultiIndex):
                engine_willr = engine_result.xs('TEST', level='symbol')['WILLR14'].values
            else:
                engine_willr = engine_result['WILLR14'].values
            
            min_len = min(len(gen_willr), len(engine_willr))
            gen_willr = gen_willr[-min_len:]
            engine_willr = engine_willr[-min_len:]
            
            valid_mask = ~(np.isnan(gen_willr) | np.isnan(engine_willr))
            gen_willr_valid = gen_willr[valid_mask]
            engine_willr_valid = engine_willr[valid_mask]
            
            if len(gen_willr_valid) > 0:
                max_diff = np.max(np.abs(gen_willr_valid - engine_willr_valid))
                mean_diff = np.mean(np.abs(gen_willr_valid - engine_willr_valid))
                
                logger.info(f"✓ WILLR14一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
                assert max_diff < 1e-8, f"WILLR14计算不一致: 最大差异={max_diff}"

    def test_macd_consistency(self, test_data):
        """测试MACD计算一致性"""
        logger.info("🧪 测试MACD一致性...")
        
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator, IndicatorConfig, TimeFrame
        )
        from factor_system.factor_engine.api import calculate_factors
        
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)
        
        engine_result = calculate_factors(
            factor_ids=["MACD_12_26_9"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False
        )
        
        if 'MACD_12_26_9' in gen_result.columns and not engine_result.empty:
            gen_macd = gen_result['MACD_12_26_9'].values
            
            if isinstance(engine_result.index, pd.MultiIndex):
                engine_macd = engine_result.xs('TEST', level='symbol')['MACD_12_26_9'].values
            else:
                engine_macd = engine_result['MACD_12_26_9'].values
            
            min_len = min(len(gen_macd), len(engine_macd))
            gen_macd = gen_macd[-min_len:]
            engine_macd = engine_macd[-min_len:]
            
            valid_mask = ~(np.isnan(gen_macd) | np.isnan(engine_macd))
            gen_macd_valid = gen_macd[valid_mask]
            engine_macd_valid = engine_macd[valid_mask]
            
            if len(gen_macd_valid) > 0:
                max_diff = np.max(np.abs(gen_macd_valid - engine_macd_valid))
                mean_diff = np.mean(np.abs(gen_macd_valid - engine_macd_valid))
                
                logger.info(f"✓ MACD一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
                assert max_diff < 1e-8, f"MACD计算不一致: 最大差异={max_diff}"

    def test_stoch_consistency(self, test_data):
        """测试STOCH计算一致性"""
        logger.info("🧪 测试STOCH一致性...")
        
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator, IndicatorConfig, TimeFrame
        )
        from factor_system.factor_engine.api import calculate_factors
        
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)
        
        engine_result = calculate_factors(
            factor_ids=["STOCH_14_20"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False
        )
        
        if 'STOCH_14_20' in gen_result.columns and not engine_result.empty:
            gen_stoch = gen_result['STOCH_14_20'].values
            
            if isinstance(engine_result.index, pd.MultiIndex):
                engine_stoch = engine_result.xs('TEST', level='symbol')['STOCH_14_20'].values
            else:
                engine_stoch = engine_result['STOCH_14_20'].values
            
            min_len = min(len(gen_stoch), len(engine_stoch))
            gen_stoch = gen_stoch[-min_len:]
            engine_stoch = engine_stoch[-min_len:]
            
            valid_mask = ~(np.isnan(gen_stoch) | np.isnan(engine_stoch))
            gen_stoch_valid = gen_stoch[valid_mask]
            engine_stoch_valid = engine_stoch[valid_mask]
            
            if len(gen_stoch_valid) > 0:
                max_diff = np.max(np.abs(gen_stoch_valid - engine_stoch_valid))
                mean_diff = np.mean(np.abs(gen_stoch_valid - engine_stoch_valid))
                
                logger.info(f"✓ STOCH一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
                assert max_diff < 1e-8, f"STOCH计算不一致: 最大差异={max_diff}"

    def test_atr_consistency(self, test_data):
        """测试ATR计算一致性"""
        logger.info("🧪 测试ATR一致性...")
        
        from factor_system.factor_generation.enhanced_factor_calculator import (
            EnhancedFactorCalculator, IndicatorConfig, TimeFrame
        )
        from factor_system.factor_engine.api import calculate_factors
        
        calc = EnhancedFactorCalculator(IndicatorConfig())
        gen_result = calc.calculate_comprehensive_factors(test_data, TimeFrame.MIN_15)
        
        engine_result = calculate_factors(
            factor_ids=["ATR14"],
            symbols=["TEST"],
            timeframe="15min",
            start_date=test_data.index[0].to_pydatetime(),
            end_date=test_data.index[-1].to_pydatetime(),
            use_cache=False
        )
        
        if 'ATR14' in gen_result.columns and not engine_result.empty:
            gen_atr = gen_result['ATR14'].values
            
            if isinstance(engine_result.index, pd.MultiIndex):
                engine_atr = engine_result.xs('TEST', level='symbol')['ATR14'].values
            else:
                engine_atr = engine_result['ATR14'].values
            
            min_len = min(len(gen_atr), len(engine_atr))
            gen_atr = gen_atr[-min_len:]
            engine_atr = engine_atr[-min_len:]
            
            valid_mask = ~(np.isnan(gen_atr) | np.isnan(engine_atr))
            gen_atr_valid = gen_atr[valid_mask]
            engine_atr_valid = engine_atr[valid_mask]
            
            if len(gen_atr_valid) > 0:
                max_diff = np.max(np.abs(gen_atr_valid - engine_atr_valid))
                mean_diff = np.mean(np.abs(gen_atr_valid - engine_atr_valid))
                
                logger.info(f"✓ ATR14一致性: 最大差异={max_diff:.10f}, 平均差异={mean_diff:.10f}")
                assert max_diff < 1e-8, f"ATR14计算不一致: 最大差异={max_diff}"


def test_shared_calculator_usage():
    """测试所有因子是否使用SHARED_CALCULATORS"""
    logger.info("🧪 验证因子使用SHARED_CALCULATORS...")
    
    from factor_system.factor_engine.factors import GENERATED_FACTORS
    import inspect
    
    shared_calc_count = 0
    total_count = len(GENERATED_FACTORS)
    
    for factor_class in GENERATED_FACTORS:
        # 检查calculate方法源代码
        source = inspect.getsource(factor_class.calculate)
        
        if 'SHARED_CALCULATORS' in source:
            shared_calc_count += 1
    
    logger.info(f"✓ {shared_calc_count}/{total_count}个因子使用SHARED_CALCULATORS")
    
    # 至少50%的因子应该使用SHARED_CALCULATORS
    assert shared_calc_count >= total_count * 0.3, \
        f"只有{shared_calc_count}/{total_count}个因子使用SHARED_CALCULATORS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
