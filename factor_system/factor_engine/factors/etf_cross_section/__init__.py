#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF横截面因子包初始化文件
"""

# 导入所有模块，使其可以被外部导入
from .factor_registry import ETFFactorRegistry, get_factor_registry, register_etf_factor, FactorCategory
from .etf_factor_factory import ETFFactorFactory
from .candidate_factor_generator import ETFCandidateFactorGenerator, FactorVariant
from .batch_factor_calculator import BatchFactorCalculator, calculate_all_etf_factors
from .ic_analyzer import ICAnalyzer
from .stability_analyzer import StabilityAnalyzer
from .factor_screener import FactorScreener, ScreeningCriteria, screen_etf_factors
from .factor_classifier import classify_etf_factors, ClassifiedFactor
# 导入统一管理器（推荐使用）
from .unified_manager import (
    ETFCrossSectionUnifiedManager,
    create_etf_cross_section_manager,
    ETFCrossSectionConfig,
    DefaultProgressMonitor
)

# 导入接口定义
from .interfaces import (
    IFactorCalculator,
    ICrossSectionManager,
    IFactorRegistry,
    IProgressMonitor,
    FactorCalculationResult,
    CrossSectionResult
)
# 注意：ETFCrossSectionFactors在etf_cross_section.py文件中定义
# 由于包名和文件名相同，不能在__init__.py中直接导入
# 用户应该使用：from factor_system.factor_engine.factors.etf_cross_section import ETFCrossSectionFactors
# 或使用延迟导入函数：get_etf_cross_section_factors()

ETFCrossSectionFactors = None  # 占位符，实际使用延迟导入

# 延迟导入函数，避免循环依赖
def get_etf_cross_section_factors():
    """
    获取传统ETF横截面因子计算器（延迟导入）
    
    Returns:
        ETFCrossSectionFactors类，如果导入失败返回None
    """
    try:
        # 从父级目录的etf_cross_section.py文件导入
        import sys
        import importlib.util
        import os
        
        # 获取etf_cross_section.py文件的路径
        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        etf_file = os.path.join(parent_dir, 'etf_cross_section.py')
        
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("etf_cross_section_legacy", etf_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module.ETFCrossSectionFactors
    except Exception as e:
        print(f"警告: 延迟导入ETFCrossSectionFactors失败: {e}")
        return None

def get_etf_cross_section_factors_enhanced():
    """获取增强ETF横截面因子计算器（延迟导入）"""
    try:
        from .etf_cross_section_enhanced import ETFCrossSectionFactorsEnhanced
        return ETFCrossSectionFactorsEnhanced
    except ImportError as e:
        print(f"警告: 无法导入ETFCrossSectionFactorsEnhanced: {e}")
        return None

def get_etf_cross_section_strategy_enhanced():
    """获取增强策略（延迟导入）"""
    try:
        from .etf_cross_section_strategy_enhanced import ETFCrossSectionStrategyEnhanced, EnhancedStrategyConfig
        return ETFCrossSectionStrategyEnhanced, EnhancedStrategyConfig
    except ImportError as e:
        print(f"警告: 无法导入策略增强模块: {e}")
        return None, None

def get_etf_cross_section_storage_enhanced():
    """获取增强存储（延迟导入）"""
    try:
        from .etf_cross_section_storage_enhanced import ETFCrossSectionStorageEnhanced
        return ETFCrossSectionStorageEnhanced
    except ImportError as e:
        print(f"警告: 无法导入存储增强模块: {e}")
        return None

__all__ = [
    # 核心组件
    'ETFFactorRegistry',
    'get_factor_registry',
    'register_etf_factor',
    'FactorCategory',
    'ETFFactorFactory',
    'ETFCandidateFactorGenerator',
    'FactorVariant',
    'BatchFactorCalculator',
    'calculate_all_etf_factors',
    'ICAnalyzer',
    'StabilityAnalyzer',
    'FactorScreener',
    'ScreeningCriteria',
    'screen_etf_factors',
    'classify_etf_factors',
    'ClassifiedFactor',

    # 传统因子计算器
    'ETFCrossSectionFactors',

    # 统一管理器（推荐使用）
    'ETFCrossSectionUnifiedManager',
    'create_etf_cross_section_manager',
    'ETFCrossSectionConfig',
    'DefaultProgressMonitor',

    # 接口定义
    'IFactorCalculator',
    'ICrossSectionManager',
    'IFactorRegistry',
    'IProgressMonitor',
    'FactorCalculationResult',
    'CrossSectionResult',

    # 延迟导入函数
    'get_etf_cross_section_factors',
    'get_etf_cross_section_factors_enhanced',
    'get_etf_cross_section_strategy_enhanced',
    'get_etf_cross_section_storage_enhanced'
]