#!/usr/bin/env python3
"""
专业级量化交易因子筛选系统 - 5维度筛选框架
作者：量化首席工程师
版本：2.0.0
日期：2025-09-29

核心特性：
1. 5维度筛选框架：预测能力、稳定性、独立性、实用性、短周期适应性
2. 多周期IC分析：1日、3日、5日、10日、20日预测能力评估
3. 严格的统计显著性检验：Benjamini-Hochberg FDR校正
4. VIF检测和信息增量分析
5. 交易成本和流动性评估
6. 生产级性能优化和错误处理
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import time
import yaml
import pickle
import os
import glob
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from collections import defaultdict
import json

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FactorMetrics:
    """因子综合指标"""
    name: str
    
    # 预测能力指标
    ic_1d: float = 0.0
    ic_3d: float = 0.0
    ic_5d: float = 0.0
    ic_10d: float = 0.0
    ic_20d: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    ic_decay_rate: float = 0.0
    ic_longevity: int = 0
    
    # 稳定性指标
    rolling_ic_mean: float = 0.0
    rolling_ic_std: float = 0.0
    rolling_ic_stability: float = 0.0
    ic_consistency: float = 0.0
    cross_section_stability: float = 0.0
    
    # 独立性指标
    vif_score: float = 0.0
    correlation_max: float = 0.0
    information_increment: float = 0.0
    redundancy_score: float = 0.0
    
    # 实用性指标
    turnover_rate: float = 0.0
    transaction_cost: float = 0.0
    cost_efficiency: float = 0.0
    liquidity_demand: float = 0.0
    capacity_score: float = 0.0
    
    # 短周期适应性指标
    reversal_effect: float = 0.0
    momentum_persistence: float = 0.0
    volatility_sensitivity: float = 0.0
    regime_adaptability: float = 0.0
    
    # 统计显著性
    p_value: float = 1.0
    corrected_p_value: float = 1.0
    is_significant: bool = False
    
    # 综合评分
    predictive_score: float = 0.0
    stability_score: float = 0.0
    independence_score: float = 0.0
    practicality_score: float = 0.0
    adaptability_score: float = 0.0
    comprehensive_score: float = 0.0
    
    # 元数据
    sample_size: int = 0
    calculation_time: float = 0.0
    data_quality_score: float = 0.0

@dataclass
class ScreeningConfig:
    """筛选配置"""
    # 多周期IC参数
    ic_horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    min_sample_size: int = 100
    rolling_window: int = 60
    
    # 显著性检验参数
    alpha_level: float = 0.05
    fdr_method: str = "benjamini_hochberg"  # "benjamini_hochberg" or "bonferroni"
    
    # 独立性分析参数
    vif_threshold: float = 5.0
    correlation_threshold: float = 0.8
    base_factors: List[str] = field(default_factory=lambda: ["MA5", "MA10", "RSI14", "MACD_12_26_9"])
    
    # 交易成本参数
    commission_rate: float = 0.002  # 0.2%佣金
    slippage_bps: float = 5.0      # 5bp滑点
    market_impact_coeff: float = 0.1
    
    # 筛选阈值
    min_ic_threshold: float = 0.02
    min_ir_threshold: float = 0.5
    min_stability_threshold: float = 0.6
    max_vif_threshold: float = 10.0
    max_cost_threshold: float = 0.01  # 1%最大交易成本
    
    # 性能参数
    max_workers: int = 4
    cache_enabled: bool = True
    memory_limit_mb: int = 2048
    
    # 评分权重
    weight_predictive: float = 0.35
    weight_stability: float = 0.25
    weight_independence: float = 0.20
    weight_practicality: float = 0.15
    weight_adaptability: float = 0.05

class ProfessionalFactorScreener:
    """专业级因子筛选器 - 5维度筛选框架"""
    
    def __init__(self, data_root: str = None, config: Optional[ScreeningConfig] = None):
        """初始化筛选器
        
        Args:
            data_root: 向后兼容参数，优先使用config中的路径配置
            config: 筛选配置对象
        """
        self.config = config or ScreeningConfig()
        
        # 路径优先级: config > data_root参数 > 默认值
        if hasattr(self.config, 'factor_data_root'):
            self.data_root = Path(self.config.factor_data_root)
        elif data_root:
            self.data_root = Path(data_root)
        else:
            self.data_root = Path("/Users/zhangshenshen/深度量化0927/factor_system/因子输出")  # 默认因子数据目录

        # 设置日志和缓存路径
        self.log_root = Path(getattr(self.config, 'log_root', './logs/screening'))
        self.cache_dir = Path(getattr(self.config, 'cache_root', self.data_root / "cache"))

        # 设置筛选报告专用目录
        self.screening_results_dir = Path("/Users/zhangshenshen/深度量化0927/factor_system/因子筛选")
        self.screening_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化增强版结果管理器（延迟记录日志）
        try:
            from enhanced_result_manager import EnhancedResultManager
            self.result_manager = EnhancedResultManager(str(self.screening_results_dir))
        except ImportError as e:
            self.result_manager = None

        # 创建必要的目录
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

        # 调试信息
        self.logger.info(f"DEBUG: data_root设置为: {self.data_root}")
        
        # 现在可以安全地记录增强版结果管理器状态
        if self.result_manager:
            self.logger.info("✅ 增强版结果管理器已启用")
        else:
            self.logger.warning("⚠️ 使用传统存储方式")
        
        # 性能监控
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        self.logger.info("专业级因子筛选器初始化完成")
        self.logger.info(f"配置: IC周期={self.config.ic_horizons}, 最小样本={self.config.min_sample_size}")
        self.logger.info(f"显著性水平={self.config.alpha_level}, FDR方法={self.config.fdr_method}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置专业级日志系统 - 改进版"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        logger.setLevel(logging.INFO)
        
        # 清除现有处理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 使用日志轮转 - 关键修复
        from logging.handlers import RotatingFileHandler
        today = datetime.now().strftime('%Y%m%d')
        log_file = self.log_root / f"screener_{today}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,           # 保留5个备份
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def _generate_factor_metadata(self, factors_df: pd.DataFrame) -> dict:
        """生成因子元数据"""
        metadata = {}
        
        for factor_name in factors_df.columns:
            meta = {
                'name': factor_name,
                'type': self._infer_factor_type(factor_name),
                'warmup_period': self._infer_warmup_period(factor_name),
                'description': self._generate_factor_description(factor_name)
            }
            
            # 计算统计信息
            factor_data = factors_df[factor_name]
            meta.update({
                'total_periods': len(factor_data),
                'missing_periods': factor_data.isnull().sum(),
                'missing_ratio': factor_data.isnull().sum() / len(factor_data),
                'first_valid_index': self._find_first_non_missing_index(factor_data),
                'valid_ratio': 1 - (factor_data.isnull().sum() / len(factor_data))
            })
            
            metadata[factor_name] = meta
            
        return metadata
    
    def _infer_factor_type(self, factor_name: str) -> str:
        """根据因子名称推断类型"""
        name_lower = factor_name.lower()
        
        if any(indicator in name_lower for indicator in ['ma', 'ema', 'sma', 'wma', 'dema', 'tema', 'trima', 'kama', 't3']):
            return 'trend'
        elif any(indicator in name_lower for indicator in ['rsi', 'stoch', 'cci', 'willr', 'mfi', 'roc', 'mom']):
            return 'momentum'
        elif any(indicator in name_lower for indicator in ['bb', 'bollinger', 'atr', 'std', 'mstd']):
            return 'volatility'
        elif any(indicator in name_lower for indicator in ['volume', 'obv', 'vwap']):
            return 'volume'
        elif any(indicator in name_lower for indicator in ['macd', 'signal', 'histogram', 'hist']):
            return 'momentum'
        elif any(indicator in name_lower for indicator in ['cdl', 'pattern']):
            return 'pattern'
        else:
            return 'unknown'
    
    def _infer_warmup_period(self, factor_name: str) -> int:
        """根据因子名称推断预热期"""
        name_lower = factor_name.lower()
        
        # 移动平均类 - 精确匹配数字
        import re
        ma_match = re.search(r'ma(\d+)', name_lower)
        if ma_match:
            return int(ma_match.group(1))
        
        sma_match = re.search(r'sma_?(\d+)', name_lower)
        if sma_match:
            return int(sma_match.group(1))
            
        ema_match = re.search(r'ema_?(\d+)', name_lower)
        if ema_match:
            return int(ema_match.group(1))
        
        # RSI类
        rsi_match = re.search(r'rsi(\d+)', name_lower)
        if rsi_match:
            return int(rsi_match.group(1))
        elif 'rsi' in name_lower:
            return 14
        
        # 布林带类
        bb_match = re.search(r'bb_(\d+)', name_lower)
        if bb_match:
            return int(bb_match.group(1))
        elif 'bb' in name_lower or 'bollinger' in name_lower:
            return 20
        
        # MACD类
        if 'macd' in name_lower:
            return 26
        
        # CCI类
        cci_match = re.search(r'cci(\d+)', name_lower)
        if cci_match:
            return int(cci_match.group(1))
        elif 'cci' in name_lower:
            return 20
        
        # WILLR类
        willr_match = re.search(r'willr(\d+)', name_lower)
        if willr_match:
            return int(willr_match.group(1))
        elif 'willr' in name_lower:
            return 14
        
        # ATR类
        atr_match = re.search(r'atr(\d+)', name_lower)
        if atr_match:
            return int(atr_match.group(1))
        elif 'atr' in name_lower:
            return 14
        
        # 默认预热期
        return 20
    
    def _generate_factor_description(self, factor_name: str) -> str:
        """生成因子描述"""
        name_lower = factor_name.lower()
        
        if 'ma' in name_lower:
            return f"移动平均线指标 - {factor_name}"
        elif 'rsi' in name_lower:
            return f"相对强弱指标 - {factor_name}"
        elif 'macd' in name_lower:
            return f"移动平均收敛散度指标 - {factor_name}"
        elif 'bb' in name_lower:
            return f"布林带指标 - {factor_name}"
        elif 'volume' in name_lower:
            return f"成交量指标 - {factor_name}"
        else:
            return f"技术指标 - {factor_name}"
    
    def _find_first_non_missing_index(self, series: pd.Series) -> int:
        """找到第一个非缺失值的索引位置"""
        non_null_mask = series.notna()
        if non_null_mask.any():
            return non_null_mask.idxmax()
        return len(series)
    
    def _smart_forward_fill(self, series: pd.Series) -> pd.Series:
        """智能前向填充"""
        result = series.copy()
        
        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()
        
        if first_valid_idx is not None:
            # 用第一个有效值填充前面的缺失值
            first_valid_value = series.loc[first_valid_idx]
            result = result.fillna(method='bfill', limit=1)
            result = result.fillna(first_valid_value)
            
            # 前向填充剩余的缺失值
            result = result.fillna(method='ffill')
        
        return result
    
    def _smart_interpolation(self, series: pd.Series) -> pd.Series:
        """智能插值"""
        result = series.copy()
        
        # 找到第一个有效值
        first_valid_idx = series.first_valid_index()
        
        if first_valid_idx is not None:
            # 用第一个有效值填充前面的缺失值
            first_valid_value = series.loc[first_valid_idx]
            result.loc[:first_valid_idx] = result.loc[:first_valid_idx].fillna(first_valid_value)
            
            # 对剩余缺失值进行线性插值
            result = result.interpolate(method='linear')
            
            # 如果还有缺失值，用前向填充
            result = result.fillna(method='ffill')
            result = result.fillna(method='bfill')
        
        return result
    
    def smart_missing_value_handling(self, factors_df: pd.DataFrame, factor_metadata: dict = None) -> tuple:
        """
        智能缺失值处理，区分正常预热期缺失和问题数据
        
        Args:
            factors_df: 因子数据DataFrame
            factor_metadata: 因子元数据，包含预热期信息
            
        Returns:
            tuple: (cleaned_df, handling_report)
        """
        if factor_metadata is None:
            factor_metadata = self._generate_factor_metadata(factors_df)
        
        handling_report = {
            'total_factors': len(factors_df.columns),
            'removed_factors': [],
            'handled_factors': [],
            'forward_filled_factors': [],
            'interpolated_factors': [],
            'dropped_factors': []
        }
        
        cleaned_df = factors_df.copy()
        
        for factor_name in factors_df.columns:
            factor_data = factors_df[factor_name]
            missing_count = factor_data.isnull().sum()
            
            if missing_count == 0:
                handling_report['handled_factors'].append(factor_name)
                continue
            
            missing_ratio = missing_count / len(factor_data)
            
            # 获取因子元数据
            meta = factor_metadata.get(factor_name, {})
            warmup_period = meta.get('warmup_period', 20)
            factor_type = meta.get('type', 'unknown')
            
            # 判断缺失值模式
            first_valid_idx = factor_data.first_valid_index()
            if first_valid_idx is not None:
                first_valid_pos = factor_data.index.get_loc(first_valid_idx)
            else:
                first_valid_pos = len(factor_data)
            
            # 决策逻辑
            if first_valid_pos <= warmup_period * 1.5:  # 允许1.5倍的预热期容忍度
                # 正常预热期缺失，进行智能填充
                if factor_type in ['momentum', 'trend', 'volatility']:
                    # 技术指标类因子使用前向填充
                    cleaned_df[factor_name] = self._smart_forward_fill(factor_data)
                    handling_report['forward_filled_factors'].append(factor_name)
                else:
                    # 其他类型因子使用插值
                    cleaned_df[factor_name] = self._smart_interpolation(factor_data)
                    handling_report['interpolated_factors'].append(factor_name)
                
                handling_report['handled_factors'].append(factor_name)
                
            elif missing_ratio > 0.8:
                # 缺失比例过高，删除
                cleaned_df = cleaned_df.drop(columns=[factor_name])
                handling_report['dropped_factors'].append(factor_name)
                handling_report['removed_factors'].append(factor_name)
                
            else:
                # 随机缺失，使用插值
                cleaned_df[factor_name] = self._smart_interpolation(factor_data)
                handling_report['interpolated_factors'].append(factor_name)
                handling_report['handled_factors'].append(factor_name)
        
        return cleaned_df, handling_report

    def validate_factor_data_quality(self, factors_df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """智能数据质量验证 - 解决根本问题而不是粗暴删除"""
        self.logger.info(f"开始智能数据质量验证: {symbol} {timeframe}")

        original_shape = factors_df.shape
        issues_found = []

        # 1. 检查非数值列（根本问题）
        non_numeric_cols = factors_df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns.tolist()
        if non_numeric_cols:
            self.logger.warning(f"发现非数值列: {non_numeric_cols}")
            factors_df = factors_df.drop(columns=non_numeric_cols)
            issues_found.append(f"移除非数值列: {non_numeric_cols}")

        # 2. 智能缺失值处理 - 核心改进
        if factors_df.isnull().any().any():
            self.logger.info("开始智能缺失值处理...")
            factor_metadata = self._generate_factor_metadata(factors_df)
            factors_df, handling_report = self.smart_missing_value_handling(factors_df, factor_metadata)
            
            self.logger.info(f"智能缺失值处理完成:")
            self.logger.info(f"  - 总因子数: {handling_report['total_factors']}")
            self.logger.info(f"  - 前向填充因子数: {len(handling_report['forward_filled_factors'])}")
            self.logger.info(f"  - 插值填充因子数: {len(handling_report['interpolated_factors'])}")
            self.logger.info(f"  - 删除因子数: {len(handling_report['removed_factors'])}")
            
            if handling_report['removed_factors']:
                issues_found.append(f"智能处理移除因子: {handling_report['removed_factors']}")

        # 3. 检查无穷值 - 修复而不是删除
        inf_cols = []
        for col in factors_df.select_dtypes(include=[np.number]).columns:
            if np.isinf(factors_df[col]).any():
                inf_cols.append(col)
                # 用极值替换无穷值
                factors_df[col] = factors_df[col].replace([np.inf, -np.inf], [factors_df[col].quantile(0.99), factors_df[col].quantile(0.01)])

        if inf_cols:
            self.logger.info(f"修复无穷值列: {inf_cols}")
            issues_found.append(f"修复无穷值列: {inf_cols}")

        # 4. 检查常量列（使用更严格的标准）
        constant_cols = []
        for col in factors_df.select_dtypes(include=[np.number]).columns:
            if factors_df[col].std() < 1e-10:  # 更严格的常量检测
                constant_cols.append(col)

        if constant_cols:
            self.logger.warning(f"发现常量列: {constant_cols}")
            factors_df = factors_df.drop(columns=constant_cols)
            issues_found.append(f"移除常量列: {constant_cols}")

        # 5. 检查重复列
        duplicate_cols = factors_df.columns[factors_df.columns.duplicated()].tolist()
        if duplicate_cols:
            self.logger.warning(f"发现重复列: {duplicate_cols}")
            factors_df = factors_df.loc[:, ~factors_df.columns.duplicated()]
            issues_found.append(f"移除重复列: {duplicate_cols}")

        final_shape = factors_df.shape

        # 报告验证结果
        retention_rate = final_shape[1] / original_shape[1]
        self.logger.info(f"智能数据质量验证完成:")
        self.logger.info(f"  - 原始形状: {original_shape}")
        self.logger.info(f"  - 最终形状: {final_shape}")
        self.logger.info(f"  - 因子保留率: {retention_rate:.1%}")
        
        if issues_found:
            for issue in issues_found:
                self.logger.info(f"  - {issue}")

        # 确保还有足够的因子数据
        if len(factors_df.columns) < 10:
            raise ValueError(f"验证后因子数量过少: {len(factors_df.columns)} < 10")

        return factors_df

    def load_factors(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载因子数据 - 增强版本"""
        start_time = time.time()
        self.logger.info(f"加载因子数据: {symbol} {timeframe}")
        
        # 处理symbol格式
        clean_symbol = symbol.replace('.HK', '')
        
        # 搜索策略：按优先级搜索不同格式的文件
        search_patterns = [
            # 新格式：timeframe子目录 (带.HK后缀)
            (self.data_root / timeframe, f"{clean_symbol}.HK_{timeframe}_factors_*.parquet"),
            (self.data_root / timeframe, f"{clean_symbol}HK_{timeframe}_factors_*.parquet"),
            (self.data_root / timeframe, f"{clean_symbol}_{timeframe}_factors_*.parquet"),
            # multi_tf格式
            (self.data_root, f"aligned_multi_tf_factors_{clean_symbol}*.parquet"),
            # 根目录格式
            (self.data_root, f"{clean_symbol}*_{timeframe}_factors_*.parquet"),
        ]
        
        for search_dir, pattern in search_patterns:
            if search_dir.exists():
                factor_files = list(search_dir.glob(pattern))
                if factor_files:
                    selected_file = factor_files[-1]  # 选择最新文件
                    self.logger.info(f"找到因子文件: {selected_file}")
                    
                    try:
                        factors = pd.read_parquet(selected_file)

                        # 数据质量检查
                        if factors.empty:
                            self.logger.warning(f"因子文件为空: {selected_file}")
                            continue

                        # 确保索引是datetime类型
                        if not isinstance(factors.index, pd.DatetimeIndex):
                            factors.index = pd.to_datetime(factors.index)

                        # Linus式数据质量验证 - 解决根本问题
                        factors = self.validate_factor_data_quality(factors, symbol, timeframe)

                        self.logger.info(f"因子数据加载成功: 形状={factors.shape}")
                        self.logger.info(f"时间范围: {factors.index.min()} 到 {factors.index.max()}")
                        self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")

                        return factors
                        
                    except Exception as e:
                        self.logger.error(f"加载因子文件失败 {selected_file}: {str(e)}")
                        continue
        
        # 详细错误信息
        self.logger.error(f"未找到因子数据:")
        self.logger.error(f"搜索路径: {self.data_root}")
        self.logger.error(f"搜索符号: {clean_symbol}")
        self.logger.error(f"时间框架: {timeframe}")
        
        available_dirs = [d.name for d in self.data_root.iterdir() if d.is_dir()]
        self.logger.error(f"可用目录: {available_dirs}")
        
        raise FileNotFoundError(f"No factor data found for {symbol} {timeframe}")
    
    def load_price_data(self, symbol: str, timeframe: str = None) -> pd.DataFrame:
        """加载价格数据 - 智能匹配时间框架（修复版）"""
        start_time = time.time()
        self.logger.info(f"加载价格数据: {symbol} (时间框架: {timeframe})")
        
        # 处理symbol格式
        if symbol.endswith('.HK'):
            clean_symbol = symbol.replace('.HK', '') + 'HK'
        else:
            clean_symbol = symbol
        
        # 原始数据路径 - 支持相对路径
        raw_data_path = self.data_root.parent / "raw" / "HK"
        if not raw_data_path.exists():
            raw_data_path = Path("/Users/zhangshenshen/深度量化0927/raw/HK")
        
        # 时间框架到文件名的映射
        timeframe_map = {
            '1min': '1min',
            '2min': '2min',
            '3min': '3min',
            '5min': '5min',
            '15min': '15m',
            '30min': '30m',
            '60min': '60m',
            'daily': '1day',
            '1d': '1day',
        }
        
        # 根据时间框架智能选择搜索模式
        if timeframe and timeframe in timeframe_map:
            file_pattern = timeframe_map[timeframe]
            self.logger.info(f"根据时间框架 '{timeframe}' 搜索 '{file_pattern}' 格式文件")
            search_patterns = [
                f"{clean_symbol}_{file_pattern}_*.parquet",  # 精确匹配
                f"{clean_symbol}_*.parquet",                  # 备用
            ]
        else:
            # 默认搜索顺序（保持向后兼容）
            self.logger.warning(f"未指定时间框架或不在映射表中，使用默认搜索")
            search_patterns = [
                f"{clean_symbol}_60m_*.parquet",    # 60分钟数据
                f"{clean_symbol}_1day_*.parquet",   # 日线数据
                f"{clean_symbol}_*.parquet",        # 任意时间框架
            ]
        
        for pattern in search_patterns:
            price_files = list(raw_data_path.glob(pattern))
            if price_files:
                selected_file = price_files[-1]  # 选择最新文件
                self.logger.info(f"找到价格文件: {selected_file}")
                
                try:
                    price_data = pd.read_parquet(selected_file)
                    
                    # 数据预处理
                    if 'timestamp' in price_data.columns:
                        price_data = price_data.set_index('timestamp')
                    
                    if not isinstance(price_data.index, pd.DatetimeIndex):
                        price_data.index = pd.to_datetime(price_data.index)
                    
                    # 确保包含必要的列
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in price_data.columns]
                    if missing_cols:
                        self.logger.error(f"价格数据缺少必要列: {missing_cols}")
                        continue
                    
                    self.logger.info(f"价格数据加载成功: 形状={price_data.shape}")
                    self.logger.info(f"时间范围: {price_data.index.min()} 到 {price_data.index.max()}")
                    self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")
                    
                    return price_data[required_cols]
                    
                except Exception as e:
                    self.logger.error(f"加载价格文件失败 {selected_file}: {str(e)}")
                    continue
        
        self.logger.error(f"未找到价格数据:")
        self.logger.error(f"搜索路径: {raw_data_path}")
        self.logger.error(f"搜索符号: {clean_symbol}")
        
        available_files = [f.name for f in raw_data_path.glob('*.parquet')][:10]
        self.logger.error(f"可用文件示例: {available_files}")
        
        raise FileNotFoundError(f"No price data found for {symbol}")
    
    # ==================== 1. 预测能力分析 ====================
    
    def calculate_multi_horizon_ic(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """计算多周期IC值 - 核心预测能力评估"""
        self.logger.info("开始多周期IC计算...")
        start_time = time.time()
        
        ic_results = {}
        horizons = self.config.ic_horizons
        
        # 预计算所有周期的收益率
        future_returns = {}
        for horizon in horizons:
            future_returns[horizon] = returns.shift(-horizon)
        
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        total_factors = len(factor_cols)
        processed = 0
        
        for factor in factor_cols:
            processed += 1
            if processed % 50 == 0:
                self.logger.info(f"多周期IC计算进度: {processed}/{total_factors}")
            
            factor_values = factors[factor].dropna()
            horizon_ics = {}
            
            for horizon in horizons:
                aligned_returns = future_returns[horizon].reindex(factor_values.index).dropna()
                common_idx = factor_values.index.intersection(aligned_returns.index)
                
                if len(common_idx) >= self.config.min_sample_size:
                    final_factor = factor_values.loc[common_idx]
                    final_returns = aligned_returns.loc[common_idx]
                    
                    try:
                        # 使用Spearman等级相关系数
                        ic, p_value = stats.spearmanr(final_factor, final_returns)
                        
                        if not (np.isnan(ic) or np.isnan(p_value)):
                            horizon_ics[f'ic_{horizon}d'] = ic
                            horizon_ics[f'p_value_{horizon}d'] = p_value
                            horizon_ics[f'sample_size_{horizon}d'] = len(final_factor)
                        
                    except Exception as e:
                        self.logger.debug(f"因子 {factor} 周期 {horizon} IC计算失败: {str(e)}")
                        continue
            
            if horizon_ics:
                ic_results[factor] = horizon_ics
        
        calc_time = time.time() - start_time
        self.logger.info(f"多周期IC计算完成: 有效因子={len(ic_results)}, 耗时={calc_time:.2f}秒")
        
        return ic_results
    
    def analyze_ic_decay(self, ic_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """分析IC衰减特征"""
        self.logger.info("分析IC衰减特征...")
        
        decay_metrics = {}
        horizons = self.config.ic_horizons
        
        for factor, metrics in ic_results.items():
            ic_values = []
            for horizon in horizons:
                ic_key = f'ic_{horizon}d'
                if ic_key in metrics:
                    ic_values.append(metrics[ic_key])
            
            if len(ic_values) >= 2:
                # 计算衰减率 (线性回归斜率)
                x = np.arange(len(ic_values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, ic_values)
                
                # 计算IC稳定性
                ic_stability = 1 - (np.std(ic_values) / (abs(np.mean(ic_values)) + 1e-8))
                
                # 计算IC持续性 (有效IC的数量)
                ic_longevity = len([ic for ic in ic_values if abs(ic) > 0.01])
                
                decay_metrics[factor] = {
                    'decay_rate': slope,
                    'ic_stability': max(0, ic_stability),
                    'max_ic': max(ic_values, key=abs),
                    'ic_longevity': ic_longevity,
                    'decay_r_squared': r_value ** 2,
                    'ic_mean': np.mean(ic_values),
                    'ic_std': np.std(ic_values)
                }
        
        self.logger.info(f"IC衰减分析完成: {len(decay_metrics)} 个因子")
        return decay_metrics
    
    # ==================== 2. 稳定性分析 ====================
    
    def calculate_rolling_ic(self, factors: pd.DataFrame, returns: pd.Series, 
                           window: int = None) -> Dict[str, Dict[str, float]]:
        """计算滚动IC - 时间序列稳定性评估"""
        if window is None:
            window = self.config.rolling_window
            
        self.logger.info(f"计算滚动IC (窗口={window})...")
        start_time = time.time()
        
        rolling_ic_results = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()
            
            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) >= window + 20:  # 确保有足够数据
                final_factor = factor_values.loc[common_idx]
                final_returns = aligned_returns.loc[common_idx]
                
                # 计算滚动IC
                rolling_ics = []
                for i in range(window, len(final_factor)):
                    window_factor = final_factor.iloc[i-window:i]
                    window_returns = final_returns.iloc[i-window:i]
                    
                    if len(window_factor) >= 30:  # 最小窗口样本量
                        try:
                            ic, _ = stats.spearmanr(window_factor, window_returns)
                            if not np.isnan(ic):
                                rolling_ics.append(ic)
                        except:
                            continue
                
                if len(rolling_ics) >= 10:  # 至少10个滚动IC值
                    rolling_ic_mean = np.mean(rolling_ics)
                    rolling_ic_std = np.std(rolling_ics)
                    
                    # 稳定性指标
                    stability = 1 - (rolling_ic_std / (abs(rolling_ic_mean) + 1e-8))
                    consistency = len([ic for ic in rolling_ics if ic * rolling_ic_mean > 0]) / len(rolling_ics)
                    
                    rolling_ic_results[factor] = {
                        'rolling_ic_mean': rolling_ic_mean,
                        'rolling_ic_std': rolling_ic_std,
                        'rolling_ic_stability': max(0, stability),
                        'ic_consistency': consistency,
                        'rolling_periods': len(rolling_ics),
                        'ic_sharpe': rolling_ic_mean / (rolling_ic_std + 1e-8)
                    }
        
        calc_time = time.time() - start_time
        self.logger.info(f"滚动IC计算完成: {len(rolling_ic_results)} 个因子, 耗时={calc_time:.2f}秒")
        
        return rolling_ic_results
    
    def calculate_cross_sectional_stability(self, factors: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """计算截面稳定性 - 跨时间的一致性"""
        self.logger.info("计算截面稳定性...")
        
        stability_results = {}
        # 只选择数值类型的列，排除价格列和非数值列
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        factor_cols = [col for col in numeric_cols
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for factor in factor_cols:
            factor_data = factors[factor].dropna()
            
            if len(factor_data) >= 100:
                # 分时段分析稳定性
                n_periods = 5
                period_size = len(factor_data) // n_periods
                period_stats = []
                
                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (i + 1) * period_size if i < n_periods - 1 else len(factor_data)
                    period_data = factor_data.iloc[start_idx:end_idx]
                    
                    if len(period_data) >= 20:
                        period_stats.append({
                            'mean': period_data.mean(),
                            'std': period_data.std(),
                            'skew': period_data.skew(),
                            'kurt': period_data.kurtosis()
                        })
                
                if len(period_stats) >= 3:
                    # 计算各统计量的变异系数
                    means = [s['mean'] for s in period_stats]
                    stds = [s['std'] for s in period_stats]
                    
                    mean_cv = np.std(means) / (abs(np.mean(means)) + 1e-8)
                    std_cv = np.std(stds) / (np.mean(stds) + 1e-8)
                    
                    # 综合稳定性得分
                    stability_score = 1 / (1 + mean_cv + std_cv)
                    
                    stability_results[factor] = {
                        'cross_section_cv': mean_cv,
                        'cross_section_stability': stability_score,
                        'std_consistency': 1 / (1 + std_cv),
                        'periods_analyzed': len(period_stats)
                    }
        
        self.logger.info(f"截面稳定性计算完成: {len(stability_results)} 个因子")
        return stability_results
    
    # ==================== 3. 独立性分析 ====================
    
    def calculate_vif_scores(self, factors: pd.DataFrame) -> Dict[str, float]:
        """计算方差膨胀因子 (VIF) - 多重共线性检测"""
        self.logger.info("计算VIF得分...")

        # 只选择数值类型的列，排除价格列和非数值列
        numeric_cols = factors.select_dtypes(include=[np.number]).columns
        factor_cols = [col for col in numeric_cols
                      if col not in ['open', 'high', 'low', 'close', 'volume']]

        if len(factor_cols) < 2:
            self.logger.warning("数值型因子不足，无法计算VIF")
            return {}

        factor_data = factors[factor_cols].dropna()

        if len(factor_data) < 50:
            self.logger.warning("数据不足，无法计算VIF")
            return {}

        # 标准化数据
        factor_data_std = (factor_data - factor_data.mean()) / factor_data.std()
        factor_data_std = factor_data_std.fillna(0)

        vif_scores = {}

        try:
            # 计算每个因子的VIF
            for i, factor in enumerate(factor_cols):
                if len(factor_data_std.columns) > 1:
                    vif_value = variance_inflation_factor(factor_data_std.values, i)

                    # 处理异常值
                    if np.isnan(vif_value) or np.isinf(vif_value):
                        vif_value = 999.0  # 设置为高值表示高共线性

                    vif_scores[factor] = min(vif_value, 999.0)  # 限制最大值
                else:
                    vif_scores[factor] = 1.0

        except Exception as e:
            self.logger.warning(f"VIF计算失败: {str(e)}")
            # 使用相关性矩阵作为备选方案
            corr_matrix = factor_data.corr().abs()
            for factor in factor_cols:
                max_corr = corr_matrix[factor].drop(factor).max()
                vif_scores[factor] = 1 / (1 - max_corr + 1e-8)

        self.logger.info(f"VIF计算完成: {len(vif_scores)} 个因子")
        return vif_scores
    
    def calculate_factor_correlation_matrix(self, factors: pd.DataFrame) -> pd.DataFrame:
        """计算因子相关性矩阵"""
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        factor_data = factors[factor_cols].dropna()
        
        if len(factor_data) < 30:
            self.logger.warning("数据不足，无法计算相关性矩阵")
            return pd.DataFrame()
        
        # 使用Spearman相关性 (对异常值更稳健)
        correlation_matrix = factor_data.corr(method='spearman')
        
        return correlation_matrix
    
    def calculate_information_increment(self, factors: pd.DataFrame, returns: pd.Series,
                                     base_factors: List[str] = None) -> Dict[str, float]:
        """计算信息增量 - 相对于基准因子的增量信息"""
        if base_factors is None:
            base_factors = self.config.base_factors
        
        self.logger.info(f"计算信息增量 (基准因子: {base_factors})...")
        
        # 筛选存在的基准因子
        available_base = [f for f in base_factors if f in factors.columns]
        if not available_base:
            self.logger.warning("没有可用的基准因子")
            return {}
        
        # 计算基准因子组合的预测能力
        base_data = factors[available_base].dropna()
        base_combined = base_data.mean(axis=1)  # 等权重组合
        
        aligned_returns = returns.reindex(base_combined.index).dropna()
        common_idx = base_combined.index.intersection(aligned_returns.index)
        
        if len(common_idx) < self.config.min_sample_size:
            self.logger.warning("基准因子数据不足")
            return {}
        
        base_ic, _ = stats.spearmanr(base_combined.loc[common_idx], 
                                   aligned_returns.loc[common_idx])
        
        if np.isnan(base_ic):
            base_ic = 0.0
        
        # 计算每个因子的信息增量
        information_increment = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume'] + available_base]
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            factor_common_idx = factor_values.index.intersection(common_idx)
            
            if len(factor_common_idx) >= self.config.min_sample_size:
                # 基准 + 新因子的组合
                base_aligned = base_combined.reindex(factor_common_idx)
                factor_aligned = factor_values.loc[factor_common_idx]
                returns_aligned = aligned_returns.loc[factor_common_idx]
                
                # 等权重组合
                combined_factor = (base_aligned + factor_aligned) / 2
                
                try:
                    combined_ic, _ = stats.spearmanr(combined_factor, returns_aligned)
                    
                    if not np.isnan(combined_ic):
                        increment = combined_ic - base_ic
                        information_increment[factor] = increment
                    
                except Exception as e:
                    self.logger.debug(f"因子 {factor} 信息增量计算失败: {str(e)}")
                    continue
        
        self.logger.info(f"信息增量计算完成: {len(information_increment)} 个因子")
        return information_increment
    
    # ==================== 4. 实用性分析 ====================
    
    def calculate_trading_costs(self, factors: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """计算交易成本 - 基于因子的实际交易成本评估"""
        self.logger.info("计算交易成本...")
        
        cost_analysis = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # 获取价格和成交量数据
        close_prices = prices['close']
        volume = prices['volume']
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            
            # 时间对齐
            common_idx = factor_values.index.intersection(close_prices.index)
            if len(common_idx) < 50:
                continue
            
            factor_aligned = factor_values.loc[common_idx]
            prices_aligned = close_prices.loc[common_idx]
            volume_aligned = volume.loc[common_idx]
            
            # 计算因子变化率 (换手率代理)
            factor_change = factor_aligned.pct_change().abs()
            turnover_rate = factor_change.mean()
            
            # 估算交易成本
            commission_cost = turnover_rate * self.config.commission_rate
            slippage_cost = turnover_rate * (self.config.slippage_bps / 10000)
            
            # 市场冲击成本 (基于成交量)
            avg_volume = volume_aligned.mean()
            volume_factor = 1 / (1 + np.log(avg_volume + 1))  # 成交量越大，冲击越小
            impact_cost = turnover_rate * self.config.market_impact_coeff * volume_factor
            
            total_cost = commission_cost + slippage_cost + impact_cost
            
            # 成本效率
            cost_efficiency = 1 / (1 + total_cost)
            
            # 换手频率
            change_frequency = (factor_change > 0.05).mean()  # 5%以上变化的频率
            
            cost_analysis[factor] = {
                'turnover_rate': turnover_rate,
                'commission_cost': commission_cost,
                'slippage_cost': slippage_cost,
                'impact_cost': impact_cost,
                'total_cost': total_cost,
                'cost_efficiency': cost_efficiency,
                'change_frequency': change_frequency,
                'avg_volume': avg_volume
            }
        
        self.logger.info(f"交易成本计算完成: {len(cost_analysis)} 个因子")
        return cost_analysis
    
    def calculate_liquidity_requirements(self, factors: pd.DataFrame, 
                                       volume: pd.Series) -> Dict[str, Dict[str, float]]:
        """计算流动性需求"""
        self.logger.info("计算流动性需求...")
        
        liquidity_analysis = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_volume = volume.reindex(factor_values.index).dropna()
            
            common_idx = factor_values.index.intersection(aligned_volume.index)
            if len(common_idx) < 30:
                continue
            
            factor_aligned = factor_values.loc[common_idx]
            volume_aligned = aligned_volume.loc[common_idx]
            
            # 计算因子极值时期的成交量需求
            factor_percentiles = factor_aligned.rank(pct=True)
            
            # 极值期间 (前10%和后10%)
            extreme_mask = (factor_percentiles <= 0.1) | (factor_percentiles >= 0.9)
            normal_mask = (factor_percentiles > 0.3) & (factor_percentiles < 0.7)
            
            if extreme_mask.sum() > 0 and normal_mask.sum() > 0:
                extreme_volume = volume_aligned[extreme_mask].mean()
                normal_volume = volume_aligned[normal_mask].mean()
                
                # 流动性需求指标
                liquidity_demand = (extreme_volume - normal_volume) / (normal_volume + 1e-8)
                liquidity_score = 1 / (1 + abs(liquidity_demand))
                
                # 容量评估
                capacity_score = np.log(normal_volume + 1) / 20  # 标准化容量得分
                
                liquidity_analysis[factor] = {
                    'extreme_volume': extreme_volume,
                    'normal_volume': normal_volume,
                    'liquidity_demand': liquidity_demand,
                    'liquidity_score': liquidity_score,
                    'capacity_score': min(capacity_score, 1.0)
                }
        
        self.logger.info(f"流动性需求计算完成: {len(liquidity_analysis)} 个因子")
        return liquidity_analysis
    
    # ==================== 5. 短周期适应性分析 ====================
    
    def detect_reversal_effects(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """检测反转效应 - 短期反转特征"""
        self.logger.info("检测反转效应...")
        
        reversal_effects = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()
            
            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) < 100:
                continue
            
            factor_aligned = factor_values.loc[common_idx]
            returns_aligned = aligned_returns.loc[common_idx]
            
            # 计算因子分位数
            factor_ranks = factor_aligned.rank(pct=True)
            
            # 高因子值 vs 低因子值的收益差异
            high_mask = factor_ranks >= 0.8
            low_mask = factor_ranks <= 0.2
            
            if high_mask.sum() > 10 and low_mask.sum() > 10:
                high_returns = returns_aligned[high_mask].mean()
                low_returns = returns_aligned[low_mask].mean()
                
                # 反转效应 (低因子值 - 高因子值)
                reversal_effect = low_returns - high_returns
                
                # 反转强度 (标准化)
                returns_std = returns_aligned.std()
                reversal_strength = abs(reversal_effect) / (returns_std + 1e-8)
                
                # 反转一致性
                high_positive_rate = (returns_aligned[high_mask] > 0).mean()
                low_positive_rate = (returns_aligned[low_mask] > 0).mean()
                reversal_consistency = abs(low_positive_rate - high_positive_rate)
                
                reversal_effects[factor] = {
                    'reversal_effect': reversal_effect,
                    'reversal_strength': reversal_strength,
                    'reversal_consistency': reversal_consistency,
                    'high_return_mean': high_returns,
                    'low_return_mean': low_returns
                }
        
        self.logger.info(f"反转效应检测完成: {len(reversal_effects)} 个因子")
        return reversal_effects
    
    def analyze_momentum_persistence(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """分析动量持续性"""
        self.logger.info("分析动量持续性...")
        
        momentum_analysis = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            aligned_returns = returns.reindex(factor_values.index).dropna()
            
            common_idx = factor_values.index.intersection(aligned_returns.index)
            if len(common_idx) < 120:  # 需要更多数据计算持续性
                continue
            
            factor_aligned = factor_values.loc[common_idx]
            returns_aligned = aligned_returns.loc[common_idx]
            
            # 计算多周期动量持续性
            momentum_correlations = []
            
            for window in [5, 10, 20]:  # 不同的动量窗口
                if len(factor_aligned) > window + 10:
                    for i in range(window, len(factor_aligned) - 5):
                        current_factor = factor_aligned.iloc[i]
                        future_returns = returns_aligned.iloc[i+1:i+6].sum()  # 未来5日收益
                        
                        momentum_correlations.append((current_factor, future_returns))
            
            if len(momentum_correlations) > 20:
                factors_vals, returns_vals = zip(*momentum_correlations)
                
                try:
                    momentum_corr, momentum_p = stats.spearmanr(factors_vals, returns_vals)
                    
                    if not np.isnan(momentum_corr):
                        # 动量一致性
                        consistent_signals = sum(1 for f, r in momentum_correlations if f * r > 0)
                        momentum_consistency = consistent_signals / len(momentum_correlations)
                        
                        momentum_analysis[factor] = {
                            'momentum_persistence': momentum_corr,
                            'momentum_consistency': momentum_consistency,
                            'momentum_p_value': momentum_p,
                            'signal_count': len(momentum_correlations)
                        }
                        
                except Exception as e:
                    self.logger.debug(f"因子 {factor} 动量分析失败: {str(e)}")
                    continue
        
        self.logger.info(f"动量持续性分析完成: {len(momentum_analysis)} 个因子")
        return momentum_analysis
    
    def analyze_volatility_sensitivity(self, factors: pd.DataFrame, returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """分析波动率敏感性"""
        self.logger.info("分析波动率敏感性...")
        
        volatility_analysis = {}
        factor_cols = [col for col in factors.columns 
                      if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # 计算滚动波动率
        rolling_vol = returns.rolling(window=20).std()
        
        for factor in factor_cols:
            factor_values = factors[factor].dropna()
            
            common_idx = factor_values.index.intersection(rolling_vol.index)
            if len(common_idx) < 100:
                continue
            
            factor_aligned = factor_values.loc[common_idx]
            vol_aligned = rolling_vol.loc[common_idx].dropna()
            
            # 再次对齐
            final_idx = factor_aligned.index.intersection(vol_aligned.index)
            if len(final_idx) < 50:
                continue
            
            factor_final = factor_aligned.loc[final_idx]
            vol_final = vol_aligned.loc[final_idx]
            
            # 分析因子在不同波动率环境下的表现
            vol_percentiles = vol_final.rank(pct=True)
            
            high_vol_mask = vol_percentiles >= 0.7
            low_vol_mask = vol_percentiles <= 0.3
            
            if high_vol_mask.sum() > 10 and low_vol_mask.sum() > 10:
                high_vol_factor = factor_final[high_vol_mask].std()
                low_vol_factor = factor_final[low_vol_mask].std()
                
                # 波动率敏感性
                vol_sensitivity = (high_vol_factor - low_vol_factor) / (low_vol_factor + 1e-8)
                
                # 稳定性得分 (波动率敏感性越低越好)
                stability_score = 1 / (1 + abs(vol_sensitivity))
                
                volatility_analysis[factor] = {
                    'volatility_sensitivity': vol_sensitivity,
                    'stability_score': stability_score,
                    'high_vol_std': high_vol_factor,
                    'low_vol_std': low_vol_factor
                }
        
        self.logger.info(f"波动率敏感性分析完成: {len(volatility_analysis)} 个因子")
        return volatility_analysis
    
    # ==================== 统计显著性检验 ====================
    
    def benjamini_hochberg_correction(self, p_values: Dict[str, float], 
                                    alpha: float = None) -> Dict[str, float]:
        """标准Benjamini-Hochberg FDR校正"""
        if alpha is None:
            alpha = self.config.alpha_level
        
        if not p_values:
            return {}
        
        # 转换为数组
        factors = list(p_values.keys())
        p_vals = np.array([p_values[factor] for factor in factors])
        
        # 按p值排序
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        sorted_factors = [factors[i] for i in sorted_indices]
        
        # 标准BH程序
        n_tests = len(p_vals)
        corrected_p = {}
        
        for i, (factor, p_val) in enumerate(zip(sorted_factors, sorted_p)):
            # BH校正公式: p_corrected = p * n / (i + 1)
            corrected_p_val = min(p_val * n_tests / (i + 1), 1.0)
            corrected_p[factor] = corrected_p_val
        
        return corrected_p
    
    def bonferroni_correction(self, p_values: Dict[str, float]) -> Dict[str, float]:
        """Bonferroni校正"""
        if not p_values:
            return {}
        
        n_tests = len(p_values)
        corrected_p = {}
        
        for factor, p_val in p_values.items():
            corrected_p[factor] = min(p_val * n_tests, 1.0)
        
        return corrected_p
    
    # ==================== 综合评分系统 ====================
    
    def calculate_comprehensive_scores(self, all_metrics: Dict[str, Dict]) -> Dict[str, FactorMetrics]:
        """计算综合评分 - 5维度加权评分"""
        self.logger.info("计算综合评分...")
        
        comprehensive_results = {}
        
        # 获取所有因子名称
        all_factors = set()
        for metric_dict in all_metrics.values():
            if isinstance(metric_dict, dict):
                all_factors.update(metric_dict.keys())
        
        for factor in all_factors:
            metrics = FactorMetrics(name=factor)
            
            # 1. 预测能力评分 (35%)
            predictive_score = 0.0
            if 'multi_horizon_ic' in all_metrics and factor in all_metrics['multi_horizon_ic']:
                ic_data = all_metrics['multi_horizon_ic'][factor]
                
                # 提取各周期IC
                metrics.ic_1d = ic_data.get('ic_1d', 0.0)
                metrics.ic_3d = ic_data.get('ic_3d', 0.0)
                metrics.ic_5d = ic_data.get('ic_5d', 0.0)
                metrics.ic_10d = ic_data.get('ic_10d', 0.0)
                metrics.ic_20d = ic_data.get('ic_20d', 0.0)
                
                # 计算平均IC和IR
                ic_values = [abs(ic_data.get(f'ic_{h}d', 0.0)) for h in self.config.ic_horizons]
                ic_values = [ic for ic in ic_values if ic != 0.0]
                
                if ic_values:
                    metrics.ic_mean = np.mean(ic_values)
                    metrics.ic_std = np.std(ic_values) if len(ic_values) > 1 else 0.1
                    metrics.ic_ir = metrics.ic_mean / (metrics.ic_std + 1e-8)
                    
                    # 预测能力得分
                    predictive_score = min(metrics.ic_mean * 10, 1.0)  # 标准化到[0,1]
            
            if 'ic_decay' in all_metrics and factor in all_metrics['ic_decay']:
                decay_data = all_metrics['ic_decay'][factor]
                metrics.ic_decay_rate = decay_data.get('decay_rate', 0.0)
                metrics.ic_longevity = decay_data.get('ic_longevity', 0)
                
                # 衰减惩罚
                decay_penalty = abs(metrics.ic_decay_rate) * 0.1
                predictive_score = max(0, predictive_score - decay_penalty)
            
            metrics.predictive_score = predictive_score
            
            # 2. 稳定性评分 (25%)
            stability_score = 0.0
            if 'rolling_ic' in all_metrics and factor in all_metrics['rolling_ic']:
                rolling_data = all_metrics['rolling_ic'][factor]
                metrics.rolling_ic_mean = rolling_data.get('rolling_ic_mean', 0.0)
                metrics.rolling_ic_std = rolling_data.get('rolling_ic_std', 0.0)
                metrics.rolling_ic_stability = rolling_data.get('rolling_ic_stability', 0.0)
                metrics.ic_consistency = rolling_data.get('ic_consistency', 0.0)
                
                stability_score = (metrics.rolling_ic_stability + metrics.ic_consistency) / 2
            
            if 'cross_section_stability' in all_metrics and factor in all_metrics['cross_section_stability']:
                cs_data = all_metrics['cross_section_stability'][factor]
                metrics.cross_section_stability = cs_data.get('cross_section_stability', 0.0)
                
                # 综合稳定性
                stability_score = (stability_score + metrics.cross_section_stability) / 2
            
            metrics.stability_score = stability_score
            
            # 3. 独立性评分 (20%)
            independence_score = 1.0  # 默认满分
            if 'vif_scores' in all_metrics and factor in all_metrics['vif_scores']:
                metrics.vif_score = all_metrics['vif_scores'][factor]
                vif_penalty = min(metrics.vif_score / self.config.vif_threshold, 2.0)
                independence_score *= (1 / (1 + vif_penalty))
            
            if 'correlation_matrix' in all_metrics:
                corr_matrix = all_metrics['correlation_matrix']
                if factor in corr_matrix.columns:
                    factor_corrs = corr_matrix[factor].drop(factor, errors='ignore')
                    if len(factor_corrs) > 0:
                        metrics.correlation_max = factor_corrs.abs().max()
                        corr_penalty = max(0, metrics.correlation_max - 0.5) * 2
                        independence_score *= (1 - corr_penalty)
            
            if 'information_increment' in all_metrics and factor in all_metrics['information_increment']:
                metrics.information_increment = all_metrics['information_increment'][factor]
                # 信息增量奖励
                info_bonus = max(0, metrics.information_increment) * 5
                independence_score = min(independence_score + info_bonus, 1.0)
            
            metrics.independence_score = max(0, independence_score)
            
            # 4. 实用性评分 (15%)
            practicality_score = 1.0
            if 'trading_costs' in all_metrics and factor in all_metrics['trading_costs']:
                cost_data = all_metrics['trading_costs'][factor]
                metrics.turnover_rate = cost_data.get('turnover_rate', 0.0)
                metrics.transaction_cost = cost_data.get('total_cost', 0.0)
                metrics.cost_efficiency = cost_data.get('cost_efficiency', 0.0)
                
                practicality_score = metrics.cost_efficiency
            
            if 'liquidity_requirements' in all_metrics and factor in all_metrics['liquidity_requirements']:
                liq_data = all_metrics['liquidity_requirements'][factor]
                metrics.liquidity_demand = liq_data.get('liquidity_demand', 0.0)
                metrics.capacity_score = liq_data.get('capacity_score', 0.0)
                
                # 综合实用性
                practicality_score = (practicality_score + metrics.capacity_score) / 2
            
            metrics.practicality_score = practicality_score
            
            # 5. 短周期适应性评分 (5%)
            adaptability_score = 0.5  # 默认中性
            if 'reversal_effects' in all_metrics and factor in all_metrics['reversal_effects']:
                rev_data = all_metrics['reversal_effects'][factor]
                metrics.reversal_effect = rev_data.get('reversal_effect', 0.0)
                reversal_strength = rev_data.get('reversal_strength', 0.0)
                
                # 适度的反转效应是好的
                adaptability_score += min(reversal_strength * 0.5, 0.3)
            
            if 'momentum_persistence' in all_metrics and factor in all_metrics['momentum_persistence']:
                mom_data = all_metrics['momentum_persistence'][factor]
                metrics.momentum_persistence = mom_data.get('momentum_persistence', 0.0)
                
                # 动量持续性奖励
                adaptability_score += abs(metrics.momentum_persistence) * 0.2
            
            if 'volatility_sensitivity' in all_metrics and factor in all_metrics['volatility_sensitivity']:
                vol_data = all_metrics['volatility_sensitivity'][factor]
                vol_stability = vol_data.get('stability_score', 0.0)
                
                # 波动率稳定性奖励
                adaptability_score = (adaptability_score + vol_stability) / 2
            
            metrics.adaptability_score = min(adaptability_score, 1.0)
            
            # 综合评分计算
            weights = getattr(self.config, 'weights', {
                'predictive_power': 0.35,
                'stability': 0.25,
                'independence': 0.20,
                'practicality': 0.10,
                'short_term_fitness': 0.10
            })
            
            metrics.comprehensive_score = (
                metrics.predictive_score * weights.get('predictive_power', 0.35) +
                metrics.stability_score * weights.get('stability', 0.25) +
                metrics.independence_score * weights.get('independence', 0.20) +
                metrics.practicality_score * weights.get('practicality', 0.10) +
                metrics.adaptability_score * weights.get('short_term_fitness', 0.10)
            )
            
            # 统计显著性
            if 'p_values' in all_metrics and factor in all_metrics['p_values']:
                metrics.p_value = all_metrics['p_values'][factor]
            
            if 'corrected_p_values' in all_metrics and factor in all_metrics['corrected_p_values']:
                metrics.corrected_p_value = all_metrics['corrected_p_values'][factor]
                metrics.is_significant = metrics.corrected_p_value < self.config.alpha_level
            
            comprehensive_results[factor] = metrics
        
        self.logger.info(f"综合评分计算完成: {len(comprehensive_results)} 个因子")
        return comprehensive_results
    
    # ==================== 主筛选函数 ====================
    
    def screen_factors_comprehensive(self, symbol: str, timeframe: str = "60min") -> Dict[str, FactorMetrics]:
        """主筛选函数 - 5维度综合筛选"""
        start_time = time.time()
        self.logger.info(f"开始5维度因子筛选: {symbol} {timeframe}")
        
        try:
            # 1. 数据加载
            self.logger.info("步骤1: 数据加载...")
            factors = self.load_factors(symbol, timeframe)
            price_data = self.load_price_data(symbol, timeframe)  # 传递timeframe参数
            
            # 2. 数据预处理和对齐
            self.logger.info("步骤2: 数据预处理...")
            close_prices = price_data['close']
            returns = close_prices.pct_change().shift(-1)  # 次日收益
            
            # 添加诊断日志 - 关键修复
            self.logger.info(f"数据对齐前诊断:")
            self.logger.info(f"  因子数据: {len(factors)} 行, 时间 {factors.index.min()} 到 {factors.index.max()}")
            self.logger.info(f"  价格数据: {len(close_prices)} 行, 时间 {close_prices.index.min()} 到 {close_prices.index.max()}")
            
            # 时间对齐
            common_index = factors.index.intersection(close_prices.index)
            
            # 如果对齐失败，尝试诊断并修复
            if len(common_index) == 0:
                self.logger.error("数据对齐失败！尝试诊断...")
                self.logger.error(f"  因子前5个时间: {factors.index[:5].tolist()}")
                self.logger.error(f"  价格前5个时间: {close_prices.index[:5].tolist()}")
                
                # 对于daily数据，尝试标准化到日期
                if timeframe == 'daily':
                    self.logger.info("检测到daily时间框架，尝试标准化到日期...")
                    factors.index = factors.index.normalize()
                    close_prices.index = close_prices.index.normalize()
                    returns.index = returns.index.normalize()
                    common_index = factors.index.intersection(close_prices.index)
                    self.logger.info(f"标准化后共同时间点: {len(common_index)}")
            
            if len(common_index) < self.config.min_sample_size:
                raise ValueError(f"数据对齐后样本量不足: {len(common_index)} < {self.config.min_sample_size}")
            
            factors_aligned = factors.loc[common_index]
            returns_aligned = returns.loc[common_index]
            prices_aligned = price_data.loc[common_index]
            
            self.logger.info(f"数据对齐完成: 样本量={len(common_index)}, 因子数={len(factors_aligned.columns)}")
            
            # 3. 5维度分析
            all_metrics = {}
            
            # 3.1 预测能力分析
            self.logger.info("步骤3.1: 预测能力分析...")
            all_metrics['multi_horizon_ic'] = self.calculate_multi_horizon_ic(factors_aligned, returns_aligned)
            all_metrics['ic_decay'] = self.analyze_ic_decay(all_metrics['multi_horizon_ic'])
            
            # 3.2 稳定性分析
            self.logger.info("步骤3.2: 稳定性分析...")
            all_metrics['rolling_ic'] = self.calculate_rolling_ic(factors_aligned, returns_aligned)
            all_metrics['cross_section_stability'] = self.calculate_cross_sectional_stability(factors_aligned)
            
            # 3.3 独立性分析
            self.logger.info("步骤3.3: 独立性分析...")
            all_metrics['vif_scores'] = self.calculate_vif_scores(factors_aligned)
            all_metrics['correlation_matrix'] = self.calculate_factor_correlation_matrix(factors_aligned)
            all_metrics['information_increment'] = self.calculate_information_increment(factors_aligned, returns_aligned)
            
            # 3.4 实用性分析
            self.logger.info("步骤3.4: 实用性分析...")
            all_metrics['trading_costs'] = self.calculate_trading_costs(factors_aligned, prices_aligned)
            all_metrics['liquidity_requirements'] = self.calculate_liquidity_requirements(factors_aligned, prices_aligned['volume'])
            
            # 3.5 短周期适应性分析
            self.logger.info("步骤3.5: 短周期适应性分析...")
            all_metrics['reversal_effects'] = self.detect_reversal_effects(factors_aligned, returns_aligned)
            all_metrics['momentum_persistence'] = self.analyze_momentum_persistence(factors_aligned, returns_aligned)
            all_metrics['volatility_sensitivity'] = self.analyze_volatility_sensitivity(factors_aligned, returns_aligned)
            
            # 4. 统计显著性检验
            self.logger.info("步骤4: 统计显著性检验...")
            
            # 收集p值
            p_values = {}
            for factor, ic_data in all_metrics['multi_horizon_ic'].items():
                # 使用1日IC的p值作为主要显著性指标
                p_values[factor] = ic_data.get('p_value_1d', 1.0)
            
            all_metrics['p_values'] = p_values
            
            # FDR校正
            if self.config.fdr_method == "benjamini_hochberg":
                corrected_p = self.benjamini_hochberg_correction(p_values)
            else:
                corrected_p = self.bonferroni_correction(p_values)
            
            all_metrics['corrected_p_values'] = corrected_p
            
            # 5. 综合评分
            self.logger.info("步骤5: 综合评分...")
            comprehensive_results = self.calculate_comprehensive_scores(all_metrics)
            
            # 6. 性能统计
            total_time = time.time() - start_time
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = current_memory - self.start_memory

            self.logger.info(f"5维度筛选完成:")
            self.logger.info(f"  - 总耗时: {total_time:.2f}秒")
            self.logger.info(f"  - 内存使用: {memory_used:.1f}MB")
            self.logger.info(f"  - 因子总数: {len(comprehensive_results)}")

            # 统计各维度表现
            significant_count = sum(1 for m in comprehensive_results.values() if m.is_significant)
            high_score_count = sum(1 for m in comprehensive_results.values() if m.comprehensive_score > 0.7)

            self.logger.info(f"  - 显著因子: {significant_count}")
            self.logger.info(f"  - 高分因子: {high_score_count}")

            # 7. 收集筛选统计信息
            screening_stats = {
                'total_factors': len(comprehensive_results),
                'significant_factors': significant_count,
                'high_score_factors': high_score_count,
                'total_time': total_time,
                'memory_used_mb': memory_used,
                'sample_size': len(common_index) if 'common_index' in locals() else 0,
                'factors_aligned': len(factors_aligned.columns) if 'factors_aligned' in locals() else 0,
                'data_alignment_successful': len(common_index) > 0 if 'common_index' in locals() else False,
                'screening_timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe
            }

            # 8. 收集数据质量信息
            data_quality_info = {
                'factor_data_shape': factors.shape if 'factors' in locals() else None,
                'price_data_shape': price_data.shape if 'price_data' in locals() else None,
                'aligned_data_shape': factors_aligned.shape if 'factors_aligned' in locals() else None,
                'data_overlap_count': len(common_index) if 'common_index' in locals() else 0,
                'factor_data_range': {
                    'start': factors.index.min().isoformat() if 'factors' in locals() and len(factors) > 0 else None,
                    'end': factors.index.max().isoformat() if 'factors' in locals() and len(factors) > 0 else None
                },
                'price_data_range': {
                    'start': price_data.index.min().isoformat() if 'price_data' in locals() and len(price_data) > 0 else None,
                    'end': price_data.index.max().isoformat() if 'price_data' in locals() and len(price_data) > 0 else None
                },
                'alignment_success_rate': len(common_index) / min(len(factors), len(price_data)) if 'factors' in locals() and 'price_data' in locals() else 0.0
            }

            # 9. 保存完整筛选信息 - 使用增强版结果管理器
            try:
                if self.result_manager is not None:
                    # 使用新的增强版结果管理器创建时间戳文件夹
                    session_id = self.result_manager.create_screening_session(
                        symbol=symbol,
                        timeframe=timeframe,
                        results=comprehensive_results,
                        screening_stats=screening_stats,
                        config=self.config,
                        data_quality_info=data_quality_info
                    )
                    
                    self.logger.info(f"✅ 完整筛选会话已创建: {session_id}")
                    screening_stats['session_id'] = session_id
                else:
                    self.logger.info("使用传统存储方式")
                
                # 保持向后兼容 - 仍然保存传统格式
                try:
                    saved_files = self.save_comprehensive_screening_info(
                        results=comprehensive_results,
                        symbol=symbol,
                        timeframe=timeframe,
                        screening_stats=screening_stats,
                        data_quality_info=data_quality_info
                    )
                    screening_stats['legacy_files'] = saved_files
                except Exception as legacy_error:
                    self.logger.warning(f"传统格式保存失败: {legacy_error}")
                    
            except Exception as e:
                self.logger.error(f"保存完整筛选信息失败: {str(e)}")
                screening_stats['save_error'] = str(e)

            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"因子筛选失败: {str(e)}")
            raise
    
    def generate_screening_report(self, results: Dict[str, FactorMetrics], 
                                output_path: str = None, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """生成筛选报告"""
        self.logger.info("生成筛选报告...")
        
        if not results:
            self.logger.warning("没有筛选结果，无法生成报告")
            return pd.DataFrame()
        
        # 转换为DataFrame
        report_data = []
        for factor_name, metrics in results.items():
            row = {
                'Factor': factor_name,
                'Comprehensive_Score': metrics.comprehensive_score,
                'Predictive_Score': metrics.predictive_score,
                'Stability_Score': metrics.stability_score,
                'Independence_Score': metrics.independence_score,
                'Practicality_Score': metrics.practicality_score,
                'Adaptability_Score': metrics.adaptability_score,
                
                'IC_Mean': metrics.ic_mean,
                'IC_IR': metrics.ic_ir,
                'IC_1d': metrics.ic_1d,
                'IC_5d': metrics.ic_5d,
                'IC_10d': metrics.ic_10d,
                
                'Rolling_IC_Stability': metrics.rolling_ic_stability,
                'IC_Consistency': metrics.ic_consistency,
                
                'VIF_Score': metrics.vif_score,
                'Max_Correlation': metrics.correlation_max,
                'Info_Increment': metrics.information_increment,
                
                'Turnover_Rate': metrics.turnover_rate,
                'Transaction_Cost': metrics.transaction_cost,
                'Cost_Efficiency': metrics.cost_efficiency,
                
                'P_Value': metrics.p_value,
                'Corrected_P_Value': metrics.corrected_p_value,
                'Is_Significant': metrics.is_significant,
                
                'Sample_Size': metrics.sample_size
            }
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        report_df = report_df.sort_values('Comprehensive_Score', ascending=False)
        
        # 保存报告（包含时间框架标识）
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # 使用传入的参数或从results中提取symbol和timeframe信息
            symbol_info = symbol or results.get('symbol', 'unknown')
            timeframe_info = timeframe or results.get('timeframe', 'unknown')
            # 使用专门的筛选报告目录
            output_path = self.screening_results_dir / f"screening_report_{symbol_info}_{timeframe_info}_{timestamp}.csv"
        
        # 确保路径是字符串格式，避免pandas Path._flavour问题
        output_path_str = str(output_path)
        report_df.to_csv(output_path_str, index=False)
        self.logger.info(f"筛选报告已保存: {output_path}")
        
        return report_df

    def save_comprehensive_screening_info(self, results: Dict[str, FactorMetrics],
                                       symbol: str, timeframe: str,
                                       screening_stats: Dict,
                                       data_quality_info: Dict = None):
        """保存完整的筛选信息，包括多个格式的报告"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"screening_{symbol}_{timeframe}_{timestamp}"

        # 1. 保存详细的CSV报告
        csv_path = self.screening_results_dir / f"{base_filename}_detailed_report.csv"
        report_df = self.generate_screening_report(results, str(csv_path), symbol, timeframe)

        # 2. 保存筛选过程统计信息
        stats_path = self.screening_results_dir / f"{base_filename}_screening_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(screening_stats, f, indent=2, ensure_ascii=False, default=str)

        # 3. 保存顶级因子摘要
        summary_path = self.screening_results_dir / f"{base_filename}_top_factors_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 因子筛选摘要报告 ===\n")
            f.write(f"股票代码: {symbol}\n")
            f.write(f"时间框架: {timeframe}\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n=== 筛选统计 ===\n")
            f.write(f"总因子数: {screening_stats.get('total_factors', 0)}\n")
            f.write(f"显著因子: {screening_stats.get('significant_factors', 0)}\n")
            f.write(f"高分因子: {screening_stats.get('high_score_factors', 0)}\n")
            f.write(f"总耗时: {screening_stats.get('total_time', 0):.2f}秒\n")
            f.write(f"内存使用: {screening_stats.get('memory_used_mb', 0):.1f}MB\n")

            # 获取前10名因子
            top_factors = self.get_top_factors(results, top_n=10, min_score=0.0, require_significant=False)
            f.write(f"\n=== 前10名顶级因子 ===\n")
            for i, factor in enumerate(top_factors, 1):
                f.write(f"{i:2d}. {factor.name:<25} 综合得分: {factor.comprehensive_score:.3f} ")
                f.write(f"预测能力: {factor.predictive_score:.3f} 显著性: {'✓' if factor.is_significant else '✗'}\n")

        # 4. 保存数据质量报告
        if data_quality_info:
            quality_path = self.screening_results_dir / f"{base_filename}_data_quality.json"
            with open(quality_path, 'w', encoding='utf-8') as f:
                json.dump(data_quality_info, f, indent=2, ensure_ascii=False, default=str)

        # 5. 保存配置参数记录
        config_path = self.screening_results_dir / f"{base_filename}_config.yaml"
        config_dict = {
            'screening_config': asdict(self.config),
            'execution_info': {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'data_root': str(self.data_root),
                'screening_results_dir': str(self.screening_results_dir)
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

        # 6. 创建一个主索引文件
        index_path = self.screening_results_dir / f"{base_filename}_index.txt"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"因子筛选完整报告索引\n")
            f.write(f"========================\n\n")
            f.write(f"基础信息:\n")
            f.write(f"  股票代码: {symbol}\n")
            f.write(f"  时间框架: {timeframe}\n")
            f.write(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"包含文件:\n")
            f.write(f"  1. {csv_path.name} - 详细因子数据 (CSV格式)\n")
            f.write(f"  2. {stats_path.name} - 筛选过程统计 (JSON格式)\n")
            f.write(f"  3. {summary_path.name} - 顶级因子摘要 (TXT格式)\n")
            if data_quality_info:
                f.write(f"  4. {quality_path.name} - 数据质量报告 (JSON格式)\n")
            f.write(f"  5. {config_path.name} - 配置参数记录 (YAML格式)\n")
            f.write(f"  6. {index_path.name} - 本索引文件\n\n")
            f.write(f"使用说明:\n")
            f.write(f"  - 查看顶级因子: 阅读 {summary_path.name}\n")
            f.write(f"  - 详细数据分析: 打开 {csv_path.name} 使用Excel或pandas\n")
            f.write(f"  - 筛选过程详情: 查看 {stats_path.name}\n")
            f.write(f"  - 配置参数参考: 查看 {config_path.name}\n")

        self.logger.info(f"完整筛选信息已保存到: {self.screening_results_dir}")
        self.logger.info(f"主索引文件: {index_path}")

        return {
            'csv_report': str(csv_path),
            'stats_json': str(stats_path),
            'summary_txt': str(summary_path),
            'data_quality_json': str(quality_path) if data_quality_info else None,
            'config_yaml': str(config_path),
            'index_txt': str(index_path)
        }

    def get_top_factors(self, results: Dict[str, FactorMetrics], 
                       top_n: int = 20, 
                       min_score: float = 0.5,
                       require_significant: bool = True) -> List[FactorMetrics]:
        """获取顶级因子"""
        
        # 筛选条件
        filtered_results = []
        for metrics in results.values():
            if metrics.comprehensive_score >= min_score:
                if not require_significant or metrics.is_significant:
                    filtered_results.append(metrics)
        
        # 按综合得分排序
        filtered_results.sort(key=lambda x: x.comprehensive_score, reverse=True)
        
        return filtered_results[:top_n]

def main():
    """主函数 - 使用示例"""
    # 初始化配置
    config = ScreeningConfig(
        ic_horizons=[1, 3, 5, 10, 20],
        min_sample_size=100,
        alpha_level=0.05,
        fdr_method="benjamini_hochberg",
        min_ic_threshold=0.02,
        min_ir_threshold=0.5
    )
    
    # 初始化筛选器
    screener = ProfessionalFactorScreener(
        "/Users/zhangshenshen/深度量化0927/factor_system/output",
        config=config
    )
    
    # 执行筛选
    symbol = "0700.HK"
    timeframe = "60min"
    
    print(f"开始专业级因子筛选: {symbol} {timeframe}")
    print("="*80)
    
    try:
        # 5维度综合筛选
        results = screener.screen_factors_comprehensive(symbol, timeframe)
        
        # 生成报告
        report_df = screener.generate_screening_report(results)
        
        # 获取顶级因子
        top_factors = screener.get_top_factors(results, top_n=10, min_score=0.6)
        
        # 输出结果
        print("\n5维度因子筛选结果:")
        print("="*80)
        print(f"总因子数量: {len(results)}")
        print(f"显著因子数量: {sum(1 for m in results.values() if m.is_significant)}")
        print(f"高分因子数量 (>0.6): {sum(1 for m in results.values() if m.comprehensive_score > 0.6)}")
        print(f"顶级因子数量: {len(top_factors)}")
        
        print(f"\n前10名顶级因子:")
        print("-"*120)
        print(f"{'排名':<4} {'因子名称':<20} {'综合得分':<8} {'预测':<6} {'稳定':<6} {'独立':<6} {'实用':<6} {'适应':<6} {'IC均值':<8} {'显著性':<6}")
        print("-"*120)
        
        for i, metrics in enumerate(top_factors[:10]):
            significance = "***" if metrics.corrected_p_value < 0.001 else "**" if metrics.corrected_p_value < 0.01 else "*" if metrics.corrected_p_value < 0.05 else ""
            
            print(f"{i+1:<4} {metrics.name:<20} {metrics.comprehensive_score:.3f}    "
                  f"{metrics.predictive_score:.3f}  {metrics.stability_score:.3f}  "
                  f"{metrics.independence_score:.3f}  {metrics.practicality_score:.3f}  "
                  f"{metrics.adaptability_score:.3f}  {metrics.ic_mean:+.4f}  {significance:<6}")
        
        print(f"\n报告文件: {report_df}")
        
    except Exception as e:
        print(f"筛选失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
