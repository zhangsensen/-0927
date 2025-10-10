#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量因子筛选引擎
支持多市场、多股票、多时间框架的批量因子筛选
"""

import gc
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

from config_manager import ConfigManager, ScreeningConfig
from data_loader_patch import patch_data_loader
from professional_factor_screener import ProfessionalFactorScreener
from utils.market_utils import discover_stocks, format_symbol_for_display
from utils.timeframe_utils import SUPPORTED_TIMEFRAMES, sort_timeframes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchScreeningStats:
    """批量筛选统计"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_factors_found: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    def record_success(self, num_factors: int):
        """记录成功"""
        self.completed_tasks += 1
        self.total_factors_found += num_factors
    
    def record_failure(self):
        """记录失败"""
        self.failed_tasks += 1
    
    def finalize(self):
        """完成统计"""
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """已用时间（秒）"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100


class BatchFactorScreener:
    """批量因子筛选引擎"""
    
    def __init__(self, config_file: str = None):
        """
        初始化批量筛选器
        
        Args:
            config_file: 配置文件路径，默认使用 batch_screening_config.yaml
        """
        self.logger = logger
        
        # 加载配置
        if config_file is None:
            config_file = "configs/batch_screening_config.yaml"
        
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
        
        # 读取配置
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.batch_config = yaml.safe_load(f)
        
        # 批量处理参数
        self.max_parallel = self.batch_config.get('max_parallel_stocks', 4)
        self.batch_size = self.batch_config.get('batch_size', 10)
        self.continue_on_error = self.batch_config.get('continue_on_error', True)
        
        # 输出目录（需要在_create_screening_config之前设置）
        self.output_root = Path(self.batch_config.get(
            'output_root',
            './batch_screening_results'
        ))
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 转换为ScreeningConfig
        self.screening_config = self._create_screening_config()
        
        # 统计信息
        self.stats = BatchScreeningStats()
        self.results = {}
        self.failed_tasks = []
        
        self.logger.info(f"批量筛选器初始化完成")
        self.logger.info(f"配置文件: {self.config_file}")
        self.logger.info(f"输出目录: {self.output_root}")
    
    def _create_screening_config(self) -> ScreeningConfig:
        """从批量配置创建筛选配置"""
        # 确保使用绝对路径
        data_root = self.batch_config.get('data_root', '../factor_output')
        raw_data_root = self.batch_config.get('raw_data_root', '../raw')
        
        config_dict = {
            'data_root': str(Path(data_root).absolute()),
            'raw_data_root': str(Path(raw_data_root).absolute()),
            'output_root': str(self.output_root),
            'ic_horizons': self.batch_config.get('ic_horizons', [1, 3, 5, 10, 20]),
            'min_sample_size': self.batch_config.get('min_sample_size', 100),
            'alpha_level': self.batch_config.get('alpha_level', 0.05),
            'fdr_method': self.batch_config.get('fdr_method', 'benjamini_hochberg'),
            'rolling_window': self.batch_config.get('rolling_window', 60),
            'min_ic_threshold': self.batch_config.get('min_ic_threshold', 0.015),
            'min_ir_threshold': self.batch_config.get('min_ir_threshold', 0.35),
            'vif_threshold': self.batch_config.get('vif_threshold', 5.0),
            'correlation_threshold': self.batch_config.get('correlation_threshold', 0.8),
            'weights': self.batch_config.get('weights', {}),
            'max_workers': self.max_parallel,
            'enable_parallel': self.batch_config.get('enable_parallel', True),
            'log_level': self.batch_config.get('log_level', 'INFO'),
        }
        
        return ScreeningConfig(**config_dict)
    
    def discover_all_stocks(self, market: str = None) -> Dict[str, List[str]]:
        """
        自动发现所有可用股票
        
        Args:
            market: 指定市场 ('HK', 'US') 或 None (全部市场)
            
        Returns:
            {'HK': ['0005HK', ...], 'US': ['AAPLUS', ...]}
        """
        data_root = Path(self.screening_config.data_root)
        stocks = discover_stocks(data_root, market)
        
        self.logger.info(f"发现股票数量:")
        for mkt, symbols in stocks.items():
            self.logger.info(f"  {mkt}: {len(symbols)} 只")
        
        return stocks
    
    def _screen_single_task(
        self,
        symbol: str,
        timeframe: str
    ) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        筛选单个任务（用于并行处理）
        
        Returns:
            (success, result_dict, error_message)
        """
        try:
            # 创建筛选器实例
            screener = ProfessionalFactorScreener(self.screening_config)
            
            # 应用数据加载器补丁
            patch_data_loader(screener)
            
            # 执行筛选
            factor_metrics = screener.screen_factors_comprehensive(
                symbol=symbol,
                timeframe=timeframe
            )
            
            # 转换结果
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'num_factors': len(factor_metrics),
                'metrics': {name: vars(metrics) for name, metrics in factor_metrics.items()}
            }
            
            return True, result, None
            
        except Exception as e:
            error_msg = f"{symbol} {timeframe}: {str(e)}"
            self.logger.error(f"❌ 筛选失败 - {error_msg}")
            return False, None, error_msg
    
    def screen_single_stock(
        self,
        symbol: str,
        timeframes: List[str] = None
    ) -> Dict[str, Any]:
        """
        筛选单只股票的多个时间框架
        
        Args:
            symbol: 股票代码
            timeframes: 时间框架列表，None表示使用配置中的全部
            
        Returns:
            {
                '5min': FactorMetrics(...),
                '15min': FactorMetrics(...),
                ...
            }
        """
        if timeframes is None:
            timeframes = self.batch_config.get('timeframes', ['5min'])
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"开始筛选: {format_symbol_for_display(symbol)}")
        self.logger.info(f"时间框架: {timeframes}")
        self.logger.info(f"{'='*60}")
        
        results = {}
        
        for tf in timeframes:
            try:
                self.logger.info(f"\n处理 {symbol} {tf}...")
                
                # 创建筛选器
                screener = ProfessionalFactorScreener(self.screening_config)
                
                # 应用补丁
                patch_data_loader(screener)
                
                # 执行筛选
                factor_metrics = screener.screen_factors_comprehensive(
                    symbol=symbol,
                    timeframe=tf
                )
                
                results[tf] = factor_metrics
                
                self.logger.info(
                    f"✅ {symbol} {tf}: 找到 {len(factor_metrics)} 个优质因子"
                )
                
            except Exception as e:
                self.logger.error(f"❌ {symbol} {tf} 筛选失败: {e}")
                if not self.continue_on_error:
                    raise
        
        return results
    
    def screen_multiple_stocks(
        self,
        symbols: List[str],
        timeframes: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量筛选多只股票（串行模式，稳定可靠）
        
        Args:
            symbols: 股票代码列表
            timeframes: 时间框架列表
            
        Returns:
            {
                '0700.HK': {'5min': metrics, '15min': metrics, ...},
                'AAPL.US': {...},
                ...
            }
        
        Note:
            如需高性能并行处理，请使用 batch_screen_all_stocks_parallel.py
        """
        if timeframes is None:
            timeframes = self.batch_config.get('timeframes', ['5min'])
        
        # 计算总任务数
        self.stats.total_tasks = len(symbols) * len(timeframes)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"批量筛选任务（串行模式）")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"股票数量: {len(symbols)}")
        self.logger.info(f"时间框架: {timeframes}")
        self.logger.info(f"总任务数: {self.stats.total_tasks}")
        self.logger.info(f"{'='*60}\n")
        
        results = {}
        
        # 串行处理（稳定可靠）
        for symbol in tqdm(symbols, desc="筛选股票"):
            formatted_symbol = format_symbol_for_display(symbol)
            try:
                symbol_results = self.screen_single_stock(symbol, timeframes)
                results[formatted_symbol] = symbol_results
                
                for tf in timeframes:
                    if tf in symbol_results:
                        self.stats.record_success(len(symbol_results[tf]))
                    else:
                        self.stats.record_failure()
                
            except Exception as e:
                self.logger.error(f"❌ {formatted_symbol} 筛选失败: {e}")
                for tf in timeframes:
                    self.stats.record_failure()
                
                if not self.continue_on_error:
                    raise
            
            # 定期垃圾回收
            gc.collect()
        
        self.stats.finalize()
        return results
    
    def screen_market(
        self,
        market: str,
        timeframes: List[str] = None,
        limit: int = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        筛选指定市场的所有股票
        
        Args:
            market: 市场代码 ('HK' 或 'US')
            timeframes: 时间框架列表
            limit: 限制处理的股票数量（用于测试）
            
        Returns:
            筛选结果字典
        """
        # 发现股票
        all_stocks = self.discover_all_stocks(market=market)
        
        if market not in all_stocks:
            raise ValueError(f"市场不存在: {market}")
        
        symbols = all_stocks[market]
        
        if limit:
            symbols = symbols[:limit]
            self.logger.info(f"限制处理前 {limit} 只股票")
        
        return self.screen_multiple_stocks(symbols, timeframes)
    
    def screen_all_markets(
        self,
        timeframes: List[str] = None,
        markets: List[str] = None,
        limit_per_market: int = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        筛选所有市场的所有股票
        
        Args:
            timeframes: 时间框架列表
            markets: 市场列表，None表示全部市场
            limit_per_market: 每个市场限制的股票数量
            
        Returns:
            {
                'HK': {symbol: {timeframe: metrics, ...}, ...},
                'US': {...}
            }
        """
        if markets is None:
            markets = self.batch_config.get('markets', ['HK', 'US'])
        
        if timeframes is None:
            timeframes = self.batch_config.get('timeframes', ['5min'])
        
        results = {}
        
        for market in markets:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"处理 {market} 市场")
            self.logger.info(f"{'='*60}")
            
            try:
                market_results = self.screen_market(
                    market=market,
                    timeframes=timeframes,
                    limit=limit_per_market
                )
                results[market] = market_results
                
            except Exception as e:
                self.logger.error(f"❌ {market} 市场处理失败: {e}")
                if not self.continue_on_error:
                    raise
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        prefix: str = "batch_screening"
    ) -> Path:
        """
        保存批量筛选结果
        
        Args:
            results: 筛选结果
            prefix: 文件名前缀
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_root / f"{prefix}_{timestamp}.json"
        
        # 准备可序列化的结果
        serializable_results = {}
        for symbol, tf_results in results.items():
            serializable_results[symbol] = {}
            for tf, metrics in tf_results.items():
                if isinstance(metrics, dict):
                    # 已经是字典格式
                    serializable_results[symbol][tf] = metrics
                else:
                    # 转换FactorMetrics对象
                    serializable_results[symbol][tf] = {
                        name: vars(m) if hasattr(m, '__dict__') else str(m)
                        for name, m in metrics.items()
                    }
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ 结果已保存: {output_file}")
        return output_file
    
    def print_summary(self):
        """打印批量筛选摘要"""
        print("\n" + "="*60)
        print("批量筛选摘要")
        print("="*60)
        print(f"总任务数: {self.stats.total_tasks}")
        print(f"完成任务: {self.stats.completed_tasks}")
        print(f"失败任务: {self.stats.failed_tasks}")
        print(f"成功率: {self.stats.success_rate:.1f}%")
        print(f"发现优质因子: {self.stats.total_factors_found} 个")
        print(f"总耗时: {self.stats.elapsed_time:.1f} 秒")
        if self.stats.completed_tasks > 0:
            print(f"平均每任务: {self.stats.elapsed_time / self.stats.completed_tasks:.2f} 秒")
        print("="*60)

