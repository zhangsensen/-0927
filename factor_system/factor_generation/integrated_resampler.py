#!/usr/bin/env python3
"""
整合重采样模块 - 支持批量因子处理系统
从1分钟数据自动生成缺失的时间框架数据

设计原则：
1. 无缝集成到批量处理流程
2. 智能检测缺失时间框架
3. 高效的内存管理
4. 完整的错误处理
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class IntegratedResampler:
    """整合重采样器 - 为批量因子处理提供重采样支持"""
    
    def __init__(self):
        """初始化重采样器"""
        self.timeframe_mapping = {
            "1min": "1min",
            "2min": "2min",
            "3min": "3min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "120min": "120min",
            "240min": "240min",
            "1day": "1day",
        }
        
        # 重采样规则映射
        self.resample_rules = {
            "1min": "1min",
            "2min": "2min",
            "3min": "3min", 
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "120min": "120min",
            "240min": "240min",
            "1day": "1D"
        }
        
        logger.info("整合重采样器初始化完成")

    def normalize_timeframe_label(self, label: str) -> str:
        """标准化时间框架标签"""
        mapping = {
            "1h": "60min",
            "60m": "60min",
            "30m": "30min",
            "15m": "15min",
            "5m": "5min",
            "2m": "2min",
            "1m": "1min",
            "2h": "120min",
            "4h": "240min",
            "120m": "120min",
            "240m": "240min",
            "1d": "1day",
            "1day": "1day",
            "daily": "1day",
        }
        lower = label.lower()
        normalized = mapping.get(lower, lower)
        return self.timeframe_mapping.get(normalized, normalized)

    def can_resample_from_1min(self, target_timeframe: str) -> bool:
        """检查是否可以从1分钟数据重采样到目标时间框架"""
        normalized = self.normalize_timeframe_label(target_timeframe)
        # 1分钟数据可以重采样到所有更高时间框架
        resampleable = ["2min", "3min", "5min", "15min", "30min", "60min", "120min", "240min", "1day"]
        return normalized in resampleable

    def resample_ohlcv(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        重采样OHLCV数据到目标时间框架
        
        Args:
            data: 1分钟OHLCV数据，必须包含timestamp索引
            target_timeframe: 目标时间框架
            
        Returns:
            重采样后的数据
        """
        normalized_tf = self.normalize_timeframe_label(target_timeframe)
        
        if normalized_tf not in self.resample_rules:
            raise ValueError(f"不支持的时间框架: {target_timeframe}")
        
        # 确保数据有正确的时间索引
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data = data.set_index(pd.to_datetime(data['timestamp']))
            else:
                data.index = pd.to_datetime(data.index)
        
        # 排序确保时间顺序
        data = data.sort_index()
        
        # 定义聚合规则
        agg_rules = {}
        if 'open' in data.columns:
            agg_rules['open'] = 'first'
        if 'high' in data.columns:
            agg_rules['high'] = 'max'
        if 'low' in data.columns:
            agg_rules['low'] = 'min'
        if 'close' in data.columns:
            agg_rules['close'] = 'last'
        if 'volume' in data.columns:
            agg_rules['volume'] = 'sum'
        if 'turnover' in data.columns:
            agg_rules['turnover'] = 'sum'
        
        if not agg_rules:
            raise ValueError("数据必须包含OHLCV列")
        
        # 执行重采样
        rule = self.resample_rules[normalized_tf]
        resampled = data.resample(rule, label='right', closed='right').agg(agg_rules)
        
        # 清理数据
        resampled.dropna(how='all', inplace=True)
        if 'close' in resampled.columns:
            resampled = resampled[resampled['close'].notna()]
        
        # 重置索引名称
        resampled.index.name = 'timestamp'
        
        logger.debug(f"重采样完成: {len(data)} -> {len(resampled)} 行 ({normalized_tf})")
        
        return resampled

    def find_missing_timeframes(self, stock_files: Dict[str, str], 
                               required_timeframes: List[str]) -> List[str]:
        """
        找出缺失的时间框架
        
        Args:
            stock_files: 股票现有文件 {timeframe: file_path}
            required_timeframes: 需要的时间框架列表
            
        Returns:
            缺失的时间框架列表
        """
        existing_tfs = set(stock_files.keys())
        required_tfs = set(required_timeframes)
        missing_tfs = required_tfs - existing_tfs
        
        # 过滤出可以从1分钟重采样的时间框架
        resampleable_missing = []
        for tf in missing_tfs:
            if self.can_resample_from_1min(tf) and '1min' in existing_tfs:
                resampleable_missing.append(tf)
        
        return resampleable_missing

    def generate_missing_data(self, stock_symbol: str, stock_files: Dict[str, str],
                            missing_timeframes: List[str], 
                            output_dir: Path) -> Dict[str, str]:
        """
        为股票生成缺失时间框架的数据
        
        Args:
            stock_symbol: 股票代码
            stock_files: 现有文件映射
            missing_timeframes: 缺失的时间框架
            output_dir: 输出目录
            
        Returns:
            新生成的文件映射 {timeframe: file_path}
        """
        if not missing_timeframes or '1min' not in stock_files:
            return {}
        
        # 读取1分钟数据
        min1_file = stock_files['1min']
        try:
            data_1min = pd.read_parquet(min1_file)
            logger.debug(f"{stock_symbol}: 读取1分钟数据 {len(data_1min)} 行")
        except Exception as e:
            logger.error(f"{stock_symbol}: 读取1分钟数据失败 - {e}")
            return {}
        
        generated_files = {}
        
        for tf in missing_timeframes:
            try:
                # 重采样数据
                resampled_data = self.resample_ohlcv(data_1min, tf)
                
                # 生成输出文件名
                # 从原文件名提取日期范围
                original_name = Path(min1_file).stem
                parts = original_name.split('_')
                if len(parts) >= 3:
                    symbol_part = parts[0]
                    date_part = '_'.join(parts[2:])  # 跳过时间框架部分
                    output_filename = f"{symbol_part}_{tf}_{date_part}.parquet"
                else:
                    output_filename = f"{stock_symbol}_{tf}_resampled.parquet"
                
                # 创建输出目录
                tf_output_dir = output_dir / tf
                tf_output_dir.mkdir(parents=True, exist_ok=True)
                
                output_path = tf_output_dir / output_filename
                
                # 保存重采样数据
                resampled_data.to_parquet(output_path, compression='snappy')
                generated_files[tf] = str(output_path)
                
                logger.info(f"{stock_symbol}: 生成 {tf} 数据 -> {output_filename} ({len(resampled_data)} 行)")
                
            except Exception as e:
                logger.error(f"{stock_symbol}: 生成 {tf} 数据失败 - {e}")
                continue
        
        return generated_files

    def ensure_all_timeframes(self, stock_symbol: str, stock_files: Dict[str, str],
                            required_timeframes: List[str], 
                            temp_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        确保股票拥有所有需要的时间框架数据
        
        Args:
            stock_symbol: 股票代码
            stock_files: 现有文件映射
            required_timeframes: 需要的时间框架
            temp_dir: 临时文件目录
            
        Returns:
            完整的文件映射 {timeframe: file_path}（键已标准化）
        """
        # 🔧 标准化现有文件的时间框架键
        normalized_stock_files = {}
        for tf, path in stock_files.items():
            normalized_tf = self.normalize_timeframe_label(tf)
            normalized_stock_files[normalized_tf] = path
        
        # 找出缺失的时间框架
        missing_tfs = self.find_missing_timeframes(normalized_stock_files, required_timeframes)
        
        if not missing_tfs:
            logger.debug(f"{stock_symbol}: 所有时间框架数据已存在")
            return normalized_stock_files.copy()
        
        logger.info(f"{stock_symbol}: 需要生成缺失时间框架: {missing_tfs}")
        
        # 设置输出目录
        if temp_dir is None:
            temp_dir = Path("./temp_resampled")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成缺失数据
        generated_files = self.generate_missing_data(
            stock_symbol, normalized_stock_files, missing_tfs, temp_dir
        )
        
        # 合并文件映射
        complete_files = normalized_stock_files.copy()
        complete_files.update(generated_files)
        
        logger.info(f"{stock_symbol}: 完成时间框架补全，共 {len(complete_files)} 个时间框架")
        
        return complete_files


def create_resampler() -> IntegratedResampler:
    """创建重采样器实例"""
    return IntegratedResampler()


# 兼容性函数，用于与现有batch_resample_hk.py的接口保持一致
def batch_resample_from_1min(data_1min: pd.DataFrame, 
                           target_timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """
    从1分钟数据批量重采样到多个时间框架
    
    Args:
        data_1min: 1分钟OHLCV数据
        target_timeframes: 目标时间框架列表
        
    Returns:
        重采样结果 {timeframe: resampled_data}
    """
    resampler = IntegratedResampler()
    results = {}
    
    for tf in target_timeframes:
        try:
            if resampler.can_resample_from_1min(tf):
                results[tf] = resampler.resample_ohlcv(data_1min, tf)
            else:
                logger.warning(f"无法从1分钟重采样到 {tf}")
        except Exception as e:
            logger.error(f"重采样到 {tf} 失败: {e}")
    
    return results
