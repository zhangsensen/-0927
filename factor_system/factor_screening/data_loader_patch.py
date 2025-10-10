#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器补丁 - 支持HK/US市场分离的数据加载

这个文件包含改进的 load_factors 和 load_price_data 方法
将被集成到 ProfessionalFactorScreener 中
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.market_utils import (
    construct_factor_file_path,
    construct_price_file_path,
    normalize_symbol,
    parse_market
)
from utils.timeframe_utils import map_timeframe

logger = logging.getLogger(__name__)


def load_factors_v2(self, symbol: str, timeframe: str) -> pd.DataFrame:
    """
    加载因子数据 - V3版本，从大宽表中提取纯因子列
    
    🔧 核心改进：
    - 从大宽表中排除价格列（open, high, low, close, volume）
    - 只返回纯因子列，避免筛选时混入价格数据
    
    Args:
        symbol: 股票代码 (支持 '0700.HK', '0700HK', 'AAPLUS' 等格式)
        timeframe: 时间框架 (如: '5min', '15min', '1day')
        
    Returns:
        因子数据DataFrame（不包含价格列）
    """
    start_time = time.time()
    self.logger.info(f"加载因子数据: {symbol} {timeframe}")
    
    try:
        # 使用新的路径构建工具
        factor_file = construct_factor_file_path(
            self.data_root,
            symbol,
            timeframe
        )
        
        if not factor_file.exists():
            raise FileNotFoundError(f"因子文件不存在: {factor_file}")
        
        self.logger.info(f"✅ 找到因子文件: {factor_file.name}")
        
        # Linus式优化：大数据使用pyarrow引擎，内存映射减少I/O
        if factor_file.stat().st_size > 50 * 1024 * 1024:  # 50MB以上
            wide_table = pd.read_parquet(factor_file, engine='pyarrow')
        else:
            wide_table = pd.read_parquet(factor_file)

        # 数据质量检查
        if wide_table.empty:
            raise ValueError(f"因子文件为空: {factor_file}")

        # 🔧 排除价格列，只保留因子列
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        factor_cols = [col for col in wide_table.columns if col not in price_cols]

        if not factor_cols:
            raise ValueError(f"大宽表中没有因子列（只有价格列）")

        factors = wide_table[factor_cols].copy()

        # 🔧 修复：正确设置时间索引（在使用wide_table之前）
        if 'timestamp' in wide_table.columns:
            # 使用timestamp列作为索引（包含实际日期时间）
            factors.index = pd.to_datetime(wide_table['timestamp'])
        elif 'datetime' in wide_table.columns:
            # 备选：使用datetime列
            factors.index = pd.to_datetime(wide_table['datetime'])
        elif not isinstance(factors.index, pd.DatetimeIndex):
            # 最后才尝试转换现有索引（如果是RangeIndex，这会产生错误的时间戳）
            factors.index = pd.to_datetime(factors.index)

        # 立即释放大宽表内存
        del wide_table
        
        self.logger.info(
            f"📊 从大宽表提取因子: 因子列数={len(factor_cols)}, 数据行数={factors.shape[0]}"
        )
        
        # 内存优化：移除全NaN列
        factors = factors.dropna(axis=1, how="all")
        
        # 内存优化：转换数据类型
        for col in factors.select_dtypes(include=["float64"]).columns:
            factors[col] = factors[col].astype("float32")
        
        # 数据质量验证
        factors = self.validate_factor_data_quality(factors, symbol, timeframe)
        
        initial_memory = factors.memory_usage(deep=True).sum() / 1024 / 1024
        self.logger.info(
            f"因子数据加载成功: 形状={factors.shape}, "
            f"内存={initial_memory:.1f}MB"
        )
        self.logger.info(
            f"时间范围: {factors.index.min()} 到 {factors.index.max()}"
        )
        self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")
        
        return factors
        
    except FileNotFoundError as e:
        self.logger.error(f"❌ 因子文件不存在: {e}")
        self.logger.error(f"搜索路径: {self.data_root}")
        self.logger.error(f"股票代码: {symbol}")
        self.logger.error(f"时间框架: {timeframe}")
        
        # 提供详细的诊断信息
        try:
            clean_symbol, market = normalize_symbol(symbol)
            market_dir = self.data_root / market
            if market_dir.exists():
                self.logger.error(f"可用时间框架目录: {[d.name for d in market_dir.iterdir() if d.is_dir()]}")
            else:
                self.logger.error(f"市场目录不存在: {market_dir}")
        except Exception:
            pass
        
        raise
        
    except Exception as e:
        self.logger.error(f"❌ 加载因子数据失败: {e}")
        raise


def load_price_data_v2(
    self, 
    symbol: str, 
    timeframe: Optional[str] = None
) -> pd.DataFrame:
    """
    加载价格数据 - V3版本，直接从大宽表因子文件中提取OHLCV数据
    
    🔧 核心改进：
    - 大宽表已包含价格数据（open, high, low, close, volume）
    - 无需访问 @raw/ 目录
    - 时间对齐问题自动解决（因子和价格在同一文件中）
    
    Args:
        symbol: 股票代码
        timeframe: 时间框架
        
    Returns:
        价格数据DataFrame (包含 open, high, low, close, volume)
    """
    start_time = time.time()
    self.logger.info(f"从大宽表加载价格数据: {symbol} {timeframe}")
    
    try:
        # 🔧 直接从因子文件加载（大宽表包含价格+因子）
        factor_file = construct_factor_file_path(
            self.data_root,
            symbol,
            timeframe
        )
        
        if not factor_file.exists():
            raise FileNotFoundError(f"因子文件不存在: {factor_file}")
        
        self.logger.info(f"✅ 从因子文件提取价格数据: {factor_file.name}")
        
        # 加载大宽表
        wide_table = pd.read_parquet(factor_file)
        
        if wide_table.empty:
            raise ValueError(f"因子文件为空: {factor_file}")
        
        # 🔧 提取价格列（OHLCV）
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in price_cols if col not in wide_table.columns]
        
        if missing_cols:
            raise ValueError(
                f"大宽表缺少价格列: {missing_cols}\n"
                f"可用列示例: {list(wide_table.columns[:10])}"
            )
        
        # 提取价格数据
        price_data = wide_table[price_cols].copy()
        
        # 🔧 修复：正确设置时间索引（价格数据）
        # 重新加载大宽表以获取timestamp信息
        if 'timestamp' in wide_table.columns:
            # 使用timestamp列作为索引（包含实际日期时间）
            price_data.index = pd.to_datetime(wide_table['timestamp'])
        elif 'datetime' in wide_table.columns:
            # 备选：使用datetime列
            price_data.index = pd.to_datetime(wide_table['datetime'])
        elif not isinstance(price_data.index, pd.DatetimeIndex):
            # 最后才尝试转换现有索引
            price_data.index = pd.to_datetime(price_data.index)
        
        # 处理timestamp列（如果存在）
        if 'timestamp' in wide_table.columns:
            # timestamp列可能是毫秒时间戳，需要转换
            if price_data.index.name != 'timestamp':
                price_data.index.name = 'timestamp'
        
        self.logger.info(
            f"✅ 价格数据提取成功: 形状={price_data.shape}"
        )
        self.logger.info(
            f"时间范围: {price_data.index.min()} 到 {price_data.index.max()}"
        )
        self.logger.info(f"加载耗时: {time.time() - start_time:.2f}秒")
        
        return price_data
        
    except FileNotFoundError as e:
        self.logger.error(f"❌ 因子文件不存在: {e}")
        self.logger.error(f"搜索路径: {self.data_root}")
        self.logger.error(f"股票代码: {symbol}")
        self.logger.error(f"时间框架: {timeframe}")
        
        # 提供详细的诊断信息
        try:
            clean_symbol, market = normalize_symbol(symbol)
            market_dir = self.data_root / market
            if market_dir.exists():
                tf_dir = market_dir / timeframe
                if tf_dir.exists():
                    available_files = list(tf_dir.glob("*.parquet"))
                    self.logger.error(f"可用因子文件: {[f.name for f in available_files[:5]]}")
        except Exception:
            pass
        
        raise
        
    except Exception as e:
        self.logger.error(f"❌ 从大宽表提取价格数据失败: {e}")
        raise


# 补丁函数：将新方法注入到ProfessionalFactorScreener类
def patch_data_loader(screener_instance):
    """
    将新的数据加载方法补丁到现有的筛选器实例
    
    Args:
        screener_instance: ProfessionalFactorScreener实例
    """
    import types
    
    # 替换load_factors方法
    screener_instance.load_factors = types.MethodType(
        load_factors_v2,
        screener_instance
    )
    
    # 替换load_price_data方法  
    screener_instance.load_price_data = types.MethodType(
        load_price_data_v2,
        screener_instance
    )
    
    logger.info("✅ 数据加载器已升级到V2版本（支持HK/US市场分离）")

