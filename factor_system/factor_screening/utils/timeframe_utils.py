#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间框架工具
处理因子文件和价格文件的时间框架映射
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# 时间框架映射表
TIMEFRAME_MAPPING = {
    # 因子文件命名 -> 价格文件命名
    '1min': '1min',
    '2min': '2min',
    '3min': '3min',
    '5min': '5min',
    '15min': '15m',      # 🔧 修复：价格文件使用15m格式
    '30min': '30m',      # 🔧 修复：价格文件使用30m格式
    '60min': '60m',      # 🔧 修复：价格文件使用60m格式
    '2h': '60m',         # 🔧 2h因子使用60m价格文件（最接近的）
    '4h': '60m',         # 🔧 4h因子使用60m价格文件（最接近的）
    '1day': '1day',
    'daily': '1day',     # 兼容旧配置
    # 额外的别名
    '1d': '1day',
    'd': '1day',
}

# 支持的所有时间框架
SUPPORTED_TIMEFRAMES = [
    '1min', '2min', '3min', '5min', '15min', 
    '30min', '60min', '2h', '4h', '1day'
]

# 时间框架分组
TIMEFRAME_GROUPS = {
    'ultra_short': ['1min', '2min', '3min'],
    'short': ['5min', '15min', '30min'],
    'medium': ['60min', '2h', '4h'],
    'long': ['1day'],
}

# 时间框架排序权重（用于排序）
TIMEFRAME_ORDER = {
    '1min': 1,
    '2min': 2,
    '3min': 3,
    '5min': 4,
    '15min': 5,
    '30min': 6,
    '60min': 7,
    '2h': 8,
    '4h': 9,
    '1day': 10,
    'daily': 10,  # 与1day相同
}


def map_timeframe(timeframe: str, target: str = 'price') -> str:
    """
    时间框架映射
    
    Args:
        timeframe: 输入时间框架
        target: 'price' (价格文件) 或 'factor' (因子文件)
        
    Returns:
        映射后的时间框架名称
        
    Examples:
        >>> map_timeframe('daily', 'price')
        '1day'
        >>> map_timeframe('5min', 'price')
        '5min'
    """
    if target == 'price':
        return TIMEFRAME_MAPPING.get(timeframe, timeframe)
    else:
        # 因子文件通常不需要映射
        return timeframe


def normalize_timeframe(timeframe: str) -> str:
    """
    标准化时间框架名称
    
    Args:
        timeframe: 原始时间框架名称
        
    Returns:
        标准化后的名称
        
    Examples:
        >>> normalize_timeframe('daily')
        '1day'
        >>> normalize_timeframe('1d')
        '1day'
        >>> normalize_timeframe('5min')
        '5min'
    """
    return TIMEFRAME_MAPPING.get(timeframe, timeframe)


def validate_timeframe(timeframe: str) -> bool:
    """
    验证时间框架是否支持
    
    Args:
        timeframe: 时间框架名称
        
    Returns:
        是否为支持的时间框架
    """
    normalized = normalize_timeframe(timeframe)
    return normalized in SUPPORTED_TIMEFRAMES


def get_timeframe_group(timeframe: str) -> Optional[str]:
    """
    获取时间框架所属分组
    
    Args:
        timeframe: 时间框架名称
        
    Returns:
        分组名称: 'ultra_short', 'short', 'medium', 'long'
        
    Examples:
        >>> get_timeframe_group('5min')
        'short'
        >>> get_timeframe_group('1day')
        'long'
    """
    normalized = normalize_timeframe(timeframe)
    
    for group, timeframes in TIMEFRAME_GROUPS.items():
        if normalized in timeframes:
            return group
    
    return None


def sort_timeframes(timeframes: List[str]) -> List[str]:
    """
    按时间长度排序时间框架
    
    Args:
        timeframes: 时间框架列表
        
    Returns:
        排序后的时间框架列表
        
    Examples:
        >>> sort_timeframes(['1day', '5min', '15min'])
        ['5min', '15min', '1day']
    """
    return sorted(
        timeframes,
        key=lambda tf: TIMEFRAME_ORDER.get(normalize_timeframe(tf), 999)
    )


def get_available_timeframes(
    data_root, 
    symbol: str, 
    market: str
) -> List[str]:
    """
    获取指定股票的所有可用时间框架
    
    Args:
        data_root: 数据根目录
        symbol: 股票代码
        market: 市场 ('HK' 或 'US')
        
    Returns:
        可用的时间框架列表
    """
    from pathlib import Path
    
    data_root = Path(data_root)
    market_dir = data_root / market
    
    if not market_dir.exists():
        logger.warning(f"市场目录不存在: {market_dir}")
        return []
    
    available = []
    
    for tf in SUPPORTED_TIMEFRAMES:
        tf_dir = market_dir / tf
        if tf_dir.exists():
            # 检查是否有该股票的因子文件
            pattern = f"{symbol}_{tf}_factors.parquet"
            if list(tf_dir.glob(pattern)):
                available.append(tf)
    
    return sort_timeframes(available)


def get_timeframe_minutes(timeframe: str) -> int:
    """
    获取时间框架对应的分钟数
    
    Args:
        timeframe: 时间框架名称
        
    Returns:
        分钟数
        
    Examples:
        >>> get_timeframe_minutes('5min')
        5
        >>> get_timeframe_minutes('2h')
        120
        >>> get_timeframe_minutes('1day')
        1440
    """
    timeframe = normalize_timeframe(timeframe)
    
    mapping = {
        '1min': 1,
        '2min': 2,
        '3min': 3,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '60min': 60,
        '2h': 120,
        '4h': 240,
        '1day': 1440,
    }
    
    return mapping.get(timeframe, 0)


def is_intraday(timeframe: str) -> bool:
    """
    判断是否为日内时间框架
    
    Args:
        timeframe: 时间框架名称
        
    Returns:
        是否为日内时间框架
    """
    return normalize_timeframe(timeframe) != '1day'


def get_compatible_ic_horizons(timeframe: str) -> List[int]:
    """
    获取适合该时间框架的IC分析周期
    
    Args:
        timeframe: 时间框架名称
        
    Returns:
        推荐的IC分析周期列表
        
    Examples:
        >>> get_compatible_ic_horizons('5min')
        [1, 3, 5, 10, 20]
        >>> get_compatible_ic_horizons('1day')
        [1, 3, 5, 10, 20, 30, 60]
    """
    minutes = get_timeframe_minutes(timeframe)
    
    if minutes <= 5:  # 超短周期
        return [1, 3, 5, 10, 20]
    elif minutes <= 30:  # 短周期
        return [1, 3, 5, 10, 20, 30]
    elif minutes <= 240:  # 中周期
        return [1, 3, 5, 10, 20, 40]
    else:  # 长周期
        return [1, 3, 5, 10, 20, 30, 60]

